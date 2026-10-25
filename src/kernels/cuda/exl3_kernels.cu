// src/kernels/cuda/exl3_kernels.cu - EXL3 routed experts on the GPU (exl3_kernels.hpp).
#include "strata/kernels/exl3_kernels.hpp"

#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "strata/kernels/exl3_expert.hpp"

namespace strata::kernels {
namespace {

namespace ex = strata::kernels::exl3;

// the codebook index of tile step p (exl3_expert.hpp FORMAT): the 16-bit window ending at the step's end bit
__device__ __forceinline__ int step_s(const uint32_t* w, int K2, int p) {
    const int words = 4 * K2, nbits = 32 * words;
    int end;
    if ((K2 & 1) == 0) end = (p + 1) * (K2 >> 1);
    else end = (p & 1) == 0 ? (p >> 1) * (K2 | 1) + (K2 >> 1) : ((p + 1) >> 1) * (K2 | 1);
    int start = end - 16;
    if (start < 0) start += nbits;
    const int wi = start >> 5, off = start & 31;
    const int wj = wi + 1 == words ? 0 : wi + 1;
    const uint64_t v = ((uint64_t) w[wi] << 32) | w[wj];
    const uint32_t state = (uint32_t) (v >> (48 - off)) & 0xFFFFu;
    const uint32_t x = state * ex::kMul1;
    return (int) ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24));
}

// LDS: a 128 x 128 block, element (r, c) at r * 128 + ((c + r) & 127) - the rotation keeps both the column passes
// (threads along c) and the row passes (threads along r) free of bank conflicts
__device__ __forceinline__ int at(int r, int c) { return r * 128 + ((c + r) & 127); }

// grid (n / 128, k / 128), 256 threads: block (j, b) of one matrix -> dst rows (row_mul * c + row_add), columns k
__global__ void __launch_bounds__(256) dequant_kernel(const uint8_t* role, int k, int n, int K2, __half* dst,
                                                     int row_mul, int row_add) {
    extern __shared__ float W[];   // 128 x 128 floats = 64 KB
    const int b = blockIdx.x, j = blockIdx.y, t = threadIdx.x;
    const size_t tile_bytes = 16u * (size_t) K2;
    const size_t trellis = (size_t) (k / 16) * (size_t) (n / 16) * tile_bytes;
    const __half* suh = reinterpret_cast<const __half*>(role + trellis);
    const __half* svh = suh + k;
    // 1. decode: 64 tiles, four threads a tile, 64 steps each
    {
        const int tile = t >> 2, q = t & 3;
        const int tkl = tile >> 3, tcl = tile & 7;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(
            role + ((size_t) (8 * j + tkl) * (size_t) (n / 16) + (size_t) (8 * b + tcl)) * tile_bytes);
        for (int p = 64 * q; p < 64 * q + 64; ++p) {
            const int r = ((p >> 3) & 3) * 2 + (p & 1) + 8 * ((p >> 1) & 1);
            const int c = (p >> 5) + 8 * ((p >> 2) & 1);
            W[at(16 * tkl + r, 16 * tcl + c)] = ex::kA * (float) step_s(w, K2, p) + ex::kB;
        }
    }
    __syncthreads();
    // 2. the left Hadamard (over r, per column), then row i times suh
    {
        const int c = t & 127, half = t >> 7;
        for (int h = 1; h < 128; h *= 2) {
            for (int u = half * 32; u < half * 32 + 32; ++u) {   // butterfly u of 64: lower index i
                const int i = (u / h) * 2 * h + (u % h);
                const float a = W[at(i, c)], bb = W[at(i + h, c)];
                W[at(i, c)] = a + bb;
                W[at(i + h, c)] = a - bb;
            }
            __syncthreads();
        }
        for (int i = half * 64; i < half * 64 + 64; ++i) W[at(i, c)] *= __half2float(suh[128 * j + i]);
    }
    __syncthreads();
    // 3. the right Hadamard (over c, per row)
    {
        const int r = t & 127, half = t >> 7;
        for (int h = 1; h < 128; h *= 2) {
            for (int u = half * 32; u < half * 32 + 32; ++u) {
                const int i = (u / h) * 2 * h + (u % h);
                const float a = W[at(r, i)], bb = W[at(r, i + h)];
                W[at(r, i)] = a + bb;
                W[at(r, i + h)] = a - bb;
            }
            __syncthreads();
        }
    }
    // 4. column c times svh / 128 -> dst row (row_mul (128 b + c) + row_add), value 128 j + i
    for (int e = t; e < 128 * 128; e += 256) {
        const int c = e >> 7, i = e & 127;
        const float v = W[at(i, c)] * (__half2float(svh[128 * b + c]) * (1.0f / 128.0f));
        dst[(size_t) (row_mul * (128 * b + c) + row_add) * (size_t) k + (size_t) (128 * j + i)] = __float2half(v);
    }
}

void launch(const void* role, int k, int n, int K2, uint16_t* dst, int row_mul, int row_add, void* stream, const char* what) {
    static bool attr = false;
    if (!attr) {
        (void) cudaFuncSetAttribute(dequant_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 128 * 4);
        attr = true;
    }
    dequant_kernel<<<dim3(n / 128, k / 128), 256, 128 * 128 * 4, (cudaStream_t) stream>>>(
        static_cast<const uint8_t*>(role), k, n, K2, reinterpret_cast<__half*>(dst), row_mul, row_add);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void exl3_dequant_gu_f16(int gu_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                         void* stream) {
    const int K2 = ex::k2_of(gu_type);
    if (!K2 || n_embd % 128 || n_ff % 128) { std::fprintf(stderr, "exl3_dequant_gu_f16: type %d\n", gu_type); std::exit(1); }
    launch(gate, (int) n_embd, (int) n_ff, K2, dst, 2, 0, stream, "exl3_dequant_gu_f16");
    launch(up, (int) n_embd, (int) n_ff, K2, dst, 2, 1, stream, "exl3_dequant_gu_f16");
}

void exl3_dequant_down_f16(int d_type, const void* down, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    const int K2 = ex::k2_of(d_type);
    if (!K2 || n_embd % 128 || n_ff % 128) { std::fprintf(stderr, "exl3_dequant_down_f16: type %d\n", d_type); std::exit(1); }
    launch(down, (int) n_ff, (int) n_embd, K2, dst, 1, 0, stream, "exl3_dequant_down_f16");
}

}  // namespace strata::kernels
