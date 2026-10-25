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

// ---- decode: exl3_expert_grouped ------------------------------------------------------------------------------------
// Compiled with -ffp-contract=off (CMakeLists.txt): every product and sum below is rounded on its own, as on the CPU.
#include "strata/kernels/sexp.hpp"

namespace strata::kernels {
namespace {

// tile step of (row r, column c): the inverse of exl3_expert.hpp's step placement
__device__ __forceinline__ int step_of(int r, int c) {
    return (r & 1) | (((r >> 3) & 1) << 1) | (((c >> 3) & 1) << 2) | (((r >> 1) & 3) << 3) | ((c & 7) << 5);
}

// one entry's scratch: the prepared inputs of gate, up and down, and h
struct EntLay {
    size_t qg, qu, qd, dg, du, dd, Qg, Qu, Qd, qs, h, bytes;
    __host__ __device__ EntLay(int64_t H, int64_t F) {
        auto al = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
        qg = 0; qu = (size_t) H; qd = 2 * (size_t) H;
        dg = al(qd + (size_t) F, 16); du = dg + 4 * (size_t) (H / 128); dd = du + 4 * (size_t) (H / 128);
        Qg = dd + 4 * (size_t) (F / 128); Qu = Qg + 4 * (size_t) (H / 128); Qd = Qu + 4 * (size_t) (H / 128);
        qs = Qd + 4 * (size_t) (F / 128);
        h = al(qs + 12, 16);
        bytes = al(h + 4 * (size_t) F, 256);
    }
};

struct RoleView {
    const uint8_t* trellis;
    const __half* suh;
    const __half* svh;
    int k, n, K2;
};

__device__ __forceinline__ RoleView role_view(const uint8_t* blob, int r, int H, int F, int gk2, int dk2) {
    size_t at = 0;
    RoleView v{};
    for (int i = 0; i <= r; ++i) {
        v.k = i < 2 ? H : F;
        v.n = i < 2 ? F : H;
        v.K2 = i < 2 ? gk2 : dk2;
        const size_t tb = (size_t) (v.k / 16) * (size_t) (v.n / 16) * 16u * (size_t) v.K2;
        v.trellis = blob + at;
        v.suh = reinterpret_cast<const __half*>(blob + at + tb);
        v.svh = v.suh + v.k;
        at += tb + 2u * (size_t) (v.k + v.n);
    }
    return v;
}

__device__ __forceinline__ float round_away(float t) {
    const float tr = truncf(t);
    return fabsf(t - tr) >= 0.5f ? tr + copysignf(1.0f, t) : tr;
}

// the 128-point Hadamard over a block's 128 threads (v in LDS, one value a thread): the scalar code's butterflies
__device__ __forceinline__ float had128_block(float* sh, float v) {
    const int i = threadIdx.x;
    sh[i] = v;
    __syncthreads();
    for (int h = 1; h < 128; h *= 2) {
        const float a = sh[i & ~h], b = sh[i | h];
        __syncthreads();
        sh[i] = (i & h) ? a - b : a + b;
        __syncthreads();
    }
    return sh[i];
}

__global__ void group_map_kernel(const int32_t* grp_start, const int32_t* n_groups, int64_t cap_groups, int32_t* ent_grp) {
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    for (int g = 0; g < ng; ++g)
        for (int e = grp_start[g] + (int) threadIdx.x; e < grp_start[g + 1]; e += blockDim.x) ent_grp[e] = g;
}

// grid (cap, roles, k / 128), 128 threads: one 128-block of one entry's input to one role, rotated and quantized
__global__ void __launch_bounds__(128) prep_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                   const int32_t* n_groups, int64_t cap_groups, const int32_t* ent_tok,
                                                   const int32_t* ent_grp, const float* x, uint8_t* scratch, int H, int F,
                                                   int gk2, int dk2, int role0) {
    const int e = blockIdx.x, role = role0 + (int) blockIdx.y, j = blockIdx.z, i = threadIdx.x;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e >= grp_start[ng]) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const RoleView m = role_view(reinterpret_cast<const uint8_t*>(grp_ptr[ent_grp[e]]), role, H, F, gk2, dk2);
    const float* src = role == 2 ? reinterpret_cast<const float*>(es + L.h) : x + (size_t) ent_tok[e] * (size_t) H;
    __shared__ float sh[128];
    __shared__ float red[128];
    __shared__ int ired[128];
    const float u = src[128 * j + i] * __half2float(m.suh[128 * j + i]);
    const float v = had128_block(sh, u);
    red[i] = fabsf(v);
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (i < s) red[i] = fmaxf(red[i], red[i + s]);
        __syncthreads();
    }
    const float d = red[0] / 127.0f;
    float q = d > 0.f ? round_away(v / d) : 0.f;
    q = fminf(fmaxf(q, -127.f), 127.f);
    const size_t qo = role == 0 ? L.qg : role == 1 ? L.qu : L.qd;
    reinterpret_cast<int8_t*>(es + qo)[128 * j + i] = (int8_t) q;
    ired[i] = (int) q;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (i < s) ired[i] += ired[i + s];
        __syncthreads();
    }
    if (i == 0) {
        reinterpret_cast<float*>(es + (role == 0 ? L.dg : role == 1 ? L.du : L.dd))[j] = d;
        reinterpret_cast<int32_t*>(es + (role == 0 ? L.Qg : role == 1 ? L.Qu : L.Qd))[j] = ired[0];
    }
}

// grid (cap, roles): qs of each prepared input, left to right over its blocks
__global__ void qs_kernel(const int32_t* grp_start, const int32_t* n_groups, int64_t cap_groups, uint8_t* scratch, int H,
                          int F, int role0) {
    const int e = blockIdx.x, role = role0 + (int) blockIdx.y;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e >= grp_start[ng] || threadIdx.x != 0) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const float* d = reinterpret_cast<const float*>(es + (role == 0 ? L.dg : role == 1 ? L.du : L.dd));
    const int32_t* Q = reinterpret_cast<const int32_t*>(es + (role == 0 ? L.Qg : role == 1 ? L.Qu : L.Qd));
    const int nb = (role == 2 ? F : H) / 128;
    float qs = 0.f;
    for (int j = 0; j < nb; ++j) qs = qs + d[j] * (float) Q[j];
    reinterpret_cast<float*>(es + L.qs)[role] = qs;
}

// output column c (thread) of block b of one role: steps 3-5, before the svh scale
__device__ __forceinline__ float matvec_col(const RoleView& m, const int8_t* q, const float* d, float qs, int b, float* sh) {
    const int c = 128 * b + (int) threadIdx.x;
    const int tn = c >> 4, cc = c & 15;
    const size_t tile_bytes = 16u * (size_t) m.K2;
    float acc = 0.f;
    for (int j = 0; j < m.k / 128; ++j) {
        int32_t I = 0;
        for (int tk = 8 * j; tk < 8 * j + 8; ++tk) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(m.trellis + ((size_t) tk * (size_t) (m.n / 16) + (size_t) tn) * tile_bytes);
            for (int r = 0; r < 16; ++r) I += (int32_t) q[16 * tk + r] * step_s(w, m.K2, step_of(r, cc));
        }
        acc = acc + d[j] * (float) I;
    }
    const float yp = ex::kA * acc + ex::kB * qs;
    const float y = had128_block(sh, yp);
    return (y * (1.0f / 128.0f)) * __half2float(m.svh[c]);
}

// grid (cap, n_ff / 128), 128 threads: gate and up block b, then h
__global__ void __launch_bounds__(128) gu_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                 const int32_t* n_groups, int64_t cap_groups, const int32_t* ent_grp,
                                                 uint8_t* scratch, int H, int F, int gk2, int dk2) {
    const int e = blockIdx.x, b = blockIdx.y;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e >= grp_start[ng]) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[ent_grp[e]]);
    __shared__ float sh[128];
    const float* qs = reinterpret_cast<const float*>(es + L.qs);
    const float g = matvec_col(role_view(blob, 0, H, F, gk2, dk2), reinterpret_cast<const int8_t*>(es + L.qg),
                               reinterpret_cast<const float*>(es + L.dg), qs[0], b, sh);
    __syncthreads();
    const float u = matvec_col(role_view(blob, 1, H, F, gk2, dk2), reinterpret_cast<const int8_t*>(es + L.qu),
                               reinterpret_cast<const float*>(es + L.du), qs[1], b, sh);
    reinterpret_cast<float*>(es + L.h)[128 * b + threadIdx.x] = (g / (1.0f + strata::kernels::sexp(-g))) * u;
}

// grid (cap, n_embd / 128), 128 threads: down block b -> out row ent_dst[e]
__global__ void __launch_bounds__(128) down_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                   const int32_t* n_groups, int64_t cap_groups, const int32_t* ent_grp,
                                                   const int32_t* ent_dst, uint8_t* scratch, float* out, int H, int F,
                                                   int gk2, int dk2) {
    const int e = blockIdx.x, b = blockIdx.y;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e >= grp_start[ng]) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[ent_grp[e]]);
    __shared__ float sh[128];
    const float y = matvec_col(role_view(blob, 2, H, F, gk2, dk2), reinterpret_cast<const int8_t*>(es + L.qd),
                               reinterpret_cast<const float*>(es + L.dd), reinterpret_cast<const float*>(es + L.qs)[2], b, sh);
    out[(size_t) ent_dst[e] * (size_t) H + (size_t) (128 * b) + threadIdx.x] = y;
}

}  // namespace

size_t exl3_expert_scratch_bytes(int64_t cap, int64_t n_embd, int64_t n_ff) {
    const EntLay L(n_embd, n_ff);
    return (size_t) cap * L.bytes + (size_t) cap * sizeof(int32_t) + 256;
}

void exl3_expert_grouped(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const unsigned long long* grp_ptr,
                         const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                         const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const float* x, void* scratch,
                         float* out, void* stream) {
    const int gk2 = ex::k2_of(gu_type), dk2 = ex::k2_of(d_type);
    const int H = (int) n_embd, F = (int) n_ff;
    if (!gk2 || !dk2 || H % 128 || F % 128 || cap_entries <= 0) {
        std::fprintf(stderr, "exl3_expert_grouped: bad arguments (types %d/%d)\n", gu_type, d_type);
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    const EntLay L(n_embd, n_ff);
    uint8_t* sc = static_cast<uint8_t*>(scratch);
    int32_t* ent_grp = reinterpret_cast<int32_t*>(sc + (size_t) cap_entries * L.bytes);
    const int cap = (int) cap_entries;
    group_map_kernel<<<1, 256, 0, s>>>(grp_start, n_groups, cap_groups, ent_grp);
    prep_kernel<<<dim3(cap, 2, H / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_tok, ent_grp, x, sc,
                                                      H, F, gk2, dk2, 0);
    qs_kernel<<<dim3(cap, 2), 32, 0, s>>>(grp_start, n_groups, cap_groups, sc, H, F, 0);
    gu_kernel<<<dim3(cap, F / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_grp, sc, H, F, gk2, dk2);
    prep_kernel<<<dim3(cap, 1, F / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_tok, ent_grp, x, sc,
                                                      H, F, gk2, dk2, 2);
    qs_kernel<<<dim3(cap, 1), 32, 0, s>>>(grp_start, n_groups, cap_groups, sc, H, F, 2);
    down_kernel<<<dim3(cap, H / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_grp, ent_dst, sc, out,
                                                   H, F, gk2, dk2);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "exl3_expert_grouped: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

}  // namespace strata::kernels
