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
    size_t qg, qu, qd, dg, du, dd, Qg, Qu, Qd, qs, h, ypg, ypu, ypd, bytes;
    __host__ __device__ EntLay(int64_t H, int64_t F) {
        auto al = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
        qg = 0; qu = 2 * (size_t) H; qd = 4 * (size_t) H;   // int16 values (int8 or Q15 range)
        dg = al(qd + 2 * (size_t) F, 16); du = dg + 4 * (size_t) (H / 128); dd = du + 4 * (size_t) (H / 128);
        Qg = dd + 4 * (size_t) (F / 128); Qu = Qg + 4 * (size_t) (H / 128); Qd = Qu + 4 * (size_t) (H / 128);
        qs = Qd + 4 * (size_t) (F / 128);
        h = al(qs + 12, 16);
        ypg = h + 4 * (size_t) F;          // the v2 path's y' (before the output rotation) of gate, up and down
        ypu = ypg + 4 * (size_t) F;
        ypd = ypu + 4 * (size_t) F;
        bytes = al(ypd + 4 * (size_t) H, 256);
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
                                                   int gk2, int dk2, int role0, int qm) {
    const int e = blockIdx.x, role = role0 + (int) blockIdx.y, j = blockIdx.z, i = threadIdx.x;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e < grp_start[0] || e >= grp_start[ng]) return;   // a PCIe pass's entries follow the VRAM pass's
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
    const float qmf = (float) qm;
    const float d = red[0] / qmf;
    float q = d > 0.f ? round_away(v / d) : 0.f;
    q = fminf(fmaxf(q, -qmf), qmf);
    const size_t qo = role == 0 ? L.qg : role == 1 ? L.qu : L.qd;
    reinterpret_cast<int16_t*>(es + qo)[128 * j + i] = (int16_t) q;
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
    if (e < grp_start[0] || e >= grp_start[ng] || threadIdx.x != 0) return;
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
__device__ __forceinline__ float matvec_col(const RoleView& m, const int16_t* q, const float* d, float qs, int b, float* sh) {
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
    if (e < grp_start[0] || e >= grp_start[ng]) return;   // a PCIe pass's entries follow the VRAM pass's
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[ent_grp[e]]);
    __shared__ float sh[128];
    const float* qs = reinterpret_cast<const float*>(es + L.qs);
    const float g = matvec_col(role_view(blob, 0, H, F, gk2, dk2), reinterpret_cast<const int16_t*>(es + L.qg),
                               reinterpret_cast<const float*>(es + L.dg), qs[0], b, sh);
    __syncthreads();
    const float u = matvec_col(role_view(blob, 1, H, F, gk2, dk2), reinterpret_cast<const int16_t*>(es + L.qu),
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
    if (e < grp_start[0] || e >= grp_start[ng]) return;   // a PCIe pass's entries follow the VRAM pass's
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[ent_grp[e]]);
    __shared__ float sh[128];
    const float y = matvec_col(role_view(blob, 2, H, F, gk2, dk2), reinterpret_cast<const int16_t*>(es + L.qd),
                               reinterpret_cast<const float*>(es + L.dd), reinterpret_cast<const float*>(es + L.qs)[2], b, sh);
    out[(size_t) ent_dst[e] * (size_t) H + (size_t) (128 * b) + threadIdx.x] = y;
}


// ---- v2: one wave per (group, role, 16-column tile column), every token of the group at once -----------------------
// Lane l owns column c = l & 15 of the tile column and rows 8 (l >> 4) .. + 7 of every tile; a tile's words sit one per
// lane and the steps fetch theirs with shuffles.  The integer sums of a 128-input block are exact in any order (the two
// halves meet by one integer shuffle-add), and the FP32 part runs block by block, left to right, as the spec says; the
// output rotation runs after (post kernels), as in v1.  Bit for bit v1 and the CPU.
constexpr int kV2MaxT = 8;

template <int K2>
__device__ __forceinline__ int step_end_t(int p) {
    if constexpr ((K2 & 1) == 0) return (p + 1) * (K2 / 2);
    else return (p & 1) == 0 ? (p >> 1) * (K2 | 1) + (K2 / 2) : ((p + 1) >> 1) * (K2 | 1);
}

template <int K2>
__global__ void __launch_bounds__(128) band_v2_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                      const int32_t* n_groups, int64_t cap_groups, uint8_t* scratch,
                                                      int H, int F, int gk2, int dk2, int role0, int units_per_role) {
    constexpr int words = 4 * K2, nbits = 32 * words;
    const int wave = (int) (blockIdx.x * 4 + (threadIdx.x >> 5)), lane = (int) (threadIdx.x & 31);
    const int roles = role0 == 2 ? 1 : 2;
    const int per_group = roles * units_per_role;
    const int g = wave / per_group, rem = wave % per_group;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (g >= ng) return;
    const int role = role0 + rem / units_per_role, tn = rem % units_per_role;
    const int e0 = grp_start[g], nt = min(grp_start[g + 1] - e0, kV2MaxT);
    const EntLay L(H, F);
    const RoleView m = role_view(reinterpret_cast<const uint8_t*>(grp_ptr[g]), role, H, F, gk2, dk2);
    const int c = lane & 15, h = lane >> 4;
    const size_t tile_bytes = 16u * (size_t) K2;
    const size_t qo = role == 0 ? L.qg : role == 1 ? L.qu : L.qd;
    const size_t dof = role == 0 ? L.dg : role == 1 ? L.du : L.dd;
    float acc[kV2MaxT];
    for (int t = 0; t < kV2MaxT; ++t) acc[t] = 0.f;
    // the 8 steps of this lane in a tile (rows 8 h + i, column c)
    int sw[8], sh8[8];
    for (int i = 0; i < 8; ++i) {
        int start = step_end_t<K2>(step_of(8 * h + i, c)) - 16;
        if (start < 0) start += nbits;
        sw[i] = start >> 5;
        sh8[i] = 48 - (start & 31);
    }
    for (int j = 0; j < m.k / 128; ++j) {
        int32_t I[kV2MaxT];
        for (int t = 0; t < kV2MaxT; ++t) I[t] = 0;
        for (int tk = 8 * j; tk < 8 * j + 8; ++tk) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(
                m.trellis + ((size_t) tk * (size_t) (m.n / 16) + (size_t) tn) * tile_bytes);
            const uint32_t mine = lane < words ? w[lane] : 0u;
            int s[8];
            for (int i = 0; i < 8; ++i) {
                const uint32_t hi = __shfl_sync(0xffffffffu, mine, sw[i]);
                const uint32_t lo = __shfl_sync(0xffffffffu, mine, sw[i] + 1 == words ? 0 : sw[i] + 1);
                const uint32_t state = (uint32_t) ((((uint64_t) hi << 32) | lo) >> sh8[i]) & 0xFFFFu;
                const uint32_t x = state * ex::kMul1;
                s[i] = (int) ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24));
            }
            for (int t = 0; t < nt; ++t) {
                const int16_t* q = reinterpret_cast<const int16_t*>(scratch + (size_t) (e0 + t) * L.bytes + qo) + 16 * tk + 8 * h;
                const uint4 qv = *reinterpret_cast<const uint4*>(q);   // 8 int16, 16-byte aligned
                const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                int32_t a = I[t];
                for (int i = 0; i < 8; ++i) a += (int32_t) (int16_t) (qw[i >> 1] >> (16 * (i & 1))) * s[i];
                I[t] = a;
            }
        }
        for (int t = 0; t < nt; ++t) {
            const int32_t tot = I[t] + __shfl_xor_sync(0xffffffffu, I[t], 16);
            const float d = reinterpret_cast<const float*>(scratch + (size_t) (e0 + t) * L.bytes + dof)[j];
            acc[t] = acc[t] + d * (float) tot;
        }
    }
    if (h == 0) {
        const size_t yo = role == 0 ? L.ypg : role == 1 ? L.ypu : L.ypd;
        for (int t = 0; t < nt; ++t) {
            uint8_t* es = scratch + (size_t) (e0 + t) * L.bytes;
            const float qs = reinterpret_cast<const float*>(es + L.qs)[role];
            reinterpret_cast<float*>(es + yo)[16 * tn + c] = ex::kA * acc[t] + ex::kB * qs;
        }
    }
}

// grid (cap, n_ff / 128), 128 threads: gate and up block b: the output rotation, svh, SwiGLU -> h
__global__ void __launch_bounds__(128) gu_post_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                      const int32_t* n_groups, int64_t cap_groups, const int32_t* ent_grp,
                                                      uint8_t* scratch, int H, int F, int gk2, int dk2) {
    const int e = blockIdx.x, b = blockIdx.y;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e < grp_start[0] || e >= grp_start[ng]) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[ent_grp[e]]);
    __shared__ float sh[128];
    const int c = 128 * b + (int) threadIdx.x;
    const RoleView mg = role_view(blob, 0, H, F, gk2, dk2), mu = role_view(blob, 1, H, F, gk2, dk2);
    const float yg = had128_block(sh, reinterpret_cast<const float*>(es + L.ypg)[c]);
    const float g = (yg * (1.0f / 128.0f)) * __half2float(mg.svh[c]);
    __syncthreads();
    const float yu = had128_block(sh, reinterpret_cast<const float*>(es + L.ypu)[c]);
    const float u = (yu * (1.0f / 128.0f)) * __half2float(mu.svh[c]);
    reinterpret_cast<float*>(es + L.h)[c] = (g / (1.0f + strata::kernels::sexp(-g))) * u;
}

// grid (cap, n_embd / 128), 128 threads: down block b: the output rotation and svh -> out row ent_dst[e]
__global__ void __launch_bounds__(128) down_post_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                        const int32_t* n_groups, int64_t cap_groups, const int32_t* ent_grp,
                                                        const int32_t* ent_dst, uint8_t* scratch, float* out, int H, int F,
                                                        int gk2, int dk2) {
    const int e = blockIdx.x, b = blockIdx.y;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e < grp_start[0] || e >= grp_start[ng]) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const RoleView md = role_view(reinterpret_cast<const uint8_t*>(grp_ptr[ent_grp[e]]), 2, H, F, gk2, dk2);
    __shared__ float sh[128];
    const int c = 128 * b + (int) threadIdx.x;
    const float y = had128_block(sh, reinterpret_cast<const float*>(es + L.ypd)[c]);
    out[(size_t) ent_dst[e] * (size_t) H + (size_t) c] = (y * (1.0f / 128.0f)) * __half2float(md.svh[c]);
}

// ---- v3: lanes own a quarter of a tile's bit stream (64 steps), a wave = the 8 tiles of one 128-input block ---------
// Lane l: tile row 8 j + (l >> 2) of the block, quarter i = l & 3 of its tile (steps 64 i .. 64 i + 63, bits
// [64 i K, 64 (i + 1) K) = words [K2 i, K2 (i + 1)), plus the word before for the windows that start earlier).  A step's
// window then sits at a compile-time bit offset; its row is a function of the step alone and its column is one of the
// lane's four (2 i, 2 i + 1, 2 i + 8, 2 i + 9).  The 8 tile rows' integer sums meet by three shuffle-adds (exact), and
// the FP32 accumulation runs block by block as the spec says: bit for bit v1 / v2 / the CPU.
// state * 0x83DCD12D (mod 2^32) for the state in the HIGH half of v (its low half: anything): one packed 16-bit multiply
// gives (state * 0x83DC) << 16 (the low half multiplies 0), one 16 x 16 + 32 mad adds state * 0xD12D.  Two full-rate
// instructions, where the compiler turns state * C into a quarter-rate v_mul_lo_u32.  Exact: integer arithmetic.
__device__ __forceinline__ uint32_t mul1_hi(uint32_t v) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
    uint32_t r, x;
    asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(r) : "v"(v), "v"(0x83DC0000u));
    asm("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(x) : "v"(v), "v"(0xD12Du), "v"(r));
    return x;
#else
    return (v >> 16) * ex::kMul1;
#endif
}

template <int K2>
__global__ void __launch_bounds__(64) band_v3_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                     const int32_t* n_groups, int64_t cap_groups, uint8_t* scratch,
                                                     int H, int F, int gk2, int dk2, int role0, int units_per_role) {
    constexpr int words = 4 * K2;
    const int wave = (int) (blockIdx.x * 2 + (threadIdx.x >> 5)), lane = (int) (threadIdx.x & 31);
    const int roles = role0 == 2 ? 1 : 2;
    const int per_group = roles * units_per_role;
    const int g = wave / per_group, rem = wave % per_group;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (g >= ng) return;
    const int role = role0 + rem / units_per_role, tn = rem % units_per_role;
    const int e0 = grp_start[g], nt = min(grp_start[g + 1] - e0, kV2MaxT);
    const EntLay L(H, F);
    const RoleView m = role_view(reinterpret_cast<const uint8_t*>(grp_ptr[g]), role, H, F, gk2, dk2);
    const int qi = lane & 3, tr = lane >> 2;
    const size_t tile_bytes = 16u * (size_t) K2;
    const size_t qo = role == 0 ? L.qg : role == 1 ? L.qu : L.qd;
    const size_t dof = role == 0 ? L.dg : role == 1 ? L.du : L.dd;
    const size_t Qof = role == 0 ? L.Qg : role == 1 ? L.Qu : L.Qd;
    float acc[kV2MaxT][4];
#pragma unroll
    for (int t = 0; t < kV2MaxT; ++t)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[t][c] = 0.f;
    const int nb = m.k / 128;
    for (int j = 0; j < nb; ++j) {
        const int tk = 8 * j + tr;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(
            m.trellis + ((size_t) tk * (size_t) (m.n / 16) + (size_t) tn) * tile_bytes);
        uint32_t buf[K2 + 2];   // the word before, the K2 chunk words, and a zero (the last window's discarded low bits)
        buf[0] = w[qi == 0 ? words - 1 : K2 * qi - 1];
        buf[K2 + 1] = 0u;
#pragma unroll
        for (int k = 0; k < K2; ++k) buf[1 + k] = w[K2 * qi + k];
        int sv[64];
#pragma unroll
        for (int u = 0; u < 64; ++u) {
            constexpr int KA = K2 / 2;
            const int endl = (K2 & 1) == 0 ? (u + 1) * KA : ((u & 1) == 0 ? (u >> 1) * K2 + KA : ((u + 1) >> 1) * K2);
            const int b = endl - 16 + 32;            // bit index in buf (buf[0] = the word before the chunk)
            const int a = b >> 5, o = b & 31;
            const uint32_t v = o == 0 ? buf[a] : ((buf[a] << o) | (buf[a + 1] >> (32 - o)));   // state in the high half
            sv[u] = (int) __builtin_amdgcn_sad_u8(mul1_hi(v), 0u, 0u);                          // its byte sum
        }
        // each token: its four columns' integer sums over this lane's tile, the 8 tile rows' sums met by three
        // shuffle-adds (exact), then the block's FP32 step - one token's sums live at a time
#pragma unroll
        for (int t = 0; t < kV2MaxT; ++t) {
            if (t < nt) {
                const uint8_t* es = scratch + (size_t) (e0 + t) * L.bytes;
                const int16_t* q = reinterpret_cast<const int16_t*>(es + qo) + 16 * tk;
                const uint4 q0 = *reinterpret_cast<const uint4*>(q), q1 = *reinterpret_cast<const uint4*>(q + 8);
                const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
                int32_t I[4] = {0, 0, 0, 0};
#pragma unroll
                for (int u = 0; u < 64; ++u) {
                    const int r = ((u >> 3) & 3) * 2 + (u & 1) + 8 * ((u >> 1) & 1);
                    const int cl = (u >> 5) + 2 * ((u >> 2) & 1);
                    const int qr = (int32_t) (int16_t) (qw[r >> 1] >> (16 * (r & 1)));
                    I[cl] += __mul24(qr, sv[u]);   // |q| <= 16383, s <= 1020: v_mad_i32_i24
                }
                const float d = reinterpret_cast<const float*>(es + dof)[j];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    int32_t v = I[c];
                    v += __shfl_xor_sync(0xffffffffu, v, 4);
                    v += __shfl_xor_sync(0xffffffffu, v, 8);
                    v += __shfl_xor_sync(0xffffffffu, v, 16);
                    acc[t][c] = acc[t][c] + d * (float) v;
                }
            }
        }
    }
    if (tr == 0) {
        const size_t yo = role == 0 ? L.ypg : role == 1 ? L.ypu : L.ypd;
#pragma unroll
        for (int t = 0; t < kV2MaxT; ++t) {
            if (t < nt) {
                uint8_t* es = scratch + (size_t) (e0 + t) * L.bytes;
                const float* d = reinterpret_cast<const float*>(es + dof);
                const int32_t* Q = reinterpret_cast<const int32_t*>(es + Qof);
                float qs = 0.f;
                for (int jj = 0; jj < nb; ++jj) qs = qs + d[jj] * (float) Q[jj];   // as qs_kernel, left to right
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int col = 2 * qi + (c & 1) + 8 * (c >> 1);
                    reinterpret_cast<float*>(es + yo)[16 * tn + col] = ex::kA * acc[t][c] + ex::kB * qs;
                }
            }
        }
    }
}

template <int K2>
void launch_band_v3(int64_t cap_groups, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    uint8_t* sc, int H, int F, int gk2, int dk2, int role0, cudaStream_t s) {
    const int units = (role0 == 2 ? H : F) / 16;
    const int waves = (int) cap_groups * (role0 == 2 ? 1 : 2) * units;
    band_v3_kernel<K2><<<(waves + 1) / 2, 64, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, sc, H, F, gk2, dk2,
                                                      role0, units);
}

bool v2_path() {
    static const bool v2 = [] { const char* e = std::getenv("STRATA_EXL3_V2"); return e && e[0] == '1'; }();
    return v2;
}

template <int K2>
void launch_band(int64_t cap_groups, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                 uint8_t* sc, int H, int F, int gk2, int dk2, int role0, cudaStream_t s) {
    const int units = (role0 == 2 ? H : F) / 16;
    const int waves = (int) cap_groups * (role0 == 2 ? 1 : 2) * units;
    band_v2_kernel<K2><<<(waves + 3) / 4, 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, sc, H, F, gk2, dk2,
                                                       role0, units);
}

void band(int K2, int64_t cap_groups, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
          uint8_t* sc, int H, int F, int gk2, int dk2, int role0, cudaStream_t s) {
    if (!v2_path()) {   // v3 (default)
        switch (K2) {
            case 5: launch_band_v3<5>(cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, role0, s); return;
            case 6: launch_band_v3<6>(cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, role0, s); return;
            case 7: launch_band_v3<7>(cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, role0, s); return;
            default: break;   // the other widths: v2
        }
    }
    switch (K2) {
        case 4: launch_band<4>(cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, role0, s); break;
        case 5: launch_band<5>(cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, role0, s); break;
        case 6: launch_band<6>(cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, role0, s); break;
        case 7: launch_band<7>(cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, role0, s); break;
        case 8: launch_band<8>(cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, role0, s); break;
        default: std::fprintf(stderr, "exl3_expert_grouped: v2 has no K2 %d\n", K2); std::exit(1);
    }
}

// ---- v3's pipeline: five launches (prepare gate/up, gate/up bands, gate/up post + prepare down, down bands, down post)
// An entry finds its group by a search over grp_start (no map kernel); the bands compute qs themselves (no qs kernel).
__device__ __forceinline__ int group_of(const int32_t* grp_start, int ng, int e) {
    int lo = 0, hi = ng - 1;   // the last g with grp_start[g] <= e
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (grp_start[mid] <= e) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// ---- wave-level 128-value blocks: lane l holds values 4 l .. 4 l + 3 (no LDS, no barriers) ------------------------
// The Hadamard's butterflies are the scalar code's (strides 1, 2 in the lane; 4 .. 64 across lanes by xor shuffles),
// each output from the same two inputs in the same order, so every value is the same bits as had128_block's.
__device__ __forceinline__ void had128_wave(float v[4]) {
    float a, b;
    a = v[0]; b = v[1]; v[0] = a + b; v[1] = a - b;
    a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b;
    a = v[0]; b = v[2]; v[0] = a + b; v[2] = a - b;
    a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b;
    const int lane = (int) (threadIdx.x & 31);
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool upper = (lane & m) != 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float o = __shfl_xor_sync(0xffffffffu, v[k], m);
            v[k] = upper ? o - v[k] : v[k] + o;   // lower: self + partner; upper: partner (lower element) - self
        }
    }
}

// one 128-block of a role's input, rotated and quantized: x4 the lane's four values (prep_kernel's arithmetic)
__device__ __forceinline__ void prep_wave(uint8_t* es, const EntLay& L, const RoleView& m, int role, int j, const float x4[4],
                                          int qm) {
    const int lane = (int) (threadIdx.x & 31);
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = x4[k] * __half2float(m.suh[128 * j + 4 * lane + k]);
    had128_wave(v);
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float qmf = (float) qm;
    const float d = am / qmf;
    int qsum = 0;
    int16_t qv[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float q = d > 0.f ? round_away(v[k] / d) : 0.f;
        q = fminf(fmaxf(q, -qmf), qmf);
        qv[k] = (int16_t) q;
        qsum += (int) q;
    }
    const size_t qo = role == 0 ? L.qg : role == 1 ? L.qu : L.qd;
    uint2 packed;
    packed.x = (uint32_t) (uint16_t) qv[0] | ((uint32_t) (uint16_t) qv[1] << 16);
    packed.y = (uint32_t) (uint16_t) qv[2] | ((uint32_t) (uint16_t) qv[3] << 16);
    *reinterpret_cast<uint2*>(reinterpret_cast<int16_t*>(es + qo) + 128 * j + 4 * lane) = packed;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qsum += __shfl_xor_sync(0xffffffffu, qsum, o);
    if (lane == 0) {
        reinterpret_cast<float*>(es + (role == 0 ? L.dg : role == 1 ? L.du : L.dd))[j] = d;
        reinterpret_cast<int32_t*>(es + (role == 0 ? L.Qg : role == 1 ? L.Qu : L.Qd))[j] = qsum;
    }
}

// grid (cap, 2, n_embd / 512), 128 threads = 4 waves: gate / up input block j = 4 z + wave of entry e
__global__ void __launch_bounds__(128) prep3_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                    const int32_t* n_groups, int64_t cap_groups, const int32_t* ent_tok,
                                                    const float* x, uint8_t* scratch, int H, int F, int gk2, int dk2, int qm) {
    const int e = blockIdx.x, role = (int) blockIdx.y, j = 4 * (int) blockIdx.z + (int) (threadIdx.x >> 5);
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e < grp_start[0] || e >= grp_start[ng] || 128 * j >= H) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const RoleView m = role_view(reinterpret_cast<const uint8_t*>(grp_ptr[group_of(grp_start, ng, e)]), role, H, F, gk2, dk2);
    const float4 xv = *reinterpret_cast<const float4*>(x + (size_t) ent_tok[e] * (size_t) H + (size_t) (128 * j) + 4 * (threadIdx.x & 31));
    const float x4[4] = {xv.x, xv.y, xv.z, xv.w};
    prep_wave(es, L, m, role, j, x4, qm);
}

// the output side of one 128-block: y = H(y') / 128 * svh for the lane's four values
__device__ __forceinline__ void post_wave(const float* yp, const __half* svh, int b, float out4[4]) {
    const int lane = (int) (threadIdx.x & 31);
    const float4 yv = *reinterpret_cast<const float4*>(yp + 128 * b + 4 * lane);
    float v[4] = {yv.x, yv.y, yv.z, yv.w};
    had128_wave(v);
#pragma unroll
    for (int k = 0; k < 4; ++k) out4[k] = (v[k] * (1.0f / 128.0f)) * __half2float(svh[128 * b + 4 * lane + k]);
}

// grid (cap, n_ff / 128), 32 threads: gate and up block b -> h (output rotation, svh, SwiGLU), then h's block b as the
// down input (block j = b: the same 128 values)
__global__ void __launch_bounds__(32) gu_post_prep_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                          const int32_t* n_groups, int64_t cap_groups, uint8_t* scratch,
                                                          int H, int F, int gk2, int dk2, int qm) {
    const int e = blockIdx.x, b = blockIdx.y;
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e < grp_start[0] || e >= grp_start[ng]) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[group_of(grp_start, ng, e)]);
    const RoleView mg = role_view(blob, 0, H, F, gk2, dk2), mu = role_view(blob, 1, H, F, gk2, dk2);
    float g[4], u[4], h[4];
    post_wave(reinterpret_cast<const float*>(es + L.ypg), mg.svh, b, g);
    post_wave(reinterpret_cast<const float*>(es + L.ypu), mu.svh, b, u);
#pragma unroll
    for (int k = 0; k < 4; ++k) h[k] = (g[k] / (1.0f + strata::kernels::sexp(-g[k]))) * u[k];
    prep_wave(es, L, role_view(blob, 2, H, F, gk2, dk2), 2, b, h, qm);
}

// grid (cap, n_embd / 512), 128 threads = 4 waves: down block b = 4 y + wave -> out row ent_dst[e]
__global__ void __launch_bounds__(128) down_post3_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                         const int32_t* n_groups, int64_t cap_groups, const int32_t* ent_dst,
                                                         uint8_t* scratch, float* out, int H, int F, int gk2, int dk2) {
    const int e = blockIdx.x, b = 4 * (int) blockIdx.y + (int) (threadIdx.x >> 5);
    const int ng = (int) min<int64_t>(*n_groups, cap_groups);
    if (e < grp_start[0] || e >= grp_start[ng] || 128 * b >= H) return;
    const EntLay L(H, F);
    uint8_t* es = scratch + (size_t) e * L.bytes;
    const RoleView md = role_view(reinterpret_cast<const uint8_t*>(grp_ptr[group_of(grp_start, ng, e)]), 2, H, F, gk2, dk2);
    float y[4];
    post_wave(reinterpret_cast<const float*>(es + L.ypd), md.svh, b, y);
    *reinterpret_cast<float4*>(out + (size_t) ent_dst[e] * (size_t) H + (size_t) (128 * b) + 4 * (threadIdx.x & 31)) =
        make_float4(y[0], y[1], y[2], y[3]);
}

bool v1_path() {
    static const bool v1 = [] { const char* e = std::getenv("STRATA_EXL3_V1"); return e && e[0] == '1'; }();
    return v1;
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
    auto v3_k2 = [](int k2) { return k2 == 5 || k2 == 6 || k2 == 7; };
    if (!v1_path() && !v2_path() && v3_k2(gk2) && v3_k2(dk2)) {   // v3 (default): five launches
        const int qm = ex::act_qmax();
        prep3_kernel<<<dim3(cap, 2, (H / 128 + 3) / 4), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_tok, x, sc, H, F,
                                                           gk2, dk2, qm);
        band(gk2, cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, 0, s);
        gu_post_prep_kernel<<<dim3(cap, F / 128), 32, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, sc, H, F, gk2, dk2, qm);
        band(dk2, cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, 2, s);
        down_post3_kernel<<<dim3(cap, (H / 128 + 3) / 4), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_dst, sc, out, H, F,
                                                             gk2, dk2);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "exl3_expert_grouped: %s\n", cudaGetErrorString(err));
            std::exit(1);
        }
        return;
    }
    group_map_kernel<<<1, 256, 0, s>>>(grp_start, n_groups, cap_groups, ent_grp);
    prep_kernel<<<dim3(cap, 2, H / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_tok, ent_grp, x, sc,
                                                      H, F, gk2, dk2, 0, ex::act_qmax());
    qs_kernel<<<dim3(cap, 2), 32, 0, s>>>(grp_start, n_groups, cap_groups, sc, H, F, 0);
    if (!v1_path()) {   // v2 (default): the bands a wave each, then the output rotations
        band(gk2, cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, 0, s);
        gu_post_kernel<<<dim3(cap, F / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_grp, sc, H, F, gk2, dk2);
        prep_kernel<<<dim3(cap, 1, F / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_tok, ent_grp, x, sc,
                                                          H, F, gk2, dk2, 2, ex::act_qmax());
        qs_kernel<<<dim3(cap, 1), 32, 0, s>>>(grp_start, n_groups, cap_groups, sc, H, F, 2);
        band(dk2, cap_groups, grp_ptr, grp_start, n_groups, sc, H, F, gk2, dk2, 2, s);
        down_post_kernel<<<dim3(cap, H / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_grp, ent_dst, sc,
                                                            out, H, F, gk2, dk2);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "exl3_expert_grouped: %s\n", cudaGetErrorString(err));
            std::exit(1);
        }
        return;
    }
    gu_kernel<<<dim3(cap, F / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_grp, sc, H, F, gk2, dk2);
    prep_kernel<<<dim3(cap, 1, F / 128), 128, 0, s>>>(grp_ptr, grp_start, n_groups, cap_groups, ent_tok, ent_grp, x, sc,
                                                      H, F, gk2, dk2, 2, ex::act_qmax());
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
