// exl3_pf_check <exl3 pack> <fixture> [layer] [--bench]
//
// exl3_pf_group (the prompt path's grouped EXL3 experts: the trellis decoded inside FP16 WMMA tiles) against the
// prompt path it replaces, rebuilt here from its own kernels: exl3_dequant_left_f16 (FP16 weights with the input
// rotation folded in), FP32-accumulated GEMMs, exl3_rotate_gu_out, the SwiGLU to FP16, exl3_rotate_d_out.  Neither is
// the parity arithmetic (FP16 activations): the two differ by rounding (~1e-3 relative); a wrong decode, fragment or
// rotation shows as O(1).  16 experts of one layer, row counts 1..300 at non-contiguous offsets, real activations
// (the fixture's inputs).  --bench: us per group.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "strata/kernels/exl3_expert.hpp"
#include "strata/kernels/exl3_kernels.hpp"

namespace {
struct LayerFmt { int gt = 0, dt = 0; unsigned long long off = 0, blob = 0; };

// Y[r][n] = sum_k X[r][k] W[n][k] (FP16 in, FP32 sums)
__global__ void ref_gemm(const __half* X, const __half* W, float* Y, int rows, int n, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= n || r >= rows) return;
    float s = 0.f;
    for (int i = 0; i < k; ++i) s += __half2float(X[(size_t) r * k + i]) * __half2float(W[(size_t) c * k + i]);
    Y[(size_t) r * n + c] = s;
}
__global__ void ref_swiglu(const float* gu, __half* h, int rows, int nff) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nff) return;
    const int r = i / nff, c = i % nff;
    const float g = gu[(size_t) r * 2 * nff + 2 * c], u = gu[(size_t) r * 2 * nff + 2 * c + 1];
    const float v = g / (1.0f + __expf(-g)) * u;
    h[i] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: exl3_pf_check <pack> <fixture> [layer] [--bench]\n"); return 2; }
    const std::string pack = argv[1];
    const int layer = argc > 3 && argv[3][0] != '-' ? std::atoi(argv[3]) : 4;
    bool bench = false;
    for (int i = 3; i < argc; ++i) bench |= std::strcmp(argv[i], "--bench") == 0;
    std::map<int, LayerFmt> lay;
    {
        std::ifstream in(pack + "/native_experts.txt");
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int l;
            LayerFmt f;
            ss >> l >> f.gt >> f.dt >> f.off >> f.blob;
            lay[l] = f;
        }
    }
    if (!lay.count(layer)) { std::fprintf(stderr, "no layer %d\n", layer); return 2; }
    const LayerFmt f = lay.at(layer);
    std::ifstream fx(argv[2], std::ios::binary);
    int32_t head[4];
    fx.read(reinterpret_cast<char*>(head), sizeof head);
    const int nrec = head[0], H = head[1], F = head[2];
    if (!strata::kernels::exl3_pf_supported(f.gt, f.dt, H, F)) { std::fprintf(stderr, "unsupported layer\n"); return 2; }
    std::vector<std::vector<float>> xs((size_t) nrec, std::vector<float>((size_t) H));
    for (int r = 0; r < nrec; ++r) {
        int32_t le[2];
        fx.read(reinterpret_cast<char*>(le), 8);
        fx.read(reinterpret_cast<char*>(xs[(size_t) r].data()), 4 * H);
        fx.seekg(4 * (F + H), std::ios::cur);
    }
    const int NE = 16;
    int cnts[NE] = {1, 2, 5, 16, 17, 31, 64, 65, 100, 127, 128, 129, 200, 250, 3, 300};
    if (const char* v = std::getenv("STRATA_PFC_AVG")) {   // (bench) row counts spread around an average: 0.25..1.75 x
        const int avg = std::atoi(v);
        for (int e = 0; e < NE; ++e) cnts[e] = std::max(1, avg / 4 + (int) ((3.0 * avg / 2.0) * ((e * 7 + 3) % NE) / (NE - 1)));
    }
    strata::kernels::Exl3PfGroup g{};
    g.n = NE;
    int rows = 0;
    for (int e = 0; e < NE; ++e) { g.off[e] = rows + 7 * e; g.cnt[e] = cnts[e]; rows += cnts[e]; }
    const int total_rows = g.off[NE - 1] + cnts[NE - 1];
    std::ifstream bin(pack + "/experts.bin", std::ios::binary);
    std::vector<void*> dblob((size_t) NE);
    for (int e = 0; e < NE; ++e) {
        std::vector<uint8_t> b(f.blob);
        bin.seekg((std::streamoff) (f.off + (unsigned long long) (13 * e + 1) * f.blob));
        bin.read(reinterpret_cast<char*>(b.data()), (std::streamsize) f.blob);
        cudaMalloc(&dblob[(size_t) e], f.blob);
        cudaMemcpy(dblob[(size_t) e], b.data(), f.blob, cudaMemcpyHostToDevice);
        g.blob[e] = static_cast<const uint8_t*>(dblob[(size_t) e]);
    }
    std::vector<__half> hx((size_t) total_rows * H);
    for (int r = 0; r < total_rows; ++r)
        for (int k = 0; k < H; ++k) hx[(size_t) r * H + k] = __float2half(xs[(size_t) (r % nrec)][(size_t) k] * (1.0f + 0.25f * (float) (r % 5)));
    __half *dX, *dHh, *dHr, *dqg, *dqd;
    float *dGU, *dDm, *dGr, *dDr;
    float* dsc;
    cudaMalloc(&dX, hx.size() * 2);
    cudaMemcpy(dX, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&dHh, (size_t) total_rows * F * 2); cudaMalloc(&dHr, (size_t) total_rows * F * 2);
    cudaMalloc(&dGU, (size_t) total_rows * 2 * F * 4); cudaMalloc(&dGr, (size_t) total_rows * 2 * F * 4);
    cudaMalloc(&dDm, (size_t) total_rows * H * 4); cudaMalloc(&dDr, (size_t) total_rows * H * 4);
    cudaMemset(dDm, 0, (size_t) total_rows * H * 4); cudaMemset(dDr, 0, (size_t) total_rows * H * 4);
    cudaMalloc(&dqg, ((size_t) 2 * F * H + 2 * F) * 2); cudaMalloc(&dqd, ((size_t) H * F + H) * 2);
    cudaMalloc(&dsc, (size_t) total_rows * 3 * 4);
    // the reference: the current prompt path, expert by expert
    for (int e = 0; e < NE; ++e) {
        const int o = g.off[e], c = g.cnt[e];
        strata::kernels::exl3_dequant_left_f16(f.gt, f.dt, g.blob[e], F, H, reinterpret_cast<uint16_t*>(dqg),
                                               reinterpret_cast<uint16_t*>(dqd), nullptr);
        ref_gemm<<<dim3((2 * F + 127) / 128, c), 128>>>(dX + (size_t) o * H, dqg, dGr + (size_t) o * 2 * F, c, 2 * F, H);
        strata::kernels::exl3_rotate_gu_out(dGr + (size_t) o * 2 * F, c, F, reinterpret_cast<const uint16_t*>(dqg) + (size_t) 2 * F * H, nullptr);
        ref_swiglu<<<(c * F + 255) / 256, 256>>>(dGr + (size_t) o * 2 * F, dHr + (size_t) o * F, c, F);
        ref_gemm<<<dim3((H + 127) / 128, c), 128>>>(dHr + (size_t) o * F, dqd, dDr + (size_t) o * H, c, H, F);
        strata::kernels::exl3_rotate_d_out(dDr + (size_t) o * H, c, H, reinterpret_cast<const uint16_t*>(dqd) + (size_t) H * F, nullptr);
    }
    strata::kernels::exl3_pf_group(f.gt, f.dt, H, F, g, reinterpret_cast<const uint16_t*>(dX), dGU,
                                   reinterpret_cast<uint16_t*>(dHh), dDm, dsc, nullptr);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err)); return 1; }
    std::vector<float> a((size_t) total_rows * H), b((size_t) total_rows * H);
    cudaMemcpy(a.data(), dDm, a.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), dDr, b.size() * 4, cudaMemcpyDeviceToHost);
    double num = 0, den = 0, worst = 0;
    int worst_row = -1, nan = 0;
    for (int e = 0; e < NE; ++e)
        for (int r = g.off[e]; r < g.off[e] + g.cnt[e]; ++r) {
            double rn = 0, rd = 0;
            for (int k = 0; k < H; ++k) {
                const double d = (double) a[(size_t) r * H + k] - (double) b[(size_t) r * H + k];
                if (!std::isfinite(a[(size_t) r * H + k])) ++nan;
                rn += d * d;
                rd += (double) b[(size_t) r * H + k] * b[(size_t) r * H + k];
            }
            num += rn;
            den += rd;
            const double rel = std::sqrt(rn / (rd > 0 ? rd : 1));
            if (rel > worst) { worst = rel; worst_row = r; }
        }
    const double rel = std::sqrt(num / den);
    std::printf("layer %d (types %d/%d), %d experts, %d rows: down output rel RMS %.3e, worst row %.3e (row %d), %d non-finite\n",
                layer, f.gt, f.dt, NE, rows, rel, worst, worst_row, nan);
    const bool ok = rel < 1e-2 && worst < 3e-2 && nan == 0;
    std::printf("%s\n", ok ? "OK (rounding-level)" : "FAILED");
    if (bench) {
        cudaStream_t st;
        cudaStreamCreate(&st);
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (int i = 0; i < 3; ++i)
            strata::kernels::exl3_pf_group(f.gt, f.dt, H, F, g, reinterpret_cast<const uint16_t*>(dX), dGU,
                                           reinterpret_cast<uint16_t*>(dHh), dDm, dsc, st);
        cudaEventRecord(t0, st);
        const int reps = 20;
        for (int i = 0; i < reps; ++i)
            strata::kernels::exl3_pf_group(f.gt, f.dt, H, F, g, reinterpret_cast<const uint16_t*>(dX), dGU,
                                           reinterpret_cast<uint16_t*>(dHh), dDm, dsc, st);
        cudaEventRecord(t1, st);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventRecord(t0, st);
        for (int i = 0; i < reps; ++i)
            for (int e = 0; e < NE; ++e)
                strata::kernels::exl3_dequant_left_f16(f.gt, f.dt, g.blob[e], F, H, reinterpret_cast<uint16_t*>(dqg),
                                                       reinterpret_cast<uint16_t*>(dqd), st);
        cudaEventRecord(t1, st);
        cudaEventSynchronize(t1);
        float dq = 0;
        cudaEventElapsedTime(&dq, t0, t1);
        const double flop = 2.0 * rows * (2.0 * F * H + (double) H * F);
        std::printf("bench: %.1f us per group of %d experts / %d rows (%.1f TFLOPS); the old path's dequant alone %.1f us\n",
                    1000.0 * ms / reps, NE, rows, flop / (1e-3 * ms / reps) / 1e12, 1000.0 * dq / reps);
    }
    return ok ? 0 : 1;
}
