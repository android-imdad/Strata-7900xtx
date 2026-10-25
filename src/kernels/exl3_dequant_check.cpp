// exl3_dequant_check <exl3 pack> <layer> <expert> <out.bin>
// The prompt path's EXL3 dequantizer (exl3_kernels.hpp) on one real expert: writes the interleaved gate/up FP16
// rows ([2 n_ff][n_embd]) then the down rows ([n_embd][n_ff]) for tools/exl3_ref.py --dequant-check to compare
// with the float64 reconstruction.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strata/kernels/exl3_expert.hpp"
#include "strata/kernels/exl3_kernels.hpp"

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: exl3_dequant_check <pack> <layer> <expert> <out.bin>\n"); return 2; }
    const std::string pack = argv[1];
    const int L = std::atoi(argv[2]), E = std::atoi(argv[3]);
    std::ifstream in(pack + "/native_experts.txt");
    std::string line;
    int gt = 0, dt = 0;
    unsigned long long off = 0, blob = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int l;
        int a, b;
        unsigned long long o, bb;
        ss >> l >> a >> b >> o >> bb;
        if (l == L) { gt = a; dt = b; off = o; blob = bb; }
    }
    const int H = 2560, FF = 640;
    std::vector<uint8_t> host(blob);
    std::ifstream bin(pack + "/experts.bin", std::ios::binary);
    bin.seekg((std::streamoff) (off + (unsigned long long) E * blob));
    bin.read(reinterpret_cast<char*>(host.data()), (std::streamsize) blob);
    const size_t gu_role = strata::kernels::exl3::role_bytes(H, FF, strata::kernels::exl3::k2_of(gt));
    void* dblob = nullptr;
    uint16_t *dgu = nullptr, *dd = nullptr;
    cudaMalloc(&dblob, blob);
    cudaMalloc(&dgu, (size_t) 2 * FF * H * 2);
    cudaMalloc(&dd, (size_t) H * FF * 2);
    cudaMemcpy(dblob, host.data(), blob, cudaMemcpyHostToDevice);
    const uint8_t* b8 = static_cast<const uint8_t*>(dblob);
    strata::kernels::exl3_dequant_gu_f16(gt, b8, b8 + gu_role, FF, H, dgu, nullptr);
    strata::kernels::exl3_dequant_down_f16(dt, b8 + 2 * gu_role, FF, H, dd, nullptr);
    std::vector<uint16_t> gu((size_t) 2 * FF * H), d((size_t) H * FF);
    cudaMemcpy(gu.data(), dgu, gu.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(d.data(), dd, d.size() * 2, cudaMemcpyDeviceToHost);
    std::ofstream out(argv[4], std::ios::binary);
    out.write(reinterpret_cast<const char*>(gu.data()), (std::streamsize) gu.size() * 2);
    out.write(reinterpret_cast<const char*>(d.data()), (std::streamsize) d.size() * 2);
    std::printf("layer %d expert %d: types %d/%d, wrote %s\n", L, E, gt, dt, argv[4]);
    // the prompt path's dequantizer (input rotation only): its time per expert, launched back to back
    {
        uint16_t *lgu = nullptr, *ld = nullptr;
        cudaMalloc(&lgu, ((size_t) 2 * FF * H + 2 * FF) * 2);
        cudaMalloc(&ld, ((size_t) H * FF + H) * 2);
        cudaStream_t st;
        cudaStreamCreate(&st);
        for (int i = 0; i < 20; ++i) strata::kernels::exl3_dequant_left_f16(gt, dt, b8, FF, H, lgu, ld, st);
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, st);
        const int reps = 500;
        for (int i = 0; i < reps; ++i) strata::kernels::exl3_dequant_left_f16(gt, dt, b8, FF, H, lgu, ld, st);
        cudaEventRecord(b, st);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("exl3_dequant_left_f16: %.1f us per expert (back to back)\n", 1000.0 * ms / reps);
        // write the left-rotated weights after the full ones for tools/exl3_ref.py --dequant-check
        std::vector<uint16_t> g2((size_t) 2 * FF * H + 2 * FF), d2((size_t) H * FF + H);
        cudaMemcpy(g2.data(), lgu, g2.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(d2.data(), ld, d2.size() * 2, cudaMemcpyDeviceToHost);
        std::ofstream o2(std::string(argv[4]) + ".left", std::ios::binary);
        o2.write(reinterpret_cast<const char*>(g2.data()), (std::streamsize) g2.size() * 2);
        o2.write(reinterpret_cast<const char*>(d2.data()), (std::streamsize) d2.size() * 2);
    }
    return 0;
}
