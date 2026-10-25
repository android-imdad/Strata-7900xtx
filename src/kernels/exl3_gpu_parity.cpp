// exl3_gpu_parity <exl3 pack> <fixture>
// The GPU's EXL3 verify groups (exl3_expert_grouped) bit for bit against tools/exl3_ref.py's fixture (the same one
// exl3_cpu_parity checks the CPU against): each record's expert alone, then all records of one expert as one group
// with the other records' inputs as extra tokens (a group of up to 8 entries), and two groups in one launch.
// --bench [layer]: one layer-window's verify groups - 32 distinct experts, 40 entries (a 4-token window, top-10) - timed
// over 200 launches back to back (us per call).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strata/kernels/exl3_expert.hpp"
#include "strata/kernels/exl3_kernels.hpp"

namespace {
struct LayerFmt { int gt = 0, dt = 0; unsigned long long off = 0, blob = 0; };
struct Rec { int layer, expert; std::vector<float> x, h, y; };
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: exl3_gpu_parity <pack> <fixture> [--bench [layer]]\n"); return 2; }
    const bool bench = argc > 3 && std::strcmp(argv[3], "--bench") == 0;
    const int bench_layer = argc > 4 ? std::atoi(argv[4]) : 4;
    const std::string pack = argv[1];
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
    std::ifstream fx(argv[2], std::ios::binary);
    int32_t head[4];
    fx.read(reinterpret_cast<char*>(head), sizeof head);
    const int nrec = head[0], H = head[1], F = head[2];
    if (head[3] != strata::kernels::exl3::act_qmax()) {
        std::fprintf(stderr, "the fixture was made for activations up to %d, this run uses %d (STRATA_EXL3_ACT)\n", head[3],
                     strata::kernels::exl3::act_qmax());
        return 2;
    }
    std::vector<Rec> recs((size_t) nrec);
    for (auto& r : recs) {
        int32_t le[2];
        fx.read(reinterpret_cast<char*>(le), sizeof le);
        r.layer = le[0]; r.expert = le[1];
        r.x.resize((size_t) H); r.h.resize((size_t) F); r.y.resize((size_t) H);
        fx.read(reinterpret_cast<char*>(r.x.data()), 4 * H);
        fx.read(reinterpret_cast<char*>(r.h.data()), 4 * F);
        fx.read(reinterpret_cast<char*>(r.y.data()), 4 * H);
    }
    std::ifstream bin(pack + "/experts.bin", std::ios::binary);
    const int CAP = 16;
    void *d_blob[2], *d_scratch;
    float *d_x, *d_out;
    int32_t* d_i32;
    unsigned long long* d_ptr;
    size_t maxblob = 0;
    for (auto& [l, f] : lay) maxblob = std::max<size_t>(maxblob, f.blob);
    cudaMalloc(&d_blob[0], maxblob); cudaMalloc(&d_blob[1], maxblob);
    cudaMalloc(&d_scratch, strata::kernels::exl3_expert_scratch_bytes(CAP, H, F));
    cudaMalloc(&d_x, (size_t) CAP * H * 4); cudaMalloc(&d_out, (size_t) CAP * H * 4);
    cudaMalloc(&d_i32, 4 * 64); cudaMalloc(&d_ptr, 8 * 4);
    auto load = [&](int l, int e, void* dst) {
        const LayerFmt& f = lay.at(l);
        std::vector<uint8_t> b(f.blob);
        bin.seekg((std::streamoff) (f.off + (unsigned long long) e * f.blob));
        bin.read(reinterpret_cast<char*>(b.data()), (std::streamsize) f.blob);
        cudaMemcpy(dst, b.data(), f.blob, cudaMemcpyHostToDevice);
    };
    // groups: list of (blob slot, record indices as entries); token t = entry index, dst row = entry index
    int offset = 0;
    auto run = [&](const LayerFmt& f, const std::vector<std::vector<int>>& groups, std::vector<std::vector<float>>& ys) {
        const int base = offset;   // entries [base, base + n): a PCIe pass's follow the VRAM pass's
        std::vector<int32_t> start{base}, ent;
        std::vector<float> xs;
        for (const auto& g : groups) {
            for (int r : g) { ent.push_back((int32_t) ent.size()); xs.insert(xs.end(), recs[(size_t) r].x.begin(), recs[(size_t) r].x.end()); }
            start.push_back((int32_t) (base + ent.size()));
        }
        const int ng = (int) groups.size(), ne = (int) ent.size();
        std::vector<int32_t> i32(64, 0);
        i32[0] = ng;
        for (int i = 0; i <= ng; ++i) i32[1 + i] = start[(size_t) i];
        for (int e = 0; e < base; ++e) { i32[8 + e] = 0; i32[32 + e] = 0; }   // another pass's entries: not this launch's
        for (int e = 0; e < ne; ++e) { i32[8 + base + e] = e; i32[32 + base + e] = e; }   // dst, tok
        cudaMemcpy(d_i32, i32.data(), 4 * 64, cudaMemcpyHostToDevice);
        unsigned long long p[2] = {(unsigned long long) d_blob[0], (unsigned long long) d_blob[1]};
        cudaMemcpy(d_ptr, p, 16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0xFF, (size_t) CAP * H * 4);
        strata::kernels::exl3_expert_grouped(f.gt, f.dt, H, F, d_ptr, d_i32 + 1, d_i32, d_i32 + 8, d_i32 + 32, 4, CAP,
                                             d_x, d_scratch, d_out, nullptr);
        std::vector<float> out((size_t) ne * H);
        cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost);
        ys.assign((size_t) ne, {});
        for (int e = 0; e < ne; ++e) ys[(size_t) e].assign(out.begin() + (size_t) e * H, out.begin() + (size_t) (e + 1) * H);
    };
    std::map<std::pair<int, int>, std::vector<int>> by;
    for (int i = 0; i < nrec; ++i) by[{recs[(size_t) i].layer, recs[(size_t) i].expert}].push_back(i);
    int checks = 0, fails = 0;
    auto cmp = [&](const std::vector<float>& got, int r, const char* what) {
        ++checks;
        int bad = 0;
        for (int i = 0; i < H; ++i) bad += std::memcmp(&got[(size_t) i], &recs[(size_t) r].y[(size_t) i], 4) != 0;
        if (bad) {
            ++fails;
            if (fails <= 10) std::printf("FAIL %s: layer %d expert %d: %d/%d differ (y0 %.9g vs %.9g)\n", what, recs[(size_t) r].layer,
                                         recs[(size_t) r].expert, bad, H, got[0], recs[(size_t) r].y[0]);
        }
    };
    std::pair<int, int> prev{-1, -1};
    for (const auto& [key, idx] : by) {
        const LayerFmt& f = lay.at(key.first);
        load(key.first, key.second, d_blob[0]);
        std::vector<std::vector<float>> ys;
        for (int r : idx) { run(f, {{r}}, ys); cmp(ys[0], r, "alone"); }
        // a group of 8: this expert's records first, then other records' inputs (their outputs are this expert's, not checked)
        std::vector<int> grp = idx;
        for (int i = 0; (int) grp.size() < 8 && i < nrec; ++i) if (recs[(size_t) i].layer != key.first || recs[(size_t) i].expert != key.second) grp.push_back(i);
        run(f, {grp}, ys);
        for (size_t i = 0; i < idx.size(); ++i) cmp(ys[i], idx[i], "group of 8");
        offset = 3;   // the same group as a launch whose entries start at 3 (a PCIe pass after a VRAM one)
        run(f, {grp}, ys);
        offset = 0;
        for (size_t i = 0; i < idx.size(); ++i) cmp(ys[i], idx[i], "group of 8 at entry 3");
        // two groups in one launch: the previous expert of the same layer in slot 1
        if (prev.first == key.first) {
            load(prev.first, prev.second, d_blob[1]);
            const auto& pidx = by.at(prev);
            run(f, {idx, pidx}, ys);
            for (size_t i = 0; i < idx.size(); ++i) cmp(ys[i], idx[i], "two groups");
            for (size_t i = 0; i < pidx.size(); ++i) cmp(ys[idx.size() + i], pidx[i], "two groups (2nd)");
        }
        prev = key;
        std::printf("layer %2d expert %3d (types %d/%d): checked\n", key.first, key.second, f.gt, f.dt);
    }
    std::printf("%s: %d of %d checks bit-exact\n", fails ? "FAILED" : "OK", checks - fails, checks);
    if (bench) {
        const LayerFmt& f = lay.at(bench_layer);
        const int NG = 32, NE = 40, CAPB = 80;
        void* dblobs = nullptr;
        cudaMalloc(&dblobs, (size_t) NG * f.blob);
        for (int g = 0; g < NG; ++g) load(bench_layer, 13 * g + 1, (uint8_t*) dblobs + (size_t) g * f.blob);
        std::vector<unsigned long long> ptr((size_t) NG);
        for (int g = 0; g < NG; ++g) ptr[(size_t) g] = (unsigned long long) ((uint8_t*) dblobs + (size_t) g * f.blob);
        std::vector<int32_t> hb(1 + (NG + 1) + 2 * CAPB, 0);
        hb[0] = NG;
        int e = 0;
        for (int g = 0; g <= NG; ++g) { hb[(size_t) (1 + g)] = e; if (g < NG) e += g < NE - NG ? 2 : 1; }   // 8 experts take 2 tokens
        for (int i = 0; i < NE; ++i) { hb[(size_t) (1 + NG + 1 + i)] = i; hb[(size_t) (1 + NG + 1 + CAPB + i)] = i % 4; }
        int32_t* db = nullptr;
        unsigned long long* dp = nullptr;
        void* dsc = nullptr;
        float *dx = nullptr, *dout = nullptr;
        cudaMalloc(&db, hb.size() * 4); cudaMalloc(&dp, ptr.size() * 8);
        cudaMalloc(&dsc, strata::kernels::exl3_expert_scratch_bytes(CAPB, H, F));
        cudaMalloc(&dx, (size_t) 4 * H * 4); cudaMalloc(&dout, (size_t) CAPB * H * 4);
        cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dp, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice);
        std::vector<float> xs((size_t) 4 * H);
        for (int t = 0; t < 4; ++t) std::memcpy(xs.data() + (size_t) t * H, recs[(size_t) (t % nrec)].x.data(), (size_t) H * 4);
        cudaMemcpy(dx, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice);
        cudaStream_t st;
        cudaStreamCreate(&st);
        auto call = [&] {
            strata::kernels::exl3_expert_grouped(f.gt, f.dt, H, F, dp, db + 1, db, db + 1 + NG + 1, db + 1 + NG + 1 + CAPB, CAPB,
                                                 CAPB, dx, dsc, dout, st);
        };
        for (int i = 0; i < 20; ++i) call();
        cudaStreamSynchronize(st);
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        const int reps = 200;
        cudaEventRecord(a, st);
        for (int i = 0; i < reps; ++i) call();
        cudaEventRecord(b, st);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("bench layer %d (types %d/%d): %d experts, %d entries: %.1f us per layer-window (%s)\n", bench_layer,
                    f.gt, f.dt, NG, NE, 1000.0 * ms / reps, std::getenv("STRATA_EXL3_V1") ? "v1" : std::getenv("STRATA_EXL3_V2") ? "v2" : "v3");
    }
    return fails ? 1 : 0;
}
