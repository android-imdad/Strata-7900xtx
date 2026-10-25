// exl3_cpu_parity <exl3 pack> <fixture> [--bench]
//
// The CPU's EXL3 experts (exl3_cpu.hpp) bit for bit against tools/exl3_ref.py (the parity spec's executable form):
// each fixture record is one token through one real expert of the pack - x, the SwiGLU output h and the expert output
// y as the reference computes them.  Every record is checked alone and inside token groups of every size 1..8 (the
// group's other tokens are other records' inputs to the same expert), so a kernel whose rows depend on the grouping
// fails here.  --bench: ms per expert and token count on one thread.
//
// The fixture: python tools/exl3_ref.py --pack <pack> --layers 0,4,44 --experts 4 --fixture exl3.fix
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "strata/kernels/cpu/exl3_cpu.hpp"

namespace x = strata::kernels::cpu::exl3c;
namespace ex = strata::kernels::exl3;

namespace {

struct LayerFmt {
    int gk2 = 0, dk2 = 0;
    uint64_t off = 0, blob = 0;
};

bool read_layout(const std::string& pack, std::map<int, LayerFmt>& lay, int& n_expert) {
    std::ifstream in(pack + "/native_experts.txt");
    if (!in) return false;
    std::string line;
    n_expert = 512;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t at = line.find("(n_expert ");
            if (at != std::string::npos) n_expert = std::atoi(line.c_str() + at + 10);
            continue;
        }
        std::istringstream ss(line);
        int l, gt, dt;
        LayerFmt f;
        ss >> l >> gt >> dt >> f.off >> f.blob;
        f.gk2 = ex::k2_of(gt);
        f.dk2 = ex::k2_of(dt);
        if (!f.gk2 || !f.dk2) return false;
        lay[l] = f;
    }
    return true;
}

struct Rec {
    int layer, expert;
    std::vector<float> x, h, y;
};

int count_diff(const float* a, const float* b, int n) {
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += std::memcmp(a + i, b + i, 4) != 0;
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: exl3_cpu_parity <exl3 pack> <fixture> [--bench]\n");
        return 2;
    }
    const std::string pack = argv[1];
    const bool bench = argc > 3 && std::strcmp(argv[3], "--bench") == 0;
    std::map<int, LayerFmt> lay;
    int n_expert = 0;
    if (!read_layout(pack, lay, n_expert)) {
        std::fprintf(stderr, "cannot read an EXL3 native_experts.txt in %s\n", pack.c_str());
        return 2;
    }
    std::ifstream fx(argv[2], std::ios::binary);
    int32_t head[3];
    if (!fx.read(reinterpret_cast<char*>(head), sizeof head)) { std::fprintf(stderr, "bad fixture\n"); return 2; }
    const int nrec = head[0], H = head[1], FF = head[2];
    std::vector<Rec> recs((size_t) nrec);
    for (auto& r : recs) {
        int32_t le[2];
        fx.read(reinterpret_cast<char*>(le), sizeof le);
        r.layer = le[0];
        r.expert = le[1];
        r.x.resize((size_t) H);
        r.h.resize((size_t) FF);
        r.y.resize((size_t) H);
        fx.read(reinterpret_cast<char*>(r.x.data()), 4 * H);
        fx.read(reinterpret_cast<char*>(r.h.data()), 4 * FF);
        fx.read(reinterpret_cast<char*>(r.y.data()), 4 * H);
    }
    if (!fx) { std::fprintf(stderr, "short fixture\n"); return 2; }

    std::ifstream bin(pack + "/experts.bin", std::ios::binary);
    auto load_blob = [&](int l, int e) {
        const LayerFmt& f = lay.at(l);
        std::vector<uint8_t> b(f.blob);
        bin.seekg((std::streamoff) (f.off + (uint64_t) e * f.blob));
        bin.read(reinterpret_cast<char*>(b.data()), (std::streamsize) f.blob);
        return b;
    };

    // one expert over a group of tokens: h and y for each
    auto run = [&](const std::vector<uint8_t>& blob, const LayerFmt& f, const std::vector<const float*>& xs,
                   std::vector<std::vector<float>>& hs, std::vector<std::vector<float>>& ys) {
        const int nt = (int) xs.size();
        const x::Mat g = x::role(blob.data(), 0, H, FF, f.gk2, f.dk2), u = x::role(blob.data(), 1, H, FF, f.gk2, f.dk2),
                     d = x::role(blob.data(), 2, H, FF, f.gk2, f.dk2);
        static x::Act ag[8], au[8], ah[8];
        const x::Act* pg[8];
        const x::Act* pu[8];
        const x::Act* ph[8];
        hs.assign((size_t) nt, std::vector<float>((size_t) FF));
        ys.assign((size_t) nt, std::vector<float>((size_t) H));
        for (int t = 0; t < nt; ++t) {
            x::prep(g, xs[(size_t) t], ag[t]);
            x::prep(u, xs[(size_t) t], au[t]);
            pg[t] = &ag[t];
            pu[t] = &au[t];
        }
        float* hp[8];
        for (int b = 0; b < FF / 128; ++b) {
            for (int t = 0; t < nt; ++t) hp[t] = hs[(size_t) t].data() + 128 * b;
            x::gu_block(g, u, pg, pu, nt, b, hp);
        }
        for (int t = 0; t < nt; ++t) {
            x::prep(d, hs[(size_t) t].data(), ah[t]);
            ph[t] = &ah[t];
        }
        float* yp[8];
        for (int b = 0; b < H / 128; ++b) {
            for (int t = 0; t < nt; ++t) yp[t] = ys[(size_t) t].data() + 128 * b;
            x::block(d, ph, nt, b, yp);
        }
    };

    int fails = 0, checks = 0;
    std::map<std::pair<int, int>, std::vector<int>> by_expert;
    for (int i = 0; i < nrec; ++i) by_expert[{recs[(size_t) i].layer, recs[(size_t) i].expert}].push_back(i);
    const int best = x::isa();
    for (int isa : {0, 2}) {
    if (isa > best) continue;
    x::set_isa(isa);
    std::printf("-- %s path\n", isa == 2 ? "AVX-512" : "scalar");
    for (const auto& [key, idx] : by_expert) {
        const LayerFmt& f = lay.at(key.first);
        const auto blob = load_blob(key.first, key.second);
        // the group: this expert's records, then other records' inputs to fill sizes up to 8
        std::vector<int> pool = idx;
        for (int i = 0; pool.size() < 8 && i < nrec; ++i)
            if (recs[(size_t) i].layer != key.first || recs[(size_t) i].expert != key.second) pool.push_back(i);
        for (int nt = 1; nt <= 8 && nt <= (int) pool.size(); ++nt) {
            for (int rot = 0; rot < (int) idx.size(); ++rot) {
                // the checked record at position rot % nt of the group
                std::vector<int> grp(pool.begin(), pool.begin() + nt);
                const int me = idx[(size_t) rot];
                const int pos = rot % nt;
                for (int t = 0; t < nt; ++t)
                    if (grp[(size_t) t] == me) std::swap(grp[(size_t) t], grp[(size_t) pos]);
                if (grp[(size_t) pos] != me) grp[(size_t) pos] = me;
                std::vector<const float*> xs;
                for (int r : grp) xs.push_back(recs[(size_t) r].x.data());
                std::vector<std::vector<float>> hs, ys;
                run(blob, f, xs, hs, ys);
                const Rec& R = recs[(size_t) me];
                const int bh = count_diff(hs[(size_t) pos].data(), R.h.data(), FF);
                const int by = count_diff(ys[(size_t) pos].data(), R.y.data(), H);
                ++checks;
                if (bh || by) {
                    ++fails;
                    if (fails <= 10)
                        std::printf("FAIL layer %d expert %d, group of %d at %d: h %d/%d, y %d/%d values differ "
                                    "(y[0] %.9g vs %.9g)\n", key.first, key.second, nt, pos, bh, FF, by, H,
                                    ys[(size_t) pos][0], R.y[0]);
                }
            }
        }
        std::printf("layer %2d expert %3d (K %.1f/%.1f): %zu records checked in groups of 1..8\n", key.first,
                    key.second, f.gk2 / 2.0, f.dk2 / 2.0, idx.size());
    }
    }
    x::set_isa(best);
    std::printf("%s: %d of %d checks bit-exact\n", fails ? "FAILED" : "OK", checks - fails, checks);

    if (bench) {
        const auto& [key, idx] = *by_expert.begin();
        const LayerFmt& f = lay.at(key.first);
        const auto blob = load_blob(key.first, key.second);
        for (int nt : {1, 2, 4, 8}) {
            std::vector<const float*> xs;
            for (int t = 0; t < nt; ++t) xs.push_back(recs[(size_t) (t % nrec)].x.data());
            std::vector<std::vector<float>> hs, ys;
            run(blob, f, xs, hs, ys);
            const int reps = 5;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) run(blob, f, xs, hs, ys);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("bench layer %d: %d token(s): %.3f ms per expert (L2-warm weights, one thread)\n", key.first, nt, ms);
        }
    }
    return fails ? 1 : 0;
}
