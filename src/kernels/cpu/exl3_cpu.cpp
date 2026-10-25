// src/kernels/cpu/exl3_cpu.cpp - EXL3 routed experts on the CPU (exl3_cpu.hpp), in exl3_expert.hpp's arithmetic.
// Compiled with -ffp-contract=off: every product and sum is rounded on its own, as the spec (and the GPU) does.
#include "strata/kernels/cpu/exl3_cpu.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/sexp.hpp"

namespace strata::kernels::cpu::exl3c {
namespace {

float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {   // subnormal: normalize
            int ee = 113;
            uint32_t mm = m;
            while (!(mm & 0x400u)) { mm <<= 1; --ee; }
            bits = s | ((uint32_t) ee << 23) | ((mm & 0x3FFu) << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7F800000u | (m << 13);
    } else {
        bits = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// the 128-point Hadamard on v (unnormalized butterflies, strides 1..64, (a + b, a - b))
void had128(float* v) {
    for (int h = 1; h < 128; h *= 2)
        for (int g = 0; g < 128; g += 2 * h)
            for (int i = g; i < g + h; ++i) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
}

float round_away(float t) {
    const float tr = std::trunc(t);
    return std::fabs(t - tr) >= 0.5f ? tr + std::copysign(1.0f, t) : tr;
}

// tile step p: its row (input) and column (output) inside the 16 x 16 tile
struct StepPos {
    uint8_t row[256], col[256];
    StepPos() {
        for (int p = 0; p < 256; ++p) {
            row[p] = (uint8_t) (((p / 8) % 4) * 2 + (p & 1) + 8 * ((p / 2) & 1));
            col[p] = (uint8_t) (p / 32 + 8 * ((p / 4) & 1));
        }
    }
};
const StepPos kPos;

// end bit (exclusive) of step p at 2K = K2
inline int step_end(int p, int K2) {
    if ((K2 & 1) == 0) return (p + 1) * (K2 / 2);
    const int KA = K2 / 2;
    return (p & 1) == 0 ? (p / 2) * (2 * KA + 1) + KA : ((p + 1) / 2) * (2 * KA + 1);
}

// s[row][col] of one tile: the codebook index (byte sum of state * mul1) of each of its 256 weights
void decode_tile(const uint32_t* w, int K2, int16_t s[16][16]) {
    const int words = 4 * K2, nbits = 32 * words;
    for (int p = 0; p < 256; ++p) {
        int start = step_end(p, K2) - 16;
        if (start < 0) start += nbits;
        const int wi = start >> 5, off = start & 31;
        const uint64_t v = ((uint64_t) w[wi] << 32) | w[(wi + 1) % words];
        const uint32_t state = (uint32_t) (v >> (48 - off)) & 0xFFFFu;
        const uint32_t x = state * exl3::kMul1;
        s[kPos.row[p]][kPos.col[p]] = (int16_t) ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24));
    }
}

}  // namespace

void prep_avx512(const Mat& m, const float* x, Act& a);
void block_avx512(const Mat& m, const Act* const* a, int nt, int b, float* const* y);

namespace {
int g_isa = -1;
}

int isa() {
    if (g_isa < 0) g_isa = cpu_avx512_ok() && std::getenv("STRATA_EXL3_SCALAR") == nullptr ? 2 : 0;
    return g_isa;
}
void set_isa(int i) { g_isa = i; }

Mat role(const uint8_t* blob, int r, int n_embd, int n_ff, int gu_k2, int d_k2) {
    size_t at = 0;
    Mat m;
    for (int i = 0; i <= r; ++i) {
        m.k = i < 2 ? n_embd : n_ff;
        m.n = i < 2 ? n_ff : n_embd;
        m.K2 = i < 2 ? gu_k2 : d_k2;
        const size_t tb = exl3::trellis_bytes(m.k, m.n, m.K2);
        m.trellis = blob + at;
        m.suh = reinterpret_cast<const uint16_t*>(blob + at + tb);
        m.svh = reinterpret_cast<const uint16_t*>(blob + at + tb + 2 * (size_t) m.k);
        at += exl3::role_bytes(m.k, m.n, m.K2);
    }
    return m;
}

void prep(const Mat& m, const float* x, Act& a) {
    if (isa() == 2) { prep_avx512(m, x, a); return; }
    float qs = 0.f;
    for (int j = 0; j < m.k / 128; ++j) {
        float v[128];
        for (int i = 0; i < 128; ++i) v[i] = x[128 * j + i] * h2f(m.suh[128 * j + i]);
        had128(v);
        float amax = 0.f;
        for (int i = 0; i < 128; ++i) amax = std::fmax(amax, std::fabs(v[i]));
        const int qm = exl3::act_qmax();
        const float qmf = (float) qm;
        const float d = amax / qmf;
        int Q = 0;
        for (int i = 0; i < 128; ++i) {
            float q = d > 0.f ? round_away(v[i] / d) : 0.f;
            q = std::fmin(std::fmax(q, -qmf), qmf);
            a.q[128 * j + i] = (int16_t) q;
            a.q4[128 * j + i] = qm == 127 ? (int32_t) (((uint32_t) (uint8_t) (int8_t) q) * 0x01010101u)
                                          : (int32_t) ((uint32_t) (uint16_t) (int16_t) q);
            Q += (int) q;
        }
        a.d[j] = d;
        qs = qs + d * (float) Q;
    }
    a.qs = qs;
}

void block(const Mat& m, const Act* const* a, int nt, int b, float* const* y) {
    if (isa() == 2) { block_avx512(m, a, nt, b, y); return; }
    const int tn_per = m.n / 16, tile_bytes = 16 * m.K2;
    float acc[8][128];
    for (int t = 0; t < nt; ++t)
        for (int c = 0; c < 128; ++c) acc[t][c] = 0.f;
    int16_t s[16][16];
    for (int j = 0; j < m.k / 128; ++j) {
        int32_t I[8][128] = {};
        for (int tk = 8 * j; tk < 8 * j + 8; ++tk)
            for (int tc = 0; tc < 8; ++tc) {
                const int tn = 8 * b + tc;
                decode_tile(reinterpret_cast<const uint32_t*>(m.trellis + ((size_t) tk * tn_per + tn) * tile_bytes),
                            m.K2, s);
                for (int t = 0; t < nt; ++t) {
                    const int16_t* q = a[t]->q + 16 * tk;
                    for (int r = 0; r < 16; ++r)
                        for (int c = 0; c < 16; ++c) I[t][16 * tc + c] += (int32_t) q[r] * s[r][c];
                }
            }
        for (int t = 0; t < nt; ++t) {
            const float d = a[t]->d[j];
            for (int c = 0; c < 128; ++c) acc[t][c] = acc[t][c] + d * (float) I[t][c];
        }
    }
    for (int t = 0; t < nt; ++t) {
        float v[128];
        const float bq = exl3::kB * a[t]->qs;
        for (int c = 0; c < 128; ++c) v[c] = exl3::kA * acc[t][c] + bq;
        had128(v);
        for (int c = 0; c < 128; ++c) y[t][c] = (v[c] * (1.0f / 128.0f)) * h2f(m.svh[128 * b + c]);
    }
}

void gu_block(const Mat& g, const Mat& u, const Act* const* ag, const Act* const* au, int nt, int b, float* const* h) {
    float gv[8][128], uv[8][128];
    float* gp[8];
    float* up[8];
    for (int t = 0; t < nt; ++t) { gp[t] = gv[t]; up[t] = uv[t]; }
    block(g, ag, nt, b, gp);
    block(u, au, nt, b, up);
    for (int t = 0; t < nt; ++t)
        for (int c = 0; c < 128; ++c) h[t][c] = (gv[t][c] / (1.0f + strata::kernels::sexp(-gv[t][c]))) * uv[t][c];
}

}  // namespace strata::kernels::cpu::exl3c
