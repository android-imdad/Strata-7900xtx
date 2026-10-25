// src/kernels/cpu/exl3_avx512.cpp - the EXL3 expert rows of exl3_cpu.cpp in AVX-512 (F, BW, VL, VNNI, VBMI): the same
// bits as the scalar code (exl3_cpu_parity checks both against tools/exl3_ref.py).  Compiled with -ffp-contract=off.
//
// A tile's bits become a big-endian byte stream (one vpshufb per 64 bytes); a tile row's 16 states are one vpermi2b
// (each lane's three stream bytes, the index wrapping past the stream's end - the trellis is tail-biting) and one
// variable shift.  The codebook index s is the byte sum of state * 0x83DCD12D, so q s is one vpdpbusd of the
// product's four bytes against the token's int8 q repeated four times: exact integers, in any order.
#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "strata/kernels/cpu/exl3_cpu.hpp"

namespace strata::kernels::cpu::exl3c {

namespace {

// per 2K: for each tile row r and output lane c, the vpermi2b index (bytes B0 B1 B2 of the state's window, placed so
// the dword is B0 << 24 | B1 << 16 | B2 << 8) and the right shift that leaves the state in the low 16 bits
struct Tables {
    alignas(64) uint8_t idx[17][16][64];
    alignas(64) uint32_t shr[17][16][16];
    bool ready[17] = {};
};
Tables g_tab;

void build(int K2) {
    if (g_tab.ready[K2]) return;
    const int nbytes = 16 * K2, nbits = 8 * nbytes;
    for (int p = 0; p < 256; ++p) {
        const int r = ((p / 8) % 4) * 2 + (p & 1) + 8 * ((p / 2) & 1);
        const int c = p / 32 + 8 * ((p / 4) & 1);
        int end;
        if ((K2 & 1) == 0) end = (p + 1) * (K2 / 2);
        else end = (p & 1) == 0 ? (p / 2) * (K2 | 1) + K2 / 2 : ((p + 1) / 2) * (K2 | 1);
        int start = end - 16;
        if (start < 0) start += nbits;
        const int b0 = start >> 3, off = start & 7;
        auto at = [&](int b) { return (uint8_t) (b % nbytes); };   // tail-biting: past the end wraps to the start
        uint8_t* ix = g_tab.idx[K2][r] + 4 * c;
        ix[0] = at(b0 + 2);   // the low byte is shifted out; any byte will do
        ix[1] = at(b0 + 2);
        ix[2] = at(b0 + 1);
        ix[3] = at(b0);
        g_tab.shr[K2][r][c] = (uint32_t) (16 - off);
    }
    g_tab.ready[K2] = true;
}

// the 128-point Hadamard on 8 registers (v[0..7] = values 0..127), each butterfly (a + b, a - b) as the scalar code
inline void had128(__m512 v[8]) {
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (int h = 1; h < 16; h *= 2) {
        const __m512i perm = _mm512_xor_si512(lane, _mm512_set1_epi32(h));
        const __mmask16 lower = _mm512_testn_epi32_mask(lane, _mm512_set1_epi32(h));
        for (int i = 0; i < 8; ++i) {
            const __m512 p = _mm512_permutexvar_ps(perm, v[i]);
            // lower lane i: v[i] + v[i + h]; upper lane i + h: v[i] - v[i + h] = p - v
            v[i] = _mm512_mask_blend_ps(lower, _mm512_sub_ps(p, v[i]), _mm512_add_ps(v[i], p));
        }
    }
    for (int h = 1; h < 8; h *= 2)
        for (int g = 0; g < 8; g += 2 * h)
            for (int i = g; i < g + h; ++i) {
                const __m512 a = v[i], b = v[i + h];
                v[i] = _mm512_add_ps(a, b);
                v[i + h] = _mm512_sub_ps(a, b);
            }
}

inline __m512 load_h(const uint16_t* p) { return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))); }

}  // namespace

void prep_avx512(const Mat& m, const float* x, Act& a) {
    const __m512 absm = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    float qs = 0.f;
    for (int j = 0; j < m.k / 128; ++j) {
        __m512 v[8];
        for (int i = 0; i < 8; ++i)
            v[i] = _mm512_mul_ps(_mm512_loadu_ps(x + 128 * j + 16 * i), load_h(m.suh + 128 * j + 16 * i));
        had128(v);
        __m512 am = _mm512_and_ps(v[0], absm);
        for (int i = 1; i < 8; ++i) am = _mm512_max_ps(am, _mm512_and_ps(v[i], absm));
        const float amax = _mm512_reduce_max_ps(am);
        const float d = amax / 127.0f;
        __m512i qsum = _mm512_setzero_si512();
        for (int i = 0; i < 8; ++i) {
            __m512 r = _mm512_setzero_ps();
            if (d > 0.f) {
                const __m512 t = _mm512_div_ps(v[i], _mm512_set1_ps(d));
                const __m512 tr = _mm512_roundscale_ps(t, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                const __m512 fr = _mm512_and_ps(_mm512_sub_ps(t, tr), absm);
                const __mmask16 up = _mm512_cmp_ps_mask(fr, _mm512_set1_ps(0.5f), _CMP_GE_OQ);
                const __m512 one = _mm512_or_ps(_mm512_set1_ps(1.0f), _mm512_andnot_ps(absm, t));
                r = _mm512_mask_add_ps(tr, up, tr, one);
                r = _mm512_min_ps(_mm512_max_ps(r, _mm512_set1_ps(-127.0f)), _mm512_set1_ps(127.0f));
            }
            const __m512i qi = _mm512_cvtps_epi32(r);
            qsum = _mm512_add_epi32(qsum, qi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(a.q + 128 * j + 16 * i), _mm512_cvtsepi32_epi8(qi));
            _mm512_storeu_si512(a.q4 + 128 * j + 16 * i, _mm512_mullo_epi32(_mm512_and_si512(qi, _mm512_set1_epi32(0xFF)),
                                                                           _mm512_set1_epi32(0x01010101)));
        }
        a.d[j] = d;
        qs = qs + d * (float) _mm512_reduce_add_epi32(qsum);
    }
    a.qs = qs;
}

void block_avx512(const Mat& m, const Act* const* a, int nt, int b, float* const* y) {
    build(m.K2);
    const int tn_per = m.n / 16, tile_bytes = 16 * m.K2;
    const __mmask64 hi_mask = tile_bytes > 64 ? (~0ull >> (128 - tile_bytes)) : 0;
    const __m512i bswap = _mm512_set4_epi32(0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203);
    const __m512i mul1 = _mm512_set1_epi32((int) exl3::kMul1), lo16 = _mm512_set1_epi32(0xFFFF);
    alignas(64) float acc[8][8][16];   // [token][tile column][lane]
    alignas(64) int32_t I[8][8][16];
    for (int t = 0; t < nt; ++t)
        for (int tc = 0; tc < 8; ++tc) _mm512_store_ps(acc[t][tc], _mm512_setzero_ps());
    const uint8_t (*idx)[64] = g_tab.idx[m.K2];
    const uint32_t (*shr)[16] = g_tab.shr[m.K2];
    for (int j = 0; j < m.k / 128; ++j) {
        for (int t = 0; t < nt; ++t)
            for (int tc = 0; tc < 8; ++tc) _mm512_store_si512(I[t][tc], _mm512_setzero_si512());
        for (int tk = 8 * j; tk < 8 * j + 8; ++tk) {
            const uint8_t* row0 = m.trellis + ((size_t) tk * tn_per + 8 * b) * tile_bytes;
            for (int tc = 0; tc < 8; ++tc) {
                const uint8_t* tp = row0 + (size_t) tc * tile_bytes;
                const __m512i s0 = _mm512_shuffle_epi8(_mm512_loadu_si512(tp), bswap);
                const __m512i s1 = _mm512_shuffle_epi8(_mm512_maskz_loadu_epi8(hi_mask, tp + 64), bswap);
                __m512i w[16];
                for (int r = 0; r < 16; ++r) {
                    const __m512i d = _mm512_permutex2var_epi8(s0, _mm512_load_si512(idx[r]), s1);
                    const __m512i st = _mm512_and_si512(_mm512_srlv_epi32(d, _mm512_load_si512(shr[r])), lo16);
                    w[r] = _mm512_mullo_epi32(st, mul1);
                }
                for (int t = 0; t < nt; ++t) {
                    const int32_t* q4 = a[t]->q4 + 16 * tk;
                    __m512i s = _mm512_load_si512(I[t][tc]);
                    for (int r = 0; r < 16; ++r) s = _mm512_dpbusd_epi32(s, w[r], _mm512_set1_epi32(q4[r]));
                    _mm512_store_si512(I[t][tc], s);
                }
            }
        }
        for (int t = 0; t < nt; ++t) {
            const __m512 d = _mm512_set1_ps(a[t]->d[j]);
            for (int tc = 0; tc < 8; ++tc)
                _mm512_store_ps(acc[t][tc], _mm512_add_ps(_mm512_load_ps(acc[t][tc]),
                                                          _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_load_si512(I[t][tc])))));
        }
    }
    const __m512 A = _mm512_set1_ps(exl3::kA), inv = _mm512_set1_ps(1.0f / 128.0f);
    for (int t = 0; t < nt; ++t) {
        const __m512 bq = _mm512_set1_ps(exl3::kB * a[t]->qs);
        __m512 v[8];
        for (int tc = 0; tc < 8; ++tc) v[tc] = _mm512_add_ps(_mm512_mul_ps(A, _mm512_load_ps(acc[t][tc])), bq);
        had128(v);
        for (int i = 0; i < 8; ++i)
            _mm512_storeu_ps(y[t] + 16 * i, _mm512_mul_ps(_mm512_mul_ps(v[i], inv), load_h(m.svh + 128 * b + 16 * i)));
    }
}

}  // namespace strata::kernels::cpu::exl3c
