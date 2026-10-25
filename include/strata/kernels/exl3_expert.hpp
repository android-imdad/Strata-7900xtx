// include/strata/kernels/exl3_expert.hpp - routed experts in EXL3 (exllamav3 trellis) form: the format and the
// arithmetic every device computes them in, bit for bit (the CPU rows, the GPU's VRAM/PCIe groups, the parity replica).
//
// FORMAT (exllamav3, MIT, turboderp; tools/exl3_pack.py writes the blobs).  A matrix with k inputs and n outputs
// (y = x W) is stored as
//   trellis  int16 [k/16][n/16][16 K]: each 16x16 tile is a tail-biting trellis of 256 steps read MSB-first from its
//            8 K uint32 words (little-endian pairs of the int16s).  Integer K: step p's state is the 16-bit window that
//            ends at bit (p + 1) K, modulo the tile's 256 K bits.  Half-integer K = KA + 0.5: the steps alternate KA
//            and KA + 1 bits (odd steps carry the extra bit), so step p ends at bit (p / 2) (2 KA + 1) + KA (p even) or
//            ((p + 1) / 2) (2 KA + 1) (p odd).  Step p sits at tile row (input) (p/8 % 4) * 2 + (p & 1) + 8 (p/2 & 1),
//            column (output) p/32 + 8 (p/4 & 1).
//   codebook mul1: s = the byte sum of state * 0x83DCD12D (mod 2^32), 0..1020; the weight is
//            w' = fp16(1024 + s) * fp16(0x1eee) + fp16(0xc931) - affine in s: w' = A s + B (exllamav3 rounds it to
//            FP16; the arithmetic below keeps the affine form exactly).
//   suh, svh FP16 [k], [n]: input and output channel scales.  With H the 128-point Sylvester Hadamard (unnormalized
//            butterflies) on each 128 consecutive values:  y = H(H(x * suh) W') * svh / 128.
// An expert blob is [trellis | suh | svh] for gate, up, then down; type id 160 + 2 K (EXL3 K2.5 = 165, K3 = 166).
//
// ARITHMETIC (the parity spec; tools/exl3_ref.py is its executable form).  All FP32, no contraction (each product and
// sum rounded on its own), divisions IEEE, round = half away from zero.  For one token and one matrix:
//   1. u_i = x_i * suh_i;  xh = H(u) per 128-block: butterflies at strides 1, 2, 4, ..., 64, each (a + b, a - b) with a
//      the lower index.
//   2. per 128-block j: amax_j = max |xh| (any order: max is exact); d_j = amax_j / QM; q = clamp(round(xh / d_j),
//      -QM, QM) (d_j = 0: q = 0).  Q_j = sum of the block's q (exact int).  QM = act_qmax(): 127 (int8, ~1% expert
//      error) or 16383 (Q15, ~0.01%; STRATA_EXL3_ACT=15).
//   3. per output c and block j: I_jc = sum over the block's 128 inputs of q_i * s_ic (exact int32: |I| <= 128 QM 1020
//      < 2^31; (float) I rounds to nearest even - exact for int8 - the same on every device).
//   4. acc_c = sum over j = 0, 1, ... of (d_j * I_jc), left to right;  qs = sum over j of (d_j * Q_j), left to right;
//      y'_c = A * acc_c + B * qs.
//   5. y = H(y') per 128-block of outputs (as in 1), then y_c = (y_c * (1/128)) * svh_c.
// gate and up each run 1-5 on x with their own suh; h_i = silu(g_i) * up_i with silu(g) = g / (1 + sexp(-g))
// (sexp.hpp); down runs 1-5 on h.  Each token's result depends on nothing but its own x: grouping tokens (a verify
// window) never changes a bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::kernels::exl3 {

constexpr int kTypeBase = 160;
/// The native type id of EXL3 at K bits (K2 = 2 K: 5 for K 2.5, 6 for K 3).
constexpr int type_of(int K2) { return kTypeBase + K2; }
/// 2 K of a native type id, or 0 when it is not an EXL3 id (K 1 .. 8 and the half-integers 1.5 / 2.5 / 3.5).
constexpr int k2_of(int type) {
    const int k2 = type - kTypeBase;
    return (k2 >= 2 && k2 <= 16 && (k2 % 2 == 0 || k2 <= 7)) ? k2 : 0;
}
constexpr bool is_exl3(int type) { return k2_of(type) != 0; }

/// The activations' integer range QM (exl3_expert.hpp step 2): STRATA_EXL3_ACT=15 -> 16383 (Q15), else 127 (int8).
/// Read once; the CPU rows and the GPU groups both use it, so they keep the same bits.
inline int act_qmax() {
    static const int q = [] { const char* v = std::getenv("STRATA_EXL3_ACT"); return v && std::atoi(v) == 15 ? 16383 : 127; }();
    return q;
}

constexpr uint32_t kMul1 = 0x83DCD12Du;
/// The codebook's affine form w' = A s + B: A = fp16(0x1eee), B = 1024 A + fp16(0xc931) (both exact in FP32).
constexpr float kA = 0.00676727294921875f;                 // fp16 0x1eee
constexpr float kBias = -10.3828125f;                       // fp16 0xc931
constexpr float kB = 1024.0f * kA + kBias;                  // exact: 6.9296875 - 10.3828125 = -3.453125

/// Bytes of one role: the trellis (k/16 x n/16 tiles of 16 K2 bytes) + suh (2 k) + svh (2 n).
constexpr size_t trellis_bytes(int64_t k, int64_t n, int K2) { return (size_t) (k / 16) * (size_t) (n / 16) * 16u * (size_t) K2; }
constexpr size_t role_bytes(int64_t k, int64_t n, int K2) { return trellis_bytes(k, n, K2) + 2u * (size_t) (k + n); }

}  // namespace strata::kernels::exl3
