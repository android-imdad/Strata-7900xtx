// include/strata/kernels/cpu/exl3_cpu.hpp - EXL3 routed experts on the CPU, in exl3_expert.hpp's arithmetic (the parity
// spec): bit for bit tools/exl3_ref.py, and the GPU's EXL3 expert kernels.
//
// A token's input to one matrix is prepared once (exl3_prep: rotation by suh and the Hadamard, int8 per 128-block),
// then any 128-output block of the matrix is one call (exl3_block), so the pool can cut an expert's outputs into
// blocks across threads.  Every block of every token depends on nothing but that token's prepared input: grouping
// tokens never changes a bit.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/kernels/exl3_expert.hpp"

namespace strata::kernels::cpu::exl3c {

/// The widest EXL3 input this code takes (Flash-Next's hidden size).
inline constexpr int kMaxK = 2560;

/// One token's prepared input to one matrix: int8 values, the per-128-block scale and the scale-weighted sum.
struct Act {
    alignas(64) int8_t q[kMaxK];
    alignas(64) int32_t q4[kMaxK];   ///< each q repeated in all four bytes (the AVX-512 VNNI dot's signed operand)
    float d[kMaxK / 128];
    float qs = 0.f;          ///< sum over blocks j of d_j * (sum of block j's q), left to right
};

/// One matrix of an expert blob.
struct Mat {
    const uint8_t* trellis = nullptr;
    const uint16_t* suh = nullptr;   ///< FP16 bits [k]
    const uint16_t* svh = nullptr;   ///< FP16 bits [n]
    int k = 0, n = 0, K2 = 0;
};

/// Role r (0 gate, 1 up, 2 down) of an expert blob [trellis | suh | svh] x 3 (tools/exl3_pack.py).
Mat role(const uint8_t* blob, int r, int n_embd, int n_ff, int gu_k2, int d_k2);

/// Steps 1-2: x (m.k floats) rotated by suh, the Hadamard, int8 per 128-block.
void prep(const Mat& m, const float* x, Act& a);

/// Steps 3-5 for the 128 outputs of block b (outputs [128 b, 128 b + 128)) of nt tokens: y[t][0..128).
void block(const Mat& m, const Act* const* a, int nt, int b, float* const* y);

/// The path prep / block take: 2 AVX-512 (F, BW, VL, VNNI, VBMI), 0 scalar (STRATA_EXL3_SCALAR=1 forces it) - the
/// same bits either way.
int isa();
void set_isa(int isa);   // tests

/// gate and up block b, then h = silu(g) * up (silu(g) = g / (1 + sexp(-g))): h[t][0..128).
void gu_block(const Mat& g, const Mat& u, const Act* const* ag, const Act* const* au, int nt, int b, float* const* h);

}  // namespace strata::kernels::cpu::exl3c
