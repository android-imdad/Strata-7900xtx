// include/strata/kernels/exl3_kernels.hpp - EXL3 routed experts on the GPU (exl3_expert.hpp's format).
//
// The prompt path's dequantizer: the full weight in the original basis, W = diag(suh) H W' H diag(svh) / 128 (both
// Hadamards block-diagonal over 128 values, so each 128 x 128 block of W comes from the same block of W'), as FP16 in
// the engine's row layout (row = one output, k values).  Not part of the parity arithmetic: the prompt path's GEMMs
// are not the decode path's.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// One expert's gate and up matrices into the prompt path's interleaved FP16 layout (row 2r = gate row r, row 2r + 1 =
/// up row r, n_embd values each); gu_type is the layer's EXL3 type id (exl3_expert.hpp).
void exl3_dequant_gu_f16(int gu_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                         void* stream);
/// One expert's down matrix: n_embd rows of n_ff values.
void exl3_dequant_down_f16(int d_type, const void* down, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream);

}  // namespace strata::kernels

namespace strata::kernels {

/// Bytes of scratch exl3_expert_grouped needs for `cap` entries (n_embd, n_ff as the layer's).
size_t exl3_expert_scratch_bytes(int64_t cap, int64_t n_embd, int64_t n_ff);

/// A verify window's EXL3 experts in exl3_expert.hpp's arithmetic - bit for bit the CPU's (exl3_cpu.hpp): group g's
/// blob at grp_ptr[g]; its entries [grp_start[g], grp_start[g + 1]) (g < *n_groups, read on the device) take token
/// ent_tok[e]'s F32 input (x + ent_tok[e] * n_embd) and write row ent_dst[e] of `out` (n_embd floats).  gu_type /
/// d_type: the layer's EXL3 ids.
void exl3_expert_grouped(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const unsigned long long* grp_ptr,
                         const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                         const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const float* x, void* scratch,
                         float* out, void* stream);

}  // namespace strata::kernels
