// include/strata/kernels/exl3_kernels.hpp - EXL3 routed experts on the GPU (exl3_expert.hpp's format).
//
// The prompt path's dequantizer: the full weight in the original basis, W = diag(suh) H W' H diag(svh) / 128 (both
// Hadamards block-diagonal over 128 values, so each 128 x 128 block of W comes from the same block of W'), as FP16 in
// the engine's row layout (row = one output, k values).  Not part of the parity arithmetic: the prompt path's GEMMs
// are not the decode path's.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// One expert's gate and up matrices into the prompt path's interleaved FP16 layout (row 2r = gate row r, row 2r + 1 =
/// up row r, n_embd values each); gu_type is the layer's EXL3 type id (exl3_expert.hpp).
void exl3_dequant_gu_f16(int gu_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                         void* stream);
/// One expert's down matrix: n_embd rows of n_ff values.
void exl3_dequant_down_f16(int d_type, const void* down, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream);

}  // namespace strata::kernels
