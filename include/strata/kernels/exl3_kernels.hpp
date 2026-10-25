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
/// Zero the scratch's counters (the fused band kernels count their waves modulo, so this runs once, after the
/// scratch is allocated and before any graph capture).
void exl3_expert_scratch_init(void* scratch, int64_t cap, int64_t n_embd, int64_t n_ff);
/// (debug) STRATA_EXL3_V4DBG=128: the last v4 launch's per-block timeline as text (block kind g loc t0 t1 t2 t3)
void exl3_v4_timeline_dump(const char* path, int nblocks);

/// A verify window's EXL3 experts in exl3_expert.hpp's arithmetic - bit for bit the CPU's (exl3_cpu.hpp): group g's
/// blob at grp_ptr[g]; its entries [grp_start[g], grp_start[g + 1]) (g < *n_groups, read on the device) take token
/// ent_tok[e]'s F32 input (x + ent_tok[e] * n_embd) and write row ent_dst[e] of `out` (n_embd floats).  gu_type /
/// d_type: the layer's EXL3 ids.  group_rows (0: cap_groups): the launch's groups side by side, each block striding
/// over the rest (a PCIe pass: a few groups of cap).
void exl3_expert_grouped(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const unsigned long long* grp_ptr,
                         const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                         const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const float* x, void* scratch,
                         float* out, void* stream, int64_t group_rows = 0);

}  // namespace strata::kernels

namespace strata::kernels {

/// The prompt path's EXL3 weights with only the INPUT rotation folded in: W_in = diag(suh) H W' (an independent
/// 128-point transform per column - a thread a column block, no shared memory), as FP16 in the engine's row layout
/// (gate/up interleaved as exl3_dequant_gu_f16's), one launch for an expert's three matrices.  The output side
/// (y = H(x W_in) * svh / 128) runs on the products: exl3_rotate_gu_out / exl3_rotate_d_out.  Each matrix's svh
/// goes after its weights: dst_gu + 2 n_ff n_embd holds svh_gate [n_ff] then svh_up [n_ff], dst_d + n_embd n_ff
/// holds svh_down [n_embd] (so the dequant buffers take kExl3GuTail / kExl3DTail more halves).
void exl3_dequant_left_f16(int gu_type, int d_type, const void* blob, int64_t n_ff, int64_t n_embd, uint16_t* dst_gu,
                           uint16_t* dst_d, void* stream);
/// gate/up products [rows][2 n_ff] (interleaved: column 2 r gate r, 2 r + 1 up r), in place: each half's output
/// rotation and svh / 128 (svh_gu: dst_gu's tail).
void exl3_rotate_gu_out(float* gu, int64_t rows, int64_t n_ff, const uint16_t* svh_gu, void* stream);
/// down products [rows][n_embd], in place: the output rotation and svh / 128 (svh_d: dst_d's tail).
void exl3_rotate_d_out(float* dm, int64_t rows, int64_t n_embd, const uint16_t* svh_d, void* stream);

}  // namespace strata::kernels

namespace strata::kernels {

/// The prompt path's EXL3 experts, a group at a time: up to 16 experts of one layer, each over its rows of the chunk
/// (Xs FP16 [rows][n_embd] in, Dm F32 [rows][n_embd] out - before that its rows hold the rotated inputs; GU F32
/// [rows][2 n_ff], Hh FP16 [rows][n_ff] and rs F32 [rows][3] are scratch rows of the same chunk).  The trellis is decoded inside FP16 WMMA tiles (no dequantized weights): the
/// activations are rotated per expert (x' = H(x * suh)), the weights enter as 1024 + s.  The caller fills blob, off,
/// cnt and n; rel and mc0 are computed here.
struct Exl3PfGroup {
    const uint8_t* blob[16];
    int32_t off[16];   // expert e's first row in the chunk
    int32_t cnt[16];   // its rows
    int32_t rel[17];   // the group's rows: expert e's at [rel[e], rel[e + 1]) of the scratch
    int32_t mc0[17];   // 64-row chunks: expert e's at [mc0[e], mc0[e + 1])
    int32_t n;
};
/// Whether exl3_pf_group takes this layer (the shapes and K it is built for).
bool exl3_pf_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
void exl3_pf_group(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const Exl3PfGroup& g, const uint16_t* Xs,
                   float* GU, uint16_t* Hh, float* Dm, float* rs, void* stream);

}  // namespace strata::kernels

