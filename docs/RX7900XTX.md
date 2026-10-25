# RX 7900 XTX (gfx1100) edition

This branch (`rx7900xtx`) is upstream Strata plus speedups measured on one AMD Radeon RX 7900 XTX. Every change keeps
the model's output **bit-identical**: reading the same texts, the engine with them gives the same log-probabilities
and the same top token at every position as the engine without them, so the same perplexity. Nothing here trades quality for speed. The exceptions are opt-in:
part 4 (`-DSTRATA_SWIGLU_SEXP` + `STRATA_CPU_PARITY=1`) shifts the output once against today's engine so that from then
on it no longer depends on where the experts sit, and part 8's EXL3 pack is another quantization of the experts
(22% smaller, the same perplexity on chat, 5% lower on long code and docs, faster).

Measured on: RX 7900 XTX 24 GB (gfx1100), Ryzen 7 9800X3D, 62 GiB RAM, Linux 7.2, ROCm 10.2 (TheRock nightly
10.2.0a20260930), Qwen3.8-Flash-Next GSQ-RCO IQ3_S, 262,144-token context, no control vector.

## Results

`tools/strata_bench.py --long`, same day (24 October), 3 alternating rounds, medians (each round itself the median of
3 runs per test). Upstream v0.2.4.1 (`8183c6b6`, engine 0.1.49.1; its defaults, every output-changing switch off)
against this fork merged with v0.2.4.1 (parts 1-6 with the parity build, `STRATA_MTP_HIP_PRIO=1`, engine `8928808b`):

| Test | Upstream v0.2.4.1 `8183c6b6` | This fork | Change |
| --- | ---: | ---: | ---: |
| Short chat, writes answers | 82.6 tokens/s | 97.5 tokens/s | **+18%** |
| Short chat, first token | 1.20 s | 0.84 s | **-30%** |
| 4K prompt, reads | 1,021 tokens/s | 1,365 tokens/s | **+34%** |
| 16K prompt, reads | 1,341 tokens/s | 1,868 tokens/s | **+39%** |
| 32K prompt, reads | 1,359 tokens/s | 1,928 tokens/s | **+42%** |
| Agent session, first token per turn | 3.98 s | 2.99 s | **-25%** |
| Agent session, writes answers | 90.3 tokens/s | 95.3 tokens/s | +5.5% |

Every round of the fork beat every round of upstream in every row but the last, where the rounds overlap (fork
91.5-102.9, upstream 89.4-93.3 tokens/s). Both engines measured 5-7% below the 14 October runs on this PC (the machine,
not the code: upstream lost the same share), so across dates compare the percentages. Report:
[bench/results/2026-10-24-...-0.2.4](../bench/results/2026-10-24-community-rx-7900-xtx-gfx1100-0.2.4/).
Merging v0.2.4.1 (1,836 upstream commits) changed none of this fork's output: kl.py over 2,557 positions after 16K,
16K and 64K prompts gives the same top token everywhere, mean KL 0.0000, perplexity 4.447 = 4.447 against the
0.1.44.1-based engine `d4b184ee`. A fresh 32K prompt with exact 32K chunks gives the 8K-chunk run's residual stream
bit for bit at every half layer (`STRATA_PF_LAYER_DUMP`); that check found and fixed one merge bug (`8928808b`:
upstream's new chunk-size variable shadowed a row width in `STRATA_PF_PIECE`'s checkpoint split, and 32K prompts with a
checkpoint inside went non-finite).

The previous comparison, against upstream 0.1.44.1 (14 October, engine `d4b184ee`):

| Test | Upstream 0.1.44.1 `296b59d2` | This fork | Change |
| --- | ---: | ---: | ---: |
| Short chat, writes answers | 88.8 tokens/s | 105.0 tokens/s | **+18%** |
| Short chat, first token | 1.12 s | 0.82 s | **-27%** |
| 4K prompt, reads | 1,081 tokens/s | 1,443 tokens/s | **+33%** |
| 16K prompt, reads | 1,397 tokens/s | 1,921 tokens/s | **+37%** |
| 32K prompt, reads | 1,406 tokens/s | 1,990 tokens/s | **+42%** |
| Agent session, first token per turn | 3.88 s | 2.93 s | **-24%** |
| Agent session, writes answers | 96.4 tokens/s | 102.5 tokens/s | **+6%** |

Reports:
[bench/results/2026-10-14-...-0.1.44-cum](../bench/results/2026-10-14-community-rx-7900-xtx-gfx1100-0.1.44-cum/); the
same comparison before part 6 (short chat +10%, agent sessions +2%):
[2026-10-14-...-0.1.44](../bench/results/2026-10-14-community-rx-7900-xtx-gfx1100-0.1.44/).
Merging 0.1.44.1 changed none of this fork's output (kl.py: 2,557 positions identical, perplexity 4.447 = 4.447, with
8K and with exact 32K chunks), and decode did not get slower: with the same drafts on both engines (5 paired rounds,
3,580 windows, every reply identical) the merged engine takes 30.49 ms a window against 30.75 before it. Regular
benchmark runs differ by a few percent in decode because the drafts follow measured timings. Upstream's agent decode improved in 0.1.44 and is now close to this fork's; its new
default STRATA_COMMIT_BATCH measured neutral here (=0: the same within noise).

The earlier comparison, against upstream main `a0116b27` (13 October, strict config, before parts 4 and 5):

| Test | Upstream main `a0116b27` | This branch | Change |
| --- | ---: | ---: | ---: |
| Short chat, writes answers | 86.8 tokens/s | 95.8 tokens/s | **+10%** |
| Short chat, first token | 1.12 s | 0.83 s | **-26%** |
| 4K prompt, reads | 1,069 tokens/s | 1,427 tokens/s | **+34%** |
| 16K prompt, reads | 1,381 tokens/s | 1,827 tokens/s | **+32%** |
| 32K prompt, reads | 1,421 tokens/s | 1,844 tokens/s | **+30%** |
| Agent session, first token per turn | 3.90 s | 2.92 s | **-25%** |
| Agent session, writes answers | 96.6 tokens/s | 107.0 tokens/s | **+11%** |

Reports: [upstream and PR #2231](../bench/results/2026-10-13-community-rx-7900-xtx-gfx1100-exact-speedups/),
[PR #2231 and this branch](../bench/results/2026-10-13-community-rx-7900-xtx-gfx1100-gu-exact/). In the prompt and
first-token rows every round of this branch beat every round of the engine it was compared with; decode moves a few
percent between runs (the drafts follow measured timings), so read its rows as approximate.

The config is a strict one: every switch that changes the output (even slightly) is off, including upstream's own opt-in
ones (`STRATA_HC_Q8`, `STRATA_PF_GEMM`, `STRATA_PF_FUSED`, WMMA prompt attention). With those on, both engines are
faster and no longer identical to each other.

## What is in it

### 1. Decode and prompt-path kernels (from PR #2231, withdrawn)

[Niko1221/Strata#2231](https://github.com/Niko1221/Strata/pull/2231): launch shapes of latency-bound kernels (router,
GDN norm, hyper-connection up, QSA top-k, the drafter's row select), DPP lane exchanges, the multi-block argmax on HIP,
the PCIe experts' landed flag written by the copy stream, the adaptive tier's copies over both DMA engines, warp
dequant of Q6_K / IQ4_XS. On by default, each with an `=0` switch; plus opt-ins listed below.

### 2. The experts' gate/up product, exact and fast (`STRATA_PF_GU_EXACT=1`)

In a prompt about half the GPU time is the routed experts' gate/up product (FP16 x FP16 -> FP32, 1280 x 2560 per
expert). It has no hipBLASLt tuning, so it falls back to rocBLAS's generic Tensile kernels, which do not use the matrix
cores and reach 6-13 TFLOPS on this card. Any faster kernel adds the products in another order, which changes the
output a little (that is why `STRATA_PF_GEMM` moves perplexity from 4.447 to 4.508).

This branch reproduces rocBLAS's order instead. Read from its kernels and checked bit for bit:

- **65 tokens or more per expert** (`MT64x32x8`, GSU 1): each output is one fused-multiply-add chain over the 2,560
  inputs, starting at input `((n / 64) % 16) * 128` (Tensile's StaggerU) and wrapping around.
- **2 to 64 tokens** (`MT128x32x16`, GSU 3): the inputs are split three ways in interleaved steps of 16 (split `g`
  takes steps `g, g+3, g+6, ...`), each split is one chain starting at its step `((n / 128) % 4) * 8`, and the result
  is `(p0 + p1) + p2`.
- **1 token** is a GEMV and stays on rocBLAS.

With the order fixed, the work is free to run faster:

- The prompt path collects 16 experts and computes them together (`src/prefill/gu_exact.cu`).
- From 65 tokens the products run on a **Tensile-generated kernel** (`src/prefill/gu_tensile_gfx1100.s`). Tensile
  tuned everything except the parameters that define the order (MacroTile0 64, GlobalSplitU 1, InnerUnroll 1,
  StaggerU 32 / 256 bytes / mapping 0, no matrix instructions). The chosen kernel, MT64x128x32, reaches 26.5 TFLOPS at
  512 tokens and 28.7 at 1,024 (rocBLAS's fallback: 12.9 / 13.1), about what rocBLAS's own tuned FP32 GEMM does on
  this card. One launch per expert, spread over 8 streams: about 24 TFLOPS on a typical group.

How it was checked:

- Offline against `hipblasGemmEx`: every token count 2..4,096 and every 37th to 16,384 for gu_exact's own kernels;
  for the Tensile kernel 748 token counts through its own client (65 to ~3,300) and 656 through this launcher (65 to
  4,100), with a negative control that must differ (it does).
- At every start: the grouped kernels against the engine's own `Gemm::f16` call at 25 token counts, bit for bit. A
  difference, a different GEMM path or another GPU turns the feature off and says so in the log.
- In the engine: kl.py (2,557 positions after 16K / 16K / 64K prompt reads) identical log-probabilities and perplexity, 100% same top token; the
  session-state fingerprint after short prompts identical.

### 3. Deterministic decode: the same tokens whatever the drafter does (`STRATA_IQ_MT_MIN=1`)

Strata verifies the drafter's guesses in windows of 1-8 tokens, and the window a token lands in follows the drafts, which
follow measured timings. Without this switch a token's logits depend on its window: reading the same text through windows
of 2, 3 or 4 tokens instead of 1 gives slightly different log-probabilities, so the same prompt can give a slightly different reply from run to run. The cause on this card is the CPU expert rows, which were grouped by window.

Upstream's `STRATA_IQ_MT_MIN=1` computes every CPU row independently of the window. With it, teacher-forced reading
through windows of **1, 2, 3, 4, 5, 6 and 8 tokens gives byte-identical log-probabilities** (2,557 positions, the same perplexity,
the same 36 GDN layer states), so every token is exactly what one-token-at-a-time decoding would give: the reply no
longer depends on the drafts or on timing. It can still change when the expert placement changes (how many experts fit
in VRAM, which other programs' VRAM use can shift by a few slots), because an expert computed on the CPU differs from the
GPU's result in the last bits; for the same placement it is the same (part 4 removes that difference too). Checked with the debug knob `STRATA_READ_WINDOW=N` (read through windows of at most N tokens) and
`kl.py`; end to end, a 2,425-token file rewrite returned byte-identical replies under four draft policies.

That makes the draft policy a pure speed setting. On rewrite turns (the model returns a file with small edits):

| Draft policy (all with `STRATA_IQ_MT_MIN=1`) | Decode, 2,425-token rewrite |
| --- | ---: |
| `--spec-min-p 0.70`, no lookup | 134 tokens/s (median of 3) |
| `--spec-min-p 0.5 --lookup-chain 3` | 145 tokens/s (median of 3), same bytes |

The MTP drafter's own depth is a speed setting too now. With `--spec 4` it stops at 3 guesses although its acceptance is
still ~85-90% there; with `--spec 6` acceptance stays ~80% down to the fifth guess. Two alternating rounds of thinking-on
questions, a coding task and a rewrite (slots pinned): `--spec 6` was fastest in every request, about +5-20% over
`--spec 4` (thinking turns most), for ~6 MiB of VRAM; `--spec 8` drafts too deep, and `--spec-min-p` 0.4 / 0.3 did not
beat 0.5. Ordinary chat and agent turns gain nothing from the lookup chain; `STRATA_IQ_MT_MIN` itself costs 0-5% there.
The reference this matches is one-token decoding, not the engine without the switch (whose replies follow the drafts).

### 4. The same output whatever the expert placement (`STRATA_CPU_PARITY=1`, build with `-DSTRATA_SWIGLU_SEXP`)

Each routed expert runs either on the GPU (from VRAM or over PCIe) or on the CPU, and today the two give results that
differ in the last bits. So the reply depends on the placement: how many expert slots fit in VRAM (other programs' VRAM
use moves it by a few slots), the adaptive tier's swaps, and VRAM-saving options such as `--kv-resident`. On the same
engine, moving 50 / 200 slots moves perplexity from 4.425 to 4.461 / 4.419.

This part makes the CPU compute exactly what the GPU computes:

- **A portable exponential in the GPU SwiGLU** (`include/strata/kernels/sexp.hpp`, selected at build time with
  `-DSTRATA_SWIGLU_SEXP`): a Cody-Waite `exp` from `fma` / `rint` / `ldexp` only, at most 0.94 ulp (`__expf` on
  gfx1100: up to 64 ulp). `g / (1 + sexp(-g))` gives the same bits on the host and on the GPU for all 2^32 inputs.
- **A CPU replica of the GPU expert arithmetic** (`src/kernels/cpu/replica_avx512.cpp`, AVX-512 VNNI): the same q8_1
  activations, each part's integer dot and scale step, `(dw * d8) * sumi` accumulated by FMA (the GPU contracts it) in
  the GPU's lane and butterfly order, for every format the model uses (gate/up IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS; down
  IQ4_NL, Q2_0). The tool `expert_replica` checks it against `native_expert_grouped`: 192/192 bit for bit (48 layers x
  1-4 tokens, VRAM and PCIe launches); plain multiply + add, the negative control, differs everywhere.

The switch turns itself off (and says why in the log) without AVX-512, if a layer's format is not in the replica, or on a
GPU build without `-DSTRATA_SWIGLU_SEXP`. Without the switch nothing changes.

The replica reads IQ3_S grid entries with one AVX-512 gather per 64 weights instead of 16 scalar loads: the same table
values, so the same integer sums (`expert_replica` on all ten IQ3_S layers: 0 of 10,240 outputs differ from the GPU).
That kernel was bound by the scalar loads; with the gather the CPU expert time per decode window is 3.5% lower and
writing answers +0.3-0.4% (only 10 of the 48 layers are IQ3_S). `STRATA_REPLICA_GATHER=0` keeps the scalar loads.

Checked with kl.py (2,557 positions, slots pinned): with parity on, 7,200 slots against 7,150 / 7,000 / 6,500 slots
gives identical log-probabilities, 100% same top token, perplexity 4.447 at every placement (without parity the
placements disagree). With `--kv-resident 32768` against without it, also identical: perplexity 4.447 = 4.447. Together with part 3 the reply
no longer depends on the drafts, the timing, the expert placement or VRAM use.

**The cost:** the new exponential rounds differently from `__expf`, a one-time shift against today's output: perplexity
4.425 -> 4.469, 96.6% same top token, the same size as the placement drift above (perplexity 4.419-4.461).
Tool calls: 36/36 valid with and without it.

**The gain:** placement becomes a free speed setting, like the drafts in part 3. `--kv-resident 32768` keeps only the
last 32K tokens of KV in VRAM and frees it for ~1,600 more expert slots (8,798 against ~7,200 here). Official benchmark,
4 alternating rounds, medians:

| Engine | Chat, writes answers | Agent session, writes answers | 16K prompt, reads |
| --- | ---: | ---: | ---: |
| Parts 1-3 | 95.4 tokens/s | 98.1 tokens/s | 1,852 tokens/s |
| + parity | 94.6 tokens/s (-0.9%) | 98.5 tokens/s (+0.4%) | 1,857 tokens/s |
| + parity + `--kv-resident 32768` | 94.8 tokens/s (-0.6%) | **105.8 tokens/s (+7.8%)** | 1,850 tokens/s |

Parity alone costs nothing measurable. With `--kv-resident` every agent round (105.4-108.0) beat the best baseline round
(103.7); chat decode was flat in this run (+4% in an earlier sweep).

### 5. Larger prompt chunks with the same bits (`--prefill auto:32768` + `STRATA_PF_PIECE=8192`)

Reading a prompt in 32K-token chunks instead of 8K is faster, mostly because every expert that is not in VRAM is
copied and dequantized once per chunk. But it changes the output (perplexity 4.447 -> 4.469), for three reasons
found by hashing every row's hidden state after every half layer (`STRATA_PF_LAYER_DUMP`):

1. **Dense products depend on a row's place in the launch.** hipBLASLt / rocBLAS stagger their summation order by
   row tile, so a token at row 9,000 of a 32K launch gets other bits than at row 808 of an 8K one.
2. **Expert products depend on how many tokens the expert gets** (the exact gate/up kernel's order class, the down
   product's tile placement).
3. **The server reads a prompt in segments ending at its checkpoints**, so 8K chunks are counted from each
   segment's start, not from token 0.

`STRATA_PF_PIECE=P` keeps the large chunk but computes in P-token pieces at exactly the row offsets P-token chunks
have: dense products launch per piece; each expert is dequantized once and multiplies each piece's tokens with the
calls a P-token chunk makes (consecutive pieces of 65+ tokens share one gate/up launch: that kernel's order depends
only on the output column; the down product stays per piece, `tools/hip/pf_piece/down_order_check` shows hipBLASLt's
solution 2106 keeps a row's bits only where `(row / 96) % 4` is unchanged); the PLE block's sub-batches follow the
pieces. Conversation checkpoints that P-token chunks would have made inside a large chunk (every
`--prompt-cache-every` tokens) are taken there too, so a new prompt that shares a prefix with an earlier one resumes
from the same place.

Checked against `--prefill auto` (8K chunks), same build, slots pinned:

- kl.py: 2,557 positions identical, perplexity 4.447 = 4.447.
- Through the server with images and a control vector: 36 tool calls, 2 images, 18.8K / 35.9K / 51.1K-token
  documents and their follow-up turns, 44/44 replies byte-identical; the checkpoints at 16,384 / 32,768 / 49,152
  hash the same (`STRATA_CKPT_HASH=1`) and the prefix reuse is the same.

Speed (prompt reading, with parts 1-4):

| Prompt | 8K chunks | 32K + `STRATA_PF_PIECE=8192` | 32K plain (perplexity 4.469) |
| --- | ---: | ---: | ---: |
| fresh 32K prompt | 21.8 s | 18.8-19.3 s (**+13-16%**) | 16.9 s |
| 18.8K-token document (server) | 13.3 s | 12.0 s (**+11%**) | - |
| 48K continuation | 31.4 s | 28.2-29.1 s (**+8-11%**) | 26.1 s |

About half of the plain 32K gain stays: experts whose pieces get 2-64 tokens must keep the slower 3-split order the
8K reference uses, and the down product runs per piece. Writing answers is unchanged.

### 6. Drafts kept while the whole chain is likely to hold (`STRATA_SPEC_CUM=0.5`)

`--spec-min-p 0.5` keeps a draft in the verify window while that draft's own probability (from the draft layer) is at
least 0.5. A window costs more for every token it carries, though, and a draft only counts if every draft before it was
accepted too. `STRATA_SPEC_CUM=S` keeps drafts while the product of the probabilities so far (the chance that the whole
chain holds) is at least `S`, and the drafter stops at the same point. With part 3 the output does not depend on the
drafts, so this changes only the speed.

Measured on this card (verify windows logged with `STRATA_SPEC_LOG=<file>`):

- A window costs 11.2 ms plus 2.4 ms per verified token, a draft step 0.74 ms. The draft layer is well calibrated for
  its first draft and overconfident after it (second drafts at probability 0.8-0.9 held 57% of the time).
- Replayed over 604 full-depth windows: `--spec-min-p 0.5` 115.8 tokens/s (model), the cumulative rule at 0.5 118.0,
  at 0.4 with probabilities calibrated per draft depth 119.6, a perfect stop (knowing which drafts hold) 140.6. The
  draft layer's probabilities are what limits any such rule.
- Six 1,000-token prompts x 300 tokens, no lookup chain, the same tokens in every run, 3 alternating rounds (medians):
  104.09 -> 105.94 tokens/s (+1.8%). The calibrated variant (`STRATA_SPEC_CAL=1`, calibration learned while serving)
  was not better (+0.3% to +1.3%).
- `tools/strata_bench.py`, the settings above with `--lookup-chain 3`, 3 alternating rounds (medians):

| Test | `--spec-min-p 0.5` | + `STRATA_SPEC_CUM=0.5` | Change |
| --- | ---: | ---: | ---: |
| Short chat, writes answers | 96.3 tokens/s | 102.5 tokens/s | **+6.4%** |
| Agent session, writes answers | 97.2 tokens/s | 99.2 tokens/s | **+2.1%** |
| Prompts, first tokens | | | unchanged |

Every round with the rule beat every round without it, in both tests.

A further opt-in, `STRATA_SPEC_LOOKUP_Q=0.95` + `STRATA_SPEC_DEPTH_DECAY=0.9` (with `STRATA_SPEC_CUM=0.3` and
`--spec-min-p 0.01`), counts a draft that the suffix lookup (prompt + output so far) also proposes as 0.95 and discounts
deeper drafts. A logistic fit over 1,167 logged windows ranked the lookup's agreement as the strongest sign that a draft
holds (the runner-up's probability and the entropy added nothing), and without the lookup chain it was +1.4% / +5.8%
on two prompt sets. With `--lookup-chain 3` and suffix drafts, which already use the lookup, the official benchmark
was a wash (short chat -2.5%, agent sessions +0.8%, 3 rounds), so it is not in the settings above.

### 7. Faster prompt experts and adaptive swaps from the PCIe fetches

Three changes with the same output (engine `89ef4423`), found by a read-only review of where the time goes and then
measured:

- **`STRATA_PF_DOWN_STREAMS=4`**: the prompt's expert down products (and their SwiGLU) of a 16-expert group run on 4
  streams forked from the compute stream and joined back before the group's buffers are reused, as part 2's gate/up
  products already do. A down product of ~100 tokens fills only ~27 of the card's workgroup slots. The calls are the
  same hipBLASLt solution with the same arguments, only those that need no workspace, each writing its own rows.
  Fresh prompts: 2K -3..-4%, 4K -2..-3%, 16K and 32K -2% read time (every round faster). 8 streams measured the same
  as 4. Do not raise `GPU_MAX_HW_QUEUES` to 8 on this card: every prompt got 10-20% slower.
- **`STRATA_PCIE_USAGE=1` + `STRATA_PCIE_KEEP=96`**: each layer sends about a third of its missed experts over PCIe
  (`--pcie-frac 0.35`) and drops the bytes after use, and between windows the adaptive tier copies ~35-50 experts from
  RAM into VRAM, which the next window waits for. Almost every one of those swaps missed one window earlier. Today's
  pick sends the last misses in routing order over PCIe, which covered 16% of the next round's swaps (replay of a
  served-settings routing dump); picking them by the tier's own usage count covers 63%. `STRATA_PCIE_KEEP=96` lands the
  fetched blobs in a 96-slot ring that keeps them for the window, and the tier copies such a swap inside VRAM. About 31
  of ~50 swaps per window now come from the ring, and the wait for the copies fell from 1.2-1.7 ms to 0.2-0.5 ms a
  window. Six 600-token prompts x 256 tokens, no lookup chain, the same tokens in every run, 5 alternating rounds
  (medians): 106.15 -> 110.48 tokens/s (**+4.1%**, every round faster). With part 4 the output does not depend on
  which experts the GPU or the CPU computes, so this changes only the speed.
- **IQ3_XXS rows on the CPU** (on by default, `STRATA_REPLICA_XXS=0` = before): part 4's CPU replica decodes IQ3_XXS
  (the gate/up type of 17 of the 48 layers in the IQ3_S pack) with one gather and a popcount sign mask instead of 16
  table loads and 8 sign lookups: the same table words, bit-identical to the GPU in `expert_replica` on all 17 layers;
  decode +0.7%.

Checked: teacher-forced KL against the previous engine 0.0000 (100% the same top token, perplexity 4.447 = 4.447),
the same generated tokens. `tools/strata_bench.py --long`, against engine `8928808b`, 3 alternating rounds (medians):

| Test | `8928808b` | `89ef4423` + the switches | Change |
| --- | ---: | ---: | ---: |
| Short chat, writes answers | 100.1 tokens/s | 102.4 tokens/s | +2.3% |
| Short chat, first token | 0.84 s | 0.83 s | -1% |
| 4K prompt, reads | 1,419 tokens/s | 1,464 tokens/s | **+3.2%** |
| 16K prompt, reads | 1,920 tokens/s | 1,955 tokens/s | +1.8% |
| 32K prompt, reads | 1,993 tokens/s | 2,027 tokens/s | +1.7% |
| Agent session, first token per turn | 2.87 s | 2.78 s | **-3.1%** |
| Agent session, writes answers | 107.2 tokens/s | 109.1 tokens/s | +1.8% |

The prompt rows: every round faster but one 16K round (equal). The writing rows overlap (the lookup chain's drafts
follow timing, ~1-2% noise), so the fixed-draft measurement above is the better number for decode.

Also tried: upstream's `STRATA_DEQUANT_WIDE=1` (8 values a thread for the dense Q4_K/Q5_K weights) is bit-exact here
but 8-14% slower than the kernel it replaces, which already writes ~1.6 TB/s.

### 8. EXL3 experts: the routed experts in exllamav3's format (opt-in pack)

A different quantization of the same model's routed experts, so a different output: turboderp's
[EXL3](https://github.com/turboderp-org/exllamav3) quant of Qwen3.8-Flash-Next at 2.68 bits a weight
([`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), revision
`2.68bpw_h6_ng8`). Its experts are trellis-coded (the `mul1` codebook: each weight is a 16-bit window of a bit stream,
multiplied out to a byte sum), 2.5 bits a weight in 40 layers and 3 bits in layers 0-3 and 44-47, with per-channel scales
and a 128-point Hadamard rotation on each side. `tools/exl3_pack.py` copies those tensors byte for byte into an
`experts.bin` and links everything else from the IQ3_S pack: attention, GDN, the shared expert, the router, the
n-gram table and the drafter still come from the IQ3_S GGUF. Only the routed experts differ, so the comparison below
is the experts alone.

| Routed experts (48 layers x 512) | IQ3_S pack (GSQ-RCO) | EXL3 2.68 pack |
| --- | ---: | ---: |
| Size | 50.3 GB | **39.5 GB** (-22%) |
| Bits a weight (scales included) | 3.33 | **2.61** |
| VRAM expert cache, same settings | ~8,800 slots | ~11,100 slots |

The rest of the model is shared: the IQ3_S GGUF (54.8 GB + 28.8 GB, which also hold the IQ3_S experts) stays on
disk, so the EXL3 pack adds 39.5 GB of disk, plus the 46.6 GB download while packing (without
`ngram_embedding.safetensors`, which Strata does not use). In RAM and VRAM the experts take 22% less.

**The same bits on the CPU and the GPU.** `include/strata/kernels/exl3_expert.hpp` is the arithmetic both compute:
`x' = H(x * suh)` in FP32 with fixed butterflies, int8 per 128 inputs (round half away from zero), exact int32 sums
`I = sum q * s` of the codebook byte sums, `acc = sum_j d_j I_j` left to right, `y' = A acc + B qs`, then `H`, `svh / 128`
and the `sexp` SwiGLU of part 4. So, like part 4, the output does not depend on which device computes an expert, and a
verify window gives every token the bits it gets alone. Checked: `tools/exl3_ref.py` (numpy) reproduces exllamav3's own
weight reconstruction exactly; `exl3_cpu_parity` and `exl3_gpu_parity` match it bit for bit (144 of 144 checks:
groups of 1-8 tokens, a PCIe pass's entry offsets, int8 and the Q15 option `STRATA_EXL3_ACT=15`); and kl.py with
11,150 or 2,000 experts in VRAM (42% or 5% of decode's expert lookups on the GPU, the rest on the CPU) gives
byte-identical log-probabilities at the same prompt chunk. (Compare at a fixed `--prefill` chunk: the automatic chunk
shrinks with the cache - 32K at 11,150 slots, 2,816 tokens at 2,000 - and a different chunk is a different
computation, for IQ3_S too. Part 5's `STRATA_PF_PIECE=8192` is exact only when the chunk is at least 8,192 tokens.)

**Kernels.**

- CPU: AVX-512 (VNNI, VBMI). A tile row's 16 states come from one `vpermi2b` of the byte-swapped tile, and `q * s` is
  one `vpdpbusd` of the product's four bytes against q repeated: 0.14 ms an expert for one token.
- GPU, decode (`STRATA_EXL3_V4=1`, default): one launch a layer. Input-transform blocks, then gate/up blocks (2 column
  groups x 10 k-waves), then down blocks (4 x 5), handing off through per-group counters (a block only ever waits for
  blocks dispatched before it). A lane decodes 32 consecutive trellis steps, which are two whole columns of a 16 x 16
  tile, so its integer sums need no shuffles; each block's `d_j I_j` meet in LDS and are summed in order. 85 us for
  a 32-expert, 40-entry layer window (the five-launch version: 117 us).
- GPU, prompts (`STRATA_PF_EXL3_FUSED=1`, default): up to 64 experts a launch on FP16 WMMA. The activations are rotated
  per expert (`x' = H(x * suh)`, FP16), each weight enters the tile as `1024 + s` (exact in FP16, the affine part
  corrected once per row), and the output rotation, SwiGLU and down's input rotation run between the two products.
  No dequantized weights and no per-expert GEMM calls: a 16K prompt's expert time fell from 6.2 s (dequantize + 2x
  hipBLASLt per expert, which IQ3_S still uses) to 2.0 s. This is the prompt path's arithmetic, not decode's (FP16
  activations): against the dequantize + GEMM path it differs by rounding (relative RMS 5e-4, `exl3_pf_check`);
  `STRATA_PF_EXL3_FUSED=0` brings that path back. With `STRATA_PF_PIECE=8192` and `--prefill auto:32768` it gives the
  8K-chunk run's log-probabilities byte for byte.

**Quality**, perplexity (see the README for what it means), paired against IQ3_S on the same positions:

| | Chat trace (45,588 positions) | Code and docs after long prompts (2,557 positions) |
| --- | ---: | ---: |
| BF16, the uncompressed model (turboderp's chart) | 1.3847 | - |
| Strata, IQ3_S experts | 1.4191 | 4.4465 |
| Strata, EXL3 2.68 experts | 1.4218 (+0.19%, CI -0.01..+0.39%) | **4.2122 (-5.3%, CI -2.8..-7.7%)** |
| exllamav3 2.68 (its own engine and weights) | 1.4166 (-0.18%) | 4.2176 |
| exllamav3 3.18 (its own engine and weights) | 1.4026 (-1.17%) | 4.2213 |

The chat trace is turboderp's in-domain set; the second set is kl.py's texts (code and documentation read after 16K,
16K and 64K prompts, text tokens only). On chat the two packs tie; on long code and docs EXL3 is clearly better. The
0.37% that exllamav3's 2.68 keeps over Strata's on the chat trace is not the activations (Q15 activations changed
nothing) - most likely its non-expert weights (6-7 bits) against the GGUF's.

**Speed**, same engine and settings, `tools/strata_bench.py --long`, 3 alternating rounds, medians:

| Test | IQ3_S experts | EXL3 2.68 experts | Change |
| --- | ---: | ---: | ---: |
| Short chat, writes answers | 103.7 tokens/s | 115.8 tokens/s | **+11.7%** |
| Short chat, first token | 0.85 s | 0.40 s | **-53%** |
| 4K prompt, reads | 1,450 tokens/s | 1,880 tokens/s | **+29.7%** |
| 16K prompt, reads | 1,937 tokens/s | 2,251 tokens/s | **+16.2%** |
| 32K prompt, reads | 2,011 tokens/s | 2,279 tokens/s | **+13.3%** |
| Agent session, first token per turn | 2.80 s | 1.66 s | **-41%** |
| Agent session, writes answers | 104.1 tokens/s | 113.5 tokens/s | **+9.0%** |

Engine `7c5de68d` for both (the settings of parts 1-7), only `--pack` differs; every EXL3 round beat every IQ3_S
round in every row. Report: [bench/results/2026-10-25-...-exl3](../bench/results/2026-10-25-community-rx-7900-xtx-gfx1100-exl3/).

Also checked before the LAN server moved to it: the real server code for 5 soak rounds (chats plain and streamed, a
6-turn conversation, 8K-32K prompts, an image, a tool call, parallel requests; no errors), and a hang that only the
engine showed (counters inside the scratch moved with the window size) fixed and covered by a check.

Notes from the kernel work (gfx1100, ROCm 10.2): write a decode step's instructions as asm groups of 8 states
(multiply x8, multiply-add x8, byte sum x8), because the compiler, scheduling for registers, otherwise chains every
step (159 -> 99 us); any scratch memory, even 152 bytes from a struct indexed at run time, slows wave launch badly
(99 -> 89 us); flat loads count in `lgkmcnt`, so an LDS wait also waits for them (use global address space); a
conditional prefetch becomes register copies that wait for the load; `v_dot4_i32_iu8` runs at half rate here.

### 9. The prompt's dequantization beside the products (`STRATA_PF_DQ_OVERLAP=1`, default on)

A GU_EXACT group's experts (part 2) are dequantized on their own stream into one of two buffer sets while the
compute stream runs the previous group's products from the other. The same kernels on the same inputs: kl.py
byte-identical with it on and off. With the LAN server's settings (8K chunks, no `STRATA_PF_DOWN_STREAMS`) fresh
IQ3_S prompts read 7-11% faster (4K 1,174 -> 1,302, 16K 1,599 -> 1,712 tokens/s). With this page's settings (part 5's
exact 32K chunks and part 7's down-product streams) it measured the same with and without (4K, 16K and 32K prompts,
2 rounds), so it matters only without those. `STRATA_PF_DQ_OVERLAP=0` turns it off.

### Tried on this card and not kept

- **Prefetching the next layer's experts.** Upstream's look-ahead (`RouterLookahead`: the next layer's router applied
  to this layer's MoE input) and Foresight swap space (`STRATA_FS_SLOTS`) run on HIP, with the same output. Here they
  were 3-4% slower: the prediction covers only 31-37% of the missed experts at about 4 copies per layer (precision
  0.35-0.48; this model's hyper-connections change the router input a lot from layer to layer), the predictor thread
  slows the CPU pool, and every 1.9 MB copy during a window adds about 14 us to the GPU's path (its reads of host
  memory slow down beside DMA). `STRATA_ROUTE_DUMP=<file>` writes the routing for offline scoring;
  `STRATA_FS_TAKE_FIRST=1` makes a landed slot take a CPU expert's place instead of a PCIe one's (better, still slower).
- **Cheaper kernel launches.** A dependent kernel in a HIP graph costs 2.8 us here, and raw AQL packets with the cache
  fence the kernel needs cost the same (HIP graphs already pre-build their packets). Without the queue's acquire fence a
  kernel costs 2.2 us and stays correct only if it invalidates its own caches first. HIP has no switch for that fence,
  and intercept queues (an HSA tools library) stall the decode window, so it is not used.
- **A cheaper draft layer.** `--mtp-q4 all`: drafts held less often, decode -7%. `--mtp-draft-vocab
  data/draft_vocab_en.bin`: drafting 16% faster with the same acceptance, decode within noise.
- **The adaptive tier's swap budget.** `--adapt-swaps` 160 / 80 / 40 decode the same within noise (fewer swaps wait
  less for their copies but leave more experts to the CPU); 0 (no adaptive tier) is 24% slower.
- **More expert slots.** `--vram-reserve-mib` 1024 / 768 / 512 gives 8,811 / 8,943 / 9,074 slots and +0.4% / +0.8%
  decode: the hit rate is already about 90%, so not worth less headroom for the desktop.
- **The hyper-connection down product on more blocks.** 4 or 2 rows a block instead of 8 (80 / 160 blocks, the same
  per-row arithmetic): -1.6% / -8%, because every block stages the activations again.

## Use it

Build the HIP engine for gfx1100 (the interleaved-copy path needs `-DSTRATA_HIP_MMVQ_IL`):

```bash
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      "-DCMAKE_HIP_FLAGS=-DSTRATA_HIP_EXP_LAYOUTS -DSTRATA_HIP_MMVQ_IL"
cmake --build build-hip --target strata
```

For part 4 add `-DSTRATA_SWIGLU_SEXP` to `CMAKE_HIP_FLAGS` (a separate build directory keeps today's output available).

Then add to the `env` of your `strata-*.json` (all bit-exact):

```json
"STRATA_PF_GU_EXACT": "1",
"STRATA_WAIT_FUSE": "1", "STRATA_MMVQ_IL_PLAIN": "1", "STRATA_COMBINE_HITS": "1", "STRATA_ROUTE_PUB": "1",
"STRATA_WAIT_COMBINE": "1", "STRATA_PF_LEAN_ISSUE": "1", "STRATA_PF_COPY2": "1", "STRATA_IO_THREADS": "64",
"STRATA_IQ_MT_MIN": "1"
```

and, for deterministic decode with faster drafting, set its `args` to `--spec 6 --spec-min-p 0.5 --lookup-chain 3`.
With the part 4 build, also `"STRATA_CPU_PARITY": "1"` in `env` and `--kv-resident 32768` in `args`.
For part 5, `"STRATA_PF_PIECE": "8192"` in `env` and `--prefill auto:32768` in place of `--prefill auto` in `args`.
For part 6, `"STRATA_SPEC_CUM": "0.5"` in `env` (keep `--spec-min-p 0.5`).
For part 8, build an EXL3 pack (`tools/exl3_pack.py`, see the README) and point `--pack` at it, with the part 4 build
and `STRATA_CPU_PARITY=1`.

The log says whether it is on:

```
strata prefill: STRATA_PF_GU_EXACT: the experts' gate/up products from 2 tokens, 16 experts a launch (checked bit for bit against rocBLAS at 25 token counts); from 65 tokens on the Tensile kernel
strata generate: STRATA_CPU_PARITY: on (the CPU expert rows in the GPU's arithmetic, bit for bit)
```

`STRATA_PF_GU_EXACT=G` sets the group size (1 = 16, at most 64; it borrows about 10 MB of VRAM per expert while a
prompt is read). `STRATA_PF_GU_TENSILE=0` keeps the 65+ token experts on gu_exact's own kernel.

## Limits

Why this is a fork rather than an upstream pull request, and thanks to Strata: see the
[README](../README.md#why-a-fork-and-not-merged-upstream).

- Measured on one PC. The Tensile kernel is gfx1100 only; other cards keep gu_exact's own kernels, which follow
  rocBLAS's fallback order generally but were only checked here.
- The bit-exactness is against this ROCm's rocBLAS fallback. The start-up check is there for a different ROCm: if
  rocBLAS's order changes, the feature turns itself off.
- Not built for CUDA or SYCL. The new code is HIP-only and compiles to stubs elsewhere.
- Decode is unchanged by part 2; its gains are part 1's.
- Part 5 reproduces 8K chunks (`STRATA_PF_PIECE` must equal the chunk the reference reads with: 8192 = `--prefill
  auto` here). Single GPU only: a layer split or peer GPU takes no checkpoints inside a chunk.
- Part 8's fast GPU paths are built for this model's shapes (2560 x 640 experts) and 2.5- / 3-bit EXL3 layers; other
  shapes or bit widths fall back to slower kernels. Its CPU path needs AVX-512 (VNNI, VBMI).
- Part 4 needs AVX-512 (VNNI) on the CPU and was checked only for this model's formats. Its replica follows the GPU
  kernels' order as built here; a change to those kernels needs the `expert_replica` check again.
