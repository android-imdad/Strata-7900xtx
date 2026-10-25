<h1 align="center">Strata for the RX 7900 XTX</h1>

<p align="center"><b>A faster <a href="https://github.com/Niko1221/Strata">Strata</a> for the AMD Radeon RX 7900 XTX, with output
bit-identical to the original</b><br>
RX 7900 XTX (gfx1100) · Linux · ROCm · free and open source</p>

[Strata](https://github.com/Niko1221/Strata) runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**,
a 125-billion-parameter model, on one gaming PC. This fork tunes it for the RX 7900 XTX. Every speedup keeps the output
**bit-identical** to upstream (the same log-probabilities and the same token at every position, so the same perplexity): faster, with nothing traded
for quality. The exceptions are opt-in: a parity build, described under [Install](#install), and the
[EXL3 experts](#exl3-experts-smaller-faster-same-quality), a smaller quantization of the model's experts.

Upstream Strata v0.2.4.1 against this fork (merged with v0.2.4.1), same PC, same day, alternating rounds, medians of 3
(RX 7900 XTX 24 GB, Ryzen 7 9800X3D, 62 GiB RAM, Linux, ROCm 10.2, IQ3_S):

| | Upstream v0.2.4.1 | RX 7900 XTX edition | |
| --- | ---: | ---: | ---: |
| Writes answers (short chat) | 82.6 tokens/s | 97.5 tokens/s | **+18%** |
| First token (short chat) | 1.20 s | 0.84 s | **-30%** |
| Reads a 4K prompt | 1,021 tokens/s | 1,365 tokens/s | **+34%** |
| Reads a 16K prompt | 1,341 tokens/s | 1,868 tokens/s | **+39%** |
| Reads a 32K prompt | 1,359 tokens/s | 1,928 tokens/s | **+42%** |
| Agent turn, first token | 3.98 s | 2.99 s | **-25%** |
| Agent session, writes answers | 90.3 tokens/s | 95.3 tokens/s | +5.5% |

The fork runs the settings below, including the opt-in parity build, exact 32K prompt chunks and the draft rule
`STRATA_SPEC_CUM=0.5` (engine `8928808b`). Upstream runs its defaults with every output-changing switch off. Every
round of the fork beat every round of upstream in every row except agent-session writing speed, where the rounds
overlap (fork 91.5-102.9, upstream 89.4-93.3 tokens/s); writing speed moves a few percent between runs. Both engines
measured 5-7% below the 14 October runs on this PC, so compare the percentages, not the absolute numbers, across dates.
Report: [bench/results/2026-10-24-...-0.2.4](bench/results/2026-10-24-community-rx-7900-xtx-gfx1100-0.2.4/)
(the earlier comparison against 0.1.44.1:
[2026-10-14-...-0.1.44-cum](bench/results/2026-10-14-community-rx-7900-xtx-gfx1100-0.1.44-cum/)).

Newer, with the same output: three more bit-exact switches (engine `89ef4423`, [Settings](#settings)) against the
engine above, same day, 3 alternating rounds, medians: reads 4K prompts **+3.2%** (1,419 -> 1,464 tokens/s), 16K +1.8%,
32K +1.7%, agent turn first token **-3.1%** (2.87 -> 2.78 s), writes answers +2.3% short chat / +1.8% agent sessions
(those rounds overlap; with the drafts held fixed, decode is **+4.8%**). Report:
[bench/results/2026-10-24-...-speed](bench/results/2026-10-24-community-rx-7900-xtx-gfx1100-speed/).

**Newest, opt-in: EXL3 experts.** The same model with its routed experts in turboderp's
[EXL3](https://github.com/turboderp-org/exllamav3) 2.68-bit format instead of IQ3_S: the experts are **22% smaller**
(39.5 GB instead of 50.3 GB), the quality is the same on chat and better on long code and documents (perplexity
below), and it is faster everywhere. Same engine (`7c5de68d`) and settings, 3 alternating rounds, medians; every EXL3
round beat every IQ3_S round in every row:

| | IQ3_S experts | EXL3 2.68 experts | |
| --- | ---: | ---: | ---: |
| Writes answers (short chat) | 103.7 tokens/s | 115.8 tokens/s | **+12%** |
| First token (short chat) | 0.85 s | 0.40 s | **-53%** |
| Reads a 4K prompt | 1,450 tokens/s | 1,880 tokens/s | **+30%** |
| Reads a 16K prompt | 1,937 tokens/s | 2,251 tokens/s | **+16%** |
| Reads a 32K prompt | 2,011 tokens/s | 2,279 tokens/s | **+13%** |
| Agent turn, first token | 2.80 s | 1.66 s | **-41%** |
| Agent session, writes answers | 104.1 tokens/s | 113.5 tokens/s | **+9%** |

Report: [bench/results/2026-10-25-...-exl3](bench/results/2026-10-25-community-rx-7900-xtx-gfx1100-exl3/). Sizes,
perplexity and how to set it up: [EXL3 experts](#exl3-experts-smaller-faster-same-quality).

Decode is also deterministic: the same question gives the same answer whatever the drafter does (and, with an opt-in
build, whatever the expert placement).

<details>
<summary><b>Read more:</b> how it works</summary>

Faster decode kernels (first opened as [PR #2231](https://github.com/Niko1221/Strata/pull/2231)), and the prompt's expert
products computed in exactly rocBLAS's summation order, but grouped and on a Tensile-generated kernel for this card
(`STRATA_PF_GU_EXACT=1`), about 2x faster for that step. With `STRATA_IQ_MT_MIN=1` decode is also **deterministic**:
every token equals one-token-at-a-time decoding whatever the drafter guesses (for a given expert placement), so faster draft policies (prompt lookup
for code rewrites, +8%) change only the speed. An opt-in build (`-DSTRATA_SWIGLU_SEXP` + `STRATA_CPU_PARITY=1`) makes
the CPU's expert rows bit-identical to the GPU's, so the output no longer depends on expert placement either; that
shifts the output once (perplexity 4.425 -> 4.469) and lets `--kv-resident 32768` free VRAM for agent sessions with
identical output (+8% agent decode). `STRATA_SPEC_CUM=0.5` keeps drafts while the whole chain is likely to hold
(+6.4% short chat, +2.1% agent decode, same output). Details, flags and how it was checked:
**[docs/RX7900XTX.md](docs/RX7900XTX.md)**.

</details>

## What you need

| | |
| --- | --- |
| **Graphics card** | AMD Radeon **RX 7900 XTX** (gfx1100, 24 GB). Other cards: use [upstream Strata](https://github.com/Niko1221/Strata). |
| **RAM** | 64 GB for the IQ3_S model measured here (it was tested with 62 GiB usable). The EXL3 experts take about 11 GB less. |
| **Processor** | Any x86-64 with AVX2. The optional placement parity build (below) and the EXL3 experts need **AVX-512** (Zen 4 / Zen 5). |
| **System** | Linux with ROCm. Measured with ROCm 10.2 (TheRock nightly 10.2.0a20260930). |
| **Disk** | About 80 GB free, on an SSD. The EXL3 experts need 40 GB more, and 47 GB for the download while packing. |

## Install

**1. Model and config.** Clone this repository and run upstream's installer, which downloads the model and writes a
`strata-<model>.json` config:

```bash
git clone https://github.com/android-imdad/Strata-7900xtx.git Strata
cd Strata
./setup.sh
```

The installer may set up upstream's engine; the next step replaces it.

**2. Build this fork's engine** for gfx1100:

```bash
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      "-DCMAKE_HIP_FLAGS=-DSTRATA_HIP_EXP_LAYOUTS -DSTRATA_HIP_MMVQ_IL"
cmake --build build-hip --target strata
```

For the placement parity build, add `-DSTRATA_SWIGLU_SEXP` to `CMAKE_HIP_FLAGS` and use another build directory (for
example `build-sexp`). It shifts the output once against the other builds (perplexity 4.425 -> 4.469), so
keep both if you want to compare.

**3. Point the config at it.** In `strata-<model>.json`, set `"exe"` to the engine you built
(`build-hip/strata` or `build-sexp/strata`) and add the settings below. `./setup.sh --setup` and `./update.sh` can
write upstream's engine back into the config, so check `"exe"` after running them.

## Settings

Add to `env` (all bit-exact):

```json
"STRATA_PF_GU_EXACT": "1",
"STRATA_WAIT_FUSE": "1", "STRATA_MMVQ_IL_PLAIN": "1", "STRATA_COMBINE_HITS": "1", "STRATA_ROUTE_PUB": "1",
"STRATA_WAIT_COMBINE": "1", "STRATA_PF_LEAN_ISSUE": "1", "STRATA_PF_COPY2": "1", "STRATA_IO_THREADS": "64",
"STRATA_IQ_MT_MIN": "1"
```

and to `args`, for deterministic decode with faster drafting:

```json
"--spec", "6", "--spec-min-p", "0.5", "--lookup-chain", "3"
```

With the parity build, also add `"STRATA_CPU_PARITY": "1"` to `env` and `"--kv-resident", "32768"` to `args`
(+8% decode in agent sessions).

Faster drafting with the same output: add `"STRATA_SPEC_CUM": "0.5"` to `env`. Drafts are kept while the chance that
the whole chain of drafts holds stays at 0.5 or more, instead of judging each draft on its own: short chat **+6.4%**,
agent sessions **+2.1%** writing answers (official benchmark, 3 alternating rounds, every round faster).

Optional, from upstream 0.1.44: `"STRATA_MTP_HIP_PRIO": "1"` gives the drafter its own GPU queue (agent decode +3% in one
benchmark, short chat within noise - inside the run-to-run spread, so not a proven gain; the output does not change). Do not set `STRATA_COPY_HIP_PRIO=1` on this card: it
hangs the engine.

Faster long prompts with the same output: add `"STRATA_PF_PIECE": "8192"` to `env` and change `"--prefill", "auto"` to
`"--prefill", "auto:32768"` in `args`. Prompts are read in 32K chunks but computed exactly as 8K chunks, so the output
and the conversation checkpoints are bit-identical: a fresh 32K prompt +13-16%, long documents +10-12%.

Newest, same output (engine `89ef4423`; KL 0.0000 and the same tokens): add
`"STRATA_PF_DOWN_STREAMS": "4", "STRATA_PCIE_USAGE": "1", "STRATA_PCIE_KEEP": "96"` to `env`. The first spreads the
prompt's expert down products over 4 GPU streams (prompts +2-3%). The other two send the experts the adaptive VRAM tier
is about to swap in over PCIe in the window before and keep them in a 96-blob ring (about 130 MB of VRAM), so most
swaps become a copy inside VRAM (decode +4.1% with the drafts held fixed). The CPU's IQ3_XXS expert rows are also
faster (+0.7%, on by default). Details: [docs/RX7900XTX.md, part 7](docs/RX7900XTX.md#7-faster-prompt-experts-and-adaptive-swaps-from-the-pcie-fetches).

The engine's log (`strata-<model>.log`) shows that they are on:

```
strata prefill: STRATA_PF_GU_EXACT: the experts' gate/up products from 2 tokens, 16 experts a launch (checked bit for bit against rocBLAS at 25 token counts); from 65 tokens on the Tensile kernel
strata generate: STRATA_CPU_PARITY: on (the CPU expert rows in the GPU's arithmetic, bit for bit)
```

If a check fails at start-up (another ROCm, another GPU), the feature turns itself off and the log says why. Every
switch, how each one was checked and its limits: **[docs/RX7900XTX.md](docs/RX7900XTX.md)**.

## EXL3 experts: smaller, faster, same quality

An opt-in pack that swaps the model's **routed experts** (the 48 layers x 512 experts that hold most of its weights)
from the GSQ-RCO IQ3_S GGUF for turboderp's [EXL3](https://github.com/turboderp-org/exllamav3) quant of the same model
at 2.68 bits a weight ([`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3),
revision `2.68bpw_h6_ng8`). Everything else (attention, the shared expert, the router, the n-gram table, the drafter)
still comes from the IQ3_S GGUF, so the comparisons below measure the experts alone. The engine computes EXL3 experts
with its own kernels on the GPU and the CPU, bit for bit the same on both
([docs/RX7900XTX.md, part 8](docs/RX7900XTX.md#8-exl3-experts-the-routed-experts-in-exllamav3s-format-opt-in-pack)).
The output differs from IQ3_S's, since the weights do.

**Sizes**

| Routed experts | IQ3_S | EXL3 2.68 |
| --- | ---: | ---: |
| Size of all 24,576 experts (each one 22% smaller in RAM and VRAM too) | 50.3 GB | **39.5 GB** (-22%) |
| Bits a weight (scales included) | 3.33 | 2.61 |
| Experts that fit in the VRAM cache (24 GB card, same settings) | 8,796 | 11,150 |

The IQ3_S GGUF stays on disk for the rest of the model (54.8 GB + 28.8 GB, IQ3_S experts included), so the pack
adds 39.5 GB of disk.

**Quality: perplexity (lower is better)**

| | Chat (turboderp's trace, 45,588 tokens) | Code and docs after long prompts (2,557 tokens) |
| --- | ---: | ---: |
| The uncompressed model (BF16, turboderp's chart) | 1.3847 | - |
| Strata, IQ3_S experts | 1.4191 | 4.4465 |
| Strata, **EXL3 2.68 experts** | 1.4218: **+0.19%** (95% CI -0.01% .. +0.39%) | 4.2122: **-5.3%** (95% CI -2.8% .. -7.7%) |
| exllamav3 2.68, its own engine (for reference) | 1.4166 | 4.2176 |
| exllamav3 3.18, its own engine (for reference) | 1.4026 | 4.2213 |

On chat the two packs tie (the confidence interval includes zero); on long code and documentation EXL3 is clearly
better, at 22% less memory. The percentages and their 95% confidence intervals compare the two packs token by token
on the same text. The exllamav3 rows use exllamav3's own weights for the rest of the model too (attention at 6-7
bits, an 8-bit n-gram table), so they are a reference point, not a like-for-like comparison.

> **What perplexity means.** At every token of a real text, the model gives a probability to the token that actually
> comes next. Perplexity is the exponential of the average of -log(that probability): roughly, how many equally
> likely tokens the model is choosing between at each step. 1.0 would be a model that is always certain and always
> right; 1.42 means it is usually close to sure and sometimes hesitates. Compression adds a little uncertainty, so
> perplexity measures how much a quantized model loses against the original: on the chat trace the uncompressed model
> scores 1.3847, IQ3_S is 2.5% above it and EXL3 2.68 2.7%. The two columns are different texts (chat is much easier to
> predict than raw code and documentation), so compare rows within a column, not across columns. On the large chat
> set, differences of a few tenths of a percent are noise; on the smaller code set, 1-2%. The confidence intervals say
> which differences are real.

**Speed:** the table at the top (same engine and settings, every round faster). Writing answers is +9-12%, reading
prompts +13-30%, and first tokens arrive 40-50% sooner, mostly because the prompt path decodes the EXL3 experts inside
its matrix kernels instead of first expanding each expert to 16-bit weights.

**Set it up**

```bash
# 1. The EXL3 quant (46.6 GB; Strata does not use its n-gram table, so skip it)
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 2.68bpw_h6_ng8 --exclude ngram_embedding.safetensors \
    --local-dir models/Qwen3.8-Flash-Next-exl3/2.68bpw_h6_ng8
# 2. A pack with its experts, linking everything else from your IQ3_S pack (the directory your config's --pack
#    points at; needs numpy; --check re-reads every part)
python tools/exl3_pack.py --exl3 models/Qwen3.8-Flash-Next-exl3/2.68bpw_h6_ng8 --base <your IQ3_S pack> \
    --out <new pack> --check
```

3. Use the parity build (`build-sexp`, see [Install](#install)) with `"STRATA_CPU_PARITY": "1"`, and in `args` point
   `--pack` at the new pack. Keep `--native` and `--ple-gguf` on the IQ3_S GGUF. The log then says
   `native pack: <new pack> experts (largest blob 1.86 MB)`. The download can be deleted after packing.

Limits: the fast GPU paths are built for this model's shapes and for 2.5- and 3-bit layers (other EXL3 quants fall
back to slower kernels); the CPU path needs AVX-512; the 3.18-bit quant (1.2% lower perplexity in exllamav3, more
memory) is not packed or measured in Strata yet. Measured on one PC like everything here.

## More speed, if you accept a small change in output

The settings above are all exact. The switches below change the output slightly, so it is no longer bit-identical to
upstream. Each was turned on alone against the exact setup (parity build, `--kv-resident 32768`, the expert cache
pinned at 8,600 slots): perplexity over 2,557 positions of code and docs after 16K, 16K and 64K prompts (lower is
better; the exact setup scores **4.447**), and prompt reading speed from the same runs (one run each, so about +/-3%).
For scale: before the parity build, shifting only 200 expert slots moved perplexity between 4.419 and 4.461.

| Switch | Perplexity | Same top token | 16K prompt read | 48K prompt read |
| --- | ---: | ---: | ---: | ---: |
| exact (the settings above) | 4.447 | 100% | - | - |
| `--prefill auto:32768` (`args`): 32K prompt chunks instead of 8K (with `STRATA_PF_PIECE=8192`: 4.447, exact, about half the gain) | 4.469 | 96.2% | **+13%** | **+21%** |
| `STRATA_HIP_WMMA=1` + `STRATA_SELECT_WMMA=1` + `STRATA_PA_FAST=1` (`env`): prompt attention on the matrix cores | 4.492 | 95.9% | **+11%** | **+16%** |
| `STRATA_HC_UPMIX=1` + `STRATA_PF_SWITCH_MIN_T=4096` (`env`) | 4.501 | 94.7% | +2% | -1% |
| `STRATA_PF_GEMM=1` + `STRATA_PF_SWITCH_MIN_T=4096` (`env`) | 4.508 | 94.2% | **-21%** | **-21%** |

- **`--prefill auto:32768`** is the best trade: the smallest change in perplexity and the largest gain on long prompts.
  Prompt attention on the matrix cores (WMMA) is next. Both speed up reading prompts only; writing answers is unchanged.
- **Do not use `STRATA_PF_GEMM`** on this fork: it takes the experts' products off rocBLAS, which turns
  `STRATA_PF_GU_EXACT` off (the log says so), and the prompt path ends up 21% slower. All the switches together
  (perplexity 4.502) were slower than `--prefill auto:32768` alone for the same reason.
- **No effect here:** `STRATA_HC_Q8` converts nothing on this build (the log reports 0.00 GiB), and `STRATA_PF_FUSED`
  changed neither the output nor the speed. `STRATA_PREFILL_CPU_SHARE=auto` acts only on short prompt chunks (agent
  tool results), which this test does not read, so it is not measured here.

Each switch is a single `env` or `args` entry; remove it to go back to exact output.

Never set `STRATA_FUSE_HEAD_GR=1`. On this card it breaks the model (perplexity 15).

## Run and measure

Start it with `./setup.sh` as usual. The app opens at `http://127.0.0.1:8080`, and apps and coding agents connect to
`http://127.0.0.1:8080/v1` (OpenAI) or `/v1/messages` (Anthropic). Everything about using it is the same as upstream:
[upstream README](https://github.com/Niko1221/Strata#using-it), [docs/INSTALL.md](docs/INSTALL.md),
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

To measure your own card, run `./bench.sh` (5-10 minutes: chat, 4K / 16K prompts and an agent-style session). The
numbers above come from this benchmark.

Problems with this fork's changes: open an [issue here](https://github.com/android-imdad/Strata-7900xtx/issues). For
everything else, see [upstream's issues](https://github.com/Niko1221/Strata/issues).

## Why a fork and not merged upstream

- **The fastest part is specific to this card.** The prompt speedup comes from a Tensile-generated gfx1100 assembly
  kernel, tuned and checked on one RX 7900 XTX with one ROCm build. On other GPUs it stays off, and its fallback
  kernels were checked only here. It is bit-exact because it copies the summation order inside rocBLAS's fallback
  kernels, and nothing guarantees that order: a ROCm update can change it (a check at start-up then turns the feature
  off). Upstream would carry a kernel that its maintainers cannot test or regenerate without this card and a patched
  Tensile setup.
- **Part of it changes the output on purpose.** CPU/GPU expert parity shifts the output once (perplexity 4.425 -> 4.469) for every build
  that uses it and needs AVX-512. Its CPU replica copies the GPU expert kernels' arithmetic exactly, so every future
  change to those kernels would have to be checked against it again. That cost belongs in an opt-in fork, not in
  every upstream change.
- **Upstream merges slowly compared with how much it receives.** On 2026-10-14 Strata had 525 open pull requests
  (the oldest from 2026-10-06) and 30 merged through GitHub in total, none in the past week, while 655 new ones were
  opened. Exact-speed work goes stale quickly while it waits, since every upstream kernel change needs the exactness checks
  again. A fork can ship it now and merge upstream on its own schedule. The decode kernels were opened as
  [PR #2231](https://github.com/Niko1221/Strata/pull/2231), later withdrawn; they live here instead.
- **It is measured on one PC.** Upstream serves NVIDIA and AMD cards on Windows and Linux. Everything here was tuned
  and checked on a single RX 7900 XTX under Linux and ROCm 10.2, and it should be judged as that.

This branch follows upstream and takes in its changes from time to time. Everything here is MIT licensed, like Strata,
so upstream or anyone else is welcome to take any part of it.

## Thanks

To **[Niko](https://github.com/Niko1221)** and **[Strata](https://github.com/Niko1221/Strata)**. Strata is what makes a
125-billion-parameter model usable on one gaming PC at all, and everything in this fork builds on it: the engine, the
expert packs, the multi-token drafting and the official benchmark used for every number here. Deterministic
decode is upstream's own `STRATA_IQ_MT_MIN` switch. Thanks also to Strata's contributors, and to AMD's ROCm, rocBLAS
and Tensile, which generated the prompt kernel.

## Credits and license

Strata is by [Niko](https://github.com/Niko1221) and its contributors. The model is
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) and Unsloth.
Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Open source under the [MIT License](LICENSE); a few parts and
every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).
