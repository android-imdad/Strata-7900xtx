# EXL3 2.68 experts against IQ3_S, RX 7900 XTX (gfx1100)

25 October 2026, two comparisons, each 3 alternating rounds of `tools/strata_bench.py --long` (medians; each round
itself the median of 3 runs per test).

## Against upstream v0.2.4.1 (`vs-upstream/`)

Upstream Strata v0.2.4.1 (`8183c6b6`, IQ3_S, its defaults with every output-changing switch off, from its own
checkout) against this fork with the EXL3 2.68 pack (engine `7c5de68d`, the settings below). Every round of the fork
beat every round of upstream in every row:

| Test | Upstream v0.2.4.1 (IQ3_S) | This fork, EXL3 2.68 | Change | Rounds (upstream / fork) |
| --- | ---: | ---: | ---: | --- |
| Short chat, writes answers | 87.3 tokens/s | 115.1 tokens/s | **+32%** | 86.3-89.8 / 115.0-117.7 |
| Short chat, first token | 1.16 s | 0.40 s | **-66%** | 1.16-1.16 / 0.40-0.41 |
| 4K prompt, reads | 1,066 tokens/s | 1,882 tokens/s | **+76%** | 1,065-1,072 / 1,867-1,885 |
| 16K prompt, reads | 1,378 tokens/s | 2,247 tokens/s | **+63%** | 1,378-1,385 / 2,243-2,251 |
| 32K prompt, reads | 1,398 tokens/s | 2,278 tokens/s | **+63%** | 1,389-1,399 / 2,273-2,279 |
| Agent session, first token per turn | 3.83 s | 1.69 s | **-56%** | 3.82-3.95 / 1.67-1.71 |
| Agent session, writes answers | 97.2 tokens/s | 110.9 tokens/s | **+14%** | 91.4-101.5 / 107.2-116.1 |

## Against this fork's own IQ3_S (`vs-fork-iq3_s/`)

The same engine (`7c5de68d`, the parity build) and the same settings for both: the README's exact
settings (parts 1-7 of [docs/RX7900XTX.md](../../../docs/RX7900XTX.md): `STRATA_CPU_PARITY=1`, `STRATA_PF_PIECE=8192`
with `--prefill auto:32768`, `STRATA_SPEC_CUM=0.5`, `STRATA_PF_DOWN_STREAMS=4`, `STRATA_PCIE_USAGE=1`,
`STRATA_PCIE_KEEP=96`, `--kv-resident 32768`, no control vector). The only difference is `--pack`: `iq3_s` (the
GSQ-RCO IQ3_S experts in the GGUF) or `exl3_268` (turboderp's EXL3 2.68 experts, built by `tools/exl3_pack.py`; the
reports' model name is the config's, `qwen3.8-flash-next-iq3_s`, in both). `tools/strata_bench.py --long`, 3
alternating rounds (IQ3_S first in each), each round itself the median of 3 runs per test:

| Test | IQ3_S | EXL3 2.68 | Change | Rounds (IQ3_S / EXL3) |
| --- | ---: | ---: | ---: | --- |
| Short chat, writes answers | 103.7 tokens/s | 115.8 tokens/s | **+11.7%** | 102.9-103.9 / 114.1-117.0 |
| Short chat, first token | 0.85 s | 0.40 s | **-53%** | 0.84-0.86 / 0.39-0.40 |
| 4K prompt, reads | 1,450 tokens/s | 1,880 tokens/s | **+29.7%** | 1,441-1,456 / 1,876-1,884 |
| 16K prompt, reads | 1,937 tokens/s | 2,251 tokens/s | **+16.2%** | 1,935-1,941 / 2,242-2,252 |
| 32K prompt, reads | 2,011 tokens/s | 2,279 tokens/s | **+13.3%** | 2,009-2,011 / 2,276-2,280 |
| Agent session, first token per turn | 2.80 s | 1.66 s | **-41%** | 2.78-2.81 / 1.65-1.71 |
| Agent session, writes answers | 104.1 tokens/s | 113.5 tokens/s | **+9.0%** | 102.8-105.9 / 112.8-117.4 |

Every EXL3 round beat every IQ3_S round in every row. The output is not the same (another quantization of the
experts): quality is compared by perplexity in the [README](../../../README.md#exl3-experts-smaller-faster-similar-quality).
One folder per round and pack: the benchmark's own report, its raw results and the engine log excerpt.
