# IQ3_S against EXL3 2.68 experts, RX 7900 XTX (gfx1100)

25 October 2026. The same engine (`7c5de68d`, the parity build) and the same settings for both: the README's exact
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
experts): quality is compared by perplexity in the [README](../../../README.md#exl3-experts-smaller-faster-same-quality).
One folder per round and pack: the benchmark's own report, its raw results and the engine log excerpt.
