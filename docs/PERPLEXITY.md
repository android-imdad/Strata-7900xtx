# What perplexity means

Perplexity is the number this fork uses to say how much quality a change costs, or how two versions of the model
compare. This page explains what it measures, how to read the numbers in the [README](../README.md) and in
[docs/RX7900XTX.md](RX7900XTX.md), and where it stops being useful.

## The idea

A language model reads text one token at a time (a token is a word or a piece of a word). Before each token it gives
every possible next token a probability. Perplexity asks: **how much probability did the model give to the token that
really came next?**

- If the model gave the real next token a probability of 1 (100%) every time, it was never surprised: perplexity 1.0.
- If it gave it 50% every time, it was as unsure as a fair coin: perplexity 2.0.
- If it gave it 10% every time, it was as unsure as choosing among 10 equally likely tokens: perplexity 10.

So perplexity is roughly **the number of equally likely tokens the model is choosing between at each step**. Lower is
better. A perplexity of 1.42 means the model is usually close to sure of the next token and sometimes hesitates.

## The formula

For a text of N tokens, where the model gave the real next token the probability p at each position:

```
perplexity = exp( average of -log(p) over the N positions )
```

The average of `-log(p)` is the model's mean *negative log-likelihood* (NLL), measured in *nats* per token. Taking the
exponential turns it back into the "number of choices" scale. Perplexity is the geometric mean of `1/p`, so a few
positions where the model was badly wrong (very small p) weigh a lot.

A small example: probabilities 0.9, 0.5 and 0.99 for three tokens give `-log(p)` = 0.105, 0.693 and 0.010, an average
of 0.269, and a perplexity of exp(0.269) = 1.31.

## Why it measures quantization loss

To fit the 125-billion-parameter model on one PC, its weights are compressed (quantized) to about 2.6-3.3 bits each.
That adds a little noise, so the model becomes slightly less sure of the right next token, and its perplexity rises a
little above the uncompressed (BF16) model's. **The smaller the gap to the uncompressed model, the less the
compression cost.** On turboderp's chat trace, for example:

| Model | Perplexity | Above the uncompressed model |
| --- | ---: | ---: |
| Uncompressed (BF16, turboderp's chart) | 1.3847 | - |
| exllamav3 3.18 bits | 1.4026 | +1.3% |
| Strata, IQ3_S experts (upstream v0.2.4.1) | 1.4187 | +2.5% |
| Strata, EXL3 2.68 experts (this fork) | 1.4218 | +2.7% |

## Comparing two versions

**Percentages.** A change from perplexity A to B is reported as `B / A - 1`, for example +0.19%. For small changes
this equals the change in mean NLL: +0.19% is 0.0019 nats per token.

**Paired comparisons and confidence intervals.** Both versions read exactly the same text, so the comparison is made
token by token: the difference in `-log(p)` at each position. That cancels most of the text's own difficulty and
leaves the real difference between the versions. From the spread of those per-token differences comes a 95%
confidence interval (CI): the range the true difference lies in, given how much text was read. **If the interval
includes zero, the two versions are not measurably different** on that text. Examples from the README, on the chat
trace: this fork's IQ3_S against upstream's is +0.03% with a CI of -0.02% .. +0.08% - a tie. EXL3 against upstream's
IQ3_S is +0.22% with a CI of +0.02% .. +0.42% - a real but very small difference. On code and docs EXL3 is -5.2% with
a CI of -2.6% .. -7.8% - a clear improvement.

**Bit-identical output.** Most of this fork's speedups give exactly the same probabilities at every position as the
engine they replace. Then perplexity is the same to the last digit, and the docs say "bit-identical" or "KL 0.0000"
instead of quoting a perplexity change.

## The two test sets

The numbers come from two texts, read with the engine's own prompt and decode paths:

- **Chat** - turboderp's in-domain trace, 45,588 positions of chat conversations: the set his exllamav3 charts are
  measured on, so Strata's numbers can be compared with his. It is the larger set, so its confidence intervals are
  narrow (a few tenths of a percent).
- **Code and docs after long prompts** - kl.py's three texts: source code and documentation read after 16K, 16K and
  64K tokens of prompt, 2,557 scored positions (text tokens only). It tests long-context reading. It is smaller, so
  its intervals are wider (1-2%).

The two sets have very different perplexities (about 1.4 and 4.4) because raw code and documentation are much harder
to predict than chat. **Compare versions within one set, never one set's number against the other's.**

## Other numbers used next to perplexity

- **Same top token** - the share of positions where both versions would pick the same most likely next token (100%
  for bit-identical changes).
- **KL divergence** - how far apart the two versions' whole probability distributions are, averaged over positions
  (0 for bit-identical changes). It catches changes that perplexity, which only looks at the real next token, misses.

## Limits

- Perplexity measures how well the model predicts these texts. It follows answer quality closely for small changes
  like quantization, but it is not a test of reasoning or of any one task.
- Differences inside the confidence interval are noise. A change that looks better by 0.1% on the code set is not a
  measured improvement.
- A different prompt chunk size, a different engine or different weights for the rest of the model change the numbers
  slightly; the comparisons here keep everything but the one change the same, and say so when they do not.
