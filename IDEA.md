# The idea behind Lightning McQwen3

A condensed version of the write-up,
[Lightning McQwen3: Making Qwen3-0.6B think more efficiently](https://blog.basta.one/blog/thinkingcap/),
which has the figures and the traces. The [README](README.md) has the recipe's exact settings.

## Goal

Make Qwen3-0.6B think less without losing accuracy, thinking still on. Training uses only
GSM8K, Skywork-OR1 (its easy band), ARC-Easy and CommonsenseQA; the four benchmarks (SVAMP,
MBPP+, IFEval, BFCL) are held out. Results are tokens saved / accuracy change in pp against the
base model on the same questions, 8 samples per question, 95% intervals in brackets. Tokens
saved is one minus the geometric mean of the per-benchmark token ratios; accuracy change is the
mean of the four changes; a truncated answer counts as wrong.

Final: **47.3% saved [45.8, 48.7] at −0.88 pp [−1.61, −0.15]**.

| Benchmark | Saved | Δ accuracy |
| --- | ---: | ---: |
| SVAMP | 66% | −3.8 pp |
| MBPP+ | 39% | −1.9 pp |
| IFEval | 32% | +1.4 pp |
| BFCL | 45% | +0.8 pp |

The headline combines the ungated runs on SVAMP, MBPP+ and IFEval, which the gate doesn't
touch, with a gated BFCL run; a full gated rerun read 46.6% / −1.33 pp, within rerun noise.

## Principles

- **No RL.** A length-reward RL post-train is the standard approach, but the compute budget was
  one 12 GB GPU plus a share of a cluster. Everything here is SFT or DPO on sampled traces.
- **Keep the traces in Qwen's own voice.** Fine-tuning on another model's thinking traces did
  poorly, presumably from the distribution shift. So short traces come only from the model
  itself, in two ways: **cut tokens** from its thinking, or **prompt it** to think differently.

## Two sources of short traces

**The truncation oracle** (`mcqwen/oracle.py`). Cut a correct thinking trace at the sentence
ends nearest 10%, 20% … 90%, close it with `</think>`, and re-answer 4 times from each cut.
The *stable* cut is the earliest one from which every later cut is still right; for training
labels, "right" means all 4 re-answers correct (ThoughtFold, ICML 2026, published this kind
of check first and accepts 3 of 4). The oracle needs hindsight, so it is a simulation, not a
model, but it measures the room: cutting the base model's traces at the stable point saved
34.5% at +1.08 pp against answering from the full thinking (BFCL 55%, SVAMP 34%, MBPP+ and
IFEval 22%). Most of the room is in tool calls.

**A distilled prompt** (`mcqwen/distill.py`). An agent searched prompts overnight. Two terse
worked examples in the system prompt beat "be concise" by far; the refined version, `ex-draft`,
is Chain of Draft (Xu et al., 2025): "only keep a minimum draft for each thinking step, with 5
words at most", with two examples. It saved 62% on the search questions, 63% on held-out
questions and 27% on unseen sources (48% counting only questions the base model finished
within the 4,096-token cap). Its real advantage over the plain examples prompt is fewer **leaks**: traces
that mention "the example", which make no sense once the prompt is removed for training (4% of
traces against 13%).

## Training them in

- **SFT on oracle cuts is safe but timid:** 11.4% / −0.34 pp, a third of the oracle's room.
  More epochs only traded accuracy for a little more saving.
- **DPO on oracle pairs.** Shortest-over-longest correct pairs saved 90% and lost 27 pp within
  50 steps: the model never saw a short trace fail, so it never learned when to keep thinking.
  The fix is three kinds of pairs, mixed 0.4 / 0.4 / 0.2: *shorten* (stable cut over the full
  trace), *guard* (stable cut over the cut just before it, which answered wrong) and *hard* (for
  questions with no stable cut, the full trace over an early wrong cut). Step 100 gave
  28.1% / −0.78 pp.
- **The collapse.** Between steps 200 and 250 that run learned to emit `<|endoftext|>` right
  after `</think>` and stopped answering; validation accuracy fell from 63% to 0.8% while
  preference accuracy and reward margin looked healthy. Both sides of every pair end with the
  same answer scaffold, so DPO can push the shared part down for free. An NLL term on the chosen
  answer (the *anchor*) prevents it. `mcqwen/dpo.py` prints a `TRIPWIRE` line when
  P(`<|endoftext|>` after `</think>`) starts to rise.
- **Prompt distillation is plain SFT:** sample with `ex-draft`, strip the prompt, drop leaking
  traces, fine-tune on the 11,247 left. Cuts only remove the end of a trace; the prompt changes
  all of it, which is where the base model's maths wordiness is.

The two are complementary:

| Model | SVAMP | MBPP+ | IFEval | BFCL | Overall |
| --- | --- | --- | --- | --- | --- |
| SFT on the oracle cuts | 7% / −1.7 | 11% / −0.7 | 9% / +0.9 | 18% / +0.2 | 11.4% / −0.34 |
| Oracle DPO (step 100) | 16% / −1.4 | 18% / −1.5 | **32% / −0.5** | **43% / +0.2** | 28.1% / −0.78 |
| Prompt distillation | **52% / −2.9** | 22% / −2.9 | 11% / +0.4 | 12% / +0.3 | 26.5% / −1.27 |

## Combining them

1. **Run the oracle on the distilled model's own traces.** Oracle pairs from the base model's
   traces undo part of the distillation, because those traces think the old way. On-policy:
   the distilled model samples 4 traces for 9,301 other questions, the oracle cuts them (197k
   cuts, 9,034 pairs), and plain DPO from the distilled model, stopped at step 100 of 561,
   gives 42.5% / −1.45 pp. The earlier trade-off line predicted −3.2 pp at that saving, and
   this run doesn't collapse when trained to the end.
2. **Anchored correctness DPO** (`mcqwen/correctness.py`). Pairs that say "be right" and
   nothing about length: the model's own right trace over its own wrong one for the same GSM8K
   or Skywork question, lengths within ±25% (732 pairs). Without an anchor this lost 5.2 pp:
   the two traces share most of their tokens, so pushing the wrong one down drags the right one
   with it. With an NLL term on the whole chosen trace (as in RPO, Pang et al. 2024) it gains
   0.6 pp at the same length: 43.2% / −0.85 pp. The gain landed on MBPP+ and IFEval, not on
   SVAMP.
3. **A think gate for tool calls** (`mcqwen/gate.py`, `evaluate.py --gate`). Often the model
   answers right without thinking. A logistic probe on the layer-21 hidden state at the end of
   the prompt, fitted on 1,500 ToolACE prompts labelled by whether the no-think call matches the
   think calls, decides when to start the reply with an empty thinking block. On BFCL it skips
   30% of prompts, mean tokens 156 → 116, at unchanged accuracy, and still declines irrelevance
   prompts. Probes on maths and code rank prompts about as well, but a wrong skip there is
   expensive (skipping the easiest 5% of SVAMP costs 1.1 pp), so the gate covers tool calls
   only. ThinkSwitcher (EMNLP Findings 2025) uses a similar probe.

## What changed in the thinking

On SVAMP the final model's setup before its first calculation is about half as long as the
base model's, and everything after it, the working and the re-checking, about a quarter. The
cost is that re-checking: where shortening hurts, it is often because the base model's longer
thinking caught its own mistake.

Side effects outside the benchmarks, from a chat-style probe: the model agrees that Sydney is
the capital of Australia 6 of 6 times (base pushes back 3 of 6), keeps "9.11 is greater than
9.9" 6 of 6 times (base fixes it 5 of 6), and invents what the user ate for breakfast 5 of 6
times (base 2 of 6). The visible answer after the thinking also shrank, from 244 tokens to 30
on maths prompts.

## What didn't work

| Idea | Result | Why |
| --- | --- | --- |
| SFT on the full trace when no cut was stable | 5% saved | Full traces pull back toward full length |
| More SFT: more epochs, a second round of cutting, 15× distillation data | No better trade-off | More training on the same targets |
| Fine-tuning the oracle-DPO model on the distilled maths traces | 28.6% / −1.09 pp\* | Copying outputs doesn't carry the other model's training over |
| Merging the two LoRAs | 51.0% / −3.93 pp\* | Near-zero thinking leaked onto easy maths |
| Dropping maths shorten pairs | SVAMP still shortened and lost | Shortening spreads across question types |
| Training the gate into the weights with DPO | −3.3 pp (1 sample) | It called tools on irrelevance prompts |
| Asking a larger model to pick cut points | ~40% / −0.8 pp (pilot) | Cuts at the first right guess: too early on hard questions, too late on easy ones |

\* 2 samples per question, which ran 0.5–1 pp optimistic against 8.

Pairs built from a single trace also can't tell luck from a real mistake: keep the first 10% of
a wrong trace and resample the rest, and 30% come out right.

## Next

- Correctness DPO at a larger scale, over every right/wrong pair of similar length in the
  training data.
- RL from the final checkpoint, if the no-RL constraint is dropped.
- A gate that survives being trained into the weights.
