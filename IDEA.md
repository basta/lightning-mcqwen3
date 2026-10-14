# IDEA

Qwen3-0.6B, taught to think shorter with thinking still on: **47.3% fewer tokens [45.8, 48.7]
at −0.88 pp accuracy [−1.61, −0.15]** against the base model, 8 samples per question. The
full story, with figures: [the write-up](https://blog.basta.one/blog/thinkingcap/). Exact
settings: the [README](README.md).

## What I tried and why

- **No RL.** A length-reward RL post-train is the obvious approach, but the budget was mostly one
  12 GB GPU. Everything is SFT or DPO on sampled traces instead.
- **Short traces in the model's own voice.** Fine-tuning on another model's thinking did
  poorly, presumably from the distribution shift, so short traces come only from Qwen itself:
  either *cut* its thinking, or *prompt* it to think tersely.
- **A truncation oracle** to find out how much can be cut. Cut a correct trace at 10%, 20% …
  90% of its thinking, close it with `</think>`, re-answer 4 times from each cut, and keep the
  earliest cut after which every cut stays right (all 4 re-answers, for training labels). It
  needs hindsight, so it is a measurement, not a model: on the base model's traces it saved
  34.5% at +1.08 pp, most of it on tool calls (BFCL 55%).
- **A terse prompt, distilled into the weights.** An agent searched prompts overnight; two
  worked examples in Chain-of-Draft style ("a minimum draft for each thinking step, with 5
  words at most") saved about 60% on training questions, and few traces mention the examples,
  so the prompt can be removed for SFT.
- **Training the oracle's cuts in**, first with SFT, then with DPO pairs.

## What worked

Each step continues from the one before; tokens saved / accuracy change:

| Step | Saved | Δ accuracy |
| --- | ---: | ---: |
| 1. SFT on the distilled prompt's traces (11,247) | 26.5% | −1.27 pp |
| 2. DPO on oracle pairs from the distilled model's own traces, stopped at step 100 | 42.5% | −1.45 pp |
| 3. Anchored correctness DPO (732 pairs) | 43.2% | −0.85 pp |
| 4. + think gate on tool calls | **47.3%** | **−0.88 pp** |

- **Guard pairs make DPO safe.** Shortest-over-longest pairs alone saved 90% and lost 27 pp
  within 50 steps. Mixing *shorten* (stable cut over full trace), *guard* (stable cut over
  the wrong cut just before it) and *hard* (full trace over an early wrong cut, for questions
  with no stable cut) at 0.4 / 0.4 / 0.2 keeps the model thinking on hard questions.
- **The two sources are complementary.** Prompt distillation shortens maths (SVAMP 52%), the
  oracle shortens tool calls and instructions (BFCL 43%, IFEval 32%). Running the oracle on
  the *distilled* model's own traces (on-policy) gets both: step 2 sits 1.8 pp above the
  earlier trade-off line.
- **An NLL anchor** on the chosen side. Right-over-wrong pairs at matched length lost 5.2 pp
  without it, because pushing the wrong trace down drags the near-identical right one along;
  with it, they recover 0.6 pp at the same length (on MBPP+ and IFEval).
- **A probe gate for tool calls.** A logistic probe on the layer-21 hidden state at the end of
  the prompt, fitted on 1,500 ToolACE prompts (not BFCL), starts the reply with an empty
  thinking block where it predicts thinking won't change the call: BFCL 25% → 45% saved at
  unchanged accuracy. The headline combines the ungated runs on the other three benchmarks
  with a gated BFCL run; a full gated rerun read 46.6% / −1.33 pp, within rerun noise.

Per benchmark: SVAMP 66% / −3.8 pp, MBPP+ 39% / −1.9 pp, IFEval 32% / +1.4 pp,
BFCL 45% / +0.8 pp. Maths and code pay for it; the base model's longer re-checking sometimes
caught its own mistake.

## What didn't

| Idea | Result | Why |
| --- | --- | --- |
| SFT on the oracle's cuts | 11.4% / −0.34 pp | Safe but timid: a third of the oracle's room; more epochs collapse accuracy |
| DPO on oracle pairs from the *base* model, trained to the end | Collapsed at step ~200: emitted `<\|endoftext|>` right after `</think>` | Both sides share the answer scaffold, so DPO pushed it down for free; an anchor fixes it but lands back on the old trade-off line (47.8% / −4.15 pp) |
| SFT on the full trace when no cut was stable | 5% saved | Full traces pull back toward full length |
| More SFT data or rounds (15× distillation data, a second cutting round) | No further shortening | Same targets, same trade-off |
| Fine-tuning the oracle-DPO model on distilled maths traces | 28.6% / −1.09 pp\* | Lost DPO's BFCL and IFEval shortening |
| Merging the two LoRAs | 51.0% / −3.93 pp\* | Near-zero thinking leaked onto easy maths |
| Gating maths and code too | Skipping the easiest 5% of SVAMP: −1.1 pp | A wrong skip is expensive there |
| Training the gate into the weights with DPO | −3.3 pp | Called tools on irrelevance prompts |
| A larger model choosing the cut points | ~40% / −0.8 pp (pilot) | Cuts at the first right guess: too early on hard questions, too late on easy ones |

\* 2 samples per question, which ran 0.5–1 pp optimistic against 8.

Side effects outside the benchmarks: the final model pushes back less (agrees Sydney is
Australia's capital 6 of 6 times, base 3 of 6) and invents missing context more often.

## Hardware

- **One RTX 4070 SUPER (12 GB)** for all DPO (about 5.5 s per step for the oracle DPO, 11–14 s
  for correctness DPO), the gate, and most evaluations (about 1.5 h for an 8-sample run).
- **MetaCentrum**, the Czech national grid, for sampling and the oracle (8 shards on 40 GB
  GPUs, a few hours each) and for SFT (a 24 GB GPU).
