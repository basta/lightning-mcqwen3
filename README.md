# Lightning McQwen3

Qwen3-0.6B, taught to think shorter: **47% fewer output tokens at −0.9 pp accuracy** on
[BottleCapAI's ThinkingCap test](CHALLENGE.md) (SVAMP, MBPP+, IFEval, BFCL; 8 samples per
question, paired with the base model).

This repository is a fork of [BottleCapAI/ThinkingCap-Test](https://github.com/BottleCapAI/ThinkingCap-Test).
The evaluation harness, the training data preparation and the challenge description
([CHALLENGE.md](CHALLENGE.md)) are theirs; this fork adds the recipe in [`mcqwen/`](mcqwen)
and a think gate option to `evaluate.py`.

- Write-up: [Lightning McQwen3](https://blog.basta.one/blog/thinkingcap/): how the recipe was
  found, what didn't work, and interactive figures; [IDEA.md](IDEA.md) is the short version:
  what I tried and why, what worked, what didn't, and the hardware
- Weights: [basta/lightning-mcqwen3](https://huggingface.co/basta/lightning-mcqwen3) on Hugging
  Face, the LoRA adapter and `gate.pt`

## Results

Against Qwen3-0.6B, 8 samples per question, 95% intervals from a paired bootstrap over
questions:

| | Tokens saved | Accuracy change |
| --- | ---: | ---: |
| SVAMP (maths) | 66.3% | −3.8 pp |
| MBPP+ (code) | 39.1% | −1.9 pp |
| IFEval (instructions) | 32.0% | +1.4 pp |
| BFCL (tool calls) | 44.6% | +0.8 pp |
| **All** | **47.3%** [45.8, 48.7] | **−0.88 pp** [−1.61, −0.15] |

Without the gate, the weights alone save 43.2% at −0.85 pp; the gate only changes BFCL
(25.3% → 44.6% saved there). The headline combines the ungated runs on SVAMP, MBPP+ and IFEval,
which the gate doesn't touch, with a gated BFCL run. A full gated rerun read 46.6% / −1.33 pp,
within rerun noise.

## How

Every step is a LoRA (r16, alpha 32, all linear layers) continued from the one before. Only
the challenge's suggested training data (`data/train.jsonl`) is used; the benchmarks are never
trained on.

1. **Prompt distillation** (`mcqwen/distill.py`, `mcqwen/sft.py`). A Chain-of-Draft system
   prompt ("only keep a minimum draft for each thinking step, with 5 words at most", two worked
   examples) answers 5,000 training questions per source twice. One correct trace per question,
   with the prompt removed and traces that mention the examples dropped, makes 11,247 SFT
   targets. SFT from the base model: lr 1e-4, 1 epoch.
2. **On-policy oracle** (`mcqwen/oracle.py`). The distilled model answers 2,500 other questions
   per source 4 times. Every correct trace is cut at the sentence ends nearest 10%, 20% … 90%
   of its thinking, closed with `</think>` and re-answered 4 times from each cut (197k cuts). A
   cut is safe when all 4 re-answers are right; the stable run is the safe cuts that reach
   the end.
3. **Oracle DPO** (`mcqwen/oracle.py pairs`, `mcqwen/dpo.py`). 9,034 pairs, 40% *shorten*
   (the stable run's second cut and its answer over the full trace), 40% *guard* (the same over
   the unsafe cut before it and a wrong answer), 20% *hard* (no stable run: the full trace over
   a cut near the middle and a wrong answer). DPO from the distilled adapter, lr 2e-5, β 0.1,
   batch 16, stopped at step 100 of 561.
4. **Anchored correctness DPO** (`mcqwen/correctness.py`, `mcqwen/dpo.py --nll 1`). From the
   traces step 2 already sampled, the model's own right maths trace over its own wrong one for
   the same question, lengths within ±25% (732 pairs), so the pairs say "be right" and nothing
   about length. DPO plus an NLL term on the whole right trace, 1 epoch (42 steps).
5. **Think gate** (`mcqwen/gate.py`, `evaluate.py --gate`). A logistic probe on the model's
   layer-21 hidden state at the end of the prompt predicts whether thinking would change its
   tool call. It is fitted on 1,500 ToolACE prompts, labelled by whether 4 sampled calls with
   thinking and 4 without agree; no correctness labels and no benchmark data. On tool-call
   prompts where it fires (half of them on ToolACE), the reply starts with an empty thinking
   block, and those forced tokens count as output.

In DPO, the reference is the starting adapter: its log-probabilities are computed once before
training. The NLL anchor in step 4 matters: right-versus-wrong pairs without it lose accuracy,
because DPO can widen the margin by lowering both sides. During DPO, `mcqwen/dpo.py` scores 64
held-out pairs every 25 steps and prints a `TRIPWIRE` line when the probability of
`<|endoftext|>` right after `</think>` starts to rise, which is how an unanchored run
collapsed into empty answers.

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
scripts/reproduce.sh          # every step, then the evaluation; outputs in work/
```

The script lists every command with its settings; each step resumes where it stopped. The
sampling steps (1 and 2) took most of the compute: they ran as 8 shards on 40 GB cluster GPUs,
a few hours each. Everything else ran on one 12 GB RTX 4070 SUPER: about 5.5 s per DPO step,
and about 1.5 h for an 8-sample evaluation, mostly MBPP+. With 8 shards merged in order, as in
the script, the data steps rebuild the published training files byte for byte from the same
samples, and the gate refits to identical weights. GPU sampling is not bit-reproducible, and
neither is the correctness DPO: two reruns from the same pairs and reference agree on the first
step's loss and already differ in the second.

To evaluate the published adapter and gate directly:

```bash
hf download basta/lightning-mcqwen3 --local-dir weights
python evaluate.py --out runs/base
python evaluate.py --out runs/final --adapter weights --gate weights/gate.pt
python compare.py runs/base runs/final
```

## Credit

The challenge, its evaluation harness and the training data selection are by
[BottleCapAI](https://www.bottlecapai.com) (Apache 2.0, see [LICENSE](LICENSE)). The prompt is
Chain of Draft (Xu et al., 2025). The truncation oracle with guard pairs is close to
ThoughtFold (ICML 2026), and the probe gate is a label-free variant of ThinkSwitcher
(EMNLP Findings 2025).
