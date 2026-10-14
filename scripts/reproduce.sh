#!/bin/bash
# The whole recipe, from base Qwen3-0.6B to the gated final model, then the evaluation.
#
#   scripts/reproduce.sh                # everything, into work/
#   WORK=/big/disk scripts/reproduce.sh
#
# Every step resumes where it stopped, so rerunning after a crash or a kill is safe. The two
# sampling steps are split into SHARDS shards run one after another here; they are independent,
# so on a cluster run each `--shard i/N` on its own GPU and merge as below. 8 shards, merged in
# order, give exactly the files the published model was trained on: a different count gives the
# same questions in a different order, and so a different random choice of SFT targets.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${WORK:-work}
SHARDS=${SHARDS:-8}
PY=${PY:-python}
shards() { for i in $(seq 0 $((SHARDS - 1))); do echo "$1/shard-$i"; done; }

# data/train.jsonl, data/val.jsonl (the challenge's suggested training data) and the benchmarks
[ -f data/train.jsonl ] || $PY utils/train/prepare_data.py
[ -f questions/questions.jsonl ] || $PY utils/eval/benchmarks.py

# 1. Prompt distillation: Chain-of-Draft traces for 5,000 questions per source, 2 each;
#    one correct trace per question with the prompt removed (11,247 targets)
for i in $(seq 0 $((SHARDS - 1))); do
  $PY -m mcqwen.distill sample --out $W/distill/shard-$i --shard $i/$SHARDS
done
$PY -m mcqwen.merge $W/distill $(shards $W/distill)
$PY -m mcqwen.distill build --out $W/distill

# 2. SFT: a fresh LoRA (r16, all linear layers) on those targets, 1 epoch
[ -f $W/distill/adapter/adapter_config.json ] ||
  $PY -m mcqwen.sft --data $W/distill/sft.jsonl --out $W/distill/adapter

# 3. On-policy oracle: the distilled model answers 2,500 other questions per source 4 times;
#    every correct trace is cut at 10..90% and re-answered 4 times from each cut
for i in $(seq 0 $((SHARDS - 1))); do
  $PY -m mcqwen.oracle sample --out $W/onpolicy/shard-$i --shard $i/$SHARDS --adapter $W/distill/adapter
  $PY -m mcqwen.oracle cut --out $W/onpolicy/shard-$i --adapter $W/distill/adapter
done
$PY -m mcqwen.merge $W/onpolicy $(shards $W/onpolicy)

# 4. Oracle DPO: shorten / guard / hard pairs (9,034), stopped at step 100 of 561
$PY -m mcqwen.oracle pairs --out $W/onpolicy
[ -f $W/onpolicy/adapter/adapter_config.json ] ||
  $PY -m mcqwen.dpo --pairs $W/onpolicy/pairs.jsonl --init $W/distill/adapter \
    --out $W/onpolicy/adapter --stop-at 100

# 5-6. Anchored correctness DPO: the sampled maths traces, right over wrong at matched
#      length (732 pairs), with an NLL anchor on the right one, 1 epoch (42 steps)
$PY -m mcqwen.correctness --traces $W/onpolicy --out $W/correctness/pairs.jsonl
[ -f $W/correctness/adapter/adapter_config.json ] ||
  $PY -m mcqwen.dpo --pairs $W/correctness/pairs.jsonl --init $W/onpolicy/adapter \
    --out $W/correctness/adapter --nll 1

# 7. The think gate, fitted on ToolACE prompts only
$PY -m mcqwen.gate prep --out $W/gate
$PY -m mcqwen.gate gen --out $W/gate --adapter $W/correctness/adapter
$PY -m mcqwen.gate feats --out $W/gate --adapter $W/correctness/adapter
$PY -m mcqwen.gate fit --out $W/gate --adapter $W/correctness/adapter

# Evaluation: 8 samples per question, paired with the base model's run
[ -f runs/base/results.json ] || $PY evaluate.py --samples 8 --out runs/base
$PY evaluate.py --adapter $W/correctness/adapter --gate $W/gate/gate.pt --samples 8 --out runs/final
$PY compare.py runs/base runs/final
