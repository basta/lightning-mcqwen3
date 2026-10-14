"""Step 5, correctness pairs: the on-policy sample's own right maths trace (chosen) against its
own wrong one (rejected) for the same training question, of about the same length, so the
pairs teach "be right" without "be longer".

Only GSM8K and Skywork questions with both a right and a wrong trace, both closed and neither
truncated; lengths within 25% of each other, up to 2 pairs per question on disjoint traces,
closest first.

    python -m mcqwen.correctness --traces work/onpolicy --out work/correctness/pairs.jsonl
"""
import argparse
from collections import Counter, defaultdict
import random

from mcqwen import common as C

SOURCES = ("gsm8k", "skywork")
TOLERANCE = 0.25
PER_QUESTION = 2


def match(right, wrong):
    """Up to PER_QUESTION (chosen, rejected) pairs with lengths within TOLERANCE, closest first."""
    cands = sorted((abs(len(r["tokens"]) / len(w["tokens"]) - 1), i, j)
                   for i, r in enumerate(right) for j, w in enumerate(wrong))
    used_r, used_w, out = set(), set(), []
    for d, i, j in cands:
        if d > TOLERANCE or len(out) == PER_QUESTION:
            break
        if i not in used_r and j not in used_w:
            used_r.add(i)
            used_w.add(j)
            out.append((right[i], wrong[j]))
    return out


def main(a):
    prompts = {q["qid"]: q["prompt_ids"] for q in C.iread(f"{a.traces}/questions.jsonl")
               if q["source"] in SOURCES}
    by = defaultdict(list)
    for t in C.iread(f"{a.traces}/traces.jsonl"):
        if t["source"] in SOURCES and t["closed"] and not t["truncated"]:
            by[t["qid"]].append(t)
    rows, n = [], Counter()
    for qid, ts in sorted(by.items()):
        right = [t for t in ts if t["correct"]]
        wrong = [t for t in ts if not t["correct"]]
        for r, w in match(right, wrong):
            n[r["source"]] += 1
            rows.append({"qid": qid, "source": r["source"], "kind": "correct",
                         "prompt_ids": prompts[qid], "chosen_ids": r["tokens"],
                         "rejected_ids": w["tokens"]})
    random.Random(0).shuffle(rows)  # so the DPO step's held-out tail is a random draw
    C.write(a.out, rows)
    print(f"{len(rows)} pairs {dict(n)} -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, help="the oracle step's directory")
    p.add_argument("--out", required=True)
    main(p.parse_args())
