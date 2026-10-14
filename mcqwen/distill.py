"""Step 1, prompt distillation: answer training questions with a Chain-of-Draft system prompt,
keep one correct trace per question and drop the prompt, giving SFT targets in which the
model writes short drafts unprompted.

    sample   --traces answers per question with the prompt (data/train.jsonl, the first
             --per-source questions of each source; --shard i/N for one job's share)
    build    one target per question: a random correct trace, never one that leaks the
             prompt (mentions its examples), maths answers only in a form SVAMP's grader parses

    python -m mcqwen.distill sample --out work/distill --per-source 5000 --traces 2
    python -m mcqwen.distill build --out work/distill
"""
import argparse
from pathlib import Path
import random
import re

from mcqwen import common as C

# Chain of Draft (Xu et al. 2025) in its few-shot form; found by scoring candidate prompts on
# validation questions for tokens saved at equal accuracy
SYSTEM = (
    "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at "
    "most, like in these two examples.\n\n"
    "Example 1\nQuestion: A shop sells pens at $3 each. Tom buys 4 pens and pays with a $20 "
    "bill. How much change does he get?\nThinking: 4 x 3 = 12; 20 - 12 = 8.\nFinal answer: 8\n\n"
    "Example 2\nQuestion: Where would you keep milk to stop it from spoiling?\nA. cupboard\n"
    "B. refrigerator\nC. oven\nD. table\nThinking: cold stops spoiling: refrigerator.\n"
    "Final answer: B")
# thinking that refers to the prompt makes no sense once the prompt is gone
LEAK = re.compile(r"\b(examples|example [12]|(?:first|second|previous|given|provided|above) "
                  r"example|system prompt)\b", re.I)
LEAK_WORDS = ["pens?", "milk", "Tom", "refrigerator", r"spoil\w*"]  # from the examples


def leaks(question, thinking):
    if LEAK.search(thinking):
        return True
    return any(re.search(rf"\b{w}\b", thinking, re.I) and not re.search(rf"\b{w}\b", question, re.I)
               for w in LEAK_WORDS)


def sample(a):
    out = Path(a.out)
    tok = C.Tok()
    qs = [q for q in C.load_questions("train", a.per_source, a.skip_per_source)
          if C.in_shard(q["qid"], a.shard)]
    # Skywork last: it is most of the tokens and the fewest solved questions, so a run that
    # runs out of time loses the least useful part
    qs.sort(key=lambda q: q["source"] == "skywork")
    for q in qs:
        q["prompt_ids"] = tok.prompt(q["prompt"])  # the target's prompt: without the system one
    C.write(out / "questions.jsonl", qs)
    system = [{"role": "system", "content": SYSTEM}]
    jobs = ({"key": q["qid"], "ids": tok.prompt(system + q["prompt"]), "seed": C.seed_of(q["qid"]),
             "max_tokens": C.CAP} for q in qs)
    llm, lora = C.engine()
    C.generate(llm, lora, tok, jobs, out / "sample-gen.jsonl", a.traces, "distill", len(qs))
    del llm
    gen = C.outputs_of(out / "sample-gen.jsonl")
    rows = []
    for q in qs:
        for k, tokens in enumerate(gen[q["qid"]]):
            correct, _, truncated = C.grade(tok, tokens, q["answer"], q["task"])
            think = tokens.index(tok.close) if tok.close in tokens else len(tokens)
            rows.append({"key": f"{q['qid']}#{k}", "qid": q["qid"], "source": q["source"],
                         "tokens": tokens, "correct": correct, "truncated": truncated,
                         "leak": leaks(q["prompt"][-1]["content"], tok.decode(tokens[:think]))})
    C.write(out / "traces.jsonl", rows)
    for source in sorted({r["source"] for r in rows}):
        mine = [r for r in rows if r["source"] == source]
        print(f"{source:15s} acc {sum(r['correct'] for r in mine) / len(mine):.1%}  "
              f"tokens {sum(len(r['tokens']) for r in mine) / len(mine):.0f}  "
              f"leaks {sum(r['leak'] for r in mine) / len(mine):.1%}")


def build(a):
    out = Path(a.out)
    tok = C.Tok()
    rng = random.Random(0)
    traces = {}
    for t in C.iread(out / "traces.jsonl"):
        traces.setdefault(t["qid"], []).append(t)
    rows = []
    for q in C.iread(out / "questions.jsonl"):
        ts = traces.get(q["qid"], [])
        good = [t for t in ts if t["correct"] and C.eval_valid(tok, t["tokens"], q["task"])
                and not t["leak"]]
        if not good:
            continue
        target = rng.choice(good)
        rows.append({"qid": q["qid"], "source": q["source"], "prompt_ids": q["prompt_ids"],
                     "completion_ids": target["tokens"], "kind": "prompted"})
        # the original script also drew a wrong trace here, for DPO pairs this recipe doesn't
        # use; the draw stays so the targets come out identical
        wrong = [t for t in ts if not t["correct"] and not t["truncated"] and not t["leak"]]
        if wrong:
            rng.choice(wrong)
    C.write(out / "sft.jsonl", rows)
    print(f"{len(rows)} SFT targets -> {out / 'sft.jsonl'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["sample", "build"])
    p.add_argument("--out", required=True)
    p.add_argument("--per-source", type=int, default=5000)
    p.add_argument("--skip-per-source", type=int, default=0)
    p.add_argument("--traces", type=int, default=2)
    p.add_argument("--shard", default="", help="i/N: this job's share of the questions")
    a = p.parse_args()
    {"sample": sample, "build": build}[a.step](a)
