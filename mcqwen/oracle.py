"""Step 3, the truncation oracle: where could the thinking have stopped?

    sample   the distilled model answers fresh training questions, --traces times each
    cut      every correct trace is cut at the sentence ends nearest 10%, 20% .. 90% of its
             thinking, closed with </think> and re-answered --answers times. A cut is safe when
             every re-answer is right; the stable run is the safe cuts that reach the end
    pairs    DPO pairs, at most one per question and kind; both sides continue the same trace:
               shorten  the stable run's second cut + its shortest right re-answer  >  the full trace
               guard    the same  >  the unsafe cut just before the run + a wrong re-answer
               hard     no stable run: the full trace  >  its unsafe cut nearest half the
                        thinking + a wrong re-answer
             The second cut, not the first, as a margin against lucky cuts. Maths answers must
             parse under SVAMP's grader. 10,000 drawn, 40/40/20 by kind, evenly across sources.

    python -m mcqwen.oracle sample --out work/onpolicy --adapter work/distill/adapter
    python -m mcqwen.oracle cut    --out work/onpolicy --adapter work/distill/adapter
    python -m mcqwen.oracle pairs  --out work/onpolicy
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import statistics

from mcqwen import common as C

FRACTIONS = [i / 10 for i in range(1, 10)]
ANSWER_CAP = 1024  # a re-answer still rambling after this counts as wrong
BOUNDARY = re.compile(r"(?<=[.!?])[ \t]+(?=\S)|\n+")  # sentence or line ends
MARGIN = 1  # earliest stable cuts not used as targets
PAIRS, MIX = 10000, {"shorten": 0.4, "guard": 0.4, "hard": 0.2}


# ---------------------------------------------------------------- sample

def sample(a):
    out = Path(a.out)
    tok = C.Tok()
    qs = [q for q in C.load_questions("train", a.per_source, a.skip_per_source)
          if C.in_shard(q["qid"], a.shard)]
    qs.sort(key=lambda q: q["source"] == "skywork")  # the long ones last, as in distill
    for q in qs:
        q["prompt_ids"] = tok.prompt(q["prompt"])
    C.write(out / "questions.jsonl", qs)
    jobs = ({"key": q["qid"], "ids": q["prompt_ids"], "seed": C.seed_of(q["qid"]),
             "max_tokens": C.CAP} for q in qs)
    llm, lora = C.engine(a.adapter or None)
    C.generate(llm, lora, tok, jobs, out / "sample-gen.jsonl", a.traces, "sample", len(qs))
    del llm
    gen = C.outputs_of(out / "sample-gen.jsonl")
    rows = []
    for q in qs:
        for k, tokens in enumerate(gen[q["qid"]]):
            correct, closed, truncated = C.grade(tok, tokens, q["answer"], q["task"])
            rows.append({"key": f"{q['qid']}#{k}", "qid": q["qid"], "source": q["source"],
                         "tokens": tokens, "correct": correct, "closed": closed,
                         "truncated": truncated})
    C.write(out / "traces.jsonl", rows)
    for source in sorted({r["source"] for r in rows}):
        mine = [r for r in rows if r["source"] == source]
        print(f"{source:15s} acc {sum(r['correct'] for r in mine) / len(mine):.1%}  "
              f"tokens {sum(len(r['tokens']) for r in mine) / len(mine):.0f}")


# ---------------------------------------------------------------- cut

def segment_ends(body):
    """Character offsets where each non-empty sentence (or line) of the thinking ends."""
    ends, start = [], 0
    for match in BOUNDARY.finditer(body):
        if body[start:match.start()].strip():
            ends.append(match.start())
        start = match.end()
    if body[start:].strip():
        ends.append(len(body))
    return ends


def thinking_body(tok, trace):
    """A trace's thinking as text, without the <think> tag, and its sentence ends."""
    body = tok.decode(trace["tokens"][:trace["tokens"].index(tok.close)])
    body = body[len("<think>"):] if body.startswith("<think>") else body
    return body, segment_ends(body)


def cut_think_ids(tok, body, ends, cut):
    """The thinking up to its cut-th sentence, closed."""
    return tok.encode(f"<think>{body[:ends[cut - 1]].rstrip()}\n</think>\n\n")


def cut_points(ends):
    """Distinct cuts on the 10%..90% grid (the most sentences ending within that share of the
    thinking's characters), at least one sentence, never all of them -> {cut: fraction}."""
    cuts = {}
    for f in FRACTIONS:
        limit = f * (ends[-1] if ends else 0)
        cuts.setdefault(sum(1 for e in ends if e <= limit), f)
    return {c: f for c, f in sorted(cuts.items()) if 0 < c < len(ends)}


def oracle_jobs(tok, qs, traces):
    """Correct traces -> (job, meta) per cut, lazily: there are ~10^5 of them."""
    for t in traces:
        if not t["correct"]:
            continue
        body, ends = thinking_body(tok, t)
        for cut, fraction in cut_points(ends).items():
            think = cut_think_ids(tok, body, ends, cut)
            key = f"{t['key']}@{cut}"
            yield ({"key": key, "ids": qs[t["qid"]]["prompt_ids"] + think, "seed": C.seed_of(key),
                    "max_tokens": min(max(C.CAP - len(think), 64), ANSWER_CAP)},
                   {"trace": t["key"], "cut": cut, "fraction": fraction,
                    "think_tokens": len(think)})


def cut(a):
    out = Path(a.out)
    tok = C.Tok()
    qs = {q["qid"]: q for q in C.iread(out / "questions.jsonl")}
    total = sum(len(cut_points(thinking_body(tok, t)[1]))
                for t in C.iread(out / "traces.jsonl") if t["correct"])
    jobs = (job for job, _ in oracle_jobs(tok, qs, C.iread(out / "traces.jsonl")))
    llm, lora = C.engine(a.adapter or None)
    C.generate(llm, lora, tok, jobs, out / "oracle-gen.jsonl", a.answers, "cut", total)


# ---------------------------------------------------------------- pairs

def graded_cuts(tok, qs, traces, sink):
    """Every cut, graded -> {trace key: its cuts in order}. A cut keeps its shortest right
    re-answer that parses and its shortest wrong one."""
    meta = {job["key"]: m for job, m in oracle_jobs(tok, qs, traces)}
    by_trace = defaultdict(list)
    for row in C.iread(sink):
        m = meta.pop(row["key"], None)
        if m is None:
            continue
        q = qs[m["trace"].split("#")[0]]
        graded = [(C.grade(tok, ans, q["answer"], q["task"], prefix=True)[0], ans)
                  for ans in row["outputs"]]
        by_trace[m["trace"]].append({
            **m, "safe": all(c for c, _ in graded),
            "answer": min((ans for c, ans in graded if c and C.eval_valid(tok, ans, q["task"])),
                          key=len, default=None),
            "wrong": min((ans for c, ans in graded if not c), key=len, default=None)})
    if meta:
        raise SystemExit(f"{len(meta)} cuts not re-answered yet, e.g. {list(meta)[:3]}")
    for cuts in by_trace.values():
        cuts.sort(key=lambda c: c["cut"])
    return by_trace


def stable_run(cuts):
    """The run of safe cuts that reaches the end of the grid, in order."""
    stable = []
    for c in reversed(cuts):
        if not c["safe"]:
            break
        stable.append(c)
    return stable[::-1]


def pairs(a):
    out = Path(a.out)
    tok = C.Tok()
    qs = {q["qid"]: q for q in C.iread(out / "questions.jsonl")}
    traces = C.read(out / "traces.jsonl")
    by_trace = graded_cuts(tok, qs, traces, out / "oracle-gen.jsonl")

    def cut_ids(t, c, answer):
        body, ends = thinking_body(tok, t)
        return cut_think_ids(tok, body, ends, c["cut"]) + answer

    best = defaultdict(dict)  # qid -> kind -> (cost, make() -> (chosen, rejected))
    for t in traces:
        if not t["correct"]:
            continue
        q = qs[t["qid"]]
        cuts = by_trace[t["key"]]
        stable = stable_run(cuts)
        usable = [c for c in stable[MARGIN:] if c["answer"] is not None]
        found = []
        if usable:
            c = usable[0]
            cost = c["think_tokens"] + len(c["answer"])
            found.append(("shorten", cost, lambda t=t, c=c: (cut_ids(t, c, c["answer"]), t["tokens"])))
            i = cuts.index(stable[0])
            before = cuts[i - 1] if i > 0 else None
            if before and before["wrong"] is not None:
                found.append(("guard", cost, lambda t=t, c=c, r=before: (
                    cut_ids(t, c, c["answer"]), cut_ids(t, r, r["wrong"]))))
        elif not stable and C.eval_valid(tok, t["tokens"], q["task"]):
            wrong = [c for c in cuts if not c["safe"] and c["wrong"] is not None]
            if wrong:
                r = min(wrong, key=lambda c: abs(c["fraction"] - 0.5))
                found.append(("hard", len(t["tokens"]), lambda t=t, r=r: (
                    t["tokens"], cut_ids(t, r, r["wrong"]))))
        for kind, cost, make in found:  # per question and kind, the cheapest chosen side
            if kind not in best[t["qid"]] or cost < best[t["qid"]][kind][0]:
                best[t["qid"]][kind] = (cost, make)

    rng = random.Random(0)
    rows, table = [], []
    for kind, share in MIX.items():
        pools = defaultdict(list)
        for qid, kinds in best.items():
            if kind in kinds:
                pools[qs[qid]["source"]].append(qid)
        for pool in pools.values():
            rng.shuffle(pool)
        want, n, taken = round(PAIRS * share), 0, defaultdict(list)
        while n < want and any(pools.values()):
            for source in sorted(pools):  # round-robin: every source as evenly as it can
                if pools[source] and n < want:
                    taken[source].append(pools[source].pop())
                    n += 1
        for source, qids in sorted(taken.items()):
            lens = []
            for qid in qids:
                chosen, rejected = best[qid][kind][1]()
                rows.append({"qid": qid, "source": source, "kind": kind,
                             "prompt_ids": qs[qid]["prompt_ids"],
                             "chosen_ids": chosen, "rejected_ids": rejected})
                lens.append((len(chosen), len(rejected)))
            table.append(f"| {kind} | {source} | {len(qids)} | {len(pools[source]) + len(qids)} "
                         f"| {statistics.mean(c for c, _ in lens):.0f} "
                         f"| {statistics.mean(r for _, r in lens):.0f} |")
    rng.shuffle(rows)  # so the DPO step's held-out tail is a random draw
    C.write(out / "pairs.jsonl", rows)
    print("| kind | source | pairs | available | chosen tokens | rejected tokens |\n"
          "|---|---|---:|---:|---:|---:|\n" + "\n".join(table))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["sample", "cut", "pairs"])
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default="", help="sample, cut: the model that writes and re-answers")
    p.add_argument("--per-source", type=int, default=2500)
    p.add_argument("--skip-per-source", type=int, default=250,
                   help="skip each source's first N questions")
    p.add_argument("--traces", type=int, default=4)
    p.add_argument("--answers", type=int, default=4, help="re-answers per cut")
    p.add_argument("--shard", default="", help="i/N: this job's share of the questions")
    a = p.parse_args()
    {"sample": sample, "cut": cut, "pairs": pairs}[a.step](a)
