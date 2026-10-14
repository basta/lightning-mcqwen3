"""Step 7, the think gate: a linear probe that tells, from the prompt alone, when the model's
tool call won't change if it skips thinking. evaluate.py --gate applies it.

    prep     1,500 ToolACE prompts (Team-ACE/ToolACE, first user turn) rendered in the BFCL
             system prompt, dropping any that shares a function name with BFCL; 15% get
             another prompt's tools, as irrelevance cases
    gen      the model samples 4 calls with thinking and 4 without for each
    feats    its hidden state at layer LAYER, last position of the prompt + "<think>\\n"
    fit      label: thinking doesn't matter (all 4 no-think calls agree, at least 3 of 4
             thinking calls agree, and the two agree); no correctness labels, no benchmark
             data. L2 logistic regression; the threshold fires on half of the ToolACE prompts
             by 5-fold cross-validated score. Saves gate.pt

    python -m mcqwen.gate prep  --out work/gate
    python -m mcqwen.gate gen   --out work/gate --adapter work/correctness/adapter
    python -m mcqwen.gate feats --out work/gate --adapter work/correctness/adapter
    python -m mcqwen.gate fit   --out work/gate --adapter work/correctness/adapter
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random

from mcqwen import common as C
from utils.eval import graders

N_EXT, SWAP, K = 1500, 0.15, 4
THINK_CAP, NOTHINK_CAP = 2048, 512
LAYER, WD, COVERAGE = 21, 0.1, 0.5
HEADER = "Here is a list of functions in json format that you can invoke.\n"
SCOPE = HEADER.strip()  # the gate only judges prompts that offer tools
DECIDE, FORCE = "<think>\n", "<think>\n\n</think>\n\n"


def prep(a):
    from datasets import load_dataset
    rows = [r for r in C.iread(C.REPO / "questions/questions.jsonl") if r["grader"] == "bfcl"]
    # BFCL is used for its system prompt's wording and to keep its functions out, nothing else
    prefix = rows[0]["messages"][0]["content"].split(HEADER)[0] + HEADER
    bfcl_names = set()
    for r in rows:
        bfcl_names |= {t["name"] for t in json.loads(r["messages"][0]["content"].split(HEADER)[1])}
    pool = []
    for i, r in enumerate(load_dataset("Team-ACE/ToolACE", split="train")):
        conv = r["conversations"]
        if not conv or conv[0]["from"] != "user":
            continue
        s = r["system"]
        try:
            tools, _ = json.JSONDecoder().raw_decode(s[s.find("[", s.find("invoke")):])
        except ValueError:
            continue
        if not tools or not all(isinstance(t, dict) and "name" in t for t in tools):
            continue
        if {t["name"] for t in tools} & bfcl_names:
            continue
        pool.append({"src": i, "tools": tools, "query": conv[0]["value"]})
    rng = random.Random(0)
    out = []
    for n, p in enumerate(rng.sample(pool, N_EXT)):
        tools, kind = p["tools"], "toolace"
        if rng.random() < SWAP:
            names = {t["name"] for t in tools}
            other = rng.choice([q for q in pool if not ({t["name"] for t in q["tools"]} & names)])
            tools, kind = other["tools"], "swapped"
        out.append({"id": f"ext{n}", "kind": kind, "src": p["src"],
                    "messages": [{"role": "system",
                                  "content": prefix + json.dumps(tools, indent=4) + "\n"},
                                 {"role": "user", "content": p["query"]}]})
    C.write(Path(a.out) / "prompts.jsonl", out)
    print(f"{len(pool)} usable ToolACE prompts; wrote {len(out)} "
          f"({sum(o['kind'] == 'swapped' for o in out)} with swapped tools)")


def gen(a):
    out = Path(a.out)
    tok = C.Tok()
    prompts = C.read(out / "prompts.jsonl")[:a.limit or None]
    think = [{"key": r["id"], "ids": tok.prompt(r["messages"]),
              "seed": C.seed_of("think:" + r["id"]), "max_tokens": THINK_CAP} for r in prompts]
    nothink = [{"key": r["id"], "seed": C.seed_of("nothink:" + r["id"]), "max_tokens": NOTHINK_CAP,
                "ids": tok.t.apply_chat_template(r["messages"], tokenize=True,
                                                 add_generation_prompt=True, enable_thinking=False)}
               for r in prompts]
    llm, lora = C.engine(a.adapter)
    C.generate(llm, lora, tok, nothink, out / "gen-nothink.jsonl", K, "nothink", len(nothink))
    C.generate(llm, lora, tok, think, out / "gen-think.jsonl", K, "think", len(think))


def canon(reply, truncated, tag):
    """A reply's tool calls, canonical; every truncated reply differs from every other."""
    if truncated:
        return f"TRUNC:{tag}"
    try:
        return json.dumps(graders.decode_calls(reply), sort_keys=True)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return "NOCALL"


def labels(tok, out):
    """-> {prompt id: thinking doesn't change the call}."""
    think, nothink = C.outputs_of(out / "gen-think.jsonl"), C.outputs_of(out / "gen-nothink.jsonl")
    got = {}
    for key in think.keys() & nothink.keys():
        tcalls = []
        for i, o in enumerate(think[key]):
            trunc = not (o and o[-1] == tok.eos) or tok.close not in o
            tcalls.append(canon("" if trunc else tok.decode(o[o.index(tok.close) + 1:]), trunc, f"t{i}"))
        ncalls = [canon(tok.decode(o), not (o and o[-1] == tok.eos), f"n{i}")
                  for i, o in enumerate(nothink[key])]
        top, count = Counter(tcalls).most_common(1)[0]
        got[key] = len(set(ncalls)) == 1 and count >= 3 and ncalls[0] == top
    return got


def feats(a):
    import torch
    out = Path(a.out)
    tok = C.Tok()
    model = C.load_model(a.adapter).eval()
    decide = tok.encode(DECIDE)
    ids, X = [], []
    with torch.no_grad():
        for n, r in enumerate(C.read(out / "prompts.jsonl")[:a.limit or None]):
            h = model(torch.tensor([tok.prompt(r["messages"]) + decide]).cuda(),
                      output_hidden_states=True).hidden_states[LAYER]
            X.append(h[0, -1].float().cpu())
            ids.append(r["id"])
            if n % 200 == 0:
                print(f"feats {n}", flush=True)
    torch.save({"ids": ids, "X": torch.stack(X)}, out / "feats.pt")


def fit(X, y):
    """Standardised L2 logistic regression -> (mu, sd, w, b)."""
    import torch
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
    yt = torch.tensor(y, dtype=torch.float32)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, yt) + WD * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return mu, sd, w.detach(), b.detach()


def cv_scores(X, y, folds=5):
    idx = list(range(len(y)))
    random.Random(0).shuffle(idx)
    scores = [0.0] * len(y)
    for f in range(folds):
        test = idx[f::folds]
        train = sorted(set(idx) - set(test))
        mu, sd, w, b = fit(X[train], [y[i] for i in train])
        for i, s in zip(test, (((X[test] - mu) / sd) @ w + b).tolist()):
            scores[i] = s
    return scores


def export(a):
    import torch
    out = Path(a.out)
    lab = labels(C.Tok(), out)
    data = torch.load(out / "feats.pt")
    pos = {pid: i for i, pid in enumerate(data["ids"])}
    keys = sorted(lab)
    y = [int(lab[k]) for k in keys]
    X = data["X"][[pos[k] for k in keys]]
    cv = sorted(cv_scores(X, y), reverse=True)
    mu, sd, w, b = fit(X, y)
    gate = {"layer": LAYER, "mu": mu, "sd": sd, "w": w, "b": b,
            "threshold": cv[int(COVERAGE * len(cv)) - 1], "scope": SCOPE, "decide": DECIDE,
            "force": FORCE, "adapter": str(a.adapter)}
    torch.save(gate, out / "gate.pt")
    print(f"{len(keys)} prompts, {sum(y) / len(y):.1%} where thinking doesn't change the call; "
          f"saved {out / 'gate.pt'} (threshold {gate['threshold']:.3f})")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["prep", "gen", "feats", "fit"])
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default="", help="gen, feats: the model the gate is for")
    p.add_argument("--limit", type=int, default=0, help="gen, feats: the first N prompts only")
    a = p.parse_args()
    {"prep": prep, "gen": gen, "feats": feats, "fit": export}[a.step](a)
