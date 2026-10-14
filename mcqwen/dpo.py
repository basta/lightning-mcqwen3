"""Steps 4 and 6, DPO: continue an adapter on a pairs file.

The reference is the starting adapter: its sequence log-probabilities (summed over the
completion) are computed once, before training, into <pairs>-ref.jsonl. --nll adds that weight
times the chosen completion's mean NLL. DPO alone only widens the margin between the two
sides, and once let `</think>` and the answer after it sink towards <|endoftext|>; the anchor
holds the chosen side's likelihood in place.

    python -m mcqwen.dpo --pairs work/onpolicy/pairs.jsonl --init work/distill/adapter \\
        --out work/onpolicy/adapter --stop-at 100
    python -m mcqwen.dpo --pairs work/correctness/pairs.jsonl --init work/onpolicy/adapter \\
        --out work/correctness/adapter --nll 1

The last HELD_OUT pairs are not trained on. Every PROBE_EVERY steps they are scored, and every
SMOKE_EVERY steps the model also answers a few validation questions; a TRIPWIRE line flags the
collapse's first signs. --stop-at ends the run early on the full run's learning-rate schedule.
Checkpoints go to <out>-ckpt; rerunning the command resumes there.
"""
import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import time

from mcqwen import common as C
from mcqwen.sft import cosine, load_checkpoint, save_checkpoint
from utils.eval.recipe import SAMPLING, SEED

LOGIT_CHUNK = 1024  # positions per output-layer chunk
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 50))
PROBE_EVERY = int(os.environ.get("PROBE_EVERY", 25))
SMOKE_EVERY = int(os.environ.get("SMOKE_EVERY", 100))
SMOKE_N = int(os.environ.get("SMOKE_N", 16))  # smoke questions per source
SMOKE_SOURCES = {"arc_easy": 1024, "commonsense_qa": 1024, "gsm8k": 2048}  # source -> token cap
HELD_OUT = 64


def smoke_questions():
    """Per source, the SMOKE_N validation questions after its first 50."""
    seen, rows = defaultdict(int), []
    for q in C.load_questions("val", 0):
        if q["source"] in SMOKE_SOURCES:
            seen[q["source"]] += 1
            if 50 < seen[q["source"]] <= 50 + SMOKE_N:
                rows.append(q)
    return rows


def main(a):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as sft
    import torch
    import torch.nn.functional as F
    import torch.utils.checkpoint
    from peft import PeftModel

    out = Path(a.out)
    ckpt = out.with_name(out.name + "-ckpt")
    tok = C.Tok()
    eot = tok.t.convert_tokens_to_ids("<|endoftext|>")
    rows = C.read(a.pairs)[:a.limit or None]
    torch.manual_seed(0)
    resume = (ckpt / "state.pt").exists()
    model = PeftModel.from_pretrained(C.load_model(trainable=True), ckpt if resume else a.init,
                                      is_trainable=True)
    base = model.get_base_model()

    def token_nll(hidden, targets):
        return F.cross_entropy(base.lm_head(hidden).float(), targets, reduction="none")

    def token_logps(prompt, completion, probe=None):
        """Per-token log-probabilities of the completion; with probe, also the next-token
        distribution at completion position probe. The output layer runs a chunk of positions
        at a time and is recomputed in the backward pass: a 4k-token completion's float
        logits alone would take 2.3 GiB."""
        ids = torch.tensor([prompt + completion], device="cuda")
        hidden = base.model(input_ids=ids).last_hidden_state[0, len(prompt) - 1:-1]
        targets = ids[0, len(prompt):]
        lps = -torch.cat([torch.utils.checkpoint.checkpoint(
            token_nll, hidden[i:i + LOGIT_CHUNK], targets[i:i + LOGIT_CHUNK], use_reentrant=False)
            for i in range(0, len(targets), LOGIT_CHUNK)])
        if probe is None:
            return lps
        return lps, base.lm_head(hidden[probe]).float().softmax(-1)

    def answer_start(completion):
        return completion.index(tok.close) if tok.close in completion else len(completion)

    ref_file = Path(a.pairs).with_name(Path(a.pairs).stem + "-ref.jsonl")
    ref = C.read(ref_file)
    if len(ref) != len(rows):
        if resume:
            raise SystemExit(f"{ref_file} is incomplete but {ckpt} exists: the reference must be "
                             f"the starting adapter, so remove {ckpt} and start over")
        model.eval()
        ref, started = [], time.time()
        with torch.no_grad():
            for i, r in enumerate(rows):
                ref.append({"chosen": token_logps(r["prompt_ids"], r["chosen_ids"]).sum().item(),
                            "rejected": token_logps(r["prompt_ids"], r["rejected_ids"]).sum().item()})
                if (i + 1) % 500 == 0:
                    print(f"[dpo] reference {i + 1}/{len(rows)} "
                          f"{(time.time() - started) / (i + 1):.2f}s/pair", flush=True)
        C.write(ref_file, ref)
    for r, f in zip(rows, ref):
        r["ref"] = f
    rows, held = rows[:-HELD_OUT], rows[-HELD_OUT:]
    smoke_qs = smoke_questions()
    smoke_file = out.with_name(out.name + "-smoke.jsonl")
    smoke_log = C.read(smoke_file)

    def smoke(step, answer):
        """Score the held-out pairs and, with answer, answer the smoke questions."""
        tick = time.time()
        model.eval()
        m, per, pair = {"step": step}, defaultdict(list), defaultdict(list)
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(SEED)
            for source, cap in SMOKE_SOURCES.items() if answer else ():
                qs = [q for q in smoke_qs if q["source"] == source]
                prompts = [tok.prompt(q["prompt"]) for q in qs]
                width, pad = max(map(len, prompts)), tok.t.pad_token_id
                ids = torch.tensor([[pad] * (width - len(x)) + x for x in prompts], device="cuda")
                mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in prompts],
                                    device="cuda")
                gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=cap,
                                     do_sample=True, **SAMPLING, use_cache=True,
                                     eos_token_id=[tok.eos, eot], pad_token_id=pad)
                for q, g in zip(qs, gen[:, width:].tolist()):
                    stop = next((i for i, t in enumerate(g) if t in (tok.eos, eot)), None)
                    g = g if stop is None else g[:stop + 1]
                    per[source].append({"correct": C.grade(tok, g, q["answer"], q["task"])[0],
                                        "think": answer_start(g),
                                        "eot": stop is not None and g[-1] == eot})
            for r in held:
                chosen_ids, k = r["chosen_ids"], answer_start(r["chosen_ids"])
                probe = k + 1 if k + 1 < len(chosen_ids) else None
                got = token_logps(r["prompt_ids"], chosen_ids, probe)
                lps, probs = got if probe is not None else (got, None)
                rejected = token_logps(r["prompt_ids"], r["rejected_ids"]).sum().item()
                margin = a.beta * ((lps.sum().item() - r["ref"]["chosen"])
                                   - (rejected - r["ref"]["rejected"]))
                pair["accuracy"].append(float(margin > 0))
                pair["margin"].append(margin)
                if probs is not None:  # what follows </think>: "\n\n", not the end of text
                    pair["p_after_close"].append(probs[chosen_ids[probe]].item())
                    pair["p_eot_after_close"].append(probs[eot].item())
        model.train()
        m.update({f"heldout_{k}": statistics.mean(v) for k, v in pair.items()})
        line = ""
        if answer:
            everything = [x for xs in per.values() for x in xs]
            m.update({"accuracy": statistics.mean(x["correct"] for x in everything),
                      "think_tokens": statistics.mean(x["think"] for x in everything),
                      "eot_stop": statistics.mean(x["eot"] for x in everything)})
            line = (f"acc {m['accuracy']:.1%} think {m['think_tokens']:.0f} "
                    f"eot-stop {m['eot_stop']:.0%} | ")
        smoke_log.append(m)
        C.write(smoke_file, smoke_log)
        print(f"[smoke] step {step} {line}held-out acc {m['heldout_accuracy']:.2f} "
              f"margin {m['heldout_margin']:+.2f} P(after </think>) {m['heldout_p_after_close']:.4f} "
              f"P(eot) {m['heldout_p_eot_after_close']:.4f} ({(time.time() - tick) / 60:.1f} min)",
              flush=True)
        trips = [why for bad, why in [
            (m["heldout_p_after_close"] < 0.9, "P(after </think>) < 0.9"),
            (m["heldout_p_eot_after_close"] > 0.01, "P(<|endoftext|> after </think>) > 0.01"),
            (m.get("eot_stop", 0) > 0.05, "stops on <|endoftext|> > 5%")] if bad]
        if trips:
            print(f"[smoke] TRIPWIRE step {step}: " + "; ".join(trips), flush=True)
        return time.time() - tick

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(rows) / a.batch)
    sched = cosine(opt, steps, max(1, steps // 10))
    print(f"[dpo] {len(rows)} pairs (+{len(held)} held out), {steps} steps, beta {a.beta}, "
          f"nll {a.nll}, lr {a.lr}", flush=True)
    rng, started = random.Random(0), time.time()
    step, log = load_checkpoint(ckpt, opt, sched)
    done, started_at, smoke_seconds = step, step, 0.0
    smoke_log = [m for m in smoke_log if m["step"] <= step]  # a killed run's later checks redo
    if not smoke_log and not a.no_smoke:
        smoke_seconds += smoke(0, True)
    model.train()
    for epoch in range(a.epochs):
        order = rows[:]
        rng.shuffle(order)  # the same order on a resume; the batches already done are skipped
        for start in range(0, len(order), a.batch):
            if done > 0:
                done -= 1
                continue
            batch = order[start:start + a.batch]
            total, stats = 0.0, defaultdict(list)
            for r in batch:
                lps = token_logps(r["prompt_ids"], r["chosen_ids"])
                chosen = lps.sum() - r["ref"]["chosen"]
                rejected = token_logps(r["prompt_ids"], r["rejected_ids"]).sum() - r["ref"]["rejected"]
                margin = a.beta * (chosen - rejected)
                loss = (-F.logsigmoid(margin) + a.nll * -lps.mean()) / len(batch)
                loss.backward()
                total += loss.item()
                stats["accuracy"].append(float(margin.item() > 0))
                stats["margin"].append(margin.item())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            means = {k: statistics.mean(v) for k, v in stats.items()}
            log.append({"step": step, "epoch": epoch, "loss": total, "lr": sched.get_last_lr()[0],
                        **means})
            if step % 10 == 0 or step == steps:
                print(f"[dpo] step {step}/{steps} loss {total:.4f} acc {means['accuracy']:.2f} "
                      f"margin {means['margin']:+.3f} "
                      f"{(time.time() - started - smoke_seconds) / (step - started_at):.1f}s/step",
                      flush=True)
            if not a.no_smoke and (step % SMOKE_EVERY == 0 or step == steps):
                smoke_seconds += smoke(step, True)
            elif not a.no_smoke and step % PROBE_EVERY == 0:
                smoke_seconds += smoke(step, False)
            if step % SAVE_EVERY == 0 and step < steps:
                save_checkpoint(ckpt, model, opt, sched, step, log)
            if a.stop_at and step >= a.stop_at:
                break
        if a.stop_at and step >= a.stop_at:
            break
    model.save_pretrained(out)
    C.write(out / "train-log.jsonl", log)
    if step < steps:
        save_checkpoint(ckpt, model, opt, sched, step, log)
        print(f"[dpo] stopped at step {step}/{steps}; a later --stop-at continues from {ckpt}",
              flush=True)
    else:
        shutil.rmtree(ckpt, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="jsonl of {prompt_ids, chosen_ids, rejected_ids}")
    p.add_argument("--init", required=True, help="the adapter to start from, also the reference")
    p.add_argument("--out", required=True, help="where the adapter goes")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--nll", type=float, default=0.0, help="weight of the chosen side's mean NLL")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--stop-at", type=int, default=0, help="stop after this step")
    p.add_argument("--limit", type=int, default=0, help="use the first N pairs (a smoke test)")
    p.add_argument("--no-smoke", action="store_true", help="skip the held-out and smoke checks")
    main(p.parse_args())
