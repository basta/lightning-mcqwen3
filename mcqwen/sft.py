"""Step 2, SFT: a fresh LoRA on the distilled targets, loss on the completion only.

    python -m mcqwen.sft --data work/distill/sft.jsonl --out work/distill/adapter

Checkpoints go to <out>-ckpt every SAVE_EVERY steps; rerunning the command resumes there.
"""
import argparse
import math
import os
from pathlib import Path
import random
import shutil
import time

from mcqwen import common as C

SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 50))


def load_checkpoint(ckpt, opt, sched):
    """A saved optimizer and schedule -> (step, log so far); (0, []) without one."""
    import torch
    if not (ckpt / "state.pt").exists():
        return 0, []
    state = torch.load(ckpt / "state.pt", weights_only=False)
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    print(f"resuming at step {state['step']}", flush=True)
    return state["step"], state["log"]


def save_checkpoint(ckpt, model, opt, sched, step, log):
    """The adapter, optimizer and schedule. The state file is written last and replaced
    atomically: it marks a complete checkpoint."""
    import torch
    model.save_pretrained(ckpt)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "log": log},
               ckpt / "state.tmp")
    (ckpt / "state.tmp").replace(ckpt / "state.pt")


def cosine(opt, steps, warmup):
    import torch
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) *
                                             0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))


def main(a):
    # a ~4k-token example's logits need one 2.3 GiB block; without this, fragmentation can
    # leave that much free but not contiguous on a 12 GB card
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    out = Path(a.out)
    ckpt = out.with_name(out.name + "-ckpt")
    rows = C.read(a.data)[:a.limit or None]
    torch.manual_seed(0)  # the LoRA initialisation
    model = C.load_model(trainable=True)
    if (ckpt / "state.pt").exists():
        model = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                                 target_modules=C.LORA_TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(rows) / a.batch)
    sched = cosine(opt, steps, max(1, steps // 20))
    print(f"[sft] {len(rows)} examples, {steps} steps", flush=True)
    rng, started = random.Random(0), time.time()
    step, log = load_checkpoint(ckpt, opt, sched)
    done, started_at = step, step
    model.train()
    for epoch in range(a.epochs):
        order = rows[:]
        rng.shuffle(order)  # the same order on a resume; the batches already done are skipped
        for start in range(0, len(order), a.batch):
            if done > 0:
                done -= 1
                continue
            batch = order[start:start + a.batch]
            targets = sum(len(r["completion_ids"]) for r in batch)
            total = 0.0
            for r in batch:  # one sequence at a time; the loss is per target token of the batch
                ids = torch.tensor([r["prompt_ids"] + r["completion_ids"]], device="cuda")
                labels = ids.clone()
                labels[0, :len(r["prompt_ids"])] = -100
                loss = model(input_ids=ids, labels=labels).loss * len(r["completion_ids"]) / targets
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log.append({"step": step, "epoch": epoch, "loss": total, "lr": sched.get_last_lr()[0]})
            if step % 10 == 0 or step == steps:
                print(f"[sft] step {step}/{steps} loss {total:.4f} "
                      f"{(time.time() - started) / (step - started_at):.1f}s/step", flush=True)
            if step % SAVE_EVERY == 0 and step < steps:
                save_checkpoint(ckpt, model, opt, sched, step, log)
    model.save_pretrained(out)
    C.write(out / "train-log.jsonl", log)
    shutil.rmtree(ckpt, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="jsonl of {prompt_ids, completion_ids}")
    p.add_argument("--out", required=True, help="where the adapter goes")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="train on the first N examples (a smoke test)")
    main(p.parse_args())
