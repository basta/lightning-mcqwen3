"""Shared pieces: the tokenizer, the vLLM engine, resumable generation, grading, data files.

Generation steps append to a jsonl "sink" chunk by chunk and skip keys already there, so a
killed job continues where it stopped. Every prompt is seeded from its key, so a question gets
the same samples whichever shard or chunk it lands in.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import zlib

import evaluate
from utils.eval import graders
from utils.eval.recipe import DEFAULT_MAX_TOKENS, SAMPLING, SEED
from utils.train.rewards import is_task_correct

REPO = Path(evaluate.__file__).resolve().parent
CAP = DEFAULT_MAX_TOKENS  # generation cap, as the evaluation's
CHUNK = int(os.environ.get("CHUNK", 500))  # prompts per generate-and-save round
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------- files

def iread(path):
    """jsonl -> rows, one at a time, tolerating a half-written last line from a crash."""
    path = Path(path)
    if not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def read(path):
    return list(iread(path))


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def seed_of(text):
    return SEED + zlib.crc32(text.encode()) % 1_000_000


def in_shard(qid, shard):
    """shard "i/N": the questions whose id hashes to i, so every shard gets the same mix."""
    if not shard:
        return True
    i, n = map(int, shard.split("/"))
    return zlib.crc32(qid.encode()) % n == i


# ---------------------------------------------------------------- model

class Tok:
    def __init__(self):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(evaluate.BASE_MODEL, revision=evaluate.BASE_REVISION)
        self.eos = self.t.eos_token_id
        self.open, self.close = self.t.convert_tokens_to_ids(["<think>", "</think>"])

    def prompt(self, messages):
        return self.t.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=True)

    def encode(self, text):
        return self.t.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.t.decode(ids, skip_special_tokens=True)


def engine(adapter=None):
    """vLLM with the evaluation's model and context length, optionally serving one LoRA."""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # as evaluate.py: no nvcc build
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    rank = json.loads((Path(adapter) / "adapter_config.json").read_text())["r"] if adapter else None
    llm = LLM(model=evaluate.BASE_MODEL, revision=evaluate.BASE_REVISION,
              tokenizer_revision=evaluate.BASE_REVISION,
              max_model_len=CAP + evaluate.PROMPT_BUDGET,
              gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.75)),
              enable_prefix_caching=True, generation_config="vllm", seed=SEED,
              enable_lora=rank is not None, max_lora_rank=rank or 16,
              **({"max_num_seqs": int(os.environ["MAX_NUM_SEQS"])}
                 if os.environ.get("MAX_NUM_SEQS") else {}))
    lora = LoRARequest("candidate", 1, str(Path(adapter).resolve())) if adapter else None
    return llm, lora


def load_model(adapter=None, trainable=False):
    """Qwen3-0.6B in bf16 on the GPU, with a LoRA adapter if given, set up for training."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        evaluate.BASE_MODEL, revision=evaluate.BASE_REVISION, dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda()
    if trainable:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return model


def open_sink(sink):
    """Append to a jsonl sink, first dropping a half-written last line left by a kill, which
    the next append would otherwise glue onto a valid row."""
    sink.parent.mkdir(parents=True, exist_ok=True)
    if sink.exists() and sink.stat().st_size:
        with sink.open("rb+") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.seek(0)
                data = stream.read()
                stream.truncate(data.rfind(b"\n") + 1)
    return sink.open("a")


def generate(llm, lora, tok, jobs, sink, n, tag, total):
    """jobs: iterable of {key, ids, seed, max_tokens}; total: how many it yields. n samples per
    job, at the evaluation's sampling settings, appended to sink as {key, outputs}."""
    from vllm import SamplingParams
    done = {row["key"] for row in iread(sink)}
    print(f"[{tag}] {total} prompts x {n}, {total - len(done)} to generate", flush=True)
    started, generated = time.time(), 0

    def run(chunk):
        nonlocal generated
        params = [SamplingParams(n=n, seed=j["seed"], max_tokens=j["max_tokens"],
                                 stop_token_ids=[tok.eos], **SAMPLING) for j in chunk]
        results = llm.generate([{"prompt_token_ids": j["ids"]} for j in chunk], params,
                               lora_request=lora, use_tqdm=False)
        for j, r in zip(chunk, results):
            stream.write(json.dumps({"key": j["key"],
                                     "outputs": [list(x.token_ids) for x in r.outputs]}) + "\n")
        stream.flush()
        generated += len(chunk)
        eta = (total - len(done) - generated) / (generated / (time.time() - started)) / 3600
        print(f"[{tag}] {len(done) + generated}/{total}  eta {eta:.1f} h", flush=True)

    with open_sink(sink) as stream:
        chunk = []
        for j in jobs:
            if j["key"] in done:
                continue
            chunk.append(j)
            if len(chunk) == CHUNK:
                run(chunk)
                chunk = []
        if chunk:
            run(chunk)


def outputs_of(sink):
    return {row["key"]: row["outputs"] for row in iread(sink)}


# ---------------------------------------------------------------- grading

def grade(tok, tokens, answer, task, prefix=False):
    """Generated tokens -> (correct, closed, truncated). prefix: the tokens are a re-answer after
    forced thinking, so they must not reopen or reclose it."""
    truncated = not (tokens and tokens[-1] == tok.eos)
    if prefix:
        closed = tok.open not in tokens and tok.close not in tokens
        text = "</think>\n\n" + tok.decode(tokens)
    else:
        closed = tok.close in tokens
        text = tok.decode(tokens)
    correct = closed and not truncated and is_task_correct(text, answer, task)
    return correct, closed, truncated


def eval_valid(tok, ids, task):
    """Would the evaluation's SVAMP parser accept this maths answer? (rewards.py is laxer.)"""
    if task != "math":
        return True
    predicted, error = graders.final_answer(tok.decode(ids).split("</think>")[-1])
    return not error and graders.to_number(predicted) is not None


# ---------------------------------------------------------------- questions

def load_questions(split, per_source, skip=0):
    """data/<split>.jsonl (utils/train/prepare_data.py): the first per_source questions of each
    source after skipping its first skip; per_source 0 for all of them."""
    rows, seen = [], defaultdict(int)
    for i, r in enumerate(read(REPO / "data" / f"{split}.jsonl")):
        n = seen[r["source"]]
        if per_source and n >= skip + per_source:
            continue
        seen[r["source"]] += 1
        if n >= skip:
            rows.append({**r, "qid": f"{split}-{i}"})
    return rows
