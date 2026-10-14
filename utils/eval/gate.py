"""Think gate: decide per prompt, before generating, whether to skip thinking.

A gate file (python -m mcqwen.gate fit) holds a linear probe on the model's hidden state
at the end of the prompt plus "<think>\\n", a threshold, and a scope string. A prompt is gated
when one of its messages contains the scope string (tool-call prompts) and the probe's score
is at or above the threshold. Gated prompts get an empty thinking block forced as the start of
the reply; those forced tokens count as generated output.

The probe runs in a child process: a CUDA context left in the parent would take memory the
vLLM engine sizes itself against.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def decide(gate_path, questions, arguments, tokenizer):
    """-> {question id: forced token ids} for the questions the gate skips thinking on."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "questions.jsonl", Path(tmp) / "fired.json"
        src.write_text("".join(json.dumps({"id": q["id"], "messages": q["messages"]}) + "\n"
                               for q in questions))
        spec = {"gate": str(Path(gate_path).resolve()), "model": arguments.model,
                "revision": arguments.revision,
                "adapter": str(Path(arguments.adapter).resolve()) if arguments.adapter else None,
                "questions": str(src), "out": str(dst)}
        subprocess.run([sys.executable, "-m", "utils.eval.gate", json.dumps(spec)], check=True,
                       cwd=Path(__file__).resolve().parents[2])
        return json.loads(dst.read_text())


def _run(spec):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gate = torch.load(spec["gate"])
    questions = [json.loads(line) for line in open(spec["questions"])]
    scoped = [q for q in questions if any(gate["scope"] in m["content"] for m in q["messages"])]
    fired = {}
    if scoped:
        tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
        model = AutoModelForCausalLM.from_pretrained(spec["model"], revision=spec["revision"],
                                                     dtype=torch.bfloat16).cuda().eval()
        if spec["adapter"]:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, spec["adapter"]).eval()
        decide_ids = tokenizer.encode(gate["decide"], add_special_tokens=False)
        force = tokenizer.encode(gate["force"], add_special_tokens=False)
        with torch.no_grad():
            for q in scoped:
                ids = tokenizer.apply_chat_template(q["messages"], tokenize=True,
                                                    add_generation_prompt=True, enable_thinking=True)
                out = model(torch.tensor([ids + decide_ids]).cuda(), output_hidden_states=True)
                h = out.hidden_states[gate["layer"]][0, -1].float().cpu()
                score = float(((h - gate["mu"]) / gate["sd"]) @ gate["w"] + gate["b"])
                if score >= gate["threshold"]:
                    fired[q["id"]] = force
    Path(spec["out"]).write_text(json.dumps(fired))
    print(f"gate: {len(fired)} of {len(scoped)} scoped prompts skip thinking "
          f"({len(questions)} questions in all)", flush=True)


if __name__ == "__main__":
    _run(json.loads(sys.argv[1]))
