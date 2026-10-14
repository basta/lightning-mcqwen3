"""Join the shard directories of a sharded step into the directory the next step reads.

    python -m mcqwen.merge work/onpolicy work/onpolicy/shard-0 ... work/onpolicy/shard-7

Every key holds the question id, so shards never share one: joining is concatenation, in the
order given. Files missing from the first shard are skipped (distill has no oracle-gen.jsonl).
"""
import json
from pathlib import Path
import sys

FILES = ["questions.jsonl", "sample-gen.jsonl", "traces.jsonl", "oracle-gen.jsonl"]

root, shards = Path(sys.argv[1]), [Path(s) for s in sys.argv[2:]]
root.mkdir(parents=True, exist_ok=True)
assert shards, "no shard directories given"
for name in FILES:
    if not (shards[0] / name).exists():
        continue
    keys = set()
    with (root / name).open("w") as out:
        for shard in shards:
            with (shard / name).open() as f:
                for line in f:
                    key = json.loads(line)["qid" if name == "questions.jsonl" else "key"]
                    assert key not in keys, f"{key} in two shards"
                    keys.add(key)
                    out.write(line)
    print(f"{name}: {len(keys)} rows from {len(shards)} shards")
