"""The recipe behind Lightning McQwen3, in the order it runs (scripts/reproduce.sh):

    distill      Chain-of-Draft traces, kept without the prompt as SFT targets
    sft          LoRA SFT on them
    oracle       on-policy traces, cut and re-answered; shorten / guard / hard pairs
    dpo          DPO from an adapter (oracle pairs, then correctness pairs with --nll 1)
    correctness  right-over-wrong maths pairs at matched length
    gate         the BFCL think gate, fitted on ToolACE
    merge        joins sharded sampling runs
"""
