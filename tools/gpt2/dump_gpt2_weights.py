"""
dump_gpt2_weights.py

Prints the first N values of every weight tensor in a GPT-2 checkpoint,
formatted to line up with the C-side `print_first_n_element` output so
the two can be diffed / eyeballed side by side.

Usage:
    python dump_gpt2_weights.py
    python dump_gpt2_weights.py --model gpt2 --n 8
"""

import argparse
from transformers import GPT2Model


def value_check(model):
	print(model.state_dict()["wte.weight"].flatten()[:8])
	print(model.state_dict()["h.0.ln_1.weight"][:8])
	print(model.state_dict()["h.3.attn.c_attn.weight"][:, :8][0])   # 注意 Q 是切片，要对应你的拆分逻辑
	print(model.state_dict()["h.5.mlp.c_fc.weight"].flatten()[:8])
	print(model.state_dict()["ln_f.weight"][:8])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="HF model name or local path")
    parser.add_argument("--n", type=int, default=8, help="number of leading values to print per tensor")
    args = parser.parse_args()

    model = GPT2Model.from_pretrained(args.model)
    state_dict = model.state_dict()

    # find the longest tensor name so the ':' columns line up,
    # matching the "%-Ns" width used on the C side
    name_width = max(len(name) for name in state_dict.keys())

    for name, tensor in state_dict.items():
        flat = tensor.flatten()
        n = min(args.n, flat.numel())
        values = flat[:n].tolist()

        formatted = ", ".join(f"{v:9.6f}" for v in values)
        print(f"{name:<{name_width}} : {formatted}")

    value_check(model)


if __name__ == "__main__":
    main()
