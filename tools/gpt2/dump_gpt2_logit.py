import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

input_ids = torch.tensor([[464, 3797, 3332, 319, 262]])

block = model.transformer.h[0]
captured = {}

def hook_ln1(mod, inp, out):  captured['ln1'] = out
def hook_attn(mod, inp, out): captured['attn'] = out[0]   # attn 回传 tuple
def hook_ln2(mod, inp, out):  captured['ln2'] = out
def hook_mlp(mod, inp, out):  captured['mlp'] = out

block.ln_1.register_forward_hook(hook_ln1)
block.attn.register_forward_hook(hook_attn)
block.ln_2.register_forward_hook(hook_ln2)
block.mlp.register_forward_hook(hook_mlp)

w = model.transformer.h[0].attn.c_attn.weight   # [768, 2304]
print("Q first row [0:8]:", w[0, 0:8].tolist())
print("K first row [0:8]:", w[0, 768:776].tolist())
print("V first row [0:8]:", w[0, 1536:1544].tolist())

b = model.transformer.h[0].attn.c_attn.bias     # [2304]
print("q_bias[0:8]:", b[0:8].tolist())
print("k_bias[0:8]:", b[768:776].tolist())
print("v_bias[0:8]:", b[1536:1544].tolist())

with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)

for k in ['ln1', 'attn', 'ln2', 'mlp']:
    print(f"{k:5s}: {captured[k][0, -1, :8].tolist()}")

hs = out.hidden_states
for i, h in enumerate(hs):
    label = "embedding" if i == 0 else f"after layer {i-1}"
    print(f"{label:20s}: {h[0, -1, :8].tolist()}")
