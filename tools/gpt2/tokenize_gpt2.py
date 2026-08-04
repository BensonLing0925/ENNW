import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
"""
tok = GPT2Tokenizer.from_pretrained("gpt2")

ids = [464, 3797, 3332, 319, 262, 4314]
with torch.no_grad():
    out = model(torch.tensor([ids]), output_hidden_states=True)
hs = out.hidden_states
logits = out.logits[0, -1]
print("next token:", logits.argmax().item())
print("logits[0:4]:", logits[:4].tolist())

wte = model.transformer.wte.weight
wpe = model.transformer.wpe.weight
emb = wte[4314] + wpe[5]
print("embedding:", emb[:4].tolist())

print("after layer 0:", hs[1][0, -1, :4].tolist())
print("after layer 1:", hs[2][0, -1, :4].tolist())
"""

ids = [464, 3797, 3332, 319, 262, 4314]
block = model.transformer.h[0]
captured = {}
def hook_attn(mod, inp, out): captured['attn'] = out[0]
h = block.attn.register_forward_hook(hook_attn)

with torch.no_grad():
    model(torch.tensor([ids]))
h.remove()
print("attn out:", captured['attn'][0, -1, :4].tolist())
