import torch
import pprint
from transformers import GPT2Model, GPT2Tokenizer

model = GPT2Model.from_pretrained('gpt2')
for name, params in model.state_dict().items():
	print(name, tuple(params.shape))

# some huggingface model use nn.Linear, which results in matrix is transposed (out_feature, in_feature)
# since this framework does not transpose any weight at all, we need to make sure the weight matrices
# are also not transposed
print(type(model.h[0].attn.c_attn))
print(type(model.h[0].attn.c_proj))
print(type(model.h[0].mlp.c_fc))

# the output is
"""
<class 'transformers.pytorch_utils.Conv1D'>
<class 'transformers.pytorch_utils.Conv1D'>
<class 'transformers.pytorch_utils.Conv1D'>
"""

# meaning the weight are not transposed (Conv1D is not actual convolution, it is just a legacy name)

# read safetensors format
from safetensors.torch import load_file, safe_open
tensors = {}
with safe_open("../weight_files/gpt2_model.safetensors", framework="pt", device=0) as f:
	pprint.pprint(f.keys())
"""
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
print(tensors)
"""
