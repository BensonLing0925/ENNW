#!/usr/bin/env python3
"""
verify_distilbert.py — Three-way numerical verification for the DistilBERT C engine.

Compares per-layer [CLS] hidden states across:
  1. PyTorch  — HuggingFace DistilBertForSequenceClassification (reference)
  2. Python-sim — numpy re-implementation of the C engine's exact algorithm
  3. C engine — distilbert_infer.exe --weights ... --verify

All three use the same weights and the same fixed input token sequence.

Usage:
  python tools/verify_distilbert.py
  python tools/verify_distilbert.py --weights distilbert_sst2.bin  # skip re-export
"""

import argparse
import math
import os
import re
import struct
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import torch
    from transformers import DistilBertForSequenceClassification
except ImportError:
    sys.exit("Requires: pip install torch transformers numpy")

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(TOOLS_DIR)

# Fixed input: [CLS] hello world [SEP] <pad>...
TOKEN_IDS = [101, 7592, 2088, 102] + [0] * 12   # 16 tokens
SEQ_LEN   = 16

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def np_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                 eps: float = 1e-12) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def np_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def np_gelu_erf(x: np.ndarray) -> np.ndarray:
    """GELU via erf — matches C's: 0.5*x*(1 + erf(x * 0.70710...))"""
    verf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + verf(x * 0.7071067811865476))

# ------------------------------------------------------------------ #
# Step 1: PyTorch reference                                           #
# ------------------------------------------------------------------ #

def run_pytorch(hf_model) -> list:
    """Return list of numpy [8] arrays: CLS first 8 dims after embedding and each layer."""
    layer_outputs = {}

    def make_hook(key):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            layer_outputs[key] = t[0, 0].float().detach().numpy().copy()
        return hook

    hooks = []
    hooks.append(hf_model.distilbert.embeddings.register_forward_hook(make_hook(0)))
    for i, layer in enumerate(hf_model.distilbert.transformer.layer):
        hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    input_ids = torch.tensor([TOKEN_IDS])
    with torch.no_grad():
        hf_model(input_ids)

    for h in hooks:
        h.remove()

    return [layer_outputs[k][:8] for k in sorted(layer_outputs)]

# ------------------------------------------------------------------ #
# Step 2: Python simulation of C engine                              #
# ------------------------------------------------------------------ #

def run_python_sim(hf_model) -> list:
    """Replicate the C engine's exact computation in numpy."""
    db   = hf_model.distilbert
    emb  = db.embeddings

    def w(t): return t.detach().float().numpy()

    word_w  = w(emb.word_embeddings.weight)    # [vocab, 768]
    pos_w   = w(emb.position_embeddings.weight)  # [512, 768]
    ln_g    = w(emb.LayerNorm.weight)
    ln_b    = w(emb.LayerNorm.bias)

    # Embedding
    tokens  = np.array(TOKEN_IDS, dtype=np.int32)
    hidden  = word_w[tokens] + pos_w[:SEQ_LEN]            # [seq, 768]
    hidden  = np_layernorm(hidden, ln_g, ln_b)

    results = [hidden[0, :8].copy()]

    for layer in db.transformer.layer:
        attn = layer.attention
        ffn  = layer.ffn

        def w(t): return t.detach().float().numpy()

        # Weight matrices: transpose from [out, in] to [in, out] for C-style gemm
        q_w = w(attn.q_lin.weight).T   # [768, 768]
        q_b = w(attn.q_lin.bias)
        k_w = w(attn.k_lin.weight).T
        k_b = w(attn.k_lin.bias)
        v_w = w(attn.v_lin.weight).T
        v_b = w(attn.v_lin.bias)
        o_w = w(attn.out_lin.weight).T
        o_b = w(attn.out_lin.bias)
        sa_g = w(layer.sa_layer_norm.weight)
        sa_b = w(layer.sa_layer_norm.bias)

        up_w  = w(ffn.lin1.weight).T   # [768, 3072]
        up_b  = w(ffn.lin1.bias)
        dn_w  = w(ffn.lin2.weight).T   # [3072, 768]
        dn_b  = w(ffn.lin2.bias)
        ol_g  = w(layer.output_layer_norm.weight)
        ol_b  = w(layer.output_layer_norm.bias)

        # QKV projections (pass-through quantize for f32: same pointer)
        Q = hidden @ q_w + q_b   # [seq, 768]
        K = hidden @ k_w + k_b
        V = hidden @ v_w + v_b

        # Multi-head attention (no mask, matching C)
        heads = 12
        hdim  = 768 // heads
        scale = hdim ** -0.5
        atten_out = np.zeros_like(hidden)
        for h in range(heads):
            sl = slice(h * hdim, (h + 1) * hdim)
            score = (Q[:, sl] @ K[:, sl].T) * scale   # [seq, seq]
            score = np_softmax(score)
            atten_out[:, sl] = score @ V[:, sl]

        # O projection + residual + LayerNorm  (post-norm)
        proj_out = atten_out @ o_w + o_b
        hidden   = np_layernorm(hidden + proj_out, sa_g, sa_b)

        # FFN + residual + LayerNorm
        inter  = np_gelu_erf(hidden @ up_w + up_b)
        ffn_out = inter @ dn_w + dn_b
        hidden  = np_layernorm(hidden + ffn_out, ol_g, ol_b)

        results.append(hidden[0, :8].copy())

    return results

# ------------------------------------------------------------------ #
# Step 3: C engine                                                    #
# ------------------------------------------------------------------ #

def export_weights(out_path: str) -> None:
    """Export distilbert-base-uncased-finetuned-sst-2-english weights."""
    sys.path.insert(0, TOOLS_DIR)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_dbert", os.path.join(TOOLS_DIR, "export_distilbert_weights.py"))
    mod = importlib.util.load_from_spec = spec  # noqa — we call via subprocess below
    # Export via subprocess to avoid namespace collisions
    subprocess.run(
        [sys.executable,
         os.path.join(TOOLS_DIR, "export_distilbert_weights.py"),
         out_path, "--sst2"],
        check=True, capture_output=True
    )

def run_c_engine(weights_path: str) -> list:
    exe = os.path.join(REPO_ROOT, "distilbert_infer.exe")
    if not os.path.exists(exe):
        sys.exit(f"Not found: {exe}  (run mingw32-make first)")
    result = subprocess.run(
        [exe, "--weights", weights_path, "--verify"],
        capture_output=True, text=True, timeout=120
    )
    output = result.stdout + result.stderr
    # Parse: "verify_layer N: f1 f2 f3 f4 f5 f6 f7 f8"
    layers = {}
    for m in re.finditer(r'verify_layer\s+(\d+):\s+([\d\s.e+\-]+)', output):
        idx    = int(m.group(1))
        vals   = list(map(float, m.group(2).split()))
        layers[idx] = np.array(vals[:8])
    if not layers:
        print("C engine output:")
        print(output)
        sys.exit("No verify_layer lines found in C output.")
    return [layers[k] for k in sorted(layers)]

# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def print_table(pt_vals, sim_vals, c_vals):
    header = f"{'Layer':<8} {'PT vs Sim':>14} {'PT vs C':>14} {'Sim vs C':>14}"
    print(header)
    print("-" * len(header))
    max_pt_c = 0.0
    for i, (pt, sim, c) in enumerate(zip(pt_vals, sim_vals, c_vals)):
        name  = "emb" if i == 0 else f"block {i}"
        d_ps  = np.max(np.abs(pt - sim))
        d_pc  = np.max(np.abs(pt - c))
        d_sc  = np.max(np.abs(sim - c))
        max_pt_c = max(max_pt_c, d_pc)
        flag  = "  OK" if d_pc < 1e-3 else "  !!"
        print(f"{name:<8}  {d_ps:14.2e}  {d_pc:14.2e}  {d_sc:14.2e}{flag}")
    print()
    return max_pt_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None,
                   help="Path to existing .bin file (skips re-export)")
    args = p.parse_args()

    print("Loading HuggingFace model (distilbert-base-uncased-finetuned-sst-2-english)...")
    hf_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english")
    hf_model.eval()
    print("  done.\n")

    # ---- PyTorch reference ----
    print("[1/3] PyTorch forward pass (with layer hooks)...")
    pt_vals = run_pytorch(hf_model)
    print(f"  {len(pt_vals)} layers captured.\n")

    # ---- Python-sim ----
    print("[2/3] Python-sim (numpy replication of C algorithm)...")
    sim_vals = run_python_sim(hf_model)
    print(f"  {len(sim_vals)} layers computed.\n")

    # ---- C engine ----
    if args.weights and os.path.exists(args.weights):
        weights_path = args.weights
        print(f"[3/3] Running C engine with --weights {weights_path} --verify...")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        weights_path = tmp.name
        print(f"[3/3] Exporting weights to {weights_path}...")
        export_weights(weights_path)
        print("  Exported. Running C engine --verify...")

    c_vals = run_c_engine(weights_path)
    print(f"  {len(c_vals)} layers captured.\n")

    if len(pt_vals) != len(sim_vals) or len(pt_vals) != len(c_vals):
        print(f"WARNING: layer count mismatch: PT={len(pt_vals)}, sim={len(sim_vals)}, C={len(c_vals)}")

    # ---- Comparison ----
    n = min(len(pt_vals), len(sim_vals), len(c_vals))
    print("=" * 60)
    print("VERIFICATION SUMMARY  (max abs diff, first 8 CLS dims)")
    print("=" * 60)
    max_gap = print_table(pt_vals[:n], sim_vals[:n], c_vals[:n])

    threshold = 1e-3
    if max_gap < threshold:
        print(f"PASS — PyTorch vs C max diff {max_gap:.2e} < {threshold}")
    else:
        print(f"FAIL — PyTorch vs C max diff {max_gap:.2e} >= {threshold}")
        # Show first diverging layer in detail
        for i, (pt, c) in enumerate(zip(pt_vals[:n], c_vals[:n])):
            if np.max(np.abs(pt - c)) >= threshold:
                name = "embedding" if i == 0 else f"block {i}"
                print(f"\nFirst divergence at {name}:")
                print(f"  PyTorch : {pt}")
                print(f"  C engine: {c}")
                print(f"  diff    : {pt - c}")
                break

    # ---- CLS values at final layer ----
    print(f"\nFinal layer [CLS] first 8 dims:")
    print(f"  PyTorch : {pt_vals[n-1]}")
    print(f"  Py-sim  : {sim_vals[n-1]}")
    print(f"  C engine: {c_vals[n-1]}")


if __name__ == "__main__":
    main()
