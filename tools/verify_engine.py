#!/usr/bin/env python3
"""
verify_engine.py — Verify the C inference engine against PyTorch.

Workflow:
  1. Load N samples from MNIST (same binary files the C engine reads).
  2. Build a CNNVIT model with a fixed seed.
  3. Run a full PyTorch forward pass → record per-sample predictions & logits.
  4. Export the weights to ENNW format.
  5. Write a mini MNIST file with just the N samples (IDX format).
  6. Run the C engine via subprocess → parse per-sample predictions.
  7. Compare predictions sample-by-sample and print a detailed report.

Usage:
  python tools/verify_engine.py [--n 100] [--seed 0]
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("Requires: pip install torch numpy")

# ---- import the model and export function from export_weights.py ----
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
from export_weights import CNNVIT, export   # noqa: E402


# ================================================================
# IDX file helpers (MNIST binary format)
# ================================================================

def read_mnist_images(path: str):
    """Return numpy array [N, H, W] uint8."""
    with open(path, 'rb') as f:
        magic, n, h, w = struct.unpack('>IIII', f.read(16))
    assert magic == 0x00000803, f"Bad image magic: {magic:#x}"
    data = np.frombuffer(open(path, 'rb').read()[16:], dtype=np.uint8)
    return data.reshape(n, h, w)


def read_mnist_labels(path: str):
    """Return numpy array [N] uint8."""
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
    assert magic == 0x00000801, f"Bad label magic: {magic:#x}"
    return np.frombuffer(open(path, 'rb').read()[8:], dtype=np.uint8)


def write_mini_mnist_images(imgs: np.ndarray, path: str) -> None:
    """Write N×H×W uint8 array as IDX3 image file."""
    n, h, w = imgs.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('>IIII', 0x00000803, n, h, w))
        f.write(imgs.tobytes())


def write_mini_mnist_labels(labels: np.ndarray, path: str) -> None:
    """Write N uint8 labels as IDX1 label file."""
    n = labels.shape[0]
    with open(path, 'wb') as f:
        f.write(struct.pack('>II', 0x00000801, n))
        f.write(labels.tobytes())


# ================================================================
# Pure-Python simulation of the C forward pass
# (to compare logits, not just final predictions)
# ================================================================

def py_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                 eps: float = 1e-5) -> np.ndarray:
    """LayerNorm along last axis, matching C tk_ops_layernorm."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def py_softmax_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def py_gelu(x: np.ndarray) -> np.ndarray:
    """GELU matching PyTorch default (erf-based)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def c_forward_single(img_u8: np.ndarray, model: CNNVIT) -> np.ndarray:
    """
    Simulate the C engine's forward pass for a single H×W uint8 image.
    Returns logits [num_classes] as float64 numpy array.
    Uses the model weights directly (numpy).
    """
    # ---- weights to numpy (float64) ----
    conv_w = model.conv.weight.detach().cpu().double().numpy()   # [F, 1, kH, kW]
    tf = model.tf_block
    attn = tf.attn
    Wq = attn.q_weight.detach().cpu().double().numpy()           # [H, H]
    Wk = attn.k_weight.detach().cpu().double().numpy()
    Wv = attn.v_weight.detach().cpu().double().numpy()
    Wup = tf.ffn_up.weight.detach().cpu().double().numpy().T     # [hidden, inter]
    Wdn = tf.ffn_dn.weight.detach().cpu().double().numpy().T     # [inter, hidden]
    ln1_g = tf.ln1.weight.detach().cpu().double().numpy()
    ln1_b = tf.ln1.bias.detach().cpu().double().numpy()
    ln2_g = tf.ln2.weight.detach().cpu().double().numpy()
    ln2_b = tf.ln2.bias.detach().cpu().double().numpy()
    fc_ws = [l.weight.detach().cpu().double().numpy().T for l in model.fc]  # each [in, out]
    fc_bs = [l.bias.detach().cpu().double().numpy()    for l in model.fc]

    num_filter  = model.seq
    kernel_size = conv_w.shape[2]
    pool_size   = img_u8.shape[0] // (img_u8.shape[0] - kernel_size + 1) + 1
    # Infer pool_size from model
    hidden_dim  = model.hidden_dim
    pooled_h    = int(math.isqrt(hidden_dim))

    H, W = img_u8.shape

    # ---- 1. Normalize ----
    x = img_u8.astype(np.float64) / 255.0   # [H, W]

    # ---- 2. Conv2D: manual correlation ----
    fH = H - kernel_size + 1
    fW = W - kernel_size + 1
    filtered = np.zeros((num_filter, fH, fW), dtype=np.float64)
    for f in range(num_filter):
        kern = conv_w[f, 0]   # [kH, kW]
        for i in range(fH):
            for j in range(fW):
                filtered[f, i, j] = (x[i:i+kernel_size, j:j+kernel_size] * kern).sum()

    # ---- 3. MaxPool2D ----
    ps = pool_size
    pH = fH // ps
    pW = fW // ps
    pooled = np.zeros((num_filter, pH, pW), dtype=np.float64)
    for f in range(num_filter):
        for i in range(pH):
            for j in range(pW):
                pooled[f, i, j] = filtered[f, i*ps:(i+1)*ps, j*ps:(j+1)*ps].max()

    # ---- 4. ReLU ----
    pooled = np.maximum(pooled, 0.0)

    # ---- 5. Reshape to [seq, hidden] ----
    x_tf = pooled.reshape(num_filter, hidden_dim)   # [seq, hidden]

    # ---- 6. Transformer (pre-norm, per-head) ----
    heads  = attn.n_heads
    hdim   = attn.head_dim
    scale  = hdim ** -0.5

    # Attention sub-layer
    ln1_out = py_layernorm(x_tf, ln1_g, ln1_b)
    Q = ln1_out @ Wq   # [seq, hidden]
    K = ln1_out @ Wk
    V = ln1_out @ Wv

    attn_out = np.zeros_like(x_tf)
    for h in range(heads):
        sl = slice(h * hdim, (h + 1) * hdim)
        Qh = Q[:, sl]   # [seq, hdim]
        Kh = K[:, sl]
        Vh = V[:, sl]
        score = Qh @ Kh.T * scale          # [seq, seq]
        score = py_softmax_rows(score)
        oh    = score @ Vh                  # [seq, hdim]
        attn_out[:, sl] = oh

    x_tf = x_tf + attn_out

    # FFN sub-layer
    ln2_out = py_layernorm(x_tf, ln2_g, ln2_b)
    ffn_mid = py_gelu(ln2_out @ Wup)       # [seq, inter]
    ffn_out = ffn_mid @ Wdn                 # [seq, hidden]
    x_tf = x_tf + ffn_out

    # ---- 7. Flatten ----
    flat = x_tf.reshape(-1)                 # [seq * hidden]

    # ---- 8. FC chain ----
    x_fc = flat
    for i, (W_fc, b_fc) in enumerate(zip(fc_ws, fc_bs)):
        x_fc = x_fc @ W_fc + b_fc
        if i < len(fc_ws) - 1:
            x_fc = np.maximum(x_fc, 0.0)   # ReLU

    return x_fc   # logits [num_classes]


# ================================================================
# C engine runner
# ================================================================

def run_c_engine(nn_exe: str, config_path: str) -> str:
    result = subprocess.run(
        [nn_exe, config_path],
        capture_output=True, text=True, timeout=300
    )
    return result.stdout + result.stderr


def parse_c_result(output: str):
    """Parse 'correct : X / Y  (Z%)' and per-sample predictions from C engine output."""
    import re
    m = re.search(r'correct\s*:\s*(\d+)\s*/\s*(\d+)', output)
    correct, total = (int(m.group(1)), int(m.group(2))) if m else (None, None)

    # Parse "sample_pred N PRED" lines
    preds = {}
    for hit in re.finditer(r'sample_pred\s+(\d+)\s+(\d+)', output):
        preds[int(hit.group(1))] = int(hit.group(2))
    c_preds = [preds[i] for i in range(len(preds))] if preds else None

    return correct, total, c_preds


# ================================================================
# Main verification
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n',    type=int, default=200,
                   help='Number of MNIST samples to verify (default 200)')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed for model init (default 0)')
    p.add_argument('--full', action='store_true',
                   help='Run on full MNIST test set (10000 samples, slower)')
    args = p.parse_args()

    # ---- Paths ----
    mnist_imgs   = os.path.join(REPO_ROOT, 'src', 'MNIST',
                                't10k-images-idx3-ubyte')
    mnist_labels = os.path.join(REPO_ROOT, 'src', 'MNIST',
                                't10k-labels-idx1-ubyte')
    nn_exe       = os.path.join(REPO_ROOT, 'nn.exe')
    weights_path = os.path.join(REPO_ROOT, 'weight_files', 'verify_weights.bin')
    cfg_path     = os.path.join(REPO_ROOT, 'weight_files', 'verify_config.json')

    for path in [mnist_imgs, mnist_labels, nn_exe]:
        if not os.path.exists(path):
            sys.exit(f"Not found: {path}")

    os.makedirs(os.path.join(REPO_ROOT, 'weight_files'), exist_ok=True)

    # ---- Architecture ----
    num_filter  = 10
    kernel_size = 3
    pool_size   = 2
    tf_n_heads  = 13
    fc_layers   = [100, 50, 10]
    img_size    = 28

    cfg_dict = {
        'num_filter':  num_filter,
        'kernel_size': kernel_size,
        'pool_size':   pool_size,
        'tf_n_heads':  tf_n_heads,
        'fc_layers':   fc_layers,
        'img_size':    img_size,
    }

    # ---- Build model ----
    torch.manual_seed(args.seed)
    model = CNNVIT(num_filter, kernel_size, pool_size, tf_n_heads, fc_layers, img_size)
    model.eval()

    # ---- Load MNIST ----
    print(f"Loading MNIST test data...")
    all_imgs   = read_mnist_images(mnist_imgs)     # [10000, 28, 28]
    all_labels = read_mnist_labels(mnist_labels)   # [10000]

    N = len(all_imgs) if args.full else min(args.n, len(all_imgs))
    imgs   = all_imgs[:N]
    labels = all_labels[:N]
    print(f"Using {N} samples.")

    # ================================================================
    # Step 1: Pure-Python simulation (matches C's exact computation)
    # ================================================================
    print(f"\n[1/3] Running pure-Python C-simulation forward pass on {N} samples...")
    py_preds = []
    py_logits_list = []
    for i, img in enumerate(imgs):
        if i % 50 == 0:
            print(f"  sample {i}/{N}...", end='\r')
        logits = c_forward_single(img, model)
        py_preds.append(int(np.argmax(logits)))
        py_logits_list.append(logits)
    print(f"  Done.                    ")
    py_correct = sum(p == int(l) for p, l in zip(py_preds, labels))
    print(f"  Python-sim accuracy: {py_correct}/{N} ({100*py_correct/N:.2f}%)")

    # ================================================================
    # Step 2: Export weights + run C engine
    # ================================================================
    print(f"\n[2/3] Exporting weights to {weights_path}...")
    export(model, cfg_dict, weights_path)

    # Write mini MNIST files
    with tempfile.TemporaryDirectory() as tmp:
        mini_img_path = os.path.join(tmp, 'imgs.idx')
        mini_lbl_path = os.path.join(tmp, 'lbls.idx')
        write_mini_mnist_images(imgs,   mini_img_path)
        write_mini_mnist_labels(labels, mini_lbl_path)

        # Write config
        infer_cfg = {
            'mode':         'TEST',
            'seed':         args.seed,
            'imgPath':      mini_img_path.replace('/', '\\'),
            'imgLabelPath': mini_lbl_path.replace('/', '\\'),
            'weights_path': weights_path.replace('/', '\\'),
            'num_filter':   num_filter,
            'kernel_size':  kernel_size,
            'pool_size':    pool_size,
            'tf_n_heads':   tf_n_heads,
            'fc_layers':    fc_layers,
        }
        with open(cfg_path, 'w') as f:
            json.dump(infer_cfg, f, indent=4)

        print(f"\n[3/3] Running C engine: {nn_exe} {cfg_path}")
        output = run_c_engine(nn_exe, cfg_path)

    c_correct, c_total, c_preds = parse_c_result(output)
    if c_correct is None:
        print("Could not parse C engine output:")
        print(output)
        sys.exit(1)
    print(f"  C engine accuracy   : {c_correct}/{c_total} ({100*c_correct/c_total:.2f}%)")

    # ================================================================
    # Step 3: Compare
    # ================================================================
    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY  ({N} samples)")
    print(f"{'='*60}")
    print(f"  Python-sim  : {py_correct:5d} / {N}  ({100*py_correct/N:.2f}%)")
    print(f"  C engine    : {c_correct:5d} / {N}  ({100*c_correct/c_total:.2f}%)")

    if c_preds and len(c_preds) == N:
        per_sample_diff = [(i, py_preds[i], c_preds[i])
                           for i in range(N) if py_preds[i] != c_preds[i]]
        if not per_sample_diff:
            print(f"\n  PASS — all {N} per-sample predictions agree.")
        else:
            print(f"\n  MISMATCH — {len(per_sample_diff)} sample(s) differ (per-sample):")
            for idx, py_p, c_p in per_sample_diff[:15]:
                true = int(labels[idx])
                print(f"    sample {idx:4d}: true={true}  Python-sim={py_p}  C={c_p}")
            if len(per_sample_diff) > 15:
                print(f"    ... and {len(per_sample_diff) - 15} more")
    elif py_correct == c_correct:
        print(f"\n  PASS — prediction counts match exactly.")
    else:
        diff = abs(py_correct - c_correct)
        print(f"\n  MISMATCH — count differs by {diff} (no per-sample data).")

    # ---- Bonus: PyTorch forward pass for reference ----
    print(f"\n--- PyTorch model reference ---")
    X = torch.tensor(imgs[:, np.newaxis].astype(np.float32) / 255.0)
    with torch.no_grad():
        logits_pt = model(X)   # [N, 10]
    pt_preds = logits_pt.argmax(dim=1).numpy()
    pt_correct = int((pt_preds == labels).sum())
    print(f"  PyTorch     : {pt_correct:5d} / {N}  ({100*pt_correct/N:.2f}%)")

    # Compare PyTorch vs Python-sim per-sample
    mismatch = [(i, int(pt_preds[i]), py_preds[i])
                for i in range(N) if int(pt_preds[i]) != py_preds[i]]
    if mismatch:
        print(f"\n  PyTorch vs Python-sim mismatches ({len(mismatch)} samples):")
        for idx, pt_p, py_p in mismatch[:10]:
            print(f"    sample {idx:4d}: PyTorch={pt_p}, Python-sim={py_p}, true={labels[idx]}")
        if len(mismatch) > 10:
            print(f"    ... and {len(mismatch)-10} more")
    else:
        print(f"  PyTorch vs Python-sim: all {N} predictions agree.")

    # ---- Show first 5 sample logits (Python-sim) ----
    print(f"\n--- Sample logits (Python-sim, first 5 samples) ---")
    for i in range(min(5, N)):
        logits = py_logits_list[i]
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        pred   = int(np.argmax(probs))
        true   = int(labels[i])
        mark   = 'OK' if pred == true else 'XX'
        print(f"  [{i}] true={true} pred={pred} {mark}  "
              f"probs=[{', '.join(f'{v:.3f}' for v in probs)}]")


if __name__ == '__main__':
    main()
