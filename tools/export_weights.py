#!/usr/bin/env python3
"""
export_weights.py — Export a PyTorch model to the ENNW binary format.

The PyTorch model architecture must match the C framework exactly:
  Conv2d(1, num_filter, kernel_size, stride=1, padding=0, bias=False)
  -> MaxPool2d(pool_size) -> ReLU
  -> CustomTransformer(seq=num_filter, hidden=pooled*pooled, n_heads)
  -> Linear chain (no output projection in attention, no output bias in FFN)

Usage:
  # Export random weights (for testing):
  python tools/export_weights.py --config config.json --out weights.bin

  # Export trained weights from a .pt state-dict file:
  python tools/export_weights.py --config config.json --pt model.pt --out weights.bin

  # Train a model and export:
  python tools/export_weights.py --config config.json --train --out weights.bin

Binary file layout:
  [Binary_Header]
  [Layer 0 meta: CONV2D]
  [Layer 1 meta: TF]
  [Layer 2 meta: FC]
  [CONV2D payload: filters float64 [F, 1, kH, kW]]
  [TF payload: Q, K, V, ffn_up, ffn_down, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta]
  [FC payload: for each linear — weights [in, out], bias [out]]
"""

import argparse
import json
import struct
import sys
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch not found. Install with: pip install torch")

# ---- Binary format constants ----
MAGIC        = b'ENNW'
VER          = 2
ENDIAN_LE    = 1
DTYPE_F64    = 2
MODEL_CNN    = 2
LAYER_FC     = 1
LAYER_CONV2D = 2
LAYER_TF     = 4


def _u32(v): return struct.pack('<I', int(v))
def _u64(v): return struct.pack('<Q', int(v))
def _f64s(arr): return arr.numpy().tobytes()

def tensor_f64(t: torch.Tensor) -> bytes:
    return _f64s(t.detach().cpu().to(torch.float64).contiguous())


# ================================================================
# PyTorch model that exactly matches the C framework
# ================================================================

class _NoOutProjAttn(nn.Module):
    """
    Multi-head self-attention WITHOUT an output projection matrix,
    matching the C implementation exactly.
    """
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}"
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads
        self.scale      = self.head_dim ** -0.5
        self.q_weight   = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.k_weight   = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.v_weight   = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.normal_(self.q_weight, std=0.02)
        nn.init.normal_(self.k_weight, std=0.02)
        nn.init.normal_(self.v_weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq, hidden]
        seq    = x.shape[0]
        hdim   = self.head_dim
        heads  = self.n_heads
        hidden = self.hidden_dim

        q = x @ self.q_weight   # [seq, hidden]
        k = x @ self.k_weight
        v = x @ self.v_weight

        q = q.view(seq, heads, hdim).transpose(0, 1)  # [heads, seq, hdim]
        k = k.view(seq, heads, hdim).transpose(0, 1)
        v = v.view(seq, heads, hdim).transpose(0, 1)

        score = (q @ k.transpose(-2, -1)) * self.scale     # [heads, seq, seq]
        score = F.softmax(score, dim=-1)
        out   = score @ v                                   # [heads, seq, hdim]
        out   = out.transpose(0, 1).contiguous().view(seq, hidden)
        return out


class TFBlock(nn.Module):
    """Pre-norm transformer block matching the C tf_block_forward."""
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        inter_dim   = 4 * hidden_dim
        self.ln1    = nn.LayerNorm(hidden_dim)
        self.attn   = _NoOutProjAttn(hidden_dim, n_heads)
        self.ln2    = nn.LayerNorm(hidden_dim)
        self.ffn_up = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.ffn_dn = nn.Linear(inter_dim,  hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn_dn(F.gelu(self.ffn_up(self.ln2(x))))
        return x


class CNNVIT(nn.Module):
    """
    Full model matching the C CNN+Transformer+FC pipeline.
    img_size: square input image side length (default 28 for MNIST).
    """
    def __init__(self, num_filter: int, kernel_size: int, pool_size: int,
                 tf_n_heads: int, fc_layers: list, img_size: int = 28):
        super().__init__()
        self.conv  = nn.Conv2d(1, num_filter, kernel_size,
                               stride=1, padding=0, bias=False)
        self.pool  = nn.MaxPool2d(pool_size)

        fmap_h     = img_size - kernel_size + 1
        pooled_h   = fmap_h // pool_size
        hidden_dim = pooled_h * pooled_h
        seq        = num_filter

        self.seq        = seq
        self.hidden_dim = hidden_dim
        self.tf_block   = TFBlock(hidden_dim, tf_n_heads)

        self.fc = nn.ModuleList()
        in_dim  = seq * hidden_dim
        for out_dim in fc_layers:
            self.fc.append(nn.Linear(in_dim, out_dim, bias=True))
            in_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 1, H, W]
        x = F.relu(self.pool(self.conv(x)))      # [N, seq, ph, pw]
        N, F_, H, W = x.shape
        x = x.view(N, F_, H * W)                 # [N, seq, hidden]

        outs = []
        for n in range(N):
            s = self.tf_block(x[n])               # [seq, hidden]
            outs.append(s.view(-1))
        x = torch.stack(outs)                     # [N, seq*hidden]

        for i, layer in enumerate(self.fc):
            x = layer(x)
            if i < len(self.fc) - 1:
                x = F.relu(x)
        return x


# ================================================================
# Weight export
# ================================================================

def export(model: CNNVIT, cfg: dict, out_path: str) -> None:
    num_filter  = cfg['num_filter']
    kernel_size = cfg['kernel_size']
    pool_size   = cfg['pool_size']
    tf_n_heads  = cfg['tf_n_heads']
    fc_layers   = cfg['fc_layers']
    img_size    = cfg.get('img_size', 28)

    fmap_h     = img_size - kernel_size + 1
    pooled_h   = fmap_h // pool_size
    hidden_dim = pooled_h * pooled_h
    seq        = num_filter
    inter_dim  = 4 * hidden_dim
    input_size = seq * hidden_dim

    # param counts
    conv_n = num_filter * 1 * kernel_size * kernel_size
    tf_n   = (hidden_dim * hidden_dim * 3 +
              hidden_dim * inter_dim +
              inter_dim  * hidden_dim +
              hidden_dim * 4)
    fc_n   = sum(in_d * out_d + out_d
                 for in_d, out_d in zip([input_size] + fc_layers[:-1], fc_layers))

    out = bytearray()

    # ---- Header ----
    out += MAGIC
    out += _u32(VER)
    out += _u32(ENDIAN_LE)
    out += _u32(DTYPE_F64)
    out += _u32(MODEL_CNN)
    out += _u32(3)              # layer_count: CONV2D + TF + FC
    out += _u32(img_size)       # input_h
    out += _u32(img_size)       # input_w
    out += _u32(1)              # input_c
    out += _u32(0) * 8          # reserved[8]

    # ---- Layer 0 meta: CONV2D ----
    out += _u32(LAYER_CONV2D)
    out += _u32(0)
    out += _u64(conv_n)
    out += _u64(conv_n * 8)
    # Binary_Conv2D_Layer_Meta (16 fields × u32)
    for v in [num_filter, 1, kernel_size, kernel_size,
              1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
        out += _u32(v)

    # ---- Layer 1 meta: TF ----
    out += _u32(LAYER_TF)
    out += _u32(1)
    out += _u64(tf_n)
    out += _u64(tf_n * 8)
    # Binary_TF_Layer_Meta (8 fields × u32)
    for v in [seq, hidden_dim, tf_n_heads, inter_dim, 0, 0, 0, 0]:
        out += _u32(v)

    # ---- Layer 2 meta: FC ----
    out += _u32(LAYER_FC)
    out += _u32(2)
    out += _u64(fc_n)
    out += _u64(fc_n * 8)
    # Binary_Net_Layer_Meta
    for v in [1, len(fc_layers), input_size, 0, 0]:   # network_type=FC_CHAIN
        out += _u32(v)
    # Per-linear meta
    in_d = input_size
    for out_d in fc_layers:
        for v in [out_d, in_d, 1, 0]:   # num_neurons, input_dim, has_bias, reserved
            out += _u32(v)
        in_d = out_d

    # ---- Payloads ----

    # CONV2D: filters [F, 1, kH, kW]
    out += tensor_f64(model.conv.weight)

    # TF
    tf = model.tf_block
    out += tensor_f64(tf.attn.q_weight)          # [hidden, hidden]
    out += tensor_f64(tf.attn.k_weight)
    out += tensor_f64(tf.attn.v_weight)
    # ffn_up weight: PyTorch Linear is [out, in] = [inter, hidden]; C wants [hidden, inter]
    out += tensor_f64(tf.ffn_up.weight.T.contiguous())
    # ffn_dn weight: PyTorch is [hidden, inter]; C wants [inter, hidden]
    out += tensor_f64(tf.ffn_dn.weight.T.contiguous())
    out += tensor_f64(tf.ln1.weight)             # gamma [hidden]
    out += tensor_f64(tf.ln1.bias)               # beta  [hidden]
    out += tensor_f64(tf.ln2.weight)
    out += tensor_f64(tf.ln2.bias)

    # FC: for each Linear — weights [in, out], bias [out]
    for layer in model.fc:
        # PyTorch Linear weight is [out, in]; C wants [in, out]
        out += tensor_f64(layer.weight.T.contiguous())
        out += tensor_f64(layer.bias)

    with open(out_path, 'wb') as f:
        f.write(out)
    print(f"[export] Wrote {len(out)} bytes to '{out_path}'")
    print(f"  conv_params={conv_n}, tf_params={tf_n}, fc_params={fc_n}")


# ================================================================
# Optional: minimal MNIST training loop
# ================================================================

def train_mnist(model: CNNVIT, img_path: str, label_path: str,
                max_iter: int, lr: float, device: torch.device) -> None:
    """
    Train on MNIST-format raw binary files (same format as the C loader).
    img_path:   uint8 flat file, skip 16-byte header, then N×H×W pixels
    label_path: uint8 flat file, skip  8-byte header, then N labels
    """
    import numpy as np

    # Load images
    with open(img_path, 'rb') as f:
        f.read(16)   # skip IDX header
        pixels = np.frombuffer(f.read(), dtype=np.uint8)

    with open(label_path, 'rb') as f:
        f.read(8)
        labels_np = np.frombuffer(f.read(), dtype=np.uint8)

    N  = labels_np.shape[0]
    hw = pixels.shape[0] // N
    h  = w = int(math.isqrt(hw))
    assert h * w == hw, f"Non-square images not supported ({hw} pixels)"

    imgs = pixels.reshape(N, 1, h, w).astype(np.float32) / 255.0
    X = torch.from_numpy(imgs).to(device)
    Y = torch.from_numpy(labels_np.astype(np.int64)).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    for epoch in range(max_iter):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss   = loss_fn(logits, Y)
        loss.backward()
        optimizer.step()

        preds   = logits.argmax(dim=1)
        correct = (preds == Y).sum().item()
        print(f"[epoch {epoch+1:3d}] loss={loss.item():.4f}  "
              f"acc={correct}/{N} ({100*correct/N:.1f}%)")


# ================================================================
# CLI
# ================================================================

def load_cfg(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {
        'num_filter':  int(raw.get('num_filter',  10)),
        'kernel_size': int(raw.get('kernel_size', 3)),
        'pool_size':   int(raw.get('pool_size',   2)),
        'tf_n_heads':  int(raw.get('tf_n_heads',  13)),
        'fc_layers':   [int(x) for x in raw.get('fc_layers', [100, 50, 10])],
        'img_size':    int(raw.get('img_size',    28)),
        'lr':          float(raw.get('lr',        0.001)),
        'max_iter':    int(raw.get('max_iter',    10)),
        'imgPath':     raw.get('imgPath',     ''),
        'imgLabelPath':raw.get('imgLabelPath',''),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description='Export PyTorch weights to ENNW binary format')
    p.add_argument('--config', required=True, help='JSON config file')
    p.add_argument('--pt',     default=None,  help='Input .pt state-dict file')
    p.add_argument('--out',    required=True, help='Output .bin file')
    p.add_argument('--train',  action='store_true',
                   help='Train on MNIST data before exporting '
                        '(requires imgPath/imgLabelPath in config)')
    args = p.parse_args()

    cfg   = load_cfg(args.config)
    model = CNNVIT(cfg['num_filter'], cfg['kernel_size'], cfg['pool_size'],
                   cfg['tf_n_heads'], cfg['fc_layers'], cfg['img_size'])

    if args.pt:
        state = torch.load(args.pt, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        print(f"[export] Loaded weights from '{args.pt}'")
    elif args.train:
        if not cfg['imgPath'] or not cfg['imgLabelPath']:
            sys.exit("--train requires imgPath and imgLabelPath in config")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[export] Training on {device} for {cfg['max_iter']} epoch(s)...")
        train_mnist(model, cfg['imgPath'], cfg['imgLabelPath'],
                    cfg['max_iter'], cfg['lr'], device)
    else:
        print("[export] No --pt or --train specified — exporting random weights")

    model.eval()
    export(model, cfg, args.out)


if __name__ == '__main__':
    main()
