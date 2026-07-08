#!/usr/bin/env python3
"""
export_distilbert_weights.py
Export HuggingFace DistilBERT weights to the DBERT binary format.

DBERT file layout
-----------------
Header (40 bytes):
  magic[8]       : b"DBERT\\x00\\x00\\x00"
  version[4]     : uint32 = 1 (base) or 2 (SST-2 with classification head)
  num_layers[4]  : uint32  (n_layers, e.g. 6)
  hidden_dim[4]  : uint32  (dim, e.g. 768)
  inter_dim[4]   : uint32  (hidden_dim in HF config = FFN size, e.g. 3072)
  vocab_size[4]  : uint32
  max_seq_len[4] : uint32  (max_position_embeddings)
  n_heads[4]     : uint32
  reserved[4]    : uint32 = 0

Embedding section:
  word_emb  float32[vocab_size, hidden_dim]
  pos_emb   float32[max_seq_len, hidden_dim]
  ln_gamma  float32[hidden_dim]
  ln_beta   float32[hidden_dim]

Per transformer layer (num_layers times):
  q_w      float32[hidden, hidden]  <- transposed from PyTorch [out=hidden, in=hidden]
  q_b      float32[hidden]
  k_w      float32[hidden, hidden]
  k_b      float32[hidden]
  v_w      float32[hidden, hidden]
  v_b      float32[hidden]
  o_proj_w float32[hidden, hidden]
  o_proj_b float32[hidden]
  sa_ln_w  float32[hidden]
  sa_ln_b  float32[hidden]
  ffn_up_w float32[hidden, inter]   <- transposed from PyTorch lin1.weight [inter, hidden]
  ffn_up_b float32[inter]
  ffn_dn_w float32[inter, hidden]   <- transposed from PyTorch lin2.weight [hidden, inter]
  ffn_dn_b float32[hidden]
  out_ln_w float32[hidden]
  out_ln_b float32[hidden]

SST-2 classification head (VERSION 2 only, appended after transformer layers):
  pre_cls_w  float32[hidden, hidden]  <- transposed pre_classifier.weight [hidden, hidden]
  pre_cls_b  float32[hidden]
  cls_w      float32[hidden, 2]       <- transposed classifier.weight [2, hidden]
  cls_b      float32[2]

Transposition rationale:
  PyTorch Linear stores weight as [out_features, in_features].
  Our C GEMM computes  C = A x B  where A=[N, in], B=[in, out].
  So we transpose every weight matrix before writing.

Usage:
  pip install torch transformers
  # Base model only (VERSION 1):
  python tools/export_distilbert_weights.py distilbert.bin
  # SST-2 fine-tuned model with classification head (VERSION 2):
  python tools/export_distilbert_weights.py distilbert_sst2.bin --sst2
"""

import sys
import struct
import argparse

try:
    import numpy as np
    import torch
    from transformers import DistilBertModel, DistilBertForSequenceClassification
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             "Install with:  pip install torch transformers numpy")

MAGIC        = b"DBERT\x00\x00\x00"   # 8 bytes
VERSION      = 1
VERSION_SST2 = 2
VERSION_INT8 = 3   # int8 quantized transformer + float32 SST-2 head


def _u32(v: int) -> bytes:
    return struct.pack("<I", int(v))


def write_f32(buf: bytearray, t: torch.Tensor) -> None:
    """Append tensor as contiguous float32 row-major bytes."""
    arr = t.detach().cpu().to(torch.float32).contiguous().numpy()
    buf += arr.tobytes()


def compute_activation_scale(activation_list, percentile=99.9, use_demo=False):
    """Compute a global activation scale from a list of activation arrays.

    Parameters
    ----------
    activation_list : list of np.ndarray
        Raw activation tensors collected during calibration.
    percentile : float
        Percentile of |activation| used as the clipping bound (robust method).
    use_demo : bool
        If True, use max(|activation|) instead of percentile (demo/simple method).

    Returns
    -------
    float
        scale = clip_value / 127.0
    """
    all_acts = np.concatenate([a.flatten() for a in activation_list])
    abs_acts = np.abs(all_acts)
    if use_demo:
        clip_val = float(abs_acts.max())
    else:
        clip_val = float(np.percentile(abs_acts, percentile))
    return clip_val / 127.0 if clip_val > 0.0 else 1.0


def calibrate_layer_act_scales(model, tokenizer, calib_sentences,
                                 percentile=99.9, use_demo=False):
    """Run calibration sentences through the model and compute per-layer
    input-activation scales for every linear GEMM in the transformer.

    Returns
    -------
    dict  {f"layer{i}_{op}": float}
        Keys: layer{i}_attn_in, layer{i}_o_proj_in, layer{i}_ffn_up_in, layer{i}_ffn_down_in
    """
    from collections import defaultdict

    acts = defaultdict(list)
    hooks = []

    num_layers = model.config.n_layers

    for i in range(num_layers):
        layer = model.distilbert.transformer.layer[i]

        def _make_hook(key):
            def _hook(module, inp, _out):
                # inp is a tuple; inp[0] is the input tensor
                acts[key].append(inp[0].detach().cpu().to(torch.float32).numpy())
            return _hook

        # Q, K, V share the same input → capture from q_lin only
        hooks.append(layer.attention.q_lin.register_forward_hook(
            _make_hook(f"layer{i}_attn_in")))
        hooks.append(layer.attention.out_lin.register_forward_hook(
            _make_hook(f"layer{i}_o_proj_in")))
        hooks.append(layer.ffn.lin1.register_forward_hook(
            _make_hook(f"layer{i}_ffn_up_in")))
        hooks.append(layer.ffn.lin2.register_forward_hook(
            _make_hook(f"layer{i}_ffn_down_in")))

    model.eval()
    with torch.no_grad():
        for sentence in calib_sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            model(**inputs)

    for h in hooks:
        h.remove()

    scales = {}
    for key, act_list in acts.items():
        scales[key] = compute_activation_scale(act_list,
                                                percentile=percentile,
                                                use_demo=use_demo)
        print(f"  calib {key}: act_scale = {scales[key]:.6f}")
    return scales


def write_i8_with_scale(buf: bytearray, t: torch.Tensor,
                         act_scale: float = 0.0) -> None:
    """Quantize weight tensor to int8 and append:
      int8_data  +  float32 weight_scale  +  float32 act_scale

    weight_scale = max(|W|) / 127  (symmetric per-tensor)
    act_scale    = calibration scale for the input activation to this layer
                   (0.0 means use dynamic quantization at inference time)
    """
    arr   = t.detach().cpu().to(torch.float32).contiguous().numpy()
    max_v = float(np.abs(arr).max())
    w_scale = max_v / 127.0 if max_v > 0.0 else 1.0
    arr_i8 = np.clip(np.round(arr / w_scale), -127, 127).astype(np.int8)
    buf += arr_i8.tobytes()
    buf += struct.pack("<f", w_scale)
    buf += struct.pack("<f", act_scale)


def _write_transformer_layers(buf: bytearray, sd: dict, prefix: str,
                               num_layers: int, quantize: bool = False,
                               act_scales: dict = None) -> None:
    """Write all transformer layers.

    act_scales: dict from calibrate_layer_act_scales() or None.
      Keys: layer{i}_attn_in, layer{i}_o_proj_in, layer{i}_ffn_up_in, layer{i}_ffn_down_in
      Missing keys → act_scale = 0.0 (dynamic at inference time).
    """
    def w(key: str) -> torch.Tensor:
        if key not in sd:
            raise KeyError(f"Missing weight key: {key}")
        return sd[key]

    def _act(key: str) -> float:
        if act_scales and key in act_scales:
            return act_scales[key]
        return 0.0   # 0.0 = dynamic quantization

    for i in range(num_layers):
        p = f"{prefix}.layer.{i}"

        if quantize:
            attn_s   = _act(f"layer{i}_attn_in")
            o_proj_s = _act(f"layer{i}_o_proj_in")
            ffn_up_s = _act(f"layer{i}_ffn_up_in")
            ffn_dn_s = _act(f"layer{i}_ffn_down_in")

            # Q, K, V share the same input → same act_scale for all three
            write_i8_with_scale(buf, w(f"{p}.attention.q_lin.weight").T.contiguous(), attn_s)
            write_f32(buf, w(f"{p}.attention.q_lin.bias"))
            write_i8_with_scale(buf, w(f"{p}.attention.k_lin.weight").T.contiguous(), attn_s)
            write_f32(buf, w(f"{p}.attention.k_lin.bias"))
            write_i8_with_scale(buf, w(f"{p}.attention.v_lin.weight").T.contiguous(), attn_s)
            write_f32(buf, w(f"{p}.attention.v_lin.bias"))
            write_i8_with_scale(buf, w(f"{p}.attention.out_lin.weight").T.contiguous(), o_proj_s)
            write_f32(buf, w(f"{p}.attention.out_lin.bias"))
            write_f32(buf, w(f"{p}.sa_layer_norm.weight"))
            write_f32(buf, w(f"{p}.sa_layer_norm.bias"))
            write_i8_with_scale(buf, w(f"{p}.ffn.lin1.weight").T.contiguous(), ffn_up_s)
            write_f32(buf, w(f"{p}.ffn.lin1.bias"))
            write_i8_with_scale(buf, w(f"{p}.ffn.lin2.weight").T.contiguous(), ffn_dn_s)
            write_f32(buf, w(f"{p}.ffn.lin2.bias"))
            write_f32(buf, w(f"{p}.output_layer_norm.weight"))
            write_f32(buf, w(f"{p}.output_layer_norm.bias"))
            calib_str = "calibrated" if act_scales else "dynamic (no calib)"
            print(f"  layer {i} exported (int8, act_scales: {calib_str})")
        else:
            write_f32(buf, w(f"{p}.attention.q_lin.weight").T.contiguous())
            write_f32(buf, w(f"{p}.attention.q_lin.bias"))
            write_f32(buf, w(f"{p}.attention.k_lin.weight").T.contiguous())
            write_f32(buf, w(f"{p}.attention.k_lin.bias"))
            write_f32(buf, w(f"{p}.attention.v_lin.weight").T.contiguous())
            write_f32(buf, w(f"{p}.attention.v_lin.bias"))
            write_f32(buf, w(f"{p}.attention.out_lin.weight").T.contiguous())
            write_f32(buf, w(f"{p}.attention.out_lin.bias"))
            write_f32(buf, w(f"{p}.sa_layer_norm.weight"))
            write_f32(buf, w(f"{p}.sa_layer_norm.bias"))
            write_f32(buf, w(f"{p}.ffn.lin1.weight").T.contiguous())
            write_f32(buf, w(f"{p}.ffn.lin1.bias"))
            write_f32(buf, w(f"{p}.ffn.lin2.weight").T.contiguous())
            write_f32(buf, w(f"{p}.ffn.lin2.bias"))
            write_f32(buf, w(f"{p}.output_layer_norm.weight"))
            write_f32(buf, w(f"{p}.output_layer_norm.bias"))
            print(f"  layer {i} exported (float32)")


def build_dbert_binary(sd: dict, cfg, sst2: bool = False,
                        quantize: bool = False, act_scales: dict = None) -> bytes:
    """Build the DBERT binary from a state dict and HF config.

    quantize=True produces VERSION_INT8 (int8 transformer weights + float32 SST-2 head).
    Requires sst2=True.
    """
    if quantize and not sst2:
        raise ValueError("--int8 requires --sst2 (int8 export always includes the SST-2 head)")

    num_layers = cfg.n_layers
    hidden_dim = cfg.dim
    inter_dim  = cfg.hidden_dim          # FFN intermediate size
    vocab_size = cfg.vocab_size
    max_seq    = cfg.max_position_embeddings
    n_heads    = cfg.n_heads

    if quantize:
        version = VERSION_INT8
    elif sst2:
        version = VERSION_SST2
    else:
        version = VERSION

    print(f"Model: layers={num_layers}, hidden={hidden_dim}, inter={inter_dim}, "
          f"vocab={vocab_size}, max_seq={max_seq}, heads={n_heads}, "
          f"version={version}")

    buf = bytearray()

    # ---- Header ----
    buf += MAGIC
    buf += _u32(version)
    buf += _u32(num_layers)
    buf += _u32(hidden_dim)
    buf += _u32(inter_dim)
    buf += _u32(vocab_size)
    buf += _u32(max_seq)
    buf += _u32(n_heads)
    buf += _u32(0)   # reserved

    # ---- Embedding section (always float32) ----
    emb_prefix = "distilbert.embeddings" if sst2 else "embeddings"
    tf_prefix  = "distilbert.transformer" if sst2 else "transformer"

    def w(key: str) -> torch.Tensor:
        if key not in sd:
            raise KeyError(f"Missing weight key: {key}")
        return sd[key]

    write_f32(buf, w(f"{emb_prefix}.word_embeddings.weight"))
    write_f32(buf, w(f"{emb_prefix}.position_embeddings.weight"))
    write_f32(buf, w(f"{emb_prefix}.LayerNorm.weight"))
    write_f32(buf, w(f"{emb_prefix}.LayerNorm.bias"))

    # ---- Transformer layers ----
    _write_transformer_layers(buf, sd, tf_prefix, num_layers,
                               quantize=quantize, act_scales=act_scales)

    # ---- SST-2 classification head (versions 2 and 3, always float32) ----
    if sst2:
        write_f32(buf, w("pre_classifier.weight").T.contiguous())  # [hidden, hidden]
        write_f32(buf, w("pre_classifier.bias"))                    # [hidden]
        write_f32(buf, w("classifier.weight").T.contiguous())       # [hidden, 2]
        write_f32(buf, w("classifier.bias"))                        # [2]
        print("  SST-2 classification head exported (float32)")

    return bytes(buf)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export HuggingFace DistilBERT weights to DBERT binary format")
    p.add_argument("output", help="Output .bin file path")
    p.add_argument("--model", default=None,
                   help="HuggingFace model name or local path. "
                        "Defaults to distilbert-base-uncased (base) or "
                        "distilbert-base-uncased-finetuned-sst-2-english (--sst2).")
    p.add_argument("--sst2", action="store_true",
                   help="Export DistilBertForSequenceClassification (SST-2 fine-tune) "
                        "with classification head. Produces VERSION 2 file.")
    p.add_argument("--int8", action="store_true",
                   help="Quantize transformer weight matrices to int8 (symmetric "
                        "per-tensor). Requires --sst2. Produces VERSION 3 file.")
    p.add_argument("--calib", action="store_true",
                   help="Run activation calibration (requires --int8). "
                        "Computes static per-layer activation scales from a small "
                        "sentence set and embeds them in the .bin file. "
                        "At inference, these replace dynamic per-batch quantization.")
    p.add_argument("--calib-percentile", type=float, default=99.9,
                   help="Percentile of |activation| used as clipping bound "
                        "(default: 99.9). Ignored when --calib-demo is set.")
    p.add_argument("--calib-demo", action="store_true",
                   help="Use max(|activation|) instead of percentile for calibration "
                        "(matches the demo method in compute_activation_scale).")
    args = p.parse_args()

    if args.int8 and not args.sst2:
        p.error("--int8 requires --sst2")
    if args.calib and not args.int8:
        p.error("--calib requires --int8")

    if args.model is None:
        args.model = ("distilbert-base-uncased-finetuned-sst-2-english"
                      if args.sst2 else "distilbert-base-uncased")

    print(f"Loading '{args.model}' from HuggingFace...")

    if args.sst2:
        model = DistilBertForSequenceClassification.from_pretrained(args.model)
    else:
        model = DistilBertModel.from_pretrained(args.model)
    model.eval()

    # ---- Optional activation calibration ----
    act_scales = None
    if args.calib:
        from transformers import DistilBertTokenizer
        tokenizer_name = args.model or "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

        calib_sentences = [
            "The moonlight shimmered over the ocean as waves gently kissed the sandy shore.",
            "A scientist discovered an unexpected pattern in the data that changed everything.",
            "The film was a masterpiece of storytelling and visual artistry.",
            "Stock markets tumbled after the central bank raised interest rates unexpectedly.",
            "Children laughed and played freely in the golden afternoon sunlight.",
            "The new software update introduced several critical security vulnerabilities.",
            "Scientists have long debated the origins of consciousness and self-awareness.",
            "I absolutely loved the restaurant — the food was incredible and the service perfect.",
            "The documentary was dull and failed to make its subject matter interesting.",
            "Despite the harsh criticism, the team remained focused and delivered great results.",
        ]
        print(f"\nRunning calibration on {len(calib_sentences)} sentences "
              f"({'demo/max' if args.calib_demo else f'percentile={args.calib_percentile}'})...")
        act_scales = calibrate_layer_act_scales(
            model, tokenizer, calib_sentences,
            percentile=args.calib_percentile,
            use_demo=args.calib_demo,
        )
        print(f"Calibration done ({len(act_scales)} scales computed).\n")

    data = build_dbert_binary(model.state_dict(), model.config,
                               sst2=args.sst2, quantize=args.int8,
                               act_scales=act_scales)

    with open(args.output, "wb") as f:
        f.write(data)

    mb = len(data) / (1024 * 1024)
    print(f"\nExported {mb:.1f} MB -> '{args.output}'")


if __name__ == "__main__":
    main()
