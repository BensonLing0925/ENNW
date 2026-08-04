#!/usr/bin/env python3
"""
tokenize_sst2.py

Download the SST-2 validation set from HuggingFace, tokenize with
DistilBertTokenizer, and write a batch binary for sst2_eval.c.

Binary format
-------------
  int32_t  num_samples
  int32_t  seq_len          (all samples padded / truncated to this length)
  per sample (num_samples times):
    int32_t  label           (0 = Negative, 1 = Positive)
    int32_t  token_ids[seq_len]

Usage
-----
  pip install datasets transformers
  python tools/tokenize_sst2.py sst2_val.bin
  python tools/tokenize_sst2.py sst2_val.bin --seq-len 128
"""

import sys
import struct
import argparse

try:
    from datasets import load_dataset
    from transformers import DistilBertTokenizerFast
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             "Install with:  pip install datasets transformers")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tokenize SST-2 validation set -> batch binary for sst2_eval")
    p.add_argument("output", help="Output .bin file path")
    p.add_argument("--seq-len", type=int, default=64,
                   help="Pad / truncate to this many tokens (default: 64)")
    p.add_argument("--split", default="validation",
                   choices=["train", "validation"],
                   help="Dataset split to export (default: validation)")
    p.add_argument("--tokenizer", default="distilbert-base-uncased",
                   help="HuggingFace tokenizer name (default: distilbert-base-uncased)")
    args = p.parse_args()

    seq_len = args.seq_len

    print(f"Loading SST-2 '{args.split}' split...")
    dataset = load_dataset("nyu-mll/glue", "sst2", split=args.split)
    print(f"  {len(dataset)} samples")

    print(f"Loading tokenizer '{args.tokenizer}'...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer)

    samples = []
    max_natural_len = 0

    for item in dataset:
        enc = tokenizer(
            item["sentence"],
            max_length=seq_len,
            padding="max_length",
            truncation=True,
        )
        ids    = enc["input_ids"]          # list of int, length == seq_len
        label  = int(item["label"])        # 0 or 1
        # track natural length (before padding) for info
        natural = sum(1 for t in ids if t != tokenizer.pad_token_id)
        max_natural_len = max(max_natural_len, natural)
        samples.append((label, ids))

    num_samples = len(samples)
    print(f"  max natural sequence length (incl. [CLS]/[SEP]): {max_natural_len}")
    print(f"  writing {num_samples} samples, seq_len={seq_len} -> '{args.output}'")

    with open(args.output, "wb") as f:
        f.write(struct.pack("<i", num_samples))
        f.write(struct.pack("<i", seq_len))
        for label, ids in samples:
            f.write(struct.pack("<i", label))
            f.write(struct.pack(f"<{seq_len}i", *ids))

    size_mb = (8 + num_samples * (1 + seq_len) * 4) / (1024 * 1024)
    print(f"  done. file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
