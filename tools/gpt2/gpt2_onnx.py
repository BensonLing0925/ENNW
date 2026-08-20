#!/usr/bin/env python3
"""
ORT GPT-2 baseline toolkit  —  for ENNW comparison

Subcommands
-----------
  inspect   <model.onnx>      Does it have KV cache? Did attention fusion fire?
  export    -o <dir>          Fresh export from HF gpt2 with use_cache=True
  optimize  <in> -o <out>     onnxruntime.transformers optimizer, gpt2 params
  bench     <model.onnx>      Prefill / decode latency, p50 / p95, pinned threads
  all       -o <dir>          export -> optimize -> inspect both -> bench both

Install
-------
  pip install "optimum[onnxruntime]" onnx onnxruntime transformers numpy

Note: thread env vars are set BEFORE importing onnxruntime, which is required
for OMP_WAIT_POLICY to take effect.
"""

import os
import sys

# ---- MUST come before importing onnxruntime / numpy-backed OMP libs ----
THREADS = os.environ.get("ENNW_BENCH_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", THREADS)
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("KMP_BLOCKTIME", "0")
# ------------------------------------------------------------------------

import argparse
import json
import re
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

GPT2_SMALL = dict(num_heads=12, hidden_size=768, num_layers=12, head_dim=64)


# =========================================================================
# inspect
# =========================================================================
def cmd_inspect(path, quiet=False):
    import onnx

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph

    in_names = [i.name for i in g.input]
    out_names = [o.name for o in g.output]

    has_past = any("past" in n for n in in_names)
    has_present = any("present" in n for n in out_names)

    ops = Counter(n.op_type for n in g.node)
    ms_nodes = Counter(n.op_type for n in g.node if n.domain == "com.microsoft")

    attention_fused = ms_nodes.get("Attention", 0) + ms_nodes.get(
        "MultiHeadAttention", 0
    )
    softmax_left = ops.get("Softmax", 0)
    layernorm_fused = ms_nodes.get("LayerNormalization", 0) + ops.get(
        "LayerNormalization", 0
    )
    skip_ln_fused = ms_nodes.get("SkipLayerNormalization", 0)
    gelu_fused = (
        ms_nodes.get("Gelu", 0)
        + ms_nodes.get("FastGelu", 0)
        + ms_nodes.get("BiasGelu", 0)
    )
    matmul_int = ops.get("MatMulInteger", 0) + ms_nodes.get("MatMulIntegerToFloat", 0)
    quantized = matmul_int > 0 or ops.get("QuantizeLinear", 0) > 0

    size_mb = path.stat().st_size / 1e6
    ext = list(path.parent.glob(f"{path.stem}*.data")) + list(
        path.parent.glob("*.onnx_data")
    )
    if ext:
        size_mb += sum(f.stat().st_size for f in ext) / 1e6

    report = dict(
        file=str(path),
        size_mb=round(size_mb, 1),
        kv_cache=has_past and has_present,
        attention_fused_nodes=attention_fused,
        softmax_remaining=softmax_left,
        skiplayernorm_fused=skip_ln_fused,
        layernorm_nodes=layernorm_fused,
        gelu_fused=gelu_fused,
        quantized=quantized,
        total_nodes=len(g.node),
        top_ops=dict(ops.most_common(10)),
        inputs=in_names[:6] + (["..."] if len(in_names) > 6 else []),
    )

    if not quiet:
        print(f"\n{'='*66}\n  INSPECT  {path.name}\n{'='*66}")
        print(f"  size                  : {report['size_mb']} MB")
        print(f"  total nodes           : {report['total_nodes']}")
        print(f"  KV cache (past/present): {'YES' if report['kv_cache'] else 'NO'}")
        print(f"  Attention fused nodes : {attention_fused}")
        print(f"  Softmax remaining     : {softmax_left}")
        print(f"  SkipLayerNorm fused   : {skip_ln_fused}")
        print(f"  Gelu fused            : {gelu_fused}")
        print(f"  quantized (INT8)      : {'YES' if quantized else 'NO'}")
        print(f"  first inputs          : {report['inputs']}")

        print("\n  --- verdict ---")
        if not report["kv_cache"]:
            print("  [FATAL] No KV cache. Every decode step recomputes the whole")
            print("          sequence. This baseline is structurally crippled —")
            print("          any speedup measured against it is not meaningful.")
        if attention_fused == 0 and softmax_left > 0:
            print("  [WARN]  Attention fusion did NOT fire (raw Softmax present,")
            print("          no com.microsoft.Attention). Run `optimize`.")
        if attention_fused > 0:
            print(f"  [OK]    Attention fusion fired ({attention_fused} nodes).")
            if softmax_left:
                print(
                    f"          ({softmax_left} Softmax left — usually fine, "
                    "check they're not in attention blocks)"
                )
        print()

    return report


# =========================================================================
# export
# =========================================================================
def cmd_export(outdir, model_id="gpt2"):
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n[export] {model_id} -> {outdir}  (task=text-generation-with-past)")
    m = ORTModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        use_cache=True,
        use_io_binding=False,
    )
    m.save_pretrained(outdir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(outdir)

    cands = sorted(outdir.glob("*.onnx"), key=lambda p: p.stat().st_size, reverse=True)
    if not cands:
        raise SystemExit("[export] no .onnx produced — check optimum version")
    main = cands[0]
    print(f"[export] main graph: {main.name}")
    print(f"[export] all files : {[p.name for p in cands]}")
    return main


# =========================================================================
# optimize
# =========================================================================
def cmd_optimize(inpath, outpath, opt_level=0, fp16=False):
    from onnxruntime.transformers.optimizer import optimize_model

    inpath, outpath = Path(inpath), Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[optimize] {inpath.name}")
    print(
        f"[optimize] model_type=gpt2 num_heads={GPT2_SMALL['num_heads']} "
        f"hidden_size={GPT2_SMALL['hidden_size']} opt_level={opt_level}"
    )

    opt = optimize_model(
        str(inpath),
        model_type="gpt2",
        num_heads=GPT2_SMALL["num_heads"],
        hidden_size=GPT2_SMALL["hidden_size"],
        opt_level=opt_level,   # 0 = let the transformer optimizer own the graph
        use_gpu=False,
        only_onnxruntime=False,
    )
    if fp16:
        opt.convert_float_to_float16(keep_io_types=True)

    stats = opt.get_fused_operator_statistics()
    print(f"[optimize] fused ops: {stats}")

    opt.save_model_to_file(str(outpath), use_external_data_format=False)
    print(f"[optimize] wrote {outpath}")

    # tokenizer / config alongside, so bench can find them
    for f in ("tokenizer.json", "vocab.json", "merges.txt", "config.json",
              "tokenizer_config.json", "special_tokens_map.json"):
        src = inpath.parent / f
        if src.exists() and not (outpath.parent / f).exists():
            shutil.copy(src, outpath.parent / f)

    return outpath


# =========================================================================
# bench
# =========================================================================
def _layer_idx(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def _past_io(sess):
    """Return (past_inputs, present_outputs, style) sorted by layer.
    style: 'split'  -> past_key_values.N.key / .value
           'merged' -> past_N with shape [2,b,h,s,d]
    """
    ins = [i.name for i in sess.get_inputs() if "past" in i.name]
    outs = [o.name for o in sess.get_outputs() if "present" in o.name]
    if not ins:
        return [], [], None
    style = "split" if any(n.endswith((".key", ".value")) for n in ins) else "merged"

    def key(n):
        return (_layer_idx(n), 0 if n.endswith(".key") else 1)

    return sorted(ins, key=key), sorted(outs, key=key), style


def _empty_past(sess, cfg):
    ins, _, style = _past_io(sess)
    h, d = cfg["num_heads"], cfg["head_dim"]
    if style == "split":
        return {n: np.zeros((1, h, 0, d), np.float32) for n in ins}
    return {n: np.zeros((2, 1, h, 0, d), np.float32) for n in ins}


def _feeds(sess, input_ids, past, past_len):
    names = {i.name for i in sess.get_inputs()}
    seq = input_ids.shape[1]
    f = {"input_ids": input_ids}
    if "attention_mask" in names:
        f["attention_mask"] = np.ones((1, past_len + seq), np.int64)
    if "position_ids" in names:
        f["position_ids"] = np.arange(past_len, past_len + seq, dtype=np.int64)[None, :]
    f.update(past)
    return {k: v for k, v in f.items() if k in names}


def cmd_bench(path, prompt_len=64, gen_tokens=64, runs=10, warmup=3,
              threads=None, ort_graph_opt="all", tag=None):
    import onnxruntime as ort

    path = Path(path)
    threads = int(threads or THREADS)

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[ort_graph_opt]

    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    ins, outs, style = _past_io(sess)
    if not ins:
        print("\n[bench] !! model has NO KV cache — decode numbers are not")
        print("[bench]    comparable to a cached engine. Fix the export first.\n")

    cfg = dict(GPT2_SMALL)
    rng = np.random.default_rng(0)

    def one_run():
        ids = rng.integers(0, 50257, size=(1, prompt_len), dtype=np.int64)
        past = _empty_past(sess, cfg) if ins else {}
        t0 = time.perf_counter()
        res = sess.run(None, _feeds(sess, ids, past, 0))
        t1 = time.perf_counter()
        prefill_ms = (t1 - t0) * 1e3

        past_len = prompt_len
        step_ms = []
        for _ in range(gen_tokens):
            if ins:
                past = {i: res[sess.get_outputs().index(
                    next(o for o in sess.get_outputs() if o.name == po))]
                    for i, po in zip(ins, outs)}
            nxt = rng.integers(0, 50257, size=(1, 1), dtype=np.int64)
            s = time.perf_counter()
            res = sess.run(None, _feeds(sess, nxt, past, past_len))
            step_ms.append((time.perf_counter() - s) * 1e3)
            past_len += 1
            if not ins:
                past = {}
        return prefill_ms, step_ms

    for _ in range(warmup):
        one_run()

    prefills, decodes = [], []
    for _ in range(runs):
        p, d = one_run()
        prefills.append(p)
        decodes.extend(d)

    def pct(v, q):
        return statistics.quantiles(v, n=100)[q - 1] if len(v) > 2 else max(v)

    r = dict(
        tag=tag or path.stem,
        file=str(path),
        threads=threads,
        prompt_len=prompt_len,
        gen_tokens=gen_tokens,
        runs=runs,
        kv_cache=bool(ins),
        prefill_p50_ms=round(statistics.median(prefills), 2),
        prefill_p95_ms=round(pct(prefills, 95), 2),
        decode_p50_ms=round(statistics.median(decodes), 3),
        decode_p95_ms=round(pct(decodes, 95), 3),
        decode_min_ms=round(min(decodes), 3),
        decode_max_ms=round(max(decodes), 3),
    )
    r["decode_variance_ratio"] = round(r["decode_max_ms"] / r["decode_min_ms"], 2)
    r["tokens_per_sec"] = round(1000.0 / r["decode_p50_ms"], 2)

    print(f"\n{'='*66}\n  BENCH  {r['tag']}\n{'='*66}")
    print(f"  threads={threads}  OMP_WAIT_POLICY={os.environ['OMP_WAIT_POLICY']}"
          f"  graph_opt={ort_graph_opt}")
    print(f"  prompt={prompt_len}  gen={gen_tokens}  runs={runs} (warmup {warmup})")
    print(f"  KV cache            : {'YES' if r['kv_cache'] else 'NO  <-- suspect'}")
    print(f"  prefill  p50 / p95  : {r['prefill_p50_ms']} / {r['prefill_p95_ms']} ms")
    print(f"  decode   p50 / p95  : {r['decode_p50_ms']} / {r['decode_p95_ms']} ms/token")
    print(f"  decode   min / max  : {r['decode_min_ms']} / {r['decode_max_ms']} ms")
    print(f"  max/min ratio       : {r['decode_variance_ratio']}x", end="")
    print("   <-- still unstable, check thread pinning"
          if r["decode_variance_ratio"] > 3 else "   (stable)")
    print(f"  throughput          : {r['tokens_per_sec']} tok/s\n")
    return r


# =========================================================================
# all
# =========================================================================
def cmd_all(outdir, prompt_len, gen_tokens, runs, threads):
    outdir = Path(outdir)
    raw = cmd_export(outdir / "raw")
    opt = cmd_optimize(raw, outdir / "opt" / "gpt2_opt.onnx")

    ri = cmd_inspect(raw)
    oi = cmd_inspect(opt)

    rb = cmd_bench(raw, prompt_len, gen_tokens, runs, threads=threads, tag="raw")
    ob = cmd_bench(opt, prompt_len, gen_tokens, runs, threads=threads, tag="optimized")

    print(f"{'='*66}\n  SUMMARY\n{'='*66}")
    print(f"  raw       attention_fused={ri['attention_fused_nodes']:3d}  "
          f"decode_p50={rb['decode_p50_ms']} ms")
    print(f"  optimized attention_fused={oi['attention_fused_nodes']:3d}  "
          f"decode_p50={ob['decode_p50_ms']} ms")
    if rb["decode_p50_ms"] and ob["decode_p50_ms"]:
        print(f"  fusion gain: {rb['decode_p50_ms']/ob['decode_p50_ms']:.2f}x")
    print("\n  ^ Use the OPTIMIZED number as your ORT baseline.")
    print("    Reporting against the raw export overstates your speedup.\n")

    res = dict(raw=dict(inspect=ri, bench=rb), optimized=dict(inspect=oi, bench=ob))
    (outdir / "baseline_report.json").write_text(json.dumps(res, indent=2))
    print(f"  wrote {outdir/'baseline_report.json'}\n")
    return res


# =========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inspect"); a.add_argument("model")

    b = sub.add_parser("export")
    b.add_argument("-o", "--out", default="gpt2_onnx/raw")
    b.add_argument("--model-id", default="gpt2")

    c = sub.add_parser("optimize")
    c.add_argument("model"); c.add_argument("-o", "--out", required=True)
    c.add_argument("--opt-level", type=int, default=0)
    c.add_argument("--fp16", action="store_true")

    d = sub.add_parser("bench")
    d.add_argument("model")
    d.add_argument("--prompt-len", type=int, default=64)
    d.add_argument("--gen-tokens", type=int, default=64)
    d.add_argument("--runs", type=int, default=10)
    d.add_argument("--warmup", type=int, default=3)
    d.add_argument("--threads", type=int, default=None)
    d.add_argument("--graph-opt", default="all", choices=["all", "basic", "none"])

    e = sub.add_parser("all")
    e.add_argument("-o", "--out", default="gpt2_onnx")
    e.add_argument("--prompt-len", type=int, default=64)
    e.add_argument("--gen-tokens", type=int, default=64)
    e.add_argument("--runs", type=int, default=10)
    e.add_argument("--threads", type=int, default=None)

    args = p.parse_args()

    if args.cmd == "inspect":
        cmd_inspect(args.model)
    elif args.cmd == "export":
        cmd_export(args.out, args.model_id)
    elif args.cmd == "optimize":
        cmd_optimize(args.model, args.out, args.opt_level, args.fp16)
    elif args.cmd == "bench":
        cmd_bench(args.model, args.prompt_len, args.gen_tokens, args.runs,
                  args.warmup, args.threads, args.graph_opt)
    elif args.cmd == "all":
        cmd_all(args.out, args.prompt_len, args.gen_tokens, args.runs, args.threads)


if __name__ == "__main__":
    sys.exit(main())
