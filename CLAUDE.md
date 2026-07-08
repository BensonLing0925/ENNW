# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
mingw32-make              # Build all targets (Windows with MinGW)
mingw32-make DEBUG=1      # Build with -O0 -g (no OpenMP)
mingw32-make clean        # Remove object files and executables
mingw32-make print        # Print Makefile variables for debugging
```

Four executables are produced: `nn.exe`, `distilbert_infer.exe`, `sst2_infer.exe`, `sst2_eval.exe`.
Entry points live in `src/NN.c` (for `nn.exe`) and `examples/*.c` (for the rest).

On Linux, use `make` instead of `mingw32-make`.

## Running the Executables

**CNN/MNIST pipeline:**
```bash
./nn.exe tools/config_infer.json
```

**DistilBERT inference (random weights):**
```bash
./distilbert_infer.exe
./distilbert_infer.exe --weights distilbert.bin
./distilbert_infer.exe --weights distilbert.bin --verify   # print CLS hidden state per layer
```

**SST-2 sentiment inference:**
```bash
./sst2_infer.exe --weights distilbert_sst2.bin [--tokens tokens.bin] [--int8]
./sst2_eval.exe  --weights distilbert_sst2.bin --data sst2_val.bin [--int8] [--limit N] [--progress]
gdb --args sst2_eval.exe --weights distilbert_sst2.bin --data sst2_val.bin --limit 20 --progress
```

## Weight Export (Python tools)

```bash
# MNIST CNN weights
python tools/export_weights.py --config tools/config_infer.json --out weights.bin

# DistilBERT float32 (VERSION 2, includes SST-2 head)
python tools/export_distilbert_weights.py distilbert_sst2.bin --sst2

# DistilBERT int8 dynamic quantization (VERSION 3)
python tools/export_distilbert_weights.py distilbert_sst2.bin --sst2 --int8

# int8 with static activation calibration (recommended for accuracy)
python tools/export_distilbert_weights.py distilbert_sst2.bin --sst2 --int8 --calib
python tools/export_distilbert_weights.py distilbert_sst2.bin --sst2 --int8 --calib --calib-demo
python tools/export_distilbert_weights.py distilbert_sst2.bin --sst2 --int8 --calib --calib-percentile 99.5
```

## Numerical Verification

```bash
python tools/verify_engine.py      # Three-way PyTorch vs Python-sim vs C, tolerance 1e-7
python tools/verify_distilbert.py  # DistilBERT-specific verification
```

## Architecture

### Memory Model
Two allocator types, never mix them:
- **`struct arena`** (`mem/arena.h`): 64KB linked-block arena. `ctx->meta_arena` holds struct metadata (layer configs, tensor descriptors); `ctx->data_arena` holds persistent weight data.
- **`struct tk_workspace`** (`src/runtime/workspaces/`): bump allocator for temporary tensors during a forward pass. Backed by a single pre-sized slab (256 MB by default). Always reset `ctx->ws->cur_offset = 0` between samples.

Use `TK_WS_BEGIN(ctx)` / `TK_WS_END(ctx)` to save and restore the workspace offset within a single forward pass (scoped scratch allocation). Use `tk_ws_tensor_alloc(ws, meta_arena, dtype, shape, ndims, &out)` to allocate workspace-backed tensors.

### Runtime Context
`struct tk_rt_ctx` (`src/runtime/rt_context.h`) is passed to every layer call:
- `rt_type`: `RT_TRAIN` / `RT_INFERENCE` / `RT_DRYRUN`
- `compute_dtype`: defaults to `TK_F64` — **must be set to `TK_F32` before DistilBERT calls**
- `use_int8`: set to `1` **before** calling `tk_distilbert_alloc` to allocate weight tensors as `TK_I8`
- `use_graph_optimize`: `1` by default — enables static-graph recording (see below)

Create with defaults via `tk_runtime_ctx_create(root_arena)`, or supply a config struct for non-default options (e.g., verify mode requires `use_graph_optimize=0`):
```c
struct tk_rt_ctx_config cfg = { .use_int8 = 0, .use_graph_optimize = 0, .graph_capacity = 0 };
ctx = tk_runtime_ctx_create_config(root_arena, cfg);
ctx->rt_type = RT_INFERENCE;
```

### Ops Vtable and Static Graph
All ops dispatch through `ctx->ops` (`struct tk_ops_vtable*`). Three vtables exist:
- `tk_record_vtable` — records ops into `ctx->static_graph` without executing; active during `RT_DRYRUN` when `use_graph_optimize=1`
- `tk_exec_vtable` — executes float ops directly
- `tk_exec_i8_vtable` — executes int8 ops directly

The static graph currently records ops for inspection (graph optimization is not yet implemented — the fusion pass is commented out in `rt_context.c`). `tk_rt_prepare` prints recorded nodes and switches `ctx->ops` to the appropriate exec vtable.

**Three-phase execution flow** (the correct pattern for multi-sample inference):
```c
// Phase 1 – dry run: sizes workspace AND records static graph
ctx->rt_type = RT_DRYRUN;
tk_distilbert_forward(ctx, model, input_ids, &hidden);
tk_sst2_cls_forward(ctx, cls_head, hidden, &logits);

// Phase 2 – prepare: prints graph, switches ctx->ops to exec vtable, resets workspace
tk_rt_prepare(ctx);  // sets rt_type = RT_INFERENCE internally

// Phase 3 – inference loop
for (int s = 0; s < N; ++s) {
    ctx->rt_type        = RT_INFERENCE;
    ctx->ws->cur_offset = 0;
    tk_distilbert_forward(ctx, model, input_ids, &hidden);
    tk_sst2_cls_forward(ctx, cls_head, hidden, &logits);
}
```

When `use_graph_optimize=0`, skip the dry run and `tk_rt_prepare` — set `ctx->rt_type = RT_INFERENCE` directly and call forward functions normally (see `--verify` mode in `examples/distilbert_infer.c`).

### Tensor System
`struct tk_tensor` (`src/ops/tensor.h`): `dtype` + `void* data` + `shape[]` + `strides[]`. Dtypes: `TK_F64`, `TK_F32`, `TK_I32`, `TK_I16`, `TK_I8`, `TK_U8`.

Type-generic ops use `TK_DISPATCH_TYPES(dtype, name, { /* scalar_t available */ })`.

All ops require contiguous tensors — checked via `tk_tensor_is_contiguous()`.

### Two Pipeline Paths

**CNN + Transformer + FC** (`src/NN.c`, entry point `nn.exe`):
```
U8 image → normalize F64 → Conv2D [10, 26, 26] → MaxPool [10, 13, 13]
→ ReLU → TransformerBlock [seq=10, hidden=169] → flatten → FC chain [1690→100→50→10]
→ softmax + cross-entropy
```
Driven by a JSON config (`tools/config_infer.json`). Key `Config` fields: `num_filter`, `kernel_size`, `pool_size`, `tf_n_heads`, `fc_layers[]`, `dtype`, `weights_path`. Config parsing uses cJSON (`config/cJSON/`), compiled as C89.

**DistilBERT** (`src/modules/transformer/distilbert/`):
```
int32 token IDs → Embedding lookup [seq, 768] → 6× TransformerBlock → hidden [seq, 768]
→ (SST-2) CLS token → pre_classifier Linear → ReLU → classifier Linear → softmax
```
`tk_distilbert_config` allocates structs; `tk_distilbert_alloc` allocates weight tensors. Both must be called before loading weights.

`model->blocks[i]` is `struct tk_distilbert_block*` — a thin wrapper; access the underlying `TransformerBlock` via `model->blocks[i]->base`.

The SST-2 head (`struct tk_sst2_cls_head`, `distilbert_cls.h`) must be separately created with `tk_sst2_cls_create` / `tk_sst2_cls_alloc` and loaded with `tk_distilbert_load_cls_weights`.

### TransformerBlock (`src/modules/transformer/tf_block.h/.c`)
Single block shared by both pipelines. Configured via `tk_tf_block_config`:
- DistilBERT uses **post-norm** (`pre_norm=0`): sublayer → residual → LayerNorm
- CNN uses **pre-norm** (`pre_norm=1`): LayerNorm → sublayer → residual

Attention uses explicit per-head gather/scatter loops (no non-contiguous views) into `[seq, hdim]` buffers allocated from the workspace.

### int8 Quantization
Weights stored as `TK_I8`; activations quantized per-GEMM call.

**Per-weight-entry binary format** (version 3 `.bin` files):
```
int8_data[n]  +  float32 weight_scale  +  float32 act_scale
```
`act_scale = 0.0` → dynamic quantization (compute `max(|x|)/127` at runtime).  
`act_scale > 0.0` → static calibration scale (loaded from file, skips runtime scan).

Scale fields in `TransformerBlock`:
- Weight scales: `q_scale`, `k_scale`, `v_scale`, `o_proj_scale`, `ffn_up_scale`, `ffn_down_scale`
- Activation scales: `attn_in_act_scale`, `o_proj_in_act_scale`, `ffn_up_in_act_scale`, `ffn_down_in_act_scale`

GEMM: `tk_ops_gemm_i8f32()` — int8×int8 → int32 accumulation (tiled, OpenMP), dequantize to float32. LayerNorm, GELU, Softmax remain float32.

### Weight File Formats
`weightio/distilbert_io.c` handles DBERT binary format:
- **Version 1**: base model, float32
- **Version 2**: SST-2 head appended, float32
- **Version 3**: SST-2 head, int8 transformer weights + calibration act_scales

`tk_distilbert_load_weights()` handles all three versions.  
`tk_distilbert_load_cls_weights()` seeks past transformer content (version-aware skip) to load the classification head.

### Error Handling
All layer functions return `int` — `0` on success, negative on error. Use `RT_FAIL(code, fmt, ...)` to set the thread-local error state and return `-1`. Use `RT_CHECK(expr)` to propagate errors up the call stack; use `RT_CHECK_GOTO(expr, label)` when cleanup is needed before returning. Inspect with `rt_err_print(stderr)`. Error codes: `RT_EINVAL`, `RT_EOOM`, `RT_EIO`, `RT_ESTATE`, `RT_EINTERNAL`.
