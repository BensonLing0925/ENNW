# ENNW: Efficient Neural Network Wrapper

**A pure C-based deep learning inference framework with custom memory management and Transformer support.**

This project aim to make a high-performance inference framework written in pure C (C99 and C23). It focuses on manual memory optimization, featuring a dual-tier arena allocation system and support for hybrid CNN-Transformer architectures.

## Why ENNW?
Most ML inference frameworks carry heavy runtime dependencies — Python runtimes, dynamic allocators, BLAS libraries. ENNW takes the opposite approach:

Predictable Memory Model — Minimal stdlib dependency. Uses an Arena-based allocation strategy that eliminates runtime heap calls
Attention-native — scaled dot-product attention implemented from scratch
Readable codebase — each layer is a single, auditable .c file

This project was built to also deeply understand what happens below PyTorch.

## Build & Run

```bash
make            # Build (outputs nn.exe on Windows, nn on Linux)
make run        # Build and run
make clean      # Remove build artifacts
make DEBUG=1    # Build with -O0 -g debug flags
make print      # Print build variables
```

Run the executable with a JSON config file:
```bash
./nn.exe path/to/config.json
```

Config JSON fields: `mode` ("TRAIN"/"TEST"), `seed` (-1 for `time(NULL)`), `lr`, `imgPath`, `imgLabelPath`, `max_iter`, `save_path`.

## Architecture

This is a CNN with attention mechanism implemented inference framework in C with custom memory management.

### Memory Model

Two-tier allocation system:
- **`struct arena`** (`mem/arena.h`) — general-purpose arena allocator using 64KB linked blocks. Used for persistent metadata (`ctx->meta_arena`) and tensor data (`ctx->data_arena`).
- **`struct tk_workspace`** (`src/runtime/workspaces/`) — stack-style bump allocator for intermediate tensors during a forward pass. Supports a `RT_DRYRUN` mode that measures peak usage without allocating, used to pre-size the workspace.

The `RT_DRYRUN` runtime type (`ctx->rt_type`) is set before the training loop to plan workspace memory; actual allocations happen during the real forward pass.

### Core Data Structure: `tk_tensor`

All layer inputs/outputs are `struct tk_tensor` (defined in `src/ops/tensor.h`): a dtype-tagged, N-dimensional array with shape/strides arrays and a raw `void* data` pointer. Supported dtypes: `TK_F64`, `TK_F32`, `TK_I16`, `TK_I8`, `TK_U8`.

### Runtime Context

`struct tk_rt_ctx` (`src/runtime/rt_context.h`) is the central handle passed to all layer operations. It owns both arenas, the workspace, and the `Model`.

### Model Composition

`struct Model` (`src/structDef.h`) holds an array of `LayerMeta` entries, each with a `layer_type` tag (`LAYER_CONV2D=2`, `LAYER_FC=1`, `LAYER_POOL=3`) and a union pointing to the typed layer struct.

### Layer Modules

| Module | Path | Key structs |
|--------|------|-------------|
| Conv2D | `src/modules/conv/` | `tk_conv2d` — filters `[num_filter, C, kH, kW]`, persistent weights in `data_arena` |
| Pooling | `src/modules/pooling/` | `tk_pooling` — `MAX_POOL`/`AVG_POOL`, kernel/stride/padding per axis |
| Fully Connected | `src/modules/fc/` | `Linear` (single layer, weights `[in, out]`) + `Network` (chain of `Linear`) |
| Transformer | `src/modules/transformer/` | `TransformerBlock` — multi-head self-attention + FFN, wired into the pipeline |

### Forward Pass Flow (as implemented in `src/NN.c`)

Per-sample loop (N iterations), then batched FC:
```
For each sample n:
  Raw U8 pixel → normalize to F64 [1, H, W]
  → tk_conv_forward()       (Conv2D: 10 filters, 3×3 → [10, 26, 26])
  → tk_pooling_forward()    (MaxPool: 2×2 → [10, 13, 13])
  → tk_tensor_relu()        (in-place)
  → tf_block_forward()      (Transformer: view as [seq=10, hidden=169])
  → copy flattened output to flat_buf row n

flat_buf [N, 1690]
  → fc_forward()            (FC chain: 1690→100→50→10, softmax + cross-entropy)
```

### Transformer Architecture

`TransformerBlock` (`src/modules/transformer/tf_block.h`):
- **config**: `seq_length`, `hidden_dim`, `n_heads`, `head_dim`, `inter_dim`
- **Weights**: Q/K/V projections `[hidden, hidden]`, FFN up `[hidden, 4*hidden]`, FFN down `[4*hidden, hidden]`, LayerNorm γ/β `[hidden]`
- **Forward**: Pre-norm (LayerNorm → Attention → residual, LayerNorm → FFN → residual)
- **Attention**: Per-head gather/scatter pattern — no non-contiguous tensor views

`tf_block_create(ctx)` + `tf_block_alloc(ctx, tf, seq, hidden, n_heads)` initialise weights from `data_arena`.

Current config in `main()`: seq=10, hidden=169 (13×13 pooled), n_heads=13, head_dim=13.

### Tensor Operations

Low-level ops live in `src/ops/` (`tensor.c`, `tensor_ops.c`): GEMM, convolution kernel, softmax, one-hot encoding, ReLU. These are called directly by layer modules rather than going through any dispatch table.

### Weight I/O

`weightio/` handles binary serialization of trained weights. The two files with current modifications are `weightio/model_io.c` and `weightio/weightio.c`.

### Config & Dataset

- Config parsing uses bundled **cJSON** (`config/cJSON/`) via `load_json()` in `config/config.c`.
- Images are loaded from BMP files via `loadImgFile()` / `loadImgLabel()` in `src/loadPic.c`.

## Validation & Tooling

To bridge the gap between high-level research and low-level C implementation, this repository includes a dedicated tooling suite for weight export and numerical verification.

> **Note on Methodology**: AI was used to co-develop the Python verification suite, enabling rapid **Cross-Framework Test-Driven Development**. This allowed for immediate detection of numerical divergence between the manual C implementation and PyTorch.

### 1. Automated Dataset Preparation
Due to the binary nature of the MNIST dataset, a helper script is provided to ensure environment reproducibility:
```bash
python tools/download_mnist.py
```

### 2. Validation
To ensure the reliability of the C implementation, a comprehensive validation pipeline is established:

- Utilize `tools/export_weights.py` to recreate the exact model architecture in PyTorch. This tool allows for training on standard datasets (like MNIST) and exporting the trained weights into a custom binary format (ENNW) tailored for the C framework.

- Using `tools/verify_engine.py` to perform a bit-exact comparison between the inference results of the PyTorch model and the C engine. This ensures that custom implementations of pointer-based tensor operations, convolution kernels, and Transformer blocks maintain algorithmic parity with industry-standard frameworks.

### 3. Bit-Exact Numerical Verification (`tools/verify_engine.py`)
This is the core validation tool that ensures the C engine's algorithmic correctness:Three-Way Comparison: It runs inference on the same sample through PyTorch Native, Python-based C-Simulation, and the Compiled C Executable.

Parity Guarantee: It performs a sample-by-sample logit comparison. Achieving PASS confirms that custom pointer arithmetic, tensor strides, and Transformer attention kernels match industry-standard results within a $11^{-7}$ tolerance.
