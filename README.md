# ENNW: Efficient Neural Network

**A pure C-based deep learning inference engine focused on edge deployment and inference optimization for Transformer architectures.**

ENNW implements all core logic and operators directly in C, with cJSON
as the only third-party dependency. It supports a DistilBERT encoder and a GPT-2
decoder with KV cache and causal masking.

### GPT-2 on Kria KV260

Cortex-A53 @ 1.33 GHz, 4 threads, GPT-2 124M fp32.

| | Baseline | Optimised | |
|---|---|---|---|
| Decode | 948 ms/token | **269 ms/token** | **3.53x** |
| Prefill (5 tokens) | 1701 ms | 1725 ms | 1.4% slower |
| 50-token generation | 46.6 s | **13.6 s** | **3.42x** |
| Peak workspace | 4.39 MB | 4.39 MB | - |

Median of three runs, variation under 0.2%. Logits and token sequence identical
across both configurations.

| | Root cause | Fix |
|---|---|---|
| 948 -> 452 ms | `omp parallel for` on the M loop; M = 1 during decode, so three of four threads idle at the barrier | parallelise over the output dimension |
| 452 -> 269 ms | removing the tile loop touched 768 pages per tile, exceeding the 512-entry L2 TLB | restore tiling inside each thread's contiguous chunk |

[Read the performance case study ->](results/README.md)  

The case study documents the profiling process, implementation changes,
measurement setup, and raw results behind these numbers.

| Stage | Question | Answer |
|---|---|---|
| 0 | Why does the same prefill take 45 ms and 542 ms? | Not known |
| 1 | Is it warm-up? | no - all six runs fluctuate, first is fastest |
| 2 | What are the threads doing? | spinning - 209 s CPU for 1.1 s of work |
| 3 | Is parallelising worth its cost? | not at this granularity 1.68 us compute vs 1300 us sync |
| 4 | (KV260) Where did 948 ms go? | M-dimension parallelism |
| 5 | (KV260) After refactoring, the result was slower than before when thread = 1 | TLB thrashing |

### DistilBERT on Kria KV260

Unlike the GPT-2 experiments, the DistilBERT comparison uses OpenBLAS as the GEMM backend for both PyTorch and ENNW. ENNW accesses OpenBLAS through its CBLAS interface. This controls for the dominant matrix-multiplication kernel and allows the experiment to focus more closely on differences in framework and runtime execution.

Benchmarked on the AMD Kria KV260 (Cortex-A53), ENNW achieves a 13.37x speedup over a PyTorch eager-mode baseline on 100-sentence DistilBERT inference (12.42 s vs. 166 s).

Because both implementations use the same OpenBLAS GEMM backend, the observed difference is not attributable simply to a faster underlying matrix-multiplication library. ENNW reduces runtime overhead through techniques including arena-based memory management, dry-run workspace pre-sizing, and operator fusion such as GEMM + bias + GELU. ENNW was compiled with -O3.

## Current Status

[OK] DistilBERT encoder with INT8 quantization, operator fusion, arena-based memory management  
[OK] GPT-2 decoder's causal masking, multi-layer KV cache, weight-tied LM head  
[OK] GPT-2 weight loading from exported checkpoints  
[OK] Deployment of GPT-2 on Kria KV260  
[WIP] Profiling and case study

## Why ENNW?
Most ML inference frameworks carry heavy runtime dependencies - Python runtimes, dynamic allocators, BLAS libraries. 
Using these dependencies brings convenience for simple deployment of deep learning models, but for those
who want greater control over the system and underlying logic, dependencies could unintentionally obfuscate the code.
Therefore, ENNW takes the opposite approach:

Minimal dependency - user only needs a GNU C Compiler to compile the inference engine
Predictable Memory Model - arena-based allocation eliminates dynamic runtime heap calls
Attention-native - scaled dot-product attention and other crucial tensor operations are implemented from scratch
Readable codebase - each layer is a single, auditable .c file.

This project was built to also deeply understand what happens below PyTorch.

## Build & Run

### TL;DR (Quick Start)
Assuming you have a GCC compiler (GCC/MinGW), Makefile, OpenMP and Python with PyTorch installed:
```bash
# Clone & Enter the repository
git clone "https://github.com/BensonLing0925/ENNW.git"
cd ENNW

# Run the script to download GPT-2 weights (~500 MB)
chmod +x ./scripts/fetch_gpt2.sh
./scripts/fetch_gpt2.sh

# 3. Build & Inference
make clean && make 
OMP_NUM_THREADS=4 ./bin/gpt2_io_test
```

## Makefile commands

```bash
make            # Build
make run        # Build and run
make clean      # Remove build artifacts
make DEBUG=1    # Build with -O0 -g debug flags
make print      # Print build variables
```

### Memory Model

Two-tier allocation system:
- **`struct arena`** (`mem/arena.h`) — general-purpose arena allocator using 64KB linked blocks. Used for persistent metadata (`ctx->meta_arena`) and tensor data (`ctx->data_arena`).
- **`struct tk_workspace`** (`src/runtime/workspaces/`) — stack-style bump allocator for intermediate tensors during a forward pass. Supports a `RT_DRYRUN` mode that measures peak usage, used to pre-size the workspace.

The `RT_DRYRUN` runtime type (`ctx->rt_type`) is set before the inference loop to plan workspace memory; allocations and memory address calculation for each tensors happened during this stage.

### Core Data Structure: `tk_tensor`

All layer inputs/outputs are `struct tk_tensor` (defined in `src/ops/tensor.h`): a dtype-tagged, N-dimensional array with shape/strides arrays and a raw `void* data` pointer. Supported dtypes: `TK_F64`, `TK_F32`, `TK_I16`, `TK_I8`, `TK_U8`.

### Runtime Context

`struct tk_rt_ctx` (`src/runtime/rt_context.h`) is the central handle passed to all layer operations. It owns both arenas, the workspace, and the `Model`.

### Transformer Module and its sub-modules

| Module | Path | Key structs |
|--------|------|-------------|
| Transformer | `src/modules/transformer/` | `TransformerBlock` - multi-head self-attention + FFN, wired into the pipeline |
| Embedding | `src/modules/transformer/embedding` | `Embedding` - basic structure of the embedding type |
| GPT-2 | `src/modules/transformer/gpt2` | gpt2 specific implementation |
| DistilBERT | `src/modules/transformer/distilbert` | distilBERT specific implementation |

### Transformer Architecture

`TransformerBlock` (`src/modules/transformer/tf_block.h`):
- **config**: `seq_length`, `hidden_dim`, `n_heads`, `head_dim`, `inter_dim`
- **Weights**: Q/K/V projections `[hidden, hidden]`, FFN up `[hidden, 4*hidden]`, FFN down `[4*hidden, hidden]`, LayerNorm gamma/beta `[hidden]`
- **Forward**: Both post-norm(distilBERT) and pre-norm(gpt-2)
- **Attention**: Per-head gather/scatter pattern - no non-contiguous tensor views

`tf_block_create(ctx)` + `tf_block_alloc(ctx, tf, seq, hidden, n_heads)` initialise weights from `data_arena`.

### Tensor Operations

Low-level ops live in `src/ops/` (`tensor.c`, `tensor_ops.c`): GEMM, convolution kernel(deprecated), softmax, one-hot encoding(deprecated), ReLU. These are called directly by layer modules rather than going through any dispatch table.

### Weight I/O

`weightio/` handles binary serialization of trained weights(HuggingFace pretrained model weights).

### Config & Dataset

- Config parsing uses bundled **cJSON** (`config/cJSON/`) via `load_json()` in `config/config.c`.

## Validation & Tooling

To bridge the gap between high-level research and low-level C implementation, this repository includes a dedicated tooling suite for weight export and numerical verification.

> **Note on Methodology**: AI was used to co-develop the Python verification suite, enabling rapid **Cross-Framework Test-Driven Development**. This allowed for immediate detection of numerical divergence between the manual C implementation and PyTorch.

### 1. Validation
To ensure the reliability of the C implementation, a comprehensive validation pipeline is established:

- Utilize `tools/export_weights.py` to recreate the exact model architecture in PyTorch. This tool allows for training on standard datasets (like MNIST) and exporting the trained weights into a custom binary format (ENNW) tailored for the C framework.

- Using `tools/verify_engine.py` to perform numerical comparison between the inference results of the PyTorch model and the C engine. This ensures that custom implementations of pointer-based tensor operations, convolution kernels, and Transformer blocks maintain algorithmic parity with industry-standard frameworks.

### 2. Numerical Verification (`tools/verify_engine.py`)
This is the core validation tool that ensures the C engine's algorithmic correctness:Three-Way Comparison: It runs inference on the same sample through PyTorch Native, Python-based C-Simulation, and the Compiled C Executable.

Parity Guarantee: It performs a sample-by-sample logit comparison. Achieving PASS confirms that custom pointer arithmetic, tensor strides, and Transformer attention kernels match industry-standard results within a $10^{-7}$ tolerance.
