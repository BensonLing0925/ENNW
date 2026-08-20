# Experimental Results

Measurements and verification runs for the ENNW inference engine.
Each subdirectory has its own README with experimental setup and conclusions.

---

## DistilBERT

### Layer-wise numerical verification
`distilbert/layerwise_verification.txt`

Per-block `[CLS]` hidden state from the C engine against a PyTorch reference,
matching to ~1e-3 across all six blocks. Establishes that the encoder path is
numerically correct before any optimisation work.

### Operator fusion
`distilbert/fusion/`

SST-2 validation set, 872 samples, x86-64:

| Configuration | Accuracy | Per-sample | Total |
|---------------|----------|------------|-------|
| No fusion | 90.37% (788/872) | 255.6 ms | 222.91 s |
| ADD+LAYERNORM, GEMM+ADD+GELU fused | 90.37% (788/872) | 213.7 ms | 186.39 s |

**16.4% latency reduction with an identical confusion matrix** (TP 397 /
TN 391 / FP 37 / FN 47), confirming fusion preserves numerical semantics
rather than trading accuracy for speed.

---

## GPT-2

### OpenMP over-subscription investigation
`gpt2/`

Prefill latency varied by 34× between runs on identical input (44–1485 ms)
on a 20-thread x86-64 laptop.
A four-stage investigation traced this to the GEMM kernel
parallelising over the M dimension, where M in this case could only be {1, 5} for every decoder shape —
at most 5 of 20 threads receive work while the rest spin at the barrier.

Key figures:

| | |
|---|---|
| CPU time, default policy | 209.0 s |
| CPU time, `OMP_WAIT_POLICY=passive` | 6.3 s |
| Single-threaded baseline | ~1.1 s |
| attention GEMM (compute) | 1.24 us |
| empty parallel region, 20 threads | 1961 us |
| empty parallel region, 4 threads | 0.831 us |

Root cause identified and quantified; the re-dimensioning fix is outstanding.
See the directory README for the full record.

---

## Reproducing

All runs use `gcc -O3 -fopenmp` on the same machine (20 logical processors,
hybrid P/E core). Model weights are not committed — see the top-level README
for download instructions.

Each subdirectory README lists the exact commands used.
