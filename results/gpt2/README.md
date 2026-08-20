# Experimental Results

This directory collects experiments related to OpenMP (omp) configuration and
GPT-2 inference speed.
The document focuses on recording a series of attempts to solve the problem of
prefill latency fluctuation and the reasoning behind what the next experiment should be.

---

## Initial discovery: anomalous difference between generate 5 and 50 tokens test
`gpt2/omp/0_gpt2_token_test`

Prefill latency differed by 12x between two runs on identical input
(45.3 ms vs 541.9 ms), despite both processing the same 5-token prompt.

Key figures:

| Config | Latency |
|---|---|
| generate 5 tokens, prefill stage | 45.281 ms |
| generate 50 tokens, prefill stage | 541.910 ms |

The experiments above both use the same prompt ("The cat sat on the"), which means the length of the inputs and
the computation paths are identical.
However, the prefill latency difference is too large to dismiss since theoretically
the difference should be small and negligible.

Another clue to this problem is the operator-level profiling:

| Config | Operator | Latency | Latency share |
|---|---|---|---|
| generate 5 tokens, prefill stage | Attention | 11.304 ms | 24.98% |
| generate 50 tokens, prefill stage | Attention | 486.268 ms | 89.74% |

The culprit of the fluctuation is located inside the Attention mechanism,
which involves operators including scale, add, softmax, and gemm.
Instinctively, I aimed at gemm, thinking it was the reason why the anomaly happened.

(For those who are wondering why the Operator gemm does not reflect on Attention accordingly,
the reason is that ENNW treats Attention as a node type, therefore the latencies of operators
that are inside of an Attention node do not reflect on other operators' record)

### Hypothesis:
H1: The LM Head computation, which has 5(prompt_len) x 768(hidden_dim) x 50257(vocab_size), causes such latency.
Rejected: LM Head executes exactly once per prefill, making it a fixed cost. A fixed cost cannot produce run-to-run variance.
H2: System-related factors caused the fluctuation, need more thorough investigation.

---

## Warmup problem

The first reason I came up with to explain the anomaly was that the above experiments only ran once per configuration:
the cold start might increase inference latency.
Even though it is unlikely that cold start can contribute to such a significant latency gap,
it is still necessary to rule it out.

The test is designed to be falsifiable: run prefill six times consecutively.
**If only the first run is slow and the remaining five agree, warm-up is
confirmed.**

Six prefill times (the warm-up test used the default thread count and its output was not saved;
the 20-thread run below reproduces the same behaviour, so it is used here instead.
At this point, I still did not know what the problem was):

44.237 1484.968 401.234 81.700 47.816 53.193 (ms)

The six prefill latencies fluctuate and the numbers are so sporadic that
the cause of the fluctuation cannot be cold start. Note that the first run is
among the fastest, which is the opposite of what warm-up would predict.

The same data undermines the initial speculation. A single configuration
reproduces the entire range, so `max_new_tokens` was never the variable.
The two runs in the previous section simply sampled different points of one distribution.
Therefore the apparent correlation with token count was coincidence.

Two conclusions follow, and the second is the more useful one:

1. Cold start is not the cause.
2. **Any single measurement here is meaningless.** Every figure from this point
   on needs repetition.

## Reproduction of the fluctuation: GEMM and OpenMP
`1_omp_thread_test`

ENNW's GEMM implementation (at `src/ops/tensor_ops.c`) uses OpenMP to accelerate GEMM operations by utilizing
parallelism.
Therefore, I shifted my focus to OpenMP, setting different thread counts to run inferences by setting OMP_NUM_THREADS.

Key figures (six consecutive prefill runs per configuration):

| Threads | Prefill (median) | Prefill range | Decode / token |
|---------|------------------|---------------|----------------|
| 1 | 93.4 ms | 92.4 - 95.7 ms | 21.5 ms |
| **4** | **41.0 ms** | 39.6 - 43.0 ms | **21.4 ms** |
| 8 | 39.5 ms | 39.2 - 40.4 ms | 30.7 ms |
| 16 | 46.3 ms | 40.8 - 47.2 ms | 88.0 ms |
| 20 | 227.2 ms | **44.2 - 1485.0 ms (34x)** | 249.2 ms |

Three important observations:
1. **Fluctuation only happened when using 20 threads**: precisely the point where
  thread count equals the number of logical processors. Every other setting is
  stable within a few percent.
2. **Prefill saturates at 8 threads** (2.37x over single-threaded) and regresses
  beyond that. Additional threads stop helping well before 20.
3. **Decode degrades monotonically past 4 threads**, reaching 249 ms/token at 20,
 a 12x regression against its own best. For the decode phase, it gains nothing from extra threads,
 so they are pure overhead.

This experiment located the fluctuation, but it did not explain why
saturating all the cores caused this problem.
To find out the reason why this happens, it is crucial to know what the threads are actually doing.

## Further investigation about threads
`2_perf_sched`

Thread count correlates with the problem, but what correlation exactly?
`perf sched timehist` records every scheduling event to see
 are these threads computing, or waiting.

Filtered to the process:

| Config | Wall | CPU time | Avg cores busy |
|--------|------|----------|----------------|
| default (active spin) | 13.37 s | **209.0 s** | 15.64 |
| `OMP_WAIT_POLICY=passive` | 4.27 s | **6.3 s** | 1.49 |

Single-threaded baseline for the same work: **~1.1 s CPU time**
(derived from 93.4 ms prefill + 50 x 21.5 ms decode).

The only variable is the barrier wait policy, and the output is byte-identical
in both cases. CPU time dropping 33x therefore isolates roughly 200 seconds as
barrier spinning that produced no computation.

With passive waiting, prefill stabilises at 47-51 ms even at 20 threads.
The fluctuation disappears entirely.

### A correction worth recording

The scheduling slice distribution initially looked like supporting evidence:

| | active | passive |
|---|---|---|
| worker slices | 21,811 | 721,599 |
| median slice | 2.985 ms | 0.001 ms |
| slices < 0.1 ms | 25.3% | **99.8%** |

25.3% of slices under 0.1 ms seemed like a busy-wait signature. The control run
disproves this — passive is *more* fragmented yet 3x faster.

Slice length is not the discriminator. What matters is whether a slice occupies
a core, which is captured by CPU time and average cores busy. Fragmentation
alone says nothing.

Run `./analyse.sh <timehist.txt>` to regenerate these figures.

## Is parallelisation worth its cost?
`3_omp_micro_gemm_test`

Five consecutive runs per thread count.

### At 4 threads - measurements are stable

| Operation | mxnxk | Median | Range |
|-----------|-------|--------|-------|
| attn score `[5,64]x[64,5]` | 1,600 | 1.68 us | 1.53 - 1.77 us |
| attn out `[5,5]x[5,64]` | 1,600 | 0.95 us | 0.84 - 1.04 us |
| decode QKV `[1,768]x[768,768]` | 590 K | 70.3 us | 69.8 - 75.5 us |
| prefill QKV `[5,768]x[768,768]` | 2.95 M | 139.7 us | 139.4 - 159.7 us |
| FFN up `[5,768]x[768,3072]` | 11.8 M | 596.3 us | 595.1 - 601.6 us |
| LM head `[1,768]x[768,50257]` | 38.6 M | 5181 us | 5081 - 5477 us |
| *empty parallel region* | - | 0.846 us | 0.747 - 0.995 us |

### At 20 threads - measurement itself breaks down

| Operation | Range | Variation |
|-----------|-------|-----------|
| prefill QKV `[5,768]x[768,768]` | 223 - 6411 us | **29x** |
| FFN up `[5,768]x[768,3072]` | 842 - 5398 us | 6.4x |
| decode QKV `[1,768]x[768,768]` | 1483 - 6035 us | 4x |
| LM head `[1,768]x[768,50257]` | 10589 - 13448 us | 1.3x |
| *empty parallel region* | 985 - 1919 us | 1.9x |

At 20 threads the arithmetic itself becomes unpredictable, not merely the
synchronisation around it. Spinning workers saturate every core, so even the
thread doing real work competes for scheduling.

The clearest case is `[1,768]x[768,768]`, where M = 1 leaves exactly one
parallel iteration: 70 us at 4 threads, 1483-6035 us at 20. Nineteen threads
with no work to do slow the one thread that has work by up to 86x.
