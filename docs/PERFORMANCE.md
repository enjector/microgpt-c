# Performance

## Character-Level vs Karpathy's microgpt.py

Measured on the **character-level name generation** workload (1,000 training steps, 20 inference samples) — MicroGPT-C vs [Karpathy's `microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95):

| Metric | Python (microgpt.py) | C (fp64) | Speedup |
|--------|--------|----------|---------| 
| **Training time** | ~93 s | **0.09 s** | **~1,000×** |
| **Training throughput** | ~0.1 k tok/s | **~616 k tok/s** | **~6,000×** |
| **Steps/sec** | ~11 | **~10,800** | **~1,000×** |
| **Inference time** | ~0.74 s | **< 1 ms** | **~700×+** |

---

## Shakespeare Character-Level (N_EMBD=128, N_LAYER=4, vocab≈84)

Multi-threaded training with auto-detected workers (uses all available CPU cores):

| Metric | Value |
|--------|-------|
| **Model parameters** | ~1M |
| **Vocab** | ~84 characters (zero `<unk>`) |
| **Throughput** | ~85 steps/s, ~64k tok/s (12 threads) |

---

## Benchmarks (N_EMBD=16, N_LAYER=1)

Run `./bench_microgpt` to reproduce on your machine. Default build uses **float32**:

| Operation | float32 (default) | double64 | Speedup |
|---|---|---|---|
| `forward_backward_one` | **530k fwd+bwd/s** | 353k fwd+bwd/s | **1.50×** |
| `adam_step` | **646k steps/s** | 289k steps/s | **2.23×** |
| `sample_token` (vocab=50) | **6.8M samples/s** | 5.4M samples/s | **1.25×** |
| Full training step (seq=8) | **677k tok/s** | 536k tok/s | **1.26×** |
| `checkpoint_save` + `load` | **5,757 rt/s** | 4,846 rt/s | **1.19×** |
| `forward_inference` (1 tok) | 1,413k infer/s | 1,724k infer/s | ~1× (noise) |

**Matrix operations** (where precision matters most):

| Size | float32 | double64 | Speedup |
|---|---|---|---|
| 128×128 (N_EMBD) | **0.86 ms** | 1.90 ms | **2.21×** |
| 512×512 (future) | **17.93 ms** | 37.99 ms | **2.12×** |

**Memory footprint** (vocab=100):

| | float32 | double64 |
|---|---|---|
| Weights | 25.5 KB | 51.0 KB |
| Optimizer | 76.5 KB | 153.0 KB |
| **Total** | **104 KB** | **208 KB** |

> **Convergence:** Both precisions reach identical loss (0.0011 after 100 steps). Float32 is the recommended default — use `-DMICROGPT_USE_FLOAT=OFF` only if you need double-precision research comparisons.
>
> **INT8 quantised build:** ~25% slower training than float32 on this tiny model, but **~8× smaller** weight storage — ideal for constrained devices.

---

## Optimisation Details

The engine includes several optimisations for training throughput:

- **`scalar_t` precision abstraction** — compile-time switch between `float` (default, 2× faster) and `double` via `-DMICROGPT_USE_FLOAT=OFF`; all math (`M_EXP`, `M_LOG`, etc.) and BLAS calls (`CBLAS_GEMV`, `CBLAS_GER`) auto-dispatch to the correct precision
- **Cache-friendly `lin_bwd`** — backward gradient accumulation uses row-major weight traversal, eliminating L1 cache thrashing for large output layers (e.g. lm_head with vocab=10003)
- **Hash-based `word_to_id`** — O(1) DJB2 hash lookup instead of O(n) linear scan across the vocabulary
- **Cosine LR with warmup** — linear warmup for `WARMUP_STEPS` followed by cosine annealing, avoiding premature LR decay
- **`restrict` + vectorisation hints** — C99 `restrict` qualifiers and Clang loop pragmas on all hot-path functions (`lin_fwd`, `lin_bwd`, `rmsnorm_fwd/bwd`) to enable full auto-vectorisation
- **Compiler flags** — `-O3 -ffast-math -march=native -flto -funroll-loops` for Release builds (LTO enables cross-file inlining)
- **Shared training helpers** — `TrainWorker` struct + `train_worker_run` thread entry, `shuffle_docs` — extracted from demos into the core library to eliminate duplication
- **Cross-platform multi-threaded batches** — all demos auto-detect CPU count and parallelise batch processing via the portable threading layer built into `microgpt.h` (pthread on Linux/macOS, Win32 threads on Windows)
- **Optional Metal GPU** — Apple Silicon GPU compute shaders for `lin_fwd`/`lin_bwd` matmuls via `-DMICROGPT_METAL=ON`
- **Optional BLAS** — Hardware-accelerated BLAS (Accelerate, OpenBLAS) for single-threaded inference via `-DMICROGPT_BLAS=ON`
