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

## Names Demo (N_EMBD=16, N_LAYER=1, vocab=27)

The simplest possible model — generates plausible human names from a 32K-name corpus:

| Metric | Value |
|--------|-------|
| **Model parameters** | 4,192 |
| **Vocab** | 27 characters |
| **Training throughput** | ~685K tok/s |
| **Training time** | <0.1s (1K steps) |
| **Inference throughput** | ~110K tok/s |
| **Checkpoint size** | 49 KB |

---

## Shakespeare Character-Level (N_EMBD=128, N_LAYER=4, vocab=84)

Multi-threaded training with auto-detected workers (uses all available CPU cores):

| Metric | Value |
|--------|-------|
| **Model parameters** | ~841K |
| **Vocab** | 84 characters (zero `<unk>`) |
| **Training throughput** | ~28K tok/s (12 threads) |
| **Training time** | ~14 min (30K steps) |
| **Inference throughput** | ~16K tok/s |

---

## Shakespeare Word-Level (N_EMBD=48, N_LAYER=1, vocab=5000)

| Metric | Value |
|--------|-------|
| **Model parameters** | ~510K |
| **Vocab** | 5,000 words (7.4% UNK rate) |
| **Training throughput** | ~12.5K tok/s (12 threads) |
| **Training time** | ~2 min (10K steps) |
| **Inference throughput** | ~40K tok/s |

> Word-level models achieve **2.5× faster inference** than character-level (40K vs 16K tok/s) because each token represents a whole word, dramatically reducing sequence length for the same text.

---

## Organelle Game Pipeline (Evaluation Speed)

Time to play 100 games using pretrained checkpoints (no training):

| Game | Pipeline time | Games/sec |
|------|-------------:|-----------:|
| **Klotski** | 1.4s | 71 |
| **Red Donkey** | 1.9s | 53 |
| **Mastermind** | 1.0s | 100 |
| **8-Puzzle** | 1.2s | 25 |
| **Sudoku** | 1.7s | 60 |
| **Tic-Tac-Toe** | 1.5s | 67 |
| **Othello** | 5.7s | 18 |
| **Pentago** | 6.5s | 15 |
| **Connect-4** | 7.1s | 14 |
| **Hex** | 11.4s | 9 |
| **Lights Out** | 15.8s | 6 |

---

## Benchmarks (N_EMBD=16, N_LAYER=1)

Run `./bench_microgpt` to reproduce on your machine. Default build uses **float32**:

| Operation | float32 (default) | double64 | Speedup |
|---|---|---|---|
| `forward_backward_one` | **551k fwd+bwd/s** | 353k fwd+bwd/s | **1.56×** |
| `adam_step` | **662k steps/s** | 289k steps/s | **2.29×** |
| `sample_token` (vocab=50) | **6.5M samples/s** | 5.4M samples/s | **1.20×** |
| Full training step (seq=8) | **642k tok/s** | 536k tok/s | **1.20×** |
| `checkpoint_save` + `load` | **4,972 rt/s** | 4,846 rt/s | **1.03×** |
| `forward_inference` (1 tok) | **1,554k infer/s** | 1,724k infer/s | ~1× (noise) |
| Auto-regressive inference (seq=16) | **1,180k tok/s** | — | — |

**Tokenisation throughput:**

| Tokeniser | Throughput |
|---|---|
| Character-level (12 chars) | **33.2M tok/s** |
| Word-level (1KB text) | **854K tok/s** |
| Vocabulary build (1KB text) | **249K builds/s** |

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

## VM Engine Benchmarks (Single Thread)

Run `./bench_microgpt_vm` to reproduce. All operations measured on a **single thread**:

| Operation | Dispatches/sec |
|---|---|
| Simple function execution | **3.7M/s** |
| Conditional branching (if/else) | **5.8M/s** |
| Function call (single param) | **4.0M/s** |
| Opaque handle + rolling mean | **643K/s** |

The VM executes deterministic validation and function calls at millions of operations per second, making it practical for O(1) judge-speed rejection in the OPA pipeline.

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
