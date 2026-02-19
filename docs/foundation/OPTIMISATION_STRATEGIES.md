# MicroGPT-C Optimisation Strategies

**A technical white paper on CPU, accelerator, and memory optimisations
for a minimal C99 GPT implementation.**

*Copyright © 2026 Ajay Soni, Enjector Software Ltd. MIT License.*

---

## Spear Summary

**Point:** For small Transformers (875K params) the CPU with compiler auto-vectorisation beats both BLAS libraries and GPU offloading — dispatch overhead kills everything else.

**Picture:** It's like hiring a helicopter to deliver a pizza across the street. The pizza is ready before the helicopter starts its engines. GPU dispatch overhead (50μs) is 10× longer than the actual 128×128 matmul (5μs).

**Proof:** CPU+SIMD: 960K tok/s inference. Metal GPU: 18K tok/s. BLAS: 280K tok/s training (slower than C loops due to thread contention). The CPU wins by 50× over GPU at this scale.

**Push:** Keep defaults as-is (`-O3 -ffast-math -march=native`). Revisit Metal and BLAS when N_EMBD reaches 512+ where dispatch overhead is amortised.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Configuration & Baseline](#2-system-configuration--baseline)
3. [Strategy 1 — SIMD Vectorisation](#3-strategy-1--simd-vectorisation)
4. [Strategy 2 — Apple Accelerate BLAS](#4-strategy-2--apple-accelerate-blas)
5. [Strategy 3 — Metal GPU Compute](#5-strategy-3--metal-gpu-compute)
6. [Strategy 4 — Compiler Optimisation Flags](#6-strategy-4--compiler-optimisation-flags)
7. [Strategy 5 — Per-Head Attention Parallelism](#7-strategy-5--per-head-attention-parallelism)
8. [Strategy 6 — Cache Tiling (Weight Blocking)](#8-strategy-6--cache-tiling-weight-blocking)
9. [Strategy 7 — Paged KV Cache](#9-strategy-7--paged-kv-cache)
10. [Strategy 8 — `scalar_t` Precision Abstraction](#10-strategy-8--scalar_t-precision-abstraction)
11. [Strategy 9 — Common Training Code Extraction](#11-strategy-9--common-training-code-extraction)
12. [Build Matrix & CMake Flags](#12-build-matrix--cmake-flags)
13. [Empirical Results & Analysis](#13-empirical-results--analysis)
14. [Conclusions & Recommendations](#14-conclusions--recommendations)

---

## 1. Executive Summary

MicroGPT-C is a zero-dependency C99 GPT-2-style transformer designed for embedded
and educational use. This document describes nine optimisation strategies explored
during development, ranging from compiler flags and SIMD hints to GPU offloading,
cache-aware memory access patterns, and configurable precision.

**Key finding**: For small models (N_EMBD=128, ~875K parameters), the CPU with
NEON SIMD auto-vectorisation outperforms both BLAS libraries and GPU offloading.
The dominant bottleneck is *dispatch overhead*, not arithmetic throughput. The
optimisations documented here provide infrastructure for when model dimensions
grow beyond L1 cache capacity (N_EMBD ≥ 512).

### Architecture Constants (Production Build)

| Constant | Value | Description |
|----------|-------|-------------|
| `N_EMBD` | 128 | Embedding dimension |
| `N_HEAD` | 8 | Number of attention heads |
| `N_LAYER` | 4 | Transformer blocks |
| `BLOCK_SIZE` | 256–512 | Maximum sequence length |
| `MAX_VOCAB` | 257 | Character-level vocabulary |
| Parameters | ~875K | Total trainable weights |

---

## 2. System Configuration & Baseline

**Target platform**: Apple M2 Max (12 CPU cores, 38 GPU cores)

### Cache Hierarchy (M2 Max)

| Level | Size | Latency | Notes |
|-------|------|---------|-------|
| L1 Data | 128 KB / core | ~3 cycles | Per performance core |
| L2 | 4 MB / cluster | ~10 cycles | Shared across cluster |
| L3 (SLC) | 32 MB | ~30 cycles | System-level cache |

### Working Set Analysis (N_EMBD=128)

Sizes shown for both precision modes (`double` = 8B, `float` = 4B per element):

| Buffer | Size (double) | Size (float) | Cache Fit |
|--------|---------------|--------------|----------|
| Attention weight W (128×128) | 128 KB | 64 KB | L1 ✅ |
| MLP weight W (512×128) | 512 KB | 256 KB | L2 ✅ |
| KV cache per layer (512×128) | 512 KB | 256 KB | L2 ✅ |
| Input vector x (128) | 1 KB | 0.5 KB | L1 ✅ |
| Gradient buffer (875K params) | ~6.8 MB | ~3.4 MB | L3 ✅ |

> **Conclusion**: At N_EMBD=128, all hot buffers fit within L2 regardless of
> precision. Float mode halves memory footprint, improving cache utilisation
> for larger models. Cache-level optimisations become critical at N_EMBD ≥ 512
> where weight matrices exceed L1/L2 capacity.

### Baseline Performance (CPU-only, scalar)

| Operation | Throughput |
|-----------|-----------|
| `forward_backward_one` (1 position) | ~333K fwd+bwd/s |
| `forward_inference` (1 token) | ~960K tok/s |
| Adam step | ~194K steps/s |
| Full training step (seq=8) | ~353K tok/s |

---

## 3. Strategy 1 — SIMD Vectorisation

**CMake flag**: `MICROGPT_SIMD=ON` (default)

### Approach

Rather than hand-writing NEON intrinsics, we rely on the compiler's
auto-vectoriser by providing the right flags:

```
-march=native -funroll-loops -ffast-math
```

On Apple Silicon, `-march=native` enables NEON 128-bit SIMD (four 32-bit floats
or two 64-bit doubles per instruction). Combined with `-funroll-loops`, the
compiler generates 2-wide `fmla` (fused multiply-add) sequences for the inner
loops of `lin_fwd` and `lin_bwd`.

### Why Not Hand-Written NEON?

1. **Portability**: Hand-written intrinsics would lock us to ARM64.
2. **Compiler maturity**: Clang 15+ on AArch64 generates near-optimal NEON code
   for simple dot-product loops when given `-O3 -ffast-math`.
3. **Maintainability**: The inner loops in `lin_fwd` and `lin_bwd` are simple
   enough that the compiler can vectorise them without pragma annotations.

### Impact

Auto-vectorisation provides approximately 2× speedup over scalar `-O0` builds
for the 128×128 matrix-vector products that dominate `lin_fwd`.

---

## 4. Strategy 2 — Apple Accelerate BLAS

**CMake flag**: `MICROGPT_BLAS=ON` (default OFF)

### Approach

Replace the hand-rolled inner loops in `lin_fwd` and `lin_bwd` with calls to
Apple's Accelerate framework. The `CBLAS_GEMV` and `CBLAS_GER` macros auto-dispatch
to the correct precision (`cblas_dgemv`/`cblas_sgemv`) based on the `scalar_t` type:

```c
// lin_fwd: y = W @ x (dispatches to cblas_dgemv or cblas_sgemv)
CBLAS_GEMV(CblasRowMajor, CblasNoTrans,
           (int)nout, (int)nin, 1.0, W, (int)nin, x, 1, 0.0, y, 1);

// lin_bwd (dx): dx += W^T @ dy
CBLAS_GEMV(CblasRowMajor, CblasTrans,
           (int)nout, (int)nin, 1.0, W, (int)nin, dy, 1, 1.0, dx, 1);

// lin_bwd (dW): dW += dy ⊗ x
CBLAS_GER(CblasRowMajor,
          (int)nout, (int)nin, 1.0, dy, 1, x, 1, dW, (int)nin);
```

### Problem: Thread Contention

Apple Accelerate internally spawns its own thread pool for large matrices.
MicroGPT-C already uses a 12-thread training pool (one thread per batch element).
When 12 training threads each call `cblas_dgemv`, Accelerate attempts to
parallelise each call internally, causing:

1. **Thread oversubscription** — up to 12 × N_THREADS contention
2. **Cache thrashing** — each Accelerate thread touches different cache lines
3. **Lock overhead** — internal synchronisation dominates for small matrices

### Result

| Configuration | Training Throughput |
|--------------|-------------------|
| C loops + SIMD (12 threads) | ~353K tok/s |
| Accelerate BLAS (12 threads) | ~280K tok/s (slower) |
| Accelerate BLAS (1 thread, inference) | ~380K tok/s (slightly faster) |

> **Recommendation**: Use `MICROGPT_BLAS=ON` only for **single-threaded inference**
> where Accelerate's internal parallelism is beneficial. For multi-threaded
> training, the simple C loops with compiler auto-vectorisation are faster.

---

## 5. Strategy 3 — Metal GPU Compute

**CMake flag**: `MICROGPT_METAL=ON` (default OFF)

### Architecture

Three new files implement the Metal acceleration layer:

```
src/microgpt_metal.h      C API declarations
src/microgpt_metal.metal   Metal Shading Language compute kernels
src/microgpt_metal.m       Objective-C bridge (device/pipeline management)
```

### Compute Kernels

Three Metal compute kernels mirror the three operations in `lin_fwd`/`lin_bwd`:

```
┌─────────────────────────────────────────────────────┐
│ Kernel              │ Operation    │ Grid Dispatch   │
├─────────────────────┼──────────────┼─────────────────┤
│ lin_fwd_kernel      │ y = W @ x   │ nout threads    │
│ lin_bwd_dx_kernel   │ dx += Wᵀdy  │ nin threads     │
│ lin_bwd_dW_kernel   │ dW += dy⊗x  │ nout×nin threads│
└─────────────────────────────────────────────────────┘
```

Each thread computes one output element. The `lin_bwd_dW_kernel` uses a flattened
1D dispatch over the full `nout × nin` matrix to parallelise the outer product.

### Precision Constraint

Metal Shading Language does **not** support `double` (64-bit float). All GPU
buffers use `float` (32-bit). When `scalar_t` is already `float` (via
`-DMICROGPT_USE_FLOAT=ON`), no conversion is needed. In the default `double`
mode, the Objective-C bridge converts at the CPU/GPU boundary:

```
CPU (scalar_t=double) → [convert to float32] → GPU Metal Buffer
                                                 ↓
                                              Compute
                                                 ↓
CPU (scalar_t=double) ← [convert to float64] ← GPU Metal Buffer
```

With `MICROGPT_USE_FLOAT=ON`:
```
CPU (scalar_t=float) → [zero-copy] → GPU Metal Buffer → Compute → [zero-copy] → CPU
```

This means float mode eliminates the precision conversion overhead entirely,
making Metal dispatch more efficient.

### Problem: Dispatch Overhead

On M2 Max with 1024 threads/threadgroup, the Metal dispatch pipeline involves:

1. Command buffer creation (~5 μs)
2. Encoder setup & buffer binding (~10 μs)
3. Kernel dispatch & commit (~5 μs)
4. GPU scheduling & execution (~5 μs for 128×128)
5. Command buffer completion wait (~25 μs)

**Total overhead: ~50 μs per call**

The actual 128×128 matrix-vector multiply takes ~5 μs of compute time. The GPU
spends 90% of wall-clock time on dispatch overhead, not arithmetic.

### Crossover Analysis

```
Dispatch overhead ≈ 50 μs (fixed)

For GPU to win: GPU_compute_time + 50μs < CPU_compute_time

At N_EMBD=128:  CPU ≈ 3μs, GPU ≈ 5μs + 50μs = 55μs  → CPU wins
At N_EMBD=512:  CPU ≈ 45μs, GPU ≈ 12μs + 50μs = 62μs → ~break-even
At N_EMBD=1024: CPU ≈ 180μs, GPU ≈ 25μs + 50μs = 75μs → GPU wins
```

> **Recommendation**: Metal acceleration becomes worthwhile at N_EMBD ≥ 512.
> At current model size (128), keep `MICROGPT_METAL=OFF`. The infrastructure
> is ready for when the model scales.

---

## 6. Strategy 4 — Compiler Optimisation Flags

### Final Compiler Configuration

```cmake
set(RELEASE_FLAGS
    -O3            # Full optimisation
    -ffast-math    # Relax IEEE 754 (allow reordering, FMA fusion)
    -march=native  # Target current CPU (enables NEON on Apple Silicon)
    -flto          # Link-time optimisation (cross-TU inlining)
    -funroll-loops # Unroll small loops for SIMD width
    -DNDEBUG       # Disable assertions
)
```

### Flag Rationale

| Flag | Effect | Risk |
|------|--------|------|
| `-O3` | Aggressive inlining, vectorisation, scheduling | None for correctness |
| `-ffast-math` | Enables FMA, allows reassociation of FP ops | Non-deterministic rounding; acceptable for ML |
| `-march=native` | NEON SIMD, LSE atomics, hardware FP16 | Binary only runs on same arch |
| `-flto` | Allows inlining across `microgpt.c` → `main.c` boundary | Longer link times |
| `-funroll-loops` | Unrolls inner loops to fill SIMD pipeline | Slight code size increase |

### `-Ofast` Migration

The deprecated `-Ofast` flag is equivalent to `-O3 -ffast-math -ffinite-math-only`.
We use `-O3 -ffast-math` explicitly for clarity and forward compatibility.

---

## 7. Strategy 5 — Per-Head Attention Parallelism

**CMake flag**: `MICROGPT_HEAD_PARALLEL=ON` (default OFF)

### Motivation

In multi-head attention, each head operates on an independent `(N_EMBD/N_HEAD)`-
dimensional subspace. The N_HEAD iterations of the attention loop are
embarrassingly parallel — no data dependency between heads.

### Implementation

The parallelism uses POSIX threads via a portable threading layer
(`mgpt_thread_create`/`mgpt_thread_join`) that works on both macOS and Linux.

#### Data Structures

```c
typedef struct {
    const double *x;           // input activation
    const double *Wq, *Wk, *Wv;  // per-head projection weights
    double *keys, *vals;       // KV cache for this layer
    size_t *cache_len;         // current cache occupancy
    size_t pos;                // sequence position
    int head;                  // head index
    double *attn_out;          // output accumulator
    /* ... more fields ... */
} AttnHeadFwdArg;

typedef struct {
    /* Similar structure for backward pass */
} AttnHeadBwdArg;
```

#### Dispatch Pattern

```c
static void attn_heads_dispatch(void *(*fn)(void *), void *args,
                                size_t arg_size) {
    mgpt_thread_t threads[N_HEAD];
    for (int h = 0; h < N_HEAD; h++)
        mgpt_thread_create(&threads[h], fn,
                           (char *)args + h * arg_size);
    for (int h = 0; h < N_HEAD; h++)
        mgpt_thread_join(threads[h]);
}
```

Three call sites are parallelised via `#ifdef MICROGPT_HEAD_PARALLEL`:

1. **Forward attention** in `forward_backward_one`
2. **Backward attention** in `forward_backward_one`
3. **Inference attention** in `forward_inference`

### Thread Interaction Matrix

```
┌──────────────────────────────────────────────────────────┐
│ Parallelism Level       │ Threads │ Where              │
├─────────────────────────┼─────────┼────────────────────┤
│ Batch-level (training)  │ 12      │ main.c train loop  │
│ Head-level (attention)  │ 8       │ attn_heads_dispatch│
│ BLAS-internal           │ N       │ Accelerate/OpenBLAS│
└──────────────────────────────────────────────────────────┘
```

> **Warning**: Enabling both batch-level and head-level parallelism simultaneously
> creates 12 × 8 = 96 threads, causing severe oversubscription on a 12-core CPU.
> Use head-level parallelism for **single-threaded inference** only.
>
> **Measured impact (names_demo, N_HEAD=2, 4192 params)**: HEAD_PARALLEL=ON
> causes an **84× slowdown** in training (117 vs 9,879 steps/s) due to
> pthread create/join overhead dominating the sub-microsecond attention math.

### Verification

- Step 1 loss = **7.0775** — identical to serial baseline ✅
- 44/44 unit tests pass ✅
- No race conditions (each head writes to a disjoint output region)

---

## 8. Strategy 6 — Cache Tiling (Weight Blocking)

**Always active** (C fallback path; BLAS/Metal paths are unaffected)

### Motivation

The naive `lin_fwd` inner loop walks the full row of W for each output:

```c
for (j = 0; j < nout; j++) {       // nout rows
    double s = 0;
    for (i = 0; i < nin; i++)       // full row walk
        s += W[j * nin + i] * x[i];
    y[j] = s;
}
```

When `nin` is large (e.g., 1024), the x[] vector and W row compete for L1 cache
lines. At N_EMBD=128, this is not yet a problem (row = 1KB), but at N_EMBD=1024
(row = 8KB), multiple rows evict x[] from L1 between uses.

### Tiled Loop Nest

```c
#define LIN_TILE_R 32   // rows per tile
#define LIN_TILE_C 64   // columns per tile
// Panel = 32 × 64 × 8B = 16 KB → fits in L1 with room for x[] and y[]

for (j0 = 0; j0 < nout; j0 += TILE_R) {          // row tiles
    j1 = min(j0 + TILE_R, nout);
    for (i0 = 0; i0 < nin; i0 += TILE_C) {        // column tiles
        i1 = min(i0 + TILE_C, nin);
        for (j = j0; j < j1; j++) {
            double s = 0;
            const double *Wrow = W + j * nin + i0;
            for (i = 0; i < i1 - i0; i++)
                s += x[i0 + i] * Wrow[i];         // 16KB panel
            y[j] += s;                             // accumulate partial sums
        }
    }
}
```

### Cache Access Pattern

```
Naive:                          Tiled:
┌─────────────────────┐        ┌──────┬──────┬──────┐
│ Row 0 (full width)  │        │ T0,0 │ T0,1 │ T0,2 │  Panel 0,0: TILE_R rows
│ Row 1 (full width)  │        │      │      │      │  × TILE_C cols = 16KB
│ Row 2 (full width)  │        ├──────┼──────┼──────┤
│ ...                 │        │ T1,0 │ T1,1 │ T1,2 │  Process all 32 rows
│ Row N (full width)  │        │      │      │      │  within one 16KB panel
└─────────────────────┘        ├──────┼──────┼──────┤  before moving to next
                               │ T2,0 │ T2,1 │ T2,2 │  column tile.
                               └──────┴──────┴──────┘
```

The key insight: within each panel, the same 512B slice of x[] is reused across
32 rows. This maximises temporal locality for x[] while keeping the W panel
resident in L1.

### Backward Pass Tiling

The backward pass applies the same tiling to two operations:

**Gradient w.r.t. input (dx += Wᵀ @ dy)**:
```c
for (j0 = 0; j0 < nout; j0 += TILE_R) {
    for (i0 = 0; i0 < nin; i0 += TILE_C) {
        for (j = j0; j < j1; j++) {
            double dyj = dy[j];
            for (i = 0; i < i1 - i0; i++)
                dx[i0 + i] += dyj * W[j * nin + i0 + i];
        }
    }
}
```

**Gradient w.r.t. weights (dW += dy ⊗ x)**:
```c
for (j0 = 0; j0 < nout; j0 += TILE_R) {
    for (i0 = 0; i0 < nin; i0 += TILE_C) {
        for (j = j0; j < j1; j++) {
            double dyj = dy[j];
            double *dWrow = dW + j * nin + i0;
            for (i = 0; i < i1 - i0; i++)
                dWrow[i] += dyj * x[i0 + i];
        }
    }
}
```

### Tile Size Selection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `LIN_TILE_R` | 32 | Panel height: 32 output elements in registers |
| `LIN_TILE_C` | 64 | Panel width: 64 doubles per row = 512B |
| Panel size | 32 × 64 × 8B = 16 KB | Fits in L1 (128KB) with room for vectors |
| x[] slice | 64 × 8B = 512B | Reused 32 times per panel |
| y[] slice | 32 × 8B = 256B | Accumulator, stays in registers |

The tile sizes are `#ifndef`-guarded, allowing override at build time:
```bash
cmake -DLIN_TILE_R=16 -DLIN_TILE_C=32 ..  # Smaller tiles for smaller L1
```

### Tail Handling

Non-tile-aligned dimensions are handled by clamping the tile boundary:
```c
size_t j1 = (j0 + LIN_TILE_R < nout) ? j0 + LIN_TILE_R : nout;
size_t i1 = (i0 + LIN_TILE_C < nin)  ? i0 + LIN_TILE_C : nin;
```

This is verified by 5 dedicated unit tests covering:
- Non-aligned dimensions (37×73)
- Tile-aligned dimensions (128×128)
- Sub-tile dimensions (5×7)

### Verification

44/44 unit tests pass ✅, including 5 new tiled algebra tests that compare
the tiled implementation against a naive reference at < 1e-10 tolerance.

---

## 9. Strategy 7 — Paged KV Cache

**CMake flag**: `MICROGPT_PAGED_KV=ON` (default OFF)

### Background: KV Cache in Transformers

The KV cache is a fundamental inference-time optimisation. During autoregressive
generation (one token at a time), without a cache you'd recompute K and V
projections for *all* previous positions at every step — O(n²) total work.
The KV cache stores previously computed K/V vectors so each new token only
computes its own pair, reducing total work to O(n).

> **Important**: KV caching is primarily an **inference** optimisation.
> In standard training, all positions are processed in parallel via masked
> attention — no cache needed. MicroGPT's `forward_backward_one` happens to
> process positions sequentially (simpler implementation), so it uses a KV
> cache during training too.

### Flat Cache (Default)

```c
keys[L] = calloc(BLOCK_SIZE * N_EMBD, sizeof(double));  // pre-allocated
vals[L] = calloc(BLOCK_SIZE * N_EMBD, sizeof(double));
// Access: keys[L][t * N_EMBD + d]  — direct pointer arithmetic
```

Fixed capacity, contiguous arrays. Simple, fast, cache-friendly.

### Paged Cache (Opt-In)

```c
#define KV_PAGE_SIZE 64   // positions per page

typedef struct {
    double *data;          // KV_PAGE_SIZE × N_EMBD doubles
} KVPage;

typedef struct {
    KVPage **pages;        // page table: pages[page_idx]
    size_t n_pages;        // allocated pages
    size_t capacity;       // max page table slots
    size_t len;            // positions stored
} PagedKVCache;
```

Demand-paged: pages are allocated on first access, retained on reset for reuse.
Access goes through `paged_kv_get(cache, t)` which resolves `page_idx = t / 64`
and `slot = t % 64`.

### Abstraction Layer

All KV access is abstracted via `KV_WRITE`/`KV_READ` macros and portable
`kv_cache_alloc()`/`kv_cache_free()`/`kv_cache_reset()` helper functions.
Callers (demos, tests) use the same API regardless of cache mode:

```c
keys[L] = kv_cache_alloc();      // flat calloc or paged_kv_create
kv_cache_free(keys[L]);           // free or paged_kv_free
```

### Performance Reality: Overhead Dominates at Small Scale

| | Flat cache | Paged cache | Ratio |
|---|---|---|---|
| **Write access** | `arr + offset` (1 add) | `sync_append` → page lookup → slot (function call + div + mod) | ~10–20× slower |
| **Read access** | `arr + offset` (1 add) | `paged_kv_get` → page lookup (function call + div + mod) | ~5–10× slower |
| **Training step** | ~3 ms | ~30–100 ms | **~10–30× slower** |
| **Memory (16 positions)** | 512 KB/layer | 16 KB/layer | 97% saved |

The overhead comes from:
1. **Function call per access** — flat mode uses direct pointer arithmetic
2. **Integer division/modulo** — `t / KV_PAGE_SIZE` and `t % KV_PAGE_SIZE`
3. **Double indirection** — `pages[page_idx]->data[slot * N_EMBD]`
4. **Sync check** — `paged_kv_sync_append` compares `c->len` vs `cache_len[L]`
   on every write
5. **Cache misses** — page table indirection defeats spatial locality

> At N_EMBD=128 with BLOCK_SIZE=64, the flat cache is 8 KB per layer — trivially
> fits in L1 cache. The paged cache's memory savings are meaningless at this scale,
> but its per-access overhead is paid thousands of times per training step.

### When Paged KV Cache Is Valuable

The paged design becomes worthwhile when:
- **Context windows are very large** (4K–128K tokens) — pre-allocating contiguous
  blocks wastes GB of RAM
- **Serving many concurrent requests** — virtual memory flexibility matters
  (the vLLM approach)
- **Beam search** — pages can be shared across beams via copy-on-write
- **Dynamic sequences** — no fixed `BLOCK_SIZE` limit

### Verification

- 48/48 unit tests pass ✅ (44 existing + 4 paged KV tests)
- All 3 demos work with both flat and paged modes ✅
- AddressSanitizer clean ✅

---

## 10. Strategy 8 — `scalar_t` Precision Abstraction

**CMake flag**: `MICROGPT_USE_FLOAT=ON` (default OFF → `double`)

### Motivation

The original implementation hardcoded `double` (64-bit) for all weights,
activations, gradients, and KV cache. While `double` provides excellent
numerical stability, it has two costs:

1. **SIMD width**: ARM NEON processes 2 doubles vs 4 floats per instruction
2. **Memory bandwidth**: 8 bytes vs 4 bytes per element — doubles the working
   set for weights and gradients

For models that fit comfortably in cache (N_EMBD ≤ 256), the SIMD throughput
doubling from float is the primary benefit.

### Implementation

A compile-time `typedef` controlled by the `MICROGPT_USE_FLOAT` preprocessor flag:

```c
#ifdef MICROGPT_USE_FLOAT
  typedef float  scalar_t;
  #define M_EXP   expf
  #define M_LOG   logf
  #define M_SQRT  sqrtf
  #define M_POW   powf
  #define M_FABS  fabsf
  #define CBLAS_GEMV cblas_sgemv
  #define CBLAS_GER  cblas_sger
#else
  typedef double scalar_t;
  #define M_EXP   exp
  #define M_LOG   log
  #define M_SQRT  sqrt
  #define M_POW   pow
  #define M_FABS  fabs
  #define CBLAS_GEMV cblas_dgemv
  #define CBLAS_GER  cblas_dger
#endif
```

### Scope of Changes

| Component | `double` → `scalar_t` | Kept as `double` |
|-----------|----------------------|------------------|
| Model weights & biases | ✅ | |
| Activations & gradients | ✅ | |
| KV cache arrays | ✅ | |
| Math library calls | ✅ (via macros) | |
| BLAS calls | ✅ (via macros) | |
| Adam hyperparameters (lr, β₁, β₂, ε) | | ✅ (stability) |
| Loop counters, sizes, token IDs | | ✅ (integer types) |

Optimiser hyperparameters are deliberately kept as `double` to avoid float32
accumulation drift in the Adam moment estimates over thousands of steps.

### Test Tolerance

Tests use a precision-aware tolerance constant:

```c
#ifdef MICROGPT_USE_FLOAT
  #define SCALAR_TOL 1e-5
#else
  #define SCALAR_TOL 1e-10
#endif
```

All 44 unit tests pass with both `double` and `float` builds.

### Impact

| Metric | `double` | `float` | Ratio |
|--------|----------|---------|-------|
| Weight memory (875K params) | ~6.8 MB | ~3.4 MB | **2× smaller** |
| NEON SIMD width | 2-wide | 4-wide | **2× throughput** |
| Metal GPU conversion | float↔double | zero-copy | **No overhead** |
| Numerical precision | ~15 digits | ~7 digits | Acceptable for ML |

### Verification

- 44/44 unit tests pass with `double` ✅
- 44/44 unit tests pass with `float` ✅
- All 6 build targets compile clean in both modes ✅

---

## 11. Strategy 9 — Common Training Code Extraction

**Always active** (library-level refactoring)

### Motivation

All three demo programs (names, shakespeare, c_codegen) independently defined
identical copies of:

1. `shuffle_docs()` — Fisher-Yates document shuffling
2. `WorkerArg` struct — per-thread training state
3. `train_batch_worker()` — thread entry point for batched training

This ~180 lines of duplication created a maintenance burden: any bug fix or
`scalar_t` migration had to be applied to all three files independently.

### Implementation

The shared code was extracted into the core library:

| Original (per-demo) | Library API | Location |
|---------------------|------------|----------|
| `shuffle_docs()` | `shuffle_docs(Docs *docs)` | `microgpt.c` |
| `WorkerArg` struct | `TrainWorker` struct | `microgpt.h` |
| `train_batch_worker()` | `train_worker_run(void *arg)` | `microgpt.c` |
| `static rand_u()` | `rand_u()` (public) | `microgpt.c` |

The `TrainWorker` struct contains all per-thread state:

```c
typedef struct {
  const Model *model;
  const Docs  *docs;
  const Vocab *vocab;
  scalar_t *grads;
  scalar_t *keys[N_LAYER];
  scalar_t *values[N_LAYER];
  size_t    cache_len[N_LAYER];
  size_t    token_buf[BLOCK_SIZE + 2];
  int       batch_start, batch_end;
  scalar_t  loss;
  size_t    positions;
  unsigned int rng_seed;
} TrainWorker;
```

### Impact

| Metric | Before | After |
|--------|--------|-------|
| Lines of duplicated code | ~180 (60 × 3 demos) | 0 |
| `scalar_t` migration points | 3 files | 1 file |
| Bug fix propagation | Manual × 3 | Automatic |

---

## 12. Build Matrix & CMake Flags

### Available Flags

| Flag | Default | Description |
|------|---------|-------------|
| `MICROGPT_SIMD` | ON | Enable `-march=native -funroll-loops` for SIMD |
| `MICROGPT_BLAS` | OFF | Use Accelerate/OpenBLAS for matmul (auto-dispatched for `scalar_t`) |
| `MICROGPT_METAL` | OFF | GPU acceleration via Metal compute shaders |
| `MICROGPT_HEAD_PARALLEL` | OFF | Per-head attention threading |
| `MICROGPT_PAGED_KV` | OFF | Paged KV cache (memory opt, trades speed) |
| `MICROGPT_USE_FLOAT` | OFF | Use `float` instead of `double` for `scalar_t` |

### Recommended Configurations

#### Multi-threaded Training (default)
```bash
cmake -DMICROGPT_SIMD=ON ..
```
Best for training with batch-level parallelism on multi-core CPUs.
BLAS, Metal, head parallelism, and paged KV are counterproductive at this scale.

#### Single-threaded Inference
```bash
cmake -DMICROGPT_BLAS=ON -DMICROGPT_HEAD_PARALLEL=ON ..
```
BLAS benefits from no thread contention. Head parallelism provides
intra-attention concurrency.

#### Large Model (N_EMBD ≥ 512)
```bash
cmake -DMICROGPT_METAL=ON -DMICROGPT_SIMD=ON -DMICROGPT_PAGED_KV=ON ..
```
GPU dispatch overhead amortised over larger matrices. Metal becomes
the fastest path. Paged KV cache becomes worthwhile for memory savings.

#### Float Precision
```bash
cmake -DMICROGPT_USE_FLOAT=ON -DMICROGPT_SIMD=ON ..
```
Doubles NEON SIMD throughput (4-wide vs 2-wide) and halves memory footprint.
Excellent for memory-constrained devices or when double precision isn't needed.

### Dispatch Priority

When multiple backends are enabled, `lin_fwd`/`lin_bwd` select the backend
via preprocessor priority:

```
#if defined(MICROGPT_METAL)
    → Metal GPU path (float32 shaders; zero-copy when scalar_t=float)
#elif defined(MICROGPT_BLAS)
    → CBLAS_GEMV / CBLAS_GER (auto-dispatched: sgemv for float, dgemv for double)
#else
    → C fallback with tiled loops (scalar_t precision)
#endif
```

---

## 13. Empirical Results & Analysis

### Benchmark Summary (N_EMBD=128, M2 Max)

| Operation | C+SIMD | BLAS | Metal |
|-----------|--------|------|-------|
| `forward_inference` (1 tok) | **960K/s** | ~380K/s* | ~18K/s |
| `forward_backward_one` | **333K/s** | ~260K/s | ~15K/s |
| `adam_step` | 194K/s | 194K/s | N/A |
| Training step (seq=8) | **353K tok/s** | ~280K/s | — |

*BLAS single-threaded inference only

### Naive vs Tiled `lin_fwd` (Benchmark)

The `bench_tiled_matmul` benchmark compares naive and tiled matrix-vector
multiply across four matrix sizes:

| Size | Description | Where Tiling Helps |
|------|-------------|-------------------|
| 32×32 | Sub-tile (all in L1) | No benefit — already fits |
| 128×128 | Current N_EMBD | Marginal — fits L1 boundary |
| 512×512 | Future model size | Significant — exceeds L1 |
| 73×97 | Non-aligned edge case | Tests tail handling |

### Scaling Projection

```
Performance model:

Time_naive  ∝ nout × nin            (streaming, cold x[])
Time_tiled  ∝ nout × nin / TILE_C   (x[] reuse within tile)
Speedup     ≈ TILE_C / (cache_miss_penalty / compute_ratio)

At N_EMBD=128:  Speedup ≈ 1.0-1.2x (matrix fits L1 regardless)
At N_EMBD=512:  Speedup ≈ 1.5-2.0x (rows exceed L1, tiling keeps panel hot)
At N_EMBD=1024: Speedup ≈ 2.0-3.0x (significant L1 miss reduction)
```

---

## 14. Conclusions & Recommendations

### What Worked

1. **Compiler auto-vectorisation** (`-O3 -ffast-math -march=native`) — the single
   highest-impact optimisation, essentially free
2. **Cache tiling** — infrastructure investment that pays off at larger model sizes
3. **Per-head parallelism** — clean, correct implementation for inference speedup

### What Didn't Work (At This Scale)

1. **BLAS** — thread pool contention with batch-level parallelism negates benefits
2. **Metal GPU** — dispatch overhead (50 μs) dwarfs compute time (5 μs) for 128×128
3. **Paged KV cache** — page table indirection adds ~10–30× overhead per KV access;
   memory savings are negligible when flat cache already fits in L1
4. **Combined parallelism** — head + batch threading causes oversubscription

### Key Insight

> For small matrices (128×128), **CPU with NEON SIMD beats GPU by 50×**.
> The crossover point where GPU offloading becomes profitable is approximately
> N_EMBD ≈ 512, where the matrix-vector product takes long enough to amortise
> the fixed ~50 μs dispatch cost. Below this threshold, the simplest approach
> (compiler vectorisation of tight C loops) is also the fastest.

### Future Work

| Priority | Optimisation | Expected Impact |
|----------|-------------|----------------|
| High | Batched Metal dispatch | Amortise GPU overhead across multiple `lin_fwd` calls |
| Medium | INT8 quantisation | 4× memory reduction, 2× throughput via NEON `smull` |
| Medium | Float benchmarking suite | Quantify NEON 4-wide vs 2-wide throughput gains systematically |
| Low | Custom NEON kernels | Diminishing returns vs auto-vectorisation |

---

*Document generated February 2026. Benchmark numbers measured on Apple M2 Max,
macOS Sonoma, Apple Clang 15, with `-O3 -ffast-math -march=native -flto`.*
