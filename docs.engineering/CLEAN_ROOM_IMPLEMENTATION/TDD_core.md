# TDD_core — Technical Design Document

**Document ID:** TDD-CORE-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_core.md`
**Sources:** `src/microgpt.h`, `src/microgpt.c`

---

## 1. Overview

The core engine is a decoder-only GPT-2-style transformer plus an Adam optimiser, KV cache, and threading harness, implemented in two C99 files (`microgpt.h`, `microgpt.c`). The engine is configured via compile-time `#define` macros so the inner loops constant-fold and unroll under `-O3`.

Single-header / two-file layout keeps the engine embeddable: a downstream project drops the two files into its build, with no other dependencies than `libc` + `libm`.

## 2. Architecture

```
   demos/main.c
      │
      ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  microgpt.h  (public API)                                        │
 │   • MicrogptConfig, microgpt_default_config()                    │
 │   • Docs, Vocab, WordVocab, Model (opaque)                       │
 │   • model_create / _free / _save / _load                         │
 │   • forward_backward_one, forward_inference, adam_step           │
 │   • sample_token, seed_rng, rand_u                               │
 │   • TrainWorker harness, mgpt_thread_* portability               │
 └──────────────────────────────────────────────────────────────────┘
      │
      ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  microgpt.c  (~3,600 lines)                                      │
 │   §1 Data loading + char tokeniser                               │
 │   §2 Model allocation + Gaussian init                            │
 │   §3 Serialisation (model_save/load, checkpoint_save/load)       │
 │   §4 Primitives (lin_fwd/bwd, rmsnorm_fwd/bwd, RoPE helpers)     │
 │   §5 forward_backward_one                                        │
 │   §6 forward_inference                                           │
 │   §7 adam_step (Adam + cosine LR + warmup + AdamW + clip)        │
 │   §8 sample_token (temperature softmax)                          │
 │   §9 Word tokeniser (frequency-ranked, hash-table lookup)        │
 │   §10 TrainWorker entry point + shuffle_docs + rand_u            │
 └──────────────────────────────────────────────────────────────────┘
```

## 3. Data flow

### Training

```
 docs ─▶ tokenize() ─▶ ids[]
   │                     │
   ▼                     ▼
 batch loop         per token loop
   │                     │
   ▼                     ▼
 forward_backward_one(model, ids[t], t, ids[t+1], keys, values, cache_len, grads)
   │
   ├─ embed (wte[ids[t]] + wpe[t])
   ├─ N_LAYER × { rmsnorm → attn(Q/K/V/O, KV cache append) → +residual → rmsnorm → mlp(fc1, ReLU, fc2) → +residual }
   ├─ rmsnorm
   ├─ lm_head → logits
   ├─ softmax → CE loss vs target
   └─ backward (accumulate grads)
       ▼
 adam_step(model, grads, m, v, step)
   ▲
   └─ cosine LR with linear warmup, AdamW (decoupled weight decay), gradient clip
```

### Inference

Single token in, logits out; KV cache appended in place. `forward_inference` skips the loss + backward path. `sample_token` draws a categorical sample under temperature scaling.

## 4. Key data structures

### 4.1 `Model`

A heap-allocated struct holding all weight matrices as separate per-matrix `scalar_t *` (or `int8_t *` plus per-matrix scales when `QUANTIZATION_INT8` is active). Matrices:

- `wte` — token embedding `[vocab_size × N_EMBD]`, row-major.
- `wpe` — position embedding `[BLOCK_SIZE × N_EMBD]`.
- `lm_head` — output projection `[vocab_size × N_EMBD]`.
- Per-layer: `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`, each `[N_EMBD × N_EMBD]`; `mlp_fc1` `[MLP_DIM × N_EMBD]`; `mlp_fc2` `[N_EMBD × MLP_DIM]`.
- (Optional) `attn_res_proj`, `mlp_res_proj` per layer when `MICROGPT_ATTN_RES`.

`model_num_params` returns the cumulative sizes; this drives the gradient/Adam buffer length.

### 4.2 KV cache

Two layouts share the helper API `kv_cache_alloc / free / reset / copy`:

- **Flat**: a contiguous `scalar_t[BLOCK_SIZE × N_EMBD]` buffer per layer. The hot path's array indexing is constant-folded by the compiler.
- **Paged** (`MICROGPT_PAGED_KV`): a `PagedKVCache` struct with a page table; pages are allocated lazily as the sequence grows. Page size is `KV_PAGE_SIZE` (default 64 positions).

The flat layout dominates the small-model regime; paged is for when `BLOCK_SIZE × N_EMBD × scalar_t` becomes too large to pre-allocate.

### 4.3 Gradient and Adam buffers

Flat `scalar_t[]` arrays of length `model_num_params`, laid out in the order documented in the `microgpt.c` comment "MEMORY LAYOUT (flat gradient / Adam buffers)":

```
[ wte | wpe | lm_head | (per-layer: wq, wk, wv, wo, fc1, fc2) | (optional: attn_res, mlp_res) ]
```

This ordering is shared with the checkpoint format (see `FS_checkpoint.md`).

## 5. Algorithms

### 5.1 Linear forward / backward

`lin_fwd(x, W, nin, nout, y)` computes `y[j] = sum_i W[j*nin + i] * x[i]`. The implementation uses cache tiling (`LIN_TILE_R × LIN_TILE_C`) to keep each panel in L1; on M2 Max with `LIN_TILE_R=32`, `LIN_TILE_C=64` the working set is ~16 KB per tile out of 128 KB L1.

`lin_bwd(x, W, dy, nin, nout, dx, dW)` computes `dx = W^T @ dy` and `dW += dy ⊗ x` accumulating into shared buffers. Both are dispatched through the `CBLAS_GEMV` / `CBLAS_GER` macros (single or double precision, BLAS-accelerated when `MICROGPT_BLAS`) and to `metal_lin_fwd` / `metal_lin_bwd` when `MICROGPT_METAL` is enabled and active.

### 5.2 RMSNorm

`rmsnorm_fwd(x, n, eps, out)` divides each element by the root-mean-square plus epsilon. There are no learnable affine parameters in this engine (a deliberate simplification matching Karpathy's reference). The backward `rmsnorm_bwd` is the closed-form derivative.

### 5.3 Multi-head causal attention

For each layer L and head h, the engine computes:

```
Q = lin_fwd(x, attn_wq[L])
K = lin_fwd(x, attn_wk[L])      (cached at position pos_id)
V = lin_fwd(x, attn_wv[L])      (cached at position pos_id)
scores[t] = Q · K[t] / sqrt(head_dim)    for t = 0..pos_id
softmax over t (causal mask is implicit by only iterating up to pos_id)
y_h = sum_t softmax[t] * V[t]
y = concat across heads
y = lin_fwd(y, attn_wo[L])
```

When `MICROGPT_QK_NORM` is enabled, Q and K are RMSNorm'd per-head before the dot product (the cache stores post-norm K so all positions are consistent). When `MICROGPT_PARTIAL_ROPE` is enabled, the last `ROPE_DIMS` of Q and K are RoPE-rotated (cos/sin tables are precomputed at model creation, sized `BLOCK_SIZE × ROPE_DIMS/2`). When `MICROGPT_ATTN_SINK` is enabled, the softmax denominator gains an extra `exp(ATTN_SINK_LOGIT)` term.

The backward pass walks the same loop in reverse; each optional flag has a closed-form gradient (no extra trainable parameters except `wpe` rotation tables which are constants).

### 5.4 MLP

Two-layer with `ReLU` activation: `h = ReLU(fc1 @ x); y = fc2 @ h`. The hidden dimension is `MLP_DIM` (4× `N_EMBD` in typical configs but freely configurable).

### 5.5 Adam with cosine LR + warmup + AdamW + clip

```
lr_t = LEARNING_RATE × min(t / WARMUP_STEPS, 0.5 × (1 + cos(π × (t − WARMUP_STEPS) / (NUM_STEPS − WARMUP_STEPS))))
m_t = β1 × m_{t−1} + (1 − β1) × g_t
v_t = β2 × v_{t−1} + (1 − β2) × g_t²
m̂_t = m_t / (1 − β1^t)
v̂_t = v_t / (1 − β2^t)
θ_t = θ_{t−1} − lr_t × (m̂_t / (sqrt(v̂_t) + EPS_ADAM) + WEIGHT_DECAY × θ_{t−1})
```

`WEIGHT_DECAY` is applied to all matrices except `wte` and `wpe`. `EPS_ADAM`, `BETA1`, `BETA2` remain `double` regardless of `scalar_t` for stability. Optional global gradient clipping (`clip_gradients`) rescales all gradients if their L2 norm exceeds `GRAD_CLIP`.

When `QUANTIZATION_INT8` is active, Adam updates an internal fp64 master copy; after each step the master is requantised back to `int8` per-matrix.

### 5.6 Sampling

`sample_token(logits, vocab_size, temperature)`:

1. Find max logit (numerical-stability shift).
2. `softmax(logits / T)`.
3. Cumulative-distribution sampling by drawing `u ~ Uniform[0,1)` from `rand_u()` and finding the first cumulative weight exceeding `u`.

### 5.7 Word tokeniser

`build_word_vocab` scans text in two passes:

1. Pass 1: hash every whitespace-delimited word into a temporary count table.
2. Pass 2: keep the top-N by frequency, assign IDs `[0..N-1]`. Append `<unk>` (N), newline (N+1), `<bos>` (N+2).

The hash table for `word_to_id` is open-addressed; capacity is rounded to the next power of two and sized so the load factor stays below ~0.5 for O(1) lookup.

`tokenize_words` splits on whitespace, treating newlines as the dedicated `newline_id` token. Out-of-vocabulary words map to `unk_id`.

## 6. Concurrency model

Training uses `TrainWorker` + `train_worker_run` — each worker owns its own gradient buffer, KV cache, token buffer, and RNG seed. Workers process disjoint slices of a batch; the main thread aggregates gradients between batches.

Inference is single-threaded by default. `MICROGPT_HEAD_PARALLEL` parallelises attention heads across threads at the cost of dispatch overhead — useful only for `N_EMBD ≥ 256`.

The portable thread API (`mgpt_thread_create`, `mgpt_thread_join`, `mgpt_cpu_count`, `mgpt_default_threads`) wraps `pthreads` on POSIX and `_beginthreadex` (with a `unsigned __stdcall` trampoline) on Windows. `clock_gettime(CLOCK_MONOTONIC, ...)` is provided as a polyfill on platforms that lack it.

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Model dimensions | Compile-time `#define` macros | Runtime config-driven loops | Constant-folding saves 10–30 % per-token cost on the small-model regime; `_microgpt_lib_for_defines` makes the build complexity manageable. |
| `scalar_t` | `float` default, `double` opt-in | `float` only | Double remains useful for research / gradient comparison against PyTorch; `SCALAR_TOL` auto-adjusts test tolerances. |
| Architecture variant | Pre-compute via library variants in CMake | Single library + runtime dispatch | Single-library + runtime dispatch loses the constant-fold benefit. |
| KV cache | Flat (default), paged (opt-in) | Always paged | Page-table overhead is measurable in the 1–10 µs per-call regime where small models live. |
| Activation | ReLU | GELU / SwiGLU | Karpathy reference parity; SwiGLU is documented as "future work" in the book. |
| Tokenisation | Char or word | BPE / SentencePiece | BPE / SentencePiece add corpus-build complexity and a runtime dependency; char + word covers the demos. |
| Sampling | Temperature-scaled softmax | Top-k / Top-p / Nucleus | Top-k / nucleus are doable in user code on top of `forward_inference`; not in the core API. |

## 8. Known limitations

- The model is autoregressive only; bidirectional encoding is not supported.
- `BLOCK_SIZE` is compile-time; sequences longer than `BLOCK_SIZE` MUST use MSA (`microgpt_msa.h`) on top of the engine.
- Training on the full 30 K-step Shakespeare run does not exercise label smoothing or weight decay by default; users opt in via `LABEL_SMOOTH` / `WEIGHT_DECAY`.
- INT8 mode does not support checkpoint save/load.
- Endianness is not handled in serialisation (see `FS_checkpoint.md` § 2).

## 9. History

The engine started as a C99 port of Karpathy's `microgpt.py`. Subsequent iterations added: word tokenisation; threading harness; weight transfer; model soup; KV-cache copy; paged KV cache; INT8; Apple Metal bridge; Block Attention Residuals; the four DeepSeek-V4 ports (Partial RoPE, Attention Sink, Q/K RMSNorm, Content-aware MSA pool). Each addition is gated by an `#ifdef` so the zero-deps baseline remains the default.

## 10. References

- Karpathy, "microgpt.py" reference implementation.
- Su et al., "RoFormer" (RoPE) 2021.
- Xiao et al., "StreamingLLM" 2024 (attention sink).
- DeepSeek-V4 §2.3.3 for partial RoPE / attention sink / Q/K RMSNorm.
- Wortsman et al., "Model soups" ICML 2022.

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
