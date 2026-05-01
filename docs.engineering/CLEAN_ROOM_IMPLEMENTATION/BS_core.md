# BS_core — Behaviour Specification

**Document ID:** BS-CORE-001
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30
**Replaces:** none

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

This document specifies the behavioural contract of the **core transformer engine**: model lifecycle, training (`forward_backward_one`, `adam_step`), inference (`forward_inference`, `sample_token`), KV cache, and the multi-threaded training harness. The scope is everything declared in `microgpt.h` outside the tokeniser (which has its own BS).

Out of scope:
- Tokenisation (covered by `BS_tokeniser.md`).
- Checkpoint serialisation (covered by `FS_checkpoint.md`).
- Organelle scaffolding (covered by `BS_organelle.md`).
- MSA / quantisation / Pipeline IR / VM (their own BSes).

## 2. Type contracts

### 2.1 `MicrogptConfig`

**Purpose:** Holds runtime hyperparameters mirroring the compile-time `#define` macros for introspection and banner display.

**Invariants:**
- INV-CORE-001: `cfg->n_embd > 0`, `cfg->n_head > 0`, `cfg->n_embd % cfg->n_head == 0`. `head_dim = n_embd / n_head`.
- INV-CORE-002: `cfg->n_layer >= 1`, `cfg->block_size >= 1`, `cfg->mlp_dim >= 1`.
- INV-CORE-003: `cfg->learning_rate > 0` and is a `double` regardless of `scalar_t`.
- INV-CORE-004: `cfg->warmup_steps <= cfg->num_steps`.
- INV-CORE-005: `cfg->max_vocab >= vocab_size` for any `Model` created against the config.
- INV-CORE-006: `microgpt_default_config()` returns a struct populated from the compile-time constants `N_EMBD`, `N_HEAD`, `N_LAYER`, `BLOCK_SIZE`, `MLP_DIM`, `NUM_STEPS`, `LEARNING_RATE`, `BATCH_SIZE`, `WARMUP_STEPS`, `TEMPERATURE`, `MAX_VOCAB`, `MAX_DOCS`, `MAX_DOC_LEN`.

**Construction:** Callers SHOULD use `microgpt_default_config()` and override fields per demo. The struct is pass-by-value in most APIs; addresses are taken only for read-only inspection.

### 2.2 `Model`

**Purpose:** Opaque handle to a heap-allocated transformer.

**Invariants:**
- INV-CORE-010: `model_create(vocab_size, cfg)` SHALL return a model whose internal config compares equal to `*cfg` for every field, accessible via `model_config(m)`.
- INV-CORE-011: `model_num_params(m)` SHALL return the cumulative size of all weight matrices in `scalar_t` units.
- INV-CORE-012: After `model_free(m)`, the pointer SHALL NOT be dereferenced (lifetime ends).
- INV-CORE-013: All weight matrices SHALL be initialised by `model_create` with samples drawn from `N(0, INIT_STD²)` where `INIT_STD = 0.08` by default.
- INV-CORE-014: `microgpt_verify_config(cfg)` SHALL return 0 if every `cfg->*` field matches the compile-time constants, and -1 otherwise.

**Thread safety:** A `Model *` is read-only during inference; multiple threads MAY call `forward_inference` against the same model with disjoint KV caches. During training, `adam_step` mutates the model and MUST run with no concurrent forward / backward / inference operations against the same model.

## 3. Operation contracts

### 3.1 `model_create`

**Signature:** `Model *model_create(size_t vocab_size, const MicrogptConfig *cfg)`

**Preconditions:** `cfg` is non-NULL; `microgpt_verify_config(cfg) == 0`; `vocab_size <= cfg->max_vocab`.

**Postconditions:** Returns a heap-allocated model whose weights are Gaussian-initialised. Returns NULL on allocation failure.

**Errors:** ERR-CORE-001 (`NULL` on OOM).

### 3.2 `forward_backward_one`

**Signature:** `scalar_t forward_backward_one(const Model *model, size_t token_id, size_t pos_id, size_t target_id, scalar_t **keys, scalar_t **values, size_t *cache_len, scalar_t *grad_buffer)`

**Preconditions:**
- `pos_id < cfg->block_size`.
- `cache_len[L]` for each layer L SHALL be the count of cached positions, with `cache_len[L] == pos_id` on entry.
- `keys[L]` and `values[L]` SHALL each be allocated by `kv_cache_alloc(cfg)` for capacity ≥ `cfg->block_size × cfg->n_embd`.
- `grad_buffer` SHALL be ≥ `model_num_params(model)` `scalar_t` elements; the caller is responsible for zeroing it at the start of each training step.

**Postconditions:**
- Returns the cross-entropy loss for predicting `target_id` from the input.
- Appends K and V vectors at `cache_len[L]` and increments `cache_len[L]` by 1.
- Accumulates gradients into `grad_buffer` (does NOT overwrite).

**Errors:** None — the function is total over the precondition. Behaviour is undefined if a precondition is violated.

### 3.3 `forward_inference`

**Signature:** `void forward_inference(const Model *model, size_t token_id, size_t pos_id, scalar_t **keys, scalar_t **values, size_t *cache_len, scalar_t *logits_out)`

**Preconditions:** As `forward_backward_one`, but `target_id` is absent and `grad_buffer` is replaced by `logits_out` of length `vocab_size` `scalar_t`.

**Postconditions:**
- Writes raw next-token logits into `logits_out`.
- Appends K, V at `cache_len[L]`; increments `cache_len[L]`.

**Errors:** None.

### 3.4 `adam_step`

**Signature:** `void adam_step(Model *model, const scalar_t *grads, scalar_t *m, scalar_t *v, int step)`

**Preconditions:**
- `grads`, `m`, `v` are each ≥ `model_num_params(model)` `scalar_t`.
- `step` is the current training step (0-indexed).

**Postconditions:**
- Updates `model`'s weights using Adam with bias correction, cosine LR with linear warmup (`WARMUP_STEPS` then cosine decay over `NUM_STEPS - WARMUP_STEPS`), optional decoupled weight decay (`WEIGHT_DECAY`) on all matrices except `wte` / `wpe`, and optional global gradient clipping (`GRAD_CLIP`).
- When `QUANTIZATION_INT8` is active, updates the internal fp64 master copy and requantises every weight matrix to int8.

**Errors:** None.

### 3.5 `sample_token`

**Signature:** `size_t sample_token(const scalar_t *logits, size_t vocab_size, scalar_t temperature)`

**Preconditions:** `logits` non-NULL; `vocab_size > 0`; `temperature > 0`.

**Postconditions:** Returns a token ID in `[0, vocab_size)` drawn from the softmax distribution `softmax(logits / temperature)`. Numerical-stability max-subtract is applied before exponentiation.

**Errors:** None.

### 3.6 KV cache helpers

`kv_cache_alloc(cfg)`, `kv_cache_free(kv)`, `kv_cache_reset(kv, cfg)`, `kv_cache_copy(src, dst, cfg, positions)` form a closed family. The contracts are:

- `kv_cache_alloc(cfg)` SHALL return a buffer suitable for caching `cfg->block_size` positions × `cfg->n_embd` `scalar_t`s.
- `kv_cache_free(kv)` SHALL free the buffer.
- `kv_cache_reset(kv, cfg)` SHALL reset internal state so a subsequent forward pass starts at position 0; it MAY retain pages (paged mode) for reuse.
- `kv_cache_copy(src, dst, cfg, positions)` SHALL copy the first `positions` cached entries from `src` to `dst`.

When `MICROGPT_PAGED_KV` is active, the returned pointer is internally a `PagedKVCache *` cast to `scalar_t *`; the engine's KV macros handle either layout transparently (INV-CORE-020).

### 3.7 Multi-threaded training harness

`TrainWorker` + `train_worker_run` form the canonical multi-thread training loop:

- INV-CORE-030: Each `TrainWorker` SHALL own its own `grads`, `keys[]`, `values[]`, `cache_len[]`, `token_buf[]`, and `rng_seed`.
- INV-CORE-031: `train_worker_run(arg)` SHALL process documents in `[batch_start, batch_end)` and accumulate `loss` and `positions` into the worker's own struct.
- INV-CORE-032: The caller SHALL aggregate `loss` and `grads` across workers before calling `adam_step`.

## 4. Invariants table

| ID | Invariant | Scope |
|---|---|---|
| INV-CORE-001 | `n_embd % n_head == 0`; `head_dim = n_embd / n_head`. | type |
| INV-CORE-002 | `n_layer ≥ 1`, `block_size ≥ 1`, `mlp_dim ≥ 1`. | type |
| INV-CORE-003 | Optimiser hyperparameters (`learning_rate`, `BETA1`, `BETA2`, `EPS_ADAM`) are double regardless of `scalar_t`. | system |
| INV-CORE-004 | `warmup_steps ≤ num_steps`. | type |
| INV-CORE-005 | `max_vocab ≥ vocab_size` for any model. | type |
| INV-CORE-006 | `microgpt_default_config()` is populated entirely from the compile-time constants. | operation |
| INV-CORE-010 | `model_create` returns a model whose `model_config(m)` reports the input config exactly. | operation |
| INV-CORE-011 | `model_num_params(m)` is the cumulative weight count. | operation |
| INV-CORE-013 | Initial weights are `N(0, 0.08²)`. | operation |
| INV-CORE-014 | `microgpt_verify_config` returns 0 only when runtime config matches compile-time constants. | operation |
| INV-CORE-020 | `kv_cache_alloc` accommodates `block_size × n_embd` positions × layers; the returned pointer transparently abstracts flat vs paged layouts. | operation |
| INV-CORE-030 | Each TrainWorker owns its mutable training scratch state. | concurrency |
| INV-CORE-031 | `train_worker_run` accumulates only into its own struct. | concurrency |
| INV-CORE-032 | Cross-worker aggregation happens between batches, not inside `train_worker_run`. | concurrency |
| INV-CORE-040 | When `MICROGPT_QK_NORM` is enabled, training SHALL be stable at `LR=0.02` (un-normed baseline diverges to PPL 731; with the flag PPL stays ≤ 205). | NFR |
| INV-CORE-041 | All four DeepSeek-V4 ports SHALL default OFF. Each SHALL be enabled independently and SHALL compose orthogonally with the others. | NFR |

## 5. Errors

| ID | Name | Conditions | Client action |
|---|---|---|---|
| ERR-CORE-001 | OOM_MODEL | `model_create` could not allocate weight matrices | Free other state and retry; reduce architecture |
| ERR-CORE-002 | CONFIG_MISMATCH | `microgpt_verify_config(cfg) != 0` | Pass a config produced by `microgpt_default_config()` or rebuild with matching `-D` flags |

The training and inference functions are total over their preconditions; behaviour is undefined if a precondition is violated.

## 6. Concurrency

Training and inference SHALL NOT run concurrently against the same model. The thread API (`mgpt_thread_create`, `mgpt_thread_join`, `mgpt_cpu_count`) wraps `pthreads` on POSIX and `_beginthreadex` on Windows; both compile from the single `microgpt.h` header.

## 7. Performance SLOs

The reference machine (Apple M2 Max, single-threaded unless noted; default float32 build) is defined once in `NFRD.md` §4. The IDs below are aliases of the rows in `NFRD.md` §4.1 and §4.5; do not duplicate machine specs here — update `NFRD.md` if the reference machine changes.

| ID | Measured target |
|---|---|
| SLO-CORE-001 | `forward_backward_one` ≥ 500K fwd+bwd/s — see `NFRD.md` §4.1 |
| SLO-CORE-002 | `adam_step` ≥ 600K steps/s — see `NFRD.md` §4.1 |
| SLO-CORE-003 | `sample_token` ≥ 6M samples/s — see `NFRD.md` §4.1 |
| SLO-CORE-004 | Full step ≥ 600K tok/s — see `NFRD.md` §4.1 |
| SLO-CORE-005 | `forward_inference` ≥ 1.5M infer/s — see `NFRD.md` §4.1 |
| SLO-CORE-006 | Auto-regressive ≥ 1M tok/s — see `NFRD.md` §4.1 |
| SLO-CORE-007 | Checkpoint round-trip ≥ 4,500 rt/s — see `NFRD.md` §4.1 |
| SLO-CORE-020 | Combined V4 stack ≥ 8.7 % PPL improvement — see `NFRD.md` §4.5 |

## 8. Scenarios

### SCN-CORE-001: Train a stem-cell model from scratch

A demo loads a corpus, builds the vocabulary, calls `model_create`, allocates `grads`, `m`, `v`, KV caches per layer, then loops `num_steps` times — each step zeroes `grads`, runs `forward_backward_one` over each token of each batch document, calls `adam_step`. After training, the demo writes a checkpoint via `checkpoint_save`.

### SCN-CORE-002: Resume training

A demo reads a checkpoint via `checkpoint_load`, recovers `step_out`, and continues the training loop from `step_out`. The cosine LR schedule continues correctly because `step` is preserved.

### SCN-CORE-003: Inference with KV cache

A demo loads weights via `model_load`, allocates KV caches, feeds a prompt token by token via `forward_inference`, then auto-regressively samples via `sample_token` until a stop condition.

## 9. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-CORE-001 | INV-CORE-001..006 | `tests/test_microgpt.c::test_default_config` |
| ACC-CORE-002 | INV-CORE-010..014 | `tests/test_microgpt.c::test_model_lifecycle` |
| ACC-CORE-003 | SLO-CORE-001..007 | `tests/bench_microgpt.c` |
| ACC-CORE-004 | NFR-010 (determinism) | `tests/test_microgpt.c` regression seeds |
| ACC-CORE-005 | INV-CORE-040 (QK-norm stability) | CMake target `test_microgpt_qk_norm` (compiled from `tests/test_microgpt.c` with `-DMICROGPT_QK_NORM` plus stress-LR defines; see `CMakeLists.txt:549-563`) |
| ACC-CORE-006 | INV-CORE-041 (V4 stack orthogonality) | CMake target `test_microgpt_rope` and the `bench_attn_sink_*`, `bench_qk_norm_*`, `bench_rope_*`, `bench_msa_*` targets — all built from existing test/bench sources with per-target `-D` defines, NOT from dedicated source files |

## 10. Cross-references

- **TDD:** `TDD_core.md`
- **FS:** `FS_checkpoint.md`
- **Source:** `src/microgpt.{h,c}`
- **Tests:** `tests/test_microgpt.c`, `tests/bench_microgpt.c`
- **Upstream dependencies:** none (this is the foundation layer).
- **Downstream dependents:** `BS_organelle.md`, `BS_msa.md`, `BS_quant.md`, `BS_pipeline_ir.md`, `BS_wiring.md`.

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
