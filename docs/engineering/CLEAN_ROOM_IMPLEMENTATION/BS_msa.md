# BS_msa — Behaviour Specification (Memory Sparse Attention)

**Document ID:** BS-MSA-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

This document specifies the behavioural contract of `MsaPool`, the routing primitives (`msa_route_top_1`, `msa_route_top_k`), the context-injection primitives (`msa_expand_context`, `_rope`), the recency ring (`MsaRecency`), and the RoPE-aware pool wrappers (`msa_pool_chunk_rope`, `msa_expand_context_rope`).

## 2. Type contracts

### 2.1 `MsaPool`

**Invariants:**
- INV-MSA-001: `pool->capacity > 0`; `pool->length ≤ pool->capacity`.
- INV-MSA-002: `pool->n_layer == cfg->n_layer` and `pool->n_embd == cfg->n_embd` for the model the pool services.
- INV-MSA-003: When `ENABLE_TURBOQUANT` or `ENABLE_ROTORQUANT` is defined at build time, the pool stores K/V as quantised triples; otherwise as `scalar_t[]`.

### 2.2 `MsaRecency`

**Invariants:**
- INV-MSA-010: `rec->capacity == n_win` (the window argument).
- INV-MSA-011: `rec->positions[i]` records the absolute `pos_id` of the K/V at slot `i` (used by RoPE re-rotation).
- INV-MSA-012: When the ring is full, `msa_recency_push` evicts the oldest entry.

## 3. Operation contracts

### 3.1 `msa_pool_create / _free`

**Postconditions:** `_create` returns a heap pool with `length = 0`. `_free` releases all owned buffers.

### 3.2 `msa_pool_chunk(pool, active_keys, active_values, chunk_len)`

**Preconditions:** `pool->length < pool->capacity`; `active_keys[L]` and `active_values[L]` allocated by `kv_cache_alloc`; the first `chunk_len` positions of each are valid.

**Postconditions:** Pools the first `chunk_len` positions across all layers into pool slot `pool->length`, increments `pool->length`, returns the new chunk index. Returns -1 if the pool is full.

The pooling weights are determined by `MSA_POOL_MODE` (compile-time): 0 uniform mean, 1 linear ramp, 2 exponential, 3 content-aware softmax (INV-MSA-020).

### 3.3 `msa_pool_chunk_rope(pool, active_keys, active_values, chunk_len, start_pos, n_head)`

**Preconditions:** As `msa_pool_chunk`. `MICROGPT_PARTIAL_ROPE` SHOULD be defined for the function to be useful; otherwise it falls back to the legacy non-rotating path.

**Postconditions:** For each token in the chunk window, K is un-rotated by `R(−(start_pos + t))` per head BEFORE pooling. The pool entry is therefore in "position-zero space".

### 3.4 `msa_route_top_1(pool, query_keys)`

**Postconditions:** Returns the index of the chunk maximising cross-layer cosine similarity to `query_keys`, or -1 if `pool->length == 0`. Ties are broken by lower index (stable).

### 3.5 `msa_route_top_k(pool, query_keys, k, indices_out, scores_out)`

**Preconditions:** `indices_out` capacity ≥ `k`; `scores_out` MAY be NULL.

**Postconditions:** Computes for each chunk `i`: `score[i] = sum_L max(0, K_q[L] · K_pool[i][L] / sqrt(n_embd))`. Returns the top-K indices in descending score order with stable tie-break, with unused slots filled with -1. Return value is the count actually written (≤ k, == 0 if pool empty).

### 3.6 `msa_expand_context(pool, chunk_idx, active_keys, active_values, pos)`

**Preconditions:** `chunk_idx ∈ [0, pool->length)`; `pos < block_size`.

**Postconditions:** Writes `pool->keys[chunk_idx][L]` to `active_keys[L][pos × n_embd + ...]` for every layer L. Same for values.

### 3.7 `msa_expand_context_rope(pool, chunk_idx, active_keys, active_values, pos, n_head)`

**Postconditions:** As `msa_expand_context`, then re-rotates the just-written K by `+pos` per head when `MICROGPT_PARTIAL_ROPE` is defined.

### 3.8 `MsaRecency` API

`_create`, `_free`, `_reset`, `_push(rec, token_keys, token_values, pos)`, `_inject(rec, active_keys, active_values, start_pos)`, `_inject_rope(rec, active_keys, active_values, start_pos, n_head)`. Tokens are written in chronological order (oldest → newest). Inject returns the count of positions written.

## 4. Invariants table

| ID | Invariant |
|---|---|
| INV-MSA-001 | `length ≤ capacity`. |
| INV-MSA-002 | `n_layer`, `n_embd` match the model. |
| INV-MSA-003 | Quantised storage iff `ENABLE_TURBOQUANT` / `ENABLE_ROTORQUANT`. |
| INV-MSA-010..012 | Recency ring: capacity, position tracking, FIFO eviction. |
| INV-MSA-020 | Pool weighting governed by `MSA_POOL_MODE`. |
| INV-MSA-021 | RoPE-aware variants store pool entries in position-zero space; non-RoPE variants store rotated K (and accept the mean-of-rotated-vectors caveat). |
| INV-MSA-022 | Routing is read-only over the pool snapshot; safe under concurrent reads. |

## 5. Errors

| ID | Function | Code | Conditions |
|---|---|---|---|
| ERR-MSA-001 | `msa_pool_chunk*` | -1 | Pool full |
| ERR-MSA-002 | `msa_route_top_1` | -1 | Pool empty |

## 6. Concurrency

The pool is single-writer (chunking events) / many-reader (routing). Routing is safe under a stable snapshot; callers SHOULD avoid concurrent chunking and routing.

## 7. Performance SLOs

Reference machine in `NFRD.md` §4.

| ID | Measured target |
|---|---|
| SLO-MSA-001 | MSA pool encode ≥ 1.3M encodes/s — see `NFRD.md` §4.4 |

## 8. Scenarios

### SCN-MSA-001: Infinite Shakespeare

`msa_infinite_shakespeare` keeps generating; when the active KV cache fills, the controller chunks the oldest half into the pool, then on each next step routes top-1 (or top-K) and injects the chunk back at slot 0. The generation continues indefinitely with O(1) memory growth.

### SCN-MSA-002: Long-context fraud companion

`msa_fraud_guardian` uses MSA to retain a long history of signals; the recency ring guarantees the last `n_win` tokens stay uncompressed.

## 9. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-MSA-001 | INV-MSA-001..003, INV-MSA-022 | `tests/test_microgpt_msa.c` |
| ACC-MSA-002 | SLO-MSA-001 | `tests/bench_microgpt_msa.c` |

## 10. Cross-references

- **TDD:** `TDD_msa.md`
- **Source:** `src/microgpt_msa.{h,c}`
- **Upstream:** `BS_core.md`
- **Optional integration:** `BS_quant.md` (when ENABLE_TURBOQUANT/ROTORQUANT)

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
