# TDD_msa — Technical Design Document

**Document ID:** TDD-MSA-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_msa.md`
**Sources:** `src/microgpt_msa.{h,c}`

## 1. Overview

Memory Sparse Attention (MSA) lifts the bounded-context-window constraint of the core engine by routing attention through a *latent pool* of compressed K/V chunks. A query token's K vectors are scored against every pool entry; the top-K pool entries are re-injected into the active KV cache, giving the model long-history awareness without the O(L²) memory growth of dense attention.

MSA is orthogonal to the rest of the engine — it composes through the existing `forward_inference` interface and does not modify the core. It optionally composes with TurboQuant / RotorQuant for additional 8× memory reduction on the pool.

## 2. Architecture

```
       (current step) active KV cache  ─────────────────────────────────────►
       per layer × n_embd × ≤ block_size positions
                                           ▲
                                           │ inject (msa_expand_context_*)
                                           │
                                  pool entry (n_layer × n_embd float)
                                           ▲
                                           │ select top-K (cosine / lightning)
                                           │
                       ┌─────────────────────────────────┐
                       │      MsaPool (latent arena)      │
                       │                                 │
                       │ keys[capacity, n_layer, n_embd] │
                       │ values[capacity, n_layer, n_embd] │
                       └──────────────────────────────────┘
                                           ▲
                                           │ pool (mean / weighted)
                                           │
                       (chunk window) ← active KV cache (chunk_len positions)
```

A second buffer, `MsaRecency`, retains the last `n_win` *uncompressed* K/V tokens across chunking events; it is independent of the pool and provides guaranteed local-coherence even when the pool's routing is wrong.

## 3. Data flow

When the active KV cache fills (at `cfg->block_size`), the controller chunks `chunk_len` (typically `block_size / 2`) of the oldest tokens into a single pool entry, frees those positions, and shifts the remaining tokens. On the next forward pass the controller scores the current query's K against the pool, picks top-K entries, and injects them at the head of the now-vacant cache positions.

## 4. Key data structures

### 4.1 `MsaPool`

```c
typedef struct {
  scalar_t *keys;     /* shape [capacity, n_layer, n_embd] */
  scalar_t *values;   /* shape [capacity, n_layer, n_embd] */
  size_t capacity, length;
  int    n_layer, n_embd;
} MsaPool;
```

When `ENABLE_TURBOQUANT` or `ENABLE_ROTORQUANT` is set, the pool's K/V are stored as quantised triples (`tq_keys_idx`, `tq_keys_qjl`, `tq_keys_rnorm`) and reconstructed at use.

### 4.2 `MsaRecency`

A ring buffer of fixed capacity `n_win`, holding `[capacity, n_layer, n_embd]` floats plus the absolute `pos_id` of each token at push time. Eviction is FIFO when full.

## 5. Algorithms

### 5.1 `msa_pool_chunk`

Mean (or weighted) pool of `chunk_len` token K/V vectors, layer-by-layer:

```
for each layer L:
    for each dim d:
        pool->keys[idx][L][d] = sum_t w[t] * active_keys[L][t * n_embd + d]
        pool->values[idx][L][d] = sum_t w[t] * active_values[L][t * n_embd + d]
```

Weights `w[t]` depend on `MSA_POOL_MODE`:
- 0 (default): `w[t] = 1 / chunk_len` (uniform).
- 1: linear ramp `w[t] ∝ 1 + t/(L−1)` (recency).
- 2: exponential `w[t] ∝ exp(t/τ)` (sharper recency).
- 3: content-aware `w[t] = softmax_t(K[t] · K_last / √n_embd)`.

Mode 3 measured −0.32 % PPL on long-context MSA evaluation; modes 1/2 are within noise.

### 5.2 `msa_route_top_1`

For each pool entry `i`, compute cross-layer cosine similarity between `query_keys` and `pool->keys[i]`; return the argmax.

### 5.3 `msa_route_top_k` (Lightning Indexer port)

```
score[i] = sum_L max(0, K_q[L] · K_pool[i][L] / √n_embd)
```

Top-K by score, descending; stable tie-break on lower index. Capped at `pool->length`. Output `indices_out[]` is partially-filled with `-1` beyond the returned count.

### 5.4 `msa_expand_context`

Writes `pool->keys[idx][L][...]` into `active_keys[L][pos * n_embd + ...]` for every layer L. `msa_expand_context_rope` additionally re-rotates the just-written K by `+pos` per head when `MICROGPT_PARTIAL_ROPE` is active.

### 5.5 RoPE-aware pool / inject (the "mean of rotated vectors" fix)

Without correction, mean-pooling K vectors that were rotated by *different* RoPE angles is meaningless. The fix:

- At pool time: `K_t' = R(−pos_t) · K_t` per token before averaging.
- At expand time: `K_pool_at_p_new = R(p_new) · pooled_mean`.

The `_rope` variants take `start_pos` (the absolute pos_id of the chunk's first token) and `n_head` (RoPE rotates per-head). When `MICROGPT_PARTIAL_ROPE` is not defined or `n_head ≤ 0`, the variants fall back to the legacy non-rotating path.

## 6. Concurrency model

The pool is single-writer / many-reader: chunking events happen on the inference thread between forward passes. Read-only routing operations (`msa_route_top_*`) are thread-safe under a stable pool snapshot.

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Pool weighting | Fixed-form (4 modes) | Learnable (CSA proper) | Learnable pooling requires backprop through the pool, which sits outside the training graph. Fixed-form lets us measure whether *any* non-uniform weighting helps before committing to the larger refactor. |
| Routing | Cosine top-1 + Lightning top-K | Learned indexer (W^DQ, W^IUQ, W^w) | Same reason as above — multi-layer ReLU-summed score captures V4's structural innovation without parameters. |
| Sliding-window recency | Separate `MsaRecency` ring | Combine with pool | Composability — recency tail can be enabled / disabled independently of pool routing, and avoids smearing local detail through chunk pooling. |
| RoPE handling | Position-zero pool + rotate-on-inject | Pool rotated K directly | Mean of rotated vectors is mathematically meaningless; the rotate-fix produces valid Q · K dot products at the new slot. |

## 8. Known limitations

- The pool has fixed capacity; eviction policy on overflow is the caller's responsibility (typical: LRU on chunk-add).
- Routing scores are computed against every pool entry every step (linear in `pool->length`); this is fine for the small-pool regime (typically ≤ 1024 chunks) but does not scale to a million-chunk arena.
- Quantised pool storage assumes a single global `g_tq` / `g_rq` instance; multi-organelle deployments must arrange to use compatible quantiser instances or partition pools per organelle.
- `MsaRecency` does not interact with the pool's routing — recency injection happens at fixed slots regardless of pool selection.

## 9. References

- `docs/research/RESEARCH_MSA.md`, `RESEARCH_MSA_USE_CASES.md`.
- DeepSeek-V4 §2.3.1 (Lightning Indexer, CSA pooling).
- `docs/research/RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md`.
- `docs/research/RESEARCH_DEEPSEEK_V4_MSA_*` (rope rotate, recency, learnable pool).

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
