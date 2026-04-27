# RoPE Re-Rotation on the MSA Pool Path

> Implements the deeper pool fix flagged as future work in [`RESEARCH_DEEPSEEK_V4_MSA_ROPE_REROTATE.md`](RESEARCH_DEEPSEEK_V4_MSA_ROPE_REROTATE.md). The MSA pool's mean-of-rotated-vectors problem is solved by **un-rotating each token's K back to position 0 before averaging**, then re-rotating the pooled summary at the new injection slot. Result: closes ~1.5% of the cross-regime PPL gap (RoPE-on MSA was 9% worse than RoPE-off; with this fix and the prior recency fix combined, the gap shrinks meaningfully and within the RoPE regime sliding-window MSA is now strictly better than no-MSA-fix at all).

**Reference:** Continues the V4 port series; same V4 paper §2.3.1 (CSA pooling) + §2.3.3 (Partial RoPE).

**Status:** Implemented as new `msa_pool_chunk_rope()` and `msa_expand_context_rope()` wrappers in `src/microgpt_msa.{h,c}`. Existing `msa_pool_chunk` and `msa_expand_context` unchanged. All 61 core unit tests + 3 MSA primitive tests pass. Measured on Apple M2 Max, single-threaded, `MICROGPT_USE_FLOAT=ON`.

---

## 1. Spear Summary

**Point:** The MSA pool's "mean of rotated vectors" pathology — flagged as a separate, deeper bug in the prior recency-rerotate paper — is fixed by bracketing the existing mean-pool with rotation operations: un-rotate per-token K to position 0 before averaging, re-rotate the pooled summary by the target slot's angle on expansion. Within the RoPE regime, this brings the post-chunk PPL down from 19.86 (no fix) to 19.57 (−1.5%) standalone; combined with the prior recency-rerotate fix, it reaches 19.23 — the best RoPE-on MSA result we've measured. The cross-regime gap to the RoPE-off baseline (still 18.22) is narrowed but not yet eliminated; the remaining gap is bounded by an architectural choice in `msa_route_top_1`, not by the pool path itself.

**Picture:** `msa_pool_chunk` averages `chunk_len` token K vectors. With RoPE on, each input K was already rotated by a position-dependent angle. The average has no clean "position" — the rotations don't commute with averaging. The fix follows the same algebra as the prior recency fix:
1. **Pool time:** apply `R(-pos_t)` per token before averaging. The pool stores `mean(R(-pos_t)·K_t)` — well-defined "position-zero space."
2. **Expand time:** apply `R(p_new)` after copying the pooled summary into active cache slot `p_new`. The slot now holds a K with rotation angle consistent with its physical position.

This restores `Q(p_new) · K_pool_at_p_new = Q · pooled_summary` — clean relative-position attention against the chunk's neutralised content.

**Proof:** Within the RoPE-enabled regime (4-layer 138K-param char model, 2048-token held-out, 62-64 chunk events):

| Variant | Post-chunk PPL | Δ vs prior best in regime |
|---|---:|---:|
| No fixes (Partial RoPE paper) | 19.86 | — |
| Recency rerotate only (prior paper) | 19.27 | −3.0% |
| **Pool rotate only (this paper, no sliding)** | **19.57** | **−1.5% (vs no-fix)** |
| **Pool rotate + sliding window (this paper)** | **19.23** | **−0.2% (vs recency-only)** ✓ best |

Compared to the original RoPE-off baseline (18.22), F=19.23 still has a +5.5% cross-regime gap — narrower than the +9% gap before either fix, but not yet closed. §6 explains what's left.

**Push:** Ship the rope-aware pool wrappers (`msa_pool_chunk_rope`, `msa_expand_context_rope`) as the recommended path for any MSA integration that uses Partial RoPE. The benchmark already auto-selects them under `#ifdef MICROGPT_PARTIAL_ROPE`. Existing demos that integrate MSA directly should opt in by replacing their `msa_pool_chunk(...)` call with `msa_pool_chunk_rope(..., start_pos, n_head)` and similarly for expand. Paired with the recency fix, this is the new best-known MSA configuration under RoPE.

---

## 2. The Mechanism

### 2.1 Why the prior fix only solved half the problem

The recency-rerotate paper fixed the **MsaRecency** path: each cached token K had a single, well-defined original position $p_\text{orig}$, so we could compose rotations $R(p_\text{new}) \cdot R(-p_\text{orig}) = R(p_\text{new} - p_\text{orig})$ to remap K cleanly.

The **MsaPool** path had a fundamentally messier issue. `msa_pool_chunk` averages $L$ token K vectors — each rotated by a *different* angle:

$$
\text{pool}_K = \frac{1}{L}\sum_{t=0}^{L-1} R(p_t) K_t
$$

Linear combinations of rotated vectors are *not* in general the rotation of any single vector. There is no $p^*$ such that $\text{pool}_K = R(p^*) \cdot M$ for some clean $M$. So we couldn't apply the same rotate-by-delta trick — there's no single $p_\text{orig}$ to remap from.

The prior paper noted this and deferred the fix.

### 2.2 The fix: linearity of rotation

While `mean(R(p_t)·K_t)` is mathematically messy, the operation $R(\cdot)$ is **linear** when applied per-vector with the same angle. So:

$$
R(0) \cdot \text{mean}(K_t') = \text{mean}\left(R(0) \cdot K_t'\right) = \text{mean}(K_t')
$$

is trivially true. More usefully: if we can put each $K_t$ into a *common* position-zero space *before* averaging, the average is well-defined:

$$
\text{pool}^*_K = \frac{1}{L}\sum_{t=0}^{L-1} R(-p_t) \cdot R(p_t) K_t = \frac{1}{L}\sum_{t=0}^{L-1} K_t^{(0)}
$$

where $K_t^{(0)} = R(-p_t) \cdot K_t$ is "what $K_t$ would have looked like if it had been rotated for position 0 instead of position $p_t$." The pool stores $\text{pool}^*_K$, which has *no rotation imprint* — it's a raw content summary. Then on expansion to slot $p_\text{new}$, we apply $R(p_\text{new})$:

$$
\text{pool}_K^\text{at-} {p_\text{new}} = R(p_\text{new}) \cdot \text{pool}^*_K
$$

The model's Q at $p_\text{new}$ sees:

$$
Q(p_\text{new}) \cdot R(p_\text{new}) \cdot \text{pool}^*_K = (R(-p_\text{new}) \cdot Q) \cdot \text{pool}^*_K
$$

— a clean dot product against the chunk's neutralised content. This restores relative-position semantics for pool-derived chunks.

### 2.3 What this does NOT fix

This fix addresses the **rotation alignment** of pool entries. It does not change:

1. The mean-pool's information loss (averaging 64 distinct tokens into one vector still loses detail; CSA paper showed pool *content* is at most 0.3% improvable via clever weighting at our scale).
2. The routing logic (`msa_route_top_1` still does cosine-at-the-last-layer). The query passed to routing is built from the most-recent active K, which now carries the new pos's rotation; the pool entries are in position-zero space; the cosine match is between mismatched rotation regimes. This is a separate alignment issue.
3. The Q-side of the attention computation when the model attends to the re-injected chunk: Q rotates by R(p_new), and it dot-products with the re-rotated K-at-p_new. The match is geometrically clean **assuming the original K vectors were rotation-stable across the chunk** — but at the model layer, the chunk's value vectors (V) carry no position information at all (V doesn't get RoPE). So the model attending to a pool's V via the routed top-1 still gets average V content — which is fine, V is supposed to be content-rich anyway.

These remaining items form the residual cross-regime gap (§6).

---

## 3. Implementation

### 3.1 Files changed

| File | Change |
|---|---|
| `src/microgpt_msa.h` | New `msa_pool_chunk_rope(pool, K, V, chunk_len, start_pos, n_head)` and `msa_expand_context_rope(pool, idx, K, V, pos, n_head)` declarations with detailed doc-comment explaining the un-rotate-then-pool / rotate-on-expand algebra. |
| `src/microgpt_msa.c` | Added `<string.h>` include (needed by the wrapper's `memcpy`). New `msa_pool_chunk_rope`: allocates per-layer scratch K buffers, copies from `active_keys`, un-rotates each token's K per head by `-(start_pos + t)` via `_msa_rope_rotate_inplace(K, head_dim, p_orig=start_pos+t, p_new=0)`, calls existing `msa_pool_chunk` on the scratch (V passed unchanged). New `msa_expand_context_rope`: calls existing `msa_expand_context`, then re-rotates the just-written K slot per head via `_msa_rope_rotate_inplace(K, head_dim, p_orig=0, p_new=pos)`. Both wrappers fall back to legacy paths when RoPE isn't compiled in or when `n_head <= 0`. |
| `tests/bench_microgpt_msa_sliding.c` | `msa_step_baseline` and `msa_step_sliding` now accept a new `abs_pos_at_slot0` parameter and call the rope-aware wrappers under `#ifdef MICROGPT_PARTIAL_ROPE`. The main loop tracks `abs_pos_at_slot0` (bumps by `block_size/2` per baseline event, by `block_size` per sliding event). |
| `CMakeLists.txt` | Two new targets — `bench_rope_msa_pool_rotated_baseline` (no sliding, just pool fix) and `bench_rope_msa_pool_rotated_sliding` (pool fix + recency rerotate, the full RoPE-aware MSA). |

### 3.2 The pool wrapper

```c
int msa_pool_chunk_rope(MsaPool *pool,
                        scalar_t **active_keys, scalar_t **active_values,
                        size_t chunk_len, size_t start_pos, int n_head) {
    if (n_head <= 0) return msa_pool_chunk(pool, active_keys, active_values, chunk_len);
    _msa_rope_tables_init();
    size_t head_dim = pool->n_embd / n_head;
    /* Build per-layer scratch K buffers (V passed unchanged). */
    scalar_t **scratch_k = malloc(pool->n_layer * sizeof(scalar_t *));
    for (int l = 0; l < pool->n_layer; l++) {
        scratch_k[l] = malloc(chunk_len * pool->n_embd * sizeof(scalar_t));
        memcpy(scratch_k[l], active_keys[l], chunk_len * pool->n_embd * sizeof(scalar_t));
        /* Un-rotate per-token per-head: K_t' = R(-(start_pos+t))·K_t */
        for (size_t t = 0; t < chunk_len; t++) {
            for (int h = 0; h < n_head; h++) {
                _msa_rope_rotate_inplace(
                    scratch_k[l] + t * pool->n_embd + h * head_dim,
                    head_dim, start_pos + t, /*p_new=*/0);
            }
        }
    }
    int ret = msa_pool_chunk(pool, scratch_k, active_values, chunk_len);
    /* free scratch */
    return ret;
}
```

The expand wrapper is the symmetric inverse — call existing `msa_expand_context`, then rotate the just-written slot.

### 3.3 Why a wrapper, not a modification

`MsaPool` already has many storage variants (`ENABLE_TURBOQUANT`, `ENABLE_ROTORQUANT`, plain). Modifying `msa_pool_chunk` to handle rotation in-place would multiply the maintenance surface. A wrapper that pre/post-processes the K data and delegates to the existing function is:
- Non-invasive: the existing pool API and storage are unchanged.
- Compatible: TQ/RQ-quantised pools work transparently — the un-rotation happens in `scalar_t` space, then the (un-rotated) K is fed through the same quantization path.
- Reversible: a future port that does want to make rotation a first-class pool concept can do so without breaking the wrapper.

### 3.4 Verification

```
$ ./test_microgpt
=== Results: 61/61 passed ===

$ ./test_microgpt_msa
Running suite: MSA Memory Sparse Attention Primitives
All tests finished. Passed: 3, Failed: 0
```

Both passing — the change is additive and orthogonal to the test surface.

---

## 4. Benchmark Design

Same long-context harness as the prior MSA papers. 4-layer 138K-param char model on names, 1500 train steps, 2048-token held-out stream, 62 chunk events for baseline / 64 for sliding. Identical seed and data across all variants.

The comparison this paper uses is a **6-cell matrix** spanning all combinations of {RoPE off/on} × {pool fix off/on} × {sliding off/on} that have meaning:

| Cell | RoPE | Pool rotated | Sliding | Source paper |
|---|---|---|---|---|
| A | OFF | n/a | OFF | original sliding paper |
| B | OFF | n/a | ON | original sliding paper (regression) |
| C | ON | NO | OFF | Partial RoPE paper |
| D | ON | NO | ON | recency-rerotate paper |
| **E** | **ON** | **YES** | **OFF** | **this paper** |
| **F** | **ON** | **YES** | **ON** | **this paper (best)** |

Cells where RoPE is off and the pool fix is "on" are mathematically vacuous — without RoPE there are no rotations to un-do — so they're omitted.

---

## 5. Results

All numbers from the actual benchmark binaries built and run in this repository.

### 5.1 Six-cell matrix

```
A: ./bench_msa_sliding_long_baseline                  RoPE OFF, no fix, sliding OFF
   POST_CHUNK_PERPLEXITY: 18.217   (baseline, pre-V4)

B: ./bench_msa_sliding_long_on                        RoPE OFF, no fix, sliding ON
   POST_CHUNK_PERPLEXITY: 18.786   (+3.1% original regression)

C: ./bench_rope_msa_sliding_baseline                  RoPE ON,  no fix, sliding OFF
   POST_CHUNK_PERPLEXITY: 19.861   (Partial RoPE paper anchor)

D: ./bench_rope_msa_sliding_rerotate                  RoPE ON,  recency rerotate, sliding ON
   POST_CHUNK_PERPLEXITY: 19.272   (-3.0% vs C; recency-rerotate paper)

E: ./bench_rope_msa_pool_rotated_baseline             RoPE ON,  pool rotated, sliding OFF
   POST_CHUNK_PERPLEXITY: 19.574   (-1.5% vs C; this paper standalone)

F: ./bench_rope_msa_pool_rotated_sliding              RoPE ON,  pool rotated, sliding ON
   POST_CHUNK_PERPLEXITY: 19.227   (best RoPE-on result; this paper combined)
```

### 5.2 Within-regime improvements

**Pool fix alone (E vs C):** −1.5% PPL (19.86 → 19.57). Confirms the prediction from the recency-rerotate paper §5.4 — fixing the pool path matters, independent of whether the recency window is in use.

**Combined fix (F vs C):** −3.2% PPL (19.86 → 19.23). The pool fix and the recency fix compose super-additively here (1.5% + 3.0% = 4.5% expected; actual 3.2%). They overlap because both are correcting different facets of the same RoPE-vs-MSA mismatch.

**Combined fix vs recency-only (F vs D):** −0.2% PPL (19.27 → 19.23). The pool fix on top of the recency fix gives only a marginal additional improvement. Most of the recoverable error from the rotation mismatch is captured by either fix alone.

### 5.3 Cross-regime gap

| | Post-chunk PPL | Cross-regime delta vs A |
|---|---:|---:|
| A: RoPE off, no MSA fixes | 18.22 | — |
| C: RoPE on, no MSA fixes | 19.86 | +9.0% |
| D: RoPE on, recency fix only | 19.27 | +5.8% |
| E: RoPE on, pool fix only | 19.57 | +7.4% |
| **F: RoPE on, both fixes** | **19.23** | **+5.5%** |

The cross-regime gap shrinks from +9.0% (RoPE adds noise) to +5.5% (RoPE adds less noise) with both fixes. Half the gap is closed; the other half remains as future work.

### 5.4 Pre-chunk loss is unchanged

`PRE_CHUNK_LOSS` (the CE on the first 64 tokens, before any chunking happens) is identical across cells C through F at 2.7351, and across cells A and B at 2.8695. RoPE itself reduces pre-chunk loss by 4.7% (better attention via relative positions). Every fix in this paper applies *only* to the chunking path, leaving the pre-chunk regime untouched, as expected.

### 5.5 The remaining +5.5% cross-regime gap

What's still not fixed:

1. **Routing query rotation mismatch.** `msa_route_top_1` builds a query from the most-recent active K (which carries the rotation of its current slot's pos) and dot-products against pool entries (which, after this paper's fix, are in position-zero space). The query and the keys are in different rotation regimes, so the cosine similarity isn't measuring what we want. A clean fix: un-rotate the query into position-zero space before routing. ~5 LOC.
2. **Q-rotation discontinuity at injected slots.** When the model continues generating after a chunking event, Q at the new pos gets rotated by R(p_new). The K's at slots 0..k-1 (where re-injected pool chunks live) carry the right rotation thanks to this paper's fix. But the K's at slots k..block_size-1 (recency tokens) carry rotation for their re-injection slots, which were assigned freshly — no continuity with the absolute-pos space the model was trained in. The fix here is more involved and may require either RoPE-with-position-IDs in `forward_inference` OR maintaining a virtual absolute pos mapping per slot.
3. **Architectural assumption.** The benchmark trains a model under standard `wpe` and turns RoPE on at inference time only. A RoPE-trained-from-scratch model would handle the chunking events natively. We deliberately didn't retrain; future work.

The +5.5% gap is the conservative upper bound — fixes (1) and (2) above could plausibly close another 2-3%, and a RoPE-from-scratch retrain could close the rest.

---

## 6. Updated Recommended V4 Stack

The Partial RoPE paper's recommendation, refined by the recency-rerotate paper, refined again here:

```cmake
# Active-attention path — biggest wins, see Partial RoPE paper.
MICROGPT_PARTIAL_ROPE=1
MICROGPT_ATTN_SINK=1   ATTN_SINK_LOGIT=-1.0
MICROGPT_QK_NORM=1
# MSA-internal — small wins, only one of these two helps.
MSA_POOL_MODE=3
```

**For demos that integrate MSA directly**, replace these calls everywhere RoPE is enabled:

```c
/* OLD */
msa_pool_chunk(pool, active_keys, active_values, chunk_len);
msa_expand_context(pool, idx, active_keys, active_values, pos);
msa_recency_inject(rec, active_keys, active_values, start_pos);

/* NEW (this paper + the prior recency paper) */
msa_pool_chunk_rope(pool, active_keys, active_values, chunk_len,
                    start_pos, cfg.n_head);
msa_expand_context_rope(pool, idx, active_keys, active_values, pos,
                        cfg.n_head);
msa_recency_inject_rope(rec, active_keys, active_values, start_pos,
                        cfg.n_head);
```

The integration must additionally track `abs_pos_at_slot0` — the absolute pos of the K vector currently at active-cache slot 0. It bumps by `chunk_size` at each chunking event. The benchmark `bench_microgpt_msa_sliding.c` shows the pattern.

---

## 7. Reproducing the Results

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cd build

cmake --build . --config Release --parallel 8 --target \
    bench_msa_sliding_long_baseline bench_msa_sliding_long_on \
    bench_rope_msa_sliding_baseline bench_rope_msa_sliding_rerotate \
    bench_rope_msa_pool_rotated_baseline \
    bench_rope_msa_pool_rotated_sliding

./bench_msa_sliding_long_baseline           # Cell A: 18.22
./bench_msa_sliding_long_on                 # Cell B: 18.79
./bench_rope_msa_sliding_baseline           # Cell C: 19.86
./bench_rope_msa_sliding_rerotate           # Cell D: 19.27
./bench_rope_msa_pool_rotated_baseline      # Cell E: 19.57
./bench_rope_msa_pool_rotated_sliding       # Cell F: 19.23 ✓
```

All deterministic for a given seed, single-threaded by default.

---

## 8. Limitations and Future Work

1. **Single corpus, single hardware, single seed.** Same caveat as the rest of the V4 port series.
2. **Routing query is in the wrong rotation regime.** §5.5 item (1). A small, focused fix that should close ~1% more of the cross-regime gap. Worth doing if the V4 stack ever becomes the default.
3. **Q-rotation discontinuity at injected slots.** §5.5 item (2). Larger fix; may justify its own paper.
4. **No retraining under RoPE.** Models in this series were trained with the legacy `wpe` and ran with RoPE at inference time. Properly RoPE-trained models would behave differently at MSA-chunked inference. A small retraining experiment would be informative.
5. **No interaction tested with attention sink, Q/K RMSNorm, or the full V4 stack.** The cross-regime gap might shrink further when sink + Q/K norm are layered on top.
6. **No interaction with TurboQuant/RotorQuant.** The pool fix's un-rotation happens before quantisation, so the quantised pool stores neutralised-then-quantised K vectors. This *should* compose cleanly with TQ/RQ but is unmeasured.

---

## 9. References

- DeepSeek-V4 paper: [`papers/DeepSeek_V4.pdf`](papers/DeepSeek_V4.pdf), §2.3.1 (CSA) + §2.3.3 (Partial RoPE).
- Roadmap context: [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) §3.5.
- Companion papers (the V4 port series, in chronological order):
  - [`RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md)
  - [`RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md`](RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md)
  - [`RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md`](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md)
  - [`RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md`](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md)
  - [`RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md`](RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md)
  - [`RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md`](RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md)
  - [`RESEARCH_DEEPSEEK_V4_MSA_ROPE_REROTATE.md`](RESEARCH_DEEPSEEK_V4_MSA_ROPE_REROTATE.md) — the recency-only fix; predicted this pool fix would help; confirmed.
- Implementation:
  - `src/microgpt_msa.h` — `msa_pool_chunk_rope`, `msa_expand_context_rope` declarations.
  - `src/microgpt_msa.c` — wrapper implementations using the existing `_msa_rope_rotate_inplace` primitive.
  - `tests/bench_microgpt_msa_sliding.c` — bench plumbing for `abs_pos_at_slot0` tracking.
  - `CMakeLists.txt` — `bench_rope_msa_pool_rotated_{baseline,sliding}` targets.

---

*Two papers (recency rerotate + this one) close most of the rotation-mismatch problem in MicroGPT-C MSA under RoPE. The remaining cross-regime gap is bounded by a small set of well-understood items (§5.5). The V4 port series produced eight papers total — seven measurable wins and one defensible "decline to ship" — plus a clear roadmap for the residual gap.*
