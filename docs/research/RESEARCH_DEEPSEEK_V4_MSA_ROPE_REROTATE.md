# RoPE Re-Rotation on MSA Recency Injection

> Implementation of the "re-rotate K on injection" fix flagged as future work in [`RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md`](RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md) §6.4. When pool-derived K vectors get re-injected into the active cache at fresh physical slots, their stale RoPE rotation angles are corrected by composing rotations: `R(p_new − p_orig) = R(p_new) · R(−p_orig)`. The sliding-window-recency port (deferred since the [original paper](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md)) finally becomes **net-positive within the RoPE-on regime: −3.0% post-chunk PPL** compared to RoPE-on without sliding.

**Reference:** Same as the original sliding-window paper (DeepSeek-V4 §2.3.3), now extended to handle the V4 caveat that "K vectors carry rotation angles for their original positions but get read at new physical slots" — fixed by re-rotating them on injection.

**Status:** Implemented additively. New `msa_recency_inject_rope()` function in `src/microgpt_msa.{h,c}`. `MsaRecency` now tracks per-token absolute positions (`size_t *positions` field). All 61 core unit tests + 3 MSA primitive tests pass. Measured on Apple M2 Max, single-threaded, `MICROGPT_USE_FLOAT=ON`.

---

## 1. Spear Summary

**Point:** The sliding-window-recency port, blocked twice in this series — first by absolute-`wpe` alignment (eliminated by Partial RoPE), then by RoPE rotation angles staying baked-in at the original position when K vectors are re-injected at fresh slots — is **finally net-positive** when both Partial RoPE *and* the new re-rotation logic are enabled. Inside the RoPE regime, sliding-window MSA brings −3.0% post-chunk PPL. The two-step prediction from the [Partial RoPE paper](RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md) §6.4 is confirmed.

**Picture:** RoPE rotates each Q and K vector by an angle proportional to its position. The active-cache attention then sees Q · K depending on the *difference* between their positions — clean relative-position attention. But MSA breaks this: when a recency-cached K (rotated for original position p_orig) gets re-injected at a fresh slot p_new, the model's Q at p_new rotates by a different angle, and the relative-angle product no longer corresponds to a real position difference. The fix: at injection time, compose two rotations to remap the K from p_orig to p_new — equivalent to "un-rotate by p_orig, re-rotate by p_new", which is itself a single rotation by Δp = p_new − p_orig.

**Proof:** Within the RoPE-enabled regime (long-context 2048-token held-out, 64 MSA chunk events):

| Variant | Post-chunk PPL | Δ vs RoPE-no-sliding |
|---|---:|---:|
| RoPE on, no sliding | 19.861 | — |
| RoPE on, sliding (no re-rotation, prior paper) | 19.849 | −0.06% (neutral) |
| **RoPE on, sliding, with re-rotation fix** | **19.272** | **−3.0%** ✓ |

Compared across the full V4-port history of the sliding-window port:

| Configuration | Post-chunk PPL | vs same-regime baseline |
|---|---:|---:|
| Sliding WIN=32, no RoPE (original paper) | 18.79 | **+3.1% regression** |
| Sliding WIN=32, RoPE on (Partial RoPE paper) | 19.85 | −0.06% (neutral) |
| **Sliding WIN=32, RoPE on, re-rotation fix (this paper)** | **19.27** | **−3.0%** ✓ |

**Push:** Promote the sliding-window-recency port from "deferred" to "ship together with `MICROGPT_PARTIAL_ROPE=ON`." The recency window now provides the local-coherence benefit V4 originally claimed, *if* — and only if — RoPE is enabled and the bench/integration calls `msa_recency_inject_rope()` rather than the legacy `msa_recency_inject()`. Update the recommended V4 stack accordingly.

---

## 2. The Mechanism

### 2.1 Why the prior fix was incomplete

The Partial RoPE paper §6.4 identified the issue:

> A pool-derived K was rotated for its original position, not for its new physical slot. Q at the new slot's pos_id dot-products with that K via a relative angle that doesn't correspond to a real position difference.

Replacing absolute `wpe` with RoPE rotation eliminates one source of misalignment (the `wpe` fingerprint of the original slot is gone), but introduces a different one (the rotation angle of the original slot is still baked in). Net: sliding-window went from regression (+3.1%) to neutral (−0.06%) but did not become beneficial.

### 2.2 Closed-form re-rotation

RoPE rotation has two crucial algebraic properties:
- **Composition:** $R(a)\,R(b) = R(a+b)$ (rotation matrices commute under angle addition).
- **Inverse:** $R(a)^{-1} = R(-a)$ (orthogonal).

So given a K that was rotated by $R(p_{\text{orig}}\theta)$, we want it rotated by $R(p_{\text{new}}\theta)$ instead. The composition is:

$$
R(p_{\text{new}}\theta) \cdot R(-p_{\text{orig}}\theta) = R((p_{\text{new}} - p_{\text{orig}})\theta)
$$

— a **single** rotation by the angle difference. We don't need to un-rotate then re-rotate as two separate operations; we just compute $\Delta\theta_p = (p_{\text{new}} - p_{\text{orig}}) \theta_p$ for each frequency $\theta_p$ and apply one rotation.

### 2.3 Computing the delta from cached cos/sin tables

We don't precompute angles for every possible delta. Instead, we use sum-of-angles identities directly on the cached cos/sin table values for $p_{\text{new}}$ and $p_{\text{orig}}$:

$$
\cos(\Delta p \cdot \theta_p) = \cos(p_{\text{new}} \theta_p)\cos(p_{\text{orig}} \theta_p) + \sin(p_{\text{new}} \theta_p)\sin(p_{\text{orig}} \theta_p)
$$

$$
\sin(\Delta p \cdot \theta_p) = \sin(p_{\text{new}} \theta_p)\cos(p_{\text{orig}} \theta_p) - \cos(p_{\text{new}} \theta_p)\sin(p_{\text{orig}} \theta_p)
$$

This is what `_msa_rope_rotate_inplace()` in `src/microgpt_msa.c` computes per token per head. Cost: 4 mul + 1 add for each frequency pair; 4 mul + 2 add for each (a, b) rotation. At the deep config (n_layer=4, n_head=4, head_dim=16, ROPE_DIMS=16, recency=32 tokens), the entire rotation pass per chunking event is ~`32 × 4 × 4 × 8` ≈ 4K ops — negligible.

### 2.4 Why this only fixes the recency, not the pool

The recency window stores **individual token K vectors**, each with a single, well-defined original position. Re-rotation works cleanly: there is exactly one $p_{\text{orig}}$ to remap from.

The MSA pool (`MsaPool`) stores **mean-pooled K vectors**, averaged across multiple tokens that were rotated by *different* angles. The mean of rotated vectors is NOT the same as the rotation of a mean — there's no single $p_{\text{orig}}$ to un-rotate by. Fixing the pool path requires a more invasive change (e.g., un-rotating each token's K back to position 0 *before* averaging, then re-rotating after expansion). This paper deliberately stops at the recency-window fix; the pool fix is a separate, follow-up port.

So the experimental claim is precisely:
- The recency window's contribution to MSA quality is finally usable with RoPE.
- The pool's chunk-injection contribution is still position-mangled and contributes its own (unchanged) error.
- Net: sliding-window is now beneficial at the margin, even though absolute MSA-with-RoPE PPL remains worse than MSA-without-RoPE for unrelated pool reasons.

---

## 3. Implementation

### 3.1 Files changed

| File | Change |
|---|---|
| `src/microgpt_msa.h` | Added `size_t *positions` to `MsaRecency`. Updated `msa_recency_push` signature to take a `pos` argument. New `msa_recency_inject_rope(rec, active_keys, active_values, start_pos, n_head)` declaration. Detailed doc-comment explaining the rotation-by-delta algebra. |
| `src/microgpt_msa.c` | (1) Allocate/free `positions[capacity]`. (2) `msa_recency_push` stores the pos. (3) Local cos/sin tables with same `BLOCK_SIZE × ROPE_DIMS/2` layout as `microgpt.c`, lazily populated. (4) `_msa_rope_rotate_inplace(x, head_dim, p_orig, p_new)` applies the delta rotation via sum-of-angles. (5) `msa_recency_inject_rope` walks the ring chronologically, computes `p_new = start_pos + i`, and rotates each cached K per head. Falls back to plain `msa_recency_inject` when RoPE isn't compiled in OR when `n_head <= 0`. |
| `tests/bench_microgpt_msa_sliding.c` | (a) `msa_recency_push` call site now passes `pos`. (b) `msa_step_sliding` uses `msa_recency_inject_rope(...)` when `MICROGPT_PARTIAL_ROPE` is defined; the legacy non-RoPE branch falls back to plain `msa_recency_inject`. |
| `CMakeLists.txt` | New `bench_rope_msa_sliding_rerotate` target — same flags as `bench_rope_msa_sliding_on`, but built after the integration update so its binary measures the post-fix behaviour for clean comparison. |

### 3.2 The `msa_recency_inject_rope` core

```c
size_t msa_recency_inject_rope(const MsaRecency *rec,
                               scalar_t **active_keys,
                               scalar_t **active_values,
                               size_t start_pos, int n_head) {
    if (!rec || rec->length == 0) return 0;
    if (n_head <= 0) return msa_recency_inject(rec, active_keys, active_values, start_pos);

    _msa_rope_tables_init();
    size_t head_dim = (size_t)(rec->n_embd / n_head);
    size_t start = (rec->length < rec->capacity) ? 0 : rec->head;

    for (size_t i = 0; i < rec->length; i++) {
        size_t ring   = (start + i) % rec->capacity;
        size_t p_orig = rec->positions[ring];
        size_t p_new  = start_pos + i;
        for (int l = 0; l < rec->n_layer; l++) {
            /* Copy K and V into active cache at p_new */
            ...copy...
            /* For each head, rotate the K slice by (p_new - p_orig) */
            for (int h = 0; h < n_head; h++) {
                _msa_rope_rotate_inplace(dst_k + h * head_dim, head_dim, p_orig, p_new);
            }
            /* V is unchanged — RoPE doesn't touch V. */
        }
    }
    return rec->length;
}
```

The function is fully orthogonal to the existing `msa_recency_inject` path. Callers that don't have access to `n_head` or that don't enable RoPE continue working exactly as before.

### 3.3 Verification

```
$ ./test_microgpt
=== Results: 61/61 passed ===

$ ./test_microgpt_msa
Running suite: MSA Memory Sparse Attention Primitives
All tests finished. Passed: 3, Failed: 0
```

The test suites don't exercise the recency-window path directly (the unit tests focus on core engine + MSA primitives), but their continued passing confirms the additive change didn't break anything else. The benchmarks below provide the actual quality measurement.

---

## 4. Benchmark Design

Same long-context harness used in [the prior MSA papers](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md). 4-layer 138K-param char model on names, 1500 train steps, 2048-token held-out stream, 62–64 MSA chunking events. Identical seed and data across all variants.

### 4.1 Four-cell A/B/C/D matrix

To isolate the contribution of the re-rotation fix, the comparison spans both the RoPE-off and RoPE-on regimes:

| Cell | RoPE | Sliding | Re-rotate fix | Notes |
|---|---|---|---|---|
| A | OFF | OFF | n/a | Pre-V4 baseline (best in non-RoPE world) |
| B | OFF | ON | n/a | Original sliding regression |
| C | ON | OFF | n/a | RoPE-only baseline (fresh comparison anchor) |
| **D** | **ON** | **ON** | **YES** | **The unblock-fix experiment** |

Reading: A/B reproduces the original sliding-window paper's negative result; C reproduces the Partial RoPE paper's "neutral" result for sliding (because *that* paper used the legacy non-rotating inject); D is new — measures whether the re-rotation fix flips the sliding-window port to net-positive within the RoPE regime.

### 4.2 What "net-positive" means

The natural comparison for D is **C** (same RoPE regime, no sliding) — measures the marginal value of adding sliding-window-recency on top of the V4 RoPE port. A direct comparison to A or B is misleading because the RoPE-on absolute PPL is shifted (RoPE has its own MSA-specific issues separate from the recency-injection issue this paper fixes — see §5.4 below).

---

## 5. Results

All numbers from the actual benchmark binaries built and run in this repository. Identical seed, identical data, 62–64 chunk events on a 2048-token stream.

### 5.1 The four-cell matrix

```
A: ./bench_msa_sliding_long_baseline              RoPE OFF, sliding OFF
   POST_CHUNK_PERPLEXITY: 18.217   chunk_events: 62
   FINAL_TRAIN_LOSS: 2.329

B: ./bench_msa_sliding_long_on                    RoPE OFF, sliding ON
   POST_CHUNK_PERPLEXITY: 18.786   chunk_events: 64
   FINAL_TRAIN_LOSS: 2.329

C: ./bench_rope_msa_sliding_baseline              RoPE ON,  sliding OFF
   POST_CHUNK_PERPLEXITY: 19.861   chunk_events: 62
   FINAL_TRAIN_LOSS: 2.263

D: ./bench_rope_msa_sliding_rerotate              RoPE ON,  sliding ON, re-rotate
   POST_CHUNK_PERPLEXITY: 19.272   chunk_events: 64
   FINAL_TRAIN_LOSS: 2.263
```

### 5.2 Within-regime comparisons

**RoPE-OFF regime (A vs B):**

| | Post-chunk PPL | Δ |
|---|---:|---:|
| Sliding OFF | 18.22 | — |
| Sliding ON  | 18.79 | **+3.1% (regression)** |

This reproduces the original sliding-window paper's finding. Sliding hurts when RoPE isn't active.

**RoPE-ON regime (C vs D):**

| | Post-chunk PPL | Δ |
|---|---:|---:|
| Sliding OFF | 19.86 | — |
| **Sliding ON + re-rotate** | **19.27** | **−3.0% (improvement)** ✓ |

The re-rotation fix flips the sign within the RoPE regime. Sliding now helps by 3.0%, equal in magnitude (and opposite in sign) to the original regression.

### 5.3 The sliding-window port's complete history

| Stage | Configuration | Sliding-vs-no-sliding within same regime | Status |
|---|---|---:|---|
| 1. Original | RoPE off, plain inject | **+3.1% (regression)** | shipped as deferred |
| 2. After Partial RoPE | RoPE on, plain inject | −0.06% (neutral) | unblocked but not yet positive |
| 3. **After this paper** | **RoPE on, re-rotation fix** | **−3.0% (improvement)** ✓ | **net-positive** |

The two-step prediction from the Partial RoPE paper §6.4 is fully confirmed: RoPE alone made the regression go away; RoPE + the re-rotation fix made the port net-positive.

### 5.4 The remaining puzzle: RoPE-on absolute PPL is worse than RoPE-off

Comparing across regimes (NOT a within-regime comparison):

| Variant | Post-chunk PPL |
|---|---:|
| A: RoPE off, no sliding | 18.22 |
| C: RoPE on, no sliding | 19.86 |
| D: **RoPE on, sliding + re-rotate (this paper)** | **19.27** |

D is better than C by −3% but still worse than A by +5.8%. Why is RoPE-on absolute MSA PPL higher than RoPE-off in the first place?

The cause is **not** the recency-window injection (which D now handles correctly). It's the **MSA pool's `msa_expand_context`** — which copies a mean-pooled K vector to position 0 of the active cache. That mean-pool aggregates K vectors that were each rotated by different angles, so the result has no well-defined rotation. When the model's Q at pos=0 dot-products with that mean-pooled K, the geometric meaning is ambiguous in the RoPE regime — significantly more so than in the non-RoPE regime, because the model has been trained to rely on rotation-encoded relative position rather than wpe-encoded absolute position. So pool-derived chunks become a noisier signal.

Fixing the pool is a separate, deeper port (it requires un-rotating each token's K back to position 0 before averaging, which means the pool needs absolute-position information for every contributing token). This paper deliberately stops at the recency fix.

### 5.5 Summary

This paper closes the loop on the V4 port series' deferred item. Within the configuration the V4 port series ultimately recommends (`MICROGPT_PARTIAL_ROPE=ON`), the sliding-window recency port is now usable, by ~3% PPL. The earlier decisions to defer it are validated — without RoPE it's a regression, with RoPE alone it's neutral, with the proper re-rotation fix it's beneficial.

---

## 6. Updated Recommended V4 Stack

The Partial RoPE paper's recommendation:

```cmake
MICROGPT_PARTIAL_ROPE=1
MICROGPT_ATTN_SINK=1   ATTN_SINK_LOGIT=-1.0
MICROGPT_QK_NORM=1
MSA_POOL_MODE=3
```

This paper extends it with the sliding-window recency port now opt-in:

```cmake
# Active-attention path — biggest wins (8.7% combined, see RoPE paper).
MICROGPT_PARTIAL_ROPE=1
MICROGPT_ATTN_SINK=1   ATTN_SINK_LOGIT=-1.0
MICROGPT_QK_NORM=1
# MSA-internal — small wins, only one of these two helps.
MSA_POOL_MODE=3
# MSA recency — now usable thanks to RoPE + this paper's re-rotation fix.
# Use msa_recency_inject_rope() (not msa_recency_inject) at injection.
# In bench_microgpt_msa_sliding.c the choice is automatic via #ifdef.
```

For demos that integrate MSA directly (not via the benchmark harness), the integration code must explicitly call `msa_recency_inject_rope()` instead of `msa_recency_inject()` to opt in to the re-rotation fix.

---

## 7. Reproducing the Results

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cd build

cmake --build . --config Release --parallel 8 --target \
    bench_msa_sliding_long_baseline bench_msa_sliding_long_on \
    bench_rope_msa_sliding_baseline bench_rope_msa_sliding_rerotate

# Cell A: RoPE off, sliding off
./bench_msa_sliding_long_baseline      # PPL ~ 18.22
# Cell B: RoPE off, sliding on (original regression)
./bench_msa_sliding_long_on            # PPL ~ 18.79
# Cell C: RoPE on, sliding off
./bench_rope_msa_sliding_baseline      # PPL ~ 19.86
# Cell D: RoPE on, sliding on, re-rotate fix
./bench_rope_msa_sliding_rerotate      # PPL ~ 19.27 ✓
```

All deterministic for a given seed.

---

## 8. Limitations and Future Work

1. **Single corpus, single hardware, single seed.** Same caveat as the rest of the V4 port series.
2. **MSA pool path is not fixed.** §5.4 explains: re-rotating mean-pooled chunks requires un-rotating each token's K to position 0 *before* averaging, which means tracking absolute positions through the pool. Implementing this would require: (a) `MsaPool` storing per-token positions; (b) `msa_pool_chunk` un-rotating each contributing K before averaging; (c) `msa_expand_context` re-rotating the result at the new injection slot. Likely 100–200 LOC, separate paper.
3. **The cross-regime comparison (RoPE-on vs RoPE-off MSA) remains unfavourable.** Even with this paper's fix, RoPE-on absolute MSA PPL (19.27) is worse than RoPE-off MSA PPL (18.22). For long-context inference where the recommended V4 stack is used, this isn't directly a problem (the RoPE-on standard-inference PPL is much better than RoPE-off — see §5.2 of the Partial RoPE paper). But it does say that under heavy MSA chunking, the V4 stack's overall benefit is partly offset by pool-rotation noise. Fixing the pool path (above) would address this.
4. **Recency-window size not swept.** This paper used `MSA_WIN=32` to match the prior paper. Whether smaller or larger windows do better with the re-rotation fix is unmeasured.
5. **No interaction with attention sink or Q/K RMSNorm.** The full V4 stack might further amplify or dampen the sliding-window benefit. Untested.

---

## 9. References

- DeepSeek-V4 paper: [`papers/DeepSeek_V4.pdf`](papers/DeepSeek_V4.pdf), §2.3.3 "Additional Branch of Sliding Window Attention" + §2.3.3 "Partial Rotary Positional Embedding".
- Roadmap context: [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) §3.4.
- Companion papers (the V4 port series — now extended with this fix-up):
  - [`RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md)
  - [`RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md`](RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md)
  - [`RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md`](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md) — the original deferred port.
  - [`RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md`](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md)
  - [`RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md`](RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md)
  - [`RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md`](RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md) — predicted this fix would work; partially confirmed in §5.4 of that paper, fully confirmed in this paper.
- Implementation:
  - `src/microgpt_msa.h` — `MsaRecency.positions` field, `msa_recency_inject_rope()` declaration.
  - `src/microgpt_msa.c` — local cos/sin tables, `_msa_rope_rotate_inplace()`, `msa_recency_inject_rope()`.
  - `tests/bench_microgpt_msa_sliding.c` — push site passes pos, sliding integration uses rope-aware inject under `#ifdef MICROGPT_PARTIAL_ROPE`.
  - `CMakeLists.txt` — `bench_rope_msa_sliding_rerotate` target.

---

*The deferred port closes. The prediction was correct. The recency window was always the right idea — it just needed RoPE *and* the rotation algebra to be done correctly at injection time. Two papers and one closed loop later, MicroGPT-C MSA finally has the V4 sliding-window recency benefit as designed.*
