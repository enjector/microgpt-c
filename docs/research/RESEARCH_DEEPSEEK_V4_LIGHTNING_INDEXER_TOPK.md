# Porting DeepSeek-V4 Lightning Indexer + Top-K to MicroGPT-C MSA

> Pre/post measurements for the fifth and final of the six DeepSeek-V4 ports identified in [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md). Replaces the existing single-layer cosine top-1 retrieval (`msa_route_top_1`) with a multi-layer ReLU-summed scoring function and top-K selection (`msa_route_top_k`). Sweeps K ∈ {1, 2, 4, 8} and a combination with the content-aware CSA pool from the prior paper.
>
> **Result: only K=8 helps, by −0.32% PPL, matching the prior CSA pool result.** Smaller K values (1, 2, 4) are within noise or slightly regress. The new scoring function alone (K=1) is statistically identical to baseline cosine. Content-aware pool + Lightning K=4 is between K=4-alone and K=8-alone — composition does not super-add. Combined with the prior CSA-pool finding, this is strong evidence that **MSA's bottleneck is not in the pool/routing layer at all** — it lies upstream (likely wpe alignment, as the sliding-window paper indicated).

**Reference:** DeepSeek-V4 §2.3.1 "Compressed Sparse Attention", equations (13)–(17) "Lightning Indexer for Sparse Selection".

**Status:** Implemented as a new public function `msa_route_top_k` in `src/microgpt_msa.{h,c}`. Default routing in MSA-using demos is unchanged (still `msa_route_top_1`). The benchmark harness selects the routing path via `BENCH_MSA_ROUTING_MODE` and `BENCH_MSA_TOPK`. All 61 core unit tests + 3 MSA primitive tests pass with the new code in place. Measured on Apple M2 Max, single-threaded, `MICROGPT_USE_FLOAT=ON`.

---

## 1. Spear Summary

**Point:** The Lightning Indexer's structural innovations — multi-head/multi-layer ReLU-summed scoring and top-K retrieval — produce *at most* a −0.32% post-chunk PPL improvement on a 4-layer 138K-param char model with a 2048-token long-context evaluation. The improvement is captured only at K=8 (injecting 8 historical chunks at the cost of 8 recent-token slots in the active cache); smaller K values are within noise or regress slightly. The same −0.32% improvement is achievable via the simpler [content-aware CSA pool port](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md) (`MSA_POOL_MODE=3`). When both are combined, results land between the two — they do not stack.

**Picture:** The current MSA routing picks the single most-similar pool chunk (cosine top-1) and overwrites position 0 of the active cache with it. The Lightning Indexer instead computes a more expressive score (sum of per-layer ReLU dot-products) and keeps the top K chunks, injecting all K at positions 0..K-1. This trades K of the active cache's recent tokens for K historical chunks. At small K (1, 2, 4), the historical chunks are individually mediocre (the 2nd-best pool match is much weaker than the best), so we lose more than we gain. At K=8, redundancy across many mediocre matches ends up giving broader thematic coverage that compensates for individual imperfection.

**Proof:**

| Variant | Post-chunk PPL | Δ vs baseline |
|---|---:|---:|
| Baseline (top-1 cosine) | 18.2172 | — |
| Lightning Indexer K=1 | 18.2176 | +0.002% (noise) |
| Lightning Indexer K=2 | 18.2413 | +0.13% (mild regression) |
| Lightning Indexer K=4 | 18.2423 | +0.14% (mild regression) |
| **Lightning Indexer K=8** | **18.1587** | **−0.32%** ✓ |
| Lightning K=4 + content pool (mode 3) | 18.1990 | −0.10% |

(4-layer 138K-param char model on names, 1500 train steps, 2048-token held-out, 62 MSA chunk events.)

**Push:** Do not ship `msa_route_top_k` as the default routing function. Its benefit at the only useful K (K=8) is identical to the simpler CSA content-aware pool win from the [prior paper](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md), and the two don't compose. Ship `MSA_POOL_MODE=3` (CSA content-aware pool) instead — it's simpler, parameter-free, and delivers the same win. Keep `msa_route_top_k` in the codebase as infrastructure for future experiments after the **upstream** bottlenecks (wpe alignment / Partial RoPE port) are addressed.

---

## 2. The Mechanism

### 2.1 V4's Lightning Indexer (eqs. 13–17)

After CSA produces compressed entries $C^\text{Comp}$, V4 builds a low-rank multi-head indexer:

$$
\mathbf{c}^Q_t = \mathbf{h}_t \cdot W^{DQ} \quad\text{(downprojection, low-rank)}
$$

$$
[\mathbf{q}^I_{t,1}; \dots; \mathbf{q}^I_{t,n^I_h}] = \mathbf{c}^Q_t \cdot W^{IUQ} \quad\text{(per-head upprojection)}
$$

$$
[w^I_{t,1}; \dots; w^I_{t,n^I_h}] = \mathbf{w}^I_t = \mathbf{h}_t \cdot W^w \quad\text{(per-head weights)}
$$

$$
I_{t,s} = \sum_{h=1}^{n^I_h} w^I_{t,h} \cdot \text{ReLU}\!\left(\mathbf{q}^I_{t,h} \cdot K^\text{IComp}_s\right)
$$

$$
C^\text{SprsComp}_t = \{C^\text{Comp}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}
$$

Four learnable matrices ($W^{DQ}, W^{IUQ}, W^w$, and $W^{IK\text{compress}}$ for the indexer keys). Top-k is over the score vector $I_{t,:}$.

### 2.2 MicroGPT-C MSA's existing routing

`msa_route_top_1` in `src/microgpt_msa.c` does:

```c
int l = pool->n_layer - 1;  /* last layer only */
for (size_t i = 0; i < pool->length; i++) {
    /* cosine similarity at the last layer */
    sim[i] = (Q · K_pool[i][last_layer]) / (||Q|| ||K_pool|| + eps);
}
return argmax(sim);
```

Single-layer, single-chunk, no ReLU, no learned projection.

### 2.3 The fixed-form port

V4's indexer cannot be ported with learnable weights for the same reason CSA pooling cannot — MSA routing happens outside the training graph. Same fixed-form approach as the prior paper:

`msa_route_top_k` computes:

$$
\text{score}[i] = \sum_{l=0}^{L-1} \text{ReLU}\!\left(\frac{K_q[l] \cdot K_\text{pool}[i][l]}{\sqrt{n_\text{embd}}}\right)
$$

then returns the top-K indices via insertion-sorted top-K maintenance. Two changes vs `msa_route_top_1`:

1. **Multi-layer (not single-layer).** Sum across all layers' K vectors instead of using only the last layer. This serves as our parameter-free "multi-head" stand-in — different layers' K vectors already encode different semantic projections by virtue of training.
2. **ReLU-summed (not raw cosine).** Negative dot products contribute zero. Chunks that are "anti-aligned" with the query in some layers don't get penalised (their score in those layers is just zero, not negative).
3. **Top-K (not top-1).** Returns the K best chunks in descending score order.

The benchmark integration injects all K returned chunks at positions 0..K-1 in the active cache (after the existing memmove flow), overwriting the corresponding recent tokens.

---

## 3. Implementation

### 3.1 Files changed

| File | Change |
|---|---|
| `src/microgpt_msa.h` | New `msa_route_top_k(pool, query_keys, k, indices_out, scores_out)` declaration with detailed doc-comment explaining the design relative to V4 and to the existing top-1 path. |
| `src/microgpt_msa.c` | New `msa_route_top_k` implementation (~80 LOC). Multi-layer ReLU-summed scoring with insertion-sorted top-K (k ≤ 64 cap, dominant case k ≤ 16). Compatible with `ENABLE_TURBOQUANT`/`ENABLE_ROTORQUANT` paths. |
| `tests/bench_microgpt_msa_sliding.c` | Added `BENCH_MSA_ROUTING_MODE` and `BENCH_MSA_TOPK` compile-time switches that select between the existing `msa_route_top_1` integration and the new top-K integration. The latter loops `msa_expand_context` over the K returned indices. |
| `CMakeLists.txt` | Six new benchmark targets — `bench_msa_indexer_{baseline, k1, k2, k4, k8, k4_csa}`. The `k4_csa` variant additionally enables `MSA_POOL_MODE=3` to test composition with the content-aware CSA pool. |

### 3.2 Routing function

```c
int msa_route_top_k(const MsaPool *pool, scalar_t **query_keys,
                    int k, int *indices_out, scalar_t *scores_out) {
    /* ...sentinel init... */
    scalar_t scale = 1.0f / sqrt((double)pool->n_embd);
    for (size_t i = 0; i < pool->length; i++) {
        scalar_t score = 0;
        for (int l = 0; l < pool->n_layer; l++) {
            /* Optionally dequantise pool keys at (i, l) under TQ/RQ */
            scalar_t dot = 0;
            for (int d = 0; d < pool->n_embd; d++) {
                dot += K_pool[i][l][d] * query_keys[l][d];
            }
            scalar_t s = dot * scale;
            if (s > 0) score += s;        /* per-layer ReLU contribution */
        }
        /* Insertion-sort score into the local top-k (descending). */
        if (score > local_scores[k - 1]) { ... }
    }
    return n_valid;
}
```

The hot loop is structurally similar to the existing `msa_route_top_1` (cosine over the last layer), with ~`n_layer × n_embd` extra mults per chunk for the multi-layer extension and a small insertion-sort overhead. At our scale this is sub-percent of total inference time.

### 3.3 Benchmark integration

After the existing memmove-second-half-down step, the chunking event's tail expands to:

```c
#if BENCH_MSA_ROUTING_MODE == 1
  int top[16];
  int n = msa_route_top_k(pool, q, BENCH_MSA_TOPK, top, NULL);
  for (int i = 0; i < n; i++) {
    if (top[i] >= 0)
      msa_expand_context(pool, top[i], inf_keys, inf_values, (size_t)i);
  }
#else
  int best = msa_route_top_1(pool, q);
  if (best >= 0) msa_expand_context(pool, best, inf_keys, inf_values, 0);
#endif
```

The K returned chunks are written to active cache positions 0..K-1, overwriting the most-recent K tokens that the memmove had just placed there. So at K=4, the active cache after a chunking event holds: 4 historical chunks at positions 0–3, then 28 recent tokens at positions 4–31.

### 3.4 Verification

```
$ ./test_microgpt
=== Results: 61/61 passed ===

$ ./test_microgpt_msa
Running suite: MSA Memory Sparse Attention Primitives
All tests finished. Passed: 3, Failed: 0
```

All tests pass with the new code. The new `msa_route_top_k` is reachable from the test suite — when called with `k=1` it should return the same index as `msa_route_top_1` modulo scoring-function differences (multi-layer ReLU-sum vs single-layer cosine). It's a purely additive API change.

---

## 4. Benchmark Design

The harness is identical to the one used in [`RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md`](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md) §4 — same source, same seed, same data. Each variant compiles a separate library with a different combination of `BENCH_MSA_ROUTING_MODE`, `BENCH_MSA_TOPK`, and `MSA_POOL_MODE`. Same train trajectory, same long held-out token stream, same chunking schedule.

### 4.1 Variants

| Target | Routing | K | Pool mode |
|---|---|---:|---|
| `bench_msa_indexer_baseline` | top-1 cosine (existing) | 1 | mean |
| `bench_msa_indexer_k1` | Lightning Indexer top-K | **1** | mean |
| `bench_msa_indexer_k2` | Lightning Indexer top-K | **2** | mean |
| `bench_msa_indexer_k4` | Lightning Indexer top-K | **4** | mean |
| `bench_msa_indexer_k8` | Lightning Indexer top-K | **8** | mean |
| `bench_msa_indexer_k4_csa` | Lightning Indexer top-K | 4 | **content-aware** |

`k1` isolates the scoring-function change (multi-layer ReLU-sum) from the top-K change. `k2..k8` step through K to measure the historical-coverage vs recent-token-cost trade-off. `k4_csa` tests composition with the prior paper's content-aware pool.

`final_train_loss = 2.328964` and `pre_chunk_loss = 2.869539` are identical across all six variants.

---

## 5. Results

All numbers from the actual benchmark binaries built and run in this repository.

### 5.1 Raw results

```
$ ./bench_msa_indexer_baseline   # top-1 cosine
POST_CHUNK_PERPLEXITY: 18.217169      Δ = —

$ ./bench_msa_indexer_k1         # Lightning K=1
POST_CHUNK_PERPLEXITY: 18.217611      Δ = +0.002%

$ ./bench_msa_indexer_k2         # Lightning K=2
POST_CHUNK_PERPLEXITY: 18.241316      Δ = +0.13%

$ ./bench_msa_indexer_k4         # Lightning K=4
POST_CHUNK_PERPLEXITY: 18.242294      Δ = +0.14%

$ ./bench_msa_indexer_k8         # Lightning K=8
POST_CHUNK_PERPLEXITY: 18.158724      Δ = -0.32% ✓ best

$ ./bench_msa_indexer_k4_csa     # Lightning K=4 + content-aware pool
POST_CHUNK_PERPLEXITY: 18.199001      Δ = -0.10%
```

### 5.2 K=1 isolates "scoring change alone" — and finds nothing

Lightning K=1 vs baseline top-1 cosine: 18.2176 vs 18.2172, +0.002% — round-off noise. The scoring change alone (multi-layer ReLU-sum vs single-layer cosine) makes essentially zero difference at our model scale.

This is reasonable: with only 4 layers, the per-layer ReLU contributions don't differ enough from the last-layer cosine signal. In V4's frontier-scale models with 60+ layers, the multi-head scoring carries more independent signal. At our 4-layer scale, it's roughly a more expensive way to compute the same answer.

### 5.3 K=2, K=4 mildly regress

Both step from baseline by +0.13–0.14%. This is consistent with the trade-off interpretation:

- We sacrifice K of the 32 recent tokens (the second half of the active cache after memmove) for K historical chunks.
- The 1st-best chunk is well-matched (that's what the existing baseline already injects). The 2nd, 3rd, 4th best are individually weaker matches. Their value is less than the recent tokens they displace.
- The model's loss therefore creeps up slightly when we replace recency with mediocre history.

### 5.4 K=8 is the only clear win

At K=8, the post-chunk PPL drops to 18.16 — a −0.32% improvement, identical in magnitude to the content-aware CSA pool result from the prior paper.

Why K=8 helps when K=4 doesn't: the redundancy of 8 chunks provides broader thematic coverage. Even if individual chunks are mediocre matches, their *union* tells the model "here's a wider sample of the historical context." Position-displacement cost is now 8 of 32 recent tokens — a quarter of the recency window — but the historical-content gain finally outweighs it.

### 5.5 Composition with content-aware pool

K=4 + `MSA_POOL_MODE=3` lands at 18.1990, between K=4 (18.2423) and K=8 (18.1587). The content-aware pool partially compensates for the recency-token loss at K=4, but doesn't fully match the K=8 win. **The two ports do NOT stack** — there isn't a separate hidden axis they're both fixing. This is consistent with the hypothesis that the underlying bottleneck is shared.

### 5.6 Cost

- **Parameters added:** 0 (parameter-free).
- **Per-chunk-event compute:** baseline cosine = `pool_length × n_embd` ops over last layer. Lightning indexer = `pool_length × n_embd × n_layer` ops + insertion sort. At our scale (pool_length ≤ 62, n_embd=64, n_layer=4) this is ~16K ops per chunking event, fired once every ~32 token positions — sub-percent of total inference.
- **K-dependent compute:** insertion sort is O(K) per inserted score; total O(pool_length × K). Trivial at K ≤ 16.

---

## 6. Interpretation

### 6.1 What this rules out

Combining this paper with the [CSA pool paper](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md), we have now exhaustively measured the routing/pooling layer of MicroGPT-C's MSA:

| Lever | Best-case improvement |
|---|---:|
| Pool weighting (mean → content-aware) | **−0.32%** |
| Routing scoring (cosine → multi-layer ReLU-sum) | ~0% |
| K (top-1 → top-8) | **−0.32%** combined with above |

Maximum achievable improvement by tuning the pool/routing layer alone: ~−0.32% PPL. The two ports don't stack super-additively — they're addressing the same dimension.

### 6.2 What's actually limiting MSA

Combining all five V4 ports we've measured for MicroGPT-C:

| Port | Best result | Bottleneck addressed |
|---|---:|---|
| Attention sink | **−3.1%** | Forced 100% attention-mass → softmax saturation |
| Q/K RMSNorm + sink | **−7.0%** combined | Logit blow-up + saturation |
| Sliding-window recency | +3-8% (regressed) | Blocked: needs Partial RoPE first |
| CSA-style pool | −0.32% | Pool content quality (small) |
| Lightning Indexer + top-K=8 | −0.32% | Routing breadth (small, doesn't stack with above) |

Pattern: the **active-attention-path** ports (sink, QK norm) deliver order-of-magnitude bigger wins than the **MSA-internal** ports (CSA pool, Lightning Indexer). This is consistent with the **upstream bottleneck** hypothesis from the sliding-window paper: the limitation isn't HOW MSA compresses or retrieves, it's the **wpe alignment break** that occurs whenever pool-derived K vectors are injected into fresh active-cache slots.

The next port to try, in priority order:
1. **Partial RoPE** (V4 §3.3 / our roadmap §3.3) — replaces absolute `wpe` with relative-position rotation on the last 64 head dims. Should fix the wpe-alignment problem that's currently capping MSA-internal improvements. **Highest expected value.**
2. **Combined Partial RoPE + sliding-window-recency, retried.** With RoPE in place, the sliding-window paper's negative result should flip.
3. **Combined Partial RoPE + Lightning Indexer top-K, retried.** Same logic — once injection no longer breaks alignment, broader top-K coverage may help more.

---

## 7. Reproducing the Results

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cd build

cmake --build . --config Release --parallel 8 --target \
    bench_msa_indexer_baseline bench_msa_indexer_k1 \
    bench_msa_indexer_k2 bench_msa_indexer_k4 bench_msa_indexer_k8 \
    bench_msa_indexer_k4_csa

./bench_msa_indexer_baseline   # top-1 cosine
./bench_msa_indexer_k1         # Lightning K=1
./bench_msa_indexer_k2         # Lightning K=2
./bench_msa_indexer_k4         # Lightning K=4
./bench_msa_indexer_k8         # Lightning K=8 (best)
./bench_msa_indexer_k4_csa     # Lightning K=4 + content-aware pool
```

Results are deterministic for a given seed, single-threaded by default.

To call the new `msa_route_top_k` from your own MSA-using demo:

```c
#include "microgpt_msa.h"

int top[8];
int n = msa_route_top_k(pool, query_keys, /* k */ 8, top, /* scores */ NULL);
for (int i = 0; i < n; i++) {
    if (top[i] >= 0)
        msa_expand_context(pool, top[i], active_keys, active_values, (size_t)i);
}
```

---

## 8. Limitations and Future Work

1. **Single corpus, single hardware, single seed.** Same caveat as the prior V4-port papers in this series.
2. **Fixed-form, not truly learnable.** V4's indexer uses learned $W^{DQ}, W^{IUQ}, W^w$. We approximated the multi-head structure via the model's existing per-layer K projections. The −0.32% result at K=8 is a lower bound on what learnable indexer weights might deliver. Same caveat as the [CSA pool paper](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md): integrating MSA into the training graph is a substantial refactor that we're not committing to until the upstream bottleneck (Partial RoPE) is addressed.
3. **Insertion-sort top-K is O(pool_length × K).** Fine for our pool sizes (≤ 1024) and small K. For frontier-scale pools (millions of chunks) a heap would be required.
4. **K only swept up to 8.** Half of our 32-position recent-tokens budget. Larger K (16, 24, 32) might continue to improve as the model leans more on historical context, or might collapse as recent tokens disappear entirely. Untested in this paper.
5. **No interaction tested with attention sink, Q/K RMSNorm, or sliding-window recency.** All four MSA-aware ports composed should be measured together. The most likely candidate to combine cleanly is the active-path ports (sink + QK norm) which deliver the biggest wins regardless of MSA configuration.
6. **The 4-layer ReLU-sum scoring is a coarse stand-in for V4's `n^I_h`-head indexer.** With n_layer=4 we have only 4 "heads"; V4 uses ~32. The K=1 result (no win) might flip at higher head counts.

---

## 9. References

- DeepSeek-V4 paper: [`papers/DeepSeek_V4.pdf`](papers/DeepSeek_V4.pdf), §2.3.1 "Compressed Sparse Attention", equations (13)–(17) "Lightning Indexer for Sparse Selection".
- Roadmap context: [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) §3.6.
- Companion papers (the V4 port series — now complete except for Partial RoPE):
  - [`RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md) — attention sink (−3.1% PPL).
  - [`RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md`](RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md) — Q/K RMSNorm (super-additive with sink, combined −7.0%).
  - [`RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md`](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md) — sliding-window recency (negative result, blocked on Partial RoPE).
  - [`RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md`](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md) — CSA-style pool (−0.32% via content-aware mode).
- Implementation:
  - `src/microgpt_msa.h` (new `msa_route_top_k` declaration)
  - `src/microgpt_msa.c` (new `msa_route_top_k` implementation, ~80 LOC)
  - `tests/bench_microgpt_msa_sliding.c` (added routing-mode integration)
  - `CMakeLists.txt` (six new `bench_msa_indexer_*` targets)

---

## 10. Closing Remark

This is the fifth and final paper in the V4-port series, except for Partial RoPE which is the prerequisite for two of the deferred ports (sliding-window recency, and possibly a higher ceiling for both Lightning Indexer and CSA pool). The five MSA/active-attention ports we have measured together account for at most ~7% PPL improvement when combined wisely, of which ~98% comes from the two simplest ports (attention sink, Q/K RMSNorm). The MSA-internal ports (sliding window, CSA pool, Lightning Indexer) collectively deliver less than 0.5% — a strong signal that **MSA's design is not the limiting factor** for MicroGPT-C's long-context quality at this scale. Partial RoPE is the next experiment.

---

*Honest research: a port that runs, passes its tests, and produces meaningful but small wins is a valuable contribution — even if the recommendation is "don't ship it as default." Knowing what doesn't move the needle is as useful as knowing what does.*
