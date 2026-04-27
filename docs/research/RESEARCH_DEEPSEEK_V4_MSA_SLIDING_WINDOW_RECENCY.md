# Porting DeepSeek-V4 Sliding-Window Recency Branch to MicroGPT-C MSA

> Pre/post measurements for the third of the six DeepSeek-V4 ports identified in [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md). Implementation, controlled A/B benchmark, sensitivity sweep, and a long-context scaling test.
>
> **Result: this port did not deliver the expected benefit on MicroGPT-C.** Across all tested window sizes (8, 16, 32 tokens) and sequence lengths (512 and 2048 tokens), the sliding-window-recency variant slightly *regresses* post-chunk perplexity by 3–8%. We document the implementation, the measurements, and the structural reason it doesn't transfer.

**Reference:** DeepSeek-V4 §2.3.3 "Additional Branch of Sliding Window Attention."

**Status:** Implemented as `MsaRecency` in `src/microgpt_msa.{h,c}` and integrated into the benchmark harness behind `BENCH_MSA_SLIDING_WINDOW=1`. All 61 core unit tests + 3 MSA primitive tests pass with the change. Measured on Apple M2 Max, single-threaded, `MICROGPT_USE_FLOAT=ON`.

---

## 1. Spear Summary

**Point:** A direct port of V4's sliding-window-recency branch into MicroGPT-C's MSA pipeline does **not** improve long-context perplexity. Post-chunk PPL on a 2048-token held-out stream goes from 18.22 (baseline MSA) to 18.79 with WIN=32 sliding recency — a 3.1% regression, consistent across window sizes and sequence lengths. The data structure works, the integration runs, but the architectural assumption underneath V4's design — separate attention branches with relative-position embeddings — does not hold in MicroGPT-C's existing MSA, which uses absolute `wpe` and an implicit memmove-based recency that the new explicit recency disrupts.

**Picture:** V4's sliding window assumes the attention layer has *two parallel branches* — one for compressed/sparse historical KV, one for uncompressed recency — and that each branch carries its own positional information (V4 uses partial RoPE on both). MicroGPT-C's MSA has only **one** attention path with one cache layout, and the model uses learned absolute `wpe`. Re-injecting recency tokens at "wrong" physical positions breaks the wpe alignment that the model implicitly relies on. The existing MSA flow accidentally avoids this because it `memmove`s recent KV vectors into adjacent physical slots, preserving absolute-position locality.

**Proof:**

| Sequence | Variant | Chunk events | Post-chunk PPL | Δ vs baseline |
|---|---:|---:|---:|---:|
| 512 tokens | baseline (memmove half) | 14 | 17.80 | — |
| 512 tokens | sliding WIN=8 | 9 | 19.23 | +8.0% |
| 512 tokens | sliding WIN=16 | 10 | 19.06 | +7.1% |
| 512 tokens | sliding WIN=32 | 15 | 19.13 | +7.5% |
| 2048 tokens | baseline | 62 | **18.22** | — |
| 2048 tokens | sliding WIN=32 | 64 | 18.79 | +3.1% |

**Push:** Do not ship the sliding-window-recency port as-is. The proper V4-style realisation would require *also* porting partial RoPE (V4 §3.3 in our roadmap) so the recency branch carries its own *relative* position information that doesn't conflict with the active cache's absolute positions. Recommend: revisit this port after Partial RoPE lands. Alternatively, re-cast as a "smaller chunk_size" variant of existing MSA (chunk only 16 oldest instead of 32), which delivers the same recency-preservation idea without disrupting `wpe` alignment — a separate experiment, not what V4 actually proposes.

---

## 2. The Mechanism

### 2.1 V4's sliding-window branch

V4 §2.3.3 "Additional Branch of Sliding Window Attention":

> In order to strictly preserve causality in CSA and HCA, each query attends to only preceding compressed KV blocks. Consequently, a query cannot access information from other tokens within its own compressed block. Meanwhile, recent tokens usually possess greater relevance to the query token in language modeling. For these reasons, we introduce a supplementary attention branch to both CSA and HCA in a sliding window manner, for better modeling of local dependencies. To be specific, for each query token, we additionally produce $n_\text{win}$ uncompressed KV entries corresponding to the recent $n_\text{win}$ tokens. In the core attention of CSA and HCA, these KV entries in the sliding window will be used along with the compressed KV entries.

The crucial structural detail — V4 has **two attention sub-paths per layer**:
1. The compressed branch (CSA or HCA): query attends to top-k compressed blocks selected by the lightning indexer.
2. The recency branch: query attends to the last $n_\text{win}$ uncompressed tokens.

Both branches contribute to the final attention output. They share queries but have disjoint KV sets and are concatenated before the softmax.

### 2.2 MicroGPT-C MSA's existing recency mechanism

MicroGPT-C's MSA (in `src/microgpt_msa.c` plus the integration in `demos/msa/infinite_shakespeare/main.c`) does not have two attention branches. It has one — the standard attention reads from the active KV cache (size `block_size`). The "recency" property emerges implicitly from the chunking flow:

```c
/* When pos hits block_size: */
size_t chunk_size = cfg.block_size / 2;
msa_pool_chunk(pool, inf_keys, inf_values, chunk_size);
for (int L = 0; L < nl; L++) {
    /* memmove the SECOND HALF (positions chunk_size..block_size-1) */
    /* down to physical positions 0..chunk_size-1 */
    memmove(inf_keys[L],
            inf_keys[L] + (chunk_size * cfg.n_embd),
            (cfg.block_size - chunk_size) * cfg.n_embd * sizeof(scalar_t));
    /* same for values */
    inf_cache_len[L] -= chunk_size;
}
pos -= chunk_size;
/* then expand the best-routed chunk into position 0 */
```

After this, the cache holds 32 tokens of *uncompressed recency* in slots 0..31 + one summary chunk overwritten at slot 0. The next forward at `pos = 32` writes into slot 32. From the model's perspective, the cache looks contiguous and the absolute position pattern (0, 1, 2, ..., 31, 32...) lines up with the wpe vectors that influenced the K projections at training time.

So MSA already has an **implicit sliding-window recency of size 32** (= block_size/2). The V4-style port is meant to make this explicit and *enlarge* it — but doing so cleanly turns out to require more than a recency ring buffer.

### 2.3 The port we implemented

A new `MsaRecency` ring buffer in `src/microgpt_msa.{h,c}`:

```c
typedef struct {
    scalar_t *keys;   /* [capacity, n_layer, n_embd] */
    scalar_t *values; /* [capacity, n_layer, n_embd] */
    size_t capacity; size_t length; size_t head;
    int n_layer; int n_embd;
} MsaRecency;
```

with three operations:
- `msa_recency_push(rec, kv_at_current_pos)` — append the just-computed K/V to the ring; on overflow, evict oldest.
- `msa_recency_inject(rec, active_keys, active_values, start_pos)` — copy the entire ring into the active cache in chronological order at `[start_pos, start_pos + length)`.
- `msa_recency_reset(rec)` — clear without freeing.

The benchmark integration (`tests/bench_microgpt_msa_sliding.c`) does the following at every chunking event when `BENCH_MSA_SLIDING_WINDOW=1`:

1. Pool ALL `block_size` active tokens into the MsaPool (no half/half split).
2. Wipe the active cache.
3. Inject the best-routed chunk at slot 0.
4. Inject the entire recency ring at slots 1..n_win.
5. Continue generation from pos = n_win + 1.

Compared to the baseline:
- The baseline keeps `block_size/2` (=32) recent tokens at their original physical slots (positions 32..63 → 0..31 via memmove).
- The sliding version keeps `n_win` (8/16/32) recent tokens in the ring, then re-injects them at slots 1..n_win after wiping the cache.

### 2.4 Where the port goes wrong

The K vectors in `MsaRecency` were computed at their *original* absolute positions $p_\text{orig}$. They embed the wpe of those positions: $K_t = W_K \cdot \text{RMSNorm}(W_E \cdot \text{tok}_t + \text{wpe}[p_\text{orig}])$. After re-injection at new physical slots $p_\text{new} \neq p_\text{orig}$, the model's attention computes:

$$
\text{score}_t = \hat q \cdot K_t
$$

where $\hat q$ is computed at the new pos (carrying its own wpe), but $K_t$'s wpe fingerprint corresponds to a much older position. The attention dot-product was implicitly trained assuming physical slot index ≈ absolute position. The sliding window breaks this alignment.

The baseline avoids this issue by accident: its memmove preserves *physical adjacency* of tokens that were already adjacent in absolute positions. The K vectors at the new slots 0..31 came from absolute positions 32..63; their wpe fingerprints encode "I'm in the second half of a sequence." When viewed from a new pos around 32, the geometry is consistent — the K vectors look like they're slightly further away than they would for a fresh sequence start, but the relative positional structure is preserved.

A proper V4-style port would handle this by adding **partial RoPE** to Q and K — making attention depend on *relative* (not absolute) position, so re-injected K vectors don't carry stale absolute-position fingerprints. RoPE is item §3.3 of [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) and is a separate, larger port.

---

## 3. Implementation

### 3.1 Files changed

| File | Change |
|---|---|
| `src/microgpt_msa.h` | New `MsaRecency` struct + 4 operations: `msa_recency_create / free / reset`, `msa_recency_push`, `msa_recency_inject`. Detailed doc-comment explaining design rationale and pointing to this paper. |
| `src/microgpt_msa.c` | Implementation of all four operations (~80 LOC). Ring-buffer semantics (eviction = head advance, length saturates at capacity). No changes to `MsaPool` — fully orthogonal. |
| `tests/bench_microgpt_msa_sliding.c` | New self-contained A/B benchmark — trains identical seeded char model on names, scores next-token CE on a long held-out token stream. Two integration paths gated by `BENCH_MSA_SLIDING_WINDOW`: baseline (existing memmove flow) vs sliding-window (pool-all-then-inject-best-chunk-plus-recency). |
| `CMakeLists.txt` | Six benchmark targets — baseline + WIN=8/16/32 at 512-token sequences, plus baseline + WIN=32 at 2048-token sequences for the scaling test. |

### 3.2 The recency ring (key snippets)

Push (append-or-evict-oldest semantics):

```c
void msa_recency_push(MsaRecency *rec,
                      scalar_t **token_keys, scalar_t **token_values) {
    if (rec->length < rec->capacity) {
        size_t slot = rec->length;          /* head still 0 while filling */
        _msa_recency_write_slot(rec, slot, token_keys, token_values);
        rec->length++;
    } else {
        _msa_recency_write_slot(rec, rec->head, token_keys, token_values);
        rec->head = (rec->head + 1) % rec->capacity;
    }
}
```

Inject (chronological copy-out):

```c
size_t msa_recency_inject(const MsaRecency *rec,
                          scalar_t **active_keys, scalar_t **active_values,
                          size_t start_pos) {
    if (!rec || rec->length == 0) return 0;
    /* Walk the ring chronologically: oldest = head when full, else 0. */
    size_t start = (rec->length < rec->capacity) ? 0 : rec->head;
    for (size_t i = 0; i < rec->length; i++) {
        size_t ring = (start + i) % rec->capacity;
        /* copy K and V across all layers into active[start_pos + i] */
    }
    return rec->length;
}
```

### 3.3 Verification

```
$ ./test_microgpt
=== Results: 61/61 passed ===

$ ./test_microgpt_msa
Running suite: MSA Memory Sparse Attention Primitives
All tests finished. Passed: 3, Failed: 0
```

Both core and MSA test suites pass with the new code in place. The new `MsaRecency` is a standalone data structure with no dependencies on existing MSA paths, so the change is non-invasive.

---

## 4. Benchmark Design

### 4.1 What we're measuring

Long-context next-token cross-entropy. We need a measurement that:
- Goes well past `block_size`, so MSA chunking is exercised many times.
- Is fully deterministic (same seed, same data path).
- Distinguishes "loss before any chunking" from "loss after chunking" so we can quantify how much information is lost at chunking events.

### 4.2 Method

`tests/bench_microgpt_msa_sliding.c`:

1. Train a 4-layer 138K-param char-level model on the names corpus (90% train split), 1500 Adam steps. Identical seed and data across both A and B variants — only `BENCH_MSA_SLIDING_WINDOW` differs.
2. Build a long held-out token stream by concatenating held-out names with BOS separators until `BENCH_LONG_SEQ_LEN` tokens are produced.
3. Feed the stream through `forward_inference` token by token. When `pos == block_size`, run the configured MSA chunking step.
4. At each token: softmax the lm_head logits and accumulate `−log p(target)` separately for positions before and after the first chunk event.

Reported metrics:
- `PRE_CHUNK_LOSS`: average CE on positions `0..block_size-1` (no chunking yet — control).
- `POST_CHUNK_LOSS`: average CE on positions `block_size..long_n-1` (chunking active).
- `POST_CHUNK_PERPLEXITY`: `exp(POST_CHUNK_LOSS)`.
- `CHUNK_EVENTS`: number of times MSA chunking ran.

The pre-chunk loss is identical across A/B variants by construction (no chunking has happened). Differences in post-chunk loss measure the integration's quality.

### 4.3 Variants

| Target | Sequence | WIN | Chunking flow |
|---|---:|---:|---|
| `bench_msa_sliding_baseline` | 512 | n/a | existing memmove half |
| `bench_msa_sliding_on_w8` | 512 | 8 | new sliding window |
| `bench_msa_sliding_on` (WIN=16) | 512 | 16 | new sliding window |
| `bench_msa_sliding_on_w32` | 512 | 32 | new sliding window |
| `bench_msa_sliding_long_baseline` | 2048 | n/a | existing memmove half |
| `bench_msa_sliding_long_on` | 2048 | 32 | new sliding window |

---

## 5. Results

All numbers are from the actual benchmark binaries built in this repository. `final_train_loss = 2.329`, `pre_chunk_loss = 2.870` are identical across all variants (same seed, same training).

### 5.1 512-token sequence sweep

| Variant | Chunk events | Post-chunk loss | Post-chunk PPL | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline | 14 | 2.879 | **17.80** | — |
| sliding WIN=8 | 9 | 2.956 | 19.23 | **+8.0%** |
| sliding WIN=16 | 10 | 2.948 | 19.06 | **+7.1%** |
| sliding WIN=32 | 15 | 2.951 | 19.13 | **+7.5%** |

**Reading.** All three sliding-window sizes regress relative to the baseline. The regression is roughly flat across WIN ∈ {8, 16, 32} — roughly 7% — suggesting the issue is **structural** (the integration disrupts something fundamental) rather than **hyperparameter-sensitive** (a particular WIN value is wrong). If WIN=32 (matched to baseline's implicit recency size) had recovered baseline performance, we'd conclude WIN=16 was just too small. It didn't, so the regression isn't about recency-window size.

### 5.2 2048-token scaling test (WIN=32)

| Variant | Chunk events | Post-chunk loss | Post-chunk PPL | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline | 62 | 2.902 | **18.22** | — |
| sliding WIN=32 | 64 | 2.933 | 18.79 | **+3.1%** |

**Reading.** At 4× longer sequences the baseline degrades (PPL 17.80 → 18.22, +2.4% — chunk-event accumulation slowly poisons the model). The sliding variant degrades less in absolute terms (PPL 19.13 → 18.79) — interesting, but it's still strictly worse than the baseline. The gap between sliding and baseline narrows from 7.5% (512 tokens) to 3.1% (2048 tokens), so the longer the context the more the sliding-window's structural issue is amortised. It still does not become net-positive.

### 5.3 What's happening

Two competing effects:

1. **Sliding window's explicit recency is more durable** than the baseline's implicit recency. The baseline's recency tokens get pooled away at every chunking event and depend on best-chunk routing for survival. The sliding window guarantees the last `n_win` tokens stay uncompressed across arbitrarily many chunking events.

2. **Sliding window's re-injection breaks `wpe` alignment.** Re-injected K vectors carry wpe fingerprints from their original (much earlier) absolute positions, but they sit at fresh small physical slots. The trained model's attention was implicitly learned with physical-slot ≈ absolute-position and is poorly served by this discontinuity.

In our benchmark, effect (2) outweighs effect (1) by a constant ~5–8% PPL margin. Effect (1) accumulates with sequence length (every chunk preserves recency that the baseline would have lost), so the gap shrinks at 2048 tokens. Extrapolating naively: the sliding window might break even or win at 16k+ tokens, but that's outside our test range and untested.

---

## 6. Why V4's Recipe Doesn't Transfer Cleanly

V4's sliding window is part of a coherent architectural package:

| V4 feature | What it does | MicroGPT-C status |
|---|---|---|
| Sliding window (§2.3.3) | Always-on uncompressed recency, attention reads BOTH recency and pool | **THIS PAPER** |
| Partial RoPE (§2.3.3) | Last 64 dims of Q/K rotated by *relative* position | Future work (port §3.3 in roadmap) |
| Two attention branches | Compressed pool + recency, concatenated before softmax | Not present — single attention path |
| Compressed-block "RoPE on output with -i" | Output of attention also carries relative-position via an inverse RoPE | Not present |
| Indexer-based top-k routing (§2.3.1) | Content-addressed retrieval, not just LRU | Future work (port §3.6 in roadmap) |

The sliding-window branch's value in V4 derives from the surrounding machinery — particularly **partial RoPE**, which makes attention positional behaviour insensitive to absolute slot index. Without RoPE, MicroGPT-C's attention is critically dependent on the absolute `wpe` fingerprint baked into K vectors, and re-injecting K vectors at new slots violates that contract.

This is consistent with the original V4 paper's design philosophy: the long-context efficiency story is a single integrated solution, and individual components don't necessarily yield wins in isolation. Compare with attention sinks (a single drop-in change with a self-contained win) and Q/K RMSNorm (a single change that helps stability without architectural co-dependence).

### 6.1 The recommended sequence

The right order to attempt this set of V4 ports for MicroGPT-C is now clearer:

1. ✅ **Attention sink** — done, +3% PPL win. Self-contained.
2. ✅ **Q/K RMSNorm** — done, stability win + super-additive with sink. Self-contained.
3. ⏭ **Partial RoPE** (§3.3 in roadmap) — should land before retrying sliding window. Provides relative-position attention that the sliding window depends on.
4. **Sliding window recency, retried** — once partial RoPE is in, retry this port; the wpe-alignment issue should disappear.
5. **CSA-style pooling** (§3.5 in roadmap) — replaces mean-pool with a learnable weighted pool. Should also benefit from RoPE.
6. **Lightning Indexer** (§3.6 in roadmap) — content-addressed top-k retrieval.

In short: this port is paused, not abandoned. The infrastructure (`MsaRecency`) is correct and reusable; the integration needs partial RoPE first.

---

## 7. Reproducing the Results

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cd build

cmake --build . --config Release --parallel 8 --target \
    bench_msa_sliding_baseline bench_msa_sliding_on \
    bench_msa_sliding_on_w8 bench_msa_sliding_on_w32 \
    bench_msa_sliding_long_baseline bench_msa_sliding_long_on

./bench_msa_sliding_baseline       # 512-token, no sliding
./bench_msa_sliding_on             # 512-token, WIN=16
./bench_msa_sliding_on_w8          # 512-token, WIN=8
./bench_msa_sliding_on_w32         # 512-token, WIN=32
./bench_msa_sliding_long_baseline  # 2048-token, no sliding
./bench_msa_sliding_long_on        # 2048-token, WIN=32
```

Results are deterministic for a given seed, single-threaded by default.

---

## 8. Limitations and Future Work

1. **Single corpus, single hardware, single seed.** Same caveat as previous V4 port papers.
2. **The negative result is conditional on the absence of Partial RoPE.** §6 explicitly predicts the result will flip once Partial RoPE is in. Re-running this benchmark after the RoPE port is the cleanest way to test that hypothesis.
3. **Scaling extrapolation is untested.** Effect (1) (durable recency) accumulates with sequence length; the 512 → 2048 trend (+8% → +3% gap) suggests the sliding window may eventually become competitive at much longer sequences. A 16K or 32K sequence test would settle this — but at that scale the baseline MSA's chunk-pool also degrades, so the comparison gets noisier.
4. **The bench's "best-chunk" injection happens after all-pool, while the baseline uses post-memmove K as query.** These two query-formulation choices are subtly different and may interact with the routing in ways that aren't clean. A more controlled test would use the same query formulation in both flows.
5. **Mean-pool quality is a confound.** Both the baseline and sliding window use the same `msa_pool_chunk` mean-pooling. If mean-pool is the limiting factor, no integration tweak will help; CSA-style learnable pooling (V4 §2.3.1, our roadmap §3.5) is the more important target.
6. **No interaction tested with attention sink or Q/K RMSNorm.** The combined paper trio (sink + Q/K norm + sliding window) may show different dynamics. Untested in this paper.

---

## 9. References

- DeepSeek-V4 paper: [`papers/DeepSeek_V4.pdf`](papers/DeepSeek_V4.pdf), §2.3.3 "Additional Branch of Sliding Window Attention".
- Roadmap context: [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) §3.4.
- Companion papers:
  - [`RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md) — attention sink (self-contained win).
  - [`RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md`](RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md) — Q/K RMSNorm (stability win, super-additive with sink).
- Existing MSA design: [`RESEARCH_MSA.md`](RESEARCH_MSA.md).
- Implementation:
  - `src/microgpt_msa.h` (new `MsaRecency` struct + ops)
  - `src/microgpt_msa.c` (implementation, ~80 LOC, no changes to `MsaPool`)
  - `tests/bench_microgpt_msa_sliding.c` (A/B benchmark)
  - `CMakeLists.txt` (six benchmark targets)

---

*Honest research: not every V4 idea drops cleanly into a different architectural family. We measured, found a structural issue, identified the prerequisite (Partial RoPE), and shelved the port until that's in. The infrastructure is in place; revisiting is straightforward.*
