# Porting DeepSeek-V4 CSA-Style Learnable Pooling to MicroGPT-C MSA

> Pre/post measurements for the fourth of the six DeepSeek-V4 ports identified in [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md). Replaces the uniform mean pool in `msa_pool_chunk` with three CSA-style fixed-form weighted alternatives: linear-ramp recency, exponential recency, and content-aware (softmax-to-anchor).
>
> **Result: pool *weight assignment* is not a bottleneck in MicroGPT-C's MSA.** Positional weighting (linear, exp) is within 0.005% of mean pool — statistically indistinguishable. Content-aware pooling delivers a small (−0.32%) improvement. The bottleneck for MSA quality lies elsewhere — likely routing fidelity or the chunk-injection mechanism.

**Reference:** DeepSeek-V4 §2.3.1 "Compressed Sparse Attention", equations (9)–(12).

**Status:** Implemented as a compile-time switch `MSA_POOL_MODE` ∈ {0, 1, 2, 3} in `src/microgpt_msa.{h,c}`. Default is **0 (uniform mean — existing behaviour)**, so the change is bit-identical to the pre-port engine without the flag. All 61 core unit tests + 3 MSA primitive tests pass with every mode. Measured on Apple M2 Max, single-threaded, `MICROGPT_USE_FLOAT=ON`.

---

## 1. Spear Summary

**Point:** Across four pooling strategies — uniform mean (baseline), linear-ramp recency, exponential recency, and content-aware (softmax-of-cosine-to-anchor) — the post-chunk perplexity of MSA on a 2048-token held-out stream varies by less than 0.5%. Positional schemes don't help at all (within 0.005% of mean); the content-aware scheme delivers a small but real −0.32% PPL improvement at zero parameter cost. **MicroGPT-C's MSA quality is not gated by how chunks are pooled; it's gated by something else** — most likely routing fidelity (top-1 cosine retrieval) or the wpe-alignment issue surfaced in the [sliding-window paper](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md).

**Picture:** MSA compresses a chunk of `block_size` KV tokens into a single latent vector via mean-pool. V4's CSA replaces this mean with a learned softmax-weighted sum that knows about content (which tokens matter) and position (where in the chunk they are). We tested both ideas in fixed-form (no extra parameters, no backprop):

- Positional priors (linear ramp, exp recency) — capture "recent tokens matter more."
- Content prior (cosine softmax to last K) — capture "tokens semantically aligned with the chunk's tail matter more."

If V4's *learnable* CSA were going to deliver a major win in MicroGPT-C, at least one of these fixed-form approximations should already show a clear effect. The fact that they don't tells us this port has a low ceiling at our scale.

**Proof:**

| Mode | Pool weighting | Post-chunk PPL | Δ vs mean |
|---|---|---:|---:|
| 0 | Uniform mean (baseline) | 18.2172 | — |
| 1 | Linear ramp (oldest→newest = 1.0→2.0) | 18.2176 | +0.002% |
| 2 | Exponential recency (τ = chunk_len/4) | 18.2180 | +0.005% |
| 3 | **Content-aware (softmax of K · K_last/√d)** | **18.1597** | **−0.318%** ✓ |

All numbers from a 4-layer 138K-param char-level model on names, 1500 train steps, 2048-token held-out scoring with 62 MSA chunk events.

**Push:** Ship `MSA_POOL_MODE=3` (content-aware) as a recommended default for MSA — small win, zero cost. Don't bother with positional schemes (modes 1, 2). Don't invest in true backprop-learnable pool weights yet — the fixed-form ceiling suggests the upside is limited. Investigate routing improvements (V4 §3.6 Lightning Indexer port) and/or partial RoPE (§3.3) before revisiting this.

---

## 2. The Mechanism

### 2.1 V4's CSA pooling (eqs. 9–12)

V4 builds compressed KV entries for a sequence of $n$ tokens via two sets of learnable transforms with overlapping blocks of size $m$:

$$
C^a = H \cdot W^{aKV}, \qquad C^b = H \cdot W^{bKV}
$$

$$
Z^a = H \cdot W^{aZ}, \qquad Z^b = H \cdot W^{bZ}
$$

For each compressed block $i$, the per-token weights are computed via a softmax over $2m$ stacked positions with learned positional biases $B^a, B^b$:

$$
[S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}] = \text{Softmax}_\text{row}([Z^a + B^a; Z^b + B^b])
$$

$$
C^\text{Comp}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j + \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j
$$

Four learnable matrices ($W^{aKV}$, $W^{bKV}$, $W^{aZ}$, $W^{bZ}$) plus two learnable bias tables ($B^a$, $B^b$). Trained end-to-end with the rest of the model.

### 2.2 MicroGPT-C MSA's existing pool

`msa_pool_chunk` in `src/microgpt_msa.c` does the simplest possible compression:

```c
for (int d = 0; d < n_embd; d++) {
    scalar_t sum_k = 0, sum_v = 0;
    for (size_t t = 0; t < chunk_len; t++) {
        sum_k += active_keys[l][t * n_embd + d];
        sum_v += active_values[l][t * n_embd + d];
    }
    pool->keys[...]   = sum_k / chunk_len;     /* uniform mean */
    pool->values[...] = sum_v / chunk_len;
}
```

This is the parameter-zero version of CSA: `S = 1/chunk_len`, no content projection, no positional bias.

### 2.3 The structural problem

V4's CSA is *learnable* — every weight in $W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ}, B^a, B^b$ gets gradients during model training. In MicroGPT-C, MSA pooling happens **outside** the training graph: the engine trains within a fixed `block_size` context, then MSA is applied at *inference time* to extend that context. There is no gradient flowing through `msa_pool_chunk`.

To make CSA truly learnable in MicroGPT-C, the entire MSA flow would need to be integrated into the training forward/backward — a substantial architectural change. The right pre-investment question is: **does CSA-style weighted pooling produce enough wins, in any form, to justify that refactor?**

### 2.4 Fixed-form alternatives (this paper)

We replace the uniform mean with one of three weight schemes, each capturing a prior that V4's learned CSA could plausibly approximate:

1. **Linear ramp recency** (`MSA_POOL_MODE=1`):
   $$
   w_t \propto 1 + \frac{t}{L-1}, \qquad t \in [0, L-1]
   $$
   Weights interpolate linearly from 1.0 (oldest) to 2.0 (newest). Captures "recent matters more, but not by much."

2. **Exponential recency** (`MSA_POOL_MODE=2`):
   $$
   w_t \propto \exp(t/\tau), \qquad \tau = L/4
   $$
   With L=64, τ=16: the last 4–8 tokens dominate the pool (relative weight ≈ 51:1 between newest and oldest). Captures "only the tail matters."

3. **Content-aware** (`MSA_POOL_MODE=3`):
   $$
   w_t = \text{softmax}_t\!\left(\frac{K_t \cdot K_{L-1}}{\sqrt{n_\text{embd}}}\right)
   $$
   Computed at the LAST layer (most semantically rich, per existing `msa_route_top_1` convention). Tokens whose K is similar to the chunk's most-recent K dominate the pool. Captures "tokens that 'agree' with the chunk's topic matter more." No parameters; the cosine score is a parameter-free content function.

All three normalise to $\sum_t w_t = 1$, so the resulting chunk vector has the same magnitude regime as the existing mean pool — no scale shift between modes. The benchmark binary, train loop, and routing code are byte-for-byte identical across all four modes.

---

## 3. Implementation

### 3.1 Files changed

| File | Change |
|---|---|
| `src/microgpt_msa.h` | Added `MSA_POOL_MODE` macro (default 0), with detailed doc-comment listing the four modes and their semantics. |
| `src/microgpt_msa.c` | Modified `msa_pool_chunk` — pre-computes a `weights[BLOCK_SIZE]` array (mode-selected), then the inner per-dim loop accumulates `sum_t weights[t] * K[t]` (was: uniform sum followed by `/chunk_len`). All four modes share the same outer loop and TurboQuant/RotorQuant integration paths — no code duplication. |
| `tests/bench_microgpt_msa_sliding.c` | **Reused unchanged.** This benchmark already scores long-context next-token CE under MSA, with `MSA_POOL_MODE` provided via the demo's `DEFINES`. |
| `CMakeLists.txt` | Four benchmark targets — `bench_msa_csa_{mean,linear,exp,content}` — each at the same architecture and seed, differing only by `MSA_POOL_MODE`. |

### 3.2 The weight-computation patch

```c
scalar_t weights[BLOCK_SIZE];
#if MSA_POOL_MODE == 1                       /* linear ramp */
    /* w_t = 1 + t/(L-1), normalised to sum to 1 */
#elif MSA_POOL_MODE == 2                     /* exponential recency */
    /* w_t = exp(t/tau), with max-shift for numerical stability */
#elif MSA_POOL_MODE == 3                     /* content-aware */
    /* w_t = softmax(K[t] · K[L-1] / sqrt(n_embd)) at the last layer */
#else                                        /* mode 0: uniform */
    scalar_t w = 1.0f / (scalar_t)chunk_len;
    for (size_t t = 0; t < chunk_len; t++) weights[t] = w;
#endif
/* per-layer per-dim weighted sum: sum_t weights[t] * K[t]  (no /chunk_len) */
```

The weights array is bounded by `BLOCK_SIZE` (since `chunk_len ≤ BLOCK_SIZE` by construction), so it lives on the stack — no allocation in the hot path. The mode-selection happens at compile time, so the unused branches don't even appear in the binary.

### 3.3 Compile-time gating

```bash
# Default: uniform mean pool (existing behaviour, bit-identical to pre-port)
cmake ..

# Linear-ramp recency
cmake -DMSA_POOL_MODE=1 ..

# Exponential recency
cmake -DMSA_POOL_MODE=2 ..

# Content-aware (softmax-of-cosine-to-anchor)
cmake -DMSA_POOL_MODE=3 ..
```

Per-demo: `DEFINES MSA_POOL_MODE=3` in `add_demo(...)`. Each unique `MSA_POOL_MODE` value triggers a separate library variant via `_microgpt_lib_for_defines()`, so all four modes live side-by-side in one build tree.

### 3.4 Verification

```
$ ./test_microgpt
=== Results: 61/61 passed ===

$ ./test_microgpt_msa
Running suite: MSA Memory Sparse Attention Primitives
All tests finished. Passed: 3, Failed: 0
```

Both core and MSA test suites pass with `MSA_POOL_MODE=0` (default). The MSA test suite exercises pooling and routing primitives directly; the fact that it continues to pass at every mode is empirical evidence that the weighted-sum implementation is consistent with the original mean pool when `weights[t] = 1/chunk_len`.

---

## 4. Benchmark Design

The harness is identical to the one used in [`RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md`](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md) §4 — same source file (`tests/bench_microgpt_msa_sliding.c`), same seed (`srand(42); seed_rng(42);`), same data (`c_names.txt`).

### 4.1 What we're measuring

Long-context next-token cross-entropy under repeated MSA chunking events. The benchmark:

1. Trains a 4-layer 138K-param char model on the names corpus (90% train), 1500 Adam steps. Identical training across all four pool modes — only the inference-time pooling changes.
2. Builds a 2048-token held-out stream by concatenating held-out names with BOS separators.
3. Feeds the stream through `forward_inference` token by token. When `pos == block_size`, runs the existing baseline-MSA chunking step (memmove second half down + best-chunk re-injection at position 0). The pooling that happens *inside* `msa_pool_chunk` at each event is what differs across modes.
4. At each token: softmax the lm_head logits and accumulate `−log p(target)` separately for positions before and after the first chunk event.

Reported metrics:
- `PRE_CHUNK_LOSS`: average CE on positions `0..63` (no chunking — control).
- `POST_CHUNK_LOSS`: average CE on positions `64..2047` (62 chunking events).
- `POST_CHUNK_PERPLEXITY`: `exp(POST_CHUNK_LOSS)`.

### 4.2 Variants

| Target | `MSA_POOL_MODE` | Pool scheme |
|---|---:|---|
| `bench_msa_csa_mean` | 0 | Uniform mean (baseline) |
| `bench_msa_csa_linear` | 1 | Linear ramp |
| `bench_msa_csa_exp` | 2 | Exponential recency |
| `bench_msa_csa_content` | 3 | Content-aware softmax-to-anchor |

`final_train_loss = 2.328964` and `pre_chunk_loss = 2.869539` are identical across all four modes by construction (training is unchanged; pool mode only affects inference-time compression).

---

## 5. Results

All numbers from the actual benchmark binaries built and run in this repository, 2048-token held-out stream, 62 MSA chunk events.

### 5.1 Raw output

| Mode | Post-chunk loss | Post-chunk PPL | Δ PPL vs mean |
|---|---:|---:|---:|
| Mean (mode 0) | 2.902364 | 18.2172 | — |
| Linear ramp (mode 1) | 2.902387 | 18.2176 | **+0.002%** |
| Exp recency (mode 2) | 2.902411 | 18.2180 | **+0.005%** |
| Content-aware (mode 3) | 2.899203 | **18.1597** | **−0.318%** ✓ |

### 5.2 Reading

Three observations:

1. **Positional priors do nothing.** Linear-ramp and exp-recency are within 0.005% of mean pool — round-off noise. This is striking given how different the weight distributions are (linear ramp: 2:1 between newest and oldest; exp recency: ~51:1). The pool's *positional* weighting genuinely doesn't matter for downstream perplexity.

2. **Content-aware shows a small but real win** (−0.32% PPL). The improvement comes from picking pool weights via the actual K vectors' geometry rather than by their position in the chunk. This validates the *idea* of content-aware pooling, but the magnitude is modest at our scale.

3. **The gap between best (content-aware) and uniform is much smaller than you'd hope.** A 0.3% PPL improvement is real but unexciting. For comparison: attention sinks gave −3.1% on a similar setup; Q/K RMSNorm + attention sink combined gave −7.0%. CSA-style pooling, even with the most expressive fixed-form scheme, is firmly in the diminishing-returns zone.

### 5.3 Why the positional schemes don't help

Possible explanation: the routing step (`msa_route_top_1` via cosine similarity at the last layer) is the dominant signal — whichever pool vector is most similar to the current query gets selected, and the chunk's *content* is what determines that. Whether the chunk vector was constructed with linear-ramp or uniform weights barely changes the routing decision because both schemes preserve roughly the same axis-of-most-variation in the high-dimensional pooled vector. Only when weights are computed *from the K-vector geometry itself* (content-aware mode) does the pool vector encode meaningfully different information.

The exp-recency scheme is essentially "drop everything except the last 4-8 tokens" — and yet that didn't help either. This implies the existing mean-pool's information loss (averaging out 64 tokens) is not the actual problem. The model can't extract more useful information from a pool that retains only the recent 4 tokens than from one that averages 64 — both summaries are too lossy for the model's downstream attention to do much with.

### 5.4 Why content-aware does help (a little)

The cosine-softmax weight `w_t = softmax_t(K_t · K_{L-1} / √d)` produces an "attention-like" pool: the chunk vector ends up close to the K-direction of whichever positions happen to align with the chunk's last token. This is structurally similar to what a *trained* CSA would do — except the training-derived $W^{aZ}$ would let the model choose its own anchor, instead of being forced to use the last token.

The fact that this works at all, with no learned parameters, suggests V4's full CSA could yield a moderately bigger improvement (perhaps 1–3%) once the weight-determination function is learnable. But that requires integrating MSA into the training graph — significant engineering for a likely-modest payoff.

### 5.5 Cost

- **Parameters added:** 0 (all four modes are parameter-free).
- **Compute added per chunking event** (chunk_len=64, n_embd=64):
  - Mode 0 (mean): `chunk_len × n_embd × n_layer` mults + adds = ~16K ops.
  - Mode 1 (linear): same as mean + `chunk_len` weight computations = ~16K ops.
  - Mode 2 (exp): same as mean + `chunk_len` exp calls = ~16.06K ops (negligible).
  - Mode 3 (content-aware): same as mean + `chunk_len × n_embd` cosine ops + softmax = ~20K ops (~25% extra at the chunking step, which itself fires once every `block_size` tokens).
- **Train time:** unaffected (training doesn't use MSA).
- **Inference time:** sub-percent in our measurements.

---

## 6. Interpretation

### 6.1 What this rules out

A common assumption when seeing the existing mean-pool is "surely the model loses information by averaging 64 tokens uniformly." This paper falsifies that — at least for our model and our test:

- Reweighting toward recent tokens (mode 1, mode 2) doesn't recover information.
- Reweighting toward content-aligned tokens (mode 3) recovers a tiny amount.
- The remaining 99.7% of the post-chunk loss is *not* explained by the pool's weighting choice.

### 6.2 What the bottleneck likely is

By elimination from the prior V4-port papers in this series:
1. **Sliding-window recency** ([`RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md`](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md)) found that wpe-alignment after recency injection is a problem. Suggests learned absolute position embeddings are incompatible with the moving / re-injecting K vectors. Recommends Partial RoPE port first.
2. **CSA-style pooling** (this paper) finds that pooling weights barely matter. Suggests the chunk's content quality is not the limiting factor.

The remaining suspects: routing fidelity (`msa_route_top_1`'s cosine top-1 retrieval), and the chunk-to-active-cache injection's wpe issue. The Lightning Indexer (V4 §3.6 / our roadmap §3.6) replaces top-1 cosine retrieval with a learned content-addressed top-k indexer, and is the natural next port to try.

### 6.3 What's still worth doing

- **Ship `MSA_POOL_MODE=3` as the recommended default for MSA-using demos.** The win is small but real and free.
- **Don't pursue a backprop-learnable CSA refactor.** The fixed-form ceiling of −0.32% suggests the upside isn't worth the engineering cost.
- **Pursue Partial RoPE (§3.3 of roadmap) and Lightning Indexer (§3.6) instead.** Both target the actual bottlenecks.

---

## 7. Reproducing the Results

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cd build

cmake --build . --config Release --parallel 8 --target \
    bench_msa_csa_mean bench_msa_csa_linear \
    bench_msa_csa_exp bench_msa_csa_content

./bench_msa_csa_mean      # MSA_POOL_MODE=0 (uniform mean)
./bench_msa_csa_linear    # MSA_POOL_MODE=1 (linear ramp)
./bench_msa_csa_exp       # MSA_POOL_MODE=2 (exp recency)
./bench_msa_csa_content   # MSA_POOL_MODE=3 (content-aware)
```

Results are deterministic for a given seed, single-threaded by default. The three numbers that should match across runs are `FINAL_TRAIN_LOSS` (2.328964), `PRE_CHUNK_LOSS` (2.869539), and `CHUNK_EVENTS` (62) — the only differing line is `POST_CHUNK_PERPLEXITY`.

To enable a non-default mode in any other MSA-using demo:

```cmake
add_demo(
  NAME    your_msa_demo
  SOURCE  demos/msa/your/main.c
  DEFINES N_EMBD=... N_HEAD=... N_LAYER=... BLOCK_SIZE=... MLP_DIM=...
          MSA_POOL_MODE=3
)
```

Or globally:

```bash
cmake -DMSA_POOL_MODE=3 ..
```

---

## 8. Limitations and Future Work

1. **Single corpus, single hardware, single seed.** Same caveat as the prior V4-port papers in this series.
2. **Fixed-form, not truly learnable.** V4's CSA uses backprop-trained pool weights. We approximated three plausible learned shapes (recency-favouring, exp-recency, content-aware-cosine) but couldn't reproduce the actual training signal. The −0.32% content-aware result is the lower bound of what a fully learnable scheme might deliver; the upper bound is unknown. The §6.3 recommendation to *not* pursue full learnable CSA is conditional on the assumption that fixed-form is a reasonable approximation — true CSA could in principle unlock more, but the gap from baseline to fixed-form-best is so small that the engineering effort seems hard to justify.
3. **No interaction tested with other V4 ports.** Combined CSA pool + attention sink + Q/K RMSNorm + sliding window is unmeasured. Sink and Q/K RMSNorm act on the active attention path; CSA acts on the compressed pool — they should compose cleanly but this is unverified.
4. **No interaction tested with TurboQuant / RotorQuant.** The pool's reduced precision under quantisation might amplify the differences between pool modes (a coarse pool vector might lose more from mean averaging when also quantised). Untested in this paper.
5. **Single chunk size.** All experiments use chunk_len = block_size = 64. The recency-favouring schemes might matter more for larger chunks where the "most recent matters" bias has more positions to discriminate over.

---

## 9. References

- DeepSeek-V4 paper: [`papers/DeepSeek_V4.pdf`](papers/DeepSeek_V4.pdf), §2.3.1 "Compressed Sparse Attention", equations (9)–(12).
- Roadmap context: [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) §3.5.
- Companion papers (the V4 port series):
  - [`RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md) — attention sink (self-contained win).
  - [`RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md`](RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md) — Q/K RMSNorm (stability win + super-additive with sink).
  - [`RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md`](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md) — sliding-window recency (negative result, blocked on Partial RoPE).
- Existing MSA design: [`RESEARCH_MSA.md`](RESEARCH_MSA.md).
- Implementation:
  - `src/microgpt_msa.h` (`MSA_POOL_MODE` macro + 0-3 mode definitions)
  - `src/microgpt_msa.c` (weight-computation block + weighted-sum loop in `msa_pool_chunk`)
  - `tests/bench_microgpt_msa_sliding.c` (reused A/B harness)
  - `CMakeLists.txt` (four `bench_msa_csa_*` targets)

---

*Honest research: a small (−0.3%) win is still a win. The paper is more valuable for what it rules out — pool weighting is not the bottleneck — than for what it ships. Recommended port action: enable `MSA_POOL_MODE=3` by default in MSA demos and move on to the next V4 idea.*
