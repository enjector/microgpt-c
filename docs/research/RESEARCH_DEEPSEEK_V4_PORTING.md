# Porting DeepSeek-V4 Architectural Ideas to MicroGPT-C

> A research roadmap for selectively adopting DeepSeek-V4 innovations into a zero-dependency, CPU-first, ~1M-parameter C99 transformer engine.

**Reference paper:** [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](papers/DeepSeek_V4.pdf) (DeepSeek-AI, 2026).

**Audience:** MicroGPT-C maintainers and contributors evaluating which V4 techniques are worth porting.

---

## 1. Executive Summary

DeepSeek-V4 (V4-Pro: 1.6T params / 49B activated, V4-Flash: 284B / 13B activated) is a frontier MoE model targeting million-token contexts. MicroGPT-C is the opposite end of the spectrum — 30K to ~1M dense parameters, single-file C99, CPU-first, edge-deployable. Most of V4's contributions (FP4 QAT, MegaMoE expert parallelism, TileLang kernels, on-disk KV) are GPU/MoE infrastructure that does not map onto MicroGPT-C's design space.

However, **six V4 architectural ideas transfer cleanly to a small dense CPU engine** and address known weaknesses in MicroGPT-C's existing MSA + TurboQuant/RotorQuant pipeline. They are:

| # | Technique | V4 location | Effort | Expected win |
|---|-----------|-------------|--------|--------------|
| 1 | Attention sink (learnable softmax-denominator logits) | §2.3.3, eq. 27 | ~1 day | Long-context stability; eliminates "no-attend" pathology |
| 2 | Q/K RMSNorm pre-dot-product | §2.3.3 | ~½ day | Numerical safety; removes need for QK-clip |
| 3 | Partial RoPE on last 64 dims (with `−i` countertrick on output) | §2.3.3 | ~2–3 days | Relative-position generalisation without `wpe` growth |
| 4 | Sliding-window recency branch alongside compressed KV | §2.3.3 | ~2–3 days | Local coherence with heavy KV compression |
| 5 | CSA-style learnable pooling for MSA chunks | §2.3.1, eqs. 9–12 | ~1 week | Better latent-chunk fidelity than mean/last pooling |
| 6 | Lightning Indexer + top-k content-addressed retrieval | §2.3.1, eqs. 13–17 | ~2 weeks | MSA becomes content-addressable, not just LRU |

This document expands on each, explains the C99 implementation strategy, and identifies V4 ideas explicitly **not** worth porting at MicroGPT-C's scale.

---

## 2. What MicroGPT-C Already Has (V3-Equivalent)

Before evaluating V4-specific upgrades, a recap of what's already in `src/microgpt.{h,c}` and the auxiliary modules:

| Capability | Status in MicroGPT-C | Comparable V3/V4 feature |
|-----------|----------------------|--------------------------|
| RMSNorm | ✅ | RMSNorm |
| SwiGLU MLP | ✅ | SwiGLU |
| Grouped-query attention | ✅ | GQA / MQA-friendly |
| AdamW (decoupled weight decay) | ✅ | AdamW (used for embeddings/static biases in V4) |
| Cosine LR + linear warmup | ✅ | Same |
| Gradient clipping, label smoothing | ✅ | Same |
| KV cache + prefix sharing for ensemble | ✅ | KV reuse |
| Paged KV cache | ✅ (`MICROGPT_PAGED_KV`) | KV cache structure (§3.6.1) |
| INT8 weight quantisation | ✅ (`QUANTIZATION_INT8`) | FP4/FP8 (lower precision still) |
| MSA — LRU-paged latent KV chunks | ✅ | Conceptually adjacent to CSA's "compress every m tokens" |
| TurboQuant — 4-bit dual-state KV | ✅ | KV memory pressure relief |
| RotorQuant — rotor-based KV compression | ✅ | KV memory pressure relief |
| Block AttentionResiduals | ✅ (`MICROGPT_ATTN_RES`) | Conceptually adjacent to mHC residual stabilisation |
| Speculative decoding / multi-token vote | ✅ | MTP (Multi-Token Prediction) |

So DeepSeek-V3 inheritance is largely covered. The interesting porting question is **what V4 added on top of V3**.

---

## 3. Recommended Ports (high ROI)

### 3.1 Attention Sink

**Source:** §2.3.3 "Attention Sink", equation (27).

V4 adds learnable per-head sink logits `{z'_1, …, z'_{n_h}}` to the softmax denominator:

$$
s_{h,i,j} = \frac{\exp(z_{h,i,j})}{\sum_k \exp(z_{h,i,k}) + \exp(z'_h)}
$$

This lets each query head spend probability mass on "attend to nothing," which has been shown to stabilise long-context attention and prevent the "all probability collapses onto one early token" pathology that affects autoregressive generation.

**Why it matters for MicroGPT-C.** The MSA infinite-context story exposes the engine to exactly the regime where attention sinks help — long sequences where most tokens are irrelevant. Today, every MSA query forces a normalised distribution over the latent pool, which can amplify noise when no chunk is genuinely relevant.

**Implementation sketch.**

- Add `scalar_t attn_sink[N_LAYER][N_HEAD]` to the `Model` struct (~`N_LAYER * N_HEAD` extra scalars; for a 4-layer 8-head model, 32 scalars — negligible).
- In `attn_softmax`, compute `denom = Σ exp(z_k) + exp(sink_h)` instead of `Σ exp(z_k)`.
- Backward: gradient of `sink_h` is `−s_sink · dL/dlogits_sum` (standard softmax-with-extra-logit derivative).
- Initialise sinks to a small negative value (e.g., `−1.0`) so they start out absorbing little mass.
- Compile-time gated: `-DMICROGPT_ATTN_SINK=ON`.

**Estimated cost:** ~50 LOC across `microgpt.c`, `microgpt.h`, `adam_step`, `model_save/load`. ~1 day including tests.

---

### 3.2 Q/K RMSNorm Before the Dot Product

**Source:** §2.3.3 "Query and Key-Value Entry Normalization."

V4 applies an extra RMSNorm to each query head and to the compressed KV head **immediately before** `Q·K^T`. The paper notes this "avoids exploding attention logits and may improve training stability," and explicitly justifies omitting QK-clip because of it.

**Why it matters for MicroGPT-C.** The book chapters claim "zero NaN instability" as a foundational property. Q/K RMSNorm is the cheapest insurance policy that makes that claim robust under more aggressive configurations (deeper stacks, lower-precision KV via TurboQuant/RotorQuant, longer contexts via MSA).

**Implementation sketch.**

- Reuse the existing RMSNorm kernel.
- Apply once to `Q[h, t, :]` per head, once to `K[t, :]` per cached position, before scaled dot product.
- Two extra norm calls per attention call. On CPU, RMSNorm is a tiny fraction of attention cost (matmul-dominated), so the overhead is sub-percent.
- Optional learnable scale per-head (V4 uses one) — adds `N_LAYER * N_HEAD * 2` scalars.
- Compile-time gated: `-DMICROGPT_QK_NORM=ON`.

**Estimated cost:** ~30 LOC, ~½ day including tests.

---

### 3.3 Partial Rotary Positional Embedding

**Source:** §2.3.3 "Partial Rotary Positional Embedding."

V4 applies RoPE to only the **last 64 dimensions** of Q and K (not all of `head_dim`), and — because compressed KV entries serve as both keys and values — also applies RoPE with position `−i` to the last 64 dims of each per-head output. This way the core attention output carries relative position information rather than absolute embeddings inherited from the weighted sum of KV entries.

**Why it matters for MicroGPT-C.** The current engine uses a learned absolute `wpe` of length `BLOCK_SIZE`. This:

1. Caps context at compile-time `BLOCK_SIZE`.
2. Wastes capacity at small `N_EMBD` (a 16-dim model spends 16×BLOCK_SIZE scalars on positions).
3. Doesn't generalise beyond training-time positions, which is precisely the failure mode MSA's infinite context surfaces.

Partial RoPE on the tail dims would let the model express relative position with no extra parameters and degrade gracefully past `BLOCK_SIZE`. It is also the V4 trick that pairs best with MSA, where chunks have ambiguous absolute positions but unambiguous *relative* ordering.

**Implementation sketch.**

- New macros: `MICROGPT_PARTIAL_ROPE` (toggle) and `ROPE_DIMS` (default `min(head_dim, 32)` for typical small models).
- Precompute `cos[t][d], sin[t][d]` tables of size `BLOCK_SIZE × ROPE_DIMS/2`. These are not parameters — pure math.
- In `attn_qkv`: after computing `Q[h, t, :]` and `K[t, :]`, rotate the last `ROPE_DIMS` dims using the cached cos/sin.
- In the output projection path, optionally apply RoPE with position `−i` on the last `ROPE_DIMS` dims of each per-head core-attention output (V4's "countertrick" — see §2.3.3 paragraph 1).
- For MSA, store the **relative chunk position** at chunk creation time and use it as `t` for RoPE; this gracefully handles eviction.

**Estimated cost:** ~150 LOC including cos/sin tables, ~2–3 days. Lots of small tests (rotation correctness, gradient flow, RoPE-with-`−i` symmetry).

**Caveat.** Removing or shrinking `wpe` is a breaking change for existing checkpoints. Initial port should keep `wpe` and just *add* RoPE on the tail dims; full `wpe` retirement is a v2 step after evaluation shows RoPE alone is sufficient.

---

### 3.4 Sliding-Window Recency Branch in MSA

**Source:** §2.3.3 "Additional Branch of Sliding Window Attention."

V4 always keeps `n_win` recent uncompressed KV entries alongside compressed/sparse-selected entries, then concatenates both into the core MQA. The justification: "recent tokens usually possess greater relevance to the query token in language modeling," and a query inside a compressed block cannot see other tokens in its own block.

**Why it matters for MicroGPT-C.** MSA today routes everything to compressed latent chunks. With aggressive compression (TurboQuant 4-bit, RotorQuant), local coherence is the first thing to suffer — exactly because the most recent ~64 tokens get bucketed in with thousands of older ones. A small uncompressed recency tail would directly fix this without weakening the compressed-pool design.

**Implementation sketch.**

- Add a small ring buffer of `n_win` uncompressed K and V vectors per layer (e.g., `n_win = 64`).
- During inference: append every new K/V into both the ring buffer and the MSA pool; when the ring buffer overflows, the evicted entry stays in the MSA pool only.
- During core attention: concatenate `[ring_buffer_KV] ⊕ [MSA_chunk_KV]` before softmax.
- Compile-time gated: `-DMSA_SLIDING_WINDOW=ON`, `-DMSA_WIN=64`.

**Estimated cost:** ~200 LOC in `microgpt_msa.c`, ~2–3 days. The forward path is straightforward; gradient flow is identical to the existing MSA backward since the ring buffer is just regular K/V.

---

### 3.5 CSA-Style Learnable Pooling for MSA Chunks

**Source:** §2.3.1 "Compressed Sparse Attention," equations (9)–(12).

V4 builds compressed KV entries by computing two parallel sets of token-level transforms `C^a, C^b ∈ R^{n×c}` and **softmax weights** `Z^a, Z^b ∈ R^{n×c}` from the input hidden states, then for each block of `m` tokens:

$$
[S^a_{m i:m(i+1)-1}; S^b_{m(i-1):mi-1}] = \text{Softmax}_{\text{row}}([Z^a_{m i:m(i+1)-1} + B^a; Z^b_{m(i-1):mi-1} + B^b])
$$

$$
C^{\text{Comp}}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j + \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j
$$

— a learnable, position-aware, overlapping-block pool with biases `B^a, B^b ∈ R^{m×c}`. This is materially better than mean/last-token pooling, the typical baseline.

**Why it matters for MicroGPT-C.** Whatever MSA currently uses to summarise a chunk (mean, last, or fixed projection) is a baseline pooling operator. V4's CSA is the principled upgrade: every chunk entry is a *learned* mix of `2m` source tokens with positional bias, so the pool retains task-relevant detail rather than mean-blurring.

**Implementation sketch.**

- Add per-layer trainable matrices `W^aKV, W^bKV, W^aZ, W^bZ ∈ R^{N_EMBD × c}` and positional biases `B^a, B^b ∈ R^{m × c}`. With `c = N_EMBD / 2` and `m = 8`, this is roughly `4 * N_EMBD * c + 2 * m * c` ≈ a few thousand scalars per layer — small.
- In MSA pool construction, replace the current pooling with eqs. (9)–(12).
- Backward: standard softmax + Hadamard derivatives. Reuse the existing `softmax_bwd`.
- Compile-time gated: `-DMSA_CSA_POOLING=ON`.

**Estimated cost:** ~500 LOC, ~1 week. Most of the effort is in the backward pass and tests confirming gradients match a finite-difference reference.

---

### 3.6 Lightning Indexer + Top-k Sparse Selection

**Source:** §2.3.1 "Lightning Indexer for Sparse Selection," equations (13)–(17).

After CSA produces compressed entries `C^{\text{Comp}}`, V4 turns MSA into a **content-addressed retrieval system**: for each query token, low-rank indexer queries score every compressed block via a ReLU-summed attention head, and only the top-k blocks proceed into the core attention.

$$
I_{t,s} = \sum_{h=1}^{n^I_h} w^I_{t,h} \cdot \text{ReLU}\!\left(\mathbf{q}^I_{t,h} \cdot K^{\text{IComp}}_s\right)
$$

$$
C^{\text{SprsComp}}_t = \{C^{\text{Comp}}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}
$$

This is the most aggressive idea in V4 and the one that most directly aligns with MicroGPT-C's edge/long-context positioning.

**Why it matters for MicroGPT-C.** Today, MSA is pure LRU recency — it stores recent chunks regardless of relevance. A query about a topic from 50,000 tokens ago has no way to retrieve it unless the matching chunk happens to still be hot. Lightning Indexer would change MSA from "remembers recently" to "remembers what's relevant," which is the difference between a working buffer and an attention-based memory system.

**Implementation sketch.**

- Per-layer indexer parameters: `W^DQ ∈ R^{N_EMBD × d_c}`, `W^IUQ ∈ R^{d_c × c^I × n^I_h}`, `W^IK_compress ∈ R^{N_EMBD × c^I}`, `W^w ∈ R^{N_EMBD × n^I_h}`. With `d_c = 32, c^I = 16, n^I_h = 4`, this is ~`N_EMBD * (32 + 16 + 4) + 32 * 16 * 4` ≈ a few thousand scalars per layer.
- Forward: compute indexer queries → score every compressed entry → top-k via partial sort (use a small heap; `k` is typically 32–64).
- The top-k mask is non-differentiable; use a **straight-through estimator** for backward (gradient flows through scores as if top-k was identity), as in the V4 reference implementation.
- Compile-time gated: `-DMSA_LIGHTNING_INDEXER=ON`, `-DMSA_TOPK=32`.

**Estimated cost:** ~1500 LOC including the heap, the STE backward, and a careful test suite verifying that top-k actually surfaces relevant chunks on a synthetic needle-in-haystack benchmark. ~2 weeks.

**Research output.** This is the only port that is genuinely novel for an edge engine — small CPU models with content-addressed long-term memory. A separate `RESEARCH_LIGHTNING_INDEXER.md` should be written if this lands, with needle-in-haystack and Shakespeare-resume benchmarks.

---

## 4. Not Recommended at MicroGPT-C's Scale

These V4 ideas were considered and rejected — either the underlying problem doesn't manifest at small scale, or the V4 solution depends on hardware MicroGPT-C doesn't target.

### 4.1 Muon Optimizer (§2.4)

V4 uses Muon (Newton–Schulz orthogonalisation per matrix per step) for non-embedding/non-norm modules. The win materialises at hundreds of millions to billions of parameters. At MicroGPT-C's 30K–841K dense params, AdamW already converges in seconds to minutes; the Newton–Schulz iterations (10 per step in V4's hybrid scheme) would *add* cost without meaningful return. **Skip.**

### 4.2 Manifold-Constrained Hyper-Connections (§2.2)

mHC projects residual mappings onto the Birkhoff polytope of doubly stochastic matrices via Sinkhorn–Knopp iterations (20 of them per step), to fix instabilities in deep stacks. The instability it solves doesn't show up at `N_LAYER ≤ 4`. The existing `MICROGPT_ATTN_RES` flag already gives learned depth-attention, which addresses the same problem space at much lower cost. **Skip unless models grow past 8 layers.**

### 4.3 Heavily Compressed Attention (§2.3.2)

HCA consolidates every `m' ≫ m` tokens into a single entry — designed for million-token contexts. MSA + TurboQuant already handles MicroGPT-C's working scales (a few thousand tokens with effectively-infinite paged history). Until edge use cases credibly demand million-token contexts, HCA's complexity is unjustified. **Skip.**

### 4.4 GPU/MoE Infrastructure (§3 entirely)

FP4 quantisation-aware training, MegaMoE expert parallelism, TileLang kernels, fine-grained communication-computation overlap, on-disk KV storage, batch-invariant deterministic kernels — all of these are GPU/cluster infrastructure that does not apply to a zero-dependency CPU C99 engine. **Skip wholesale.**

### 4.5 DeepSeekMoE (§2.1)

MicroGPT-C's organelle architecture is a *deliberate* alternative to MoE: instead of one mega-model with routed experts, it runs multiple complete tiny models behind a coordination pipeline (`microgpt_organelle.{h,c}`). The two are philosophically different — MoE shares a backbone and routes per-token, organelles share a *protocol* and route per-task. MoE belongs in a different project. **Skip.**

### 4.6 Multi-Token Prediction Modules (§2.1)

V4 retains V3's MTP heads as auxiliary training objectives. MicroGPT-C already does multi-candidate / speculative decoding at inference time. Adding MTP heads at training time is feasible but is an independent research thread, not a V4-specific idea. **Track separately, not as a V4 port.**

---

## 5. Suggested Implementation Order

If implemented as one PR per feature, the following order maximises early wins and de-risks later work:

1. **Q/K RMSNorm** (§3.2) — half a day, removes a class of numerical bugs, prerequisite for safely doing RoPE and indexer experiments.
2. **Attention Sink** (§3.1) — one day, biggest stability/quality-per-LOC ratio.
3. **Partial RoPE** (§3.3) — 2–3 days, real perplexity win on word-level Shakespeare and VM codegen, and the right time to revisit `wpe`.
4. **Sliding-Window Recency Branch in MSA** (§3.4) — 2–3 days, fixes an obvious local-coherence weakness in heavily compressed MSA.
5. **CSA-Style Learnable Pooling** (§3.5) — ~1 week, infrastructure for (6).
6. **Lightning Indexer + Top-k** (§3.6) — ~2 weeks, headline research output.

After (6), MSA stops being "LRU recency" and becomes "content-addressed working memory plus recency window." That is a publishable result for an edge engine.

---

## 6. Engineering Constraints

All ports must respect MicroGPT-C's design principles:

- **C99 only** in core engine; no C11/C23 features.
- **Zero dependencies** in `microgpt.{h,c}` and the new feature files. RMSNorm, softmax, top-k, RoPE, Sinkhorn — all pure C, no BLAS or math libs beyond `libm`.
- **Compile-time gated.** Every new feature ships behind a `-DMICROGPT_*=ON` flag. Default-off; existing demos must continue to compile and pass tests bit-identically.
- **Constant-folding-friendly.** New macros (`ROPE_DIMS`, `MSA_TOPK`, `MSA_WIN`, etc.) follow the existing `#define` convention so the compiler can unroll inner loops.
- **`scalar_t`-uniform.** Never hardcode `float` or `double`; all new weights/activations use `scalar_t`. Optimizer state stays `double`.
- **Tests first.** Each feature gets a `tests/test_microgpt_<feature>.c` mirroring the existing pattern, with a numerical-gradient check for any new backward path.

---

## 7. Open Questions

1. **Does Lightning Indexer recover information that LRU MSA loses?** Synthetic needle-in-haystack benchmark required.
2. **Does Q/K RMSNorm let us safely remove gradient clipping?** Currently `GRAD_CLIP=1.0` is set on most demos. V4's claim is yes.
3. **Does partial RoPE let us shrink `wpe` to zero, or do we keep both?** Empirical question; run with `wpe` enabled first, then ablate.
4. **Is CSA pooling worth the parameters at our scale, or does mean-pool plus Lightning Indexer suffice?** Test both pooling choices independently before combining.
5. **Attention sink + partial RoPE interaction.** Both touch attention scores; ensure they compose correctly under MSA + sliding window.

---

## 8. References

- DeepSeek-V4 paper: [`docs/research/papers/DeepSeek_V4.pdf`](papers/DeepSeek_V4.pdf), DeepSeek-AI, 2026.
- DeepSeek-V4 reference implementation: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference
- Existing MicroGPT-C research notes:
  - [`RESEARCH_MSA.md`](RESEARCH_MSA.md) — current Memory Sparse Attention design.
  - [`RESEARCH_TURBO_QUANT.md`](RESEARCH_TURBO_QUANT.md) — 4-bit dual-state KV compression.
  - [`RESEARCH_ROTOR_QUANT.md`](RESEARCH_ROTOR_QUANT.md) — rotor-based KV compression.
  - [`RESEARCH_ATTN_RES.md`](RESEARCH_ATTN_RES.md) — Block Attention Residuals (related to V4's mHC).
  - [`RESEARCH_ATTENTION_MECHANISMS.md`](RESEARCH_ATTENTION_MECHANISMS.md) — broader attention design space.
  - [`RESEARCH_TRANSFORMER.md`](RESEARCH_TRANSFORMER.md) — core transformer choices.

---

## 9. Status — Roadmap Closed

This document was originally written as a **scoping roadmap** before any port had been implemented. All six recommended ports have now been measured. The series produced **eight research papers** (six per-port plus two follow-up rotation-fix papers); seven landed positive results, one ([Lightning Indexer](RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md)) ships infrastructure but is not the recommended default.

| # | Port | Status | Best result | Paper |
|---|---|---|---|---|
| 1 | Attention sink | ✅ Shipped | **−3.1% PPL** | [RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md) |
| 2 | Q/K RMSNorm pre-dot | ✅ Shipped | super-additive with sink, 3.6× recovery from divergence at high LR | [RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md](RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md) |
| 3 | **Partial RoPE** | ✅ Shipped | **−1.6% solo, capstone of −8.7% full stack** | [RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md](RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md) |
| 4 | Sliding-window MSA recency | ✅ Shipped (after rotation fixes) | **−3.0% within RoPE regime** | [RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md) → [RESEARCH_DEEPSEEK_V4_MSA_ROPE_REROTATE.md](RESEARCH_DEEPSEEK_V4_MSA_ROPE_REROTATE.md) → [RESEARCH_DEEPSEEK_V4_MSA_POOL_ROPE_REROTATE.md](RESEARCH_DEEPSEEK_V4_MSA_POOL_ROPE_REROTATE.md) |
| 5 | CSA-style pool (`MSA_POOL_MODE=3`) | ✅ Shipped | −0.32% PPL | [RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md) |
| 6 | Lightning Indexer + top-K | ✅ Infrastructure shipped, not recommended default | −0.32% at K=8, doesn't compose with #5 | [RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md](RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md) |

### Recommended default V4 stack

```cmake
MICROGPT_PARTIAL_ROPE=1
MICROGPT_ATTN_SINK=1   ATTN_SINK_LOGIT=-1.0
MICROGPT_QK_NORM=1
MSA_POOL_MODE=3
```

Combined: **−8.7% held-out PPL on the deep config (4-layer 138K-param char model)**, zero new parameters, ~1% extra training runtime, all 61 core unit tests pass under each flag combination tested. Demos that integrate MSA directly should additionally use the rope-aware wrappers (`msa_pool_chunk_rope`, `msa_expand_context_rope`, `msa_recency_inject_rope`) — see `msa_infinite_shakespeare_v4` for a worked example.

### Answering the open questions from §7

1. **Does Lightning Indexer recover information that LRU MSA loses?** Partially. Top-K=8 captures −0.32% PPL improvement. Smaller K (1, 2, 4) is within noise or slightly regresses. Not the recommended default; see paper #6.
2. **Does Q/K RMSNorm let us safely remove gradient clipping?** Inferable: Q/K RMSNorm at LR=0.02 (20× normal) keeps the model bounded where the un-normed baseline diverges. We didn't formally remove `GRAD_CLIP=1.0` from existing demos as part of this work, but the safety margin is now measured.
3. **Does partial RoPE let us shrink `wpe` to zero?** Untested — we deliberately kept both for backward compatibility. Future work.
4. **Is CSA pooling worth the parameters at our scale?** No — fixed-form pooling (the parameter-free port we shipped) caps at −0.32% PPL. The learnable variant V4 uses likely doesn't unlock more at our scale; backprop-through-pool refactor is not justified by the upside. See paper #5.
5. **Attention sink + partial RoPE interaction.** Compose super-additively: sink alone −3.1%, RoPE alone −1.6%, combined −3.8%. The full stack with Q/K RMSNorm reaches −8.7%. See paper #3 §5.3.

### What's still open

- Removing `wpe` cleanly under RoPE (item 3 above) — checkpoint-format-breaking, deferred.
- Closing the residual ~+5.5% cross-regime gap in MSA-with-RoPE evaluation. Three named items in [RESEARCH_DEEPSEEK_V4_MSA_POOL_ROPE_REROTATE.md](RESEARCH_DEEPSEEK_V4_MSA_POOL_ROPE_REROTATE.md) §5.5: routing-query rotation alignment, Q-rotation continuity at injected slots, and RoPE-from-scratch retraining.
- High-LR stability ablations across the full stack (V4 cites Muon optimiser as a partner; we used AdamW).

The series is otherwise complete.

---

*This document was a research roadmap, not a commitment. The roadmap is now closed: each port was re-justified at implementation time, measured, documented, and shipped (or explicitly not shipped, with reasoning). See the eight per-port papers for full results.*
