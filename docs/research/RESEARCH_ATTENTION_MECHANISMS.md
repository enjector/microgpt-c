# Design Document: Attention Mechanisms

## A Practical Guide to Attention Variants for Sub-1M Parameter Transformers

**Author:** Ajay Soni, Enjector Software Ltd.

**Date:** February 2026

---

## Spear Summary

**Point:** Standard multi-head attention wastes memory when K/V heads duplicate each other — Grouped Query Attention (GQA) is the single highest-impact upgrade for MicroGPT-C, cutting KV cache by up to 4× with minimal quality loss.

**Picture:** Imagine eight students in a classroom (Q heads) each taking their own notes (K/V heads). In GQA, four students share notes in pairs — same understanding, half the paper. In MQA, everyone shares one set of notes — minimal paper, slightly less personalised.

**Proof:** Llama 2 70B moved from MHA to GQA and saw **no measurable quality loss** on standard benchmarks while cutting KV cache memory by 4×. At MicroGPT-C's scale (128-dim, 8 heads), GQA with 2 KV groups would reduce KV cache from 8 KB to 2 KB per layer — the difference between fitting on an ESP32 and not.

**Push:** Implement GQA as a compile-time option (`-DN_KV_HEAD=2`). It's a ~50-line change to the attention loop and weight allocation, with immediate benefits for the multi-organelle pipeline's total memory footprint.

---

## Table of Contents

1. [Current Implementation: Standard MHA](#1-current-implementation-standard-mha)
2. [Strategy 1 — Grouped Query Attention (GQA)](#2-strategy-1--grouped-query-attention-gqa)
3. [Strategy 2 — Multi-Query Attention (MQA)](#3-strategy-2--multi-query-attention-mqa)
4. [Strategy 3 — Sliding Window Attention](#4-strategy-3--sliding-window-attention)
5. [Strategy 4 — Multi-Head Latent Attention (MLA)](#5-strategy-4--multi-head-latent-attention-mla)
6. [Strategy 5 — Dynamic Sparse Attention](#6-strategy-5--dynamic-sparse-attention)
7. [Comparison Matrix](#7-comparison-matrix)
8. [Implementation Priority](#8-implementation-priority)

---

## 1. Current Implementation: Standard MHA

### Architecture

MicroGPT-C implements standard multi-head attention as described in *Attention Is All You Need* (Vaswani et al., 2017). Each layer has three full projection matrices:

```
Input x (N_EMBD)
    │
    ├─── Wq [N_EMBD × N_EMBD] ──→ Q (N_EMBD)
    ├─── Wk [N_EMBD × N_EMBD] ──→ K (N_EMBD)  ──→ KV Cache
    └─── Wv [N_EMBD × N_EMBD] ──→ V (N_EMBD)  ──→ KV Cache
```

Q, K, V are split into `N_HEAD` independent sub-spaces of dimension `head_dim = N_EMBD / N_HEAD`. Each head computes:

```
Attention(Q_h, K_h, V_h) = softmax(Q_h · K_h^T / √head_dim) · V_h
```

Results from all heads are concatenated and passed through the residual connection.

### Memory Budget (current defaults)

| Component | Size (float32) | Per Layer | Total (4 layers) |
|-----------|---------------|-----------|-------------------|
| Wq weights | N_EMBD² = 16,384 B | 16 KB | 64 KB |
| Wk weights | N_EMBD² = 16,384 B | 16 KB | 64 KB |
| Wv weights | N_EMBD² = 16,384 B | 16 KB | 64 KB |
| KV cache (BLOCK_SIZE=256) | 2 × 256 × 128 × 4 = 256 KB | 256 KB | 1,024 KB |
| **Attention total** | | **304 KB** | **1,216 KB** |

> **Key observation:** At N_EMBD=128, the KV cache dominates attention memory — it's **5× larger** than the weight matrices themselves. Any optimisation that reduces KV cache size has outsized impact.

### Code Path

```c
// Forward: project Q, K, V (full N_EMBD × N_EMBD matmuls)
lin_fwd(x_norm1, model->Wq[L], ne, ne, q);
lin_fwd(x_norm1, model->Wk[L], ne, ne, k);
lin_fwd(x_norm1, model->Wv[L], ne, ne, v);

// Store full K, V vectors in cache
memcpy(KV_WRITE(keys, L, cache_len, ne), k, ne * sizeof(scalar_t));
memcpy(KV_WRITE(values, L, cache_len, ne), v, ne * sizeof(scalar_t));

// Per-head attention loop
for (int h = 0; h < nh; h++) {
    size_t hoff = h * hd;
    // Each head reads its slice of Q, K, V
    // Compute attention weights, softmax, weighted sum
}
```

### Strengths and Limitations

| ✅ Strengths | ❌ Limitations |
|-------------|--------------|
| Simple, well-understood | K/V heads often learn redundant representations |
| Maximum representational capacity | KV cache scales linearly with N_HEAD |
| Battle-tested implementation | Overkill for sub-1M parameter models |
| Easy to debug and verify | Memory-heavy for multi-organelle pipelines |

---

## 2. Strategy 1 — Grouped Query Attention (GQA)

**Priority: HIGH — recommended next implementation**

**CMake flag:** `-DN_KV_HEAD=2` (proposed)

### What It Is

GQA (Ainslie et al., 2023) reduces the number of K/V heads while keeping all Q heads. Multiple Q heads share the same K/V projections:

```
Standard MHA (8 Q heads, 8 KV heads):
  Q₀ ↔ K₀,V₀    Q₁ ↔ K₁,V₁    Q₂ ↔ K₂,V₂    Q₃ ↔ K₃,V₃
  Q₄ ↔ K₄,V₄    Q₅ ↔ K₅,V₅    Q₆ ↔ K₆,V₆    Q₇ ↔ K₇,V₇

GQA-2 (8 Q heads, 2 KV groups):
  Q₀ Q₁ Q₂ Q₃ ↔ K₀,V₀         Q₄ Q₅ Q₆ Q₇ ↔ K₁,V₁

GQA-4 (8 Q heads, 4 KV groups):
  Q₀ Q₁ ↔ K₀,V₀    Q₂ Q₃ ↔ K₁,V₁    Q₄ Q₅ ↔ K₂,V₂    Q₆ Q₇ ↔ K₃,V₃
```

### The Math

```
N_HEAD     = 8          (query heads, unchanged)
N_KV_HEAD  = 2          (key/value head groups)
head_dim   = N_EMBD / N_HEAD = 128 / 8 = 16
kv_dim     = N_KV_HEAD × head_dim = 2 × 16 = 32
group_size = N_HEAD / N_KV_HEAD = 8 / 2 = 4  (Q heads per KV group)

Wq: N_EMBD × N_EMBD  = 128 × 128 = 16,384 params  (unchanged)
Wk: N_EMBD × kv_dim  = 128 × 32  = 4,096 params   (4× smaller)
Wv: N_EMBD × kv_dim  = 128 × 32  = 4,096 params   (4× smaller)
```

### Memory Savings

| Component | MHA (8 KV heads) | GQA-2 (2 KV groups) | Savings |
|-----------|-----------------|---------------------|---------|
| Wk weights per layer | 16 KB | 4 KB | **4×** |
| Wv weights per layer | 16 KB | 4 KB | **4×** |
| KV cache per layer | 256 KB | 64 KB | **4×** |
| **Attention total (4 layers)** | **1,216 KB** | **400 KB** | **3×** |

For a **3-organelle pipeline** (like 8-puzzle):

| | MHA | GQA-2 | Savings |
|--|-----|-------|---------|
| Total KV cache | 3 × 1,024 KB = **3 MB** | 3 × 256 KB = **768 KB** | **4×** |

> **This is the difference between fitting on an ESP32 (520 KB SRAM) and not.**

### Implementation Sketch

The change affects three areas:

**1. Weight allocation** — Wk and Wv become `N_EMBD × kv_dim` instead of `N_EMBD × N_EMBD`:

```c
#ifndef N_KV_HEAD
#define N_KV_HEAD N_HEAD    // default: standard MHA
#endif
#define KV_DIM (N_KV_HEAD * (N_EMBD / N_HEAD))

// In model_create:
model->Wk[L] = calloc(N_EMBD * KV_DIM, sizeof(scalar_t));  // was N_EMBD²
model->Wv[L] = calloc(N_EMBD * KV_DIM, sizeof(scalar_t));  // was N_EMBD²
```

**2. KV projection** — project to smaller dimension:

```c
lin_fwd(x_norm1, model->Wq[L], ne, ne, q);          // full Q: N_EMBD → N_EMBD
lin_fwd(x_norm1, model->Wk[L], ne, KV_DIM, k);      // reduced K: N_EMBD → KV_DIM
lin_fwd(x_norm1, model->Wv[L], ne, KV_DIM, v);      // reduced V: N_EMBD → KV_DIM
```

**3. Attention loop** — map each Q head to its KV group:

```c
for (int h = 0; h < nh; h++) {
    int kv_group = h / (N_HEAD / N_KV_HEAD);    // which KV group this Q head uses
    size_t q_off  = h * hd;                      // Q offset: per Q head
    size_t kv_off = kv_group * hd;               // KV offset: shared within group

    for (size_t t = 0; t < T; t++) {
        const scalar_t *kt = KV_READ(keys, L, t, KV_DIM) + kv_off;
        scalar_t s = 0;
        for (size_t d = 0; d < hd; d++)
            s += q[q_off + d] * kt[d];
        attn_weights[h * bs + t] = s * scale;
    }
    // ... softmax, weighted sum using kv_off for values ...
}
```

### Quality Impact

| Study | Model Size | MHA → GQA | Quality Loss |
|-------|-----------|-----------|-------------|
| Llama 2 (Meta, 2023) | 70B params | 64 → 8 KV heads | **None measurable** |
| Mistral 7B (2023) | 7B params | 32 → 8 KV heads | **None measurable** |
| MicroGPT-C (projected) | 875K params | 8 → 2 KV heads | **TBD — experiment needed** |

> **Hypothesis:** At sub-1M scale, the redundancy between KV heads is likely *higher* than at large scale (less capacity to differentiate heads), so GQA should be even more effective.

### Backward Pass

The backward pass mirrors the forward: gradients for Wk and Wv accumulate into the smaller `N_EMBD × KV_DIM` matrices. The Q gradient path is unchanged. The `kv_group` mapping must be applied in reverse:

```c
// Backward: each Q head's gradient flows to its KV group
int kv_group = h / (N_HEAD / N_KV_HEAD);
// dK[kv_off + d] += dAttn[h,t] * q[q_off + d]  (accumulated across group)
```

---

## 3. Strategy 2 — Multi-Query Attention (MQA)

**Priority: MEDIUM — falls out of GQA for free**

### What It Is

MQA (Shazeer, 2019) is the extreme case of GQA where `N_KV_HEAD = 1`: all Q heads share a single K/V projection.

```
MQA (8 Q heads, 1 KV head):
  Q₀ Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇  ↔  K₀,V₀   (one shared set)
```

### Implementation

Zero additional code beyond GQA:

```bash
cmake -DN_KV_HEAD=1 ..    # MQA: all heads share one KV
cmake -DN_KV_HEAD=2 ..    # GQA-2: 4 heads per group
cmake -DN_KV_HEAD=4 ..    # GQA-4: 2 heads per group
cmake -DN_KV_HEAD=8 ..    # Standard MHA (default)
```

### Memory vs Quality Tradeoff

| Config | KV cache per layer (float32) | KV params per layer | Risk |
|--------|----------------------------|--------------------|----|
| MHA (N_KV_HEAD=8) | 256 KB | 32,768 | Baseline |
| GQA-4 (N_KV_HEAD=4) | 128 KB | 16,384 | Very low |
| GQA-2 (N_KV_HEAD=2) | 64 KB | 8,192 | Low |
| **MQA (N_KV_HEAD=1)** | **32 KB** | **4,096** | **Moderate** |

At MicroGPT-C's scale, the concern is that a single KV head (16 dimensions) may be too small to capture diverse attention patterns. **Start with GQA-2, test MQA as an extreme configuration.**

---

## 4. Strategy 3 — Sliding Window Attention

**Priority: LOW — relevant at BLOCK_SIZE ≥ 1024**

### What It Is

Standard attention computes scores over all previous positions — O(n²) in sequence length. Sliding Window Attention (SWA) limits attention to the most recent W positions:

```
Standard (BLOCK_SIZE=256):
  Position 200 attends to: [0, 1, 2, ..., 199, 200]  ← all 201 positions

Sliding Window (W=64):
  Position 200 attends to: [137, 138, ..., 199, 200]  ← only 64 positions
```

### When It Helps

```
Attention compute:  O(T × T × head_dim)

Standard at T=256:   256 × 256 × 16 = 1,048,576 ops
SWA at T=256, W=64:  256 × 64 × 16  = 262,144 ops     (4× fewer)

Standard at T=1024:  1,024 × 1,024 × 16 = 16,777,216 ops
SWA at T=1024, W=64: 1,024 × 64 × 16    = 1,048,576 ops  (16× fewer)
```

### Why Not Yet

At MicroGPT-C's current BLOCK_SIZE (16–256), the attention computation is already sub-microsecond. SWA's complexity savings don't outweigh the implementation cost. **But** for future organelle pipelines with long context (e.g., processing entire source files), SWA becomes essential.

### Implementation Notes

The change is localised to the attention loop — clamp the starting position:

```c
size_t t_start = (T > WINDOW_SIZE) ? (T - WINDOW_SIZE) : 0;
for (size_t t = t_start; t < T; t++) {
    // ... same attention computation ...
}
```

KV cache can then be made circular (ring buffer), eliminating the need to store the full sequence:

```c
size_t cache_pos = pos % WINDOW_SIZE;  // wrap around
memcpy(keys[L] + cache_pos * hd, k_h, hd * sizeof(scalar_t));
```

This changes KV cache from `BLOCK_SIZE × head_dim` to `WINDOW_SIZE × head_dim` — potentially a huge saving for long sequences.

### Mistral's Approach

Mistral 7B combines SWA (W=4096) with GQA (8 KV heads). For MicroGPT-C, the combination of GQA + SWA would be:

```
Window=64, GQA-2 (2 KV groups):
  KV cache per layer = 2 × 64 × 16 × 4 bytes = 8 KB
  vs. current:         8 × 256 × 16 × 4 bytes = 128 KB
  Savings: 16×
```

---

## 5. Strategy 4 — Multi-Head Latent Attention (MLA)

**Priority: RESEARCH — relevant for multi-organelle shared context**

### What It Is

MLA (DeepSeek-V2, 2024) compresses K/V into a low-rank latent space before caching:

```
Standard: x → Wk → K (N_EMBD) → cache
MLA:      x → W_down → latent (d_latent) → cache → W_up → K (N_EMBD)

Where d_latent << N_EMBD (e.g., 32 vs 128)
```

The KV cache stores the compressed latent, not the full K/V vectors. At inference time, the latent is decompressed back to K/V.

### Memory Analysis

```
Standard KV cache entry: N_EMBD × sizeof(scalar_t) = 128 × 4 = 512 bytes
MLA KV cache entry:      d_latent × sizeof(scalar_t) = 32 × 4 = 128 bytes (4× smaller)
```

### Why It's Interesting for Organelles

In a multi-organelle pipeline, organelles could share a compressed latent representation of the context instead of the full KV vectors. This is speculation, but it aligns with the pipe-separated wire format philosophy — keep the communication channel narrow.

### Why Not Yet

MLA requires additional projection matrices (W_down, W_up) that add parameters. At sub-1M scale, the parameter overhead may outweigh the cache savings. GQA achieves similar memory reductions with strictly fewer parameters.

---

## 6. Strategy 5 — Dynamic Sparse Attention

**Priority: RESEARCH — not recommended at this scale**

### What It Is

Instead of attending to all positions (dense) or a fixed window (SWA), Dynamic Sparse Attention selects which positions to attend to based on the content:

```
Dense:   attend to all T positions
SWA:     attend to last W positions  
Sparse:  attend to top-K positions by some scoring function
```

Examples include BigBird (random + window + global), Longformer (window + global), and routing-based approaches.

### Why Not at This Scale

1. **Overhead dominates:** The scoring function to select which positions to attend costs more than just computing all attentions at T ≤ 256
2. **Complexity:** Requires custom masking logic, non-trivial backward pass
3. **Unpredictable memory:** Sparse patterns make KV cache management harder
4. **No evidence of benefit:** At head_dim=16, each attention head is already extremely focused — forcing sparsity may hurt

### When It Becomes Relevant

If MicroGPT-C scales to BLOCK_SIZE ≥ 4096 with N_EMBD ≥ 512, sparse attention becomes necessary. At that point, a combination of SWA (for local context) + global tokens (for long-range) would be the recommended approach, following the Longformer pattern.

---

## 7. Comparison Matrix

### At Current Scale (N_EMBD=128, BLOCK_SIZE=256, N_HEAD=8)

| Variant | KV Cache/Layer | Weight Params/Layer | Quality Risk | Impl Complexity | Recommended? |
|---------|---------------|--------------------|-----------|----|---|
| **Standard MHA** | 256 KB | 49,152 | None (baseline) | None (current) | ✅ Current |
| **GQA-2** | 64 KB | 24,576 | Very low | ~50 lines | 🎯 **Next** |
| **GQA-4** | 128 KB | 32,768 | Low | Same as GQA-2 | ✅ Option |
| **MQA** | 32 KB | 20,480 | Moderate | Same as GQA-2 | ⚠️ Test first |
| **Sliding Window** | Depends on W | Same | Low (local tasks) | ~30 lines | 🔮 Future |
| **MLA** | 4× smaller | More params | Unknown | ~200 lines | 🔬 Research |
| **Dynamic Sparse** | Unpredictable | Same | High at small scale | ~500+ lines | ❌ Not yet |

### At Pipeline Scale (3 organelles × 4 layers)

| Config | Total KV Cache | Fits on ESP32? |
|--------|---------------|---------------|
| MHA | 3,072 KB | ❌ No (520 KB SRAM) |
| GQA-2 | 768 KB | ❌ No (but close) |
| GQA-2 + SWA(64) | 96 KB | ✅ **Yes** |
| MQA + SWA(64) | 24 KB | ✅ Yes (with room) |

---

## 8. Implementation Priority

### Phase 1: GQA (Q2 2026)

**Scope:** ~50 lines of code change

1. Add `N_KV_HEAD` compile-time constant (default = `N_HEAD` for backward compatibility)
2. Resize Wk/Wv allocation to `N_EMBD × KV_DIM`
3. Update `forward_backward_one` and `forward_inference` attention loops with `kv_group` mapping
4. Update `model_num_params` to reflect smaller KV weight count
5. Add unit tests comparing GQA output against MHA reference
6. Benchmark memory savings on 8-puzzle pipeline

**Success criteria:** All existing tests pass with `N_KV_HEAD=N_HEAD`. 8-puzzle solve rate remains ≥ 90% with `N_KV_HEAD=2`.

### Phase 2: SWA (Q3 2026)

**Scope:** ~30 lines + ring buffer KV cache

1. Add `WINDOW_SIZE` compile-time constant (default = `BLOCK_SIZE` for backward compatibility)
2. Clamp attention loop start position
3. Implement circular KV cache indexing
4. Test on Shakespeare (long-context task)

### Phase 3: MLA (Research, Q4 2026+)

**Scope:** ~200 lines, requires careful evaluation

Only pursue if GQA + SWA is insufficient for target deployment platforms.

---

## References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Vaswani et al., *Attention Is All You Need* | 2017 | Standard MHA |
| Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* | 2019 | MQA |
| Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* | 2023 | GQA |
| Jiang et al., *Mistral 7B* | 2023 | SWA + GQA combination |
| DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient MoE Model* | 2024 | MLA |
| Zaheer et al., *Big Bird: Transformers for Longer Sequences* | 2020 | Sparse attention |
| Beltagy et al., *Longformer: The Long-Document Transformer* | 2020 | Window + global tokens |

---

*Document generated February 2026. Analysis based on MicroGPT-C N_EMBD=128, N_HEAD=8, BLOCK_SIZE=256, float32 precision.*
