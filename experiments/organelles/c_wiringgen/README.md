# Observation: Prompt-Conditioned C Function Composition with MicroGPT-C

**Date:** 18 February 2026
**Model:** MicroGPT-C character-level Transformer (wiring organelle)
**Author:** Ajay Soni, Enjector Software Ltd.

---

## 1. Objective

To determine whether a **875K-parameter character-level Transformer** can learn to
**compose** known primitive functions into higher-level operations. While the
`c_codegen` organelle (see `OBSERVATION_C_CODEGEN.md`) proved that the model can
**retrieve** individual function bodies from prompts, this experiment tests whether
a dedicated "wiring" model can learn the **composition patterns** — how to chain,
pipeline, and aggregate functions together.

The key questions:

1. Can the model learn `/* composition description */ → function body that calls known primitives`?
2. Can it reproduce known wiring patterns (e.g. "smooth then differentiate")?
3. **Can it generalise to novel compositions** that chain primitives in unseen ways?
4. Does a corpus of compositions provide better generalisation than a corpus of implementations?

### Motivation: The Two-Organelle Architecture

Experiment 5 in the c_codegen report definitively proved that a single model cannot
compose novel functions — it operates as a retrieval system. This organelle tests
the hypothesis that **separation of concerns** can unlock composition:

```
  c_codegen organelle          c_wiringgen organelle
  ──────────────────           ─────────────────────
  Learns: WHAT each            Learns: HOW to compose
  function does                functions together
  Corpus: 2,081 raw            Corpus: 864 composition
  C function bodies            patterns calling primitives
  Output: function body        Output: wiring/glue code
```

Together, these form a **two-organelle pipeline** where the wiring model decomposes
intent into primitive calls, and the code model generates each primitive's body.

---

## 2. Architecture & Configuration

| Parameter       | Value   | Same as c_codegen? |
|-----------------|---------|--------------------|
| `N_EMBD`        | 128     | ✅ Yes              |
| `N_LAYER`       | 4       | ✅ Yes              |
| `N_HEAD`        | 4       | ✅ Yes              |
| `BLOCK_SIZE`    | 512     | ✅ Yes              |
| `BATCH_SIZE`    | 16      | ✅ Yes              |
| `LEARNING_RATE` | 0.0003  | ✅ Yes              |
| `NUM_STEPS`     | 50,000  | ✅ Yes              |
| `GEN_LEN`       | 400     | ✅ Yes              |
| `CODEGEN_TEMP`  | 0.3     | ✅ Yes              |
| Total params    | 868,096 | ≈ Same (875K)      |

The model architecture is identical to the c_codegen organelle — same decoder-only
Transformer with RMSNorm, RoPE, multi-head attention, SwiGLU, and AdamW. The only
difference is the training corpus.

---

## 3. Training Corpus

The corpus (`c_wiring.txt`) contains **864 function compositions** — functions whose
bodies **call known primitives** from the c_codegen vocabulary rather than implementing
raw logic.

### Corpus Statistics

| Metric               | c_codegen          | c_wiringgen        | Notes |
|----------------------|--------------------|--------------------|-------|
| Functions            | 2,081              | 864                | 2.4× smaller |
| Corpus size          | 492 KB             | 171 KB             | 2.9× smaller |
| Lines                | ~15,000            | 5,710              | 2.6× smaller |
| Unique characters    | 90                 | 63                 | Simpler vocab |
| Avg function length  | ~237 chars          | ~198 chars         | Shorter bodies |
| Params/byte ratio    | 1.78:1             | **5.08:1**         | Over-parameterised |
| Base functions       | 462 originals      | **174 originals**  | + variations |
| Variations           | 1,619 (3.5× each)  | **690 (4× each)**  | Comment variants |

> [!IMPORTANT]
> The params-to-data ratio of **5.08:1** is significantly higher than c_codegen's
> 1.78:1. This means the model can trivially memorise the entire corpus. The
> question is whether the memorised **patterns** (function chaining, pipeline
> composition) are more generalisable than memorised **implementations** (raw
> loop bodies, arithmetic expressions).

### Corpus Categories

All compositions are **domain-agnostic** (horizontal), using primitives that could
apply to finance, defence, automotive, scientific computing, or any domain:

| Category | Base Count | Example Composition |
|----------|-----------|---------------------|
| Array transforms (map, fill, scale, clamp) | ~25 | `sigmoid_array` — applies `sigmoid()` to each element |
| Two-pass (stat → transform) | ~20 | `normalize_z` — `mean()` → subtract → `stddev()` → divide |
| Pipeline / chaining | ~30 | `filter_downsample` — `lowpass()` → `downsample()` |
| Aggregation (multi-array → scalar) | ~15 | `variance_ratio` — `variance(a)` / `variance(b)` |
| Windowed operations | ~20 | `rolling_zscore` — `rolling_mean()` + `running_stddev()` per element |
| Conditional routing | ~5 | `adaptive_smooth` — `variance()` → choose smoothing period |
| Multi-step workflows | ~25 | `detrend_fft` — `rolling_mean()` → subtract → `fft_radix2()` |
| Utility (copy, pad, interleave) | ~20 | `zero_pad` — for-loop fill + memcpy pattern |
| Matrix/vector compositions | ~10 | `matvec_relu` — `mat_vec()` → `relu()` per element |

### Design Principles

1. **Horizontal, not domain-specific.** No finance-specific or physics-specific
   compositions. Domain adaptation is a higher-level organelle responsibility.

2. **Only calls known primitives.** Every function body calls primitives from the
   c_codegen vocabulary (`mean`, `stddev`, `rolling_mean`, `fft_radix2`, `lowpass`,
   `vec_normalize`, `dot`, `softmax`, etc.). The model learns to wire, not to implement.

3. **Compositional patterns, not algorithms.** The corpus teaches "A then B",
   "if X then A else B", "compute X, use X to transform Y" — the grammar of
   function composition rather than the content of individual functions.

### Primitive Vocabulary

The wiring corpus references ~100 distinct primitives from the c_codegen organelle:

**Scalar primitives:** `mean`, `stddev`, `variance`, `median`, `min_val`, `max_val`,
`dot`, `cosine_similarity`, `correlation`, `covariance`, `entropy`, `signal_energy`,
`signal_rms`, `snr`, `mse`, `mae`, `rmse`, `vec_norm`, `autocorrelation`, etc.

**Array primitives:** `rolling_mean`, `ema_series`, `sma`, `cumsum`, `diff_central`,
`softmax`, `vec_add`, `vec_scale`, `vec_normalize`, `bubble_sort`, `insertion_sort`,
`minmax_normalize`, `zscore_normalize`, `histogram`, `linreg`, `fir_filter`,
`lowpass`, `bandpass_filter`, `downsample`, `upsample`, `fft_radix2`,
`fft_magnitude`, `fft_phase`, `hamming_window`, `hilbert_envelope`,
`median_filter`, `moving_avg`, `moving_max`, `moving_min`, `compute_ranks`,
`running_stddev`, `gradient_descent`, `mat_vec`, `outer_product`, etc.

### Example Corpus Entries

```c
/* normalize array to zero mean unit variance */
void normalize_z(double *out, const double *x, int n) {
  double m = mean(x, n);
  double s = stddev(x, n);
  for (int i = 0; i < n; i++) out[i] = (x[i] - m) / s;
}

/* chain lowpass filter then downsample */
void filter_downsample(double *out, const double *x, int n, int factor) {
  double tmp[1024];
  lowpass(tmp, x, n, 0.5 / factor);
  downsample(out, tmp, n, factor);
}

/* compute rolling z-score */
void rolling_zscore(double *out, const double *x, int n, int period) {
  double rm[1024], rs[1024];
  rolling_mean(rm, x, n, period);
  running_stddev(rs, x, n, period);
  for (int i = 0; i < n; i++) out[i] = (x[i] - rm[i]) / (rs[i] + 1e-10);
}

/* two layer linear transform */
void two_layer(double *out, const double *M1, const double *M2,
               const double *x, int d1, int d2, int d3) {
  double tmp[1024];
  mat_vec(tmp, M1, x, d2, d1);
  for (int i = 0; i < d2; i++) tmp[i] = relu(tmp[i]);
  mat_vec(out, M2, tmp, d3, d2);
}
```

Note the pattern: each function is a **composition** — allocate temp buffer, call
primitive A, optionally transform, call primitive B, return. The model sees hundreds
of variations of this pattern.

---

## 4. Inference Method

Identical to c_codegen:

1. **BOS token** initialises the KV cache.
2. Prompt string (e.g. `/* normalize then smooth array */\n`) is fed character-by-character.
3. Model autoregressively generates characters using temperature-scaled sampling (T=0.3).
4. Generation stops at double newline (end of function) or after `GEN_LEN` characters.

Confidence scoring (softmax probability of the next predicted character) tracks
prompt recognition and generation quality.

---

## 5. Test Prompts

### Novel Wiring (Not in Corpus)

These prompts describe compositions that the model has **never seen**, but that
should be constructible from the patterns it has learned:

| # | Prompt | Reasoning |
|---|--------|-----------|
| 1 | `/* normalize then smooth array */` | Should chain `zscore_normalize()` → `rolling_mean()` |
| 2 | `/* denoise and downsample */` | Should chain `lowpass()` → `downsample()` — similar to `filter_downsample` |
| 3 | `/* bandpass then compute energy */` | Should chain `bandpass_filter()` → `signal_energy()` |
| 4 | `/* sort then trim and average */` | Should chain `bubble_sort()` → `trimmed_mean()` |
| 5 | `/* ema followed by differentiation */` | Should chain `ema_series()` → `diff_central()` |
| 6 | `/* detrend then compute spectrum */` | Similar to `detrend_fft` but asking for spectrum not raw FFT |
| 7 | `/* rank then compute correlation */` | Should chain `compute_ranks()` → `correlation()` |
| 8 | `/* filter then compute autocorrelation */` | Should chain `lowpass()` → `autocorrelation()` |
| 9 | `/* smooth two arrays then compute distance */` | Should smooth A, smooth B, then `vector_distance()` |
| 10 | `/* normalize then compute cosine similarity */` | Should normalize A, normalize B, then `cosine_similarity()` |

### Control Prompts (In Corpus)

| # | Prompt | Expected Output |
|---|--------|-----------------|
| 1 | `/* normalize array to zero mean unit variance */` | `normalize_z()` — exact corpus match |
| 2 | `/* smooth then differentiate signal */` | `smooth_diff()` — exact corpus match |
| 3 | `/* chain lowpass filter then downsample */` | `filter_downsample()` — exact corpus match |
| 4 | `/* compute rolling z-score */` | `rolling_zscore()` — exact corpus match |
| 5 | `/* two-stage smoothing with different periods */` | `double_smooth()` — exact corpus match |

---

## 6. Training Dynamics

> [!NOTE]
> Training has been started. This section will be updated with loss curves,
> convergence data, and generation results as the 50,000-step training progresses.

### Initial Status

```
loaded 864 wiring compositions (multi-line documents)
total characters: 173080 (169.0 KB)
vocab: 63 characters
N_EMBD=128 BLOCK_SIZE=512 N_LAYER=4 N_HEAD=4

params: 868096 | batch 16 | steps 50000 | lr 0.0003 | threads 12

step     1 / 50000 | loss 6.6080
```

### Predictions

- **Params/byte ratio:** 5.08:1 — the model will memorise the corpus easily
- **Convergence speed:** Faster than c_codegen (less data), expected final loss < 0.05
- **Vocab:** Only 63 characters (vs 90 for c_codegen) — simpler distribution to learn
- **Control prompts:** Should achieve near-perfect recall (same as c_codegen at 1.78:1)

### Key Hypothesis

The critical question is **novel prompt #2**: `/* denoise and downsample */`. The
corpus contains `/* chain lowpass filter then downsample */` → `filter_downsample()`.
If the model truly learns the **composition pattern** (filter → downsample), it
should recognise "denoise and downsample" as a variant of this pattern, even though
the exact words differ — because it has learned many examples of "A then B" wiring.

Similarly, `/* ema followed by differentiation */` should map to the same
`ema_series() → diff_central()` chain as `/* smooth then differentiate signal */`,
just with different comment wording.

This is precisely where the wiring organelle might outperform c_codegen for novel
prompts: the corpus teaches **composition grammar** rather than specific implementations,
so the model has more exposure to structural variation.

---

## 7. Expected Results & Analysis Framework

### Success Criteria

| Tier | Condition | What It Proves |
|------|-----------|----------------|
| **Tier 1** | Controls work (5/5 exact recall) | Model memorises compositions like c_codegen memorises functions |
| **Tier 2** | Novel prompts #2, #5 work (similar to corpus entries) | Model generalises within close semantic neighbours |
| **Tier 3** | Novel prompts #7–#10 work (structurally novel chains) | Model has learned composition grammar |
| **Tier 4** | All 10 novel prompts work | Model is a genuine composition engine |

### Comparison Framework

After training completes, results will be compared against c_codegen's Experiment 5:

```
  c_codegen (Exp 5)          c_wiringgen
  ──────────────────         ──────────────────
  Corpus: raw C bodies       Corpus: composition patterns
  Novel prompts: 0/10 ❌      Novel prompts: ?/10
  Controls: 3/5 ✅            Controls: ?/5
  Composition: IMPOSSIBLE    Composition: ???
```

If the wiring organelle achieves even **1/10 novel compositions**, it would prove
that a corpus of composition patterns generalises better than a corpus of raw
implementations — validating the two-organelle architecture.

---

## 8. Architectural Context

### The Organelle Chain Vision

```
  User: "compute z-scored rolling average"
          │
          ▼
  ┌──────────────────────┐
  │  c_wiringgen          │  Recognises: "normalize → smooth"
  │  (wiring organelle)   │  Emits: zscore_normalize() → rolling_mean()
  └────────┬─────────────┘
           │
           ▼
  ┌──────────────────────┐
  │  c_codegen            │  Retrieves: zscore_normalize body + rolling_mean body
  │  (code organelle)     │  Emits: complete C implementations
  └────────┬─────────────┘
           │
           ▼
  Complete C function with composition + implementation
```

### Domain Organelles (Future Work)

The wiring corpus is deliberately **horizontal** — it teaches generic composition
patterns (A→B, if-then-else, two-pass, pipeline). Domain-specific composition
(e.g. "compute risk-adjusted return" → Sharpe pipeline) would be handled by
a third **domain organelle** that maps domain language to horizontal patterns:

```
  Domain organelle: "risk-adjusted return" → "normalize then compute ratio"
  Wiring organelle: "normalize then compute ratio" → mean() / stddev() chain
  Code organelle: mean() body, stddev() body
```

This three-tier architecture separates **domain knowledge**, **composition grammar**,
and **implementation detail** into independently trainable, composable models.

---

## 9. Reproduction

```bash
# Build
cmake -B build -DMICROGPT_BLAS=ON
cmake --build build --target c_wiringgen

# Train (50K steps, ~3–5 hours on 12 threads)
cd build && ./c_wiringgen

# Resume from checkpoint (automatic)
./c_wiringgen   # loads c_wiringgen.ckpt if present

# Generate variations (if modifying base corpus)
cd examples/c_wiringgen
python3 generate_variations.py
```

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
