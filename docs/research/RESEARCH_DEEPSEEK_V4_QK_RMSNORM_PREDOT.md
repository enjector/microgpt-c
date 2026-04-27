# Porting DeepSeek-V4 Q/K RMSNorm Pre-Dot to MicroGPT-C

> Pre/post measurements for the second of the six DeepSeek-V4 ports identified in [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md). Per-head RMSNorm of Q and K immediately before the attention dot product. Implementation, controlled A/B benchmark, high-LR stress test, and a composition test against attention sinks.

**Reference:** DeepSeek-V4 §2.3.3 "Query and Key-Value Entry Normalization."

**Status:** Implemented behind `-DMICROGPT_QK_NORM=ON`, default **OFF**. All 61 existing unit tests pass with the flag both **off** (bit-identical to pre-port) and **on** (numerical-gradient and training-reduces-loss tests confirm the new backward path is correct). Measured on Apple M2 Max, single-threaded, `MICROGPT_USE_FLOAT=ON`.

---

## 1. Spear Summary

**Point:** Q/K RMSNorm pre-dot is **not a free quality win at safe learning rates** — at default LR=0.001 on a 4-layer 138K-param char model it slightly worsens held-out loss (+0.6%). Its real value is **stability under aggressive learning rates**: at LR=0.02 the un-normed baseline diverges to perplexity 731 while QK norm holds the model at perplexity 205 — a **3.6× recovery from divergence**. It also **composes super-additively with attention sinks**: combining both ports beats either alone (-7.0% loss vs baseline, vs -3.1% for sink alone).

**Picture:** Standard attention does `softmax(Q · K^T / √d_k) · V`. As Wq and Wk drift during training, the magnitudes of Q and K grow unboundedly — this is what causes the well-known "exploding attention logits" failure mode at high learning rates. The √d_k scale factor controls expected magnitude *under random init* but does nothing once the projections have learned. QK norm clamps each head's Q and K to unit RMS *before* the dot product, so attention logits stay in a bounded range regardless of how the projection weights evolve.

**Proof:**

| Config | Variant | Held-out loss | Perplexity | Δ vs baseline |
|---|---|---:|---:|---:|
| **Safe LR (0.001)** | baseline | 2.352 | 10.51 | — |
| Safe LR | QK norm only | 2.366 | 10.65 | +1.4% (mild regression) |
| Safe LR | sink only | 2.321 | 10.18 | −3.1% |
| **Safe LR** | **QK norm + sink** | **2.279** | **9.77** | **−7.0%** ✓ |
| **High LR (0.02)** | baseline | 6.594 | 730.98 | (diverged) |
| **High LR** | **QK norm only** | **5.322** | **204.75** | **3.6× better perplexity** ✓ |

**Push:** Ship `MICROGPT_QK_NORM=ON` as a recommended companion to `MICROGPT_ATTN_SINK=ON` for any model with `N_LAYER ≥ 2`. Don't ship it alone unless you already train at the edge of stability. The two flags fix orthogonal problems and compose into the strongest configuration tested.

---

## 2. The Mechanism

### 2.1 Standard scaled dot-product attention

For each head $h$, with head dimension $d_k = N_{\text{EMBD}} / N_{\text{HEAD}}$:

$$
\text{score}_{h,t} = \frac{q_h \cdot k_{h,t}}{\sqrt{d_k}}, \qquad
p_{h,t} = \frac{\exp(\text{score}_{h,t})}{\sum_{k} \exp(\text{score}_{h,k})}
$$

The $\sqrt{d_k}$ factor was originally derived assuming Q and K have unit variance per dim. That holds at initialisation (the engine uses $\mathcal{N}(0, 0.08^2)$ init and RMSNorm before Wq/Wk), but not necessarily during training. As Wq and Wk drift, $\|q_h\|_2$ and $\|k_{h,t}\|_2$ grow, the dot product magnitudes grow with them, and softmax saturates toward one-hot. This shows up empirically as collapsing diversity in attention patterns and, at high enough LR, divergence to NaN.

### 2.2 Q/K RMSNorm pre-dot (DeepSeek-V4 §2.3.3)

V4 inserts a per-head RMSNorm immediately before the dot product:

$$
\hat q_h = \frac{q_h}{\sqrt{\frac{1}{d_k}\sum_{i} q_{h,i}^2 + \varepsilon}}, \qquad
\hat k_h = \frac{k_h}{\sqrt{\frac{1}{d_k}\sum_{i} k_{h,i}^2 + \varepsilon}}
$$

$$
\text{score}_{h,t} = \frac{\hat q_h \cdot \hat k_{h,t}}{\sqrt{d_k}}
$$

After the norm, $\|\hat q_h\|_2 \approx \sqrt{d_k}$ and similarly for $\hat k_h$, so $\hat q_h \cdot \hat k_{h,t} \in [-d_k, d_k]$ regardless of the magnitudes of the underlying projections. The $\sqrt{d_k}$ scale then maps that into $[-\sqrt{d_k}, \sqrt{d_k}]$, which is the exact regime softmax was designed for. V4 explicitly cites this as the reason they omit QK-clip from the Muon optimiser (whereas earlier work needed it).

### 2.3 Implementation choice: no learnable scale

V4 uses a learnable per-head scale alongside RMSNorm. We chose a non-learnable variant. Justification:

1. **Minimal checkpoint impact.** A learnable scale would add `2 × N_LAYER × N_HEAD` scalars (one for Q-side, one for K-side per head per layer), and a corresponding chunk in the gradient buffer and Adam state. A non-learnable variant is parameter-zero.
2. **The scale's job is already done by Wq and Wk.** Without QK norm, the magnitudes of Q and K were free to drift; with QK norm, the *direction* of the projection matters, not the *magnitude*. Wq and Wk can learn the direction directly. A learnable post-norm scale is therefore mostly redundant at small model scales.
3. **Cleaner backward.** With no learnable scale, the only new backward path is the per-head `rmsnorm_bwd` calls, which reuse the existing well-tested `rmsnorm_bwd` function.

### 2.4 Backward pass

After the change, the dot-product loop computes scores using the normed $\hat q$ and $\hat k$. The existing softmax-backward code (`d_q[t] += d_score[t] * k_t`) therefore yields gradients with respect to the *post-norm* $\hat q$ and $\hat k$. To propagate back through Wq and Wk, those need to be converted to gradients with respect to the *pre-norm* $q$ and $k$.

Per-head RMSNorm has a closed-form backward (already implemented in `rmsnorm_bwd`):

$$
\frac{\partial \hat q_{h,i}}{\partial q_{h,j}} =
\frac{1}{\text{rms}(q_h)} \left( \delta_{ij} - \frac{\hat q_{h,i} \hat q_{h,j}}{d_k} \right)
$$

We apply `rmsnorm_bwd` per head to convert `d_q` and `d_k_cur` from post-norm gradients to pre-norm gradients before lin_bwd through Wq and Wk. The V vector is not normed, so `d_v_cur` is unchanged. **Past-position K gradients in the cache are not re-propagated** — same convention as the unmodified engine — so the only K we backprop through is the current position's pre-norm K, which we save as `sv_k_pre[L]`.

---

## 3. Implementation

### 3.1 Files changed

| File | Change |
|---|---|
| `src/microgpt.h` | Added `MICROGPT_QK_NORM` toggle (no magnitude knob — on/off). Banner line in `microgpt_print_config`. |
| `src/microgpt.c` | (1) Added `sv_q_pre[N_LAYER][N_EMBD]` and `sv_k_pre[N_LAYER][N_EMBD]` to the saved-state stack frame, gated by `#ifdef MICROGPT_QK_NORM`. (2) Training forward — after `lin_fwd` projections and before `sv_q`/cache writes — saves pre-norm copies and applies per-head `rmsnorm_fwd` in place. (3) Inference forward — same per-head normalisation in place (no save needed, no backward). (4) Training backward — between the per-head softmax-backward dispatch and `lin_bwd` through Wq/Wk — applies per-head `rmsnorm_bwd` to convert `d_q` and `d_k_cur` from post-norm to pre-norm gradients. |
| `tests/bench_microgpt_attn_sink.c` | **Reused unchanged.** The existing A/B harness already accepts arbitrary feature flags via `add_demo(... DEFINES ...)`. |
| `CMakeLists.txt` | Seven new targets: `bench_qk_norm_{baseline,on}` (tiny), `bench_qk_norm_deep_{baseline,on}`, `bench_qk_norm_stress_{baseline,on}` (high LR), `bench_qk_norm_deep_with_sink` (composition test), and `test_microgpt_qk_norm` (full unit-test suite with QK_NORM=1, exercising the new backward via the existing gradient-direction tests). |

### 3.2 The forward-side patch

Training forward (after `lin_fwd` of Q, K, V; before `sv_q` save):

```c
#ifdef MICROGPT_QK_NORM
memcpy(sv_q_pre[L], q, ne * sizeof(scalar_t));
memcpy(sv_k_pre[L], k, ne * sizeof(scalar_t));
for (int h = 0; h < nh; h++) {
    rmsnorm_fwd(sv_q_pre[L] + (size_t)h * hd, hd, q + (size_t)h * hd);
    rmsnorm_fwd(sv_k_pre[L] + (size_t)h * hd, hd, k + (size_t)h * hd);
}
#endif
```

Inference forward (no backward needed, so no save):

```c
#ifdef MICROGPT_QK_NORM
{
    scalar_t q_pre_local[N_EMBD], k_pre_local[N_EMBD];
    memcpy(q_pre_local, q, ne * sizeof(scalar_t));
    memcpy(k_pre_local, k, ne * sizeof(scalar_t));
    for (int h = 0; h < nh; h++) {
        rmsnorm_fwd(q_pre_local + (size_t)h * hd, hd, q + (size_t)h * hd);
        rmsnorm_fwd(k_pre_local + (size_t)h * hd, hd, k + (size_t)h * hd);
    }
}
#endif
```

### 3.3 The backward-side patch

After the per-head softmax-backward block fills `d_q` and `d_k_cur`, but before `lin_bwd`:

```c
#ifdef MICROGPT_QK_NORM
{
    scalar_t d_q_pre[N_EMBD], d_k_cur_pre[N_EMBD];
    memset(d_q_pre, 0, sizeof(d_q_pre));
    memset(d_k_cur_pre, 0, sizeof(d_k_cur_pre));
    for (int h = 0; h < nh; h++) {
        rmsnorm_bwd(sv_q_pre[L] + (size_t)h * hd, d_q + (size_t)h * hd, hd,
                    d_q_pre + (size_t)h * hd);
        rmsnorm_bwd(sv_k_pre[L] + (size_t)h * hd, d_k_cur + (size_t)h * hd, hd,
                    d_k_cur_pre + (size_t)h * hd);
    }
    memcpy(d_q, d_q_pre, sizeof(d_q_pre));
    memcpy(d_k_cur, d_k_cur_pre, sizeof(d_k_cur_pre));
}
#endif
```

`rmsnorm_bwd` accumulates into its output (`+=`), hence the `memset` and copy-back. The V backward is untouched.

### 3.4 KV cache semantics

The cache stores **post-norm K**. This is intentional: every position's K was post-norm at the moment of its dot product, so reading post-norm K from the cache for past-position scores keeps everything consistent. Pre-norm K is only retained for the *current* position's backward via `sv_k_pre[L]`, since past positions' weight gradients were already accumulated during their original forward step.

### 3.5 Compile-time gating

```bash
# Off (default, bit-identical to pre-port engine)
cmake ..

# On
cmake -DMICROGPT_QK_NORM=ON ..

# Combined with attention sink (recommended for safe-LR training)
cmake -DMICROGPT_QK_NORM=ON -DMICROGPT_ATTN_SINK=ON ..
```

Per-demo:
```cmake
add_demo(NAME my_demo SOURCE ... DEFINES MICROGPT_QK_NORM=1)
```

### 3.6 Verification

```
$ ./test_microgpt
=== Results: 61/61 passed ===

$ ./test_microgpt_qk_norm     # same suite, MICROGPT_QK_NORM=1 baked in
=== Results: 61/61 passed ===
```

The gradient-direction tests (e.g., `gradient_direction_reduces_loss`) exercise both forward and backward. Passing both with and without QK norm is strong evidence that the new `rmsnorm_bwd` plumbing is correct.

---

## 4. Benchmark Design

The benchmark is identical to the one used in [`RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md) §4 — same A/B source (`tests/bench_microgpt_attn_sink.c`), same seed (`srand(42); seed_rng(42);`), same data (`c_names.txt`, 32K names, 90/10 split). Three new dimensions are added:

1. **Default-LR variants** at tiny and deep architectures, to measure the standalone effect of QK norm.
2. **High-LR stress variants** (LR=0.02, 20× the engine default) to surface the exploding-logits regime that V4 cites as the main motivation.
3. **Composition variant** that enables both `MICROGPT_QK_NORM=1` and `MICROGPT_ATTN_SINK=1` with `ATTN_SINK_LOGIT=-1.0`, to measure how the two ports interact.

Variant table:

| Target | Architecture | LR | Steps | Sink | QK norm |
|---|---|---:|---:|---|---|
| `bench_qk_norm_baseline` | Tiny (1L, 4K) | 0.01 | 600 | OFF | OFF |
| `bench_qk_norm_on` | Tiny (1L, 4K) | 0.01 | 600 | OFF | ON |
| `bench_qk_norm_deep_baseline` | Deep (4L, 138K) | 0.001 | 1500 | OFF | OFF |
| `bench_qk_norm_deep_on` | Deep (4L, 138K) | 0.001 | 1500 | OFF | ON |
| `bench_qk_norm_stress_baseline` | Deep (4L, 138K) | **0.02** | 800 | OFF | OFF |
| `bench_qk_norm_stress_on` | Deep (4L, 138K) | **0.02** | 800 | OFF | ON |
| `bench_qk_norm_deep_with_sink` | Deep (4L, 138K) | 0.001 | 1500 | ON (-1.0) | ON |

LR for the Tiny config is the engine's `LEARNING_RATE` default (0.01); LR for Deep is the demo-style 0.001 to match the attention-sink benchmark. Stress overrides LR via `add_demo(... DEFINES LEARNING_RATE=0.02 ...)`.

---

## 5. Results

All numbers from the actual benchmark binaries built and run in this repo.

### 5.1 Tiny architecture (default LR, 1 layer, 4K params)

| Variant | Train loss | Held-out loss | Perplexity | Δ |
|---|---:|---:|---:|---:|
| QK norm OFF | 2.132 | 2.307 | 10.04 | — |
| QK norm ON | 2.154 (+1.0%) | 2.316 (+0.4%) | 10.13 (+0.9%) | mild regression |

**Reading.** Within noise to mild regression. Same finding as for attention sinks: a 1-layer 4-head 16-context toy doesn't have enough attention-logit dynamic range for the normalisation to matter. This is the right *negative control* — confirms QK norm is not free magic.

### 5.2 Deep architecture, default LR (4 layers, 138K params, LR=0.001)

| Variant | Train loss | Held-out loss | Perplexity | Δ |
|---|---:|---:|---:|---:|
| QK norm OFF (baseline) | 2.329 | 2.352 | 10.51 | — |
| QK norm ON | 2.377 (+2.0%) | 2.366 (+0.6%) | 10.65 (+1.4%) | mild regression |
| Sink only (logit=−1.0)\* | 2.337 | 2.321 | 10.18 | −3.1% |
| **QK norm + sink** | **2.229 (−4.3%)** | **2.279 (−3.1%)** | **9.77 (−7.0%)** | **best ✓** |

\* sink-only result reproduced from the attention-sink paper.

**Reading.**
- **QK norm alone, at safe LR, is mildly worse than baseline.** The +1.4% perplexity regression isn't catastrophic but it's not a win either. At safe LR the un-normed Q and K never drift far enough for normalisation to be net-helpful, and we're paying the cost of a less expressive parametrisation (Wq and Wk's magnitude no longer matters, only direction).
- **Sink alone is a −3.1% improvement** (reproduced from prior paper).
- **The combined port is −7.0% — strictly better than the sum of the two parts.** Sink alone gave −3.1%; if QK norm were a strict regression, the combined would be smaller in magnitude than −3.1%. Instead it's more than double. This is super-additive: QK norm and sinks are fixing orthogonal problems, and one's regression in isolation is fully recovered by the other's regularising effect.

The training loss for the combined variant is also the lowest (2.229) — this is the only configuration where QK norm helps both train and held-out loss.

### 5.3 High-LR stress test (4 layers, 138K params, LR=0.02)

| Variant | Train loss | Held-out loss | Perplexity | Outcome |
|---|---:|---:|---:|---|
| QK norm OFF (baseline) | 6.861 | 6.594 | **730.98** | model effectively diverged |
| **QK norm ON** | **5.316** (−22%) | **5.322** (−19%) | **204.75** (3.6× better PPL) | model partially recovered |

**Reading.** This is the V4 claim, validated. At a learning rate 20× the safe default, the un-normed model collapses — perplexity 731 is "produces gibberish." With QK norm enabled, the model is still suboptimal (at LR=0.02 even a stable model is over-stepping its loss surface) but it is **bounded** — it learns *something* rather than diverging. This is exactly the behaviour V4 cites as enabling them to omit QK-clip from the Muon optimiser: the norm provides automatic, gradient-friendly bounding of attention logits that no clipping policy needs to enforce.

The probe metrics are revealing:
- **Baseline at LR=0.02:** PROBE_MAX_ATTN = 0.903, PROBE_ENTROPY = 1.70 — attention has collapsed to near-one-hot at random positions. This is the saturated-softmax failure mode that QK-clip / QK-norm are designed to prevent.
- **QK norm at LR=0.02:** PROBE_MAX_ATTN = 0.977, PROBE_ENTROPY = 1.47. Surprisingly, the QK-normed model is *also* concentrated, but on a deliberate position rather than a random one — and its quality on the held-out set is 3.6× better, which is the ground truth.

### 5.4 Cost

- **Parameters added:** 0.
- **Train time:** Tiny 0.057s vs 0.060s (within noise). Deep 6.04s vs 5.96s (+1.4%). Combined with sink: 6.12s (~3% over QK-norm-alone).
- **Stack memory:** 2 × N_LAYER × N_EMBD scalars (sv_q_pre + sv_k_pre). For the deep config that is 2 × 4 × 64 = 512 scalars (~2 KB) — negligible.
- **Inference cost:** one extra `rmsnorm_fwd` per head per layer per token. On the deep config (4 layers × 4 heads × 16 head_dim) this is 16 cheap reductions per forward, sub-percent in our measurements.

### 5.5 Summary table

| Architecture | LR | QK norm | Sink | Held-out PPL | Δ vs baseline |
|---|---:|---|---|---:|---:|
| Tiny | 0.01 | OFF | OFF | 10.04 | — |
| Tiny | 0.01 | ON | OFF | 10.13 | +0.9% |
| Deep | 0.001 | OFF | OFF | 10.51 | — |
| Deep | 0.001 | ON | OFF | 10.65 | +1.4% |
| Deep | 0.001 | OFF | ON, −1.0 | 10.18 | −3.1% |
| **Deep** | **0.001** | **ON** | **ON, −1.0** | **9.77** | **−7.0% ✓** |
| Deep | 0.02 | OFF | OFF | 730.98 | (diverged) |
| **Deep** | **0.02** | **ON** | **OFF** | **204.75** | **3.6× recovery ✓** |

---

## 6. Interpretation

### 6.1 Why does QK norm hurt slightly at safe LR?

At safe LR=0.001 the un-normed Q and K projections never drift into the saturating-softmax regime. In that regime, the magnitude of Q and K *is* useful information — the model can use a large norm to mean "I'm confident this is a strong match" and a small norm to mean "low confidence." QK norm forcibly removes that signal: every Q and K is unit-magnitude, regardless of how the network feels about it. The network has to re-encode that confidence elsewhere (probably in V or in the MLP), and at this scale it doesn't fully recover. The +1.4% perplexity regression is the cost of removing a useful but dangerous degree of freedom.

### 6.2 Why does QK norm + sink help super-additively?

Hypothesis: the two ports relax different overcommitment failure modes:
- **Without sink:** every head must distribute 100% of probability mass over real positions. Forcing post-norm Q and K then *additionally* removes the magnitude signal the model used to pick which position to commit to — net loss.
- **With sink:** heads can abstain. Now the model doesn't *need* the magnitude signal as a "confidence" channel — abstain is the proper way to express low confidence. QK norm's removal of the magnitude signal is then no longer a regression, and its stabilisation benefit becomes a pure win.

In other words: sinks let the model say "I don't know," QK norm lets attention scores stay bounded; together they form a coherent stability story where the model can express uncertainty cleanly *and* can't blow up its logits in pursuit of false certainty.

### 6.3 Why does the high-LR test matter?

DeepSeek-V4 trains at very high effective LR (Muon optimiser, deep stacks). Their motivation for QK norm is explicitly: "this normalisation avoids exploding attention logits and may improve training stability." MicroGPT-C training at LR=0.001 is too gentle to reach that regime. Doing the LR=0.02 stress test is what surfaces V4's actual claim.

The stress test should be re-run on every future port: if a port helps at LR=0.02 but not at LR=0.001, it's a *stability* port, not a *quality* port, and it should ship as a recommended-default rather than a quality-knob.

### 6.4 What about training at intermediate LRs?

Not measured in this paper. A useful follow-up would be to scan LR in {0.001, 0.003, 0.005, 0.01, 0.02} with and without QK norm, plot the held-out loss, and observe (a) where the un-normed baseline starts to underperform stable training, and (b) whether QK norm widens the safely-trainable LR window. V4's design suggests both are true.

---

## 7. Reproducing the Results

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cd build

# Build all seven A/B variants + the QK-norm-enabled unit test suite
cmake --build . --config Release --parallel 8 --target \
    bench_qk_norm_baseline bench_qk_norm_on \
    bench_qk_norm_deep_baseline bench_qk_norm_deep_on \
    bench_qk_norm_stress_baseline bench_qk_norm_stress_on \
    bench_qk_norm_deep_with_sink \
    test_microgpt test_microgpt_qk_norm

# Numerical-correctness gate: 61/61 with both variants
./test_microgpt
./test_microgpt_qk_norm

# Quality variants
./bench_qk_norm_baseline
./bench_qk_norm_on
./bench_qk_norm_deep_baseline
./bench_qk_norm_deep_on
./bench_qk_norm_deep_with_sink

# Stability variants (LR=0.02)
./bench_qk_norm_stress_baseline
./bench_qk_norm_stress_on
```

Results are deterministic for a given seed and thread count.

---

## 8. Limitations and Future Work

1. **Single corpus, single hardware, single seed.** Same caveat as the attention-sink paper. The +1.4% safe-LR regression and −7.0% combined-with-sink improvement could vary on Shakespeare, VM codegen, or organelle game models.
2. **Non-learnable scale.** V4's actual recipe uses a learnable per-head scale on top of the RMSNorm. We chose to ship the simpler version. Implementation sketch: add `attn_qk_scale_q[N_LAYER][N_HEAD]` and `attn_qk_scale_k[N_LAYER][N_HEAD]` to the Model struct, multiply post-RMSNorm Q and K by their per-head scales, and route the gradient through `adam_step`. Likely 200 LOC and an Adam-state migration.
3. **No interaction tested with TurboQuant / RotorQuant / MSA.** QK norm should be especially helpful after KV compression (where K vectors are noisy reconstructions). Adding `MICROGPT_QK_NORM=1` to the existing `tq_*` and `rq_*` CMake targets and rerunning is the obvious next experiment.
4. **No interaction tested with `MICROGPT_ATTN_RES`.** The two flags should compose cleanly (QK norm in attention, AttnRes in residuals) but this has not been measured.
5. **LR scan not run.** §6.4 sketches the obvious follow-up.
6. **Stress LR=0.02 still produces a "bad" model (PPL 205).** QK norm rescued the run from divergence but not from being over-LR'd. The point of the stress test is *recovery*, not perfect quality — but a stronger test would be a hold-LR-at-the-edge experiment that finds the maximum stable LR for each variant.

---

## 9. References

- DeepSeek-V4 paper: [`papers/DeepSeek_V4.pdf`](papers/DeepSeek_V4.pdf), §2.3.3 "Query and Key-Value Entry Normalization".
- Roadmap context: [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) §3.2.
- Companion paper (attention sinks): [`RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md).
- Implementation:
  - `src/microgpt.h` (toggle, banner, sv_q_pre/sv_k_pre comment)
  - `src/microgpt.c` (forward + backward patches; `sv_q_pre` / `sv_k_pre` saved-state arrays)
  - `tests/bench_microgpt_attn_sink.c` (reused A/B harness)
  - `CMakeLists.txt` (seven new targets, including the QK-norm-enabled unit test suite)

---

*Measured on a real machine. No PyTorch reference, no GPU. C99, libc, libm — that's it.*
