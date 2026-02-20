# MicroGPT-C Training Strategies

**Learning rate scheduling, hyperparameter tuning, and capacity scaling
for the MicroGPT-C organelle training system.**

*Copyright © 2026 Ajay Soni, Enjector Software Ltd. MIT License.*

---

## Spear Summary

**Point:** LR scheduling (warmup + cosine decay) already exists in `adam_step()`, but the defaults are tuned for small models (~462K params). At 1.2M+ params, the default `WARMUP_STEPS=100` is catastrophically insufficient — training diverges after step ~7K.

**Picture:** It's like flooring the accelerator before the engine has warmed up. The first 100 steps aren't enough for Adam's moment estimates to stabilize at 1.2M params, so when peak lr kicks in, the gradients explode and never recover.

**Proof:** c_compose at 462K (WARMUP_STEPS=100, lr=0.001): best loss 0.072, 96% parse. c_compose at 1.2M (same settings): loss diverged to 2.0+ at step 7K, 20% parse.

**Push:** Set `WARMUP_STEPS = 5-10% × NUM_STEPS` and `lr ∝ 1/√(num_params)` for models above 500K params. Use the recommended hyperparameter table at the end of this document.

---

## Table of Contents

1. [LR Schedule: Linear Warmup + Cosine Decay](#1-lr-schedule-linear-warmup--cosine-decay)
2. [Warmup Ratio Guidelines](#2-warmup-ratio-guidelines)
3. [LR-Capacity Scaling](#3-lr-capacity-scaling)
4. [Empirical Evidence: c_compose Experiments](#4-empirical-evidence-c_compose-experiments)
5. [Recommended Hyperparameters by Scale](#5-recommended-hyperparameters-by-scale)
6. [Adam Optimizer Configuration](#6-adam-optimizer-configuration)

---

## 1. LR Schedule: Linear Warmup + Cosine Decay

MicroGPT's `adam_step()` implements a two-phase learning rate schedule:

```
LR │  peak_lr
   │  /‾‾‾‾‾‾‾‾‾‾‾\
   │ /              \  cosine decay
   │/                \
   │                  \___
   └────────────────────── step
   │warmup│    decay phase
```

### Phase 1: Linear Warmup (step 0 → WARMUP_STEPS)

```c
lr = peak_lr * (step + 1) / WARMUP_STEPS;
```

LR ramps linearly from 0 to `peak_lr`. This allows Adam's moment estimates
(m and v) to stabilize before the full learning rate is applied.

### Phase 2: Cosine Decay (step WARMUP_STEPS → NUM_STEPS)

```c
progress = (step - warmup) / (total - warmup);
lr = peak_lr * 0.5 * (1.0 + cos(progress * π));
```

LR follows a half-cosine curve from `peak_lr` to ~0. Smoother than linear
decay — avoids the sudden drop at the end that can destabilize training.

### Implementation

The schedule is built into `adam_step()` in `microgpt.c` (line ~2271).
It uses compile-time constants `WARMUP_STEPS`, `NUM_STEPS`, and
`LEARNING_RATE`, all overridable via CMake defines:

```cmake
DEFINES ... WARMUP_STEPS=2500 LEARNING_RATE=0.0005 NUM_STEPS=50000
```

---

## 2. Warmup Ratio Guidelines

The warmup period must be long enough for Adam's second-moment estimate (v)
to converge to a reasonable approximation of the gradient variance. With
β₂ = 0.99, the half-life of v is ~70 steps, and it takes ~500 steps for v
to capture 99% of the true variance.

### Default vs Recommended

| Model Size | Default WARMUP | Issue | Recommended |
|-----------|---------------|-------|------------|
| <200K params | 100 (10% of 1K) | ✅ Works fine | 100 |
| ~462K params | 100 (0.4% of 25K) | ⚠️ Marginal | 500-1000 |
| ~1.2M params | 100 (0.2% of 50K) | ❌ **Diverges** | 2500-5000 |

### Rule of Thumb

```
WARMUP_STEPS = max(100, NUM_STEPS * 0.05)    # minimum 5% warmup
```

For longer runs (>50K steps), 5% is sufficient. For shorter runs (<5K),
keep the absolute minimum at 100 to allow v estimates to stabilize.

### Why 100 Steps Diverges at 1.2M Params

At 1.2M params with WARMUP_STEPS=100:

1. **Step 0-100**: LR ramps 0 → 0.001. Adam's v estimates are still
   dominated by the initial zero bias — they underestimate gradient variance
   by 10-100×.
2. **Step 100**: Peak lr=0.001 hits. v is still too small, so the
   *effective* per-parameter lr is much larger than intended.
3. **Step 500-1000**: Gradient norms spike as attention weights saturate.
4. **Step 5000-10000**: Training enters a divergent regime — loss increases
   monotonically from 0.2 → 2.0+ and never recovers.

With WARMUP_STEPS=2500:

1. **Step 0-2500**: LR ramps slowly. v estimates converge to true variance.
2. **Step 2500**: Peak lr arrives when Adam is correctly calibrated.
3. **Step 2500-50000**: Cosine decay smoothly reduces lr to ~0.

---

## 3. LR-Capacity Scaling

### The lr ∝ 1/√(params) Rule

Standard practice in deep learning: larger models need smaller learning rates.
The gradient contribution of each parameter scales as ~1/√(n), so:

```
peak_lr = base_lr × √(base_params / current_params)
```

### MicroGPT Scaling Table

| Config | Params | base_lr | Scaled lr | Recommended |
|--------|--------|---------|-----------|-------------|
| 16/4/1 (tests) | ~4K | 0.01 | - | 0.01 |
| 64/4/2 (names) | ~50K | 0.003 | - | 0.003 |
| 96/8/4 (c_compose v1) | 462K | 0.001 | - | 0.001 |
| 128/8/6 (c_compose v2) | 1.2M | - | 0.001 × √(462K/1.2M) = 0.0006 | **0.0005** |
| 256/8/8 (hypothetical) | ~5M | - | 0.001 × √(462K/5M) = 0.0003 | 0.0003 |

### Why Peak LR Matters More Than Schedule Shape

The cosine decay provides a good schedule shape, but the *peak value* is the
critical parameter. An lr that's 2× too high at peak will diverge regardless
of how smooth the decay is — the damage happens early when weights get pushed
to bad regions, and cosine decay can't undo it.

---

## 4. Empirical Evidence: c_compose Experiments

Three experiments on the c_compose pipeline (intent → function plan):

### v1 Baseline: Small + Short

| Parameter | Value |
|-----------|-------|
| Config | 96/8/4 (462K params) |
| Training | 25K steps, lr=0.001, WARMUP=100 |
| Corpus | 512 train |

| Metric | Result |
|--------|--------|
| Planner best loss | 0.072 |
| Plan parse rate | **96%** (123/128) |
| Registry hit | 4% (5/128) |
| Exact match | 2% (2/128) |
| Training time | 7 min |

**Analysis**: Schedule works well. 100 warmup steps = 0.4% of 25K is marginal
but the smaller 462K model is forgiving. Loss converges smoothly.

### v2 All Improvements (Diverged)

| Parameter | Value |
|-----------|-------|
| Config | 128/8/6 (1.2M params) |
| Training | 50K steps, lr=0.001, WARMUP=100 |
| Corpus | 1452 train (expanded) |
| Constrained decoding | Edit distance ≤3 |

| Metric | Result |
|--------|--------|
| Planner best loss | 0.14 (early checkpoint) |
| Plan parse rate | **20%** (26/128) ↓76% |
| Registry hit | 11% (14/128) ↑7% |
| Exact match | 0% (0/128) ↓2% |
| Training time | 45 min |

**Analysis**: Loss diverged after step ~7K from 0.14 → 2.0+. Best-checkpoint
mechanism preserved step-7K weights, but those were still severely underfit.
Root cause: WARMUP=100 at 50K steps = 0.2%, combined with lr=0.001 being
too high for 1.2M params.

### v3 Corrected Schedule (Pending)

| Parameter | Value |
|-----------|-------|
| Config | 128/8/6 (1.2M params) |
| Training | 50K steps, lr=0.0005, **WARMUP=2500** |
| Corpus | 1452 train (expanded) |
| Constrained decoding | Edit distance ≤3 |

Hypothesis: With 5% warmup and halved peak lr, the 1.2M model should train
stably through all 50K steps, achieving lower final loss than both v1 and v2.

---

## 5. Recommended Hyperparameters by Scale

These are the recommended CMake defines for different model sizes:

### Tiny (< 100K params) — tests, names_demo

```cmake
DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=2
        NUM_STEPS=5000 LEARNING_RATE=0.003 WARMUP_STEPS=100
```

### Small (100K–500K params) — c_codegen, c_compose v1

```cmake
DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4
        NUM_STEPS=25000 LEARNING_RATE=0.001 WARMUP_STEPS=500
```

### Medium (500K–2M params) — c_compose v3, game demos

```cmake
DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=6
        NUM_STEPS=50000 LEARNING_RATE=0.0005 WARMUP_STEPS=2500
```

### Large (2M+ params) — future experiments

```cmake
DEFINES N_EMBD=256 N_HEAD=8 N_LAYER=8
        NUM_STEPS=100000 LEARNING_RATE=0.0003 WARMUP_STEPS=5000
```

---

## 6. Adam Optimizer Configuration

### Fixed Hyperparameters

These are compile-time defaults in `microgpt.h` and should rarely need changing:

| Parameter | Default | Notes |
|-----------|---------|-------|
| β₁ | 0.85 | 1st moment decay (lower than standard 0.9 for faster adaptation) |
| β₂ | 0.99 | 2nd moment decay (standard value) |
| ε | 1e-8 | Numerical stability (standard value) |

### Why β₁ = 0.85 (Not 0.9)

MicroGPT uses β₁ = 0.85, slightly lower than the standard 0.9. At small
scale (< 1M params), the gradient signal is less noisy, so a shorter momentum
window (decay half-life ~4.3 steps vs ~6.6 steps) allows faster adaptation
to changing gradient directions. This is particularly important for
organelle training where each corpus is small (< 5K docs).

### Warmup Interaction with β₂

The warmup period needs to be long enough for β₂-driven v estimates to
converge. With β₂ = 0.99:

- **Half-life of v**: ~69 steps
- **99% convergence**: ~460 steps
- **99.9% convergence**: ~690 steps

This is why `WARMUP_STEPS ≥ 500` is recommended for production models —
it allows >7 half-lives of v to accumulate, ensuring the per-parameter
adaptive learning rate is well-calibrated before peak lr arrives.
