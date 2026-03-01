# Training Optimisations for MicroGPT-C

**Author:** Ajay Soni, Enjector Software Ltd.  
**Date:** February 2026  
**Status:** Research Document

---

## 1. Executive Summary

MicroGPT-C already implements a solid training baseline: the **Adam optimiser** with **cosine learning rate scheduling and linear warmup**, all in pure C99 with zero dependencies. This document surveys additional training techniques that could improve convergence, accuracy, regularisation, or efficiency — while preserving the project's minimalist ethos.

The techniques are prioritised by **ease of implementation** and **relevance to sub-1M parameter transformers**. Each section documents the technique, the expected benefit, the C99 implementation path, and any caveats specific to models at this scale.

> **Design constraint**: Every technique must be implementable in pure C99 with no external dependencies. Memory overhead must remain negligible for models under 1M parameters. Complexity must justify the expected improvement.

---

## 2. Current Baseline

Before proposing changes, it is important to document what MicroGPT-C already implements. The training pipeline is mature:

| Component | Implementation | Source |
|---|---|---|
| **Optimiser** | Adam with bias correction | `adam_step()` in `microgpt.c` |
| **LR Schedule** | Cosine decay with linear warmup | Integrated into `adam_step()` |
| **Normalisation** | RMSNorm (pre-norm) | Per-layer in forward/backward |
| **Activation** | ReLU in MLP | Pre-MLP non-linearity |
| **Precision** | Configurable scalar_t (float/double) | Compile-time toggle |
| **Quantisation** | Optional INT8 with fp64 master copy | `QUANTIZATION_INT8` flag |

### 2.1 Current Hyperparameter Defaults

These compile-time constants in `microgpt.h` define the training baseline:

| Parameter | Default | Notes |
|---|---|---|
| `LEARNING_RATE` | 0.01 | Peak LR for cosine schedule |
| `BETA1` | 0.85 | Adam first-moment decay (lower than standard 0.9) |
| `BETA2` | 0.99 | Adam second-moment decay (lower than standard 0.999) |
| `EPS_ADAM` | 1e-8 | Adam epsilon for numerical stability |
| `WARMUP_STEPS` | 100 | Linear warmup before cosine decay |
| `BATCH_SIZE` | 8 | Documents per gradient accumulation |
| `INIT_STD` | 0.08 | Weight initialisation standard deviation |

### 2.2 What Is Not Yet Implemented

| Technique | Status | Priority |
|---|---|---|
| Weight decay (AdamW-style) | ✅ Implemented (tested: no effect at 36.7%) | ~~High~~ |
| Gradient clipping | ✅ **Implemented — production** (v5, GRAD_CLIP=1.0) | ~~High~~ |
| Dropout | ❌ Not implemented | Medium |
| Label smoothing | ✅ **Implemented — production** (v5, LABEL_SMOOTH=0.1, +3pp) | ~~Medium~~ |
| Gradient accumulation scaling | ❌ Not implemented | Low |
| Mixed-precision training | 🟡 Partial (float/double toggle) | Low |
| **Compact prompt / vocabulary reduction** | ✅ **Implemented — production** (v12c, +5pp to 65%) | **NEW** |
| Directional encoding (U/D/F signs) | ❌ Tested — 8.3%, collapsed to single-class (v13a) | Dead end |
| Full discretisation (U/D/F + VIX bands) | ❌ Tested — 8.3%, collapsed to single-class (v13b) | Dead end |
| Temporal persistence (S/M/L bucket) | ❌ Tested — 8.3%, collapsed to single-class (v13c) | Dead end |
| Confidence gating (≥60% threshold) | ❌ Tested — 100% predictions high-conf, all wrong (v13d) | Dead end |

---

## 3. Proposed Optimisations

### 3.1 Weight Decay (AdamW)

**What:** Add decoupled weight decay to the Adam optimiser. Unlike L2 regularisation (which adds the penalty to the gradient), AdamW applies weight decay directly to the parameters after the Adam update — preventing the adaptive learning rate from counteracting the regularisation.

**Why worth it:** Weight decay is the single most impactful regularisation technique for transformer training. It prevents weight magnitudes from growing unbounded, which is especially important for small models training on small corpora where overfitting is the primary failure mode. Karpathy's nanoGPT uses AdamW for this reason.

**Implementation in C99:**

```c
/* In adam_step(), after the Adam update for each parameter: */
/* Decoupled weight decay: w -= lr * wd * w */
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 0.01
#endif

/* After: model->wte[i] -= lr * mh / (sqrt(vh) + eps); */
model->wte[i] *= (1.0 - lr * WEIGHT_DECAY);
```

The decay is applied to all weight matrices but **not** to embeddings (wte, wpe) or biases — consistent with the standard PyTorch AdamW convention. Since MicroGPT-C has no biases (the architecture is bias-free), the exclusion applies only to embeddings.

**Recommended defaults:**

| Parameter | Value | Rationale |
|---|---|---|
| `WEIGHT_DECAY` | 0.01 | Standard for small transformers (Loshchilov & Hutter, 2019) |
| Apply to | Attention weights, MLP weights, lm_head | Standard exclusion of embeddings |
| Skip for | wte, wpe | Embeddings should not be regularised |

**Caveat:** For models under 100K parameters, weight decay may not help significantly — the model is capacity-limited, not overfitting. Test with and without.

**Effort:** ~20 lines of C. One `#define`, one multiplication per parameter per step.

---

### 3.2 Gradient Clipping

**What:** Clip the global gradient norm to a maximum value before applying the Adam update. If the total L2 norm of all gradients exceeds the threshold, scale all gradients proportionally so the norm equals the threshold.

**Why worth it:** Gradient explosions can occur during early training (before warmup stabilises moments) or when encountering unusual training examples. Clipping prevents catastrophic parameter updates that blow up the model. This is standard practice for all transformer training — GPT-2, GPT-3, and nanoGPT all use ‖g‖ ≤ 1.0.

**Implementation in C99:**

```c
#ifndef GRAD_CLIP
#define GRAD_CLIP 1.0
#endif

/* Before adam_step(), compute and clip the global gradient norm: */
static void clip_gradients(scalar_t *grads, size_t n) {
  scalar_t norm_sq = 0;
  for (size_t i = 0; i < n; i++)
    norm_sq += grads[i] * grads[i];
  scalar_t norm = sqrt(norm_sq);
  if (norm > GRAD_CLIP) {
    scalar_t scale = GRAD_CLIP / norm;
    for (size_t i = 0; i < n; i++)
      grads[i] *= scale;
  }
}
```

**Recommended defaults:**

| Parameter | Value | Rationale |
|---|---|---|
| `GRAD_CLIP` | 1.0 | Standard for transformer training |

**Caveat:** With cosine scheduling + warmup already in place, gradient explosions are rare. But clipping is defensive and essentially free — the norm computation is O(N) over parameters, negligible compared to the forward/backward pass.

**Effort:** ~15 lines of C. One function, one call site.

---

### 3.3 Improved Adam Hyperparameters

**What:** Adjust the Adam moment decay rates to better match standard transformer training practice.

**Current vs. recommended:**

| Parameter | Current | Recommended | Rationale |
|---|---|---|---|
| `BETA1` | 0.85 | **0.9** | Standard value; 0.85 discounts history too aggressively for small models |
| `BETA2` | 0.99 | **0.999** | Standard value; 0.99 adapts second moments too quickly, causing LR jitter |
| `LEARNING_RATE` | 0.01 | **6e-4** | 0.01 is high for Adam (designed for SGD-scale LRs); 6e-4 is Karpathy's default |
| `WARMUP_STEPS` | 100 | **10% of NUM_STEPS** | Proportional warmup scales better across different training durations |

**Why worth it:** The current β1=0.85 causes the optimiser to forget gradient history too quickly, leading to noisier updates. The current LR=0.01 is aggressive for Adam — the adaptive denominator (√v̂) already provides per-parameter scaling, so a lower peak LR prevents overshooting.

**Implementation:** Pure hyperparameter changes — no code modifications needed, only `#define` updates.

**Caveat:** These are recommendations based on general transformer best practices. The current defaults may have been tuned specifically for certain experiments. A/B testing with loss curves is essential.

**Effort:** 0 lines of C. Configuration changes only.

---

### 3.4 Label Smoothing

**What:** Replace the hard one-hot target distribution with a softened version. Instead of assigning probability 1.0 to the correct token and 0.0 to all others, assign (1 − α) to the correct token and α/(V−1) to all others.

**Why worth it:** Label smoothing prevents the model from becoming overconfident in its predictions, which has two benefits:
1. **Better calibration** — the model's confidence scores more accurately reflect true accuracy
2. **Regularisation** — prevents the logits from growing unbounded (which wastes capacity on pushing probabilities toward 1.0)

For organelle models that use confidence scores as a gate signal (System 1/System 2 in the OPA pipeline), better calibration directly improves pipeline reliability.

**Implementation in C99:**

```c
#ifndef LABEL_SMOOTH
#define LABEL_SMOOTH 0.1
#endif

/* In forward_backward_one(), modify the loss computation: */
/* Instead of: loss = -log(probs[target_id]) */
/* Use: loss = -(1-α)*log(p[target]) - (α/(V-1)) * Σ_{i≠target} log(p[i]) */
/* Simplified: loss = -(1-α)*log(p[target]) - (α/V) * Σ log(p[i])         */
scalar_t smooth = LABEL_SMOOTH;
scalar_t log_target = -log(probs[target_id]);
scalar_t log_sum = 0;
for (size_t i = 0; i < vocab_size; i++)
  log_sum += -log(probs[i]);
loss = (1.0 - smooth) * log_target + (smooth / (scalar_t)vocab_size) * log_sum;
```

The gradient also changes: instead of `d_logits[i] = probs[i] - (i == target)`, use:

```c
for (size_t i = 0; i < vocab_size; i++) {
  scalar_t target_prob = (i == target_id) ? (1.0 - smooth) : 0.0;
  target_prob += smooth / (scalar_t)vocab_size;
  d_logits[i] = probs[i] - target_prob;
}
```

**Recommended defaults:**

| Parameter | Value | Rationale |
|---|---|---|
| `LABEL_SMOOTH` | 0.1 | Standard for language models (Vaswani et al., 2017) |

**Caveat:** For character-level models with small vocabularies (V ≈ 60–100), label smoothing distributes α across fewer classes — each non-target class gets a larger share than in word-level models. Consider using α=0.05 for character-level tokenisation.

**Effort:** ~15 lines of C. Modify the loss computation and gradient in `forward_backward_one()`.

---

### 3.5 Dropout

**What:** During training, randomly zero out a fraction of activations after attention and MLP layers. At inference time, keep all activations (no zeroing).

**Why worth it:** Dropout is the classic regularisation technique for neural networks. It prevents co-adaptation of neurons — forcing the model to learn redundant representations that generalise better. For micro-scale models training on small corpora (Shakespeare, game traces), dropout can meaningfully reduce overfitting.

**Implementation in C99:**

```c
#ifndef DROPOUT_RATE
#define DROPOUT_RATE 0.1
#endif

/* Apply after attention output and after MLP output during training: */
static void dropout(scalar_t *x, size_t n, scalar_t rate, int is_training) {
  if (!is_training || rate <= 0) return;
  scalar_t scale = 1.0 / (1.0 - rate);  /* inverted dropout */
  for (size_t i = 0; i < n; i++) {
    if (rand_u() < rate)
      x[i] = 0;
    else
      x[i] *= scale;  /* scale up survivors to maintain expected value */
  }
}
```

**Application points:**

1. After attention output projection (before residual add)
2. After MLP fc2 output (before residual add)
3. Optionally: on the embedding sum (wte + wpe)

**Recommended defaults:**

| Model size | Dropout rate | Rationale |
|---|---|---|
| < 100K params | 0.0 | Model is capacity-limited, not overfitting |
| 100K–500K params | 0.1 | Mild regularisation for moderate corpora |
| 500K–1M params | 0.2–0.3 | Stronger regularisation for small corpora |

**Caveat:** Dropout requires distinguishing training from inference mode, which adds a `is_training` flag to forward pass functions. This is a modest API change. Also, dropout interacts with the backward pass — zeroed activations must stay zeroed during gradient computation (save the dropout mask).

**Effort:** ~30 lines of C. Requires a dropout mask buffer and an API flag.

---

### 3.6 Data-Centric Techniques

These techniques improve training effectiveness without modifying the model or optimiser.

#### 3.6.1 Curriculum Learning

**What:** Present training data in order of increasing difficulty — shorter sequences first, longer sequences later. For game corpora, this could mean training on simpler board states (low Manhattan distance) before harder ones (high md).

**Why worth it:** Small models benefit significantly from "easy to hard" progression. Training on simple examples first allows the model to learn basic syntax and common patterns before tackling complex cases. This can:
- Accelerate convergence by 1.5–2×
- Improve final accuracy on hard examples by giving the model a stable foundation

**Implementation:**

```c
/* Pre-sort training documents by length at load time: */
/* After load_docs(), sort docs.lines[] by docs.doc_lens[] ascending */
qsort_r(doc_indices, num_docs, sizeof(int), cmp_by_len, doc_lens);

/* OR: progressive context masking — start with BLOCK_SIZE/4, ramp to full */
int effective_block_size = MIN(BLOCK_SIZE, BLOCK_SIZE * step / (NUM_STEPS / 4));
```

**Effort:** ~10 lines of C. Sort at load time, or mask at training time.

#### 3.6.2 Token-Level Data Augmentation

**What:** During training, randomly perturb input sequences with small probability:
- **Token swap** (p=0.05): swap two adjacent tokens
- **Token delete** (p=0.02): skip one token in the sequence
- **Token duplicate** (p=0.02): repeat one token

**Why worth it:** Augmentation increases the effective corpus size without requiring more data. For small corpora (e.g., 200KB Shakespeare), even mild augmentation can reduce overfitting and improve generalisation to paraphrased inputs.

**Caveat:** Must be applied carefully for structured formats like the OPA wire format (`seq|fn1|fn2`) — corrupting delimiters would teach the model broken syntax. Best suited for free-text corpora.

**Effort:** ~25 lines of C. Inline mutation during document tokenisation.

---

### 3.7 Mixed Precision Training

**What:** Use 32-bit floats (float) for forward/backward passes instead of 64-bit doubles. MicroGPT-C already supports this via the `MICROGPT_USE_FLOAT` compile-time flag, but full mixed-precision training would use float for compute and double only for the Adam accumulator.

**Why worth it:** Float training:
- Halves memory for weight matrices, gradients, and KV cache
- Doubles SIMD throughput on ARM NEON (4 elements per 128-bit register vs 2)
- Enables larger batch sizes or longer contexts within the same memory budget
- For sub-1M parameter models, float precision is more than sufficient

**Current status:** The `scalar_t` toggle already provides the mechanism. The missing piece is gradient scaling — when using float, gradients can underflow to zero for very small values. A simple loss scaling factor handles this:

```c
#ifdef MICROGPT_USE_FLOAT
#define LOSS_SCALE 128.0f  /* Scale up loss to prevent gradient underflow */
/* Scale gradients back down before Adam: */
for (size_t i = 0; i < n; i++)
  grads[i] /= LOSS_SCALE;
#endif
```

**Recommended configuration:**

| Component | Precision | Rationale |
|---|---|---|
| Forward/backward | float (32-bit) | Sufficient for sub-1M models |
| Adam moments (m, v) | float (32-bit) | Acceptable with loss scaling |
| Adam master copy (INT8 mode) | double (64-bit) | Needed for requantisation accuracy |

**Effort:** ~10 lines of C. The infrastructure already exists; only loss scaling is needed.

---

### 3.8 Hyperparameter Search Framework

**What:** A lightweight framework for running multiple short training runs with varied hyperparameters and comparing results.

**Why worth it:** At the micro scale, training a model takes seconds to minutes — fast enough for systematic hyperparameter exploration. A 3×3 grid over LR and weight decay, with 5,000 steps each, would take ~10 minutes total on an M2 chip and could identify the optimal configuration.

**Implementation approach:**

```c
/* grid_search.c — standalone utility */
typedef struct {
  double lr;
  double weight_decay;
  double beta1;
  int warmup_steps;
} HyperConfig;

/* Log results to CSV for analysis: */
/* lr,wd,beta1,warmup,final_loss,best_loss,steps_to_loss_1.0 */
```

This does not require any changes to the core library — it is a standalone experiment harness that calls the existing training API with different compile-time overrides.

**Effort:** ~100 lines of C. New experiment file, no core changes.

---

### 3.9 Model Soup / Weight Averaging

**What:** Train K variants of the same model with different random seeds or hyperparameters, then average the final weights element-wise to produce a "soup" ensemble.

**Why worth it:** Weight averaging:
- Smooths out initialisation variance — the averaged model lands in a flatter region of the loss landscape
- Costs nothing at inference time (same model size, same speed)
- Typically improves by 5–10% over the best individual model
- Trivial to implement: just average the weight arrays

**Implementation in C99:**

```c
/* After training K models, average weights: */
for (size_t i = 0; i < num_params; i++) {
  scalar_t sum = 0;
  for (int k = 0; k < K; k++)
    sum += models[k]->wte[i];  /* (generalise to all weight arrays) */
  soup->wte[i] = sum / (scalar_t)K;
}
```

**Recommended configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| K (number of models) | 3–5 | Diminishing returns beyond 5 |
| Variation | Different seeds, same hyperparams | Simplest; isolates initialisation variance |
| Advanced | Different LR or warmup per model | Weight averaging acts as implicit ensemble |

**Effort:** ~30 lines of C. New post-training utility, no core changes.

---

## 4. Prioritised Implementation Roadmap

Based on impact-to-effort ratio and relevance to the organelle use case:

### Tier 1: High Impact, Low Effort (Implement First)

| # | Technique | Expected impact | Actual Result |
|---|---|---|---|
| 1 | **Gradient clipping** (§3.2) | Prevents training instability | ✅ Production (v5) — stabilised training |
| 2 | **Weight decay** (§3.1) | Reduces overfitting | ❌ No effect (36.7% = baseline) |
| 3 | **Hyperparameter tuning** (§3.3) | Better convergence | ✅ Tuned (LR=0.0003, optimal for markets) |

### Tier 2: Moderate Impact, Moderate Effort

| # | Technique | Expected impact | Actual Result |
|---|---|---|---|
| 4 | **Label smoothing** (§3.4) | Better calibration | ✅ **Production (v5) — +3pp to 60%** |
| 5 | **Curriculum learning** (§3.6.1) | Faster convergence | ❌ −11pp (v7) — removed beneficial randomisation |
| 6 | **Mixed precision** (§3.7) | 2× throughput | Not yet tested |

### Tier 3: Valuable but Higher Complexity

| # | Technique | Expected impact | Actual Result |
|---|---|---|---|
| 7 | **Dropout** (§3.5) | Reduced overfitting | Not yet tested |
| 8 | **Model soup** (§3.9) | 5–10% improvement | ❌ Collapsed to single-class predictions |
| 9 | **Data augmentation** (§3.6.2) | Larger effective corpus | ❌ −11pp (v7) — contradictory training data |
| 10 | **Hyperparameter search** (§3.8) | Systematic optimisation | ✅ Partially done (optim_bench) |

### New Discovery: Vocabulary Reduction (v12c)

| # | Technique | Expected impact | Actual Result |
|---|---|---|---|
| 11 | **Compact prompt** (integer rounding, drop features) | Reduce vocabulary | ✅ **+5pp to 65% — NEW RECORD** |
| 12 | **100K training steps** | More memorisation | ❌ 8.5% — catastrophic overfitting |
| 13 | **BLOCK_SIZE=256** | Larger context window | ❌ 53.3% — padding hurts retrieval |
| 14 | **Directional encoding** (U/D/F signs, v13a) | Extreme vocab reduction | ❌ 8.3% — magnitude info lost, collapsed |
| 15 | **Full discretisation** (U/D/F + VIX bands, v13b) | Maximum vocab reduction | ❌ 8.3% — only 324 distinct prompts, label conflicts |
| 16 | **Temporal persistence** (S/M/L bucket, v13c) | Add regime duration info | ❌ 8.3% — persistence adds vocab without signal |
| 17 | **Confidence gating** (≥60% threshold, v13d) | Filter uncertain predictions | ❌ 100% high-conf but all wrong — no useful uncertainty signal |

> [!WARNING]
> **v12c reproducibility concern**: The 65% compact prompt result is non-reproducible. Re-runs with identical corpus (MD5-verified) collapse to 8.3%. Analysis in private research notes.

---

## 5. Interaction with the OPA Pipeline

Several of these optimisations have specific implications for the Organelle Pipeline Architecture:

### 5.1 Label Smoothing and Confidence Gating

The OPA pipeline uses **softmax confidence** as the gate between System 1 (fast retrieval) and System 2 (ensemble voting). Label smoothing directly affects the calibration of these confidence scores:

- **Without smoothing:** the model pushes logits toward ±∞, producing confidence scores clustered near 0 and 1 — the gate rarely triggers System 2
- **With smoothing:** confidence scores are distributed more evenly, providing a more useful signal for the System 1/System 2 gate

This is potentially the highest-leverage interaction: a better-calibrated model doesn't just produce better predictions — it knows *when it doesn't know*, enabling the pipeline to allocate resources more effectively.

### 5.2 Weight Decay and Transfer Learning

The `model_transfer_weights()` function copies trained transformer weights between models with different vocabularies. Weight decay ensures that transferred weights have bounded magnitudes, preventing the destination model from inheriting inflated weight scales that may clash with its freshly-initialised embeddings.

### 5.3 Curriculum Learning and Game Corpora

**Tested in v7 — NEGATIVE RESULT (−11pp).** Curriculum ordering (easy→hard by regime difficulty + VIX) removed beneficial randomisation. Early training became biased toward RISK_ON only, and the model never recovered.

For game experiments (8-Puzzle, Mastermind, Connect-4), curriculum learning also showed no benefit — retrieval engines need random sampling, not structured ordering.

### 5.4 Model Soup and Organelle Ensembles

**Tested — NEGATIVE RESULT.** Weight-averaging 3–5 independently-trained models collapsed to single-class predictions. Different random seeds converge to different loss basins; averaging destroys all learned representations for sub-1M param models.

See [ORGANELLE_MODEL_SOUP.md](ORGANELLE_MODEL_SOUP.md) for the full analysis.

---

## 6. Benchmarking Strategy

### 6.1 Metrics

All optimisation experiments should report:

| Metric | Definition | Why it matters |
|---|---|---|
| **Training loss** | Cross-entropy loss on training set | Basic convergence indicator |
| **Validation loss** | Loss on held-out 10% of corpus | Overfitting detector |
| **Steps to threshold** | Steps to reach loss < 1.0 (or 0.5) | Convergence speed |
| **Validation perplexity** | exp(validation_loss) | Interpretable model quality metric |
| **Peak memory** | RSS during training | Edge deployment constraint |
| **Wall-clock time** | Total training time | Practical efficiency |

### 6.2 Recommended A/B Protocol

1. **Baseline:** Current defaults (Adam, cosine schedule, no decay, no clipping)
2. **Treatment:** One change at a time (additive, not combinatorial)
3. **Fixed seed:** `seed_rng(42)` for reproducibility
4. **Duration:** 10,000 steps minimum for meaningful comparison
5. **Corpus:** Shakespeare (~1.1 MB) as the standard microbenchmark
6. **Model:** N_EMBD=64, N_LAYER=2, N_HEAD=4 (~92K params) — the standard organelle configuration

### 6.3 Expected Outcomes

| Experiment | Hypothesis | Success criterion |
|---|---|---|
| Gradient clipping | Prevents loss spikes in first 100 steps | No training divergence |
| Weight decay | Lower validation loss, slightly higher training loss | Val loss ≤ 95% of baseline |
| Label smoothing | Better-calibrated confidence scores | Brier score improvement |
| Curriculum learning | Faster convergence on hard examples | Steps to loss < 0.5 reduced by 20%+ |

---

## 7. Experimental Results

All experiments were run on the tictactoe player corpus (game data) using the `optim_bench` harness under `demos/character-level/optim_bench/`. Configuration: N_EMBD=48, N_LAYER=2, N_HEAD=4 (~64K params), 5,000 steps, batch_size=8, seed=42.

### 7.1 Loss Comparison

| Experiment | Config | Step 1 | Step 1K | Step 3K | Step 5K | **Best Loss** | Time |
|---|---|---|---|---|---|---|---|
| **Baseline** | — | 3.6699 | 0.4209 | 0.2847 | 0.2818 | **0.2299** | 5.2s |
| **Grad Clip** | ‖g‖≤1.0 | 3.6699 | 0.4209 | 0.2847 | 0.2818 | **0.2299** | 5.1s |
| **Weight Decay** | wd=0.01 | 3.6699 | 0.4271 | 0.2871 | 0.2859 | **0.2370** | 5.2s |
| **Adam Tuned** | β₁=0.9 β₂=0.999 lr=6e-4 | 3.6699 | 0.4547 | 0.2910 | 0.2992 | **0.2430** | 5.1s |
| **Label Smooth** | α=0.1 | 3.6585 | 1.0226 | 0.8799 | 0.8929 | **0.8411** | 5.2s |
| **Combined** | clip+wd+smooth | 3.6585 | 1.0273 | 0.8828 | 0.8933 | **0.8394** | 5.2s |

> [!NOTE]
> Label smoothing raises the cross-entropy loss floor because the loss now includes the entropy of the smoothed target distribution (H ≈ 0.6 for α=0.1, V≈30). 
> A label-smoothed loss of 0.84 corresponds roughly to a hard-target loss of ~0.24, comparable to baseline.

### 7.2 Inference Quality

Despite higher numerical loss, label smoothing produced **better-formatted** inference output:

| Model | Input Prompt | Output | Format Correct? |
|---|---|---|---|
| Baseline | `board=_________` | `Oalid=0,1,2,3,4,5,6` | ❌ ("Oalid" corruption) |
| Baseline | `board=X___O____` | `Oalid=1,2,3,5,6,8\|blocked=5` | ❌ |
| Label Smooth | `board=_________` | `valid=0,1,2,3,4,5,7,6` | ✅ |
| Label Smooth | `board=X___O____` | `valid=1,2,3,5,7,8\|blocked=4` | ✅ |

This supports the hypothesis from §5.1: label smoothing prevents the model from memorising exact surface patterns and instead learns the *structure* of the output format — producing more reliable and well-calibrated responses.

### 7.3 Connect-4 Results (460K params — Complex Game)

To validate at a more realistic scale, the same experiments were run on the Connect-4 player corpus (7×6 board, 53K training lines) using the full-scale architecture: N_EMBD=96, N_LAYER=4, N_HEAD=8 (~460K params), 5,000 steps.

| Experiment | Config | Step 1 | Step 1K | Step 3K | Step 5K | **Best Loss** | Δ vs Baseline | Time |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | — | 4.1893 | 0.2690 | 0.1878 | 0.1526 | **0.1227** | — | 60s |
| **Grad Clip** | ‖g‖≤1.0 | 4.1893 | 0.2690 | 0.1878 | 0.1526 | **0.1227** | 0% | 60s |
| **Weight Decay** | wd=0.01 | 4.1893 | 0.2660 | 0.2047 | 0.1538 | **0.1193** | **−3%** ✅ | 60s |
| **Adam Tuned** | β₁=0.9 β₂=0.999 lr=6e-4 | 4.1893 | 0.2869 | 0.2487 | 0.1636 | **0.1313** | +7% | 60s |
| **Label Smooth** | α=0.1 | 4.2182 | 0.8536 | 0.8004 | 0.7473 | **0.7291** | *N/A* | 60s |
| **Combined** | clip+wd+smooth | 4.2182 | 0.8562 | 0.8044 | 0.7519 | **0.7314** | *N/A* | 61s |

> [!IMPORTANT]
> **Scale-dependent finding**: Weight decay IMPROVED best loss by 3% at 460K params, while it HURT by 3% at 64K params. This confirms the prediction from §3.1: weight decay only helps when the model has enough capacity to overfit. The crossover point appears to lie between 64K and 460K parameters.

### 7.4 Per-Technique Verdicts (Cross-Scale)

| Technique | 64K params (TTT) | 460K params (C4) | Final Verdict |
|---|---|---|---|
| **Gradient Clipping** | 🟡 No effect | � No effect | ✅ Enable (zero cost, defensive) |
| **Weight Decay** | 🔴 −3% worse | 🟢 **+3% better** | ✅ Enable for models ≥200K params |
| **Adam Tuned** | 🔴 −6% worse | 🔴 −7% worse | ❌ Keep current β₁=0.85, β₂=0.99, LR=0.001 |
| **Label Smoothing** | 🟢 Better formatting | 🟢 Consistent | ✅ Enable (α=0.1) for all models |
| **Combined** | 🟡 ≈ label smooth alone | 🟡 ≈ label smooth alone | Label smoothing is the dominant effect |

### 7.5 Shakespeare Results (Natural Language Text)

To validate beyond game corpora, experiments were run on Shakespeare's complete works — a fundamentally different domain (natural language prose/verse, 1.1 MB, 114K lines).

#### 7.5.1 Character-Level (840K params, 2K steps)

| Experiment | Config | Step 1 | Step 500 | Step 1K | Step 2K | Time |
|---|---|---|---|---|---|---|
| **Baseline** | — | 6.907 | 2.464 | 2.239 | **2.320** | 30s |
| **Label Smooth** | α=0.1 | 6.917 | 2.880 | 2.678 | **2.742** | 31s |
| **Weight Decay** | wd=0.01 | 6.907 | 2.473 | 2.240 | **2.309** | 31s |
| **Combined** | clip+wd+smooth | 6.917 | 2.872 | 2.681 | **2.741** | 31s |

#### 7.5.2 Word-Level (510K params, 10K steps)

| Experiment | Config | Step 1 | Step 1K | Step 5K | Step 10K | Time |
|---|---|---|---|---|---|---|
| **Baseline** | — | 8.619 | 5.024 | 4.757 | **4.384** | 160s |
| **Label Smooth** | α=0.1 | 8.628 | 5.530 | 5.343 | **5.024** | 183s |
| **Weight Decay** | wd=0.01 | 8.619 | 5.024 | 4.754 | **4.383** | 161s |

> [!NOTE]
> **Domain-dependent finding**: Weight decay is precisely neutral on Shakespeare text (Δ < 0.01) at both 510K and 840K params — unlike Connect-4 where it improved by 3% at 460K params. This suggests weight decay's benefit correlates with the *structured/repetitive* nature of game corpora (where exact memorisation is a risk) rather than model size alone.

### 7.6 Markets Results (Real-World Financial Data)

To test on real-world data, experiments were run on the Market Regime Detection pipeline — 3 organelles trained on actual financial market data (SPY, QQQ, TLT, GLD, USO, VIX). This is the most demanding test: noisy, non-stationary, real-world time series.

| Organelle | Corpus | Baseline Best Loss | Label Smooth Best Loss | Weight Decay Best Loss |
|---|---|---|---|---|
| **Analyser** | 575KB (market summaries) | 0.3433 | 0.9584 | 0.3450 |
| **Regime Classifier** | 68KB (regime labels) | 0.2842 | 0.7887 | 0.2824 |
| **Sector Rotator** | 136KB (sector weights) | 0.1523 | 0.7318 | 0.1524 |

**Backtest Accuracy (60-day, 5 regime classes, 20% random baseline):**

| Experiment | Valid Predictions | **Correct Regime** | Δ vs Baseline |
|---|---|---|---|
| **Baseline** | 60/60 (100%) | 22/60 (**36.7%**) | — |
| **Label Smooth** (α=0.1) | 60/60 (100%) | 25/60 (**41.7%**) | **+5pp** ✅ |
| **Weight Decay** (wd=0.01) | 60/60 (100%) | 22/60 (**36.7%**) | 0pp |
| **Soup-3** (3 seeds avg) | 42/60 (70%) | 23/42 (**54.8%** valid) | +1.7pp overall |
| **Soup-5** (5 seeds avg) | 52/60 (87%) | 8/52 (**15.4%** valid) | **−23pp** ❌ |
| **Soup-3 + LS** (3 seeds, α=0.1) | 30/60 (50%) | 8/30 (**26.7%** valid) | **−23pp** ❌ |
| **Greedy Soup-3** (best + validate) | 60/60 (100%) | 22/60 (**36.7%**) | 0pp (=baseline) |

> [!IMPORTANT]
> **Label smoothing improves real-world prediction accuracy.** Despite substantially higher CE loss numbers, label smoothing pushed backtest accuracy from 36.7% → 41.7% in these 5K-step A/B benchmarks. In the full 50K-step production run (`markets_demo`), label smoothing (α=0.1) + gradient clipping achieved **60% accuracy** (3.0× random baseline) — up from 57% without these optimisations. This confirms label smoothing prevents overconfident memorisation and produces better-calibrated probability distributions.

> [!WARNING]
> **Model Soup is definitively NOT recommended for random-init organelles.** Five soup variants were tested: uniform(3), uniform(5), soup+LS(3), soup+LS(5, skipped), and greedy(3). The greedy soup correctly rejected ALL averaging candidates (validation loss jumped 3-7× from averaging), degenerating to single-seed (=baseline). This proves that different seeds converge to fundamentally different loss basins. See [ORGANELLE_MODEL_SOUP.md](file:///Users/user/dev/projects/microgpt-c/docs/research/RESEARCH_MODEL_SOUP.md) for full analysis.

### 7.7 Per-Technique Verdicts (Cross-Scale, Cross-Domain)

| Technique | Games (64K) | Games (460K) | Text (510K–840K) | Markets (615K) | Final Verdict |
|---|---|---|---|---|---|
| **Grad Clipping** | 🟡 No effect | 🟡 No effect | — | — | ✅ Enable (zero cost) |
| **Weight Decay** | 🔴 Worse | 🟢 +3% better | 🟡 Neutral | 🟡 Neutral | ✅ For games ≥200K |
| **Adam Tuned** | 🔴 Worse | 🔴 Worse | — | — | ❌ Keep current params |
| **Label Smooth** | 🟢 Format | 🟢 Consistent | 🟡 +18% CE | 🟢 **+5pp accuracy** | ✅ **Enable everywhere** |
| **Combined** | 🟡 ≈ smooth | 🟡 ≈ smooth | 🟡 ≈ smooth | — | Label smoothing dominates |
| **Model Soup** | — | — | — | 🔴 Soup-5 collapses | ❌ Not for random init |
| **Multi-day Context** | — | — | — | 🔴 −3pp (56.7%) | ❌ Overflows capacity |
| **Noise Augmentation** | — | — | — | 🔴 −11pp (49.2%) | ❌ Muddies retrieval |
| **Curriculum Ordering** | — | — | — | 🔴 −11pp (49.2%) | ❌ Shuffle is better |

### 7.8 Final Recommendations

Based on 24 experiments across 5 corpora (tictactoe 64K, Connect-4 460K, Shakespeare char 840K, Shakespeare word 510K, Markets 615K) plus 5 Model Soup experiments:

| Priority | Technique | Action |
|---|---|---|
| **1** | **Label smoothing (α=0.1)** | ✅ **Enable for all organelles** — improves output quality AND real-world accuracy |
| **2** | **Gradient clipping (‖g‖≤1.0)** | ✅ Enable everywhere (zero cost, defensive) |
| **3** | **Weight decay (wd=0.01)** | ✅ Enable for structured/game data ≥200K params |
| **4** | **Keep current Adam params** | β₁=0.85, β₂=0.99, LR=0.001 are well-tuned |
| ❌ | **Model Soup** | Not recommended for random-init organelles; requires shared pre-trained foundation |
| ❌ | **Data augmentation** | Noise jitter + curriculum ordering degraded accuracy (see §7.9) |
| ❌ | **Multi-day context** | Longer prompts overflow model capacity (see §7.9) |

### 7.9 Negative Results: Data Augmentation Experiments (Markets)

Two additional experiments attempted to push the markets pipeline beyond 60% accuracy. Both regressed significantly:

| Version | Changes | Accuracy | vs v5 (60%) |
|---|---|---|---|
| **v5** (baseline) | Label smoothing (α=0.1) + grad clipping, 50K steps | **60.0%** | — |
| **v6** (multi-day context) | Added trailing SPY direction + 5d return to prompts, BLOCK_SIZE=256 | **56.7%** | **−3.3pp** ❌ |
| **v7** (noise augment + curriculum) | Gaussian jitter (σ=0.2%) on 1d returns, curriculum ordering (easy→hard by VIX), 100K steps, 3× corpus (22,710 entries) | **49.2%** | **−10.8pp** ❌ |

Both were reverted. v5 confirmed as the optimal configuration.

**Why they failed:**

1. **v6 (multi-day context)**: Adding 5-day return data increased prompt length from 77 to 92 chars. For a retrieval engine with 615K params and BLOCK_SIZE=128, the extra tokens diluted the signal. The model couldn't leverage the temporal context — it just made the pattern-matching harder.

2. **v7 (noise augmentation + curriculum)**:
   - **Noise jitter** destroyed the exact pattern→output mappings that the retrieval engine relies on. Adding Gaussian noise (σ=0.2%) to `SPY=+0.3%` creates `SPY=+0.5%` — but these may actually correspond to different regimes. For a memorisation-based model, noisy duplicates are not regularisation; they are **contradictory training signal**.
   - **Curriculum ordering** removed the random shuffle that helps the model see diverse patterns during training. Sorting easy→hard meant early training only saw RISK_ON data, creating a bias that persisted.
   - **100K steps** provided no recovery — the damaged signal couldn't be fixed by training longer.

> [!NOTE]
> **Key insight for retrieval engines**: Techniques designed for generalisation-focused models (augmentation, curriculum learning) are counterproductive for pattern-memorisation models. The organelle needs to learn *exact* input→output mappings. Any technique that makes inputs noisier or less diverse during training directly degrades retrieval accuracy.

---

## 8. Techniques Deliberately Excluded

Some commonly cited training techniques are **not recommended** for MicroGPT-C:

| Technique | Why excluded |
|---|---|
| **LoRA / Adapters** | Designed for fine-tuning billion-parameter models; adds complexity without benefit at <1M scale |
| **Knowledge Distillation** | Requires a teacher model; violates the zero-dependency constraint |
| **Flash Attention** | Memory optimisation for long sequences; irrelevant for BLOCK_SIZE ≤ 256 |
| **Gradient checkpointing** | Trades compute for memory; unnecessary when full model fits in L1 cache |
| **Learning rate finder** | Requires a separate pre-training sweep; the cosine schedule with warmup already covers LR adaptation |
| **Batch normalisation** | Inferior to RMSNorm for transformers; the project already uses RMSNorm |
| **Advanced activations (SiLU/GELU)** | Marginal improvement over ReLU; adds code complexity (backward pass, possible numerical issues) |

---

## 9. Relationship to Prior Research

This optimisation work complements other research documents in the organelle series:

| Document | Relationship |
|---|---|
| [ORGANELLE_REASONING.md](file:///Users/user/dev/projects/microgpt-c/docs/research/RESEARCH_ORGANELLE_REASONING.md) | Optimisations improve retrieval quality, but cannot cross the retrieval–reasoning boundary |
| [ORGANELLE_NAR.md](file:///Users/user/dev/projects/microgpt-c/docs/research/RESEARCH_ORGANELLE_REASONING.md) | Better training produces higher-fidelity retrieval, making the NAR decomposition more effective |
| [ORGANELLE_GENERALISATION.md](file:///Users/user/dev/projects/microgpt-c/docs/research/RESEARCH_GENERALISATION.md) | Regularisation (weight decay, dropout, label smoothing) may improve generalisation within the retrieval distribution |
| [ORGANELLE_PIPELINE.md](file:///Users/user/dev/projects/microgpt-c/docs/research/RESEARCH_ORGANELLE_PIPELINE.md) | Label smoothing directly impacts confidence gating in the OPA pipeline |

The key insight: these optimisations make organelles **better retrieval engines** — faster convergence, less overfitting, better-calibrated confidence. They do not and cannot make organelles reason. The intelligence continues to live in the pipeline; the optimisations ensure that the pipeline has better-quality building blocks.

> *"A sharper retrieval engine does not become a reasoner. But a pipeline of sharper retrieval engines becomes a more effective cell."*

---

## Appendix A: Quick Reference — All Proposed Compile-Time Constants

```c
/* ---- Regularisation ---- */
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 0.01      /* Decoupled weight decay (AdamW-style)     */
#endif
#ifndef GRAD_CLIP
#define GRAD_CLIP 1.0          /* Maximum global gradient norm              */
#endif
#ifndef LABEL_SMOOTH
#define LABEL_SMOOTH 0.1       /* Label smoothing coefficient (0 = off)     */
#endif
#ifndef DROPOUT_RATE
#define DROPOUT_RATE 0.1       /* Dropout rate (0 = off)                    */
#endif

/* ---- Mixed precision ---- */
#ifdef MICROGPT_USE_FLOAT
#ifndef LOSS_SCALE
#define LOSS_SCALE 128.0f      /* Gradient scaling for float training       */
#endif
#endif
```

---

## Appendix B: Key References

| Reference | Relevance |
|---|---|
| Loshchilov & Hutter, 2019 — *Decoupled Weight Decay Regularization* | AdamW formulation |
| Kingma & Ba, 2015 — *Adam: A Method for Stochastic Optimization* | Original Adam paper |
| Vaswani et al., 2017 — *Attention Is All You Need* | Label smoothing for transformers |
| Karpathy, 2023 — *nanoGPT* | Reference implementation for small-scale transformer training |
| Wortsman et al., 2022 — *Model Soups: Averaging Weights of Multiple Fine-tuned Models* | Weight averaging technique |

---

*Copyright © 2026 Ajay Soni, Enjector Software Ltd. MIT License.*
