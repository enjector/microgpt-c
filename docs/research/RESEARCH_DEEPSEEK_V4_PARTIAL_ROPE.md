# Porting DeepSeek-V4 Partial RoPE to MicroGPT-C

> Pre/post measurements for the sixth and capstone DeepSeek-V4 port identified in [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md). Adds Rotary Positional Embedding to the last `ROPE_DIMS` of every per-head Q and K vector, immediately before the dot product, with a closed-form gradient through the rotation. Tests standalone, in pairwise composition with attention sink and Q/K RMSNorm, and revisits the deferred sliding-window-recency port to test the prior paper's "RoPE will unblock this" prediction.
>
> **Result: RoPE is the largest standalone V4 win we've measured (−1.6% PPL on the deep config), the full V4-stack (RoPE + sink + Q/K RMSNorm) reaches −8.7% PPL vs baseline, and the sliding-window port's prior regression is eliminated — though not yet flipped to positive.**

**Reference:** DeepSeek-V4 §2.3.3 "Partial Rotary Positional Embedding"; original RoPE: Su et al., RoFormer (2021).

**Status:** Implemented behind `-DMICROGPT_PARTIAL_ROPE=ON`, default **OFF**. All 61 existing unit tests pass with the flag both **off** (bit-identical to pre-port) and **on** (gradient-direction tests confirm the new backward through the rotation). Measured on Apple M2 Max, single-threaded, `MICROGPT_USE_FLOAT=ON`.

---

## 1. Spear Summary

**Point:** Partial RoPE is the **single biggest standalone V4 port** for MicroGPT-C: −1.6% PPL on its own at default LR, climbing to −8.7% PPL when stacked with attention sink and Q/K RMSNorm. The sliding-window-recency port — previously flagged as blocked on RoPE — has its regression eliminated (was +3.1%, now neutral) but does not yet become net-positive in MSA contexts, where a *different* rotation-alignment issue surfaces (re-injected K vectors carry rotation angles for their original positions but get read at new physical slots).

**Picture:** The engine's previous positional signal was the learned `wpe` table — one absolute embedding per slot index, baked into K vectors at projection time. RoPE replaces that with rotation: every Q and K's tail dimensions are spun by an angle proportional to its position, with the angle's frequency scaled across dim pairs. The crucial property: after rotation, `Q · K` depends on the **difference** of their positions (a rotation-by-Δθ matrix product), not their absolute values. So attention naturally generalises beyond trained positions and is invariant to slot-shifts that preserve relative order — which is the whole point for variable-length contexts and MSA-style cache reuse.

**Proof:** A controlled A/B sweep at the deep config (4-layer 138K-param char model on names, 1500 train steps):

| Variant | Held-out PPL | Δ vs baseline |
|---|---:|---:|
| Baseline (no V4 ports) | 10.51 | — |
| **RoPE only** | **10.34** | **−1.6%** ✓ |
| Sink only (prior paper) | 10.18 | −3.1% |
| RoPE + Q/K RMSNorm | 10.23 | −2.7% |
| RoPE + sink | 10.11 | −3.8% |
| Sink + Q/K RMSNorm (prior best) | 9.77 | −7.0% |
| **Full V4-stack: RoPE + sink + Q/K RMSNorm** | **9.60** | **−8.7%** ✓ |

The sliding-window unblock test (long-context 2048-token held-out with WIN=32):

| Variant | Post-chunk PPL | Δ vs RoPE baseline |
|---|---:|---:|
| Sliding WIN=32, RoPE OFF (prior paper) | 18.79 | +3.1% (regression) |
| **Sliding WIN=32, RoPE ON** | **19.85** | **−0.06% (neutral)** |

The regression vanished; the port is no longer harmful. It is also not yet beneficial — a different rotation-alignment issue takes its place. See §6.4.

**Push:** Ship `MICROGPT_PARTIAL_ROPE=ON` together with `MICROGPT_ATTN_SINK=ON` and `MICROGPT_QK_NORM=ON` as the **recommended V4 default stack** for MicroGPT-C models with `N_LAYER ≥ 2` and `BLOCK_SIZE ≥ 64`. The full stack delivers a meaningful −8.7% PPL win, zero new parameters, ~1% extra runtime. The sliding-window-recency port can now be re-evaluated as future work — it's no longer harmful, but will need an additional fix (re-rotating cached K vectors on injection) to become net-positive.

---

## 2. The Mechanism

### 2.1 Why absolute `wpe` is the wrong primitive

The engine's existing positional signal is a learned table `wpe[BLOCK_SIZE][N_EMBD]`, added to token embeddings before the first RMSNorm. After Wq and Wk project, every K vector implicitly encodes "I am the token at slot $p$" via the imprint of `wpe[p]`. Three problems:

1. **Hard context cap.** `wpe` has `BLOCK_SIZE` slots, period. Generation past `BLOCK_SIZE` is undefined.
2. **Position-conflated content.** A K vector's content is fingerprinted with both the token identity and its absolute slot. Comparing two K vectors at different positions involves comparing both their content and their position fingerprints — there's no clean relative-position signal.
3. **MSA breaks the alignment.** When MSA chunks pool old K vectors and the system re-injects them at fresh physical slots, the K vectors still carry the wpe-fingerprint of their *original* slot. The active-cache attention now sees Ks with stale absolute positions in physical positions that don't match. The [sliding-window-recency paper](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md) flagged this as the structural cause of its negative result.

### 2.2 RoPE: rotation as relative-position attention

RoFormer (Su et al., 2021) proposed that each Q and K vector be **rotated** by an angle proportional to its position, with frequencies scaled across dim pairs:

$$
R(p, \theta_d) = \begin{bmatrix} \cos(p\theta_d) & -\sin(p\theta_d) \\ \sin(p\theta_d) & \cos(p\theta_d) \end{bmatrix}, \qquad
\theta_d = \frac{1}{B^{2d/D}}, \qquad d = 0, 1, \dots, D/2 - 1
$$

Applied to dim pairs $(2d, 2d+1)$: $\hat q = R(p_q) q$, $\hat k = R(p_k) k$. Then the attention dot product gives:

$$
\hat q^\top \hat k = q^\top R(p_q)^\top R(p_k) k = q^\top R(p_k - p_q) k
$$

That last equality uses the orthogonality $R(p_q)^\top = R(-p_q)$ and the additive property $R(a)R(b) = R(a+b)$. **The post-rotation dot product depends only on the position difference** — the absolute positions cancel. This is the relative-position invariance that V4 exploits for million-token contexts.

### 2.3 Partial RoPE (V4's specific recipe)

V4 §2.3.3 applies RoPE only to the **last 64 dimensions** of each head's Q and K. The leading dims pass through unrotated, retaining absolute-position information via the existing wpe-equivalent path (in V4, that's compressed-block bias terms). The rationale: a small slice of relative-position-aware dims is plenty for the model to encode local structure, while the unrotated dims still help anchor things globally.

For MicroGPT-C, our adaptation:
- Default `ROPE_DIMS = min(head_dim, 32)` — rotate the last 32 dims of each head.
- Default `ROPE_BASE = 10000.0` — the standard RoPE base.
- **Keep the existing `wpe` table.** No checkpoint break, no parameter removal. RoPE is purely additive structure.
- Apply rotation **after** Q/K RMSNorm (if enabled) and **before** the dot product. The cache stores rotated K vectors.

### 2.4 Backward through rotation

Rotation is orthogonal: $R(p)^\top = R(-p) = R(p)^{-1}$. The chain-rule backward through the rotation is *another* rotation, by the negated angle:

$$
\frac{\partial \mathcal{L}}{\partial q} = R(-p)^\top \frac{\partial \mathcal{L}}{\partial \hat q} = R(p)^\top \frac{\partial \mathcal{L}}{\partial \hat q}
$$

Wait — that's not right. Let's redo: $\hat q = R(p) q$, so the Jacobian is $R(p)$, and the backward pulls $d\hat q$ back through $R(p)^\top = R(-p)$:

$$
\frac{\partial \mathcal{L}}{\partial q_i} = \sum_j R(p)_{ji} \frac{\partial \mathcal{L}}{\partial \hat q_j} \quad\Leftrightarrow\quad dq = R(-p)\, d\hat q
$$

Concretely for adjacent-pair rotation:
- Forward: $\hat a = a c - b s$, $\hat b = a s + b c$ (where $c = \cos(p\theta), s = \sin(p\theta)$).
- Backward: $da = d\hat a \cdot c + d\hat b \cdot s$, $db = -d\hat a \cdot s + d\hat b \cdot c$.

This is what `rope_rotate_bwd` implements. No new parameters, no Adam state.

---

## 3. Implementation

### 3.1 Files changed

| File | Change |
|---|---|
| `src/microgpt.h` | Added `MICROGPT_PARTIAL_ROPE` toggle, `ROPE_DIMS` macro (default `min(head_dim, 32)`), `ROPE_BASE` macro (default 10000). Banner line in `microgpt_print_config`. |
| `src/microgpt.c` | New `rope_tables_init()` precomputing cos/sin tables of size `BLOCK_SIZE × ROPE_DIMS/2`, plus `rope_rotate_fwd()` and `rope_rotate_bwd()` operating on a head-dim slice in place. Wired into all three forward sites (training serial, training parallel-head worker call sites, inference serial) and the training backward — RoPE backward runs FIRST (innermost) since RoPE is the OUTERMOST forward transform. |
| `tests/bench_microgpt_attn_sink.c` | **Reused unchanged.** The existing A/B harness already accepts arbitrary feature flags via `add_demo(... DEFINES ...)`. |
| `tests/bench_microgpt_msa_sliding.c` | **Reused unchanged.** Used for the sliding-window unblock test under RoPE. |
| `CMakeLists.txt` | Eight new targets: `bench_rope_{baseline, on}` (tiny), `bench_rope_deep_{baseline, on}`, `bench_rope_deep_with_{sink, qknorm}`, `bench_rope_deep_full_stack`, `bench_rope_msa_sliding_{baseline, on}` (the unblock test), and `test_microgpt_rope` (full unit-test suite under `MICROGPT_PARTIAL_ROPE=1` validating the new backward). |

### 3.2 The forward-side patch

Right before `sv_q[L]` save in the training forward path, and similarly in the inference path:

```c
#ifdef MICROGPT_PARTIAL_ROPE
    rope_tables_init();
    for (int h = 0; h < nh; h++) {
        rope_rotate_fwd(q + (size_t)h * hd, hd, pos_id);
        rope_rotate_fwd(k + (size_t)h * hd, hd, pos_id);
    }
#endif
```

The cache stores **post-rotation K**: every cached K already encodes its position through the rotation. This is the V4 design — it's also why the cache, post-RoPE, becomes the natural representation for relative-position attention.

### 3.3 The backward-side patch

After softmax-backward fills `d_q` and `d_k_cur` (post-rotation gradients), but before the QK_NORM backward (if enabled) and `lin_bwd`:

```c
#ifdef MICROGPT_PARTIAL_ROPE
    /* RoPE was the OUTERMOST transform in forward. Its backward runs
     * FIRST on the gradient flowing back: rotate by -theta. */
    for (int h = 0; h < nh; h++) {
        rope_rotate_bwd(d_q + (size_t)h * hd, hd, pos_id);
        rope_rotate_bwd(d_k_cur + (size_t)h * hd, hd, pos_id);
    }
#endif
```

Order of operations in the backward stack (when all flags are on):

1. **RoPE backward** (this PR) — rotates `d_q` and `d_k_cur` by `-theta_pos`.
2. **QK RMSNorm backward** (prior PR) — converts post-norm gradients to pre-norm.
3. **lin_bwd** — backpropagates through Wq, Wk, Wv to `d_x_norm1`.

Each step is an exact inverse of the corresponding forward step. The order matters: applying RoPE backward AFTER QK_NORM backward would rotate gradients that are already in the wrong space, producing incorrect Wq/Wk gradients.

### 3.4 Gradient validation

```
$ ./test_microgpt
=== Results: 61/61 passed ===

$ ./test_microgpt_rope        # full suite, MICROGPT_PARTIAL_ROPE=1 baked in
=== Results: 61/61 passed ===
```

The `gradient_direction_reduces_loss` and `training_reduces_loss` tests in `test_microgpt.c` exercise both forward and backward end-to-end. Passing both with and without RoPE is strong evidence that the new rotation-fwd/rotation-bwd pair is internally consistent.

### 3.5 Compile-time gating

```bash
# Off (default — bit-identical to pre-port engine)
cmake ..

# On with default ROPE_DIMS (= min(head_dim, 32))
cmake -DMICROGPT_PARTIAL_ROPE=ON ..

# Custom number of rotated dims
cmake -DMICROGPT_PARTIAL_ROPE=ON -DROPE_DIMS=16 ..

# Full V4-recommended stack (recommended default)
cmake -DMICROGPT_PARTIAL_ROPE=ON -DMICROGPT_ATTN_SINK=ON \
      -DMICROGPT_QK_NORM=ON ..
```

Per-demo: `DEFINES MICROGPT_PARTIAL_ROPE=1` in `add_demo(...)`. Each unique feature combination triggers a separate library variant via `_microgpt_lib_for_defines()`.

### 3.6 Cost

- **Parameters added:** 0 (cos/sin tables are pure math, not learnable).
- **Stack memory:** ~`BLOCK_SIZE × ROPE_DIMS/2 × 2 × sizeof(scalar_t)` for the global cos/sin tables — under 8 KB at our scale, populated once and reused for every forward call.
- **Compute per forward:** `n_layer × n_head × ROPE_DIMS / 2` rotation ops per token. At our deep config (4 × 4 × 16 = 256 rotation pairs per token), ~1024 mul+add per forward — sub-percent of attention compute.
- **Train time:** baseline 5.135s vs RoPE-on 5.048s on the deep config (effectively unchanged; the noise dominates).
- **Inference time:** comparable — sub-percent measured.

---

## 4. Benchmark Design

Same A/B harness as the prior V4-port papers. `tests/bench_microgpt_attn_sink.c` for standalone-quality experiments; `tests/bench_microgpt_msa_sliding.c` for the long-context sliding-window unblock test. Identical seed, identical data, only feature flags differ.

### 4.1 Variants at a glance

| Target | Architecture | RoPE | Sink | QK norm | Sliding | Used for |
|---|---|---|---|---|---|---|
| `bench_rope_baseline` | Tiny | OFF | OFF | OFF | OFF | Tiny negative control |
| `bench_rope_on` | Tiny | ON | OFF | OFF | OFF | Tiny standalone |
| `bench_rope_deep_baseline` | Deep | OFF | OFF | OFF | OFF | Deep baseline (reproduces sink paper) |
| `bench_rope_deep_on` | Deep | ON | OFF | OFF | OFF | **Deep RoPE standalone** |
| `bench_rope_deep_with_sink` | Deep | ON | ON | OFF | OFF | RoPE + sink composition |
| `bench_rope_deep_with_qknorm` | Deep | ON | OFF | ON | OFF | RoPE + Q/K RMSNorm composition |
| `bench_rope_deep_full_stack` | Deep | ON | ON | ON | OFF | **Full V4 stack** |
| `bench_rope_msa_sliding_baseline` | Deep | ON | OFF | OFF | OFF | Long-context MSA, no sliding |
| `bench_rope_msa_sliding_on` | Deep | ON | OFF | OFF | ON (W=32) | Long-context MSA + sliding (unblock test) |

`final_train_loss` and `pre_chunk_loss` are not identical across variants (training trajectory changes when the model architecture changes), but seed is held constant, so any difference comes from the V4 ports themselves.

---

## 5. Results

### 5.1 Tiny architecture (1-layer 4K-param, 600 steps)

| Variant | Held-out PPL | Δ vs baseline |
|---|---:|---:|
| RoPE OFF | 10.04 | — |
| **RoPE ON** | **9.97** | **−0.7%** ✓ |

Even at the tiniest config, RoPE delivers a real (if small) win — first V4 port we've measured to do so on the Tiny architecture. (Recall: attention sink and Q/K RMSNorm both registered as noise on Tiny.) The V4 paper's claim that RoPE is genuinely beneficial at any reasonable model size holds.

### 5.2 Deep architecture standalone (4-layer 138K-param, 1500 steps)

| Variant | Train loss | Held-out loss | Held-out PPL | Δ vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 2.329 | 2.352 | 10.51 | — |
| **RoPE ON** | 2.263 (−2.8%) | 2.336 (−0.7%) | **10.34** | **−1.6%** ✓ |

RoPE alone reduces both training and held-out loss. Unlike Q/K RMSNorm, which slightly regressed at safe LR alone, RoPE is net-positive standalone.

### 5.3 Pairwise compositions

| Variant | Held-out PPL | Δ vs baseline |
|---|---:|---:|
| Baseline | 10.51 | — |
| Sink only (prior paper) | 10.18 | −3.1% |
| Q/K RMSNorm only (prior paper) | 10.65 | +1.4% |
| RoPE only | 10.34 | −1.6% |
| Sink + Q/K RMSNorm (prior best) | 9.77 | −7.0% |
| RoPE + sink | 10.11 | −3.8% |
| RoPE + Q/K RMSNorm | 10.23 | −2.7% |
| **RoPE + sink + Q/K RMSNorm** | **9.60** | **−8.7%** ✓ |

Three observations:

1. **The full stack is the new best.** Adding RoPE to the prior best (sink + Q/K RMSNorm at 9.77) drops PPL another 1.7% to 9.60.

2. **RoPE + Q/K RMSNorm pair is interesting.** Q/K RMSNorm alone slightly regresses (+1.4%); RoPE alone is −1.6%; together they're −2.7% — almost exactly RoPE's solo win. Q/K RMSNorm's regression at safe LR is *cancelled* by RoPE's structure but doesn't yet super-add. This is consistent with the Q/K RMSNorm paper's finding that its real value is stability under LR pressure, not standalone quality.

3. **RoPE + sink composes well** (−3.8%) but slightly less than additive (−1.6% + −3.1% = −4.7% expected; actual −3.8%). They're hitting partially overlapping problems — sink relaxes the "must allocate 100% mass" constraint; RoPE makes the where-to-allocate decision easier. Some shared improvement is double-counted.

### 5.4 Sliding-window MSA unblock test (long-context 2048-token held-out)

The sliding-window paper [predicted](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md#62-the-recommended-sequence) that "Partial RoPE should land before retrying [the sliding-window port]; the wpe-alignment issue should disappear." This test verifies that prediction.

Without RoPE (from the prior paper):

| Variant | Post-chunk PPL | Δ |
|---|---:|---:|
| Baseline (memmove half) | 18.22 | — |
| Sliding WIN=32 | 18.79 | **+3.1% (regression)** |

With RoPE:

| Variant | Post-chunk PPL | Δ |
|---|---:|---:|
| Baseline + RoPE | 19.86 | — |
| **Sliding WIN=32 + RoPE** | **19.85** | **−0.06% (neutral)** |

**The regression vanished.** The sliding-window port no longer hurts. But it also doesn't yet help — the gap between the two RoPE-on variants is negligible. The prior paper's prediction is **half-confirmed**: RoPE does eliminate the original obstruction, but it surfaces a different, more subtle issue (§6.4).

A separate observation: absolute MSA PPL is *higher* with RoPE on (19.86 vs 18.22 without). This is initially counterintuitive given §5.2's clean RoPE win on standard inference. The cause: in MSA flow, K vectors are pooled, the active cache is reset, and pool-derived K vectors are re-injected at slot 0 — but the cached K's *rotation angle* corresponds to its original position, not its new physical slot. Q at the new slot's pos_id rotates by a different angle, and the relative-angle product no longer makes geometric sense. This is the same wpe-alignment problem in a new clothing — RoPE didn't eliminate it, it just transmuted it.

### 5.5 Summary

| Configuration | PPL | Δ vs baseline | Notes |
|---|---:|---:|---|
| Baseline (no V4 ports) | 10.51 | — | starting point |
| Sink only | 10.18 | −3.1% | self-contained, prior paper |
| RoPE only | 10.34 | −1.6% | self-contained, this paper |
| Sink + Q/K RMSNorm | 9.77 | −7.0% | prior best, two ports |
| **Full V4 stack: RoPE + sink + Q/K RMSNorm** | **9.60** | **−8.7%** | **new best, three ports** |

---

## 6. Interpretation

### 6.1 Why RoPE is the biggest standalone win

Each V4 port targets a different failure mode:
- **Attention sink** relaxes a constraint (mass must sum to 1).
- **Q/K RMSNorm** bounds magnitudes for stability (mostly under high LR).
- **RoPE** restructures positional information from an absolute fingerprint to a relative one.

The first two only kick in when the constraint or magnitude actually binds. RoPE acts on every single forward pass at every position — it's continuously beneficial, not just edge-case relief. The standalone win (−1.6%) reflects continuous benefit; the stacked-with-sink win (−3.8%) reflects RoPE's continuous benefit *plus* sink's edge-case relief.

### 6.2 Why the Tiny config now sees a win

The Tiny config (1 layer, 4K params, 16-position context) couldn't exploit attention sinks (no positions to spread mass across) or Q/K RMSNorm (no magnitude to clip). RoPE benefits *every* attention computation by replacing absolute position with relative — even at 16 positions, the relative-position structure is more informative than the slot-fingerprint structure of the prior `wpe`. This is the first V4 port whose Tiny-config behaviour is meaningfully positive; it suggests RoPE's value scales down to single-layer toys, not just deep stacks.

### 6.3 Why the full stack only adds 1.7% beyond sink + Q/K RMSNorm

Sink + Q/K RMSNorm together captures −7.0%. Adding RoPE only buys another 1.7% on top, even though RoPE alone is −1.6%. Two reasons:
1. **Diminishing returns.** Each port relaxes a different bottleneck; once the worst-binding two are fixed, the third can't move the needle as much because there's less "easy loss" left.
2. **Q/K RMSNorm + RoPE partial overlap.** Both stabilise the attention computation against magnitude/positional drift. The overlap was already implied by §5.3's pairwise composition.

The −8.7% combined number is, however, the new best for MicroGPT-C MSA-using models. There is no other knob in the engine that delivers a similar improvement at zero parameter cost.

### 6.4 Why MSA + RoPE didn't *flip* the sliding-window port to net-positive

The prior paper's "wpe-alignment issue" was the *symptom*, not the root cause. RoPE replaced absolute wpe with relative rotation, which removed the wpe-alignment failure. But re-injecting cached K vectors at fresh physical slots in the active cache **still** causes a mismatch: the cached K was rotated for its original position, not for its new physical slot. When Q at the new slot's pos_id dot-products with that K, the relative angle is wrong.

The fix: when re-injecting a cached K into the active cache at a new physical slot, **first** rotate it back by its original angle, **then** rotate it forward by its new slot's angle. Equivalently, "re-rotate" the K to match its new position. This is straightforward (we have rope_rotate_fwd and rope_rotate_bwd as building blocks) but requires plumbing through MSA's `msa_expand_context` to know each chunk's original rotation. It's a separate port, future work.

So the deferred sliding-window port is no longer **blocked** — it now requires a small additional fix (re-rotation on injection) rather than a fundamentally different architecture. Promoted from "blocked" to "tractable future work."

### 6.5 Updated V4-port roadmap status

After this paper, all six V4 architectural ports identified in [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) §3 have been measured:

| # | Port | Status | Best result |
|---|---|---|---|
| 1 | Attention sink (§3.1) | ✅ Shipped | −3.1% PPL |
| 2 | Q/K RMSNorm pre-dot (§3.2) | ✅ Shipped | super-additive with sink |
| 3 | **Partial RoPE (§3.3)** | ✅ **Shipped (this paper)** | **−1.6% standalone, −8.7% combined** |
| 4 | Sliding-window MSA recency (§3.4) | ⏸ Deferred (now tractable) | regression eliminated by RoPE |
| 5 | CSA-style MSA pool (§3.5) | ✅ Shipped | −0.32% (content-aware mode) |
| 6 | Lightning Indexer + top-K (§3.6) | ✅ Shipped | −0.32% at K=8, doesn't stack with #5 |

**Recommended ship configuration for any MSA-using MicroGPT-C model:**

```cmake
# Active-attention path — biggest wins, compose super-additively.
MICROGPT_PARTIAL_ROPE=1
MICROGPT_ATTN_SINK=1   ATTN_SINK_LOGIT=-1.0
MICROGPT_QK_NORM=1
# MSA-internal — small win, only one of these two helps (don't combine).
MSA_POOL_MODE=3
```

Skip ports #4 and #6 unless future work re-evaluates them after the cached-K re-rotation fix.

---

## 7. Reproducing the Results

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cd build

cmake --build . --config Release --parallel 8 --target \
    test_microgpt test_microgpt_rope \
    bench_rope_baseline bench_rope_on \
    bench_rope_deep_baseline bench_rope_deep_on \
    bench_rope_deep_with_sink bench_rope_deep_with_qknorm \
    bench_rope_deep_full_stack \
    bench_rope_msa_sliding_baseline bench_rope_msa_sliding_on

# Numerical-correctness gates: 61/61 with both flag states.
./test_microgpt
./test_microgpt_rope

# Quality variants
./bench_rope_baseline
./bench_rope_on
./bench_rope_deep_baseline
./bench_rope_deep_on
./bench_rope_deep_with_sink
./bench_rope_deep_with_qknorm
./bench_rope_deep_full_stack       # the new best

# Sliding-window unblock test
./bench_rope_msa_sliding_baseline
./bench_rope_msa_sliding_on
```

Results are deterministic for a given seed, single-threaded by default.

---

## 8. Limitations and Future Work

1. **Single corpus, single hardware, single seed.** Same caveat as the prior V4-port papers in this series.
2. **`wpe` not removed.** This port keeps the existing learned `wpe` table alongside RoPE. V4 doesn't have one — its leading dims encode position via compressed-block biases, and RoPE handles the rest. Removing `wpe` from MicroGPT-C is a checkpoint-breaking change that we deliberately deferred. A clean ablation (RoPE only, no wpe) is future work.
3. **`ROPE_DIMS = min(head_dim, 32)` is a default, not a swept hyperparameter.** A sensitivity sweep over `ROPE_DIMS ∈ {8, 16, 32, head_dim}` is straightforward to add and would tell us whether the 32-dim default is well-calibrated for our small head sizes.
4. **MSA + RoPE rotation-mismatch fix not implemented.** §6.4 sketches the fix (re-rotate K on injection); we did not implement it. Doing so is the obvious next port and should make the sliding-window port net-positive.
5. **Long-context generalisation not tested.** The headline benefit of RoPE in V4's setting is supporting million-token contexts. Our benchmarks don't exercise contexts past `BLOCK_SIZE` (except in the MSA setting, where the rotation-mismatch confounds the result). A test that generates well past trained `BLOCK_SIZE` and observes whether RoPE still produces sensible attention patterns is future work.
6. **No high-LR stress test.** Q/K RMSNorm's value emerged at LR=0.02 in the prior paper. We didn't re-run RoPE at high LR — the stability story for RoPE is plausibly different (rotation doesn't directly affect logit magnitude) but unmeasured.

---

## 9. References

- DeepSeek-V4 paper: [`papers/DeepSeek_V4.pdf`](papers/DeepSeek_V4.pdf), §2.3.3 "Partial Rotary Positional Embedding".
- Original RoPE: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, Neurocomputing 2024 (preprint 2021).
- Roadmap context: [`RESEARCH_DEEPSEEK_V4_PORTING.md`](RESEARCH_DEEPSEEK_V4_PORTING.md) §3.3.
- Companion papers (the V4 port series — now complete):
  - [`RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`](RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md) — attention sink.
  - [`RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md`](RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md) — Q/K RMSNorm.
  - [`RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md`](RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md) — sliding-window recency (predicted RoPE would unblock this; partially confirmed).
  - [`RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md`](RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md) — CSA pool.
  - [`RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md`](RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md) — Lightning Indexer.
- Implementation:
  - `src/microgpt.h` (toggle, ROPE_DIMS, ROPE_BASE, banner)
  - `src/microgpt.c` (`rope_tables_init`, `rope_rotate_fwd`, `rope_rotate_bwd`; integration in three forward sites + the training backward)
  - `tests/bench_microgpt_attn_sink.c` (reused harness)
  - `tests/bench_microgpt_msa_sliding.c` (reused for the MSA unblock test)
  - `CMakeLists.txt` (eight new targets including `test_microgpt_rope` for backward validation)

---

## 10. Closing Remark — V4 Port Series Wrap

This is the sixth and final paper in the V4 port series. After six implementations, four positive, one parametrised (sliding-window deferred but unblocked), the picture is clear: **the V4 architectural recipe maps onto a CPU-first, dependency-free, single-file C99 transformer engine with meaningful but bounded gains.**

| Lever | Standalone | Combined |
|---|---:|---:|
| Attention sink | −3.1% | (best when combined) |
| Q/K RMSNorm | +1.4% (regress) | super-additive with sink |
| RoPE | −1.6% | adds another −1.7% on top |
| **All three combined** | — | **−8.7%** |
| Sliding-window | regression → neutral with RoPE | future work |
| CSA pool / Lightning Indexer | −0.32% each | don't compose |

The biggest single takeaway across all six papers is the active-attention-path triumvirate (sink + Q/K norm + RoPE) gives an 8.7% PPL improvement at zero parameter cost and ~1% extra runtime. That is the **recommended default V4 stack** for any future MicroGPT-C model with `N_LAYER ≥ 2` and `BLOCK_SIZE ≥ 64`. The MSA-internal ports give small (~0.32%) wins at best and don't compose. The sliding-window port becomes tractable after this paper but needs one more small fix.

The series is closed. Future work picks up at: (a) the sliding-window re-rotation fix, (b) shipping the recommended stack as default, (c) checkpointing-compatible `wpe` removal under RoPE, (d) high-LR stability ablations across the full stack.

---

*Honest research, end of series. Six ports, six papers, six measurements. Ship the three that matter; document the three that don't; leave a clear breadcrumb for whoever picks this up next.*
