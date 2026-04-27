# Build Options

## SIMD Auto-Vectorisation (ON by default)

The compiler targets the best available instruction set (`-march=native` on GCC/Clang, `/arch:AVX2` on MSVC). To disable:

```bash
cmake -DMICROGPT_SIMD=OFF ..
```

---

## INT8 Quantised Build

Weights stored as 8-bit integers with per-matrix scales:

```bash
cmake -DQUANTIZATION_INT8=ON ..
```

---

## Apple Metal GPU Acceleration (macOS only)

Offloads `lin_fwd`/`lin_bwd` matmuls to Metal compute shaders. Best for larger models (N_EMBD ≥ 512):

```bash
cmake -DMICROGPT_METAL=ON ..
```

> **Note:** For small models (N_EMBD=128), GPU dispatch overhead exceeds compute time. Multi-threaded CPU is faster.

---

## Apple Accelerate / BLAS

Uses `CBLAS_GEMV`/`CBLAS_GER` macros (auto-dispatched to `cblas_dgemv`/`cblas_sgemv` based on `scalar_t`):

```bash
cmake -DMICROGPT_BLAS=ON ..
```

> **Note:** Accelerate's internal threading conflicts with multi-threaded training. Best for single-threaded inference.

---

## Float Precision

Switch all weights, activations, and gradients from `double` to `float` (32-bit). Useful for ARM NEON throughput (4-wide vs 2-wide SIMD) and memory-constrained devices:

```bash
cmake -DMICROGPT_USE_FLOAT=ON ..
```

> **Note:** Optimizer hyperparameters (learning rate, Adam β1/β2/ε) remain `double` for numerical stability. Test tolerances auto-adjust via `SCALAR_TOL`.

---

## DeepSeek-V4 Port Stack

Four flags port architectural ideas from DeepSeek-V4 §2.3 onto MicroGPT-C's CPU-first engine. Each is independently opt-in; they compose. Default OFF — the engine is bit-identical to its pre-port behaviour without any flag set. Combined, they deliver −8.7% held-out PPL on the deep config (4-layer 138K-param) at zero parameter cost. Full measurements in `docs/research/RESEARCH_DEEPSEEK_V4_PORTING.md` and the per-port papers it indexes.

### Attention Sink (`MICROGPT_ATTN_SINK`)

Adds `exp(ATTN_SINK_LOGIT)` to the softmax denominator in every attention head, so heads can route mass to "attend to nothing":

```bash
cmake -DMICROGPT_ATTN_SINK=ON ..
cmake -DMICROGPT_ATTN_SINK=ON -DATTN_SINK_LOGIT=-1.0 ..   # explicit magnitude (default)
```

> **Standalone effect:** −3.1% PPL on deep configs. No effect on 1-layer toys. See `RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md`.

### Q/K RMSNorm Pre-Dot (`MICROGPT_QK_NORM`)

Per-head RMSNorm on Q and K immediately before the attention dot product, with a closed-form gradient through the norm:

```bash
cmake -DMICROGPT_QK_NORM=ON ..
```

> **Standalone effect:** mild regression (+1.4%) at safe LR — its real value is **stability under aggressive LR**. At LR=0.02 the un-normed baseline diverges to PPL 731; with the flag, PPL stays at 205 (3.6× recovery). Composes super-additively with attention sink. See `RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md`.

### Partial RoPE (`MICROGPT_PARTIAL_ROPE`)

Rotates the last `ROPE_DIMS` of every per-head Q and K by a position-dependent angle, restoring relative-position attention:

```bash
cmake -DMICROGPT_PARTIAL_ROPE=ON ..
cmake -DMICROGPT_PARTIAL_ROPE=ON -DROPE_DIMS=16 ..        # rotate fewer dims
cmake -DMICROGPT_PARTIAL_ROPE=ON -DROPE_BASE=10000.0 ..   # default base
```

| Macro | Default | Description |
|---|---|---|
| `ROPE_DIMS` | `min(head_dim, 32)` | Number of trailing dims rotated per head |
| `ROPE_BASE` | `10000.0` | Standard RoPE frequency base |

> **Standalone effect:** −1.6% PPL on deep configs, −0.7% even on Tiny — the only V4 port that registers at 1-layer scale. The closed-form rotation backward (rotate by −θ) reuses the same cos/sin tables as the forward, no new parameters, no Adam state. See `RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md`.

### MSA Pool Weighting (`MSA_POOL_MODE`)

Selects how `msa_pool_chunk` weights the tokens it averages into a chunk summary:

| Value | Mode | Notes |
|---|---|---|
| `0` (default) | Uniform mean | Existing pre-port behaviour |
| `1` | Linear ramp recency | Newest token weighted ~2× oldest |
| `2` | Exponential recency | Last few tokens dominate |
| `3` | **Content-aware** (softmax of cosine-to-anchor) | **Recommended** |

```bash
cmake -DMSA_POOL_MODE=3 ..   # recommended
```

> **Effect:** mode 3 gives −0.32% PPL on long-context MSA evaluation; modes 1 and 2 are within noise of mode 0. See `RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md`.

### Recommended V4 stack

For **free-text generation** models (Shakespeare, names, prose) with `N_LAYER ≥ 2` and `BLOCK_SIZE ≥ 64`, enable the active-attention triumvirate plus content-aware pool:

```bash
cmake -S . -B build \
      -DMICROGPT_PARTIAL_ROPE=ON \
      -DMICROGPT_ATTN_SINK=ON   -DATTN_SINK_LOGIT=-1.0 \
      -DMICROGPT_QK_NORM=ON \
      -DMSA_POOL_MODE=3 ..
```

Combined effect: −8.7% held-out PPL on the deep benchmark, zero new parameters, ~1% extra runtime. For demos that integrate MSA directly, also opt in to the rope-aware injection by replacing `msa_pool_chunk` → `msa_pool_chunk_rope`, `msa_expand_context` → `msa_expand_context_rope`, `msa_recency_inject` → `msa_recency_inject_rope`. See `msa_infinite_shakespeare_v4` in `CMakeLists.txt` for a worked example.

> **Caveat — do NOT enable the V4 stack for grammar-rigid generation tasks** (VM codegen, structured DSL output, anything where token positions follow a strict template). Measured ablation on `w_vm_codegen_deep` at the same depth shows every V4 flag REGRESSES codegen pass rate — including in isolation. The least-harmful single flag is `MICROGPT_ATTN_SINK` (−10pp); the full stack is −30pp.
>
> | Variant | Controls | Novel | Total | Δ vs no-V4 |
> |---|---:|---:|---:|---:|
> | Baseline | 5/5 | 3/5 | 8/10 (80%) | — |
> | Sink only | 4/5 | 3/5 | 7/10 (70%) | −10pp (least harmful) |
> | RoPE only | 4/5 | 2/5 | 6/10 (60%) | −20pp |
> | Q/K RMSNorm only | 5/5 | 1/5 | 6/10 (60%) | −20pp |
> | Full V4 stack | 4/5 | 1/5 | 5/10 (50%) | −30pp |
>
> Hypothesis: code is **absolute-position-rigid** (a function template's tokens occupy fixed slots). RoPE replaces absolute `wpe` with relative-position rotation, removing structure that codegen relies on. Q/K RMSNorm strips magnitude information that may carry token-frequency priors. The active-attention V4 ports were validated on free-text PPL benchmarks and do not generalise to grammar-rigid generation. See the targets `w_vm_codegen_deep_v4_*` in `CMakeLists.txt` for reproducing this measurement.

**Heuristic for which workloads benefit:**

| Workload type | V4 stack | Examples |
|---|---|---|
| Free-text generation | ✅ Enable | Shakespeare, names, prose, dialogue |
| Long-context generation under MSA | ✅ Enable | infinite_shakespeare, context_extender |
| **Grammar-rigid generation** | ❌ **Skip** | **VM codegen, C codegen, JSON/DSL output** |
| Game playing (organelle pipeline) | ⚠ Untested | Connect-4, Tic-Tac-Toe, etc. — measure first |
| Quantised KV (TurboQuant/RotorQuant) | ⚠ Saturates | At safe LR + ample steps, metric ceilings hide differences |

---

## Custom Architecture

Each demo uses the `DEFINES` parameter in CMakeLists.txt to set its architecture. For ad-hoc overrides:

```bash
cmake -DN_EMBD=128 -DN_HEAD=8 -DN_LAYER=4 -DBLOCK_SIZE=256 ..
```

> **Note:** All demos already have their optimal architecture sizes baked into `CMakeLists.txt` via `add_demo(... DEFINES ...)`. Manual overrides affect only the default library target.

---

## Architecture Parameters

All architecture parameters are compile-time constants (`#define` macros in `microgpt.h`):

| Parameter | Default | Override | Effect |
|-----------|---------|----------|--------|
| `N_EMBD` | 16 | `-DN_EMBD=128` | Embedding dimension |
| `N_HEAD` | 4 | `-DN_HEAD=8` | Attention heads |
| `N_LAYER` | 1 | `-DN_LAYER=4` | Transformer blocks |
| `BLOCK_SIZE` | 16 | `-DBLOCK_SIZE=256` | Maximum sequence length |
| `MLP_DIM` | 64 | `-DMLP_DIM=512` | MLP hidden dimension |
| `WARMUP_STEPS` | 100 | `-DWARMUP_STEPS=500` | LR warmup duration |
| `scalar_t` | `float` | `-DMICROGPT_USE_FLOAT=OFF` | Switch all weights/activations to `double` |

---

## Platform Accelerators

> **Zero dependencies means zero dependencies.** The core engine (`microgpt.h` + `microgpt.c`) requires nothing beyond a C99 compiler, `libc`, and `libm`. It compiles and runs on any platform — from a Raspberry Pi to a mainframe.
>
> The following are **opt-in platform accelerators** that you enable explicitly via CMake flags. They are never required:
>
> | Accelerator | Flag | Requires | When To Use |
> |------------|------|----------|-------------|
> | Metal GPU | `-DMICROGPT_METAL=ON` | macOS + Apple Silicon | Models with N_EMBD ≥ 512 |
> | BLAS | `-DMICROGPT_BLAS=ON` | Accelerate / OpenBLAS / MKL | Single-threaded inference |
>
> If you don't set these flags, you get pure C99 with compiler auto-vectorisation — which, at current model sizes, is actually the fastest option (see [Optimisation Strategies](foundation/OPTIMISATION_STRATEGIES.md)).
