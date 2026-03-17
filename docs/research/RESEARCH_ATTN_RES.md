# Block Attention Residuals in MicroGPT-C

## Overview

**Attention Residuals** (AttnRes) is a technique from [Moonshot AI](https://github.com/MoonshotAI/Attention-Residuals) that replaces the standard additive residual connection `x = x + sublayer(x)` with a **learned depth-attention mechanism** that selectively aggregates representations from all prior layers.

In standard Pre-Norm Transformers, each layer's contribution is simply added to a running sum. Over many layers, early representations (which encode the original prompt semantics) get diluted by later layers' contributions — what the paper calls **PreNorm dilution**. AttnRes addresses this by letting each layer choose *how much* to attend to prior layers rather than blindly accumulating.

| Standard Residual | Block AttnRes |
|---|---|
| `x_ℓ = x_{ℓ-1} + f_ℓ(norm(x_{ℓ-1}))` | `x_ℓ = Σ α_i · h_i` where `α = softmax(q · RMSNorm(h_i))` |
| Fixed, uniform weighting | **Learned, selective** weighting over depth |
| Information dilutes with depth | Early layers remain accessible |

**Reference:** [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals) — Block AttnRes variant for memory-efficient depth attention.

---

## MicroGPT-C Implementation

AttnRes is implemented as a **compile-time opt-in** feature, gated behind `#ifdef MICROGPT_ATTN_RES`. When disabled, the engine compiles and runs identically to before — zero overhead.

### Architecture

**Block AttnRes** groups transformer layers into blocks of `ATTN_RES_BLOCK_SIZE` layers. Each block's output hidden state is stored as a "block representation." At the end of each layer, AttnRes computes a softmax-weighted combination of all prior block representations plus the current layer's output:

```
For layer L:
  1. Compute standard attention + MLP → layer output h_L
  2. Store h_L as block rep at block boundaries
  3. x₀ = Σ α_i · h_i  (attention over all blocks + current)
     where α_i = softmax(proj · RMSNorm(h_i))
```

**Within** each layer, the standard additive residual is preserved for the attention sublayer (`x = x + Wo @ attn_out`). AttnRes replaces only the **inter-layer** residual.

### Parameters

Each layer adds one projection vector `attn_res_proj[L]` of dimension `N_EMBD`. For a 4-layer model with `N_EMBD=128`, this adds just **512 scalars** (2KB) — a negligible 0.06% parameter increase.

### Files Changed

| File | Changes |
|------|---------|
| `src/microgpt.h` | `ATTN_RES_BLOCK_SIZE`, `ATTN_RES_MAX_BLOCKS` macros, config banner |
| `src/microgpt.c` | Model struct, `count_params`, `model_create/free`, `attn_res_fwd/bwd`, `forward_backward_one`, `forward_inference`, `adam_step`, serialisation |
| `CMakeLists.txt` | `MICROGPT_ATTN_RES` option, A/B benchmark targets |

### How to Use

```bash
# Enable globally
cmake -S . -B build -DMICROGPT_ATTN_RES=ON

# Or use per-target defines in CMakeLists.txt
DEFINES ... MICROGPT_ATTN_RES=1 ATTN_RES_BLOCK_SIZE=2

# Pre-configured A/B targets
cmake --build build --target c_optim_shk_baseline c_optim_shk_attnres
cmake --build build --target w_vm_codegen_v2 w_vm_codegen_deep w_vm_codegen_attnres
```

---

## Experimental Results

### Experiment 1: Shakespeare Character-Level (N_LAYER=4)

2,000-step training on 5.2MB Shakespeare corpus, character-level tokenisation.

| Metric | Baseline | AttnRes |
|--------|:--------:|:-------:|
| Params | 840,704 | 841,728 (+0.12%) |
| Loss @ 2K steps | **2.32** | **2.73** |
| Training time | 26s | 25s |
| Throughput | 77 steps/s | 80 steps/s |
| Config banner | `AttnRes = OFF` | `AttnRes = ON (block=2)` |

**Observation:** AttnRes training is stable and convergent, but loss is 18% higher than baseline at 2K steps. The projection vectors need more training iterations to learn useful aggregation patterns. At 4 layers, PreNorm dilution is minimal, so the benefit of depth-attention is limited.

### Experiment 2: VM Codegen 3-Way A/B (N_LAYER=2 vs 4 vs 4+AttnRes)

5,000-step training on 280KB pre-tokenised VM function corpus. Word-level tokenisation (861 tokens). Binary evaluation: does generated code pass `vm_module_compile()`?

| Metric | L=2 Baseline | L=4 Deep | L=4 AttnRes |
|--------|:----:|:----:|:----:|
| Params | 411,072 | 632,256 | 633,024 |
| Final loss | 0.14 | 0.15 | 0.14 |
| Best loss | 0.110 | 0.110 | **0.108** |
| Controls (in-corpus) | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) |
| Novel (out-of-corpus) | 2/5 (40%) | 2/5 (40%) | 2/5 (40%) |
| **Total** | **6/10 (60%)** | **6/10 (60%)** | **6/10 (60%)** |

**Observation:** AttnRes achieved the best training loss (0.108) and generated more prompt-faithful novel function names (`is_leap_year`, `is_palindrome`) compared to baseline — but the pass rate was identical. The ceiling is not depth or residual quality; it's a generalisation problem (the "Recall Wall").

---

## Analysis

### When AttnRes Helps

The paper demonstrates gains with **12+ layer** models on large-scale language modelling. AttnRes solves a real problem — PreNorm dilution — but that problem only manifests at significant depth.

MicroGPT-C demos currently use **1–6 layers**. At this depth:
- PreNorm dilution hasn't accumulated enough to degrade representations
- The overhead of learning depth-attention projections isn't amortised
- Other bottlenecks (corpus size, model capacity, the Recall Wall) dominate

### When AttnRes Would Help

AttnRes becomes relevant as MicroGPT-C scales to:
- **N_LAYER ≥ 8** — enough depth for dilution to become measurable
- **BPE-32K tokenisation** — deeper models with larger embedding dimensions
- **Distillation from large teachers** — where the student needs to preserve teacher representations through depth

### Qualitative Signal

Despite identical pass rates, AttnRes showed a **qualitative improvement** in prompt encoding fidelity — novel prompts generated more semantically relevant function names. This suggests the depth-attention mechanism successfully preserves prompt information through layers, even if the model lacks the capacity to compose novel algorithms from it.

### Design Decision: Keep as Opt-In

AttnRes is maintained as a dormant feature:
- Zero cost when compiled without `-DMICROGPT_ATTN_RES`
- Research infrastructure for future deep-model experiments
- Clean `#ifdef` gating with no impact on existing demos
- A/B benchmark targets included for reproducible evaluation

---

## References

- [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals) — original paper and reference implementation
- [RESEARCH_ORGANELLE_REASONING](RESEARCH_ORGANELLE_REASONING.md) — the Recall Wall analysis
- [BUILD_OPTIONS](../BUILD_OPTIONS.md) — compile-time feature flag documentation
