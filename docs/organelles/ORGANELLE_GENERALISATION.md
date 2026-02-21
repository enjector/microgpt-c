# Organelle Generalisation Research Report

**Topic:** From Memorisation to Composition — the experimental lineage of C code generation in MicroGPT-C  
**Date:** February 2026  
**Author:** Ajay Soni, Enjector Software Ltd.  
**Status:** Results in — planner excellent, wiringgen body generation needs corpus redesign

---

## Spear Summary

**Point:** A single organelle cannot generalise. The experiments in this lineage prove that fact definitively, then construct a three-organelle pipeline (`c_planner` → `c_wiringgen` → `c_judge`) that achieves *compositional* code generation from natural language intents using coordinated retrieval — not model reasoning.

**Picture:** A single organelle is a lookup table for code. Two organelles form a lookup table with an index. Three organelles, a syntax judge, and an OpaKanban retry loop form something that *writes new code* — not because any component is creative, but because the pipeline searches the space of known compositions and validates the output deterministically.

**Proof:** `c_codegen` scored 100% on in-corpus prompts and 0/10 on novel ones. `c_compose_v3` ran 128 held-out intents: 98% parse rate, 83% exact plan match, 96% neural judge pass — all strong. But `c_wiringgen` produced syntactically valid C on only 1/117 attempts (1%). The planner is excellent; the wiring body generation is the gap. Root cause: `c_wiringgen`'s corpus teaches *plan structure* (`seq|fn1|fn2`), not *C function body syntax* — it generates plan strings instead of compilable code.

**Push:** The puzzle8_reasoning experiments provide the theoretical framework: the pipeline *is* the reasoning layer. The organelles are retrieval engines; composition, validation, and retry are handled deterministically. The question is not whether organelles can reason — they cannot — but whether a well-designed pipeline can simulate compositional code generation through coordinated retrieval.

---

## 1. The Research Question

> **Can a sub-1M parameter organelle system generate *novel* C code from natural language descriptions?**

The word "novel" is key. There are two possible answers:

1. **No, not directly.** A single organelle cannot generalise to unseen prompts — it memorises strings, not semantics. (`c_codegen` proved this.)
2. **Yes, indirectly.** A pipeline of organelles, each trained on a different decomposition of the problem, can achieve compositional generation by retrieving and assembling *known patterns* in new combinations. (`c_compose_v3` is testing this.)

This distinction — **single-model retrieval vs. pipeline-level composition** — is the organelle generalisation research question.

---

## 2. Experimental Lineage

The experiments form a directed chain, each one exposing a gap that motivates the next:

```
c_codegen             c_wiringgen          c_compose          c_compose_v3
─────────────         ───────────          ─────────          ────────────
"Can one model        "Can a model         "Can planner       "Can three organelles
retrieve C code?"     learn to compose?"   + judge work?"     produce valid C?"
     │                     │                    │                    │
     ▼                     ▼                    ▼                    ▼
100% in-corpus        Corpus: plan strings  96% parse rate     98% parse rate ✅
0/10 novel            ← WRONG target        4% registry hit    83% exact match ✅
                      Needs: C bodies        LR divergence v2   1% wiring syntax ❌
                      ← Next fix                                (corpus redesign
                                                                 needed)
```

---

## 3. Phase 1 — c_codegen: Establishing the Retrieval Baseline

### What It Proved

`c_codegen` trained an 875K-parameter character-level Transformer on **2,081 C function bodies** (492 KB corpus). The training corpus spans five numerical domains: FFT, linear algebra, statistics, signal processing, and technical analysis.

| Metric | Result |
|--------|--------|
| Corpus recall (50K steps, loss 0.034) | **100% byte-perfect** (7/7 tested) |
| Novel prompts (0/10) | **0/10 — all garbled** |
| Confidence gate | 80% threshold cleanly separates known/unknown |
| Params/byte ratio | 1.78:1 |

### The Paraphrase Test

The definitive proof of memorisation:

```
/* sort values in ascending order */   →  bubble_sort() — PERFECT (100% confidence)
/* ascending sort */                   →  token soup    (35% confidence)
```

Same semantic concept, different string — total failure. The model learned `string → string` mappings, not `concept → implementation` mappings.

### Architecture Lesson: Comment→Code Document Structure

The breakthrough was treating `/* comment */\n` + function body as a **single multi-line training document**. Line-by-line splitting failed because the model never learned the association between the prompt and the body. This structural insight carries through every subsequent experiment.

### Why Scaling Did Not Help

Scaling from 142K to 875K parameters (6×) improved corpus recall from partial → byte-perfect, but novel prompt accuracy remained **0/10 throughout**. More parameters built a bigger lookup table, not a smarter programmer. This established the **retrieval boundary**: no amount of scaling within the sub-1M range will produce generalisation from a corpus of raw implementations.

### What `c_codegen` Left Unresolved

The confidence score reliably distinguishes known from unknown prompts (~80% threshold). The **missing piece** was a front-end that could map novel intent descriptions to the *known* prompts the model could handle. This is the motivation for `c_wiringgen`.

---

## 4. Phase 2 — c_wiringgen: Testing Composition Grammar

### The Core Hypothesis

`c_codegen` failed on novel prompts because the training corpus contains *implementations* — raw loop bodies, arithmetic — and those do not generalise. **What if the corpus contained *compositions* instead?**

A composition pattern (`/* smooth then differentiate */` → `void smooth_diff(...) { rolling_mean(...); diff_central(...); }`) teaches the model the *grammar* of chaining — "A then B", "if X then A else B", "compute X, transform Y by X" — rather than the *content* of each function. The hypothesis is that grammar generalises better than content.

### Architecture and Corpus

`c_wiringgen` uses an **identical architecture** to `c_codegen` (875K / 868K params, N_EMBD=128, N_LAYER=4, N_HEAD=4, BLOCK_SIZE=512), but is trained on **864 function compositions** (171 KB) referencing ~100 primitives from the `c_codegen` vocabulary.

| Corpus Category | Base Count | Example |
|---|---|---|
| Array transforms | ~25 | `sigmoid_array` — applies `sigmoid()` elementwise |
| Two-pass (stat → transform) | ~20 | `normalize_z` — `mean()` → subtract → `stddev()` → divide |
| Pipeline / chaining | ~30 | `filter_downsample` — `lowpass()` → `downsample()` |
| Aggregation | ~15 | `variance_ratio` — `variance(a) / variance(b)` |
| Windowed operations | ~20 | `rolling_zscore` — `rolling_mean()` + `running_stddev()` per element |
| Multi-step workflows | ~25 | `detrend_fft` — `rolling_mean()` → subtract → `fft_radix2()` |

**Key design principle:** The corpus is deliberately *horizontal* (domain-agnostic). Every composition calls known primitives. The model learns to wire, not to implement.

### The Over-Parameterisation Problem

The params/byte ratio is **5.08:1** — nearly 3× higher than `c_codegen`'s 1.78:1. The model can trivially memorise the corpus. The research question then becomes: does the memorised *grammar* (function chaining patterns) generalise better than memorised *content* (loop bodies)?

### Key Test Case

The critical novel prompt is `/* denoise and downsample */`. The corpus contains `/* chain lowpass filter then downsample */`. If the model has learned the composition grammar, it should recognise "denoise" as semantically equivalent to "lowpass filter" and produce the correct chain — even though the exact words differ. `c_codegen` would produce garbled output in this case. `c_wiringgen` may not.

### Status (February 2026)

`c_wiringgen` standalone: 20K steps, **best loss 0.054** (2896s, M2 chip). Embedded in `c_compose_v3`: 20K steps, best loss 0.069.

**Finding:** The corpus teaches plan string retrieval (`seq|fn1|fn2`) — not C function body syntax. When asked to generate C code, the model produces plan-format text, which `gcc -fsyntax-only` rejects. The composition grammar hypothesis is untested at the body level because the training signal never included valid C function bodies as targets. **This is the primary gap to fix.**

---

## 5. Phase 3 — c_compose: Two-Organelle Pipeline (v1 / v2)

### What It Added

`c_compose` introduced a **two-organelle pipeline** on top of `c_wiringgen`:

```
/* intent comment */  →  c_planner  →  "seq|fn1|fn2"
                      →  c_judge    →  PASS / FAIL
```

- **c_planner**: Maps `/* smooth then differentiate */` → `seq|rolling_mean|diff_central`  
- **c_judge**: Validates the flat-string plan against a 523-function registry  
- **Flat-string wire format**: `seq|fn1|fn2` — the same pipe-delimited OPA protocol used throughout the project  
- **Constrained decoding**: Edit-distance nearest-name filter to fix near-miss function names

### v1 Results (462K params, 25K steps, 512 training intents)

| Metric | Result |
|--------|--------|
| Plan parse rate | **96%** (123/128) |
| All functions in registry | 4% (5/128) |
| Judge PASS | 65% (83/128) |
| Exact match | 2% (2/128) |
| Planner training time | 276s |
| Judge training time | 156s |

**96% parse rate** — the flat-string format is trivially learnable at 462K params. The bottleneck is function *naming*, not format: the model generates plausible-but-near-miss names (`difff_cententpe` instead of `diff_central`). Only 4% of plans had all function names present in the 523-name registry.

### v2 Results (1.2M params, 50K steps, 1452 training intents)

| Metric | v1 | v2 |
|--------|----|----|
| Plan parse rate | 96% | **20%** |
| Judge PASS | 65% | **2%** |

> [!CAUTION]
> **v2 catastrophically regressed.** Scaling from 462K → 1.2M params at the same `LEARNING_RATE=0.001` caused divergence after step ~7K. The model produced garbage from that point onward. The `best_checkpoint` mechanism preserved the step-7K weights, but those were severely underfit.

### The LR Divergence Lesson

Larger models are more sensitive to learning rate. `lr=0.001` was appropriate for 462K params but destructive at 1.2M. The fix — `lr=0.0005` with `WARMUP_STEPS=2500` followed by cosine decay — is now a standard configuration in `c_compose_v3`.

### What v2 Left Unresolved

Even v1's 4% registry hit left the pipeline short of producing usable code. The plan `seq|rolling_mean|diff_central` is correct but incomplete — it tells *what* to call, but not *how* to wire the calls together into a compiling C function body. That is `c_wiringgen`'s role, which v1/v2 left disconnected from the pipeline.

---

## 6. Phase 4 — puzzle8_reasoning: The Theoretical Framework

While the C code generation lineage was underway, the **puzzle8_reasoning** experiments independently validated the core theoretical thesis that governs `c_compose_v3`.

### Why This Is Structurally Relevant

The 8-puzzle experiments prove, quantitatively, the principles that `c_compose_v3` relies on:

| Puzzle8 Insight | c_compose_v3 Application |
|---|---|
| **Representation beats capacity** (0% → 60% by encoding change alone) | Flat-string wire format compresses planner output; `gcc -fsyntax-only` compresses the validation signal |
| **Scaffolding is a capacity bridge** (64K: 20% assisted → 3% bare; 460K: 90% assisted = 90% bare) | The OpaKanban retry loop is the scaffold — it compensates for planner/wiringgen misfires |
| **5-organelle pipeline achieves 90% solve rate** on unseen puzzles | 3-organelle pipeline tests whether the same coordination achieves novel code generation |
| **Reasoning traces safe to augment** (no regression at 13% enrichment) | Reasoning trace infrastructure is available for `c_compose_v3` organelles |

### Key Structural Lesson for c_compose_v3

The puzzle8 experiments showed that 460K-parameter organelles, with the right encoding, can handle prompts that 64K organelles cannot. The `c_compose_v3` planner/judge use **128-dim, 6-layer, 8-head** Transformers (~1.2M params) with a fixed `LEARNING_RATE=0.0005` and `WARMUP_STEPS=2500` — lessons learned directly from puzzle8's capacity scaling experiments.

---

## 7. Phase 5 — c_compose_v3: Three-Organelle Pipeline with Syntax Gate

### The Key Advance

`c_compose_v3` closes the gap left by `c_compose` v1/v2 by:

1. **Adding `c_wiringgen` as a third organelle** — after the planner produces a flat-string plan, the wiring organelle converts it into an actual C function body
2. **Adding a deterministic C syntax gate** — `gcc -fsyntax-only` validates the generated C before the neural judge sees it
3. **Fixing the LR/capacity mismatch** — all organelles use `lr=0.0005` with warmup
4. **OpaKanban retry loop** — up to 3 retries per intent, with the plan as context for re-generation

### Full Pipeline

```
Natural language intent
        │
        ▼
┌───────────────────┐
│   c_planner       │  Maps /* intent */ → "seq|fn1|fn2"
│   N_EMBD=128      │  Temp=0.2, Ensemble K=3
│   N_LAYER=6       │
│   N_HEAD=8        │
└────────┬──────────┘
         │  "seq|normalize_z|rolling_mean"
         ▼
┌────────────────────────────────────┐
│  plan_to_wiring_prompt()           │  Converts flat plan to C comment prompt
│  /* pipeline: normalize_z then    │
│     rolling_mean */                │
└────────┬───────────────────────────┘
         │
         ▼
┌───────────────────┐
│  c_wiringgen      │  Generates C function body
│  (same arch as    │  Temp=0.3, GEN_LEN=400
│  planner/judge)   │
└────────┬──────────┘
         │  C function candidate
         ▼
┌───────────────────┐
│ gcc -fsyntax-only │  Deterministic syntax gate
│ (C Syntax Judge)  │  ~5ms, 0% false positives
└────────┬──────────┘
         │  PASS / FAIL
         ▼
┌───────────────────┐
│   c_judge         │  Neural semantic validation
│   (same arch)     │  Temp=0.1
└────────┬──────────┘
         │  PASS / FAIL
         ▼
┌───────────────────┐
│  OpaKanban retry  │  Up to MAX_RETRIES=3
│                   │  Blocks failed plans
└───────────────────┘
```

### Architecture Configuration

| Organelle | Params | Training | Best Loss | Result |
|---|---|---|---|---|
| `c_planner` | ~1.2M | 50K steps | 0.085 | ✅ 98% parse, 83% exact |
| `c_judge` | ~1.2M | 50K steps | 0.132 | ✅ 96% PASS on valid plans |
| `c_wiringgen` | ~1.2M | 20K steps | 0.069 | ❌ 1% syntax-valid C (corpus mismatch) |

All three organelles share `N_EMBD=128, N_HEAD=8, N_LAYER=6, BLOCK_SIZE=128, MLP_DIM=512` — a deliberate architectural unification required because `checkpoint_load` and `checkpoint_save` use compile-time macros, not runtime config values.

### The N_LAYER Alignment Problem (Solved)

> [!NOTE]
> A subtle architectural constraint of the MicroGPT-C engine: `checkpoint_load`, `model_create`, and `train_worker_run` all use compile-time `N_LAYER`, `N_HEAD`, `BLOCK_SIZE`, and `MLP_DIM` macros, rather than the runtime `MicrogptConfig` struct values. This means a single binary can only support *one* architecture across all organelles — different architectures require separate binaries or a runtime-switchable engine.
>
> The segfault fix was: set `WIRING_N_LAYER=6, WIRING_N_HEAD=8` (matching the planner/judge compile-time constants), accepting that `c_wiringgen` would be over-parameterised for its simpler corpus. This is a current limitation of the codebase that deserves a future fix (parameterise all macros through cfg struct).

### The Deterministic Syntax Gate

The `gcc -fsyntax-only` check is the most important structural contribution of `c_compose_v3`. It provides:

- **0% false positives** — gcc either accepts or rejects; there is no ambiguity
- **~5ms per check** — no object file is produced, just syntax parsing
- **Filter before neural judge** — the neural judge only sees syntactically valid code, which concentrates its capacity on semantic validation
- **Hard signal for the retry loop** — syntax failures trigger immediate OpaKanban retry

This is a direct application of the **OPA Principle**: place deterministic correctness checks *before* neural validators, so the neural models operate only on a pre-filtered input distribution.

### Test Configuration

- **128 held-out intents** from `test_intents.txt` (same split as `c_compose` v1/v2)
- **3 retries** per intent via OpaKanban
- **Evaluation metrics:** plan parse rate, C syntax pass rate, neural judge pass rate, overall success rate per retry tier
- **Baseline comparison:** `c_compose` v1 (96% parse rate, 4% registry hit, 65% judge pass)

---

## 8. Theoretical Context: Why Three Organelles?

### The Retrieval–Composition Boundary

From `ORGANELLE_REASONING.md`:

> *A single organelle maps descriptions to memorised patterns with byte-level fidelity — but cannot compose new code.*

The two-organelle (`c_planner` + `c_judge`) design of `c_compose` v1/v2 proved that the *format* is learnable (96% parse rate) but the *vocabulary* is not reliable enough (4% registry hit). The problem was not the pipeline architecture but the **absence of a wiring step**: the planner produced a plan, but nothing converted that plan into executable code.

`c_wiringgen` fills exactly this gap. The three-organelle design maps to the classic OPA Planner → Worker → Judge separation:

| OPA Role | c_compose_v3 Organelle | What It Learns |
|---|---|---|
| **Planner** | `c_planner` | Maps intent description → flat-string composition plan |
| **Worker** | `c_wiringgen` | Maps composition plan → C function body via known primitives |
| **Judge** | `gcc` (deterministic) + `c_judge` (neural) | Validates syntax and semantics |

### Coordinated Retrieval vs. Single-Model Generalisation

The key insight — proven by `c_codegen` and being tested by `c_compose_v3` — is:

```
Single organelle:  Novel intent → ??? → Garbled output
                   (retrieval fails — no matching string pattern)

OPA pipeline:      Novel intent → Decompose into known plan → Convert to known wiring 
                   → Validate → Retry if needed
                   (coordinated retrieval may succeed)
```

The pipeline does not make any organelle smarter. It creates a search problem over the space of known patterns, guided by a deterministic progress signal (syntax pass/fail), and managed by a deterministic retry coordinator (OpaKanban). The intelligence is in the coordination, not the components.

### The Composition Grammar Hypothesis

`c_wiringgen`'s corpus teaches 174 base compositions (expanded to 864 with variations), all of which follow a small number of structural patterns: A→B chaining, two-pass (stat→transform), conditional routing, aggregation. The hypothesis is that these patterns — being more abstract than raw implementations — generalise better to novel intents.

A critical test: `/* denoise and downsample */`. The corpus contains `/* chain lowpass filter then downsample */`. If `c_wiringgen` has learned the pattern ("filter-type operation followed by downsample") rather than the string ("chain lowpass filter then downsample"), it will produce a valid chaining body even for the novel description. This is equivalent to the 8-puzzle's MD-delta insight: **make the learnable pattern explicit**, and even a small model can retrieve it reliably.

---

## 9. Results (February 2026)

### Actual vs Target

| Tier | Target | Actual | Status |
|---|---|---|---|
| **Tier 1** — parse rate | >90% | **98%** (126/128) | ✅ Exceeded |
| **Tier 2** — registry hit | >80% | **91%** (117/128) | ✅ Exceeded |
| **Tier 3** — neural judge PASS | >50% | **96%** (123/128) | ✅ Exceeded |
| **Tier 4** — exact plan match | >80% | **83%** (106/128) | ✅ Met |
| **Tier 5** — wiring syntax OK | >30% | **1%** (1/117) | ❌ Failed |

### c_compose v1 vs c_compose_v3 Comparison

| Metric | c_compose v1 | c_compose_v3 | Change |
|---|---|---|---|
| Plan parse rate | 96% | **98%** | +2% ✅ |
| All fns in registry | 4% | **91%** | +87% ✅ |
| Neural judge PASS | 65% | **96%** | +31% ✅ |
| Exact plan match | 2% | **83%** | +81% ✅ |
| Valid C function body | N/A | **1%** | New (❌ needs fix) |

### What the Results Mean

The **planner pipeline** is now high-performing across every metric. The v2 LR-divergence regression is fully reversed. The 83% exact match at 1.2M params vs. 2% at 462K params confirms that the architectural scaling (with correct LR) was the right move.

The **wiring bottleneck** is a corpus design problem, not a model capacity or architecture problem. `c_wiringgen` learned to retrieve plan strings (`seq|fn1|fn2`) because that is what its training corpus contains. It was never shown a C function body as a target output. The gcc syntax gate is working exactly as intended — it is correctly rejecting non-C output. The fix is: redesign the `c_wiring.txt` corpus so each document is a `/* plan comment */` → `void fn(...) { ... }` pair, not a plan string.

---

## 10. Open Research Questions

### 10.1 Does corpus-of-compositions generalise better than corpus-of-implementations?

**Partially answered.** The `c_wiringgen` corpus was plan strings, not C bodies — so plan retrieval generalises well (83% exact match via `c_planner`), but we have not yet tested whether *C body compositions* generalise better than raw implementations. The next corpus redesign will test this directly.

### 10.2 Is the `gcc -fsyntax-only` gate the right architecture?

**Confirmed: yes.** The gate correctly rejected 99% of `c_wiringgen`'s output. It caught the corpus mismatch immediately — without it, non-C output would have flooded the neural judge and produced meaningless PASS/FAIL signals. The 0% false-positive guarantee was essential.

### 10.3 How much does the OpaKanban retry loop contribute?

**Cannot measure yet** — 466 syntax failures with 350 retries, but no retry succeeded (because all attempts generated plan strings). Once the corpus is fixed, retry contribution can be measured. Architecture is correct; the input to retry was simply always invalid.

### 10.4 What is the correct corpus structure for `c_wiringgen`?

**New primary question.** The corpus must contain `/* comment */` → valid C function body pairs where the body calls primitives from `c_codegen`'s vocabulary. Format:
```c
/* normalize then compute rolling mean */
void normalize_then_mean(float *out, const float *in, int n, int w) {
    zscore_normalize(out, in, n);
    rolling_mean(out, out, n, w);
}
```
Each document must be a compilable C function — not a plan string. The corpus generator needs updating.

### 10.5 The Architecture Constraint Problem

The compile-time macro constraint (all organelles must share `N_LAYER`, `N_HEAD`, `BLOCK_SIZE`) means `c_wiringgen` is over-parameterised (1.2M params for a 171KB corpus). Longer term, parameterising through `MicrogptConfig` at runtime would allow right-sizing per organelle within a single binary.

---

## 11. Connection to the Reasoning Report

`ORGANELLE_REASONING.md` frames the entire organelle programme around the **retrieval–reasoning boundary**: organelles retrieve, they do not reason; the pipeline provides what looks like reasoning through coordinated search.

`c_compose_v3` is the code generation instantiation of this thesis:

| ORGANELLE_REASONING Concept | c_compose_v3 Instantiation | Status |
|---|---|---|
| Single organelle is a neural grep | `c_codegen` — 0/10 novel, 100% corpus recall | ✅ Confirmed |
| Composition grammar generalises better | `c_wiringgen` plan retrieval — 83% exact | ✅ Plans yes; C bodies untested |
| OPA provides compositional novelty | Three-organelle pipeline with OpaKanban retry | 🔄 Unblocked once corpus fixed |
| Deterministic Judge gates the neural loop | `gcc -fsyntax-only` — confirmed 99% rejection accurate | ✅ Confirmed |
| Retrieval + coordination ≈ reasoning | Planner metrics excellent; wiring is the gap | 🔄 Next iteration |

The **gradient descent without calculus** framing from §8 of the reasoning report applies directly:

- **Loss function**: syntax pass / judge pass / retry count
- **Proposal distribution**: `c_wiringgen` at temperature 0.3
- **Accept/reject**: `gcc` (hard) + `c_judge` (soft)
- **Momentum**: OpaKanban blocked plan list (prevents repeating failed compositions)
- **Convergence**: syntactically valid, semantically coherent C function body

The pipeline optimises over the space of possible wiring patterns through rejection sampling. The organelles provide the proposals; the pipeline provides the search.

---

## 12. Next Steps

| Priority | Action | Expected Outcome |
|---|---|---|
| **P0** | Redesign `c_wiring.txt` corpus: each doc = `/* comment */` + compilable C function body calling known primitives | Fix the root cause of 1% syntax pass |
| **P0** | Retrain `c_wiringgen` on C-body corpus (start with 20K steps) | Establish whether composition grammar generalises in C-body form |
| **P0** | Re-run `c_compose_v3` inference on same 128 intents | Measure wiring syntax pass rate with fixed corpus |
| **P1** | Analyse gcc failure modes from current run — what class of error dominates? | Confirm plan-string-as-C is the root cause (not tokenisation) |
| **P1** | Test standalone `c_wiringgen` (C-body corpus) on 10 novel prompts vs `c_codegen` | Test composition grammar generalisation hypothesis directly |
| **P2** | Measure OpaKanban retry contribution once wiring produces valid C | Validate retry architecture for code generation |
| **P2** | Explore S-expression wire format between planner and wiring | Preserve structural nesting in the composition plan |
| **P3** | Parameterise N_LAYER/N_HEAD/BLOCK_SIZE through MicrogptConfig at runtime | Enable right-sized per-organelle architectures in one binary |
| **P3** | OpaTrace capture for `c_compose_v3` pipeline runs | Reasoning trace training data for future fine-tuning |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
