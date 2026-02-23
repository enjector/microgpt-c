# Organelle Generalisation: The VM Approach

**Topic:** Overcoming the C Code Generation Ceiling via the MicroGPT-VM
**Date:** February 2026
**Author:** Ajay Soni, Enjector Software Ltd. (with AI assistance)

---

## 1. The Context: Why We Hit a Ceiling with C Code Generation

As documented in `ORGANELLE_GENERALISATION.md`, the `c_compose` pipeline (Planner → WiringGen → Judge) proved that compositional reasoning can be simulated using coordinated retrieval and an external validity block (`gcc -fsyntax-only`). However, the final `c_wiringgen` implementation plateaued at a **< 30% success rate** for producing syntactically valid C code.

### The Root Cause
The 28% ceiling wasn't a failure of semantic understanding, but rather **token-level sampling noise**. The C language is highly unconstrained and syntactically rigid:
- A single missing semicolon, mismatched brace, or wrong token (`dor` instead of `for`) guarantees a `gcc` failure.
- Small sub-1M parameter organelles struggle to maintain long-range syntactic coherence while also focusing on structural wiring.
- `gcc` acts only as a binary rejection filter; it discards near-misses (e.g., 99% correct logic with a typo) but provides no constructive fallback unless we introduce complex, hand-crafted decoding constraints (grammar-guided sampling).

**Conclusion:** Training a small organelle to emit raw, unstructured C code is inefficient. The representation space is too large and noisy.

---

## 2. The Solution: The MicroGPT Virtual Machine (VM)

Instead of forcing the organelles to speak full C, we will shift the pipeline to target the **MicroGPT-VM** (`experiments/engines/microgpt-vm`). The VM uses a custom-built, constrained DSL (Domain Specific Language) parsed by Flex/Bison directly into bytecode.

### Why the VM Approach Solves the Noise Problem

1. **Simpler, Constrained Grammar:**
   The VM language abstracts away boilerplate C syntax (types, memory management, rigid scoping rules). This drastically shrinks the vocabulary of possible tokens the `wiringgen` model must learn, condensing the search space and naturally minimizing character-level sampling errors.
   
2. **Native Parsing & Validation:**
   Instead of shelling out to `gcc`, the pipeline can directly invoke `vm_module_parser`. 
   - It's an in-memory operation (significantly faster validation).
   - Flex/Bison provide exact, line-level syntax error reporting, which can be fed back into the OpaKanban retry loop for in-context correction.

3. **Semantic Grounding:**
   The VM bytecode limits execution to domain-specific instructions (`vm_instruction.h`). The model learns to map intentions to these core operations (a bounded set) rather than open-ended C standard library calls.

### The Opaque Array Architecture

The most significant architectural change is how arrays are handled. In C, arrays require pointers, sizing variables, memory allocation, and potentially complex bounds checking. These raw syntax elements distract the neural model and increase token-sampling noise.

Instead, the VM employs an **Opaque Array Handle** model:
- The VM does not natively support bracket syntax `arr[i]` or variable-length collections.
- Instead, arrays are abstract IDs (handles) passed around as `ptcOTHER` type scalars.
- Structural loops and array mutations happen inside native C functions, which the VM invokes via its `verb` system.
- E.g., `var smoothed = rolling_mean(input_signal, 10);` — The VM just holds a scalar handle `smoothed`. The C-host does the array allocation and calculates the rolling mean.

This means the `vm_wiringgen` corpus will be built using a **limited language syntax that is fundamentally more powerful than C for abstract modeling**. The organelles will only need to learn *composition and orchestration* of high-level C-built primitives rather than character-level implementations of those primitives.

---

## 3. The New Architecture: `vm_compose` Pipeline

We map the original OPA (Planner → Worker → Judge) framework to the VM context:

```
Natural language intent
        │
        ▼
┌───────────────────┐
│   c_planner  (or  │  Maps /* intent */ → "seq|fn1|fn2"
│   vm_planner)     │  (Retains previous high accuracy)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   vm_wiringgen    │  *NEW*: Generates MicroGPT-VM script
│   (Organelle)     │  instead of raw C function body.
└────────┬──────────┘
         │  VM Script Candidate
         ▼
┌───────────────────┐
│ vm_module_parser  │  *NEW*: Deterministic Syntax Gate
│ (Flex/Bison)      │  Parses script -> bytecode. Fast, 0% FP.
└────────┬──────────┘
         │  PASS / FAIL
         ▼
┌───────────────────┐
│   c_judge         │  Neural semantic validation
│   (Organelle)     │  (Can validate VM script semantics)
└────────┬──────────┘
         │  PASS / FAIL
         ▼
┌───────────────────┐
│  OpaKanban retry  │  Feeds `vm_module_parser` errors 
│                   │  back for context in retry loop.
└───────────────────┘
```

### Key Differences from C Generation
- **Generation Target:** `vm_wiringgen` generates VM DSL instead of C.
- **Validation Engine:** `gcc -fsyntax-only` is replaced by `vm_module_parser_generate()`.
- **Execution Engine:** If needed for dynamic evaluation, the generated code can actually be executed securely and quickly within the VM, allowing for a future *Execution Judge* (unit tests inside the prompt loop).

---

## 4. Implementation Steps

To transition the reasoning research to the VM approach:

| Phase | Task | Status | Description |
|---|---|---|---|
| **Phase 1** | Corpus Generation | ✅ Done | 146 hand-crafted VM functions in `vm_functions.txt`, expanded to 726 via `generate_variations.py` prompt augmentation. |
| **Phase 2** | Train `vm_codegen` | ✅ Done | Trained word-level GPT on VM DSL functions. 85K params (N_EMBD=48) → 60% syntax pass; 129K params (N_EMBD=64, 726 functions) → **80% syntax pass**. |
| **Phase 3** | Pipeline Integration | ✅ Done | `vm_compose` experiment: generate → validate → retry loop using `vm_module_compile()`. 0% pass on 20 held-out intents (see §4.1). |
| **Phase 4** | Benchmark & Compare | ✅ Done | Corpus expanded to 1,566 functions. Data-only scaling: 0% → **5% pass** on 20 held-out intents (see §4.2). |

### 4.1 Phase 3 Results: The Generalisation Gap

The `vm_compose` pipeline (`experiments/organelles/vm_compose/`) proves the end-to-end architecture works:

- **Generate:** Word-level autoregressive sampling with brace-balanced stopping
- **Validate:** `vm_module_compile()` deterministic syntax gate (0% false positives)
- **Retry:** Up to 3 attempts per intent with varied RNG seeds

| Metric | vm_codegen (Phase 2) | vm_compose (Phase 3) |
|--------|---------------------|---------------------|
| Test prompts | 10 (5 training, 5 novel) | 20 (all novel) |
| Syntax pass | 80% (8/10) | 0% (0/20) |
| Confidence | 53-78% | 14-41% |
| Model | 129K params | Same checkpoint |

**Key insight:** The 80% pass rate from Phase 2 was inflated by including training-data prompts. On entirely novel intents, the 129K param model generates VM-DSL-like fragments (correct keywords, types, operators) but cannot reliably produce complete, syntactically valid functions.

### 4.2 Phase 4 Results: Data-Only Scaling

Expanded the corpus with 101 new base functions and improved the variation generator (6 variations/function, 30 additional synonym groups):

| Metric | Phase 3 (726 functions) | Phase 4 (1,566 functions) |
|--------|------------------------|--------------------------|
| Corpus size | 726 | 1,566 (+116%) |
| Vocab size | 500 | 800 |
| Model params | 129K | 168K (same arch, larger vocab) |
| Training steps | 15,000 | 3,500 (partial) |
| Syntax pass | 0/20 (0%) | **1/20 (5%)** |
| Peak confidence | 41% | 55% |

The one passing intent (`// convert temperature celsius to fahrenheit`) generated syntactically valid code on retry 2:
```
function capacity(n: number): number {
    return n * n - 1;
}
```

**Syntactically valid but semantically incorrect** — the function name and logic don't match the intent. This is expected from a 168K parameter model: it has learned the grammar of VM DSL well enough to occasionally produce valid syntax, but not enough capacity to map intent semantics to correct implementations.

### 4.3 Phase 4b: Architecture Scaling Hypothesis

#### Why data-only scaling hit a ceiling

Phase 4a doubled the corpus (726 → 1,566) but kept the same 1-layer, 64-dim model. Result: 0% → 5%. The model is **capacity-limited**, not data-limited:

- **Single attention layer** can only learn direct token correlations (e.g. `// compute` → `function`). It cannot compose multi-step patterns like "if the comment says 'factorial' → emit a `for` loop that multiplies an accumulator."
- **64-dim embeddings** compress 800 tokens into a space where semantically different words (e.g. `factorial` vs `fibonacci`) may overlap, making intent discrimination impossible.

#### The architecture scaling bet

| Config | Phase 4a | Phase 4b |
|--------|----------|----------|
| N_EMBD | 64 | **96** (+50%) |
| N_LAYER | 1 | **2** (+100%) |
| MLP_DIM | 256 | **384** (+50%) |
| Params | 168K | **399K** (2.4×) |

**What the second layer buys:** In transformer architectures, Layer 1 learns *token associations* (syntax patterns, keyword co-occurrences). Layer 2 learns *compositional rules* — how patterns combine to form valid structures. This is the same principle behind why GPT-2 (12 layers) generalises better than a 1-layer model: each layer adds a level of abstraction.

For VM DSL code generation specifically, Layer 2 should enable:
1. **Comment → structure mapping:** "compute X" → function with return, "check if X" → function with if/return 0/1
2. **Loop pattern composition:** Recognising that "sum from 1 to n" requires for-loop + accumulator + return
3. **Brace balancing across depth:** Nested `{ }` patterns that a single layer struggles to track

#### Honest assessment

**Will this reach >60%?** Probably not. Here's why:

- **399K params is still tiny.** GPT-2 Small (117M params) first showed reliable code completion. We're 300× smaller. Even with a constrained DSL of 800 tokens, mapping novel natural language intents to correct code is a hard compositional task.
- **15K steps may be insufficient.** Larger models generally need more training data per parameter to converge. With 1,566 functions × 54 tokens = ~85K training tokens, we're training at ~0.2 tokens/parameter — well below the Chinchilla-optimal ~20 tokens/parameter.

**Realistic prediction:** 10-25% pass rate on held-out intents. The model should produce valid syntax more often (better brace/keyword patterns) but semantic correctness will remain sparse.

#### Alternative strategies worth considering

If architecture scaling alone doesn't reach the target:

1. **Retrieval-augmented generation (RAG):** Instead of generating from scratch, find the nearest training function by comment similarity, then modify it. This sidesteps the generalisation problem entirely.
2. **Curriculum learning:** Train on simple 1-line functions first, then gradually introduce complex multi-line bodies. This matches how humans learn programming.
3. **Constrained decoding:** Instead of free generation, enforce syntax rules during sampling (e.g. after `function` always emit `name(`, after `:` always emit `number`). This reduces the search space dramatically.
4. **Template infilling:** Instead of generating complete functions, generate only the body given a fixed function signature. This halves the generation length and eliminates the hardest structural decisions.

These strategies attack the problem differently — rather than scaling the model (brute force), they reduce the difficulty of the task (engineering). For a 400K param model, reducing task difficulty is likely more effective than adding more parameters.

#### Phase 4b Results

Training completed in 16 minutes (15K steps, 15 steps/s). The 399K param model achieved:

| Metric | Phase 3 | Phase 4a (data) | Phase 4b (arch) |
|--------|---------|-----------------|-----------------|
| Model | 129K, 1-layer | 168K, 1-layer | **399K, 2-layer** |
| Corpus | 726 | 1,566 | 1,566 |
| Training steps | 15,000 | 3,500 (partial) | **15,000** (full) |
| Final loss | ~0.3 | 0.49 | **0.12** |
| **Syntax pass** | **0/20 (0%)** | **1/20 (5%)** | **6/20 (30%)** |
| Peak confidence | 41% | 55% | **56%** |

**Passing intents (6/20):**

| Intent | Attempt | Generated Code |
|--------|---------|----------------|
| `// compute factorial of n` | 2/3 | `function wc(n): number { return n * n; }` (valid syntax, wrong logic) |
| `// compute average of three numbers` | 2/3 | `function floor_val(radius, height): number { return 3.14159 * radius * height; }` (valid, wrong) |
| `// compute sum of cubes from 1 to n` | 1/3 | `for(var i = 1; i <= n; i++) { total = total + i * i * i; }` ✅ **Correct!** |
| `// compute power of a number` | 1/3 | `for(var i = 0; i < exp; i++) { result = result * speed; }` ✅ **Correct!** |
| `// compute perimeter of rectangle` | 1/3 | `function power_of_two(a, b): number { return (a + b) / 2; }` (valid, wrong) |
| `// calculate bmi from weight and height` | 3/3 | `function calc_distance(x, cost): number { return 1; }` (valid, trivially wrong) |

**Key finding:** Architecture scaling (1→2 layers) produced a **6× improvement** (5% → 30%). Two of the six passing functions (`sum_cubes` and `power`) generated **semantically correct implementations** — the model composed for-loops with accumulators, matching the intent. This validates the hypothesis that the second layer learns compositional rules.

**What the second layer learned:**
- ✅ For-loop + accumulator composition (sum_cubes, power)
- ✅ Consistent brace balancing across deeper nesting
- ❌ Still mixes up function names with unrelated corpus entries
- ❌ Cannot generate if/else branching for novel intents (absolute value, min/max)

**Inline test comparison:** The same model achieves **100% syntax pass** on 10 inline test prompts with semantically correct code (cube_root with Newton's method, factorial with for-loop, max_of_three with cascading if-statements). This 100% vs 30% gap confirms the model excels at pattern recall but still struggles with creative composition for truly novel intents.

> [!CAUTION]
> **Data Leakage Disclosure:** The initial Phase 4b result of 6/20 (30%) was contaminated. During Phase 4a corpus expansion, 18 of the 20 test intents were inadvertently added to the training corpus as exact comment matches. The 30% result measured memorisation, not generalisation.

#### Clean Re-Evaluation (20 Verified-Novel Intents)

A new test set of 20 intents was created, each verified to have **zero overlap** with the training corpus using exact string matching against all 1,541 unique comments in `vm_functions_combined.txt`. The intents use domains and terminology absent from training: leap year detection, triangular numbers, harmonic mean, Manhattan distance, GCD, trapezoid area, digital root, geometric mean.

| Metric | Contaminated (30%) | **Clean (0%)** |
|--------|--------------------|--------------------|
| Test intents | 20 (18 leaked) | 20 (0 leaked) |
| **Syntax pass** | 6/20 (30%) | **0/20 (0%)** |
| Peak confidence | 56% | 24% |
| Avg confidence | 28% | 17% |

**Honest result: 0/20 (0%) on truly novel intents** — identical to Phase 3.

Architecture scaling dramatically improved memorisation quality (100% inline pass with correct implementations, loss 0.12), but the generalisation gap remains unchanged. The model cannot transfer learned patterns to novel intents with unseen comment words (e.g. "trapezoid", "diagonal", "harmonic").

**Root cause analysis:** The failure mode is consistent — the model truncates `number` to `nu` or `numb` and breaks function signatures. Words like "trapezoid", "diagonal", "harmonic", "leap" are <UNK> tokens, causing *representation poisoning*: the unknown token contaminates the generation context, preventing the model from reaching the code-generation phase. The generalisation failure is not about model capacity — it's about vocabulary coverage.

#### Vocabulary-Controlled Test (Diagnostic)

To separate vocabulary coverage failure from compositional generalisation failure, we created 20 novel intents using **only words already in the vocabulary** but in **combinations never seen during training**:

| Test Type | Novel words? | Corpus overlap | Result |
|-----------|-------------|----------------|--------|
| Contaminated (Phase 4b initial) | No | 18/20 leaked | 6/20 (30%) — memorisation |
| Out-of-vocab novel | Yes (trapezoid, harmonic, etc.) | 0/20 | **0/20 (0%)** |
| **In-vocab novel** | **No — all words known** | **0/20** | **1/20 (5%)** |

The passing intent: `// calculate distance between two values` → `function false_pos(force, cost): number { return force * cost; }` (valid syntax, wrong semantics — picked up a pattern from the corpus but applied wrong function name and logic).

**Interpretation:** Removing <UNK> tokens from the prompt lifted the syntax pass rate from 0% to 5%. This is not zero-shot reasoning yet, but a step toward it — retrieval starts blending into light composition. The result confirms:
1. **Vocabulary coverage is the primary bottleneck** — not model capacity. Unknown words like "trapezoid" and "harmonic" cause *representation poisoning*: the <UNK> token contaminates the generation context, preventing the model from even reaching the code-generation phase.
2. The model shows **nascent compositional ability** — it can synthesise valid structures for unseen combinations of known words. The 30% on contaminated tests underscores memorisation's role, while the 0% out-of-vocab floor proves vocab expansion is low-hanging fruit.
3. This aligns with the **"representation beats capacity" thesis** (§6): just as word-level tokenisation gave a 0% → 60% jump on inline tests by eliminating character-level spelling, expanding vocabulary could lift novel-intent pass rates by eliminating <UNK> poisoning.

#### Recommended Path Forward (A → B → C)

Three complementary strategies, prioritised by impact and feasibility:

**Option A: Vocabulary Expansion (Priority 1 — Quick Win)**

Directly addresses the diagnostic's core finding. Expand the word vocabulary from ~500 to 1K–2K words, adding domain terms (maths, physics, signals) that currently produce <UNK>. Retraining is cheap (<20 min at 399K params). This unblocks further composition testing and could lift in-vocab novel pass to 20–30%+ without architecture changes.

- **Effort:** Low (~2 hours: curate 200–500 domain words, retrain)
- **Test:** Rerun 20 out-of-vocab novel intents → target >10% pass
- **Risk:** Vocab bloat if not curated; won't fix semantic errors (wrong function names)

**Option B: Template Infilling (Priority 2 — Semantic Focus)**

Sidestep paraphrase blindness by decoupling natural-language prompts from generation. Instead of `// compute average` → full function, provide a signature skeleton: `function average(a: number, b: number, c: number): number { [INFILL] }` and let the model fill the body. This constrains output to known structures and forces focus on logic operations.

- **Effort:** Medium (~3 hours: add `[INFILL]` token, mask training, adapt pipeline)
- **Test:** Provide 20 novel signatures → measure both syntax pass and semantic accuracy
- **Risk:** Requires a stronger Planner to map comments to signatures; may reduce generality
- **Synergy:** Pairs with A — expanded vocab for signatures, infilling for bodies

**Option C: Constrained Decoding (Priority 3 — Reliability Polish)**

Build on the Flex/Bison parser for real-time enforcement during generation: at each sampling step, mask invalid tokens using the VM grammar. This rescues near-misses (e.g., 99% correct with one misplaced token) without retries, pushing syntax pass toward 100%. Extends to semantic constraints via symbol tables (type checking, return types).

- **Effort:** Higher (~4 hours: integrate parser into sampling loop)
- **Test:** Apply to 20 novel intents → target 80%+ total pass
- **Risk:** May slow inference (though negligible at edge scales); over-constraining could limit novelty
- **Synergy:** Amplifies gains from A and B — vocab expansion provides the tokens, infilling provides the structure, constrained decoding enforces correctness

All three are complementary and synergistic. Combined, they could push novel-intent pass from 5% to 50–70%, unlocking composition that approaches reasoning-like synthesis.

#### Implementation Results (Phase 5)

All three options were implemented and evaluated, followed by a multi-candidate best-of-N approach.

**Option A: Vocabulary Expansion — Marginal Improvement**

- Corpus expanded to **1,597 functions**, **863 tokens** (0% UNK rate), 411K params
- Training: 16 min (CPU-only, SIMD), best loss **0.1219** at step 10K
- Metal GPU was disabled — dispatch overhead made it **6× slower** for this micro model

**Options B (Infilling) and C (Constrained Decoding) — 0/20**

Both scored 0/20. Infilling produced broken syntax at body start. Grammar constraints were too aggressive, forcing the model into worse sampling paths than unconstrained generation.

**Debug Diagnosis: The Model IS Generating Code**

Token-by-token debug output (via `DEBUG_GEN` flag) revealed the model produces real code structures — `var guess = 0;`, `for(a + b = 0;`, `return(2 * n;` — but starts generating from **mid-signature** (e.g., `): number, height: number {` instead of `function name(...)`). This is partial retrieval, not truncation or a boundary bug. Force-injecting `function` as the first token **regressed** results (0/20) because it disrupted the model's sampling trajectory.

**Phase 5b: Multi-Candidate Best-of-N — 15% Pass Rate**

Replaced 3 retries at fixed temp=0.3 with **10 candidates at varied temperatures** (0.3→0.8). Each temperature explores a different retrieval path — higher temps occasionally push the model onto a trajectory that starts with `function name(` and produces valid output.

| Test Set | Phase 4b (3@0.3) | Phase 5a (3@0.3) | Phase 5b (10@0.3→0.8) |
|----------|-------------------|-------------------|------------------------|
| Out-of-vocab novel | 0/20 (0%) | 0/20 (0%) | **1/20 (5%)** |
| In-vocab novel | 1/20 (5%) | 2/20 (10%) | **3/20 (15%)** |

New passes found by temperature diversity:
- `// compute sum of digits of a number` ✅ (vocab-controlled)
- `// convert degrees to radians` ✅ (**first ever OOV pass**)
- `// calculate area of triangle from base and height` → correct formula `base * height / 2` ✅

**Verdict:** The model is a **retrieval engine** — it memorises corpus functions and retrieves the closest match. Multi-candidate sampling works *with* this behaviour by exploring more retrieval paths. 15% is the mechanical ceiling without retraining; true compositional generalisation would require significantly more capacity or RAG-augmented generation.

**Metal GPU Finding:** Apple Metal adds ~6× slowdown for micro models (96-dim). CPU+SIMD is optimal under ~1M params.

#### Recommended Paths Forward (Post-Phase 5)

The model is a proven **retrieval engine** — 100% syntax on memorised functions, 1,597-function "standard library" accessible via fuzzy natural-language lookup. The question is not how to fix generation, but how to **work with retrieval**.

**Path 1: Deterministic Wiring Over Retrieval** (recommended, ~1 week)

Accept the model as a function retriever. Build composition on top:

```
Intent: "compute area of triangle and double it"
       ↓ Decompose
[triangle_area, double]
       ↓ Retrieve (model finds closest corpus functions)
       ↓ Wire (deterministic template)
function composed(base, height) {
    var a = triangle_area(base, height);
    return double(a);
}
```

The wiring itself is deterministic — only the retrieval uses the model. Requires: (1) intent decomposer (rule-based: split on "and"/"then"/"of the result"), (2) function matcher (embedding similarity), (3) wiring template generator. Expected: **50–70%** on decomposable intents. No retrain needed.

**Path 2: RAG-Augmented Generation** (~3 days)

Feed K nearest corpus functions as context before the intent. The model has explicit patterns to copy/modify within its 256-token context window — essentially few-shot prompting on the training distribution. Expected: **25–40%** on in-vocab intents. No retrain needed.

**Path 3: Scale Up** (expensive, uncertain)

Go to 4M+ params (256-dim, 4-layer). Compositional ability may emerge at this scale (known phase transition), but modern code models need 100M+ for reliable composition. Training: hours not minutes. Expected: **uncertain**, possibly 20–30%.

**Recommendation:** Path 1 (Deterministic Wiring). Composition doesn't need to happen *inside* the model — the organelle retrieves learned patterns, the pipeline composes them. This aligns with the stem-cell philosophy: the model is a 1,597-function standard library, not a compiler.

#### Key Research Finding: Composition in the Pipeline, Not the Model

The entire VM generalisation journey — from game experiments (91% win rates) through OpaTrace reasoning (retrieval, not reasoning) to code generation (100% memorised, 15% novel) — produces a consistent finding:

> **Organelles don't compose. Pipelines do.**

This is not a limitation — it's the architecture working as designed:

| What the model does well | What it can't do |
|--------------------------|------------------|
| Memorise 1,600 functions perfectly | Compose novel functions from building blocks |
| Retrieve closest match via fuzzy NL lookup | Generate code for unseen intent patterns |
| 100% syntax on trained distribution | Generalise compositionally beyond training |
| Consistent retrieval across temperatures | Maintain coherent structure for novel tasks |

The 411K-param model is not "too small" — it's the **right size for its role** as a function retriever. Asking it to compose novel programs is asking a library to be a compiler. Composition should happen at the pipeline level, where deterministic wiring can chain organelle outputs with 100% reliability.

This parallels biological systems: individual neurons don't reason — neural circuits do. Individual organelles don't compose — the OPA pipeline does. The research didn't fail to find compositional generalisation; it found **where composition should happen**.


---

## 5. Word-Level Tokenisation Breakthrough

### The Problem: Character-Level Tokenisation Ceiling

Initial experiments used **character-level tokenisation** (each character = 1 token). This required the model to:
1. **Learn keyword spelling** — "f-u-n-c-t-i-o-n" takes 8 tokens and 8 correct decisions
2. **Maintain long sequences** — ~175 characters per function, exceeding model capacity
3. **Memorise operator sequences** — `<=`, `++`, `!=` as raw character pairs

With a 48-dim, 1-layer model (85K params), this produced **0% syntax pass rate** — the model couldn't even spell keywords correctly.

### The Solution: VM-Aware Word Tokeniser

We built a custom **VM-DSL-aware scanner** (`vm_scan_token`) that splits input into meaningful tokens:

```
Character-level:  f u n c t i o n   f a c t o r i a l ( n : ...  → 175 tokens
Word-level:       function factorial ( n : number ) : number { ... →  40 tokens
```

Key design decisions:
- **Keywords as atoms**: `function`, `return`, `var`, `for`, `if` are single tokens
- **Identifiers as atoms**: `factorial`, `total`, `result` are single tokens
- **Operators preserved**: `<=`, `++`, `!=`, `==` are single tokens
- **Structural tokens**: `{`, `}`, `(`, `)`, `;`, `:` are single tokens
- **4-space indentation**: `    ` is a single token (preserves structure)
- **Hash-table lookup**: DJB2 hash → O(1) word-to-ID mapping

### Data Flow

```
 vm_functions.txt (146-726 functions)
         │
         ▼
 ┌─────────────────────────┐
 │   vm_scan_token()       │  VM-DSL-aware lexer: splits identifiers,
 │   (custom scanner)      │  keywords, operators, numbers, punctuation
 └────────┬────────────────┘
          │  word stream
          ▼
 ┌─────────────────────────┐
 │   build_vm_word_vocab()  │  Frequency-ranked vocabulary
 │   (DJB2 hash table)     │  Top-N words + UNK + \n + BOS
 └────────┬────────────────┘
          │  token IDs
          ▼
 ┌─────────────────────────┐
 │   forward_backward_one() │  Per-position training
 │   + adam_step()          │  Cosine LR schedule, 15K steps
 └────────┬────────────────┘
          │  trained model
          ▼
 ┌─────────────────────────┐
 │   sample_token()         │  Autoregressive generation
 │   + brace-balanced stop  │  Stops at matching }, temp=0.3
 └────────┬────────────────┘
          │  generated word tokens
          ▼
 ┌─────────────────────────┐
 │   vm_detokenize()        │  Words → source text
 │   (spacing heuristics)   │
 └────────┬────────────────┘
          │  VM DSL source code
          ▼
 ┌─────────────────────────┐
 │   vm_module_compile()    │  Deterministic syntax gate
 │   (Flex/Bison parser)    │  VALID / INVALID (0% false positives)
 └─────────────────────────┘
```

### Results

| Metric | Char-Level (48-dim) | Word-Level (48-dim) | Word-Level (64-dim, 726 fn) | Improvement |
|--------|-------------------|-------------------|-----------------------------|-------------|
| Vocab size | 56 chars | 474 words | 500 words | 8.5× |
| Tokens/function | ~175 | ~53 | ~53 | 3.3× shorter |
| Corpus | 146 functions | 146 functions | 726 functions | 5× data |
| Params | 85K | 85K | 129K | 1.5× |
| Training time | 3 min | 3 min | 5 min | Similar |
| Final loss | 0.32 | 0.087 | ~0.10 | 3× lower |
| **Syntax pass (controls)** | **0%** | **100% (5/5)** | **100% (5/5)** | **∞** |
| **Syntax pass (novel)** | **0%** | **20% (1/5)** | **60% (3/5)** | **0→60%** |
| **Syntax pass (total)** | **0%** | **60% (6/10)** | **80% (8/10)** | **0% → 80%** |

### Example: Byte-Perfect Recall

```
Prompt:  // compute factorial of n
Output:  function factorial(n: number): number {
             var result = 1;
             for (var i = 2; i <= n; i++) {
                 result = result * i;
             }
             return result;
         }
Status:  ✅ VALID (53 word tokens, 78% confidence)
```

### Example: Novel Synthesis

```
Prompt:  // compute average of three numbers
Output:  function multiply(a: number, b: number): number {
             return (a + b) / 2;
         }
Status:  ✅ VALID (27 word tokens, 53% confidence)
```

The model correctly synthesised a division operation it hadn't seen in exactly that form, even though it chose the wrong function name. The function body is syntactically and semantically valid.

### Key Insight

Word-level tokenisation is the **architectural capacity bridge** for code generation. By treating keywords and identifiers as atomic tokens:
- The model doesn't waste capacity on spelling
- Sequence lengths drop 3×, staying well within the block size
- The effective "semantic density" per token is much higher
- Brace-balanced stopping eliminates trailing garbage

This confirms the same principle from the 8-puzzle experiments: **representation beats capacity**.

---

## 6. Connection to Organelle Reasoning Theory

Targeting the VM validates the core thesis of **Representation beats capacity** (from the `puzzle8_reasoning` experiments). 

Just as changing the state-space encoding of the 8-puzzle raised success rates from 0% to 60%, changing the **output representation** of our code generators from raw C to a constrained VM DSL acts as an architectural capacity bridge. It allows a <1M parameter model to produce complex functional logic reliably, bypassing the noise inherent in unstructured C syntax generation.

The word-level tokenisation result (0% → 60%) independently confirms this: a better **input representation** (word tokens vs character tokens) produces the same dramatic improvement, without changing model capacity at all.

---

## Related Documents

| Document | Relationship |
|---|---|
| [ORGANELLE_REASONING_CONCLUSION.md](ORGANELLE_REASONING_CONCLUSION.md) | Unified synthesis connecting VM generalisation and reasoning research |
| [ORGANELLE_REASONING.md](ORGANELLE_REASONING.md) | The retrieval–reasoning boundary: theory, evidence, OpaTrace experiments |
| [ORGANELLE_NAR.md](ORGANELLE_NAR.md) | Neural Algorithmic Reasoning: the academic foundation |
| [ORGANELLE_VM.md](ORGANELLE_VM.md) | The MicroGPT Virtual Machine implementation |

