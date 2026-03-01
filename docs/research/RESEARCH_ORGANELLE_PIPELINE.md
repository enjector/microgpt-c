# White Paper: The Organelle Pipeline Architecture

## Structured Inter-Organelle Communication for Autonomous Code Synthesis

**Author:** Ajay Soni, Enjector Software Ltd.

**Date:** February 18, 2026

---

## Spear Summary

**Point:** The wire format matters more than the model — pipe-separated flat strings let sub-1M parameter Transformers communicate reliably where free-form C fails completely.

**Picture:** It's like two people who can only speak in broken sentences trying to build something together. Give them a walkie-talkie protocol — "alpha, bravo, charlie" — and suddenly they coordinate perfectly. The protocol does the heavy lifting.

**Proof:** The c_wiringgen model generates garbled C (`doublec_ve`, `pevarnounorig`) that is unparseable. The same composition as a flat string — `seq|normalize_z|fft_magnitude` — uses only 25 characters and is parseable even if partially garbled (fuzzy-match catches `noramlize_z` at edit distance 1).

**Push:** Retrain c_wiringgen on the flat-string corpus (estimated ~30 KB vs 169 KB of C) and compare novel composition accuracy. The reduced vocab (38 chars vs 63) should free model capacity for semantics.

---

### Abstract

MicroGPT-C demonstrates that sub-1M parameter transformer models are **high-fidelity
retrieval systems** — they can reproduce known patterns with near-perfect accuracy but
cannot compose novel solutions from untrained prompts. The Multi-Organelle Architecture
(see `ORGANELLES.md`) addresses this by separating intent decomposition from code
retrieval across specialised micro-models. However, a critical gap remains: **how do
organelles communicate reliably when each is an imprecise, character-level generator?**

This paper proposes the **Organelle Pipeline Architecture (OPA)** — a structured
multi-agent system where specialised micro-models communicate via a constrained
**delimited flat-string** intermediate representation (pipe-separated tokens),
orchestrated by a deterministic pipeline controller with feedback loops. The
architecture draws directly from both biological cell signalling and modern AI agent
playbook patterns (Planner → Workers → Judge), adapted for the extreme constraints
of sub-1M parameter models. The flat-string format is a **regular language** (no
nesting, no balanced delimiters), minimising syntactic overhead and maximising the
proportion of model capacity available for semantic learning.

> [!NOTE]
> **Implementation Status (Feb 2026):** The core pipeline primitives (`Organelle`,
> `OpaKanban`, `OpaCycleDetector`, `organelle_train`, `organelle_generate`) are now
> implemented as a shared C library in [`microgpt_organelle.c|h`](../../src/microgpt_organelle.h).
> Three game experiments (puzzle8, tictactoe, connect4) validate the architecture
> with **87–90% success rates** using 460K-param organelles (N_EMBD=96, N_LAYER=4).
> Ensemble voting + valid-move pre-filtering eliminate invalid moves entirely.
> The 8-Puzzle achieves **90% solve** with zero parse errors after capacity scaling.

### 1. The Communication Problem

#### 1.1 Experimental Evidence

Training results from two organelles reveal the core challenge:

| Organelle | Corpus | Params | Best Loss | In-corpus accuracy | Novel accuracy |
|---|---|---|---|---|---|
| **c_codegen** (Library) | 2,081 C functions (481 KB) | 875K | 0.034 | ~100% (byte-perfect) | ~0% (garbled) |
| **c_wiringgen** (Architect) | 864 compositions (169 KB) | 868K | 0.112* | ~95% structural match | ~20% (partial structure) |

*\*c_wiringgen at step 11,000/50,000 — training in progress*

When c_wiringgen generates output for novel prompts, it produces structurally plausible
but lexically garbled code:

```c
/* Intended: normalize then compute cosine similarity */
void noinm_ve(double *out, const double *x, int n) {
  doublec_ve = 0;
  strs s pevarnounorig(x, n);
  r (i > < n) out[i] out[i] - = - 0;
}
```

The **structural template** is correct (function signature, local variables, loop,
function call), but the **symbols are corrupted** (`noinm_ve`, `pevarnounorig`). If
another organelle receives this output as input, it cannot parse it — **garbage in,
garbage out**.

#### 1.2 The Biological Analogy

Real biological organelles do not communicate in prose. They use **molecular signals
with rigid molecular shapes** — ATP, cAMP, calcium ions. These signals are:

- **Structurally rigid** — the receiver either recognises the shape or doesn't
- **Unambiguous** — one molecule, one meaning
- **Cheap to produce** — cells don't synthesise novels, they emit simple signals
- **Validatable** — receptors reject malformed molecules instantly

The equivalent for micro-models is a **constrained symbolic wire format** — a
communication protocol simple enough that even an imprecise 800K-parameter model
can produce it reliably, and a deterministic parser can validate it instantly.

---

### 2. The Agent Playbook Pattern

The Organelle Pipeline Architecture adapts the established AI Agent Playbook pattern —
a multi-agent system where a **Planner** decomposes tasks, dispatches to specialised
**Workers**, and routes all outputs through a **Judge** that provides feedback for
iterative refinement.

#### 2.1 Agent Playbook Structure

```
                    ┌───────────┐
                    │ Summarise │
                    └─────▲─────┘
                          │
              ┌───────────┼────────── feedback ──┐
              │           │                      │
   ┌──────────┴──┐   ┌───┴────┐   ┌──────────┐  │
   │ Receive Task ├──▶│ Planner├──▶│ Composer │──┼──▶ Judge
   └─────────────┘   │(Kanban)│──▶│          │  │     │
                      │        │   └──────────┘  │     │
                      │        │──▶┌──────────┐  │     │
                      │        │   │Web Surfer│──┘     │
                      │        │   └──────────┘        │
                      │        │──▶┌──────────┐        │
                      │        │   │Let's Think│───────▶│
                      └────────┘   └──────────┘        │
                           ▲                           │
                           └───────────────────────────┘
```

Each agent is specialised: the Planner breaks work into tasks (Kanban), Workers execute
in parallel, and the Judge validates all outputs before accepting or feeding back for
revision.

#### 2.2 Mapping to MicroGPT-C Organelles

| Agent Role | Organelle | Function | Status |
|---|---|---|---|
| **Receive Task** | User intent | Natural language comment | ✅ Exists |
| **Planner** | `c_wiringgen` | Decomposes intent into primitive call sequences | 🟡 Training |
| **Composer** | `c_codegen` | Generates C function bodies for each primitive | ✅ Trained |
| **Web Surfer** | `c_registry` | Validates function names against known symbol table | ❌ Proposed |
| **Let's Think** | `c_reason` | Infers types and validates argument compatibility | ❌ Proposed |
| **Summarise** | `c_docgen` | Generates comments from code (reverse mapping) | ❌ Proposed |
| **Judge** | `c_judge` | Validates output (compilability, structural correctness) | ❌ Proposed |

The coordination between these roles is managed by a **kanban state machine** — a shared flat-string protocol that gives stateless models the equivalent of short-term memory. Each organelle reads a pipe-delimited state string containing the current task, blocked moves, and move history. This prevents repeated errors and breaks oscillation cycles, turning individually unreliable models into a system that achieves 87–90% success rates.

![The Kanban State Machine — Planner-Worker-Judge loop, flat-string protocol, blocked move tracking, cycle breaking, and pipeline performance across 8-Puzzle, Connect-4, and Tic-Tac-Toe](images/The%20Kanban%20State%20Machine%20Infographic.jpg)

---

### 3. The Inter-Organelle Wire Format

#### 3.1 Why Free-Form C Fails

The current implicit wire format is free-form C code. This fails for three reasons:

1. **Large character set** — 90+ characters, many with syntactic significance
2. **Context-sensitive grammar** — C requires a full parser to validate
3. **Error detection is intractable** — is `doublec_ve` a typo or a valid identifier?

A micro-model generating free-form C is like a cell trying to communicate by
synthesising entire proteins from scratch — too complex, too error-prone.

#### 3.2 Candidate Wire Formats

| Format | Char set | Parse complexity | Error detection | Nesting | Model capacity |
|---|---|---|---|---|---|
| Free-form C | 90+ | Full C parser | Very hard | Unlimited | High |
| **Delimited Flat Strings** | **~25** | **`strtok` / split** | **~90% immediate** | **None (linear)** | **Very low** |
| S-expressions | ~40 | Recursive descent | ~80% (balanced parens) | Unlimited | Low |
| Prolog | ~50 | Prolog engine | Built-in (unification) | Unlimited | Medium |
| JSON | ~60 | Standard parser | Standard | Unlimited | Medium |

#### 3.3 Recommended Format: Delimited Flat Strings (Pipe-Separated)

For sub-1M parameter models, **nesting is the enemy**. Generating balanced
parentheses or brackets requires the model to learn a context-free grammar —
this consumes parameter capacity that should be spent on semantics. Analysis of
the c_wiringgen corpus reveals that **>90% of compositions are linear pipelines**
(A → B → C), not deeply nested trees.

The strongest recommendation is therefore a **flat, non-nested string format**
with fixed delimiters:

```
seq|normalize_z|fft_magnitude
```

This means: "execute `normalize_z` then `fft_magnitude` in sequence." No parens,
no brackets, no recursion — pure linear structure.

**Format specification:**

```
<type>|<fn1>|<fn2>|...|<fnN>

Types:
  seq    — sequential pipeline (output of fn_i feeds fn_{i+1})
  par    — parallel execution (independent, results collected)
  agg    — aggregation (all outputs reduced to single result)
  if     — conditional: if|<condition>|<then_fn>|<else_fn>
  win    — windowed:    win|<window_size>|<fn>
```

**Realistic examples from the wiring corpus:**

```
;; normalize array to zero mean unit variance
seq|mean|stddev|vec_affine

;; smooth then differentiate signal
seq|rolling_mean|differentiate

;; chain lowpass filter then downsample
seq|lowpass|downsample

;; compute rolling z-score
win|20|zscore_normalize

;; MACD histogram
seq|macd|sma|vec_subtract

;; conditional routing: high variance path vs low variance path
if|high_variance|denoise|identity|seq|normalize_z|fft_magnitude

;; two-stage smoothing with different periods
seq|rolling_mean;5|rolling_mean;10
```

Arguments to functions use `;` as a sub-delimiter:
`rolling_mean;5` means `rolling_mean(data, period=5)`.

**Advantages over S-expressions:**

1. **Zero nesting** — no balanced delimiters to learn; purely linear token sequences
2. **Trivially parseable** — `strtok(line, "|")` in C, under 10 lines of code
3. **~90% error detection** — wrong delimiter count, extra/missing `|`, or unknown
   function name all cause immediate parse failure
4. **Minimal character set** — `|`, `;`, `a-z`, `0-9`, `_` (~25 characters total)
5. **Maximum semantic capacity** — near-zero syntactic overhead leaves the full
   parameter budget for learning composition patterns
6. **Natural for retrieval** — the model memorises short delimiter-separated
   sequences rather than syntactic trees

**Comparison with current C output:**

```
Current (unreliable — 90 chars, context-sensitive):
  void noinm_ve(double *out, const double *x, int n) {
    doublec_ve = 0; strs s pevarnounorig(x, n); ...
  }

Proposed flat string (reliable — 25 chars, linear):
  seq|normalize_z|cosine_similarity
```

The flat string is **parseable even if partially garbled** — a mistyped function
name (`noramlize_z`) can be fuzzy-matched against the registry (edit distance 1),
whereas garbled C (`doublec_ve`) is unrecoverable. And unlike S-expressions, the
model never needs to track nesting depth.

#### 3.4 Wire Format Grammar

```ebnf
blueprint   := type '|' fn_list
fn_list     := fn_call ('|' fn_call)*
fn_call     := fname (';' arg)*

type        := 'seq' | 'par' | 'agg' | 'if' | 'win'
fname       := [a-z_][a-z0-9_]*
arg         := [a-z0-9_.]*
```

This grammar is **regular** (no recursion, no nesting) — it can be validated by
a finite automaton. The entire validator is a single `strtok` loop with field-count
checks.

**Total character set:** `a-z 0-9 _ | ; .` — **38 characters** including the
comment prefix. Compare with C's 90+ characters.

#### 3.5 Handling Nested Compositions (Escape Hatch)

For the rare cases that require deeper structure (estimated <10% of compositions),
two strategies are available:

1. **Multi-message protocol** — the Planner emits multiple flat strings that the
   orchestrator chains:
   ```
   ;; denoise, then bandpass, then compute energy
   seq|denoise|bandpass        ;; message 1: first stage
   seq|compute_energy          ;; message 2: applied to output of message 1
   ```

2. **S-expression fallback** — for genuinely hierarchical compositions, a second
   organelle variant trained on S-expressions handles the rare nested case:
   ```lisp
   (seq (call denoise) (par (call bandpass) (call highpass)) (call energy))
   ```
   This keeps the primary (flat) organelle simple and fast, with the S-expression
   variant as a specialist for complex cases.

#### 3.6 Training Corpus Transformation

The existing c_wiringgen corpus of 864 C compositions can be mechanically
transformed to delimited flat strings:

```
Before (C — 169 KB corpus, 63-char vocab):
  /* normalize array to zero mean unit variance */
  void normalize_z(double *out, const double *x, int n) {
    double m = mean(x, n);
    double s = stddev(x, n);
    for (int i = 0; i < n; i++) out[i] = (x[i] - m) / s;
  }

After (flat string — estimated ~30 KB corpus, 38-char vocab):
  ;; normalize array to zero mean unit variance
  seq|mean|stddev|vec_affine
```

The flat string corpus will be **~5× smaller** than the C corpus, with a **40%
smaller vocabulary**. This gives the model dramatically more capacity per pattern
and is predicted to substantially improve generalisation on novel compositions.

```
  Capacity allocation comparison:

  Free-form C:     [████████████ syntax ████████████][███ semantics ███]
  S-expressions:   [████ syntax ████][█████████ semantics █████████████]
  Flat strings:    [█ syn █][█████████████████ semantics █████████████████]
```

---

### 4. The Full Pipeline Architecture

#### 4.1 System Diagram

```
   User Intent: "/* normalize then compute spectrum */"
          │
          ▼
   ┌──────────────┐     Flat string IR
   │  c_wiringgen  │──────────────────────────────────┐
   │  (Planner)    │  seq|normalize_z|fft_magnitude   │
   │               │                                  │
   └──────────────┘                                   │
          │                                            │
          ▼                                            ▼
   ┌──────────────┐                            ┌──────────────┐
   │  c_registry   │◄─────── lookup ──────────│  Parser       │
   │  (Symbol Table)│  "normalize_z" → FOUND   │  (strtok loop)
   │               │  "fft_magnitude" → FOUND  │  ~10 lines C
   └──────┬───────┘                            └──────────────┘
          │ confirmed symbols
          ▼
   ┌──────────────┐     C function bodies
   │  c_codegen    │──────────────────────────┐
   │  (Composer)   │  void normalize_z(...){  │
   │               │    mean(); stddev();     │
   └──────────────┘  }                        │
          │                                    │
          ▼                                    ▼
   ┌──────────────┐                     ┌──────────────┐
   │  c_judge      │◄── assembled code ─│  Template     │
   │  (Validator)  │                    │  Expander     │
   │               │                    │  (Deterministic)
   └──────┬───────┘                    └──────────────┘
          │
     ┌────┴────┐
     │         │
    PASS     FAIL ──▶ feedback to c_wiringgen
     │                 (re-prompt with constraint)
     ▼
   Compilable C output
```

#### 4.2 Component Roles

| Component | Type | Description |
|---|---|---|
| **c_wiringgen** | Neural (organelle) | Generates flat-string composition plans from natural language |
| **Parser** | Deterministic | Validates flat string syntax (`strtok` split, field count, known keywords) |
| **c_registry** | Neural or table | Confirms function names exist and returns signatures |
| **c_codegen** | Neural (organelle) | Generates C function bodies for each confirmed primitive |
| **Template Expander** | Deterministic | Assembles flat-string plan + function bodies into compilable C |
| **c_judge** | Neural (organelle) | Validates assembled output (structure, completeness, types) |

**Key design principle:** Neural organelles handle the creative/uncertain steps
(planning, code generation, validation). Deterministic components handle the
structural steps (parsing, assembly, lookup). This **minimises the surface area
where imprecision can cause failures**.

#### 4.3 Feedback Loop

When the Judge rejects output, the feedback flows back to the Planner:

```
Attempt 1:
  c_wiringgen: seq|noramlize_z|fft_magnitude
  c_registry:  "noramlize_z" → NOT FOUND (fuzzy match: "normalize_z", dist=1)
  Action:      Re-prompt with constraint: "normalize_z" (corrected)

Attempt 2:
  c_wiringgen: seq|normalize_z|fft_magnitude
  c_registry:  FOUND + FOUND
  c_codegen:   generates bodies
  c_judge:     PASS
```

The feedback mechanism uses **constrained decoding** — on retry, the planner
receives the registry's suggested correction, narrowing the output space.

---

### 5. Proposed New Organelles

#### 5.1 c_registry — The Symbol Table

**Purpose:** Maps function names to signatures. Acts as the shared vocabulary
between all organelles.

**Implementation options:**

| Approach | Pros | Cons |
|---|---|---|
| **Static hash table** | Deterministic, zero error | Cannot generalise to new functions |
| **Neural organelle** | Can fuzzy-match novel names | May hallucinate false matches |
| **Hybrid** | Best of both — exact match first, neural fallback | Slightly more complex |

**Recommended:** Hybrid approach. The registry is a static lookup table derived
from c_codegen's training corpus, with a neural fuzzy-matcher for near-misses.

**Training corpus format:**
```
normalize_z : void (double* out, const double* x, int n)
mean : double (const double* x, int n)
fft_radix2 : void (double* re, double* im, int n)
```

#### 5.2 c_judge — The Validation Organelle

**Purpose:** Binary classifier that determines whether assembled code is
structurally valid.

**Training corpus format:**
```
/* INPUT */ seq|mean|stddev
/* VERDICT */ PASS — valid pipeline, known functions, compatible types

/* INPUT */ seq|noramlize_z
/* VERDICT */ FAIL — unknown function "noramlize_z" (suggest: "normalize_z")
```

**Key insight:** The Judge does not need to understand C. It validates the
flat-string IR, which has a much simpler structure than C code. This dramatically
reduces the model capacity required.

#### 5.3 c_docgen — The Reverse Mapping Organelle

**Purpose:** Generates natural language descriptions from flat-string plans.
Closes the verification loop — if the generated description matches the original
intent, the plan is semantically correct.

**Training corpus:** The reverse of c_wiringgen's corpus:
```
Input:  seq|normalize_z|fft_magnitude
Output: /* normalize array then compute frequency spectrum */
```

**Use case:** The Planner generates a plan → c_docgen describes it → the
orchestrator compares the description with the original intent → if they
diverge, retry.

---

### 6. Implementation Roadmap

#### Phase 1: Wire Format (Immediate)

1. Write a mechanical transformer to convert the existing 864 C wiring compositions
   to delimited flat-string format
2. Retrain c_wiringgen on the flat-string corpus
3. Compare generation quality: garbled C vs structured flat strings
4. Implement the deterministic flat-string parser (`strtok` loop, ~10 lines of C)

#### Phase 2: Registry + Judge (Q2 2026)

1. Extract all function signatures from c_codegen's corpus to build the static registry
2. Train c_judge on PASS/FAIL pairs derived from the registry
3. Implement the pipeline controller (orchestrator) as a simple C program
4. End-to-end test: intent → flat string → lookup → code → judge

#### Phase 3: Feedback Loop (Q3 2026)

1. Implement constrained retry: on Judge FAIL, re-prompt Planner with corrections
2. Train c_docgen for semantic verification
3. Measure pipeline success rate on novel compositions
4. Benchmark: single-model vs pipeline accuracy on composition tasks

#### Phase 4: Domain Organelles (Q4 2026)

1. Introduce domain-specific Planners (financial, signal processing, etc.)
2. Each domain organelle translates domain language to generic flat-string IR
3. The pipeline becomes: Domain → Plan → Lookup → Code → Judge

---

### 7. Theoretical Analysis

#### 7.1 Why Composition Requires Structure

A fundamental result from formal language theory: **context-free languages cannot
be learned from positive examples alone** (Gold's theorem, 1967). Character-level
models trained on code samples receive only positive examples. They can memorise
the training distribution but cannot reliably generalise to novel compositions.

The delimited flat-string wire format sidesteps this by **reducing the generation
task from context-sensitive (C) to regular (pipe-delimited strings)**. The model
no longer needs to learn C's full grammar — only the trivially simple flat-string
format. This is a **two-level reduction** in the Chomsky hierarchy (context-sensitive
→ regular), a strictly stronger simplification than S-expressions (which remain
context-free due to nesting).

#### 7.2 Capacity Allocation

With a fixed parameter budget (~800K), the model must allocate capacity between:
- **Syntax** — learning the grammar of the output format
- **Semantics** — learning which functions to call and in what order

Free-form C consumes significant capacity on syntax (operator precedence, declaration
syntax, pointer notation). Even S-expressions require learning balanced parentheses.
Flat strings consume almost **zero** syntactic capacity — a single delimiter character
(`|`) is the entire grammar — freeing the maximum possible parameter budget for
semantic learning (composition patterns).

```
  Free-form C:     [████████ syntax ████████][██ semantics ██]
  S-expressions:   [███ syntax ███][████████ semantics ████████]
  Flat strings:    [█][█████████████ semantics █████████████████]
```

This predicts that a flat-string-trained Planner will generalise better on novel
compositions than either a C-trained or S-expression-trained one — the same model
capacity is focused entirely on the *right* problem.

#### 7.3 Error Surface Comparison

| Property | Free-form C | S-expression | Flat string |
|---|---|---|---|
| Characters where a typo causes semantic error | ~60% | ~10% | ~5% |
| Characters where a typo causes parse failure | ~30% | ~80% | ~90% (caught immediately) |
| Characters that are "noise" (syntax sugar) | ~40% | ~5% | ~2% |
| Nesting depth to track | Unlimited | Unlimited | **0** |
| Grammar class | Context-sensitive | Context-free | **Regular** |

Flat strings make errors **loud and fast** — nearly all mistakes cause parse
failures, not silent semantic corruption. And the zero nesting depth means the
model never needs to maintain a stack — a key advantage for sub-1M parameter
models with limited working memory.

---

### 8. Comparison with Related Work

| System | Model size | Communication | Composition | Edge-capable |
|---|---|---|---|---|
| GPT-4 (monolithic) | ~1.8T params | N/A (single model) | Yes (emergent) | ❌ No |
| AutoGen (Microsoft) | LLM agents | Natural language | Yes | ❌ No |
| CrewAI | LLM agents | Natural language | Yes | ❌ No |
| GGML/llama.cpp | 1B–70B | N/A (single model) | Limited | 🟡 Partial |
| **MicroGPT-C OPA** | **~800K × N** | **Delimited flat strings** | **Pipeline** | ✅ **Yes** |

The key differentiator is the combination of **structured communication** with
**sub-1M parameter models**. All existing multi-agent systems assume LLM-class
models that can communicate in natural language. OPA is the first architecture
designed for the constraints of micro-models — where natural language communication
is unreliable and must be replaced with structured symbolic protocols.

---

### 9. Conclusion

The Multi-Organelle Pipeline Architecture transforms MicroGPT-C from a collection
of independent micro-models into a **coordinated system capable of novel composition**.
The key insight is that **the wire format matters more than the model** — by
constraining inter-organelle communication to a simple delimited flat-string
representation, we:

1. **Reduce the generation complexity** from context-sensitive (C) to regular
   (flat strings), allowing models to focus all capacity on semantics
2. **Eliminate nesting entirely** — zero stack depth, zero balanced-delimiter
   errors, pure linear token sequences
3. **Enable instant error detection** through trivial field-count validation
4. **Support graceful degradation** through fuzzy matching and feedback loops
5. **Maintain edge deployment** — the entire pipeline (5–6 organelles × ~460K
   params each) fits in under 50 MB of RAM

The architecture follows a proven pattern from AI agent systems — Planner → Workers
→ Judge — adapted for the extreme constraints of sub-1M parameter models. Each
organelle remains simple. The intelligence emerges from the pipeline.

This is the software equivalent of biological specialisation: no single organelle
is intelligent, but the cell — the pipeline — is.

---

*MicroGPT-C Organelle Pipeline Architecture — Enjector Software Ltd. MIT License.*
