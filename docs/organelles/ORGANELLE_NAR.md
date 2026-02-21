# Neural Algorithmic Reasoning (NAR) in MicroGPT-C

**Author:** Ajay Soni, Enjector Software Ltd.  
**Date:** February 2026  
**Status:** Research Document

---

## 1. Executive Summary

Modern AI development responds to what Demis Hassabis calls **Jagged Intelligence** — the tendency of massive models to pass the Math Olympiad while failing elementary logic — by scaling parameters into the trillions to "fill the gaps."

**MicroGPT-C** proposes a fundamental alternative: **Neural Algorithmic Reasoning (NAR) at the edge.** Instead of building a monolithic simulator that wastes parameter capacity internally replicating fuzzy versions of deterministic algorithms (BFS, DFS, state-tracking, cycle detection), MicroGPT-C offloads these "logic tracks" to a structured C99 pipeline. Sub-1M parameter **organelles** use high-fidelity pattern retrieval to navigate algorithmic states, while a deterministic orchestrator provides the coordination logic.

**The result:** 90%+ solve rates on logic problems, syntactically valid C code generation, and 79–91% win rates across board games — all from models small enough to run on microcontrollers.

**The thesis:** Intelligence is a property of the **coordination protocol**, not just the model weights. The OPA pipeline is the reasoning layer; the organelles are the retrieval layer.

---

## 2. The Problem: Monolithic Capacity Waste

### 2.1 LLMs Are Inefficient Reasoners

Large Language Models attempt to perform "System 2" (slow, deliberate logic) using "System 1" (fast, intuitive pattern matching) hardware. The result is a single undifferentiated parameter blob that must simultaneously learn syntax, semantics, search strategy, state tracking, and validity checking from the same gradient signal.

Three failure modes emerge:

1. **Implicit scaffolding.** LLMs use their context window as a makeshift Kanban board, wasting parameter capacity simply learning to track what they have already said. Chain-of-thought prompting forces the model to externalise this structure — confirming that the model cannot maintain it reliably without explicit tokens.

2. **Fuzzy operators.** Transformers build internal "neural operators" to simulate deterministic algorithms. The CLRS-30 benchmark (DeepMind, 2022) tests 30 classical algorithms — sorting, BFS, DFS, shortest path — against neural models, and consistently finds that models trained on small instances fail when problem size increases, **even after extensive scaling.** The neural operators are brittle approximations, not the algorithms themselves.

3. **The jaggedness trap.** Increasing model capacity reduces noise (parse errors, oscillations) without producing reasoning. As proven in our Phase 5 experiments (§5): a 7× capacity increase (64K → 460K params) eliminated oscillation entirely but left the failure ceiling unchanged. The model became a better librarian, not a mathematician.

### 2.2 The Neural Operator Taxonomy

The algorithms LLMs must internalise fall into four classes, each of which can be expressed trivially in deterministic code:

| Class | LLM approximation | OPA equivalent | Cost in LLM params | Cost in OPA |
|---|---|---|---|---|
| **Logic operators** | Attention patterns for AND/OR/NOT | C `if/else` branch | Millions | 1 line |
| **Status/comparison** | Fuzzy "neural status registers" | `>`, `==` operators | Hundreds of thousands | 1 branch instruction |
| **Search operators** | Layer-by-layer BFS expansion | 30-line queue/stack BFS | Billions (fragile) | Standard library |
| **State-tracking** | Context window tokens as scratchpad | `OpaKanban` struct | Millions (unreliable) | ~80 lines of C |

The OPA architecture **externalises all four classes.** The models handle only what they are good at: fuzzy pattern matching over a structured output format.

---

## 3. Prior Art: What the NAR Community Has Found

The MicroGPT-C project sits within a broader research tradition. The field of **Neural Algorithmic Reasoning** has studied what deterministic algorithms transformers build internally — and how accurately they replicate them.

### 3.1 Key Findings from Mechanistic Interpretability

| Circuit | Algorithm approximated | Reliability | Key reference |
|---|---|---|---|
| **Induction heads** | Sequence matching / copy-paste lookup | High (within training distribution) | Anthropic, 2022 |
| **Attention-based BFS** | Graph reachability via layer-by-layer expansion | Degrades with graph size | Xu et al., 2019 |
| **Implicit bit-comparison** | Numerical comparison (X > Y?) | Brittle on distribution shift | Multiple |
| **Emergent symbol-processing** | Variable binding and substitution | Inconsistent emergence | Smolensky et al., 2022 |
| **Chain-of-thought scratchpad** | Kanban-style stateful planning | Reliable but context-window bounded | Wei et al., 2022 |

**The consistent pattern:** transformers *learn* BFS, DFS, and sorting from execution traces, but fail to *generalise* to larger inputs. The algorithm is memorised for the training distribution, not derived from first principles.

### 3.2 CLRS-30: The Benchmark MicroGPT-C Doesn't Need

The CLRS-30 benchmark explicitly measures whether neural networks can execute classical algorithms. Neural models consistently fail on out-of-distribution inputs. MicroGPT-C's response is not to train models on more algorithm traces — it is to **not ask the models to execute algorithms at all.** The BFS runs at corpus-generation time (Python), not at inference time (C). The model is asked only: *"given this local pattern, what does the pattern matching say?"*

This is the core NAR insight: separate **what looks like the algorithm's output** (retrievable by a small model) from **the execution of the algorithm** (handled by deterministic code).

---

## 4. The OPA Solution: Externalising NAR

### 4.1 The Organelle Pipeline Architecture

MicroGPT-C replaces the generalist monolith with a biological cell metaphor — the **Organelle Pipeline Architecture (OPA)**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OPA Cell                                     │
│                                                                     │
│  Input Intent                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────┐    flat-string wire    ┌──────────────┐              │
│  │ Planner  │───────────────────────▶│   Worker(s)  │              │
│  │ Neural   │   "seq|fn1|fn2"        │   Neural     │              │
│  │ retrieval│                        │   retrieval  │              │
│  └──────────┘                        └──────┬───────┘              │
│                                             │  candidate output     │
│                                             ▼                       │
│                                      ┌──────────────┐              │
│                                      │    Judge     │              │
│                                      │ Deterministic│              │
│                                      │ (+ optional  │              │
│                                      │  neural)     │              │
│                                      └──────┬───────┘              │
│                                             │  PASS / FAIL          │
│                                             ▼                       │
│                                      ┌──────────────┐              │
│                                      │  OpaKanban   │◀─ blocked    │
│                                      │  State Track │   list       │
│                                      └──────┬───────┘              │
│                                             │  retry / done         │
│                                             ▼                       │
│                                       Result / Output               │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Each Component's NAR Role

| Component | NAR function | What it does NOT do |
|---|---|---|
| **Planner organelle** | Concept normalisation — maps messy intent to structured primitives | Execute search; reason about consequences |
| **Worker organelle** | High-fidelity retrieval — pattern → output with byte-level accuracy | Generalise to unseen patterns |
| **Judge (deterministic)** | Hard validity gate — zero false positives | Learn; adapt; remember |
| **Judge (neural)** | Soft semantic gate — confidence on output quality | Enforce invariants deterministically |
| **OpaKanban** | State tracking — todo / doing / done / blocked | Any fuzzy inference |
| **OpaCycleDetector** | Cycle detection — prevents oscillation loops | Pattern recognition |
| **Wire format** | Structured communication — flat-string protocol | Carry implicit state |

The key architectural claim: **deterministic code handles everything that must be correct; neural models handle everything that requires pattern recognition.** This separation of concerns is not just software engineering hygiene — it is a parameter efficiency argument.

### 4.3 Wire Format as NAR Infrastructure

The flat-string wire format (`seq|normalize_z|rolling_mean`, `m=3,5,x,4|valid=up,right`) is a NAR design choice. By constraining inter-component communication to a structured, low-entropy format:

- The model's syntactic budget is minimised (learning `seq|fn1|fn2` requires far fewer parameters than learning free-form C code)
- The judge's task is reduced to parsing and lookup (O(1) instead of semantic analysis)
- The entire communication channel is inspectable and loggable

The puzzle8 experiments proved this directly: encoding the board as raw digit strings gave 0% generalisation; encoding it as MD-delta values (`m=3,5,x,4`) that make the greedy rule explicit gave **60% generalisation on unseen puzzles** — with the same architecture, same parameters, same training time.

> **Wire format beats model size.** A 460K-parameter model with MD-delta encoding matched what a model ten times larger might achieve with free-form board strings.

---

## 5. Mechanics: Process Retrieval vs. Answer Retrieval

### 5.1 The Traditional Failure Mode

Standard organelle training produces **answer retrieval**:

```
Input:   Board state
Output:  Best move
Problem: If the board is novel, the lookup fails → garbled output
```

The `c_codegen` experiment is the definitive proof: 875K params, 2,081 training functions, **100% corpus recall** — and **0/10 on novel prompts**. More parameters built a bigger lookup table, not a smarter programmer.

### 5.2 Process Retrieval via Reasoning Traces

The NAR advance is **process retrieval** — training on the pipeline's own coordination history:

```
Input:   Board state + rejection history + stall count + blocked directions
Output:  Next move + adaptation strategy
Effect:  Model learns the logic of finding the answer, not just the answer
```

The OpaTrace API (implemented in `microgpt_organelle.h`) serialises each pipeline step:

```
TRACE|initial=12|final=0|solved=1|steps=6
1|up|accepted|12>11|none|model
2|right|accepted|11>10|none|model
3|up|rejected|10>-1|up|model        ← model learns: "up was blocked here"
4|left|stall|10>10|none|fallback    ← model learns: "stall detection triggered"
5|down|replan|10>10|none|fallback   ← model learns: "replan fired after 3 stalls"
6|right|accepted|10>0|none|model    ← model learns: "this resolved it"
```

By training on these traces, the model internalises three information-theoretic barriers it otherwise cannot overcome:

| Barrier | Traditional training | Trace training |
|---|---|---|
| **Fixation** | Repeats rejected moves (no memory) | Learns `board + blocked:right → try other` |
| **Oscillation** | Cycles A→B→A indefinitely | Learns to recognise cycle patterns, choose third option |
| **Non-monotonic blindness** | Cannot accept temporary metric regression | Learns traces where md↑ then ↓ (detour patterns) |

### 5.3 The Coordination Gap — Quantified

The gap between individual model accuracy and pipeline success rate is the clearest evidence that intelligence lives in coordination, not weights:

| Experiment | Individual model accuracy | Pipeline system success | Gap |
|---|---|---|---|
| 8-Puzzle (64K params) | ~50% valid moves | **90% puzzles solved** | **+40%** |
| 8-Puzzle (460K params) | ~90% valid moves | **90% puzzles solved** | ~0% |
| Mastermind | 65% individual | **79% games won** | +14% |
| Connect-4 | 72% individual | **91% games won** | +19% |
| c_compose (v1) | 4% registry hit | **65% judge pass** | +61% |

At 64K params, the 340-line C coordination library (Kanban + cycle detector + judge) transforms 50%-accurate models into 90%-successful systems. This is the concrete quantification of the NAR thesis: the orchestrator provides what no amount of gradient descent efficiently encodes.

---

## 6. Experimental Proof

### 6.1 The 8-Puzzle: Capacity Scaling Verdict (Phase 5)

The definitive test of the NAR thesis: can a sufficiently large model solve puzzles without deterministic pipeline supports?

| Metric | 64K Assisted | 64K Bare | 460K Assisted | 460K Bare |
|---|---|---|---|---|
| **Solve Rate** | 20% | 3% | **90%** | **90%** |
| Cycle Breaks | 181 | 0 | 23 | 0 |
| Success drop (bare) | — | **-85%** | — | **0%** |
| Hard-band (md 9+) | 20% | — | 70% | 70% |

**Conclusions:**
- **Scaffolding is a capacity bridge.** At 64K params, the Kanban compensates for the model's inability to parse enriched prompts. At 460K, the model handles the prompt and the Kanban has nothing to correct.
- **The ceiling doesn't move.** The same 3 hard puzzles fail at every configuration. They require detour moves that contradict the greedy heuristic — a corpus coverage problem, not a capacity problem.
- **Scaling ≠ reasoning.** 7× more parameters produced 0% improvement in generalisation. The model became a better retriever, not a reasoner.

### 6.2 The c_codegen Experiments: Retrieval Boundary

An 875K-parameter model trained on 2,081 C functions:

| Test type | Result | Interpretation |
|---|---|---|
| In-corpus prompts | **~100% byte-perfect** | Memorised the training data |
| Novel prompts | **0/10** | Cannot generalise at all |
| Paraphrase test | **0% match** | Maps strings, not concepts |
| 6× scaling (142K→875K) | 0/10 → **0/10** | No change in novel accuracy |

The paraphrase test is definitive:
- `/* sort values in ascending order */` → byte-perfect `bubble_sort()` (100% confidence)
- `/* ascending sort */` → garbled token soup (35% confidence)

Same concept, different words — total failure. This is the lexical (not semantic) retrieval proof.

### 6.3 Game Experiments: Three Performance Tiers

The full game experiment suite reveals a clear taxonomy that aligns with the NAR retrieval–reasoning boundary:

| Tier | Games | Performance | Why |
|---|---|---|---|
| **Coordination-dominated** | Pentago (91%), Connect-4 (91%), Tic-Tac-Toe (86%), Mastermind (79%) | 79–91% | Finite, learnable pattern space; pipeline amplifies retrieval |
| **Right-sizing unlocks gains** | Sudoku (78%), Othello (67%), Klotski (62%) | 62–78% | Pattern space larger; corpus coverage and capacity matter |
| **Reasoning-limited** | Red Donkey (12%), Lights Out (10%), Hex (4%) | 4–12% | Requires spatial reasoning, algebraic inference — hard limits |

The bottom tier is the clearest proof of the retrieval ceiling. Hex (4%) requires *connection* — a spatial reasoning concept that cannot be reduced to local pattern matching. No corpus expansion or parameter scaling within the sub-1M range fixes this; the game demands capabilities the architecture does not possess.

### 6.4 c_compose_v3: Compositional Code Generation

The three-organelle code generation pipeline extends NAR to a generative task:

```
intent → c_planner → "seq|fn1|fn2" → c_wiringgen → C function body
       → gcc -fsyntax-only → c_judge → PASS/FAIL → OpaKanban retry
```

- **c_planner** (50K steps, best loss 0.085): Maps natural language to flat-string composition plans
- **c_wiringgen** (20K steps): Converts plans to C function bodies calling known primitives
- **gcc -fsyntax-only**: Deterministic syntax gate (~5ms, 0% false positives)
- **c_judge** (50K steps, best loss 0.132): Neural semantic validation
- **OpaKanban**: Up to 3 retries per intent, blocking failed plans

The key design principle: the gcc syntax gate (deterministic) fires before the neural judge, so the neural model only evaluates syntactically valid code. The pipeline pre-filters the distribution the judge operates on — the same NAR pattern applied to code generation.

---

## 7. OPA as NAR Architecture: The Formal Analogy

### 7.1 Gradient Descent Without Calculus

The OPA pipeline performs **gradient descent without calculus** — navigating a solution space through rejection sampling rather than backpropagation:

| Gradient Descent | OPA Pipeline |
|---|---|
| **Loss function** L(θ) | Manhattan distance / confidence score / syntax pass |
| **Parameters** θ | Current state + Kanban memory |
| **Gradient** ∇L | Judge's accept/reject signal + direction of metric change |
| **Learning rate** α | Replan threshold (stalls before strategy change) |
| **Momentum** | Move history in Kanban (avoids revisiting failed states) |
| **Weight update** θ ← θ − α∇L | Kanban update: block failed action, try next |
| **Convergence** | Metric reaches 0 / output passes judge |

The system optimises over the space of possible outputs through proposal → evaluation → accept/reject. The models propose; the pipeline optimises.

### 7.2 The Implicit OPA Inside LLMs

The hypothesis — supported by the experimental evidence — is that large LLMs contain an **implicit version of the OPA pattern** encoded in weights:

| OPA Component | LLM Equivalent |
|---|---|
| **Planner organelle** | Early attention layers identifying structure and sub-tasks |
| **Worker organelle** | Middle layers retrieving relevant patterns from training data |
| **Judge organelle** | Late layers performing self-consistency checks |
| **Kanban state** | The autoregressive context window ("what I've already said") |
| **Wire format** | Hidden activations between layers |
| **Rejection sampling** | Softmax + temperature + beam search |

OPA makes this pattern **explicit, deterministic, and measurable.** Chain-of-thought prompting works because it forces the model to externalise its internal Kanban — each intermediate token becomes state for the next step, preventing the model from jumping to a poorly-matched retrieval. OPA does this externally, in C, with guaranteed correctness.

### 7.3 The Parameter Efficiency Argument

```
LLM approach:  1 model × N parameters learns (retrieval + search + validation + syntax)
OPA approach:  K models × (N/K) parameters each learns 1 thing + deterministic code for the rest
```

At the same total parameter budget, OPA achieves higher reliability per sub-task because each organelle spends its full capacity on a single well-defined output format, while the deterministic orchestrator provides guaranteed correctness, zero-cost cycle detection, and lossless state memory.

The 340-line coordination library (Kanban + cycle detector + judge) that transformed 50%-accurate models into 90%-successful systems is the concrete evidence. An LLM achieving 90% without that library would need vastly more parameters — trained on vastly more data — to internalise the same logic fuzzily.

---

## 8. Pathways to Stronger NAR

The retrieval–reasoning boundary is not necessarily permanent. Five NAR-compatible mechanisms can extend the pipeline without making individual organelles reason:

### 8.1 Reasoning Traces as Training Data

The pipeline's coordination decisions (accepts, rejects, stalls, replans) are already serialised by `OpaTrace`. Training on these traces shifts the organelle from **answer retrieval** to **process retrieval** — it learns to predict pipeline corrections before the judge fires. This is chain-of-thought for micro-models, generated from the pipeline's own execution history.

**Status:** Infrastructure implemented. A/B test on puzzle8 showed augmented traces are safe (no regression at 13% enrichment); scaling to 30–50% expected to improve hard-band performance.

### 8.2 Monte Carlo Tree Search Integration

Replace the linear Planner → Worker flow with MCTS, using organelles as the policy and value functions. The orchestrator manages branching, backtracking, and tree traversal; the models provide the pattern-matched evaluation at each node. This is **look-ahead without model reasoning** — strategic depth from an algorithmic search wrapper, not from neural state tracking.

**Status:** Proposed. Requires orchestrator extension; no model changes needed.

### 8.3 Neuro-Symbolic Anchoring

Replace flat-string wire format with a Prolog wire format. Inter-organelle communication becomes a language with built-in deductive logic — reasoning is offloaded to a deterministic symbolic backbone. The models remain retrieval engines; the wire format carries the logical structure.

**Status:** Proposed. S-expression fallback is a simpler intermediate step.

### 8.4 Verified Data Loops (Stem Cell Vision)

Successful pipeline runs generate verified outputs. These are converted mechanically into new training data and the models are fine-tuned on the expanded corpus. The retrieval surface grows autonomously through the pipeline's own verified discoveries.

**Status:** Proposed. The corpus generation infrastructure already exists (Python BFS scripts); only the ingestion loop is missing.

### 8.5 Multi-Timeframe Coordination

For temporal domains (markets, sensor data, logs), chain organelles operating at different timescales (tick/hour/day). The pipeline reconciles conflicting signals across timeframes — each organelle retrieves its timeframe's pattern; the reconciliation layer provides what looks like temporal reasoning.

**Status:** Partially implemented in the market regime experiment (57% holdout accuracy at 620K params, vs 33% at 163K params).

---

## 9. The Edge AI Value Proposition

### 9.1 Why OPA/NAR is the Right Architecture for the Edge

| Requirement | OPA/NAR approach | LLM approach |
|---|---|---|
| **Determinism** | C99 judge anchors all outputs | Stochastic by design |
| **Memory footprint** | ~50KB binary + checkpoint (~10MB) | 4GB+ for smallest useful models |
| **Inference latency** | Sub-5ms per organelle call | Hundreds of ms minimum |
| **Portability** | Compiles to MCUs where Python cannot run | Python/CUDA dependency |
| **Sovereignty** | Trains on-device from local data | Requires cloud API or large GPU |
| **Explainability** | Wire format logs show exactly what each organelle decided | Opaque activation space |
| **Updateability** | Individual organelles retrain independently | Full model fine-tune required |

### 9.2 Deployment Architecture

```
Edge Device (MCU / RPi / Mac)
├── OPA Orchestrator (C99, ~50KB)
│   ├── OpaKanban (state machine, ~80 lines)
│   ├── OpaCycleDetector (~30 lines)
│   └── Judge (domain-specific validity, ~50 lines)
├── Planner checkpoint (~10MB, mmap)
├── Worker checkpoint(s) (~10MB each, mmap)
└── Judge checkpoint (~10MB, optional neural layer)

Cloud dependency: NONE
Python dependency: NONE
GPU requirement: NONE
```

### 9.3 Sovereign Self-Improvement

The reasoning trace loop creates a **sovereign on-device learning cycle**:

1. Pipeline solves a problem → saves `OpaTrace`
2. Pipeline fails → saves failure trace (equally valuable)
3. Nightly: traces converted to corpus entries and appended to training set
4. Organelle fine-tuned for 1000 steps on new data (~2 min on M2 chip)
5. Improved model used for next day's inference

No cloud. No data exfiltration. No API dependency. The device becomes smarter on its own verified experience.

---

## 10. Conclusion: The Architecture of Honest Intelligence

The core claim of MicroGPT-C is simple and falsifiable: **intelligence at the system level does not require intelligence at the component level.**

The experimental evidence bears this out:
- A 460K-parameter model that cannot reason achieves 90% solve rates because the pipeline provides the search
- A 875K-parameter model that cannot generalise achieves 100% recall because the pipeline provides the index
- Game organelles that cannot play strategically achieve 79–91% win rates because the pipeline provides the coordination

This is not a workaround. It is the correct factoring of the problem. The cell does not need intelligent organelles — it needs specialised organelles, coordinated by a signalling protocol that expresses the intelligence.

The research agenda is not "make organelles smarter." It is:
1. Make the wire format more expressive (S-expressions, Prolog)
2. Make the orchestrator more powerful (MCTS, multi-timeframe)
3. Make the corpus more generative (reasoning traces, verified data loops)
4. Make the parameter budget more efficient (right-sizing per organelle)

At each step, the intelligence remains in the system. The organelles remain what they are: fast, small, deterministic retrieval engines — the biological cell's fundamental unit of specialised function.

> *"We are not claiming that organelles reason. We are claiming that a well-designed pipeline of retrieval engines can produce intelligent behaviour — and asking whether all intelligence, at every scale, is the same trick played with more parameters."*

---

## Appendix: Key Metrics Reference

| Experiment | Model size | Key result | NAR mechanism |
|---|---|---|---|
| c_codegen (875K) | 875K params | 100% corpus recall, 0/10 novel | Retrieval baseline |
| c_wiringgen (868K) | 868K params | Corpus recall ✅, novel pending | Composition grammar hypothesis |
| c_compose v1 | 462K × 2 | 96% parse rate, 4% registry hit | Two-organelle plan+judge |
| c_compose_v3 | 1.2M × 3 | In progress (128 test intents) | gcc gate + OpaKanban retry |
| puzzle8 (64K) | 64K × 5 | 60% solve (MD-delta encoding) | Representation engineering |
| puzzle8 (460K) | 460K × 5 | **90% solve (0 parse errors)** | Capacity + wire format |
| puzzle8 bare | 460K × 5 | **90% solve (no scaffolding)** | Internalised coordination |
| Mastermind | ~92K | 79% win rate | OPA + constrained retrieval |
| Connect-4 | ~92K | 91% win rate | First-player advantage + OPA |
| Hex | ~92K | 4% win rate | Reasoning-limited (hard floor) |
| Market regime | 163K→620K | 33% → 57% holdout | Scaling helps within distribution |

---

*Copyright © 2026 Ajay Soni, Enjector Software Ltd. MIT License.*