# The Retrieval–Reasoning Boundary

**Why organelles solve logic problems but cannot reason — and why that matters.**

---

## Spear Summary

**Point:** Organelles perform high-fidelity pattern retrieval, not reasoning. They can reproduce learned solutions to logic problems — which *looks* like solving them — but they cannot generalise to novel instances. Genuine novelty requires either reasoning from first principles (which sub-1M models lack capacity for) or compositional retrieval via the OPA pipeline.

**Picture:** An organelle is a librarian who knows exactly where every book is and can hand you the right one instantly — but cannot write a single novel sentence. The OPA pipeline is the editor who takes passages from many books and assembles them into something new. The librarian is essential; the editor provides the creativity.

**Proof:** The `c_codegen` organelle reproduces trained C functions with byte-perfect accuracy but scores **0/10 on novel prompts** — even trivial ones like "reverse an array." Mastermind organelles solve 78% of games via memorised `feedback → guess` mappings, but paraphrase the prompt and the output collapses to garbled tokens. The models memorise strings, not concepts.

**Push:** This boundary is a feature, not a bug. Organelles are designed as retrieval engines; the OPA pipeline provides compositional novelty. Document this distinction clearly in all claims about organelle intelligence.

---

## Latest: Phase 5 — Capacity Scaling Verdict

> **Phase 5 tests the "Assists-Off" hypothesis**: can a sufficiently large model solve puzzles without deterministic pipeline supports?

**Result: Yes — but it's a capacity story, not a reasoning story.**

The 460K-parameter model (N_EMBD=96, N_LAYER=4) achieves **90% solve rate with zero scaffolding** — matching the best result recorded with pipeline assists enabled. Key findings:

| Metric | 64K Assisted | 64K Bare | 460K Assisted | 460K Bare |
|---|---|---|---|---|
| **Solve Rate** | 20% (6/30) | 3% (1/30) | 90% (27/30) | **90% (27/30)** |
| Cycle Breaks | 181 | 0 | 23 | **0** |
| Success Drop | — | **-85%** | — | **0%** |

- **Internalized logic**: At 460K params the model handles the extended `valid=` prompts cleanly, producing correct output 100% of the time. Oscillation never occurs, so the cycle breaker has nothing to do.
- **Scaffolding is a capacity bridge**: At 64K params, disabling assists drops performance from 20% → 3%. The pipeline was compensating for the model's inability to parse its own inputs — not providing reasoning.
- **Remaining ceiling unchanged**: The same 3 hard puzzles (md=9–10) fail in every configuration — with scaffolding, without it, and at both scales. Those puzzles require moves that contradict the greedy heuristic. This is a **corpus coverage problem, not a reasoning problem**: the model has not seen training examples for those board positions. Solving them would require a richer training corpus or a deterministic search algorithm (BFS/A∗) integrated into the orchestrator — neither of which is reasoning.

### Updated Experiment Status

| Component | Status |
|---|---|
| Phase 4: Enriched-only A/B | ✅ 10% vs 90% (corpus too small — negative result) |
| Phase 4b: Combined A/B | ✅ 90% vs 90% (augmentation is safe, no regression) |
| **Phase 5: Capacity Scaling** | ✅ **90% bare = 90% assisted (scaffolding unnecessary at 460K)** |
| **Final Verdict** | ✅ Scaffolding removed; reasoning still absent; ceiling at 90% |

> **Key Insight**: At sufficient capacity, failure modes caused by undercapacity vanish, but the model's fundamental ceiling does not change. More parameters eliminate the need for runtime workarounds — they do not produce reasoning. Adding lookahead, BFS, or RL to the orchestrator would improve the **system's** solve rate, but that intelligence would live in the orchestrator, not the model.

---


## 1. The Distinction

| Capability | Retrieval (what organelles do) | Reasoning (what organelles lack) |
|---|---|---|
| **Mechanism** | Statistical pattern matching from training corpus | Deduction from first principles |
| **Strength** | Exact reproduction of known solutions | Generalisation to unseen problems |
| **Failure mode** | Collapses on novel/paraphrased inputs | N/A at this parameter scale |
| **Evidence** | 78% Mastermind solve, 0/10 novel codegen | Paraphrase failure, garbled token soup |
| **Analogy** | Neural grep — finds the right page instantly | Mathematician — derives the answer from axioms |

Reasoning requires building an internal chain of inference: *"if A then B, if B then C, therefore A implies C."* This demands working memory, compositional state tracking, and the ability to manipulate abstract representations — capabilities that emerge in models with billions of parameters but are absent at the sub-1M scale.

Retrieval, by contrast, requires only a high-fidelity mapping from input patterns to output patterns. A 92K-parameter model with 0.10 per-character cross-entropy loss has effectively memorised its corpus — it predicts each next character with ~90% confidence. This is powerful, but it is not reasoning.

---

## 2. Evidence: Organelles Retrieve, They Don't Reason

### 2.1 The c_codegen Experiment

The `c_codegen` organelle (875K params) was trained on 2,081 C functions (481 KB corpus). Results:

| Test type | Accuracy | Interpretation |
|---|---|---|
| **In-corpus prompts** | ~100% (byte-perfect) | The model memorised the training data |
| **Novel prompts** | **0/10** | The model cannot generalise at all |

Even trivially simple novel tasks — "reverse an array," "compute factorial" — produce garbled, unparseable output if those exact functions were not in the training corpus.

### 2.2 Paraphrase Failure

When a known prompt is reworded (e.g., "sort values" → "ascending sort"), the model fails or generates token soup. This proves:
- The model associates **exact string patterns** with outputs, not **concepts** with implementations
- There is no internal representation of "sorting" as an abstract operation
- The mapping is lexical, not semantic

### 2.3 Mastermind: Pattern Matching, Not Deduction

The Mastermind organelle solves 78% of games — dramatically above the 0% random baseline. But the mechanism is corpus retrieval:
- The model saw thousands of `board_state + feedback → next_guess` examples during training
- It reproduces the statistically best-matching guess for each feedback pattern
- It does not deduce which colours are correct by reasoning about the feedback logic

A human Mastermind player eliminates possibilities through logical inference: *"2 black pegs with ABCD, 1 black peg with ABCE → D is correct in its position."* The organelle does not perform this inference. It pattern-matches the feedback string to the closest training example and retrieves the associated guess.

### 2.4 The Capacity Argument

With a budget of ~800K parameters, the model must allocate capacity between:

```
Syntax:     Learning the grammar of the output format
Semantics:  Learning which patterns map to which outputs
Reasoning:  Maintaining internal state for multi-step inference
```

At this scale, syntax and semantics consume nearly all capacity. The internal state required for reasoning — tracking hypotheses, backtracking, maintaining a working memory of intermediate conclusions — simply does not fit. This is not a training failure; it is a fundamental capacity constraint.

---

## 3. Why This Matters

### 3.1 Honest Claims

The distinction between retrieval and reasoning is critical for making defensible claims about organelle intelligence:

| Defensible claim | Indefensible claim |
|---|---|
| "Organelles learn pattern-matched strategies from training corpora" | "Organelles reason about logic problems" |
| "The OPA pipeline achieves novel composition through coordination" | "Organelles generate novel solutions" |
| "Trained models outperform random baselines by 37–78 points" | "Small models can replace LLMs for reasoning tasks" |

### 3.2 The OPA Workaround

Because individual organelles cannot reason, the project achieves compositional novelty through the **Organelle Pipeline Architecture (OPA)**:

1. A **Wiring organelle** decomposes a novel intent into a sequence of *known* primitives
2. A **Code organelle** retrieves the implementation for each primitive
3. A **Judge** (deterministic or neural) validates the assembled output
4. **Feedback loops** correct errors through constrained retry

This is analogous to how a team of specialists with narrow expertise can solve problems that none of them could solve individually — not through individual reasoning, but through **coordinated retrieval**.

```
Single organelle:  Novel prompt → ??? → Garbled output
                   (retrieval fails — no matching pattern)

OPA pipeline:      Novel prompt → Decompose into known parts → Retrieve each → Assemble
                   (compositional retrieval succeeds)
```

### 3.3 The Biological Parallel

Real biological organelles do not reason either. A ribosome does not "understand" the protein it synthesises — it follows a mechanical read-decode-assemble process. Intelligence in biological cells emerges from the *coordination* of many non-reasoning components through structured signalling protocols.

MicroGPT-C organelles follow the same pattern: no single organelle is intelligent, but the pipeline — the cell — achieves intelligent behaviour through coordination.

---

## 4. Three Performance Tiers

The game experiments reveal three distinct tiers that map directly to the retrieval–reasoning boundary:

| Tier | Games | Performance | Why |
|---|---|---|---|
| **Coordination dominates** | Pentago (91%), Connect-4 (91%), Tic-Tac-Toe (86%), Mastermind (79%) | 79–91% | Finite, learnable pattern space; pipeline amplifies retrieval |
| **Right-sizing unlocks gains** | Sudoku (78%), Othello (67%), Klotski (62%) | 62–78% | Pattern space is larger; corpus coverage and model capacity matter |
| **Reasoning-limited** | Red Donkey (12%), Lights Out (10%), Hex (4%) | 4–12% | Requires spatial reasoning, connection logic, or algebraic inference that retrieval cannot approximate |

The bottom tier is the clearest proof that retrieval has limits. Hex (4% win rate) requires understanding *connection* — a spatial reasoning concept that cannot be reduced to local pattern matching. No amount of corpus expansion or parameter scaling within the sub-1M range will fix this; the game demands a capability the architecture does not possess.

---

## 5. Does Scaling Fix This?

A natural response is: *"what if you just made the model bigger?"* The experimental evidence shows that increasing parameter count improves **retrieval capacity** but does not produce reasoning. A larger model is a better librarian with a better memory — it does not spontaneously become a mathematician.

### 5.1 Scaling Improves Recall, Not Logic

In the `c_codegen` experiments, scaling from 142K to 875K parameters (6×) significantly improved corpus recall:

| Metric | 142K params | 875K params | Change |
|---|---|---|---|
| **In-corpus accuracy** | Partial | ~100% (byte-perfect) | Better lookup table |
| **Novel prompt accuracy** | 0/10 | **0/10** | No change |

The larger model memorised 2,081 C functions perfectly. But it still scored **0/10 on novel prompts** — the 6× parameter increase produced zero improvement in generalisation. More parameters resulted in a "better lookup table," not a "smarter programmer."

### 5.2 The Paraphrase Test Is Scale-Invariant

Regardless of model size, paraphrase failure persists:

- **Exact match**: `/* sort values in ascending order */` → perfect code (string was memorised)
- **Semantic variation**: `/* ascending sort */` → garbled token soup

If the model had learned the *concept* of sorting, both prompts would produce valid output. The failure proves the mapping is **lexical** (string → string), not **semantic** (concept → implementation). Scaling does not change this — a 10× larger lookup table still cannot answer questions that aren't in the table.

### 5.3 Reasoning Is a Structural Requirement, Not a Scale Threshold

The project's architecture analysis argues that reasoning is a function of **architecture and communication protocols**, not parameter density:

- **Syntactic vs. semantic budget**: Small models spend a large proportion of their parameter budget learning output syntax (C's 90+ character grammar, operator precedence, pointer notation). Scaling increases the total budget but does not change the proportion consumed by syntax.
- **The regular language advantage**: The flat-string wire format (`seq|normalize_z|fft_magnitude`) reduces the syntactic load to near zero, freeing capacity for semantic patterns — but the freed capacity goes to *more retrieval*, not reasoning.
- **Pipeline coordination**: Intelligence emerges from the OPA pipeline (Planner → Workers → Judge), not from any single model. Scaling one component does not substitute for the coordination of many.

### 5.4 The Loss Floor

Training logs show that models hit a convergence plateau regardless of capacity:

| Organelle | Start Loss | Best Loss | Reduction | Further training |
|---|---|---|---|---|
| Mastermind Planner | 3.68 | 0.08 | 45× | Diminishing returns |
| Connect-4 Player | 4.99 | 0.10 | 48× | Diminishing returns |

Once a model has memorised its training patterns (loss ~0.08–0.11), further training or moderate scaling yields marginal improvement. In the market regime experiments, scaling from 163K to 620K parameters improved holdout accuracy from 33% to 57% — but did not reach the 70% training accuracy, indicating that scaling helps generalisation only up to the limit of the learned patterns.

The loss floor is the mathematical signature of the retrieval–reasoning boundary: the model has extracted all learnable patterns from the corpus, and no amount of additional capacity will produce the internal state required for multi-step logical deduction.

### 5.5 LLMs as "Monolithic Simulators" — The Neural Operator Waste Hypothesis

> **The deeper question**: if organelles cannot do BFS, DFS, or Kanban internally, are LLMs actually doing these things — just wastefully, in fuzzy weights?

The answer appears to be yes. Large LLMs do build internal "neural operators" for search and coordination but they are brittle approximations of algorithms that can be expressed in deterministic code:

| What LLMs do internally | OPA equivalent | Cost in LLM params | Cost in OPA |
|---|---|---|---|
| Chain-of-thought as working memory | `OpaKanban` state struct | Millions of parameters | ~80 lines of C |
| Self-consistency sampling to avoid wrong answers | `organelle_generate()` ensemble vote | Temperature tuning | 3 inference calls |
| Retry on invalid output | `Judge` organelle + kanban replan | Implicit in RLHF | ~40 lines deterministic |
| A↔B oscillation detection | `OpaCycleDetector` | Emergent (unreliable) | ~30 lines of C |
| BFS-optimal path selection | `generate_corpus.py` + training | Memorised implicitly | 50-line Python BFS |

The MicroGPT-C experiments provide direct evidence of this waste:

- **The 340-line gap**: A ~340-line C coordination library (Kanban + cycle detector + judge) transformed 50%-accurate models into 90%-successful systems. An LLM achieving 90% accuracy without that library would need vastly more parameters — trained on vastly more data — to internalize the same logic fuzzily.
- **Chain-of-thought is makeshift Kanban**: When an LLM reasons step-by-step, it uses its context window as a stateful scratch pad — exactly what `OpaKanban` provides deterministically. The difference is reliability: the Kanban never forgets a blocked move; the LLM eventually does.
- **Wire format beats model size**: The project proved that the pipe-string wire format (`up|m=3,5,x,4|valid=up,right`) reduced syntactic load enough that a 460K-parameter model could match what a model ten times larger might achieve with free-form output.

#### Why This Matters Architecturally

LLMs are trained end-to-end to predict the next token. This means they must internally learn everything — syntax, semantics, search strategy, validity checking, backtracking — from the same gradient signal. There is no way for gradient descent to say "encode BFS here, encode validation there." The result is a single undifferentiated blob of fuzzy weights that approximately simulates all of these processes at once.

The OPA insight is that **separation of concerns is not just a software engineering preference — it is a capacity efficiency argument**:

```
LLM approach:  1 model × N parameters learns (retrieval + search + validation + syntax)
OPA approach:  K models × (N/K) parameters each learns 1 thing + deterministic code for the rest
```

At the same total parameter budget, OPA achieves higher reliability on each sub-task because each organelle spends its full capacity on a single well-defined output format, while the deterministic orchestrator provides what no amount of gradient descent can efficiently encode: guaranteed correctness, zero-cost cycle detection, and lossless state memory.

#### The Emergent Reasoning Hypothesis

This reframes how to think about LLM "reasoning":

- LLMs do not reason — they compress the *output patterns of reasoning processes* into weights
- Chain-of-thought prompting works because it forces the model to generate intermediate tokens that match the output patterns of reasoning (the "scratchpad" provides the structure that the model's weights approximate)
- Scaling helps because with more capacity, the model can compress more reasoning trace patterns — but this is still retrieval of reasoning outputs, not execution of reasoning algorithms

**The OPA verdict:** Rather than training a model to internalize BFS, train a small model to generate BFS-interpretable outputs, and wrap it in a BFS orchestrator. The model handles the fuzzy (what direction looks promising?) and the code handles the exact (is this move valid? have we visited this state?). This is not a workaround — it is the correct factoring of the problem.

---

## 6. Implications for the Project

### What Organelles Are Good For

- **Finite-pattern domains**: Games, codes, and structured outputs where the space of valid responses is bounded and learnable
- **Format-constrained generation**: Producing syntactically valid output (92–97% valid move rate) even when semantic accuracy varies
- **High-speed retrieval**: Sub-5ms inference on edge hardware for known-pattern lookups
- **Building blocks for composition**: As primitives in an OPA pipeline where the pipeline provides the "reasoning"

### What Organelles Cannot Do

- **Generalise to novel inputs** outside the training distribution
- **Perform multi-step logical inference** (chain-of-thought, hypothesis elimination)
- **Transfer learning across domains** — a Mastermind organelle knows nothing about Connect-4
- **Adapt to paraphrased or reformulated versions** of known problems

### The Design Philosophy

This is by design. Organelles are **retrieval engines** — fast, small, and reliable within their training distribution. The reasoning happens at the pipeline level, where deterministic orchestration, kanban state tracking, and feedback loops provide the coordination that individual models lack.

> *"No single organelle is intelligent, but the cell — the pipeline — is."*

---

## 7. The Elephant in the Room: Are LLMs Just Doing This Internally?

If intelligence in MicroGPT-C emerges from the *coordination* of retrieval-only components — not from any single model — then an uncomfortable question follows: **are large language models doing the same thing, just hidden inside a single monolithic weight matrix?**

### 7.1 The Neural Grep at Scale

The MicroGPT-C project characterises its sub-1M parameter models as "high-fidelity retrieval systems" — neural greps that map descriptions to memorised patterns with byte-level fidelity. What happens when you scale this mechanism by a million?

- A **1M-parameter organelle** memorises ~2,000 functions. Its retrieval is obvious and its limits are immediately visible.
- A **trillion-parameter LLM** is potentially a neural grep with a near-infinite library — trained on trillions of tokens spanning every domain, language, and format humans have produced.

The critical difference is not *mechanism* but *coverage*. Large LLMs appear to "reason" because their retrieval library is so vast that almost any "novel" prompt is actually a semantic neighbour of something they have already seen. The `c_codegen` experiments showed that increasing model size improved recall but did not spontaneously generate novel implementations (0/10 at 875K params). This suggests that what looks like "reasoning" in LLMs may often be **highly sophisticated pattern-matched retrieval** over an astronomically large training distribution.

### 7.2 The Implicit Pipeline Hypothesis

OPA achieves intelligence through explicit, separate components: Planner → Worker → Judge. The hypothesis is that **LLMs contain an implicit version of this same pipeline** built directly into their weights:

| OPA Component | LLM Equivalent |
|---|---|
| **Planner** (decomposes intent) | Early attention layers identifying structure and sub-tasks |
| **Worker** (retrieves implementations) | Middle layers activating relevant knowledge from compressed training data |
| **Judge** (validates output) | Late layers performing self-consistency checks |
| **Kanban state** (tracks progress) | The autoregressive context window — "what I've already said" |
| **Wire format** (structured communication) | Hidden activations between layers |
| **Rejection sampling** (accept/reject loop) | Internal token selection via softmax + temperature |

When an LLM generates a multi-step response, it is essentially maintaining its own **internal Kanban** — using its previous output tokens as the shared state for the next step of its internal planner. The context window *is* the Kanban: it tracks what has been said (`done`), what is being generated (`doing`), and implicitly constrains what should come next (`todo`).

The MicroGPT-C experiments proved that a ~340-line coordination library could turn 50%-accurate models into 90%-successful systems. Large LLMs may be performing this same coordination internally, where the "pipeline" is encoded in the weights rather than written in C.

### 7.3 Two Paths to the Same Destination

| Property | OPA (Decomposed) | LLMs (Monolithic) |
|---|---|---|
| **Where intelligence lives** | In the pipeline coordination | In the weight matrix (claimed) |
| **Retrieval** | Explicit — each organelle is a trained lookup | Implicit — attention heads retrieve from compressed training data |
| **Planning** | Explicit — a dedicated Planner organelle | Implicit — early layers may decompose intent |
| **Validation** | Explicit — a Judge organelle or deterministic validator | Implicit — later layers may self-correct |
| **Communication** | Structured flat strings between components | Hidden activations between layers |
| **Transparency** | Fully observable — you can inspect each stage | Opaque — "emergent reasoning" is unfalsifiable |

OPA makes the Planner → Worker → Judge pattern **explicit and deterministic**. Each component is separately trainable, inspectable, and replaceable. An LLM performs an analogous decomposition internally, but this is inferred from behaviour, not directly observed.

### 7.4 Evidence From LLM Behaviour

Several well-documented LLM failure modes are consistent with "sophisticated retrieval" rather than "true reasoning" — and they mirror organelle failure modes at a different scale:

| LLM failure | Retrieval explanation | Organelle parallel |
|---|---|---|
| **Hallucinations** | Interpolation between patterns produces plausible but incorrect output | Garbled token soup when `c_codegen` encounters novel prompts |
| **Sensitivity to prompt phrasing** | Different phrasings activate different retrieval paths | Paraphrase failure: "sort values" works, "ascending sort" produces garbage |
| **Reversal curse** | "A is B" doesn't imply "B is A" — a retrieval system only recalls the trained direction | Organelles only map in one direction (prompt → code, never code → prompt) |
| **Arithmetic failures on novel numbers** | The lookup table has gaps for unusual combinations | 0/10 novel accuracy despite 100% in-corpus accuracy |
| **Chain-of-thought sensitivity** | "Let's think step by step" improves results by forcing sequential retrieval | OPA's Kanban forces sequential move-by-move generation, improving from 50% to 90% |

The chain-of-thought parallel is especially revealing. When an LLM is prompted to "think step by step," it generates intermediate tokens that effectively *create its own Kanban state* — each step becomes context for the next, preventing the model from jumping to a poorly-matched retrieval. This is exactly what OPA's Kanban does externally: force sequential generation with state tracking to prevent fixation and oscillation.

### 7.5 What OPA Makes Visible

The value of OPA is not that it is "better" than an LLM — it is that it makes the retrieval–reasoning boundary **observable and measurable**:

- In OPA, you can measure exactly how much of the output came from model retrieval (92–97%) vs pipeline fallback (3–8%)
- In OPA, you can test novel vs in-corpus accuracy and get a clear answer (0/10 vs 100%)
- In OPA, the "wire format" makes inter-component communication inspectable — you can see exactly what the Planner asked the Worker to do

An LLM provides none of this visibility. When GPT-4 solves a logic puzzle, we cannot determine whether it *reasoned* about the puzzle or *retrieved* a solution to a similar puzzle from its training data. OPA's transparency is a research advantage: it proves that coordination of retrieval components can produce intelligent behaviour, and it measures exactly where that behaviour breaks down.

### 7.6 The Implication

If LLMs are performing an internal version of the OPA pattern — decomposing intent across layers, retrieving from compressed training data, and self-validating through later layers — then the difference between OPA and an LLM is not one of *kind* but of *scale and opacity*:

- **OPA** achieves coordination with ~5 explicit components at ~460K parameters each, communicating via parseable flat strings. The mechanism is transparent, the limits are measurable, and the system fits in 50 MB.
- **An LLM** achieves coordination with ~96 implicit layers at billions of parameters, communicating via opaque hidden states. The mechanism is theorised but not observed, the limits are discovered empirically, and the system requires gigabytes of RAM.

Both may be doing the same fundamental thing: **coordinated retrieval that approximates reasoning when the retrieval surface is dense enough**. OPA simply does it with the covers off.

| MicroGPT-C Concept | LLM Equivalent |
|---|---|
| **Neural grep** | Retrieval of patterns from a trillion-parameter library |
| **OPA pipeline** | An implicit, internal loop where layers act as Planner, Worker, and Judge |
| **Kanban state** | The autoregressive context window, tracking "what I've already said" |
| **Rejection sampling** | Token-level softmax selection + beam search + temperature |
| **Manhattan distance** | Perplexity / confidence — the scalar signal that guides generation |
| **Wire format** | Hidden layer activations — the "language" between internal components |

> *"We are not claiming that organelles reason. We are asking whether anyone does — or whether intelligence, at every scale, is retrieval all the way down."*

## 8. Gradient Descent Without Calculus

The previous sections establish that organelles retrieve and that the pipeline coordinates. But *how* does the pipeline coordinate? The answer reveals something unexpected: **OPA is performing gradient descent — without calculus**.

### 8.1 The Kanban as Optimizer

The OPA pipeline uses a shared **Kanban state string** as its central coordination mechanism. This state is purely deterministic — written in C, not learned — and it provides the "memory" that stateless organelles lack:

```
Kanban State: todo | doing | done | blocked
```

| Field | Role | Gradient descent analogy |
|---|---|---|
| **todo** | Remaining sub-tasks | Remaining loss to minimise |
| **doing** | Current proposed move | Current parameter update |
| **done** | Move history | Training history |
| **blocked** | Rejected moves | Negative gradients — directions that increase loss |

The Kanban prevents two critical failure modes:
- **Fixation**: Repeating a rejected move (stuck in a local minimum)
- **Oscillation**: Cycling between two states (bouncing between two minima)

By tracking what has been tried and failed, the Kanban narrows the search space on each iteration — exactly what a gradient optimizer does by following the steepest descent direction.

### 8.2 Manhattan Distance as Loss Function

In the 8-Puzzle experiments, the **Manhattan distance (md)** between the current board state and the goal state serves as the progress metric. The orchestrator monitors this metric and triggers adaptive behaviour:

- If md decreases → progress is being made → continue current strategy
- If md is unchanged for N moves → stall detected → re-invoke the Planner
- If md increases → regression → block the last move and try a different direction

This is a **loss function**. It maps the current state to a scalar value that indicates how far the system is from the goal. The pipeline then *optimises* this scalar — not by computing gradients through backpropagation, but by **rejection sampling**: propose a move, evaluate the loss, accept or reject.

### 8.3 Rejection Sampling as the Gradient Step

The pipeline's optimisation loop follows the classic rejection sampling pattern:

```
repeat:
    1. Planner proposes a direction          (sample from proposal distribution)
    2. Worker generates a specific move      (produce a candidate)
    3. Judge evaluates the move              (compute acceptance probability)
    4. If valid and progress ≥ threshold:
         accept → update Kanban → continue   (accept sample)
    5. If invalid or regressing:
         reject → block move → re-plan       (reject sample, narrow distribution)
until done or budget exhausted
```

This is **gradient descent with rejection sampling** — the system navigates a solution landscape by iteratively proposing, evaluating, and accepting-or-rejecting steps, without computing any derivatives.

### 8.4 The Formal Analogy

| Gradient Descent | OPA Pipeline |
|---|---|
| **Loss function** L(θ) | Manhattan distance, confidence score, or win/loss |
| **Parameters** θ | Current board state + Kanban memory |
| **Gradient** ∇L | Judge's accept/reject signal + direction of md change |
| **Learning rate** α | Replan threshold (how many stalls before changing strategy) |
| **Momentum** | Move history in Kanban (avoids revisiting failed states) |
| **Weight update** θ ← θ − α∇L | Kanban update: block failed move, try next best |
| **Convergence** | md reaches 0 (puzzle solved) or game won |

The key difference: gradient descent computes *exact* derivatives through the model's computational graph. OPA computes *approximate* descent directions through rejection — propose, evaluate, accept or reject. The result is the same: iterative progress toward a minimum (solution), guided by a scalar signal (loss / md).

### 8.5 Generalising the Progress Metric

Manhattan distance works for spatial puzzles, but the "gradient descent without calculus" pattern generalises to any domain with a measurable progress metric:

| Domain | Progress metric ("loss function") | What triggers re-planning |
|---|---|---|
| **8-Puzzle** | Manhattan distance to goal | md unchanged for 5 moves |
| **Mastermind** | Number of remaining possibilities | No black/white peg improvement |
| **Connect-4** | Longest connected chain | No chain growth for 3 moves |
| **Code generation** | Softmax confidence score | Confidence below 80% threshold |
| **Market regime** | Classification confidence | Confidence drop on holdout data |

In every case, the pattern is identical: a **scalar progress signal** drives an **iterative propose-evaluate-accept/reject loop** managed by a **deterministic orchestrator**. The models provide the proposals; the pipeline provides the optimisation.

### 8.6 Why This Matters

This framing resolves the apparent contradiction in the project: *how can a system built from non-reasoning retrieval components achieve 90–97% success rates on logic problems?*

The answer: the organelles don't solve the problem — they **propose candidates**. The pipeline **optimises** over those candidates using a deterministic feedback loop that is structurally equivalent to gradient descent. The "intelligence" is in the optimisation loop, not in the models.

This also explains the three performance tiers (§4):
- **Tier 1 (79–91%)**: The progress metric is clear and the solution landscape is smooth — gradient descent converges reliably
- **Tier 2 (62–78%)**: The landscape has local minima — the optimizer sometimes gets stuck despite cycle-breaking
- **Tier 3 (4–12%)**: The progress metric is poorly defined or the landscape is deceptive — gradient descent cannot navigate it

> *"OPA is not teaching models to reason. It is wrapping them in an optimisation loop that navigates solution space through rejection sampling — gradient descent without calculus."*

---

## 9. Pathways to Reasoning: Five Mechanisms That Could Close the Gap

The retrieval–reasoning boundary is not necessarily permanent. The OPA architecture suggests several mechanisms that could unlock reasoning-like behaviour without abandoning the retrieval foundation — by making the *pipeline* smarter rather than the individual models.

### 9.1 Neuro-Symbolic Anchoring

The current OPA uses delimited flat strings as its wire format. But the architecture explicitly proposes more expressive formats for tasks that require genuine logical structure:

- **Prolog wire format**: Replace the Judge's simple string validation with a **unification engine**. Organelles communicate in a language with built-in deductive logic — reasoning is offloaded to a deterministic symbolic backbone rather than demanded from neural models.
- **S-expression fallback**: For non-linear task decompositions, S-expressions allow the Planner to express nested dependencies (`(seq (call denoise) (par (call bandpass) (call highpass)))`), giving the system a "hierarchy of thought" absent in linear flat strings.

**What this unlocks**: Rule-based deduction and hierarchical planning without increasing model capacity. The models remain retrieval engines; the wire format carries the logical structure.

### 9.2 Ensemble Voting as System 2 Thinking

The `microgpt_organelle.h` library already implements ensemble voting — running N parallel inferences with temperature jitter and majority-voting the result. This can be extended into a dual-process architecture:

- **System 1 (fast)**: Single inference, high confidence → accept immediately. This is the retrieval path — pattern recognised, answer returned.
- **System 2 (slow)**: Confidence below threshold → trigger N-way ensemble vote, cycle detection, and Planner re-invocation. This is the deliberation path — multiple candidates generated and evaluated.

The confidence score (softmax entropy) acts as the **gate** between System 1 and System 2. When the model "knows" the answer (high confidence), it retrieves instantly. When it doesn't (low confidence), it switches to a slower, more deliberate process that approximates reasoning through collective evaluation.

**What this unlocks**: Noise filtering, hallucination reduction, and a measurable "thinking harder" mode triggered by uncertainty.

### 9.3 Synthetic Evolution via Verified Data Loops

The "stem cell" vision suggests that reasoning can emerge through **autonomous corpus expansion**:

```
1. Pipeline discovers a novel strategy (e.g., a new 8-puzzle sequence)
2. Judge verifies the strategy is correct
3. Verified sequence is mechanically converted into new training data
4. Model is fine-tuned on expanded corpus
5. Repeat — the retrieval surface grows with each verified discovery
```

This creates a **recursive self-improvement loop** where the retrieval system constantly expands its "mountain range of truth" based on its own verified successes. The model never reasons — but its corpus grows to cover an increasingly large portion of the problem space, effectively *learning to recall reasoning chains* by generating and verifying them.

**What this unlocks**: Progressive generalisation without architectural changes. The system "learns to reason" by accumulating verified examples of reasoning, then retrieving them.

### 9.4 Monte Carlo Tree Search Integration

The current pipeline uses a linear Planner → Player flow with rejection sampling. Replacing this with **MCTS** would unlock look-ahead reasoning:

```
Current (linear):
    Planner → Player → Judge → accept/reject → repeat

Proposed (MCTS):
    Root state
    ├── Organelle proposes move A → simulate → evaluate
    ├── Organelle proposes move B → simulate → evaluate
    └── Organelle proposes move C → simulate → evaluate
        ├── From C, propose D → simulate → evaluate
        └── From C, propose E → simulate → evaluate
    Select best path through tree
```

In this model, the organelles serve as the **policy function** (proposing moves) and the **value function** (evaluating board positions), while the OPA orchestrator manages the branching, backtracking, and tree traversal. The models remain retrieval-based; the search algorithm provides the strategic depth.

**What this unlocks**: Multi-step look-ahead, strategic planning, and the ability to evaluate consequences before committing — the closest analog to "thinking ahead" achievable without genuine neural reasoning.

### 9.5 Multi-Timeframe Coordination

For complex real-world domains (markets, sensor data, logs), reasoning about *change over time* can be achieved by chaining organelles that operate at different temporal scales:

| Organelle | Timeframe | Function |
|---|---|---|
| **Edge detector** | Short (ticks, seconds) | Detects transitions and anomalies |
| **Regime classifier** | Medium (hours, days) | Identifies the current operating mode |
| **Trend analyser** | Long (weeks, months) | Tracks macro-level direction |

By forcing the pipeline to **reconcile** short-term noise with long-term trends — each produced by a different retrieval-based organelle — the system approximates temporal reasoning. No single organelle reasons about time; the pipeline's reconciliation of conflicting timeframe signals produces behaviour that *looks like* temporal reasoning.

**What this unlocks**: Anomaly detection, regime change awareness, and the ability to distinguish signal from noise across multiple timeframes.

### 9.6 The Secret Sauce: Reasoning Traces as Training Data

The five mechanisms above improve the pipeline's *coordination logic*. But there may be one mechanism that matters more than all of them combined — and it's hiding in plain sight.

Right now, the pipeline generates a *solution* (a sequence of moves) and the Judge evaluates the *outcome* (win/loss, md=0 or not). But the pipeline throws away its most valuable artifact: **the trace of how it got there** — which moves were tried, which were rejected, why the Planner re-planned, and what the Kanban state looked like at each decision point.

That trace *is* a reasoning chain. Training on it is the **memory consolidation phase** of the OPA stem cell vision — internalising the pipeline's coordination logic into the models' weights.

#### The Coordination Gap

Current experiments show a massive gap between individual model accuracy (~50%) and pipeline system success (~90%). This gap is entirely filled by the deterministic coordination of the **Kanban** — approximately 340 lines of C code.

```
Step 1: Planner="up"    md: 12→11  accepted
Step 2: Planner="right" md: 11→10  accepted
Step 3: Planner="up"    md: 10→11  REJECTED → blocked[up]
Step 4: Planner="left"  md: 10→9   accepted
Step 5: stall (3 moves, no progress) → REPLAN triggered
Step 6: Planner="down"  md: 9→8    accepted (new strategy)
...
Step N: md=0 → SOLVED
```

- **The current state**: The pipeline uses `blocked[]`, `last[]`, and `stalls` to catch a model's errors and force corrections.
- **The proposal**: Train on the *trace* of these corrections, so the model learns to predict the correction before the Judge ever sees the move.
- **The effect**: The "intelligence" shifts from the 340-line C orchestrator into the neural weights themselves.

#### Solving Information-Theoretic Barriers

The Adaptive Organelle Planner identifies three barriers that stateless models cannot overcome: **fixation** (repeating rejected moves), **oscillation** (cycling between two states), and **non-monotonic blindness** (inability to accept temporary regression for long-term progress).

Training on traces addresses all three:

| Barrier | How traces solve it |
|---|---|
| **Fixation** | Training on `board_state + blocked:right → move:up` teaches the model the *concept* of rejection — it learns to avoid blocked directions without the Judge |
| **Oscillation** | Traces include cycle-breaking events — the model learns to recognise oscillation patterns and pre-emptively choose a third alternative |
| **Non-monotonic blindness** | Traces that show md temporarily increasing before decreasing teach the model that regression can precede progress — the hardest lesson for a retrieval system |

The model stops learning "the answer" and starts learning "how to find the answer when the obvious answer fails." This is exactly the concept normalisation capability described as the Planner's core skill.

#### Decomposition as Reasoning

The OPA Planner already generates reasoning traces when it decomposes intents. When the Planner decomposes "denoise and downsample" into `todo:lowpass,downsample`, it is performing a reasoning step. Training on these decompositions — especially those that required a re-plan after a low-confidence failure — creates a recursive self-improvement loop. The model learns to synonym-map messy, novel intents into verified primitive chains, effectively "understanding" the intent by recalling how similar intents were successfully decomposed in the past.

#### Data Loops vs Process Loops

| Feature | §9.3 Verified Data Loop | Reasoning Trace Proposal |
|---|---|---|
| **Input** | `board_state` | `board_state + rejection_history + stalls` |
| **Output** | Verified best move | Best move + adaptation strategy |
| **Learning type** | Factual (result-oriented) | Procedural (process-oriented) |
| **Emergence** | Larger lookup table | Emergent reasoning patterns |
| **Model becomes** | A better librarian | A librarian who keeps a diary |

#### Why This Is the Secret Sauce

1. **Chain-of-thought for free**: The reasoning trace is literally a chain of thought. Training on traces teaches the organelle to generate its own chain-of-thought by recalling chains that previously led to successful outcomes.

2. **The Kanban already generates it**: No new infrastructure is needed. Every game demo already tracks `blocked[]`, move history, stall counts, and replan triggers. The data exists — it just needs to be serialised into corpus format instead of discarded.

3. **It's self-improving**: Each successful solve generates one new reasoning trace. Train on it, and the model gets better at navigating similar situations. Which generates more successful solves. Which generates more traces. This is the recursive loop from §9.3 (Verified Data Loops), but applied to *process knowledge* rather than *factual knowledge*.

4. **It explains why chain-of-thought works in LLMs**: LLMs that "think step by step" are retrieving serialised reasoning traces from their training data — Stack Overflow debugging logs, textbook worked examples, step-by-step proofs. OPA would do the same thing, but with traces it generated and verified itself.

5. **It's still retrieval — and that's fine**: The organelle would be *recalling* reasoning patterns, not *performing* reasoning from first principles. But if you've accumulated enough reasoning traces for a domain, recalling the right one is functionally equivalent to reasoning. This is arguably what humans call "intuition" — pattern-matching against previous problem-solving experiences. The first time is hard. The hundredth time is instant.

#### The Sovereign Steward

Training on reasoning traces allows the organelle to evolve from a **Worker** (executing moves) to a **Sovereign Steward** of the process. The organelle essentially learns to *simulate the pipeline internally* — it "thinks" about what the Judge would reject and adjusts its proposal accordingly, before the Judge ever sees it.

This is the path from "intelligent system with dumb components" to "intelligent system with components that have internalised the system's intelligence." The pipeline bootstraps the reasoning; the organelle memorises it; future inference replays the reasoning without needing the full pipeline overhead.

**What this unlocks**: The organelle shortcutts the pipeline by recalling how the pipeline *would have thought* — internalising the Kanban's rejection logic, the stall detection, and the re-planning strategy into its own weights. The pipeline builds the reasoning traces; the organelle memorises them; future inference replays them at retrieval speed.

> *"The secret sauce isn't making organelles smarter. It's making them remember how the pipeline thinks — and then letting them shortcut the pipeline by recalling those thoughts directly. You aren't just making the librarian smarter; you're giving the librarian a diary of every time they successfully found a difficult book."*

### 9.7 Summary: Reasoning Catalysts

| Mechanism | Reasoning analogue | Implementation status |
|---|---|---|
| **Prolog wire format** | Deductive, rule-based logic | ❌ Proposed (Phase 4) |
| **Ensemble voting** | System 2 consensus and noise filtering | 🟡 Implemented (basic); gate logic proposed |
| **Verified data loops** | Learning from successful experience | ❌ Proposed (stem cell vision) |
| **MCTS orchestration** | Strategic look-ahead and backtracking | ❌ Proposed (game experiments) |
| **Multi-timeframe coordination** | Temporal reasoning via reconciliation | 🟡 Partial (market regime experiment) |
| **Reasoning traces as training data** | Chain-of-thought via process recall | 🟡 Capture infrastructure implemented |

None of these mechanisms require the organelles themselves to reason. In every case, the "reasoning" is a property of the **pipeline's coordination logic** — the models remain fast, small, and retrieval-based. The intelligence continues to live in the system, not the components. But §9.6 suggests a path where the organelle can *internalise* the system's intelligence by remembering how it operated — collapsing the pipeline into a single, faster retrieval step.

> *"A ribosome cannot reason about the protein it builds. But a cell — with hundreds of ribosomes coordinated by signalling cascades — can adapt, respond, and survive. The roadmap is not to make smarter organelles. It is to build a smarter cell — and then teach the organelles to remember what a smart cell looks like."*

### 9.8 Implementation: The OpaTrace Recorder

The reasoning trace concept described in §9.6 is now backed by a concrete implementation in the OPA core library. This section documents the hypothesis, approach, API design, and experimental methodology.

#### Hypothesis

> **If the pipeline's coordination decisions (accepts, rejects, stalls, replans, cycle-breaks) are serialised as structured traces and fed back as training data, then organelles will learn to predict the pipeline's corrections — shifting "intelligence" from the deterministic orchestrator into the neural weights.**

Specifically, we predict:

1. **Trace-trained models will require fewer pipeline interventions** — fewer rejects, fewer stalls, fewer replans — because they will have internalised the rejection logic.
2. **Trace-trained models will handle non-monotonic situations better** — they will learn to accept temporary metric regression (detours) by recalling traces where regression preceded recovery.
3. **The improvement will be measurable as a reduction in the coordination gap** — the ~50%→90% gap between individual model accuracy and pipeline system success should narrow.

#### The Missing Link

Reasoning Traces as Training Data represent the most compelling "missing link" for the Organelle Pipeline Architecture. They shift the learning paradigm from **Result-Oriented Retrieval** (what is the answer?) to **Process-Oriented Retrieval** (how do I find the answer when my first guess fails?) — arguably the most viable path to emergent reasoning in sub-1M parameter models.

This is not a speculative leap. It is a direct consequence of the OPA's existing design: the Kanban already generates the data, the corpus loader already knows the format, and the training pipeline already knows how to learn from it. The only missing piece was serialisation — and that is now implemented.

#### Why This Approach Is Effective

**1. Internalises Coordination Logic.** Currently, the "intelligence" of the system is split between the neural worker (which guesses) and the deterministic C orchestrator (which corrects). Training on traces allows the model to internalise those corrections, effectively learning to *predict* the orchestrator's behaviour. The model doesn't need to reason — it needs to remember what the orchestrator would have done and shortcut it.

**2. Solves Information-Theoretic Barriers.** Traditional training fails at **Fixation** (repeating the same wrong answer) and **Oscillation** (toggling between two wrong answers) because the models are stateless. By including `rejection_history` and `stall_counts` in the training data, the trace provides the model with temporal context — the information it needs to "reason" its way out of a loop, without actually reasoning.

**3. Chain-of-Thought for Micro-Models.** While LLMs use millions of parameters to maintain a hidden chain-of-thought, this approach leverages the **Kanban state** to make that chain explicit and learnable. The model learns patterns like *"when stalled for 3 moves with `up` blocked, try a lateral move"* as a retrieved behavioural pattern — not as a deduced strategy, but as a recalled experience. The chain-of-thought is not hidden in the weights; it is literally written in the trace.

**4. Low Implementation Overhead.** The OPA is already designed to capture this data via `OpaKanban` (blocked actions, stall counts, replan triggers) and `OpaCycleDetector` (oscillation patterns). Serialising this into a corpus — `board + blocked + last → next_move + why` — is a high-leverage move that requires minimal new infrastructure. The `OpaTrace` recorder adds ~220 lines to the library; integration into any experiment takes ~10 lines.

#### Retrieval vs. Reasoning: The Critical Distinction

The model is still performing retrieval — it has not learned to reason from first principles. But it is now **retrieving a process** rather than just a fact. This is a qualitative shift in what the retrieval engine can do:

| | Traditional Training | Reasoning Trace Training |
|---|---|---|
| **Input** | Board state | Board + rejection history + stalls |
| **Output** | Best move | Next move + adaptation strategy |
| **Learning question** | *"What is the answer?"* | *"How do I find the answer if my first guess fails?"* |
| **Failure mode** | Repeats same wrong move | Recalls that this move was already rejected |
| **Detour capability** | None (greedy only) | Can recall non-monotonic recovery patterns |
| **Model capacity needed** | Maps states → moves | Maps *trajectories* → moves |

The distinction matters because it determines the ceiling. A result-trained model can never exceed the quality of its training corpus — it can only reproduce answers it has seen. A trace-trained model can potentially *compose* behaviours from partial traces, combining "when stalled, try lateral" with "when blocked, try detour" to navigate situations no single trace covers entirely. This is not reasoning; it is **compositional retrieval** — and it may be sufficient.

> *"The gap between retrieval and reasoning is not binary. There is a spectrum: retrieving facts, retrieving procedures, retrieving strategies, and finally, composing novel strategies from retrieved fragments. Reasoning traces push the organelle from the first rung to the third. The fourth rung — genuine composition — is data-driven MCTS (§9.4). But for a 92K-parameter model trained on a 200KB corpus, the third rung is already remarkable."*

#### What: The OpaTrace API

The implementation lives in the core OPA library (`microgpt_organelle.h` / `microgpt_organelle.c`) so that every experiment can use it without duplication.

**Data structures:**

```c
/* Five possible outcomes for each pipeline step */
typedef enum {
  OPA_STEP_ACCEPTED,    /* move valid, made progress */
  OPA_STEP_REJECTED,    /* move invalid or out-of-bounds */
  OPA_STEP_STALL,       /* move valid but no progress */
  OPA_STEP_REPLAN,      /* stall threshold reached, planner re-invoked */
  OPA_STEP_CYCLE_BREAK  /* oscillation detected and broken */
} OpaStepOutcome;

/* One step in a reasoning trace */
typedef struct {
  int step;                   /* 1-indexed step number */
  char action[16];            /* proposed action (e.g. "up", "ABCD") */
  OpaStepOutcome outcome;     /* what happened */
  int metric_before;          /* progress metric before this step */
  int metric_after;           /* progress metric after (-1 if rejected) */
  char blocked_snapshot[64];  /* kanban blocked[] at decision time */
  int from_model;             /* 1 = model-sourced, 0 = fallback */
} OpaTraceStep;

/* Complete trace for one pipeline run */
typedef struct {
  OpaTraceStep steps[OPA_TRACE_MAX_STEPS]; /* max 64 steps */
  int num_steps;
  int initial_metric;
  int final_metric;
  int solved;
} OpaTrace;
```

**API functions:**

| Function | Purpose |
|---|---|
| `opa_trace_init()` | Initialise a trace with the starting metric |
| `opa_trace_record()` | Record one pipeline step (action, outcome, metrics, blocked state) |
| `opa_trace_finalise()` | Mark trace as complete (final metric, solved flag) |
| `opa_trace_to_corpus()` | Serialise to pipe-delimited corpus text |
| `opa_trace_write()` | Append serialised trace to a file |
| `opa_trace_count()` | Count steps matching a given outcome |
| `opa_trace_has_recovery()` | Detect non-monotonic recovery (metric↑ then ↓) |

#### How: Corpus Serialisation Format

Each trace is serialised as a multi-line document with a header and one line per step:

```
TRACE|initial=12|final=0|solved=1|steps=6
1|up|accepted|12>11|none|model
2|right|accepted|11>10|none|model
3|up|rejected|10>-1|up|model
4|left|stall|10>10|none|fallback
5|down|replan|10>10|none|fallback
6|right|accepted|10>0|none|model
```

This format is:
- **Pipe-delimited** — consistent with the OPA flat-string protocol
- **Human-readable** — inspectable during debugging
- **Corpus-compatible** — can be loaded by `opa_load_docs_multiline()` for training
- **Compact** — typically 200–500 bytes per trace

#### Why: The Experimental Rationale

The experiment is designed to answer one question: **does training on process traces improve pipeline performance?**

The null hypothesis is that trace-trained models perform identically to result-trained models — that the additional context (rejection history, stall counts, blocked directions) provides no useful signal for the character-level model to learn from.

If the null hypothesis is rejected, the implications are significant:

| Outcome | If confirmed | Implication |
|---|---|---|
| Fewer pipeline interventions | The model internalised rejection logic | Intelligence is migrating from C to weights |
| Better detour handling | The model learned non-monotonic strategies | Process knowledge transfers across instances |
| Narrower coordination gap | Individual accuracy approaches system success | The pipeline is becoming redundant — the ultimate goal |

#### Experimental Methodology

##### Why the 8-Puzzle Is the Ideal Testbed

Not all game experiments are equal candidates for measuring reasoning trace impact. The 8-puzzle stands out on every dimension that matters:

| Game | Progress Metric | Non-Monotonic? | Pipeline Richness | Trace Value |
|---|---|---|---|---|
| **8-Puzzle** | Manhattan distance (continuous, per-move) | ✅ Detours required for hard puzzles | 5 organelles, cycle detection, stalls, replans | **Highest** |
| Mastermind | Black/white pegs (discrete, per-guess) | ❌ Always converging | 2 organelles, simpler loop | Medium |
| Connect-4 | Win/loss (binary) | ❌ No per-move progress metric | 2 organelles | Low |
| Tic-Tac-Toe | Win/loss (binary) | ❌ No per-move progress metric | 2 organelles | Low |
| Hex/Othello | Board score (coarse) | Partially | 2 organelles | Low–Medium |

**Why specifically:**

1. **Manhattan distance is continuous and per-step.** Unlike binary win/loss games, the 8-puzzle provides a progress signal at *every single move*. This gives traces maximum information density — each step tells the model whether it moved toward or away from the goal, by exactly how much.

2. **Hard puzzles require non-monotonic recovery.** Puzzles with manhattan distance ≥ 9 cannot be solved greedily — the solver must sometimes *increase* the metric (accept a temporary regression) to escape local minima. This is exactly the pattern §9.6 identifies as hardest to learn and most valuable to capture. If trace-trained models improve hard-band performance, it is direct evidence of non-monotonic reasoning from process recall.

3. **The richest pipeline in the project.** Five collaborating organelles (Strategist, Greedy-Mover, Detour-Detector, Detour-Mover, Judge), oscillation cycle detection, stall/replan logic, valid-move filtering, and fallback paths. More decision points = richer traces = more coordination knowledge to internalise.

4. **KPIs are already stratified by difficulty.** The experiment already reports solve rate, average moves, and pipeline interventions per band (easy md 1–4, medium md 5–8, hard md 9+). This gives an automatic control structure — if trace training helps the hard band without degrading the easy band, the signal is unambiguous.

5. **The hardest KPI is already identified.** Hard-band solve rate is the weakest metric in current results. If trace-trained models improve *specifically* on hard puzzles — where non-monotonic detours are essential — that is the strongest possible evidence that process knowledge is being internalised.

> *"The 8-puzzle isn't just convenient — it's the perfect diagnostic. Manhattan distance is the stethoscope: it measures the heartbeat of every decision. A monotonically decreasing trace is trivial. A trace that goes up, then down, then solves — that's a detour. That's the signature of reasoning. And if the model can learn to reproduce that signature from memory alone, we have our answer."*

##### Four-Phase Experiment

The experiment proceeds in four phases. Here is the concrete flow.

**Phase 1: Trace Capture** — Wire the `OpaTrace` recorder into `puzzle8_reasoning_demo`. Run 1,000+ puzzles across easy/medium/hard bands. At each pipeline step, record what happened. After each puzzle, append the trace to a file. *Status: Infrastructure implemented. Integration pending.*

A captured trace looks like this:

```
TRACE|initial=12|final=0|solved=1|steps=8
1|up|accepted|12>11|none|model
2|right|accepted|11>10|none|model
3|up|rejected|10>-1|up|model           ← blocked, pipeline corrected
4|left|stall|10>10|none|fallback       ← valid move but no progress
5|down|replan|10>10|none|fallback      ← stall threshold hit, planner re-invoked
6|right|accepted|10>11|none|model      ← regression (detour!)
7|down|cycle_break|11>9|up,left|model  ← oscillation detected and broken
8|left|accepted|9>0|none|model         ← solved!
```

Every trace is a complete record of the pipeline's decision-making: what was proposed, what happened, why it failed or succeeded, and what the Kanban state looked like at each step. The annotations on the right are implicit in the data — the model learns to associate them with outcomes.

**Phase 2: Corpus Generation** — Transform raw traces into training data. The key difference is what the model *sees as input*:

```
# Traditional corpus (what the Mover currently learns):
# Input: board state only → Output: direction
m=8,10,x,12|b=4|valid=up,down,right
down

# Trace-enriched corpus (what the trace-trained Mover would learn):
# Input: board state + pipeline context → Output: direction
m=8,10,x,12|b=4|valid=up,down,right|blocked=up|stalls=2|last=up,right
down
```

The enriched input includes **pipeline state**: what's blocked (`blocked=up`), how many consecutive failures (`stalls=2`), and recent move history (`last=up,right`). The model learns to associate this richer context with the correct action — not just "what move is best for this board?" but "what move is best given that I've already tried `up` and it was rejected twice?"

Filter criteria for corpus generation:
- Include successful solves (primary training signal)
- Include traces with non-monotonic recovery (most valuable for detour learning)
- Optionally include failed solves (teaches what *not* to do)
- Format: one trace = one corpus document, separated by blank lines

**Phase 3: Trace-Trained Model** — Train a new Mover organelle on the trace-enriched corpus instead of the standard `board_state → move` corpus. The model learns:
- `board_state + rejection_history + stalls → next_move` (process retrieval)
- Instead of: `board_state → next_move` (answer retrieval)

The architecture, hyperparameters, and training steps remain identical. Only the corpus changes. This isolates the effect of process knowledge.

**Phase 4: A/B Comparison** — Run the same 1,000 test puzzles with both models on identical random seeds:

| Variant | Source | Mover trained on |
|---|---|---|
| **Control** (`puzzle8_demo`) | Original experiment | `board → move` (answer retrieval) |
| **Experiment** (`puzzle8_reasoning_demo`) | Copied variant | `board + context → move` (process retrieval) |

Measured metrics:

| Metric | Result-trained (control) | Trace-trained (experiment) |
|---|---|---|
| Solve rate | Baseline | Expected ↑ |
| Avg moves per solve | Baseline | Expected ↓ |
| Pipeline interventions (rejects + stalls + replans) | Baseline | Expected ↓↓ |
| Non-monotonic solve rate (hard band) | Baseline | Expected ↑↑ |
| Model-sourced move % | Baseline | Expected ↑ |

**The hypothesis in one sentence:** the trace-trained Mover will propose fewer rejected and stalled moves because it has already "seen" which moves the pipeline would have rejected, and learned to avoid them — effectively shortcutting the orchestrator by recalling its corrections from memory.

#### Key Metrics

The primary metric is the **intervention ratio**: the number of pipeline interventions (rejects + stalls + replans) per successful solve. If trace-trained models reduce this ratio significantly, it demonstrates that process knowledge has been internalised.

The secondary metric is the **detour success rate**: the fraction of puzzles requiring non-monotonic moves (metric temporarily increases) that are solved. This directly measures whether the model learned from recovery traces — the hardest pattern for a retrieval system to acquire.

#### Integration Code

Any experiment can add trace capture with ~10 lines:

```c
OpaTrace trace;
opa_trace_init(&trace, manhattan_distance(board));

/* inside pipeline loop: */
OpaStepOutcome outcome = is_valid ? 
    (md_after < md_before ? OPA_STEP_ACCEPTED : OPA_STEP_STALL) :
    OPA_STEP_REJECTED;
opa_trace_record(&trace, dir, outcome, md_before, md_after, 
                 kb.blocked, from_model);

/* after loop: */
opa_trace_finalise(&trace, manhattan_distance(board), solved);
opa_trace_write(&trace, "puzzle8_traces.txt");
```

#### Phase 1 Results: Trace Capture

Phase 1 has been completed. The `puzzle8_reasoning_demo` successfully trained 5 organelles and solved 30 stratified test puzzles while capturing every pipeline decision.

##### Pipeline Performance (Baseline)

| Band | Solved | Rate | Avg Moves |
|---|---|---|---|
| Easy (md 1–4) | 10/10 | 100% | 2.9 |
| Medium (md 5–8) | 10/10 | 100% | 6.1 |
| Hard (md 9+) | 7/10 | 70% | 9.7 |
| **Overall** | **27/30** | **90%** | |

##### Trace Statistics

The experiment generated 30 traces totalling 377 lines in `puzzle8_reasoning_traces.txt`:

| Outcome | Count | Notes |
|---|---|---|
| Accepted (progress) | 278 | Model-sourced moves that improved md |
| Stall (no progress) | 61 | Valid moves that didn't improve best md |
| Cycle break | 23 | Oscillation detected and broken |
| Replan | 16 | Stall threshold hit, blocked cleared |
| Rejected (OOB) | 0 | No out-of-bounds moves in this run |
| **Recovery traces** | **8/30 (27%)** | Traces with non-monotonic md recovery |

Three puzzles were not solved (hard band), producing 50+ step traces that captured the failure patterns — these are valuable negative examples for future training.

##### Example: Captured Recovery Trace

This trace from a hard-band puzzle (initial md=9) shows the full lifecycle — including a cycle break at step 5 that temporarily regressed the metric (md 5→6), followed by successful recovery:

```
TRACE|initial=9|final=0|solved=1|steps=12
1|up|accepted|9>8|none|model
2|right|accepted|8>7|none|model
3|down|accepted|7>6|none|model
4|right|accepted|6>5|none|model
5|down|cycle_break|5>6|none|model    ← regression (md increased)
6|up|stall|5>6|none|fallback         ← fallback, no progress
7|down|accepted|6>5|none|model       ← recovery begins
8|down|accepted|5>4|none|model
9|left|accepted|4>3|none|model
10|up|accepted|3>2|none|model
11|right|accepted|2>1|none|model
12|down|accepted|1>0|none|model      ← solved!
```

This trace captures precisely the pattern that matters most: a temporary regression (step 5) followed by successful recovery (steps 7–12). A trace-trained model would learn to associate "cycle detected" with "accept the regression and navigate away" — the non-greedy reasoning pattern that human solvers use intuitively.

##### Key Observations

1. **27% of traces contain recovery patterns** — this is a substantial fraction, meaning the trace corpus has enough non-monotonic examples to teach detour behaviour.

2. **Zero rejected moves** — the valid-direction filter and ensemble voting are so effective that no out-of-bounds moves were proposed. The pipeline's "intelligence" manifests primarily through stalls and cycle breaks, not rejections.

3. **Cycle breaks are the dominant intervention** (23 events across 30 puzzles) — this confirms that oscillation breaking is the pipeline's most impactful coordination mechanism, and the one most worth internalising.

4. **Hard-band failures captured** — the 3 failed puzzles (md 8–10, 50+ steps each) provide negative examples showing extended stall-replan loops. These can teach the model what *not* to do.

#### Phase 2 Results: Enriched Corpus Generation

The Phase 1+2 combined run generated an enriched training corpus alongside traces. At each accepted model-sourced step, the pipeline wrote the mover's prompt enriched with stalls and move history:

```
# Enriched corpus entry (actual captured data):
m=3,3,3,1|b=4|valid=up,down,left,right|stalls=0|last=up
right
```

Compared to the standard corpus entry:
```
m=3,3,3,1|b=4|valid=up,down,left,right
right
```

**Corpus statistics:**

| Corpus | Docs | Lines | Vocab |
|---|---|---|---|
| Standard (`puzzle8_mover.txt`) | 1,686 | 5,057 | 30 |
| Enriched (`puzzle8_mover_enriched.txt`) | 255 | 765 | 35 |

The enriched corpus is significantly smaller because it only captures model-sourced accepted moves from 30 puzzle solves, while the standard corpus contains hand-crafted examples covering all board configurations.

#### Phase 3–4 Results: A/B Comparison

Both models were trained with identical architecture (96d/8h/4L, 461K params) for 25K steps and tested on 30 identical puzzles.

##### Head-to-Head Results

| Metric | Standard (Control) | Enriched-Only (Experiment) | Δ |
|---|---|---|---|
| **Solve rate** | **27/30 (90%)** | 3/30 (10%) | **−80pp** |
| Easy band | 10/10 (100%) | 2/10 (20%) | −80pp |
| Medium band | 10/10 (100%) | 1/10 (10%) | −90pp |
| Hard band | 7/10 (70%) | 0/10 (0%) | −70pp |
| Parse errors | 0 | **776** | ↑↑↑ |
| Cycle breaks | 23 | 217 | +194 |
| Pipeline time | 1.24s | 6.23s | +5× |

##### Analysis: Why the Enriched-Only Model Failed

The result is **negative but highly informative**. The enriched-only model catastrophically underperformed due to two compounding factors:

**1. Catastrophically small corpus.** The enriched corpus has only 255 training examples vs 1,686 in the standard corpus — an 85% reduction. The model simply hasn't seen enough board configurations to generalise. The 30-puzzle capture run generates far too few examples for a character-level model that needs pattern coverage.

**2. Prompt length exceeds context window.** The enriched prompts (`m=3,3,3,1|b=4|valid=up,down,left,right|stalls=0|last=up,right,down`) average ~60–80 characters — close to the 128-character `BLOCK_SIZE`. With the response included, many training examples overflow the context window, causing the model to learn truncated patterns. This directly explains the 776 parse errors — the model outputs garbled partial tokens.

**3. No standard-corpus foundation.** The enriched model was trained *only* on trace-enriched data, with no base of standard examples. It never learned the basic `md_delta → direction` mapping that the standard model masters first.

##### Key Insight: Augmentation, Not Replacement

The experiment definitively answers the question: **enriched traces cannot replace the standard corpus.** They must **augment** it. The correct approach for Phase 4b is:

1. **Concatenate** standard + enriched corpora into `puzzle8_mover_combined.txt`
2. **Increase context window** from 128 to 192+ to accommodate enriched prompts
3. **Run at scale** — capture 200+ puzzles to generate a corpus matching the standard in size
4. Re-run the A/B comparison with the combined corpus

> *"This negative result is actually the most valuable finding so far. It proves that process knowledge cannot replace factual knowledge — it can only supplement it. The model must first learn 'what move is good for this board' before it can learn 'what move is good for this board given what I've tried before.' Retrieval of process requires a foundation of retrieval of fact."*

#### Phase 4b Results: Combined Corpus (Standard + Enriched)

Following the Phase 4 negative result, we concatenated the standard and enriched corpora into `puzzle8_mover_combined.txt` (5,822 lines) and increased the context window from 128 to 192 to prevent truncation.

##### 3-Way Comparison

| Metric | Standard | Enriched-Only | **Combined** |
|---|---|---|---|
| Corpus | 1,686 docs | 255 docs | **1,940 docs** |
| **Solve rate** | **27/30 (90%)** | 3/30 (10%) | **27/30 (90%)** |
| Easy | 10/10 (100%) | 2/10 (20%) | **10/10 (100%)** |
| Medium | 10/10 (100%) | 1/10 (10%) | **10/10 (100%)** |
| Hard | 7/10 (70%) | 0/10 (0%) | **7/10 (70%)** |
| Parse errors | 0 | 776 | **0** |
| Cycle breaks | 23 | 217 | **23** |
| Pipeline time | 1.24s | 6.23s | **1.19s** |
| Context (BLOCK_SIZE) | 128 | 128 | **192** |

##### Analysis

The combined corpus **perfectly matches baseline performance** while incorporating process knowledge. Key takeaways:

1. **No regression.** The combined model solves exactly the same puzzles (27/30) with identical per-band rates, proving that augmenting the corpus with enriched trace data does not degrade the model's learned board→move mappings.

2. **Zero parse errors.** Increasing `BLOCK_SIZE` from 128 to 192 completely eliminates parse errors, confirming that the enriched-only model's catastrophic failure (776 parse errors) was caused by context window overflow, not by the enriched data itself.

3. **Identical cycle breaks (23).** The combined model triggers the oscillation breaker exactly as often as the standard model, meaning the enriched examples have not yet altered the model's move selection behaviour in a measurable way.

##### Research Conclusion

> **The enriched corpus is safe to deploy but not yet impactful.**

The current 255 enriched examples (from 30 puzzles) represent only 13% of the combined corpus. To produce measurable improvement in the hard band, the enriched fraction likely needs to be 30–50% — requiring 200–500 puzzle captures. The infrastructure for this is now fully operational.

The experiment has established three important findings:

1. **Replacment fails** — process-only corpus catastrophically underperforms (10% vs 90%)
2. **Augmentation is safe** — combined corpus preserves baseline performance exactly
3. **Scale is needed** — 13% enriched fraction is insufficient for measurable impact

The path forward is clear: increase the puzzle capture volume to grow the enriched corpus, then re-test to find the enrichment fraction that produces measurable improvement on hard-band detour solving.

#### Phase 5 Results: Scaffolding Removal (The Reasoning Test)

The ultimate test: can the model perform without pipeline scaffolding? We disabled three assist mechanisms — the cycle breaker, the blocked-direction tracker, and the replan logic — and re-ran both the standard and combined models "bare."

##### What Was Disabled

| Assist | Purpose | Effect When Disabled |
|---|---|---|
| **Cycle breaker** | Forces an unexplored direction when A↔B oscillation detected | Model must avoid oscillation on its own |
| **Blocked tracker** | Remembers failed directions, excludes them from prompts (`\|x=...`) | Model sees all directions every time, no memory of failures |
| **Replan logic** | Clears blocked list after 6 stalls, allowing fresh attempts | No replan — model must get it right without resets |

##### 4-Way Comparison

| Metric | Standard + Assists | Standard Bare | Combined + Assists | **Combined Bare** |
|---|---|---|---|---|
| **Solve rate** | **27/30 (90%)** | **27/30 (90%)** | **27/30 (90%)** | **27/30 (90%)** |
| Easy | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Medium | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Hard | 7/10 (70%) | 7/10 (70%) | 7/10 (70%) | 7/10 (70%) |
| Parse errors | 0 | 0 | 0 | 0 |
| Cycle breaks | 23 | **0** | 23 | **0** |
| OOB rejections | 0 | 0 | 0 | 0 |
| Total moves | 278 | **270** | 278 | **270** |
| Recovery traces | 8 (27%) | 4 (13%) | 8 (27%) | 4 (13%) |

##### Verification

The results were code-reviewed and verified:

- **Flag is compiled in**: Output confirms `Pipeline assists: DISABLED (bare model)` and `Assists: DISABLED` in both bare runs
- **Same trained model**: Bare targets reuse cached checkpoints from assisted runs (same `.ckpt` files). This is correct — we're testing the same model with different runtime configurations
- **Cycle breaker truly disabled**: 0 `CYCLE DETECTED` entries in bare output vs 23 in assisted output
- **No hidden assists**: 0 stalls, 0 OOB rejections in bare runs confirm no fallback paths are active

##### Analysis

> **The result is genuine, but smaller than it first appears.**

The 23 cycle breaks in the assisted run were **mostly futile**:

```
Assisted failures (3 hard puzzles):
  board=134726580  md 10→8  40 moves + 12 cycle breaks = 52 steps — STILL FAILED
  board=124056873  md  9→7  40 moves + 13 cycle breaks = 53 steps — STILL FAILED
  board=123708456  md 10→6  40 moves + 10 cycle breaks = 50 steps — STILL FAILED

Bare failures (same 3 puzzles, different paths):
  board=134726058  md 10→8  40 moves, 0 cycle breaks = 40 steps — FAILED
  board=124063758  md  9→7  40 moves, 0 cycle breaks = 40 steps — FAILED
  board=123476085  md 10→4  40 moves, 0 cycle breaks = 40 steps — FAILED
```

The cycle breaker fires 23 times in assisted mode but **doesn't save any puzzle** — the 3 failures are the same whether the scaffolding intervenes or not. The scaffolding was trying to fix a problem (oscillation) that the 460K model doesn't actually have.

##### Why the Model Doesn't Oscillate at 460K Params

At 64K params (BLOCK_SIZE=128, N_EMBD=48, N_LAYER=2), the model had ~64K parameters and a 128-char context window. The `valid=` prompt extension made prompts too long to parse (98% parse errors), which forced fallback to random valid directions, which caused oscillation, which needed the cycle breaker.

At 460K params (N_EMBD=96, N_LAYER=4), the model handles the extended prompts cleanly (0 parse errors), which means it produces valid directions from inference rather than fallback. Since the model was trained on greedy-optimal moves, its predictions naturally avoid loops — it learned to pick the direction that reduces manhattan distance, which inherently doesn't oscillate.

##### The 64K Contrast: Scaffolding as a Crutch
To confirm this, we tested the **64K parameter model** (N_EMBD=48, N_LAYER=2) on the same 30 puzzles:

| Metric | 64K Assisted | 64K Bare | 460K Assisted | 460K Bare |
| :--- | :--- | :--- | :--- | :--- |
| **Solving Rate** | **20% (6/30)** | **3% (1/30)** | **90% (27/30)** | **90% (27/30)** |
| **Cycle Breaks** | 181 | 0 | 23 | 0 |
| **Success Drop** | - | **-85%** | - | **0%** |

This proves that **scaffolding is a "capacity bridge"**. In the 64K model, the "bare" model oscillates constantly (firing the cycle breaker 181 times when enabled). Disabling the assists causes it to fail nearly every puzzle because it lacks the capacity to represent "visited states" or "optimal paths" in its weights.

At 460K, the model doesn't need the crutch. Its successful puzzles are solved with the same efficiency in bare mode as in assisted mode, and its failures are failures of basic search depth, not of silly oscillation loops.

**The scaffolding was solving a problem caused by insufficient capacity, not insufficient reasoning.**

##### Important Caveats

1. **Same 3 hard puzzles fail in all 4 configurations** — the model's ceiling is 90% regardless of scaffolding or corpus enrichment. The remaining 10% require moves that contradict the greedy heuristic (detours to escape local minima), which no amount of runtime assists can provide.

2. **The 270 vs 278 move difference** is because assisted runs waste moves on failed puzzles via cycle-break insertions that add extra steps without helping. The model's choices on *solvable* puzzles are identical.

3. **This does not prove emergent reasoning** — it proves that sufficient capacity eliminates the need for *runtime compensation for insufficient capacity*. The model has learned better pattern matching, not reasoning.

> **The model doesn't need scaffolding because it doesn't oscillate. It doesn't oscillate because it has enough capacity to parse the prompt correctly. This is a capacity story, not a reasoning story.**

#### Current Status

| Component | Status |
|---|---|
| `OpaTrace` API (structs, enum) | ✅ Implemented in `microgpt_organelle.h` |
| 7 API functions | ✅ Implemented in `microgpt_organelle.c` |
| 12 unit tests | ✅ All passing (49/49 total) |
| 8-puzzle integration | ✅ Wired into `puzzle8_reasoning_demo` |
| Phase 1: Trace capture | ✅ 30 traces, 377 lines, 8 recovery traces |
| Phase 2: Enriched corpus | ✅ 255 docs, 765 lines generated |
| Phase 3: Enriched-only model | ✅ Trained — loss 0.084 |
| Phase 4: Enriched-only A/B | ✅ 10% vs 90% (negative — corpus too small) |
| Phase 4b: Combined A/B | ✅ 90% vs 90% (no regression) |
| Phase 5: Scaffolding removal | ✅ **90% bare = 90% assisted (scaffolding unnecessary)** |

#### Conclusion

The results for the **8-Puzzle Reasoning Demo** indicate that while the initial experiment with reasoning traces is technically successful, it currently faces a data-scarcity bottleneck for the "enriched-only" model.

##### 1. Trace Capture & Corpus Generation

The experiment successfully implemented the `OpaTrace` API to record decision-making processes instead of just results.

- **Trace Capture Phase**: Collected **30 traces** totalling **377 lines** of reasoning data, including **8 specific recovery traces** where the model successfully navigated out of a stall.
- **Enriched Corpus**: Generated **255 documents** (765 lines) for training, mapping board states and rejection histories to adaptation strategies.
- **Training Convergence**: An enriched-only model achieved a training loss of **0.084**, indicating it successfully learned the patterns within the small trace dataset.

##### 2. A/B Comparison Results

The experiment compared models trained on traditional "best-move" data against those trained on "reasoning traces":

- **Enriched-only A/B (10% Solve Rate)**: This model performed poorly compared to the baseline (90%). The failure is attributed to the **corpus being too small** to cover the vast state-space of the 8-puzzle.
- **Combined A/B (90% Solve Rate)**: Augmenting the standard corpus with reasoning traces resulted in no regression (**90% vs. 90%**). This confirms that adding reasoning-process data is "safe" and does not corrupt the model's existing retrieval capabilities.

##### 3. Key Technical Metrics

The demo's implementation (`main.c`) tracks several new "reasoning" metrics to quantify the pipeline's adaptive behaviour:

- **Recovery Traces**: Measures the percentage of successful solves that required a detour or adaptation after an initial rejection.
- **Dispatch Tracking**: Differentiates between "Greedy" moves (standard MD-reduction) and "Detour" moves (adaptive reasoning).
- **Cycle Breaking**: Monitors how often the `OpaCycleDetector` forced an unexplored direction to escape an A↔B oscillation.

##### 4. Scaffolding Removal (Phase 5)

Code-reviewed and verified: disabling all pipeline assists (cycle breaker, blocked tracker, replan logic) causes **zero regression** — the model achieves 90% with 0 cycle breaks in bare mode. However, this is not evidence of emergent reasoning. The 23 cycle breaks in the assisted run were **futile** — all 23 fired on the same 3 hard puzzles that failed regardless. The scaffolding was compensating for capacity limitations (64K model couldn't parse extended prompts → random fallback → oscillation), not providing reasoning. At 460K params, the model parses prompts correctly, so it never oscillates, and the scaffolding has nothing to do.

##### 5. Verdict and Next Steps

- **Current Verdict**: The scaffolding is unnecessary at 460K params, but this is a **capacity story, not a reasoning story**. The model has learned better pattern matching — it does not reason.
- **Remaining gap**: The 3 unsolved hard puzzles (md=9–10) require moves that contradict the greedy heuristic — there are no training examples for those board positions. This is a **corpus gap, not a reasoning gap**. Integrating BFS or A∗ into the orchestrator would fill it, but that intelligence would live in the orchestrator, not the model. The organelle itself would still be a retrieval engine serving a smarter search system.
- **Key Insight**: At sufficient model capacity, failure modes caused by undercapacity vanish, but the model's ceiling (90%) is unchanged. More capacity doesn't produce reasoning — it just eliminates the need for runtime workarounds.

---

## 10. Related Documents

| Document | Relationship |
|---|---|
| [ORGANELLE_INTELLIGENCE.md](ORGANELLE_INTELLIGENCE.md) | Experimental proof that organelles learn (retrieval verification) |
| [ORGANELLE_PIPELINE.md](ORGANELLE_PIPELINE.md) | The OPA architecture that provides compositional novelty |
| [ORGANELLE_GAMES.md](ORGANELLE_GAMES.md) | Game leaderboard showing the three performance tiers |
| [ORGANELLE_WHY_LOGIC_GAMES.md](ORGANELLE_WHY_LOGIC_GAMES.md) | Why logic games are the right testbed for these capabilities |

---

*Copyright © 2026 Ajay Soni, Enjector Software Ltd. MIT License.*
