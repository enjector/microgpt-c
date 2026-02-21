# Multi-Organelle Experiments

Sub-1M parameter Transformers that can't compose alone — but chain into pipelines that can.

---

## Spear Summary

**Point:** Single micro-models are retrieval systems not generators — but pipelines of specialised micro-models achieve what no single model can.

**Picture:** It's like having a team where each person only knows one thing. The librarian finds the book, the architect draws the plan, and the builder follows instructions. Alone they're useless at novel tasks — together they build things none of them could imagine.

**Proof:** Across 16 experiments covering 11 game domains, 2 real-world signal tasks, and 3 code generation tasks, pipelines consistently elevate weak models. The 8-puzzle pipeline (5 organelles) solves 90% of unseen puzzles, Connect-4 wins 88% with zero invalid moves, and market regime detection achieves 57% accuracy (2.8× random baseline). The lottery experiment validates the architecture's integrity by hitting an irreducible entropy floor (loss ~0.50) on genuinely unpredictable data.

**Push:** Apply compact encoding patterns (from markets) and MD-delta heuristics (from puzzle8) to the lower-performing game experiments. Expand corpus sizes for Klotski and Red Donkey.

---

## Experiments

### Games & Puzzles

| Experiment | What It Tests | Key Result |
|-----------|---------------|------------|
| [Pentago](pentago/) | Place + twist adversarial game | **90% win** vs random — best adversarial result |
| [8-Puzzle](puzzle8/) | 5-organelle pipeline + cycle breaking | **90% solve** (100% easy, 100% med, 70% hard) |
| [Connect-4](connect4/) | Pipeline rescue + ensemble voting | **88% win** vs random, zero invalid moves |
| [Tic-Tac-Toe](tictactoe/) | Planner→Player→Judge pipeline | **87% win+draw** vs random |
| [Mastermind](mastermind/) | Feedback loops + sequential deduction | **86% solve**, 5.3 avg guesses, 0 invalid |
| [Sudoku 4×4](sudoku/) | Constraint satisfaction | **76% solve** — harder puzzles easier (inverse difficulty) |
| [Othello 6×6](othello/) | Adversarial positional play | **56% win** vs random |
| [Klotski](klotski/) | Multi-piece sliding blocks | **59% solve** — fallback-dominated (87% parse errors) |
| [Red Donkey](reddonkey/) | Asymmetric sliding blocks (2×2 donkey) | **30% solve** — corpus-limited (199 entries) |
| [Lights Out](lightsout/) | Toggle logic + coupled constraints | **12% solve** — coupled constraints too complex |
| [Hex 7×7](hex/) | Connectivity-based adversarial strategy | **10% win** — global path reasoning needed |

### Real-World Signal

| Experiment | What It Tests | Key Result |
|-----------|---------------|------------|
| [Market Regime Detection](markets/) | Cross-asset correlations, learnable signal | **57% accuracy** (2.8× random), loss 0.03–0.06 |
| [Lottery Prediction](lottery/) | Independent random events (negative control) | Loss floor **~0.50** — confirms no learnable signal |

### Code Generation

| Experiment | What It Tests | Key Result |
|-----------|---------------|------------|
| [C Code Composition](c_compose/) | Planner→Judge pipeline + LR scheduling | **83% exact match**, 98% parse, 1.2M params |
| [C Code Generation](c_codegen/) | Retrieval fidelity + novel composition | 7/7 byte-perfect recall, 0/10 novel |
| [C Wiring Generation](c_wiringgen/) | Composition grammar hypothesis | Training in progress |

### The Coordination Funnel

![The pipeline acts as a filter — half the model's moves are illegal but the system still wins 90% of games](../organelles/OPA.png)

**Point:** A model that's wrong half the time still wins 90% of games when wrapped in a coordination pipeline.

**Picture:** It's like a chess player who keeps trying to move pieces off the board. Instead of training a better player you hire a referee who says "nope — try again." The player is still bad but the *system* is smart.

**Proof:** Connect-4's player organelle produces ~50% invalid moves (609 out of ~1,200). The kanban pipeline catches every one and replans — resulting in an 88% win rate against a random opponent.

**Push:** Don't build a bigger model. Build a pipeline that filters a small model's mistakes. The shared organelle library (`microgpt_organelle.c|h`) gives you the kanban + judge + replanning loop in ~340 lines of C.

---

## Key Findings Across All Experiments

### 1. The Random Gap — Proof of Intelligence

| Game | Random Baseline | Trained Model | Gap |
|------|:-:|:-:|:-:|
| Mastermind | 0% solved | 78% solved | **+78 pts** |
| Connect-4 | 54% wins | 91% wins | **+37 pts** |

Trained models produce valid, parseable moves **92–97% of the time** — the intelligence comes from the neural model, not the pipeline filters.

### 2. Signal vs Entropy — The Architecture Knows the Difference

| Domain | Training Loss | Learnable? |
|--------|:-:|:-:|
| Markets (cross-asset) | **0.03–0.06** | ✅ Yes — regime persistence |
| Lottery (independent) | **0.50–0.61** | ❌ No — irreducible entropy |

The 10–12× loss difference between markets and lottery proves the architecture distinguishes genuine signal from noise.

### 3. Corpus Size Drives Performance

| Corpus Size | Example | Parse Errors | Solve/Win % |
|:-:|---|:-:|:-:|
| 20,000 | Sudoku | 184 | 76% |
| ~2,000 | Mastermind | 25 | 86% |
| 232 | Klotski | 1,849 | 59% |
| 199 | Red Donkey | 2,286 | 30% |

Below ~500 corpus entries, parse errors dominate and the fallback mechanism does the work.

### 4. Encoding Complexity Matters

| Encoding Type | Example | Performance |
|---|---|---|
| Single char (APL-style) | Markets regime (`R`, `I`) | 57% accuracy, 0.29s inference |
| 4-char code | Mastermind (`ABCD`) | 86% solve, 0 invalid |
| Cell coordinates | Othello (`R3C4`) | 56% win |
| Flat board string | Hex (49 chars) | 10% win, 50% parse errors |

Simpler output formats → fewer parse errors → better performance.

---

## Shared Library

All game/puzzle experiments share generic OPA infrastructure via [`microgpt_organelle.c|h`](../../src/microgpt_organelle.h):

| Component | Purpose |
|-----------|--------|
| `Organelle` struct | Bundles model + vocab + docs into one trainable/inferrable unit |
| `OpaKanban` | Blocked actions, move history, stall tracking |
| `OpaCycleDetector` | A↔B oscillation breaking |
| `organelle_train()` | Full training lifecycle with multi-threaded gradient accumulation |
| `organelle_generate()` | Temperature-controlled inference from prompt |

## Design Documents

Architecture and protocol docs that explain the theory behind these experiments:

| Document | Scope |
|----------|-------|
| [ORGANELLE_PIPELINE.md](../../docs/organelles/ORGANELLE_PIPELINE.md) | Wire format design — why pipe-separated flat strings beat free-form C |
| [ORGANELLE_PLANNER.md](../../docs/organelles/ORGANELLE_PLANNER.md) | Kanban coordination protocol — stateful adaptation for stateless models |
| [ORGANELLE_INTELLIGENCE.md](../../docs/organelles/ORGANELLE_INTELLIGENCE.md) | Proof of model intelligence — random gap + loss convergence evidence |
| [ORGANELLE_GAMES.md](../../docs/organelles/ORGANELLE_GAMES.md) | Game leaderboard + recommended next puzzles |
| [ORGANELLE_WHY_LOGIC_GAMES.md](../../docs/organelles/ORGANELLE_WHY_LOGIC_GAMES.md) | Why games are the experimental apparatus, not the goal |
| [ORGANELLE_VISION.md](../../docs/organelles/ORGANELLE_VISION.md) | CLI tooling design — `microgpt create/train/infer/pipeline` |

---

## The Multi-Organelle Ecosystem

### A Composable Framework for Autonomous Edge Intelligence

**Author:** Ajay Soni, Enjector Software Ltd.

**Date:** February 18, 2026

---

### 1. Executive Summary

Traditional Large Language Models (LLMs) operate as monolithic generalists, requiring massive infrastructure and high-latency protocols to solve narrow tasks. **MicroGPT-C** proposes a paradigm shift: **Specialized Micro-Intelligence**. By implementing a high-performance, C99-native Transformer engine with built-in training capabilities, we enable the creation of "Intelligent LEGO Blocks" (Organelles). These blocks are designed to differentiate, specialize, and evolve directly on the edge, providing a low-power, autonomous alternative to centralized AI.

### 1.1 The Composition Problem

Experimental evidence (see `OBSERVATION_C_CODEGEN.md`, Experiments 1–5) demonstrates
that models under 1M parameters are **high-fidelity retrieval systems, not code
generators**. A 875K-parameter model trained on 2,081 C functions achieves near-perfect
reproduction of known functions but scores **0/10 on genuinely novel compositions** —
even for trivially simple operations like `/* create array of square numbers */`.

The params-to-data ratio tells the story:

```
  Params/data    3.1:1       1.78:1         Both ratios
  ─────────      ─────────   ──────────     ──────────
  Behaviour:     Memoriser   Memoriser      NOT compositors
  Exact match:   ✅ Perfect   ✅ Perfect      Retrieval works
  Novel prompt:  ❌ Garbled   ❌ Garbled      Composition fails
```

**A single model cannot both understand intent AND generate code at this scale.**
This is why organelles exist — composition requires separation of concerns.

---

### 2. The Multi-Organelle Approach

To unlock "Novel Composition" — the ability to solve tasks the model has never seen —
we have developed a **Two-Organelle Architecture** that separates natural language
intent from algorithmic implementation.

#### A. The Wiring Organelle (The Architect)

This model is trained on **composition grammar** rather than implementation details.

* **Function:** Maps a prompt (e.g., "denoise and downsample") to a structural sequence of primitive function calls.
* **Generalization Hypothesis:** By focusing on the "grammar" of logic (e.g., "A then B", "compute X, use X to transform Y"), it is expected to recognize novel compositions that share structural similarities with its training corpus. This hypothesis is currently under test (see `OBSERVATION_C_WIRINGGEN.md`).
* **Corpus:** 864 domain-agnostic compositions across 8 categories (pipelines, two-pass, aggregation, windowed, conditional routing, etc.).

#### B. The Code Organelle (The Library)

This model acts as a high-speed retrieval engine for bit-perfect C implementations.

* **Function:** Provides the exact function bodies for primitives like `fft_radix2()`, `mean()`, or `zscore_normalize()`.
* **Reliability:** At step 40.5K / 50K (loss 0.038), corpus-matching prompts achieve near-perfect byte-level reproduction across all five domains (numerical, linear algebra, statistics, signal processing, technical analysis).
* **Quality Gate:** Softmax-based **Confidence Scoring** reliably separates known prompts (86–93% confidence) from unknown ones (38–58%), providing a built-in uncertainty filter for production use.

#### C. Concrete Pipeline Example

```
  User prompt: "compute z-scored rolling average"
           │
           ▼
  ┌──────────────────────────┐
  │  Wiring Organelle         │  Parses intent → known composition pattern
  │  Output:                  │  zscore_normalize() → rolling_mean()
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  Code Organelle           │  Retrieves function bodies (99% confidence)
  │  Output:                  │  void zscore_normalize(...) { mean(); stddev(); ... }
  │                           │  void rolling_mean(...) { for (...) ... }
  └──────────┬───────────────┘
             │
             ▼
  Complete, compilable C function with composition + implementation
```

---

### 3. Future Directions: The Autonomous Pipeline

The roadmap focuses on transitioning individual organelles into a self-evolving ecosystem.

#### A. Organelle Chaining Protocol (Q4 2026)

Formalizing the communication between organelles through a lightweight Inter-Process Communication (IPC) layer. This allows the **Wiring Organelle** to output a blueprint that is immediately populated by the **Code Organelle**.

#### B. Domain Organelles (Future Work)

A third tier will be introduced to handle domain-specific language, completing the
three-layer separation of concerns:

```
  Domain organelle:  "risk-adjusted return"  →  "normalize then compute ratio"
  Wiring organelle:  "normalize then ratio"  →  mean() / stddev() chain
  Code organelle:    mean() body, stddev() body
```

This separates **domain knowledge**, **composition grammar**, and **implementation
detail** into independently trainable, composable models. Each layer can be
retrained or swapped without affecting the others.

#### C. Federated Differentiation (Research Direction, 2027+)

Edge devices could contribute gradient updates to a shared organelle without sharing
raw data. This would enable an ecosystem where organelles learn from local environments
(e.g., specific sensor noise or user quirks) while maintaining privacy. This remains
a research direction pending the validation of organelle chaining in production
deployments.

---

### 4. Specialized Organelle Taxonomy

Beyond code generation, the MicroGPT-C engine can be differentiated into various
functional organelles:

| Organelle Type | Input | Output | Intelligence Task |
| --- | --- | --- | --- |
| **The Architect** | Natural language | Primitive call sequence | Composition and wiring |
| **The Library** | Function name/comment | C function body | High-fidelity code retrieval |
| **The Validator** | Data patterns | "Valid" / "Invalid" | Pattern-based classification |
| **The Editor** | Raw strings | Corrected text | Semantic character-level correction |
| **The Forecaster** | Time-series | Predicted value | Time-series forecasting for sensor streams |
| **The Gatekeeper** | Decision logits | Confidence score | Autonomous safety and triage layer |

Each organelle shares the same ~64K-parameter architecture and C99 engine. Only
the training corpus and inference protocol differ — demonstrating that intelligence
is a function of **data**, not architecture, at this scale.

---

### 5. Conclusion

The future of AI isn't just "bigger." It is **faster, smaller, and more autonomous**. The Multi-Organelle Ecosystem provides the C99 baseline for this future — a world where intelligence is a composable, low-power, and self-evolving component of every device we touch.

The experimental evidence is clear: single micro-models are excellent retrieval
systems but cannot compose. By separating intent decomposition from code retrieval,
the multi-organelle architecture transforms a fundamental limitation into a design
principle — each organelle does one thing well, and their composition achieves what
no single model can.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*