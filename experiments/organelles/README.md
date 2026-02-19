# Multi-Organelle Experiments

Sub-1M parameter Transformers that can't compose alone — but chain into pipelines that can.

---

## Spear Summary

**Point:** Single micro-models are retrieval systems not generators — but pipelines of specialised micro-models achieve what no single model can.

**Picture:** It's like having a team where each person only knows one thing. The librarian finds the book, the architect draws the plan, and the builder follows instructions. Alone they're useless at novel tasks — together they build things none of them could imagine.

**Proof:** The c_codegen model scores 0/10 on novel prompts despite byte-perfect recall of 2,081 known functions. But the 8-puzzle pipeline (3 organelles × 64K params each) solves 96.7% of test puzzles through kanban coordination.

**Push:** Run the experiments below to see the retrieval-vs-composition gap firsthand. Then read the design docs to understand how pipelines bridge it.

---

## Experiments

| Experiment | What It Tests | Key Result |
|-----------|---------------|------------|
| [Tic-Tac-Toe](tictactoe/) | Planner→Player→Judge pipeline | 82% win+draw vs random |
| [8-Puzzle](puzzle8/) | Kanban coordination + capacity scaling | 96.7% solve rate (29/30) |
| [Connect-4](connect4/) | Pipeline rescue of weak models | 85% wins despite 60% invalid moves |
| [C Code Generation](c_codegen/) | Retrieval fidelity + novel composition | 7/7 byte-perfect recall, 0/10 novel |
| [C Wiring Generation](c_wiringgen/) | Composition grammar hypothesis | Training in progress |

## Design Documents

Architecture and protocol docs that explain the theory behind these experiments:

| Document | Scope |
|----------|-------|
| [ORGANELLE_PIPELINE.md](../../docs/organelles/ORGANELLE_PIPELINE.md) | Wire format design — why pipe-separated flat strings beat free-form C |
| [ORGANELLE_PLANNER.md](../../docs/organelles/ORGANELLE_PLANNER.md) | Kanban coordination protocol — stateful adaptation for stateless models |
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

Each organelle shares the same ~875K-parameter architecture and C99 engine. Only
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