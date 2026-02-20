# Why Logic Games?

**Using deterministic games as a microscope for pipeline coordination.**

---

## Spear Summary

**Point:** Logic games are controlled laboratories — they have fixed rules, measurable outcomes, and exact right answers, making them ideal for isolating *what the pipeline adds* versus what any single model knows.

**Picture:** It's like testing a new car engine on a dyno before putting it on the road. You don't use the dyno because you care about dynos — you use it because it removes every variable except the one you're measuring.

**Proof:** Three games, three different coordination challenges — all solved by the same ~340-line shared library:

| Game | Challenge | Pipeline Result | What It Proves |
|------|-----------|----------------|----------------|
| 8-Puzzle | Sequential planning under constraints | 60% solve (100% easy, 50% med, 30% hard) | Kanban + cycle breaking rescues greedy search |
| Tic-Tac-Toe | Threat detection + adversarial play | 90% win+draw vs random | Planner→Player→Judge loop filters errors |
| Connect-4 | Deeper lookahead + column reasoning | 90% win despite 50% invalid moves | Pipeline turns a coin-flip model into a winner |

**Push:** These results validate the OPA *coordination protocol*, not the games themselves. The next step is applying the same pipeline to non-game domains where the same pattern holds: decompose → propose → validate → adapt.

---

## 1. Why Games and Not "Real" Tasks?

Logic games are **not the goal** — they are the **experimental apparatus**. We chose them because they provide properties that natural-language or real-world tasks lack:

| Property | Why It Matters |
|----------|---------------|
| **Deterministic rules** | No ambiguity in what's legal — we can count exact errors |
| **Measurable outcomes** | Win/lose/draw, solved/unsolved — no subjective evaluation |
| **Scalable difficulty** | Easy → medium → hard puzzles test generalisation directly |
| **Cheap ground truth** | BFS/minimax generate optimal corpora for free |
| **Fast iteration** | 100 games in < 1 second — thousands of experiments per hour |

This makes it possible to answer a precise question: **Does the coordination pipeline add measurable value beyond what the individual models provide?**

The answer, across all three games, is unambiguously yes.

---

## 2. What Is Being Observed

### 2.1 The Core Finding: Coordination Compensates for Weakness

The single most important observation is that **pipeline coordination transforms weak models into competent systems**. Each worker organelle (~64K parameters) is a retrieval system trained on a small corpus of optimal moves. Alone, it hallucinates on unseen boards. Wrapped in a kanban pipeline with a judge and planner, it wins.

```
   Individual model accuracy:    ~50% valid moves
   Pipeline system win rate:     ~90%
   
   The gap is entirely due to coordination.
```

### 2.2 Error Anatomy: Workers Are Retrievers, Not Reasoners

Workers don't "understand" game rules. They pattern-match against training examples. When the board state is familiar, they produce correct moves. When it's novel, they guess — and ~50% of guesses are illegal.

This is not randomness. It's **out-of-distribution failure** — the same failure mode that plagues all small language models. The pipeline's value is in catching and correcting these failures in real time, without retraining.

### 2.3 The Kanban Loop: Rejection Sampling with Memory

The pipeline implements a feedback loop:

```
  Worker proposes → Judge validates → Kanban records
       ↑                                    |
       └──── Planner replans with context ──┘
```

Key behaviours observed:
- **Blocked-action memory** prevents repeating rejected moves
- **Cycle detection** breaks A↔B oscillation patterns (critical in 8-puzzle)
- **Stall counting** triggers replanning after consecutive failures
- **History tracking** gives the planner context for better proposals

This is analogous to **gradient descent with rejection sampling**: the planner proposes directions, the judge evaluates loss, and the kanban adjusts the step — all without modifying the underlying models.

### 2.4 Difficulty Scaling Reveals Pipeline Limits

The 8-puzzle results show how pipeline effectiveness degrades gracefully:

| Difficulty | Optimal Moves | Solve Rate | Observation |
|------------|---------------|------------|-------------|
| Easy (≤5)  | 1–5           | 100%       | Retrieval alone suffices |
| Medium (6–12) | 6–12       | 50%        | Kanban rescues some failures |
| Hard (13+) | 13–20         | 30%        | Pipeline runs out of retry budget |

This tells us exactly where the pipeline stops adding value — when the search depth exceeds what rejection sampling can cover efficiently. This is a measurable, reproducible finding that guides the next research phase.

---

## 3. What the Games Teach Us About Non-Game Domains

Each game isolates a coordination pattern that maps directly to real-world applications:

| Game Pattern | Real-World Analogue |
|-------------|---------------------|
| **8-Puzzle:** Sequential state transformation with backtracking | Multi-step API orchestration, workflow engines |
| **Tic-Tac-Toe:** Threat detection + defensive prioritisation | Anomaly detection, security rule evaluation |
| **Connect-4:** High error rate rescued by validation loop | Any LLM-powered system that needs output filtering |

The shared library (`microgpt_organelle.c|h`) is already domain-agnostic — `OpaKanban`, `OpaCycleDetector`, and the pipe-string protocol don't know anything about games. They implement the coordination pattern. The game logic lives entirely in each experiment's `main.c`.

---

## 4. What's Next in the Research

### Phase 1: Reduce Worker Error Rate (Current Focus)

The 50% invalid-move rate is the pipeline's main bottleneck. Three approaches, in order of effort:

| Approach | Effort | Expected Impact |
|----------|--------|-----------------|
| **Heuristic pre-filtering** — constrain proposals to legal moves before inference | Low | 30–50% error reduction |
| **Augmented corpora** — add structural labels (threats, deltas) to training data | Medium | 20–40% error reduction |
| **Hybrid workers** — deterministic fallback when retrieval confidence is low | High | ~50% error reduction |

Pre-filtering is the immediate next step: pass `valid_moves=up,left` in the prompt so the worker can only propose from a legal subset. This is already supported by the pipe-string format.

> [!NOTE]
> Active research on this problem is tracked in [Reducing Invalid Moves Progress](../../journals/Reducing%20Invalid%20Moves%20Progress.md).

### Phase 2: New Game Domains (Validation)

To confirm that the pipeline generalises, test on games with different combinatorial profiles:

| Candidate | Why |
|-----------|-----|
| **Othello/Reversi** | Higher branching factor (~10–20), tests scalability |
| **Lights Out** | State-toggle puzzle, tests non-sequential reasoning |
| **Sokoban** | Box-pushing with irreversible moves, tests planning depth |

Each game reuses the *same shared library* with a new `main.c` — if the pattern holds, OPA is validated as a general coordination protocol.

### Phase 3: Non-Game Applications (Transfer)

Apply the kanban + judge + replanning loop to:
- **Structured output generation** — generate JSON/SQL with a validator judge
- **Multi-step tool use** — decompose → act → validate → adapt
- **Edge inference pipelines** — chain sensor-reading organelles with decision organelles

The hypothesis: any task that can be decomposed into *propose → validate → adapt* benefits from the same coordination infrastructure.

---

## 5. Summary

Logic games are the **dyno**, not the destination. They provide:

1. **Isolation** — fixed rules remove confounding variables
2. **Measurement** — exact error counts and win rates
3. **Reproducibility** — deterministic games give identical results across runs
4. **Scalability** — difficulty parameters test generalisation directly

The research finding so far: **a ~340-line coordination library turns 50%-accurate models into 90%-successful systems**. The next steps are reducing the error rate, validating on harder games, and transferring the pattern to non-game domains.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*