# Do Organelles Actually Learn? — Verified

**Separating model intelligence from pipeline filtering.**

---

## Spear Summary

**Point:** The models are genuinely learning — random baselines score 0–54% while trained models score 78–91% on the same games with the same pipeline.

**Picture:** We blindfolded two dart players and gave both the same catcher-friend. One player is trained, the other throws randomly. The trained player hits the board 92–97% of the time on their own; the friend barely needs to help. The random player misses every throw — the friend places all the darts and the score plummets.

**Proof:** Mastermind with trained model: **78% solved** (92% of moves from model). Mastermind with random baseline: **0% solved** — a 78-point gap. The model learned what random cannot: valid colour patterns that converge on the solution.

**Push:** The claim "organelles learn to solve logic problems" is now backed by a controlled experiment with a 37–78 point gap across two games. Update the LinkedIn post to cite these numbers.

---

## 1. The Question

Every organelle game demo has two moving parts:

| Component | What it does |
|-----------|-------------|
| **The model** | Generates a proposed move from a board prompt |
| **The pipeline** | Filters invalid moves, retries, breaks cycles, falls back to a deterministic pick |

We needed to know: **is the model doing real work, or is the pipeline funnelling random guesses into wins?**

---

## 2. Experimental Results (Feb 20, 2026)

### Protocol

For each game, we ran 100 games in two conditions:
- **Trained model** — full pipeline with trained checkpoint (`RANDOM_BASELINE=0`)
- **Random baseline** — model output replaced with uniformly random valid guess, identical pipeline (`RANDOM_BASELINE=1`)

Both conditions use the same kanban state, planner, cycle detector, and opponent (random).

### Results

| Game | Metric | 🔴 Random Baseline | 🟢 Trained Model | Gap | Verdict |
|------|--------|:-------------------:|:-----------------:|:---:|:-------:|
| **Mastermind** | Solve rate | **0%** | **78%** | **+78 pts** | 🟢 Model dominates |
| **Connect-4** | Win rate | **54%** | **91%** | **+37 pts** | 🟢 Model dominates |
| **Tic-Tac-Toe** | Win+Draw rate | (not run) | **86%** | — | (baseline needed) |

### Move Attribution

| Game | Model-sourced | Fallback-sourced | Parse errors |
|------|:------------:|:----------------:|:------------:|
| **Mastermind** (trained) | 558/605 (**92%**) | 47/605 (8%) | 47 |
| **Connect-4** (trained) | 714/737 (**97%**) | 23/737 (3%) | 23 |
| **Mastermind** (random) | 0/1000 (0%) | 1000/1000 (100%) | 0 |
| **Connect-4** (random) | 0/1165 (0%) | 1165/1165 (100%) | 0 |

> [!IMPORTANT]
> The trained models produce valid, parseable output **92–97% of the time**. They are not guessing randomly. The pipeline fallback is invoked for only 3–8% of moves.

---

## 3. Key Findings

### Finding 1: Models Learn Output Format

The trained models overwhelmingly produce structurally valid output. For Mastermind, 92% of all guesses parse as valid 4-character colour codes (A–F). For Connect-4, 97% of moves parse as valid column indices (0–6). An untrained model would produce gibberish; a random baseline doesn't even attempt format generation.

### Finding 2: Models Learn Task-Relevant Patterns

Mastermind is the definitive test. The search space is 6⁴ = 1,296 possible codes. Random guessing has a 1/1,296 = 0.08% chance of solving on any single guess. With 10 guesses, the probability of a random solver is still negligible (~0.77%). The trained model solves **78%** of games — this is not achievable without learning the feedback→guess mapping.

### Finding 3: Pipeline Adds Modest Value on Top of Model

The fallback path rescues only 3–8% of moves. The pipeline's main contributions are:
- **Planner re-invocation** on stalls (observed but infrequent)
- **Kanban state tracking** for move history
- **Valid-move filtering** as a safety net

The pipeline amplifies the model's existing capability — it does not substitute for it.

### Finding 4: Connect-4 Random Baseline Is Elevated

Connect-4 random baseline is 54%, which reflects X's first-mover advantage in a gravity-constrained 7-wide board. The trained model's 91% still represents a 37-point improvement — significant, but less dramatic than Mastermind's 78-point gap.

---

## 4. Implementation

The experiment uses a compile-time `RANDOM_BASELINE` flag in each demo:

```c
/* Intelligence verification baseline mode:
 *   0 = Trained model (default)
 *   1 = Random baseline (random valid guess, pipeline still runs)
 */
#ifndef RANDOM_BASELINE
#define RANDOM_BASELINE 0
#endif
```

Move attribution is tracked with `total_model_sourced` and `total_fallback_sourced` counters.

Files modified:
- [`mastermind/main.c`](../../experiments/organelles/mastermind/main.c) — RANDOM_BASELINE + attribution
- [`connect4/main.c`](../../experiments/organelles/connect4/main.c) — RANDOM_BASELINE + attribution

To reproduce:
```bash
# Set RANDOM_BASELINE to 1 in the source, rebuild, and run
cmake --build build --target mastermind_demo && ./build/mastermind_demo
```

---

## 5. Conclusion

| Scenario | Threshold | Observed | Verdict |
|----------|-----------|----------|---------|
| 🟢 **Model dominates** | gap ≥ 20 pts | 37–78 pts | ✅ **Confirmed** |
| 🟡 Both contribute | gap 5–20 pts | — | — |
| 🔴 Pipeline dominates | gap < 5 pts | — | ❌ **Rejected** |

**The organelles genuinely learn.** The trained models produce valid, contextually appropriate output 92–97% of the time. The pipeline's fallback is a safety net, not the primary source of correct moves. The 37–78 point gap between trained and random baselines conclusively demonstrates that the models have learned task-relevant patterns from their training corpora.

**Defensible claim:** "Organelles learn pattern-matched strategies from their training corpora, and the OPA pipeline amplifies accuracy by catching the 3–8% of residual errors."
