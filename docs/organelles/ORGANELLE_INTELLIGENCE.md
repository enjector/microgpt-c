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


| Component        | What it does                                                                      |
| ------------------ | ----------------------------------------------------------------------------------- |
| **The model**    | Generates a proposed move from a board prompt                                     |
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


| Game            | Metric        | 🔴 Random Baseline | 🟢 Trained Model |     Gap     |      Verdict      |
| ----------------- | --------------- | :------------------: | :----------------: | :-----------: | :------------------: |
| **Mastermind**  | Solve rate    |       **0%**       |     **78%**     | **+78 pts** | 🟢 Model dominates |
| **Connect-4**   | Win rate      |      **54%**      |     **91%**     | **+37 pts** | 🟢 Model dominates |
| **Tic-Tac-Toe** | Win+Draw rate |     (not run)     |     **86%**     |     —     | (baseline needed) |

### Move Attribution


| Game                     |   Model-sourced   | Fallback-sourced | Parse errors |
| -------------------------- | :-----------------: | :----------------: | :------------: |
| **Mastermind** (trained) | 558/605 (**92%**) |   47/605 (8%)   |      47      |
| **Connect-4** (trained)  | 714/737 (**97%**) |   23/737 (3%)   |      23      |
| **Mastermind** (random)  |    0/1000 (0%)    | 1000/1000 (100%) |      0      |
| **Connect-4** (random)   |    0/1165 (0%)    | 1165/1165 (100%) |      0      |

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

## 5. Secondary Verification: Training Loss Convergence

### Why Loss Curves Matter

The baseline comparison (Section 2) proves models outperform random. But loss convergence answers a *different* question: **did the model's internal weights actually change in a meaningful direction during training?**

If the loss curve is flat or oscillating near the initial value, the model failed to learn — even if pipeline filtering masks this at game time. Conversely, a smooth descent from initial loss (~3.5–5.0) to a low plateau (~0.08–0.11) proves the optimiser found a minimum in the loss landscape and the weights encode the training corpus.

### What We Measured

Each game was re-trained from scratch with loss logged every 1,000 steps. The organelle logs are saved alongside checkpoints in `models/organelles/*.ckpt.log`.

#### Mastermind Loss Curves


|     Step | Planner Loss | Player Loss |
| ---------: | :------------: | :-----------: |
|        1 |    3.6821    |   3.4790   |
|    1,000 |    0.2164    |   0.3656   |
|    5,000 |    0.1595    |   0.2136   |
|   10,000 |    0.1499    |   0.1710   |
|   15,000 |    0.1221    |   0.1451   |
|   20,000 |    0.1001    |   0.1378   |
|   25,000 |    0.1178    |   0.1332   |
| **Best** |  **0.0819**  | **0.1102** |

Reduction: Planner **45×** (3.68 → 0.08), Player **32×** (3.48 → 0.11).

#### Connect-4 Loss Curves


|     Step | Planner Loss | Player Loss |
| ---------: | :------------: | :-----------: |
|        1 |    5.1600    |   4.9851   |
|    1,000 |    0.1829    |   0.2311   |
|    5,000 |    0.1661    |   0.1759   |
|   10,000 |    0.2194    |   0.1347   |
|   15,000 |    0.2282    |   0.1415   |
|   20,000 |    0.1658    |   0.1102   |
|   25,000 |    0.1717    |   0.1092   |
| **Best** |  **0.1131**  | **0.1041** |

Reduction: Planner **46×** (5.16 → 0.11), Player **48×** (4.99 → 0.10).

### Interpretation


| Evidence                                     | What it proves                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| Loss drops 30–48× from initial             | Model learned to predict the corpus; weights are not random noise                |
| Loss plateaus around 0.08–0.11              | Convergence is real; further training would yield diminishing returns            |
| Convergence within first 5,000 steps         | Most learning happens early; the remaining 20,000 steps fine-tune                |
| Trained model generates valid format 92–97% | Model learned the output grammar (e.g. "ABCD" for Mastermind, "3" for Connect-4) |

> [!NOTE]
> The loss values are per-character cross-entropy. A loss of 0.10 means the model predicts each next character with ~90% confidence — it has effectively memorised the corpus patterns. This is expected for small corpora (2,000–10,000 documents) with a tiny model (92K params).

### Log Files

Fresh training logs with full loss curves are at:

- [`mastermind_planner.ckpt.log`](../../models/organelles/mastermind_planner.ckpt.log)
- [`mastermind_player.ckpt.log`](../../models/organelles/mastermind_player.ckpt.log)
- [`connect4_planner.ckpt.log`](../../models/organelles/connect4_planner.ckpt.log)
- [`connect4_player.ckpt.log`](../../models/organelles/connect4_player.ckpt.log)

---

## 6. Conclusion


| Scenario              | Threshold     | Observed   | Verdict         |
| ----------------------- | --------------- | ------------ | ----------------- |
| 🟢**Model dominates** | gap ≥ 20 pts | 37–78 pts | ✅**Confirmed** |
| 🟡 Both contribute    | gap 5–20 pts | —         | —              |
| 🔴 Pipeline dominates | gap < 5 pts   | —         | ❌**Rejected**  |

**The organelles genuinely learn.** The trained models produce valid, contextually appropriate output 92–97% of the time. The pipeline's fallback is a safety net, not the primary source of correct moves. The 37–78 point gap between trained and random baselines conclusively demonstrates that the models have learned task-relevant patterns from their training corpora.

**Defensible claim:** "Organelles learn pattern-matched strategies from their training corpora, and the OPA pipeline amplifies accuracy by catching the 3–8% of residual errors."

---


## 7. Do Organelles Actually Learn? We Ran the Experiment.

When we reported 78–91% win rates on logic games using 92K-parameter organelle models, an obvious question hung in the air: **is the model doing the work, or is the pipeline just filtering random noise into wins?**

We ran a controlled experiment to find out.

### The Setup

Each game demo has two components — a trained neural model that proposes moves, and a pipeline (OPA) that validates, retries, and falls back if the model produces garbage. To isolate the model's contribution, we replaced the trained model with **uniformly random guesses** while keeping the pipeline identical.

If the pipeline is doing all the work, random guesses should score similarly to the trained model.

### The Results


| Game           | 🔴 Random Baseline | 🟢 Trained Model | Gap         |
| ---------------- | -------------------- | ------------------ | ------------- |
| **Mastermind** | 0% solved          | 78% solved       | **+78 pts** |
| **Connect-4**  | 54% wins           | 91% wins         | **+37 pts** |

The Mastermind result is the clearest proof. The search space is 1,296 possible codes. Random guessing **literally never solves it** in 10 attempts (0/100 games). The trained model solves 78/100. That gap is not achievable by accident.

### Move Attribution

We also tracked where each move came from — model output or pipeline fallback:


|            | Model-sourced | Fallback |
| ------------ | --------------- | ---------- |
| Mastermind | **92%**       | 8%       |
| Connect-4  | **97%**       | 3%       |

The trained models produce valid, parseable output the vast majority of the time. The pipeline's fallback catches only 3–8% of residual errors. The models aren't guessing — they're generating structurally and contextually valid responses.

### Training Loss Convergence

We re-trained all models from scratch with loss logging. Every organelle shows clear convergence:


| Organelle          | Start → Best Loss | Reduction |
| -------------------- | -------------------- | ----------- |
| Mastermind Planner | 3.68 → 0.08       | **45×**  |
| Mastermind Player  | 3.48 → 0.11       | **32×**  |
| Connect-4 Planner  | 5.16 → 0.11       | **46×**  |
| Connect-4 Player   | 4.99 → 0.10       | **48×**  |

At a final loss of ~0.10, each model predicts the next character with ~90% confidence — it has learned the output grammar and contextual move patterns from its training corpus.

### What This Means

The organelles perform **pattern-matched retrieval** — they memorise board-state → move associations from their training corpora and reproduce them at inference time. The OPA pipeline amplifies this by catching residual errors. Both components contribute, but the **model is the engine; the pipeline is the safety net** .

This is a legitimate form of learned behaviour for a 92K-parameter character-level transformer. It's closer to "expert system via pattern retrieval" than "reasoning from first principles" — and that's exactly what these tiny models are designed to do.

**TL;DR:** Organelles genuinely learn. The 37–78 point gap between trained models and random baselines, combined with 30–48× loss convergence, conclusively proves the models have learned task-relevant patterns. The pipeline amplifies — it does not substitute.
