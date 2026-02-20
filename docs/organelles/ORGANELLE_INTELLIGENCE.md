# Do Organelles Actually Learn? A Verification Framework

**Separating model intelligence from pipeline filtering.**

---

## Spear Summary

**Point:** Right now we cannot prove the models are learning because the pipeline's fallback logic could funnel random guesses into wins — we need a controlled baseline to measure the gap.

**Picture:** Imagine a blindfolded dart player with a friend who catches every dart that misses the board and places it on a valid scoring ring. The player's accuracy looks great but the friend is doing all the work. To know if the player has skill you need to compare against a *different* blindfolded player who throws completely at random — same friend and same catch-and-place rules.

**Proof:** In Pentago (line 204–210 of `main.c`), when the model's output fails to parse, the pipeline falls back to `empties[0]` — the first empty cell — deterministically. The demo reports `parse_errors` but never reports what percentage of *winning moves* came from the model versus this fallback. Without that number, the 91% win rate is not attributable to the model.

**Push:** Run the 3-tier experiment defined below. If the trained model beats random-baseline by ≥20 percentage points, the model genuinely learned. If the gap is <5 points, the pipeline is doing all the work and the "intelligence" claim needs to be reframed.

---

## 1. The Question

Every organelle game demo has two moving parts:

| Component | What it does |
|-----------|-------------|
| **The model** | Generates a proposed move from a board prompt |
| **The pipeline** | Filters invalid moves, retries, breaks cycles, falls back to a deterministic pick |

The reported win rates (e.g. Pentago 91%, Connect-4 90%) are **system-level results** — they measure the combined effect of both components. The question is: **how much of each?**

Three scenarios are possible:

| Scenario | What it means |
|----------|--------------|
| 🟢 **Model dominates** | Trained model >> random baseline. The model learned real patterns. |
| 🟡 **Both contribute** | Trained model > random baseline by a modest margin. Model helps, pipeline rescues. |
| 🔴 **Pipeline dominates** | Trained model ≈ random baseline. The pipeline is doing all the work. |

All three outcomes are scientifically interesting. Even scenario 🔴 would validate OPA as a powerful coordination framework — just not as evidence of "model intelligence."

---

## 2. The Experiment: 3-Tier Baseline Comparison

For each game, run 100 games across three conditions:

### Tier 1: Random Baseline (🔴)

Replace the model's output with a **uniformly random pick** from the valid moves list. The pipeline (kanban, cycle detector, valid-move filter, fallback) still runs identically.

This answers: *What does the pipeline alone achieve against a random opponent?*

### Tier 2: Untrained Model (🟡)

Create a model with `model_create()` (random weights, zero training) and run it through the same pipeline. Same ensemble voting, same temperature.

This answers: *Does an untrained neural network produce better-than-random output just from its architecture?*

### Tier 3: Trained Model (🟢)

The current behaviour — trained model with full pipeline.

This answers: *What does training add on top of the pipeline?*

### Expected Results If Models Learn

| Game | Random (🔴) | Untrained (🟡) | Trained (🟢) | Gap (🟢 − 🔴) |
|------|:-----------:|:--------------:|:------------:|:--------------:|
| Tic-Tac-Toe | ~60–70% | ~60–70% | 90% | **~20–30 pts** |
| Connect-4 | ~30–50% | ~30–50% | 90% | **~40–60 pts** |
| Pentago | ~40–60% | ~40–60% | 91% | **~30–50 pts** |
| Mastermind | ~5–15% | ~5–15% | 79% | **~60–70 pts** |

> [!NOTE]
> The random baseline for tic-tac-toe will be high (~60%+) because X goes first against a random opponent. This is expected. The *gap* is what matters, not the absolute number.

---

## 3. Implementation Plan

### 3.1 Add a `--random-baseline` Flag

Each game demo gets a compile-time or runtime flag:

```c
#define RANDOM_BASELINE 0  /* set to 1 for random baseline, 2 for untrained model */
```

When `RANDOM_BASELINE == 1`, skip the `organelle_generate_ensemble` call and instead pick a random valid move:

```c
#if RANDOM_BASELINE == 1
  /* Random baseline: pick uniformly from valid moves */
  int ri = rand_r(&game_seed) % num_empties;
  proposed_pos = empties_arr[ri];
#elif RANDOM_BASELINE == 2
  /* Untrained model: use model_create with random weights */
  /* (organelle loaded but never trained) */
  organelle_generate_ensemble(player, &cfg, prompt, output, ...);
#else
  /* Normal: trained model */
  organelle_generate_ensemble(player, &cfg, prompt, output, ...);
#endif
```

### 3.2 Add Attribution Tracking

Every game demo should track and report:

```
Model-sourced moves:     142 / 200 (71%)
Fallback-sourced moves:   58 / 200 (29%)
Parse errors:             58
```

This is a one-line addition — increment a counter when the fallback path is taken vs when the model's output is accepted directly.

### 3.3 Run the Matrix

```bash
# For each game (tictactoe, connect4, pentago, mastermind):
cmake -DRANDOM_BASELINE=0 --build build && ./build/tictactoe_demo  # Trained
cmake -DRANDOM_BASELINE=1 --build build && ./build/tictactoe_demo  # Random
cmake -DRANDOM_BASELINE=2 --build build && ./build/tictactoe_demo  # Untrained
```

Collect results in a table for the README.

---

## 4. Secondary Verification: Training Loss Convergence

Beyond win rates, training loss proves the model learned *something*:

| Evidence | What it proves |
|----------|---------------|
| Loss converges to near-zero | Model memorised the corpus (expected for small datasets) |
| Loss plateaus above 1.0 | Model failed to learn — checkpoint is noise |
| Trained model generates valid format | Model learned the output grammar (e.g. "P23Q0DC" for Pentago) |
| Trained model generates contextually appropriate moves | Model learned board→move mappings, not just format |

The current checkpoint logs are retroactive summaries. To strengthen the evidence, add loss logging to `organelle_train`:

```
Step 1000/25000  loss=2.341
Step 5000/25000  loss=0.412
Step 25000/25000 loss=0.034  ← convergence
```

---

## 5. What Each Outcome Means for the Project

### 🟢 Model dominates (gap ≥ 20 pts)

**Claim:** "Organelles learn to solve logic problems, and the pipeline amplifies their intelligence."

This is the strongest narrative. The model contributes real pattern recognition; the pipeline catches the remaining errors.

### 🟡 Both contribute (gap 5–20 pts)

**Claim:** "Organelles learn output format and partial strategy; the pipeline does the heavy lifting on correctness."

Still valid. Reframe from "intelligence" to "specialised retrieval + robust coordination." The pipeline becomes the star, which is honestly a more interesting and defensible claim.

### 🔴 Pipeline dominates (gap < 5 pts)

**Claim:** "OPA is a powerful coordination framework that achieves high win rates regardless of model quality."

This is still publishable and valuable — it means the *architecture* is the innovation, not the model. The honest version of this claim may actually be more impactful than an overclaimed model-intelligence story.

---

## 6. Priority Order

The games most likely to show genuine model intelligence (larger action spaces where random play fails badly):

| Priority | Game | Why |
|----------|------|-----|
| 1 | **Mastermind** | Combinatorial — random guesses almost never solve it |
| 2 | **Connect-4** | Strategic depth — random play loses to any pattern recognition |
| 3 | **Pentago** | Complex action format (place + rotate) — random format generation is unlikely |
| 4 | **Tic-Tac-Toe** | Simple — random baseline will be high, hard to show a gap |

Start with Mastermind. If the random baseline is ~10% and the trained model is ~79%, that's a 69-point gap — the most convincing proof of model intelligence in the entire project.
