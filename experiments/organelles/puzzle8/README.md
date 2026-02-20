# 8-Puzzle Multi-Organelle Pipeline

Five 460K-parameter neural networks solve sliding tile puzzles by learning **structural heuristics** — not memorising board states. **90% solve rate** with zero parse errors after capacity scaling.

---

## Spear Summary

**Point:** Scaling from 64K→460K params per organelle (N_EMBD=96, N_LAYER=4) combined with ensemble voting + valid-move pre-filtering achieves **90% solve rate** with **zero parse errors** — up from 0% when the same valid-filter was tried at 64K params.

**Picture:** It's like two people solving a puzzle by passing notes — the Strategist identifies which tile is most out of place, and the Mover picks which direction to slide based on the *consequences* of each move. At 64K params, the valid-move prompt made the notes too long to read. At 460K, the model handles the extra information and solves 90% of puzzles.

**Proof:** 27/30 unseen puzzles solved (100% easy, 100% medium, 70% hard). Zero parse errors (was 1,176 at 64K), zero out-of-bounds rejections. 23 cycle breaks. Pipeline executes in 1.20s.

**Push:** Add `trap=1` detour signal to let the model learn *when* to deviate from greedy descent. Feed Strategist output as priority hint to Mover for multi-step planning.

---

## How It Works

```
┌────────────┐  "m=3,5,x,4|md=4"   ┌──────────┐   "up"    ┌──────────┐
│ Strategist │─────────────────────▶│  Mover   │─────────▶│  Judge   │
│  (neural)  │  "up" (priority)    │ (neural) │          │ (determ.)│
│  64K params│                     │ 64K params│          │apply_move│
└────────────┘                      └──────────┘          └────┬─────┘
      ▲                                                        │
      │ cleared if stalls > 6       ┌──────────┐              │
      └─────────────────────────────│ Blocked  │◀─────────────┘
                                    │  Tracker │  blocked directions
                                    └──────────┘
```

### MD-Delta Encoding (Key Innovation)

Instead of feeding the raw board string (`board=742153806`), the orchestrator **pre-computes the manhattan distance after each possible move**:

```
m=3,5,x,4    ← "if you go up→md=3, down→md=5, left→illegal, right→md=4"
```

The model's job reduces from "parse a 9-digit string and somehow figure out which tile goes where" to **"mostly pick the smallest number"** — a structural rule that generalises across all 181,440 board states using just 428 unique input patterns.

- **Strategist** sees md-deltas → outputs the best direction (priority hint)
- **Mover** sees md-deltas + blank position + `valid=` directions → outputs a direction word
- **Ensemble** runs 3 inferences with temperature jitter, majority-votes the result
- **Valid Filter** checks result against `valid=` list; falls back to first valid non-blocked direction
- **Judge** is deterministic: `apply_move()` boundary check
- **Blocked tracker** prevents repeating invalid moves; clears after 6 stalls

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 5 (Strategist + Greedy-Mover + Detour-Detector + Detour-Mover + Judge) |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| MLP_DIM | 384 |
| BLOCK_SIZE | 128 |
| Params/organelle | ~460,000 |
| Total params | ~2,300,000 |
| Inference temp | 0.2 |

## Training

| Organelle | Corpus | Entries | Size | Signal |
|-----------|--------|---------|------|--------|
| Strategist | `puzzle8_strategist.txt` | 422 | 7.5 KB | Greedy-only (100% consistent) |
| Greedy-Mover | `puzzle8_mover.txt` | 1,686 | 45 KB | Greedy-only (100% consistent) |
| Detour-Detector | `puzzle8_detour_detector.txt` | 427 | 8 KB | Binary: g/d classifier |
| Detour-Mover | `puzzle8_detour_mover.txt` | 255 | 5.5 KB | Non-greedy BFS moves only |
| Judge | `puzzle8_judge.txt` | 57,344 | 1.6 MB | Valid/invalid moves |

Corpora generated from BFS-optimal solutions of 5,000 unique solvable puzzles (md 1–22). 25,000 training steps per organelle.

## Results

### v3 — Mixed Corpus (baseline for this encoding)

| Band | MD Range | Solved | Rate | Avg Moves |
|------|----------|--------|------|-----------|
| **EASY** | 1–4 | 9/10 | **90%** | 2.6 |
| **MEDIUM** | 5–8 | 7/10 | **70%** | 6.6 |
| **HARD** | 9+ | 2/10 | **20%** | 20.5 |
| **Overall** | — | **18/30** | **60%** | — |

### v3b — Greedy/Detour Split (current)

| Band | MD Range | Solved | Rate | Avg Moves |
|------|----------|--------|------|-----------|
| **EASY** | 1–4 | 10/10 | **100%** ✓ | 3.5 |
| **MEDIUM** | 5–8 | 4/10 | **40%** | 7.5 |
| **HARD** | 9+ | 2/10 | **20%** | 11.0 |
| **Overall** | — | **16/30** | **53%** | — |

### v3b + Oscillation Breaker (64K params)

| Band | MD Range | Solved | Rate | Avg Moves |
|------|----------|--------|------|-----------|
| **EASY** | 1–4 | 10/10 | **100%** ✓ | 3.7 |
| **MEDIUM** | 5–8 | 5/10 | **50%** | 10.4 |
| **HARD** | 9+ | 3/10 | **30%** | 11.7 |
| **Overall** | — | **18/30** | **60%** | — |

Cycle breaks: **73**.

### v3b + Ensemble+Valid at 64K params (FAILED)

Adding `valid=` to mover prompts at 64K params caused **98% parse errors** — the prompts were too long for the model. Fallback picks random valid directions → **0% solve rate**. This was the motivation for capacity scaling.

### v3b + Ensemble+Valid + Capacity (96/4, 460K params) — CURRENT

| Band | MD Range | Solved | Rate | Avg Moves |
|------|----------|--------|------|-----------|
| **EASY** | 1–4 | 10/10 | **100%** ✓ | 2.9 |
| **MEDIUM** | 5–8 | 10/10 | **100%** ✓ | 6.1 |
| **HARD** | 9+ | 7/10 | **70%** | 9.7 |
| **Overall** | — | **27/30** | **90%** | — |

Cycle breaks: **23** (↓69% from 73). Parse errors: **0**. Pipeline time: **1.20s**.

### Full Iteration History

| Version | Encoding | Params | Solve Rate | Key Change |
|---------|----------|--------|------------|------------|
| v2 (baseline) | Raw board string | 64K | 96.7% (train overlap) | Memorisation, not generalisation |
| v3-disp | Per-tile displacement | 64K | 17% | Too many patterns for 64K params |
| v3-md | MD-delta (mixed) | 64K | 60% | Structural rule becomes learnable |
| v3b+cycle | MD-delta (clean + breaker) | 64K | 60% (100% easy) | Orchestration → +10% medium, +10% hard |
| v3b+valid | MD-delta + ensemble+valid | 64K | **0%** | Prompt too long → 98% parse errors |
| **v3b+valid+cap** | **MD-delta + ensemble+valid** | **460K** | **90% (100%/100%/70%)** | **Capacity → 0 parse errors, +30%** |

### Key Observations

1. **Capacity scaling was transformative** — 0% → 90% solve rate by going from 64K to 460K params
2. **Zero parse errors at 460K** — the model handles `valid=` prompt extension perfectly
3. **Cycle breaks dropped 69%** (73 → 23) — the model makes fewer oscillation-prone moves
4. **428 unique MD-delta patterns** vs 181,440 possible board states — 424× input space compression
5. **Medium puzzles: 50% → 100%** — the largest improvement from capacity; hard went 30% → 70%

## Forensic Corpus Analysis

### The "Pick the Smallest" Rule Is 64.4% of the Story

Analysing all 427 Strategist training entries against pure greedy descent (always pick the direction with the lowest post-move MD) reveals a significant gap:

| Category | Count | % of Corpus |
|---|---|---|
| **Greedy-optimal** (BFS agrees with min-md) | 275 | **64.4%** |
| **Tie-breaking** (BFS picks a different direction with same md) | 99 | 23.2% |
| **Non-greedy detours** (BFS picks a *higher* md) | 53 | **12.4%** |

**35.6% of the training data contradicts "pick the smallest number."** Examples of non-greedy BFS moves:

```
m=6,4,6,x  → BFS says "up" (md=6), but greedy = "down" (md=4)
m=x,5,7,x  → BFS says "left" (md=7), but greedy = "down" (md=5)
m=10,8,10,x → BFS says "up" (md=10), but greedy = "down" (md=8)
```

These are positions where the optimal solution requires a *temporary increase* in manhattan distance to escape a local minimum. The model has no signal in the input to distinguish "greedy works here" from "you need a detour" — it's learning a noisy approximation.

### This Explains the Performance Curve

| Band | Solve Rate | Why |
|---|---|---|
| **EASY** (md 1–4) | 90% | Greedy is almost always BFS-optimal at low md |
| **MEDIUM** (md 5–8) | 70% | Some detours needed; model sometimes guesses right |
| **HARD** (md 9+) | 20% | Detours are frequent; model can't tell when to deviate |

The model has learned the greedy rule well enough for easy/medium cases, but it has no input signal to distinguish positions where greedy works from positions requiring a detour.

### Comparison with Connect4

| | Connect4 | Puzzle8 v2 (raw) | **Puzzle8 v3 (MD-delta)** |
|---|---|---|---|
| Unique inputs | 3,746 boards | 1,649 boards | **428 patterns** |
| Unseen-state performance | ~0% (pipeline rescue) | ~0% | **60% (including hard)** |
| Parse/invalid errors | 60% invalid rate | — | **0%** |
| Source of performance | Kanban rescue + first-player advantage | Memorisation | **Learned structural rule** |

This is the strongest evidence in the organelle experiments that **representation engineering beats capacity scaling**. Same architecture, same params, same pipeline — 0% → 60% by changing the encoding alone.

### The Greedy/Detour Split Experiment

Splitting the corpus into greedy-only (100% consistent) and detour-only training sets revealed a sharp tradeoff:

| Run | Easy | Medium | Hard | Overall | Signal quality |
|-----|------|--------|------|---------|----------------|
| v3 mixed | 90% | **70%** | 20% | **60%** | 12.4% contradictory |
| v3b dual-dispatch | 90% | 30% | 10% | 43% | Clean, but detour Mover too weak |
| **v3b greedy-only** | **100%** | 40% | 20% | 53% | **100% consistent** |

**The same noise was simultaneously hurting easy performance and helping medium performance.** The contradictory detour examples prevented the model from achieving 100% on easy puzzles, but those same examples occasionally caused the correct non-greedy move on medium puzzles *by accident*. You can't have both without a disambiguation signal.

The Detour Detector (trained on 427 examples) correctly identifies all oscillation/failure positions — its `[D]` annotations align perfectly with positions where the greedy solver gets trapped. The detector has learned a real signal; it just can't act on it yet because the 255-entry detour Mover hasn't learned meaningful alternative moves.

## Key Findings

### Representation is everything

The same model architecture (48-dim, 2-layer transformer) went from 0% to 60% on unseen puzzles purely through encoding changes. No capacity increase needed. This validates the VISION.md thesis: *"the task is constrained enough that a few thousand parameters can capture the pattern"* — but only if the pattern is **made explicit** in the input.

### Greedy heuristics have limits — and the corpus teaches them

The MD-delta encoding teaches a *mostly*-greedy policy, but 12.4% of the training data contains BFS-optimal moves that actively increase MD. This creates training noise: the model absorbs contradictory examples without any input feature to disambiguate them. The result is a reliable greedy core with unpredictable behaviour at decision boundaries.

### The organelle decomposition matters

Separating Strategist (strategic assessment) from Mover (tactical execution) keeps each model's task simple. The Mover learns "given these consequences, pick the best direction"; the Strategist learns "which direction looks best overall."

## Build & Run

```bash
# From repo root
mkdir build && cd build
cmake .. && cmake --build . --target puzzle8_demo
./puzzle8_demo    # trains 5 organelles, then solves 30 puzzles
```

Trains: Strategist, Greedy-Mover, Detour-Detector, Detour-Mover, Judge. Auto-resumes from checkpoints (`puzzle8_*_v3b.ckpt`).

## Corpus Generation

```bash
python3 experiments/organelles/puzzle8/generate_corpus.py
```

Generates five corpus files (strategist, greedy mover, detour detector, detour mover, judge) from BFS-optimal solutions of 5,000 puzzles.

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| ~~P1~~ | ~~Oscillation breaker~~ | ✅ Done — 73 breaks, +10% medium, +10% hard |
| ~~P1~~ | ~~Split corpus (greedy/detour)~~ | ✅ Done — 100% easy, 40% medium, 20% hard |
| ~~P1~~ | ~~Increase model capacity (N_EMBD=96, N_LAYER=4)~~ | ✅ Done — **0% → 90% solve, 0 parse errors** |
| **P1** | **Detour signal**: add `trap=1` flag to input when greedy leads to backtracking | Lets the model learn *when* to deviate from greedy |
| **P2** | Feed Strategist output into Mover prompt as priority hint | Multi-step planning to overcome local minima |
| **P3** | More training steps (50K+) at 460K params | Better convergence |
| **P3** | Solve the remaining 30% hard puzzles | May need lookahead or A* integration |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
