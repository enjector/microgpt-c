# 8-Puzzle Multi-Organelle Pipeline

Three tiny neural networks solve sliding tile puzzles by passing sticky notes to each other — and crack 96.7% of them.

---

## Spear Summary

**Point:** Three 64K-parameter brains coordinating via plain-text messages solve 29 out of 30 randomly scrambled 8-puzzles in under 0.02 seconds.

**Picture:** It's like three people solving a Rubik's Cube by passing notes — one writes the strategy ("move top row first"), another picks which tile to slide, and a referee checks if the slide was legal. Nobody sees the whole picture but together they almost always nail it.

**Proof:** The single biggest unlock was model size — jumping from 18K to 64K params per organelle took solve rate from 40% to 90%. Three cross-seed runs then confirmed 96.7% (29/30). The only failure is one specific board where the Mover oscillates right↔left instead of going up.

**Push:** Add a greedy manhattan-distance fallback when oscillation is detected — that one stuck puzzle is a known pattern. Then bump to N_EMBD=64 to close the last gap to 100%.

---

## How It Works

```
┌──────────┐  "board=102453786|md=3"  ┌──────────┐   "up"    ┌──────────┐
│ Planner  │─────────────────────────▶│  Mover   │─────────▶│  Judge   │
│ (neural) │  "todo=move,check,move"  │ (neural) │          │ (determ.)│
│ 64K params│                         │ 64K params│          │apply_move│
└──────────┘                          └──────────┘          └────┬─────┘
      ▲                                                         │
      │ replan if stalls > 4          ┌──────────┐              │
      └───────────────────────────────│  Kanban  │◀─────────────┘
                                      │  State   │  blocked + last + done
                                      └──────────┘
```

- **Planner** sees board + manhattan distance → outputs a priority chain (`todo=move,check,move,check`)
- **Mover** sees board + blank position + any blocked directions → outputs a **single word** (`up`, `down`, `left`, `right`)
- **Judge** is fully deterministic — calls `apply_move()` to check boundary validity. No neural network
- **Kanban** tracks blocked directions, move history, done list, and stall count; triggers re-plan after 4 consecutive failures

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 3 (Planner + Mover + Judge) |
| N_EMBD | 48 |
| N_HEAD | 4 |
| N_LAYER | 2 |
| MLP_DIM | 192 |
| BLOCK_SIZE | 128 |
| Params/organelle | ~64,000 |
| Total params | ~192,000 |
| Inference temp | 0.2 |

## Training

| Organelle | Corpus | Entries | Size | Vocab |
|-----------|--------|---------|------|-------|
| Planner | `puzzle8_planner.txt` | 200 | 10.9 KB | 27 chars |
| Mover | `puzzle8_mover.txt` | 1,649 | 88.5 KB | 33 chars |
| Judge | `puzzle8_judge.txt` | 3,013 | 228.4 KB | 34 chars |

Corpora generated from BFS-optimal solutions of 200 unique solvable puzzles (solution lengths 2–12). 25,000 training steps per organelle.

## Results

### Iteration History

| Config | Params | Solve Rate | Parse Errors | Key Change |
|--------|--------|------------|--------------|------------|
| Baseline (v1) | 18K × 3 | 60% | 43 | Neural Judge + simple Mover output |
| Iter 1: Kanban | 18K × 3 | 10% | 279 | BLOCK_SIZE too small — truncated docs |
| Iter 2: BLOCK_SIZE fix | 18K × 3 | 30% | 114 | Neural Judge false negatives |
| Iter 3a: Deterministic Judge | 18K × 3 | 40% | 0 | Eliminated Judge errors |
| **Iter 3b: Capacity increase** | **64K × 3** | **90%** | **0** | 2-layer model generalises to unseen states |

### Cross-Seed Validation (Final Config — 30 Puzzles)

| | Run 1 (seed 12345) | Run 2 (seed 98765) | Run 3 (seed 777) |
|---|---|---|---|
| Solve rate | 9/10 (90%) | **10/10 (100%)** | **10/10 (100%)** |
| Avg moves | 5.3 | 1.9 | 2.9 |
| Parse errors | 0 | 0 | 0 |
| Re-plans | 3 | 0 | 0 |
| Pipeline time | 0.02s | 0.01s | 0.01s |

**Combined: 29/30 (96.7%)**

### Solve Rate by Difficulty

| Manhattan Distance | Puzzles | Solved | Rate |
|--------------------|---------|--------|------|
| 0–2 | 15 | 15 | **100%** |
| 3 | 4 | 4 | **100%** |
| 4 | 10 | 9 | **90%** |
| 5 | 1 | 1 | **100%** |

The only failure (1/30) is board `123746058` (md=4) — right↔left oscillation where the correct move is `up`.

## Key Findings

### Capacity was the dominant bottleneck

The jump from 1-layer/18K params to 2-layer/64K params (3.5×) was the single largest factor. Protocol changes alone (kanban, blocked directions) only reached 40%. Capacity took it to 90%.

### Deterministic validation beats neural validation

The neural Judge had 100% precision on boundary detection but introduced false negatives on valid moves. Replacing it with `apply_move()` eliminated this error source entirely. For complete-information tasks where the orchestrator has ground truth — use deterministic checks.

### Simplified output format is critical

Reducing Mover output from `move|dir=up|result=123456780` to just `up` eliminated the memorisation burden and dramatically improved convergence.

### Failures are protocol failures not architecture failures

The remaining oscillation failure can be fixed by enriching pipe messages (adding history context) — no model architecture changes needed. This is the core advantage of structured communication over implicit neural coordination.

## Build & Run

```bash
# From repo root
mkdir build && cd build
cmake .. && cmake --build . --target puzzle8_demo
./puzzle8_demo    # trains 3 organelles, then solves 10 puzzles
```

Auto-resumes from checkpoints (`puzzle8_planner_v2.ckpt`, `puzzle8_mover_v2.ckpt`, `puzzle8_judge_v2.ckpt`).

## Corpus Generation

```bash
python3 experiments/organelles/puzzle8/generate_corpus.py
```

Generates all three corpus files from BFS-optimal solutions.

## Recommended Next Steps

| Priority | Change | Impact |
|----------|--------|--------|
| **P1** | Greedy manhattan-distance fallback on oscillation | Breaks last failure mode |
| **P1** | Augment Mover corpus with more md=4 states | Improves generalisation |
| **P2** | Try N_EMBD=64, N_LAYER=3 | May push to 100% without heuristics |
| **P2** | Re-enable neural Judge as non-gating secondary check | Adds learned validation alongside deterministic |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
