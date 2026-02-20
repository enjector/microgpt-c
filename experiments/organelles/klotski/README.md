# Klotski Multi-Organelle Pipeline

A 30K-parameter Transformer solves simplified Klotski sliding block puzzles — **62% solve rate** from only 232 training positions with heavy reliance on fallback moves.

---

## Spear Summary

**Point:** Even a tiny corpus of 232 positions achieves 59% solve rate because the kanban fallback mechanism does the heavy lifting — 1849 parse errors mean the model fails to produce valid output 87% of the time but the deterministic safety net still moves blocks toward the goal.

**Picture:** A blindfolded assistant randomly slides blocks while a supervisor checks after each move whether the big block reached the exit. The assistant gets lucky 59% of the time — not because they learned the puzzle but because random valid moves in a small space sometimes work.

**Proof:** 100 puzzles: 59 solved, 1849 parse errors (87% parse failure rate), 108 re-plans. The player corpus had only 232 entries — the smallest of all 8 games.

**Push:** Expand the corpus to 2000+ positions by using longer scramble sequences and include more diverse block configurations to give the model enough signal to learn real sliding strategies.

---

## How It Works

```
┌──────────┐  "board=A..BC..|empty=8"    ┌──────────┐"A:D"  ┌──────────┐
│ Planner  │────────────────────────────▶│  Player  │──────▶│  Judge   │
│ (neural) │  "todo=slide,check,slide"   │ (neural) │       │ (determ.)│
│  30K par.│                             │  30K par.│       │collision │
└──────────┘                             └──────────┘       └────┬─────┘
      ▲                                                          │
      │ replan if stalls > 3          ┌──────────┐              │
      └───────────────────────────────│  Kanban  │◀─────────────┘
                                      │  State   │  blocked + history
                                      └──────────┘
```

- **Board** is a 20-char string (4×5 grid): block IDs + `.` for empty
- **Player** sees board + valid moves + blocked → outputs "B:D" (block ID, direction)
- **Judge** is deterministic — checks collision bounds, applies multi-cell slide
- **Kanban** tracks blocked moves and stall history; triggers re-plan after 3 failures
- **Fallback** picks first valid unblocked move on parse errors

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| Board | 4×5 grid (20 cells) |
| Blocks | Multi-cell (1×1, 1×2, 2×1) |
| Params/organelle | ~30,000 (Tier 1 — Micro) |
| Total neural params | ~920,000 |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `klotski_planner.txt` | 232 | 12 KB | 0.07 | ~10 min |
| Player | `klotski_player.txt` | 232 | 15 KB | 0.07 | ~11 min |

**Total: ~21 minutes. 25,000 steps each, multi-threaded.**

### Corpus Generation

- BFS solver from randomly scrambled positions
- Starting layout: multi-cell blocks arranged on 4×5 grid
- 500 scramble attempts, 232 solvable positions found
- Only first move of each BFS solution used for training

## Results (100 Puzzles)

| Metric | Value |
|--------|-------|
| **Puzzles solved** | **59 / 100 (59%)** |
| Total slides | 2,117 (avg 21.2) |
| Parse errors | 1,849 (87%) |
| Planner re-plans | 108 |
| Pipeline time | 22.92s |

### Key Observations

1. **59% solve rate despite 87% parse failure** — the kanban fallback mechanism is doing almost all the work, selecting valid moves when the model produces unparseable output
2. **232-entry corpus is too small** — the model cannot learn meaningful sliding strategies from so few examples; compare to Mastermind's 2K entries achieving 86%
3. **High re-plan rate (108)** — the model gets stuck frequently, requiring the kanban state machine to reset and try different approaches
4. **Fallback solves by random walk** — the simplified 4×5 grid has a small enough state space that random valid moves eventually reach the goal for most puzzles

### Comparison with Other Sliding Puzzles

| Dimension | Puzzle-8 | **Klotski** |
|-----------|----------|-------------|
| Board size | 3×3 (9 cells) | 4×5 (20 cells) |
| Block types | Single tiles | Multi-cell (1×1, 1×2, 2×1) |
| Corpus entries | 1,686 mover | **232 player** |
| Solve rate | 90% | **59%** |
| Parse errors | 0 | **1,849** |
| Key encoding | MD-delta (pre-computed) | Raw board string |

The gap between Puzzle-8 (90%) and Klotski (59%) stems from two factors: (1) Klotski's corpus is 7× smaller, and (2) Klotski uses raw board strings instead of Puzzle-8's MD-delta encoding which pre-computes move consequences.

## Build & Run

```bash
# Generate corpus
cd experiments/organelles/klotski
python3 generate_corpus.py

# Build and run
cd build
cmake .. && cmake --build . --target klotski_demo
./klotski_demo    # trains both organelles, then solves 100 puzzles
```

Auto-resumes from checkpoints (`klotski_planner.ckpt`, `klotski_player.ckpt`).

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| **P1** | Expand corpus to 2000+ positions (longer scrambles) | 10-20× more training signal for learnable patterns |
| **P1** | MD-delta encoding: pre-compute "goal distance change per block slide" | Reduce problem to greedy heuristic (like puzzle8, which got 90%) |
| **P2** | Increase WARMUP_STEPS to 500 (per TRAINING_STRATEGIES.md) | Better gradient stability during early training |
| **P3** | Scale to Medium tier (128/8/6, lr=0.0005, WARMUP=2500) | Better representation for multi-cell block patterns |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
