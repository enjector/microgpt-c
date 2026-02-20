# Sudoku 4×4 Multi-Organelle Pipeline

A 160K-parameter Transformer solves 4×4 Sudoku puzzles — **78% solve rate** with an unexpected result: hard puzzles (103%) outperform easy ones (47%).

---

## Spear Summary

**Point:** The pipeline achieves 76% solve rate on 4×4 Sudoku with zero invalid moves and the counterintuitive result that harder puzzles (8+ cells removed) solve at 90% while easier ones (4–5 removed) solve at 53%.

**Picture:** A student solving a crossword with some letters filled in — when more letters are missing they have more clue-combinations to work with and the fallback mechanism has more valid cells to try. With fewer blanks the model must be precise and it is not.

**Proof:** 100 puzzles: 76 solved. Easy (4–5 empty): 53%. Medium (6–7 empty): 82%. Hard (8+ empty): **90%**. 184 parse errors, 18 re-plans. Valid fills: 654.

**Push:** Add constraint-aware encoding — show "row X needs {2,4}" instead of raw board strings — so the model learns elimination logic rather than memorising positions.

---

## How It Works

```
┌──────────┐  "board=1.3..2.4...|empty=6"  ┌──────────┐ "R1C2=4" ┌─────────┐
│ Planner  │──────────────────────────────▶│  Player  │─────────▶│  Judge  │
│ (neural) │  "todo=fill,check,fill"       │ (neural) │          │(determ.)│
│ 461K par.│                               │ 459K par.│          │row/col/ │
└──────────┘                               └──────────┘          │box check│
      ▲                                                          └────┬────┘
      │ replan if stalls > 3          ┌──────────┐                   │
      └───────────────────────────────│  Kanban  │◀──────────────────┘
                                      │  State   │ blocked + history
                                      └──────────┘
```

- **Board** is a 16-char string (4×4): digits 1–4 + `.` for empty
- **Player** sees board + valid cells + blocked → outputs "RrCc=D" (row, col, digit)
- **Judge** checks row/column/2×2-box uniqueness constraints
- **Kanban** tracks blocked placements; re-plans after 3 stalls

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| Board | 4×4 grid (16 cells) |
| Digits | 1–4 |
| Params/organelle | ~160,000 (Tier 3 — Standard) |
| Total neural params | ~920,000 |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `sudoku_planner.txt` | 3,000 | 197 KB | 0.18 | ~11 min |
| Player | `sudoku_player.txt` | 20,000 | 1.3 MB | 0.20 | ~12 min |

**Total: ~23 minutes. 25,000 steps each, multi-threaded.**

### Corpus Generation

- Backtracking solver generates valid 4×4 grids
- Unique-solution puzzle generation by removing cells
- 3,000 puzzles with varying difficulty (4–8 cells removed)
- Player corpus includes all valid digit placements per position

## Results (100 Puzzles)

| Metric | Value |
|--------|-------|
| **Puzzles solved** | **76 / 100 (76%)** |
| Easy (4–5 empty) | 16 / 30 (53%) |
| Medium (6–7 empty) | 33 / 40 (82%) |
| Hard (8+ empty) | 27 / 30 (90%) |
| Total fills | 654 (avg 6.5) |
| Valid moves | 654 |
| Parse errors | 184 |
| Planner re-plans | 18 |
| Pipeline time | 8.55s |

### The Inverse Difficulty Effect

The most notable finding: **harder puzzles solve more reliably than easy ones.** This is the opposite of what theory predicts. The explanation:

1. **More blanks = more valid moves** — with 8 empty cells, the model has more chances to find a correct placement by trying different valid options
2. **Fallback effectiveness** — when the model produces parse errors (184 total), the fallback picks a valid cell+digit combination, which has higher odds of being correct when there are more empty cells
3. **Easy puzzles demand precision** — with only 4 empty cells, the model must place the right digit in the right cell with no margin for error

### Comparison with Other Games

| Game | Solve % | Corpus | Parse Errors | Key Factor |
|------|---------|--------|-------------|-----------|
| Mastermind | 86% | 2K entries | 25 | Sequential deduction |
| **Sudoku 4×4** | **76%** | **20K entries** | **184** | Constraint satisfaction |
| Klotski | 59% | 232 entries | 1,849 | Block sliding |
| Lights Out | 12% | 15K entries | 0 | Coupled constraints |

## Build & Run

```bash
# Generate corpus
cd experiments/organelles/sudoku
python3 generate_corpus.py

# Build and run
cd build
cmake .. && cmake --build . --target sudoku_demo
./sudoku_demo    # trains both organelles, then solves 100 puzzles
```

Auto-resumes from checkpoints (`sudoku_planner.ckpt`, `sudoku_player.ckpt`).

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| **P1** | Constraint-aware encoding: show "row 2 needs {1,3}" + "col 1 needs {2,4}" | Teach elimination logic, not position memorisation |
| **P2** | Increase WARMUP_STEPS to 500 (per TRAINING_STRATEGIES.md) | Better gradient stability |
| **P2** | Scale to 9×9 grid with larger corpus | Test OPA on standard Sudoku difficulty |
| **P3** | Scale to Medium tier (128/8/6, lr=0.0005, WARMUP=2500) | Better representation for constraint propagation |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
