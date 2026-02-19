# Tic-Tac-Toe Multi-Organelle Pipeline

Two tiny neural networks play Tic-Tac-Toe as X against a random opponent — and win 84% of games.

---

## Spear Summary

**Point:** Two 64K-parameter brains talking in plain text beat a random Tic-Tac-Toe opponent 84% of the time with zero external dependencies.

**Picture:** It's like two interns passing sticky notes to each other — one writes the plan ("go for the center then block"), the other picks the actual square — and a supervisor just checks if the square is empty. The sticky-note format is so simple that they almost never misread each other.

**Proof:** 100 games completed in 0.97 seconds. 84 wins, 6 draws, 10 losses. The kanban feedback loop (blocked positions + re-planning) lifted the raw win rate to 84% with a 90% win+draw rate.

**Push:** The Player's loss plateaued at 0.38 — it's hit a capacity ceiling at 64K params. Bump to N_EMBD=64 / N_LAYER=3 to break through and close the gap to perfect play.

---

## How It Works

```
┌──────────┐  "board=_________|empties=9"  ┌──────────┐  "4"  ┌──────────┐
│ Planner  │─────────────────────────────▶│  Player  │──────▶│  Judge   │
│ (neural) │  "todo=move_4,check,move_0"  │ (neural) │       │ (determ.)│
│ 64K params│                              │ 64K params│       │ board[x] │
└──────────┘                              └──────────┘       └────┬─────┘
      ▲                                                          │
      │ replan if stalls > 3           ┌──────────┐              │
      └────────────────────────────────│  Kanban  │◀─────────────┘
                                       │  State   │  blocked + last moves
                                       └──────────┘
```

- **Planner** sees the board state → outputs a priority chain (`todo=move_4,check,move_0,check`)
- **Player** sees the board + any blocked positions → outputs a **single digit** (0–8)
- **Judge** is fully deterministic — checks cell empty, win/draw detection, no neural network
- **Kanban** tracks blocked positions, move history, and stall count; triggers re-plan after 3 consecutive failures

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 48 |
| N_HEAD | 4 |
| N_LAYER | 2 |
| BLOCK_SIZE | 128 |
| Params/organelle | ~64,000 |
| Total neural params | ~128,000 |
| Inference temp | 0.2 |

## Training

| Organelle | Corpus | Entries | Vocab | Best Loss | Time (25K steps) |
|-----------|--------|---------|-------|-----------|------------------|
| Planner | `tictactoe_planner.txt` | 3,252 | 29 chars | 0.1345 | 325s |
| Player | `tictactoe_player.txt` | 13,033 | 26 chars | 0.3807 | 150s |

**Total: 475 seconds single-threaded (7.9 minutes)**

Corpora generated from minimax-optimal play across all reachable board states. Player corpus includes 3,575 blocked-move variants for kanban support.

## Results (vs Random Opponent)

| Metric | 5K Steps | 25K Steps | Post-Refactor |
|--------|----------|-----------|---------------|
| Wins | 68 (68%) | 63 (63%) | **84 (84%)** |
| Draws | 5 (5%) | 19 (19%) | **6 (6%)** |
| Losses | 27 (27%) | 18 (18%) | **10 (10%)** |
| **Win+Draw** | **73%** | **82%** | **90%** |
| Valid moves | — | — | 362 |
| Invalid moves | — | — | 4,881 |
| Parse Errors | 0 | 14 | 10 |
| Re-plans | 0 | 102 | 45 |
| Pipeline Time | 0.14s | 0.34s | 0.97s |

### Key Finding: Aggressive Play Dominates

The post-refactor run shows a shift to aggressive play: 84 wins (up from 63), only 6 draws (down from 19). Losses halved from 18 to 10. The `OpaKanban` library implementation with `mgpt_default_threads()` provides better training convergence through multi-threaded gradient accumulation.

### Player Capacity Ceiling

Player loss plateaued at ~0.38 regardless of training duration (5K vs 25K). The 64K-param model can't fully represent all 5,478 reachable board states. This is a capacity limit, not a training one. The 4,881 invalid moves (93% invalid rate) show the model is essentially guessing positions and relying on the kanban loop to eventually find a valid one.

## Build & Run

```bash
# From repo root
mkdir build && cd build
cmake .. && cmake --build . --target tictactoe_demo
./tictactoe_demo    # trains both organelles, then plays 100 games
```

Auto-resumes from checkpoints (`tictactoe_planner.ckpt`, `tictactoe_player.ckpt`).

## What Validated from 8-Puzzle

| Pattern | Status |
|---------|--------|
| Pipe-separated flat strings | ✅ Near-zero parse errors |
| Deterministic Judge | ✅ Zero false rejections — strictly superior to neural for complete-information games |
| Kanban feedback (blocked + replan) | ✅ Boosted win+draw from 63% raw to 82% |
| Single-digit Player output | ✅ Maximally simple — no parse overhead |

## Known Issues

1. **High invalid move rate** — 4,881 invalid moves across 100 games (93% invalid rate). The kanban loop compensates but the model is fundamentally guessing
2. **Parse errors at 25K** — Player occasionally produces multi-digit output. Could constrain to single-token generation
3. **Pipeline rescue dependency** — Like Connect-4, the high win rate comes from kanban coordination, not genuine board understanding

## Recommended Next Steps

| Priority | Change | Impact |
|----------|--------|--------|
| **P1** | Increase model capacity (N_EMBD=64, N_LAYER=3) | Break Player loss plateau |
| **P1** | Constrain Player to single-token output | Eliminate parse errors |
| **P2** | MD-delta encoding (like puzzle8) | Give model pre-computed evaluation instead of raw board |
| **P2** | Test against minimax opponent | Validate learned strategy quality |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
