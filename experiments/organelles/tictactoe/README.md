# Tic-Tac-Toe Multi-Organelle Pipeline

Two 460K-parameter neural networks play Tic-Tac-Toe as X against a random opponent — win 81%, draw 6%, with **zero invalid moves** and only **18 parse errors**.

---

## Spear Summary

**Point:** Scaling model capacity (48→96 embed, 2→4 layers) slashed parse errors by 91% (193→18), while ensemble voting + valid-move pre-filtering maintain zero invalid moves.

**Picture:** Three interns each guess a square, they vote, and a checklist of empty squares ensures no one picks an occupied cell. A bigger brain (7× more params) means 10× fewer interns guess nonsense.

**Proof:** 100 games: 0 invalid moves, 81 wins + 6 draws = 87% win+draw rate. Only 18 parse errors (↓91% from 193 at 64K). 

**Push:** Parse errors now at 5% — diminishing returns from capacity alone. MD-delta encoding and minimax opponent testing are the next frontiers.

---

## How It Works

```
┌──────────┐  "board=_________|empties=9"  ┌──────────┐  "4"  ┌──────────┐
│ Planner  │─────────────────────────────▶│  Player  │──────▶│  Judge   │
│ (neural) │  "todo=move_4,check,move_0"  │ (neural) │       │ (determ.)│
│ 460K par.│                              │ 460K par.│       │ board[x] │
└──────────┘                              └──────────┘       └────┬─────┘
      ▲                                                          │
      │ replan if stalls > 3           ┌──────────┐              │
      └────────────────────────────────│  Kanban  │◀─────────────┘
                                       │  State   │  blocked + last moves
                                       └──────────┘
```

- **Planner** sees the board state → outputs a priority chain (`todo=move_4,check,move_0,check`)
- **Player** sees the board + `valid=` list of empty positions + blocked → outputs a **single digit** (0–8)
- **Ensemble** runs 3 inferences with temperature jitter, majority-votes the result
- **Valid Filter** checks result against `valid=` list; falls back to first valid non-blocked cell
- **Judge** is fully deterministic — checks cell empty, win/draw detection, no neural network
- **Kanban** tracks blocked positions, move history, and stall count; triggers re-plan after 3 failures

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| MLP_DIM | 384 |
| BLOCK_SIZE | 128 |
| Params/organelle | ~460,000 |
| Total neural params | ~920,000 |
| Inference temp | 0.2 |

## Training

| Organelle | Corpus | Entries | Vocab | Best Loss | Time (25K steps) |
|-----------|--------|---------|-------|-----------|------------------|
| Planner | `tictactoe_planner.txt` | 3,252 | 29 chars | ~0.06 | ~10 min |
| Player | `tictactoe_player.txt` | 13,033 | 26 chars | ~0.20 | ~10 min |

**Total: ~20 minutes multi-threaded.**

Corpora generated from minimax-optimal play across all reachable board states. Player corpus includes 3,575 blocked-move variants for kanban support.

## Results (vs Random Opponent)

| Metric | Post-Refactor (48/2) | Ensemble+Valid (48/2) | **Capacity (96/4)** |
|--------|---------------------|----------------------|---------------------|
| Wins | 84 (84%) | 75 (75%) | **81 (81%)** |
| Draws | 6 (6%) | 8 (8%) | **6 (6%)** |
| Losses | 10 (10%) | 17 (17%) | **13 (13%)** |
| **Win+Draw** | **90%** | **83%** | **87%** |
| Valid moves | 362 | 421 | **348** |
| **Invalid moves** | **4,881** | **0** | **0** |
| Parse Errors | 10 | 193 | **18** |
| Pipeline Time | 0.97s | 0.38s | **1.52s** |

### Impact of Capacity Scaling (48→96 embed, 2→4 layers)

| Metric | 64K params | 460K params | Δ |
|--------|-----------|------------|---|
| Win+Draw | 83% | **87%** | +4% |
| Parse errors | 193 | **18** | **↓91%** |
| Invalid moves | 0 | 0 | — |

The largest improvement from capacity increase across all three demos. The 64K model struggled with position outputs (31% parse rate); the 460K model handles them easily (5%).

### How Invalid Moves Hit Zero

Three mechanisms work together:
1. **Ensemble voting** (3 votes with ±0.05 temperature jitter) — reduces noise in model output
2. **`valid=` pre-filtering** — prompt includes legal positions so the model learns constraints
3. **`opa_valid_fallback()`** — deterministic safety net picks first valid non-blocked cell on parse errors

## Build & Run

```bash
# From repo root
mkdir build && cd build
cmake .. && cmake --build . --target tictactoe_demo
./tictactoe_demo    # trains both organelles, then plays 100 games
```

Auto-resumes from checkpoints (`tictactoe_planner.ckpt`, `tictactoe_player.ckpt`).

## Recommended Next Steps

| Priority | Change | Impact |
|----------|--------|--------|
| **P1** | MD-delta encoding (like puzzle8) | Give model pre-computed evaluation |
| **P2** | Test against minimax opponent | Validate learned strategy quality |
| **P3** | More training steps (50K+) | Better convergence at 460K params |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
