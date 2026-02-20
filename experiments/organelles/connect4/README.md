# Connect-4 Multi-Organelle Pipeline

A 460K-parameter Transformer plays Connect-4 on a board with 4.5 trillion possible states — trained on fewer than 2,000 of them — and wins 88% against random with **zero invalid moves**.

---

## Spear Summary

**Point:** Scaling model capacity (48→96 embed, 2→4 layers) combined with ensemble voting + valid-move pre-filtering produces 88% wins with zero invalid moves and only 47 parse errors.

**Picture:** Three darts are thrown blindfolded, the crowd votes on which one was closest, and a safety net ensures no dart lands outside the board. A bigger arm (7× more params) means fewer darts miss entirely.

**Proof:** 100 games, 0 invalid moves, 88% win rate, 47 parse errors (↓32% from 69 at 64K params). The larger model handles the `valid=` prompt extension without choking.

**Push:** Parse errors (47/802 = 6%) are now low. MD-delta encoding and minimax opponent testing are the next frontiers.

---

## How It Works

```
┌──────────┐  "board=...|empties=42"   ┌──────────┐  "3"  ┌──────────┐
│ Planner  │──────────────────────────▶│  Player  │──────▶│  Judge   │
│ (neural) │  "todo=centre,check,drop" │ (neural) │      │ (determ.)│
│ 460K par.│                           │ 460K par.│      │drop_piece│
└──────────┘                           └──────────┘      └────┬─────┘
      ▲                                                       │
      │ replan if stalls > 3          ┌──────────┐            │
      └───────────────────────────────│  Kanban  │◀───────────┘
                                      │  State   │  blocked + last cols
                                      └──────────┘
```

- **Planner** sees 42-char board string + empty count → outputs a priority chain
- **Player** sees board + `valid=` list of legal columns + blocked → outputs a **single digit** (0–6)
- **Ensemble** runs 3 inferences with temperature jitter, majority-votes the result
- **Valid Filter** checks result against known legal moves; falls back to first valid non-blocked column
- **Judge** is fully deterministic — `drop_piece()` checks column-not-full + 4-in-a-row detection
- **Kanban** tracks blocked columns, move history, and stall count; triggers re-plan after 3 failures

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| MLP_DIM | 384 |
| BLOCK_SIZE | 128 |
| Board string | 42 characters (7×6) |
| Params/organelle | ~460,000 |
| Total neural params | ~920,000 |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `connect4_planner.txt` | 5,715 | 573 KB | ~0.05 | ~15 min |
| Player | `connect4_player.txt` | 17,472 | 1,004 KB | ~0.10 | ~15 min |

**Total: ~30 minutes. 25,000 steps each, multi-threaded.**

### Corpus Generation

- Alpha-beta minimax with depth 4
- 500 self-play games (random + optimal + mixed strategies)
- 1,969 unique X-turn positions, mirror-symmetry augmented
- Blocked-variant examples for kanban support

## Results (100 Games vs Random)

| Metric | Original (48/2) | Post-Refactor | Ensemble+Valid (48/2) | **Capacity (96/4)** |
|--------|-----------------|---------------|----------------------|---------------------|
| **Wins** | 85 (85%) | 90 (90%) | 84 (84%) | **88 (88%)** |
| Draws | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| Losses | 15 (15%) | 10 (10%) | 16 (16%) | **12 (12%)** |
| Valid moves | 619 | 611 | 535 | **802** |
| **Invalid moves** | **919 (60%)** | **609 (50%)** | **0** | **0** |
| Parse errors | 6 | 4 | 69 | **47** |
| Avg moves/game | 11.5 | 11.3 | 9.9 | **15.2** |
| Pipeline time | 1.23s | 0.66s | 1.00s | **7.14s** |

### How Invalid Moves Hit Zero

Three mechanisms work together:
1. **Ensemble voting** (3 votes with ±0.05 temperature jitter) — reduces noise in model output
2. **`valid=` pre-filtering** — prompt includes legal columns so the model learns constraints
3. **`opa_valid_fallback()`** — deterministic safety net picks first legal non-blocked column on parse errors

### Impact of Capacity Scaling (48→96 embed, 2→4 layers)

| Metric | 64K params | 460K params | Δ |
|--------|-----------|------------|---|
| Win rate | 84% | **88%** | +4% |
| Parse errors | 69 | **47** | ↓32% |
| Invalid moves | 0 | 0 | — |

The larger model handles the `valid=` prompt extension more reliably, producing cleaner ensemble votes.

### Comparison with Tic-Tac-Toe

| Dimension | TTT | Connect-4 |
|-----------|-----|-----------|
| Board cells | 9 | 42 |
| State space | ~5,478 | ~4.5 trillion |
| Win+Draw rate | 87% | **88%** |
| Invalid moves | **0** | **0** |
| Parse errors | 18 | **47** |

## Build & Run

```bash
# From repo root
mkdir build && cd build
cmake .. && cmake --build . --target connect4_demo
./connect4_demo    # trains both organelles, then plays 100 games
```

Auto-resumes from checkpoints (`connect4_planner.ckpt`, `connect4_player.ckpt`).

## Key Insight

**Pipeline coordination + ensemble voting + valid-move pre-filtering + capacity scaling creates a robust zero-invalid-move system.** The 7× capacity increase (64K→460K) reduced parse errors by 32% and improved win rate by 4%. The generalist `opa_valid_filter()` and `opa_valid_fallback()` APIs work across all three game demos without game-specific code.

## Recommended Next Steps

| Priority | Change | Impact |
|----------|--------|--------|
| **P1** | MD-delta encoding (like puzzle8) | Pre-compute board evaluation per column |
| **P2** | Test against minimax opponent | Measure real strategic quality |
| **P3** | More training steps (50K+) | Better convergence at 460K params |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
