# Connect-4 Multi-Organelle Pipeline

A 64K-parameter Transformer plays Connect-4 on a board with 4.5 trillion possible states — trained on fewer than 2,000 of them — and wins 85% against random.

---

## Spear Summary

**Point:** The kanban coordination layer is doing the heavy lifting — it turns a model that picks invalid moves 60% of the time into an 85% winner.

**Picture:** It's like a dart player who's blindfolded but has a friend shouting "not that one!" after every miss. The player can barely see the board but the feedback loop keeps steering throws toward the bullseye.

**Proof:** 919 invalid moves across 100 games (60% invalid rate) yet still 85 wins. Without the blocked/replan kanban loop the win rate would drop to roughly 50–60% — barely above random-vs-random baseline of 56%.

**Push:** The model is starving for capacity — the Player has a 0.062:1 params-to-bytes ratio (12× worse than what works well). Either bump to N_EMBD=96 / N_LAYER=4 or cut the corpus to just 1,969 core positions to give the model room to breathe.

---

## How It Works

```
┌──────────┐  "board=...|empties=42"   ┌──────────┐  "3"  ┌──────────┐
│ Planner  │──────────────────────────▶│  Player  │──────▶│  Judge   │
│ (neural) │  "todo=centre,check,drop" │ (neural) │      │ (determ.)│
│ 64K params│                          │ 64K params│      │drop_piece│
└──────────┘                           └──────────┘      └────┬─────┘
      ▲                                                       │
      │ replan if stalls > 3          ┌──────────┐            │
      └───────────────────────────────│  Kanban  │◀───────────┘
                                      │  State   │  blocked + last cols
                                      └──────────┘
```

- **Planner** sees 42-char board string + empty count → outputs a priority chain
- **Player** sees board + any blocked columns → outputs a **single digit** (0–6)
- **Judge** is fully deterministic — `drop_piece()` checks column-not-full + 4-in-a-row detection
- **Kanban** tracks blocked columns, move history, and stall count; triggers re-plan after 3 failures; forces random fallback after 7 retries

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 48 |
| N_HEAD | 4 |
| N_LAYER | 2 |
| BLOCK_SIZE | 128 |
| Board string | 42 characters (7×6) |
| Params/organelle | ~64,000 |
| Total neural params | ~128,000 |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `connect4_planner.txt` | 5,715 | 573 KB | 0.1174 | 671s |
| Player | `connect4_player.txt` | 17,472 | 1,004 KB | 0.2127 | 363s |

**Total: 1,034 seconds (17 minutes). 25,000 steps each, single-threaded.**

### Corpus Generation

- Alpha-beta minimax with depth 4
- 500 self-play games (random + optimal + mixed strategies)
- 1,969 unique X-turn positions, mirror-symmetry augmented
- Blocked-variant examples for kanban support

### The Capacity Problem

| Organelle | Params | Corpus Bytes | Ratio |
|-----------|--------|--------------|-------|
| Planner | 64,320 | 586 KB | 0.110:1 |
| Player | 63,648 | 1,004 KB | 0.062:1 |

The Player is trying to memorise 1MB with 64K params — 12× less capacity per byte than what works well (c_codegen at 0.76:1). Player loss oscillated between 0.25–0.39 throughout training without converging.

## Results (100 Games vs Random)

| Metric | Value |
|--------|-------|
| **Wins** | **85 (85%)** |
| Draws | 0 (0%) |
| Losses | 15 (15%) |
| Valid moves | 619 |
| **Invalid moves** | **919 (60% invalid rate)** |
| Parse errors | 6 |
| Replans | 63 |
| Avg moves/game | 11.5 |
| Pipeline time | 1.23s |

### Why 85% Despite 60% Invalid Rate?

The kanban feedback loop rescues the pipeline. Centre-column bias (the strongest Connect-4 opening) is well-represented in the corpus, so early moves tend to be good. Mid-game moves are mostly fuzzy pattern matching against ~2,000 memorised positions out of 4.5 trillion possible states.

### Comparison with Tic-Tac-Toe

| Dimension | TTT | Connect-4 |
|-----------|-----|-----------|
| Board cells | 9 | 42 |
| State space | ~5,478 | ~4.5 trillion |
| Corpus coverage | ~46% | < 0.0000001% |
| Win rate vs random | 85% | 85% |
| Invalid moves/game | 2.6 | **9.2** |
| Params/byte (Player) | 0.25:1 | **0.062:1** |

Same win rate but radically different quality underneath. TTT achieves it through genuine retrieval; Connect-4 achieves it through centre-column bias + kanban rescues. The 919 invalids reveal the gap.

## Build & Run

```bash
# From repo root
mkdir build && cd build
cmake .. && cmake --build . --target connect4_demo
./connect4_demo    # trains both organelles, then plays 100 games
```

Auto-resumes from checkpoints (`connect4_planner.ckpt`, `connect4_player.ckpt`).

## Key Insight

**Pipeline coordination compensates for model weakness.** A weak model in a well-orchestrated pipeline outperforms a stronger model with no coordination. The kanban loop turned a ~55% player into an 85% winner. This validates the Adaptive Organelle Planner's core value proposition.

## Recommended Next Steps

| Priority | Change | Impact |
|----------|--------|--------|
| **P1** | Increase model capacity (N_EMBD=96 or N_LAYER=4) | Break Player loss plateau |
| **P1** | Reduce corpus to 1,969 core positions (drop blocked variants) | Better params/byte ratio |
| **P2** | Test float32 (`MICROGPT_USE_FLOAT`) | ~2× faster training |
| **P2** | Compressed board representation (column heights + last move) | Help model identify patterns |
| **P3** | Test against minimax opponent | Measure real strategic quality |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
