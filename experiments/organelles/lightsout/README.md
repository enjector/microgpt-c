# Lights Out Multi-Organelle Pipeline

A 160K-parameter Transformer tackles 5×5 Lights Out — a toggle puzzle with 2²⁵ (~33 million) board states. **10% solve rate** with zero parse errors and zero invalid moves.

---

## Spear Summary

**Point:** Lights Out exposes the limits of flat-string encoding for coupled-constraint puzzles — the model solves 27% of easy boards but 0% of hard ones because toggle mechanics require global reasoning that character-level patterns cannot capture.

**Picture:** Imagine pressing a light switch that also flips every neighbour's switch. Now do that on a 5×5 grid where every press affects up to 5 cells. The model learns to press individual cells correctly but cannot chain the cascading effects across the whole board.

**Proof:** 100 puzzles: 12 solved (8 easy, 4 medium, 0 hard). Zero parse errors, zero invalid moves. All 12 solves were 1-3 press puzzles; anything requiring 4+ coordinated presses failed.

**Push:** Add MD-delta style pre-computation — show the model "if you press cell X the board changes by Y lit cells" so it can learn a greedy heuristic without needing to simulate cascades.

---

## How It Works

```
┌──────────┐  "board=...25chars...|lit=8"  ┌──────────┐"P12" ┌──────────┐
│ Planner  │──────────────────────────────▶│  Player  │─────▶│  Judge   │
│ (neural) │  "todo=toggle,check,toggle"   │ (neural) │      │ (determ.)│
│ 160K par.│                               │ 160K par.│      │toggle+chk│
└──────────┘                               └──────────┘      └────┬─────┘
      ▲                                                            │
      │ replan if stalls > 3          ┌──────────┐                │
      └───────────────────────────────│  Kanban  │◀───────────────┘
                                      │  State   │  blocked + history
                                      └──────────┘
```

- **Board** is a 25-char string: `0` = off, `1` = on
- **Planner** sees board + lit count → outputs toggle/check chain
- **Player** sees board + valid cells + blocked → outputs "PRC" (press row R, col C)
- **Judge** is deterministic — toggles the pressed cell + all orthogonal neighbours, checks all-off
- **Kanban** tracks blocked cells and stall count

### Toggle Mechanics

Pressing cell (r,c) flips the state of (r,c) and all orthogonal neighbours:
```
    [r-1,c]
[r,c-1] [r,c] [r,c+1]
    [r+1,c]
```

A corner press flips 3 cells; an edge press flips 4; a centre press flips 5.

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| MLP_DIM | 384 |
| Board cells | 25 (5×5) |
| State space | 2²⁵ ≈ 33 million |
| Params/organelle | ~160,000 (Tier 3 — Standard) |
| Total neural params | ~920,000 |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `lightsout_planner.txt` | 9,989 | 798 KB | 0.19 | ~10 min |
| Player | `lightsout_player.txt` | 14,476 | 1.5 MB | 0.17 | ~18 min |

**Total: ~28 minutes. 25,000 steps each, multi-threaded.**

### Corpus Generation

- GF(2) Gaussian elimination solver over the toggle matrix
- Generates solvable boards by applying random press sequences
- Planner and player corpora from BFS-optimal press sequences
- Press counts range from 1 to 15+ presses

## Results (100 Puzzles)

| Metric | Value |
|--------|-------|
| **Puzzles solved** | **12 / 100 (12%)** |
| Easy (1–3 presses) | 8 / 30 (27%) |
| Medium (4–7 presses) | 4 / 40 (10%) |
| Hard (8+ presses) | 0 / 30 (0%) |
| Total toggles | 2,664 (avg 26.6) |
| Valid moves | 2,664 |
| **Invalid moves** | **0** |
| **Parse errors** | **0** |
| Planner re-plans | 0 |
| Pipeline time | 51.10s |

### Key Observations

1. **Zero parse errors** — the "PRC" format (press row-col) is simple enough for 100% parsing; the model never produces malformed output
2. **12% solve rate is instructive** — the low rate reveals that coupled-constraint puzzles need **pre-computed encoding**, not raw board strings
3. **All 12 solves were ≤3 presses** — the model can learn individual cell presses but cannot chain 4+ presses where each affects its neighbours
4. **Medium puzzles: 10%** — some medium solves happened through lucky fallback sequences, not learned coordination
5. **Hard puzzles: 0%** — with 8+ required presses, the cascading toggle effects are too complex for character-level pattern matching

### Why Lights Out Is Harder Than Other Games

| Dimension | Tic-Tac-Toe | Mastermind | **Lights Out** |
|-----------|-------------|------------|----------------|
| State space | ~5,478 | ~1,296 | **33 million** |
| Action coupling | Independent | Independent | **Coupled** (5 cells affected per press) |
| Solve rate | 87% | 86% | **12%** |
| Key challenge | Strategy | Deduction | **Global constraint propagation** |

The fundamental difference: in Mastermind and Tic-Tac-Toe, each move is independent. In Lights Out, pressing one cell changes up to 5 others, creating cascading effects. The model needs to reason about the **system of equations** (GF(2) linear algebra), which flat-string encoding cannot represent.

## Build & Run

```bash
# Generate corpus
cd experiments/organelles/lightsout
python3 generate_corpus.py

# Build and run
cd build
cmake .. && cmake --build . --target lightsout_demo
./lightsout_demo    # trains both organelles, then solves 100 puzzles
```

Auto-resumes from checkpoints (`lightsout_planner.ckpt`, `lightsout_player.ckpt`).

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| **P1** | MD-delta encoding: pre-compute "lit-count change per cell press" | Reduce problem to greedy heuristic (like puzzle8) |
| **P1** | Add "inverse press" signal for cells with known solutions | Let model learn which presses cancel which effects |
| **P2** | Increase WARMUP_STEPS to 500 (per TRAINING_STRATEGIES.md) | Minor training stability improvement |
| **P3** | Scale to Medium tier (128/8/6, lr=0.0005, WARMUP=2500) | Better representation for complex toggle patterns |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
