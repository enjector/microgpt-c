# Red Donkey (Huarong Dao) Multi-Organelle Pipeline

A 30K-parameter Transformer solves simplified 4×3 Red Donkey sliding block puzzles — **12% solve rate** from only 199 training positions, limited by a tiny corpus and high parse error rate.

---

## Spear Summary

**Point:** The simplified 4×3 Red Donkey achieves 30% solve rate with 93 re-plans and 2286 parse errors — demonstrating that even BFS-optimal training data cannot compensate for a corpus smaller than the model's capacity.

**Picture:** Imagine teaching someone to solve a 12-piece sliding puzzle by showing them only 199 examples, then asking them to solve 100 new ones. They learn the basic idea of "slide a block" but cannot reliably produce valid moves, so the safety net picks random valid slides — which sometimes stumble into a solution.

**Proof:** 100 puzzles: 30 solved, 2286 parse errors (54%), 93 re-plans. Player corpus had only 199 entries — the absolute minimum for this game. BFS-optimal solutions help quality but cannot fix quantity.

**Push:** Expand corpus to 2000+ positions by using deeper scrambles from more starting layouts. The classic 5×4 Red Donkey remains intractable for BFS; consider A* with Manhattan distance heuristic.

---

## How It Works

```
┌──────────┐  "board=AA.BC.DE...|goal=A@bottom" ┌──────────┐"A:D" ┌──────────┐
│ Planner  │────────────────────────────────────▶│  Player  │─────▶│  Judge   │
│ (neural) │  "todo=slide,check,slide"           │ (neural) │      │ (determ.)│
│  30K par.│                                     │  30K par.│      │collision │
└──────────┘                                     └──────────┘      └────┬─────┘
      ▲                                                                 │
      │ replan if stalls > 3           ┌──────────┐                    │
      └────────────────────────────────│  Kanban  │◀───────────────────┘
                                       │  State   │ blocked + history
                                       └──────────┘
```

- **Board** is a 12-char string (4×3 grid): `A` = 2×2 donkey, `B`–`E` = 1×1 blocks, `.` = empty
- **Goal:** Move the 2×2 donkey (A) to the bottom-left corner (rows 2–3, cols 0–1)
- **Player** sees board + valid moves + blocked → outputs "B:D" (block, direction)
- **Judge** checks collision bounds for multi-cell block (A is 2×2), applies slide
- **Kanban** tracks blocked moves and stall count

### Simplified Layout

The classic 5×4 Red Donkey (Huarong Dao) has ~10¹⁰ states, making BFS intractable. This simplified 4×3 variant has ~50,000 reachable states, enabling BFS-optimal corpus generation.

```
Starting:     Goal:
A A B         . . .
A A C         . . .
. . D         A A .
. . E         A A .
```

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| Board | 4×3 grid (12 cells) |
| Block types | 2×2 (donkey) + 1×1 × 4 |
| State space | ~50,000 reachable |
| Params/organelle | ~30,000 (Tier 1 — Micro) |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `reddonkey_planner.txt` | 199 | 10 KB | 0.08 | ~17 min |
| Player | `reddonkey_player.txt` | 199 | 9 KB | 0.10 | ~16 min |

**Total: ~33 minutes. 25,000 steps each, multi-threaded.**

### Corpus Generation

- BFS solver with max 50,000 states explored
- Starting layout scrambled via random valid moves
- BFS finds optimal solution path for each scrambled position
- 500 scramble attempts → 199 solvable positions (88 unique BFS paths)

## Results (100 Puzzles)

| Metric | Value |
|--------|-------|
| **Puzzles solved** | **30 / 100 (30%)** |
| Total slides | 4,249 (avg 42.5) |
| Parse errors | 2,286 (54%) |
| Planner re-plans | 93 |
| Pipeline time | 49.99s |

### Key Observations

1. **30% solve rate despite 54% parse errors** — unlike Klotski (59% with 87% parse errors), the 4×3 grid has fewer lucky random walks to the goal
2. **199-entry corpus is critically small** — compare to Mastermind (2K entries → 86%) and Sudoku (20K → 76%)
3. **93 re-plans** — the model gets stuck frequently, with the kanban triggering replans almost once per puzzle
4. **2×2 block constraint is harder** — moving the donkey requires checking 4 cells simultaneously, unlike 1×1 blocks in Klotski
5. **Training loss was excellent** (planner 0.08, player 0.10) — the model memorised the tiny corpus perfectly but couldn't generalise

### Corpus Size vs Performance

| Game | Corpus Size | Solve Rate | Parse Errors |
|------|-------------|-----------|--------------|
| Sudoku | 20,000 | 76% | 184 |
| Mastermind | 1,975 | 86% | 25 |
| Klotski | 232 | 59% | 1,849 |
| **Red Donkey** | **199** | **30%** | **2,286** |

Clear trend: below ~500 corpus entries, parse errors dominate and solve rate drops sharply.

## Build & Run

```bash
# Generate corpus
cd experiments/organelles/reddonkey
python3 generate_corpus.py

# Build and run
cd build
cmake .. && cmake --build . --target reddonkey_demo
./reddonkey_demo    # trains both organelles, then solves 100 puzzles
```

Auto-resumes from checkpoints (`reddonkey_planner.ckpt`, `reddonkey_player.ckpt`).

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| **P1** | Expand corpus to 2000+ (more scramble depths, multiple starts) | Reduce parse errors from 54% to <20% |
| **P1** | MD-delta encoding: pre-compute donkey distance-to-goal per move | Guide model toward optimal slides |
| **P2** | Increase WARMUP_STEPS to 500 (per TRAINING_STRATEGIES.md) | Better gradient stability |
| **P3** | Scale to classic 5×4 grid with A* solver (not BFS) | Test OPA on full Huarong Dao complexity |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
