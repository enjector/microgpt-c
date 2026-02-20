# Hex 7×7 Multi-Organelle Pipeline

A 92K-parameter Transformer plays 7×7 Hex as X against a random opponent — **4% win rate**, revealing that connection-based games demand spatial reasoning beyond character-level pattern matching.

---

## Spear Summary

**Point:** Hex exposes the fundamental limit of flat-string board encoding — the model wins only 4% against random because forming a connected path across a 7×7 hexagonal grid requires spatial reasoning that a 92K-parameter character model cannot learn from 5K training examples.

**Picture:** Imagine trying to describe a road network in a single line of text and asking someone to build a continuous highway across the map. They can learn to place road tiles in valid locations but cannot "see" which tiles connect to form a path — that requires a spatial map in their mind.

**Proof:** 100 games: 10 wins, 0 draws, 90 losses. 2,038 parse errors (50%). 4,118 total moves. Pipeline time: 53.3s. The 10% win rate is close to random play (~7% on 7×7).

**Push:** Connection-aware encoding: label each cell with "distance-to-connected-component" or "bridge cell" markers so the model can learn path-building heuristics.

---

## How It Works

```
┌──────────┐  "board=49chars|turn=X"      ┌──────────┐ "R3C4" ┌──────────┐
│ Planner  │─────────────────────────────▶│  Player  │───────▶│  Judge   │
│ (neural) │  "todo=place,check,place"    │ (neural) │        │ (determ.)│
│  92K par.│                              │  92K par.│        │path check│
└──────────┘                              └──────────┘        └────┬─────┘
      ▲                                                            │
      │ replan if stalls > 3          ┌──────────┐                │
      └───────────────────────────────│  Kanban  │◀───────────────┘
                                      │  State   │ blocked + history
                                      └──────────┘
```

- **Board** is a 49-char string (7×7): `X`, `O`, `.` for empty
- **X goal:** Connect left edge to right edge
- **O goal:** Connect top edge to bottom edge
- **Judge** checks for connected path using flood-fill
- **Hex property:** One side must win (no draws possible)

### Hex Connectivity

Each cell connects to up to 6 hexagonal neighbours. The player must form an unbroken chain of their stones from one edge to the opposite:

```
   0 1 2 3 4 5 6
 0  . . . . . . .
  1  . . . . . . .    X connects: left ↔ right
   2  . . . . . . .   O connects: top ↔ bottom
    3  . . . . . . .
     4  . . . . . . .
      5  . . . . . . .
       6  . . . . . . .
```

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| Board | 7×7 hexagonal (49 cells) |
| Win condition | Connected path edge-to-edge |
| Params/organelle | ~92,000 (Tier 2 — Small) |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `hex_planner.txt` | 4,981 | 495 KB | 0.16 | ~25 min |
| Player | `hex_player.txt` | 4,981 | 563 KB | 0.13 | ~32 min |

**Total: ~57 minutes. 25,000 steps each, multi-threaded.**

### Corpus Generation

- Monte Carlo evaluation-guided self-play
- 500 games with ~40 moves each
- Move selection biased toward cells with higher Monte Carlo win probability

## Results (100 Games vs Random)

| Metric | Value |
|--------|-------|
| **Games won (X)** | **10 / 100 (10%)** |
| Games drawn | 0 |
| Games lost | 90 |
| **Win+Draw rate** | **10%** |
| Total moves | 4,118 (avg 41.2) |
| Parse errors | 2,038 (50%) |
| Planner re-plans | 0 |
| Pipeline time | 53.25s |

### Why Hex Is the Hardest Adversarial Game

| Game | Win % | State Space | Key Reasoning | Encoding Challenge |
|------|-------|-------------|---------------|-------------------|
| Pentago | 90% | Large | Place+twist | Twist creates easy wins |
| Othello | 56% | Medium | Positional | Flip counting |
| **Hex** | **10%** | **Large** | **Path connectivity** | **Global spatial graph** |

The fundamental problem: Hex requires **global path reasoning** across a hexagonal grid. The model must understand:
1. Which cells are connected to which edges
2. Where "bridge" connections (implicit two-cell paths) exist
3. How to prioritise edge-adjacent placements
4. When to block opponent's forming connections

None of this is visible in a 49-char flat string.

### Key Observations

1. **10% ≈ random play** — on 7×7 Hex, random play wins ~7% as first player; the model is barely above random
2. **2,038 parse errors (50%)** — the 49-char board + valid moves exceeds BLOCK_SIZE=128, causing severe truncation
3. **Connection is invisible** — unlike Othello (where piece flips are local) or Pentago (where wins are 5-in-a-row), Hex wins require tracing graph connectivity across the entire board
4. **Zero re-plans despite 90% loss rate** — the planner doesn't know it's losing; it produces consistent (but wrong) plans
5. **Hex is provably hard for local methods** — even Shannon's switching game formulation suggests Hex strategy requires global information

## Build & Run

```bash
# Generate corpus
cd experiments/organelles/hex
python3 generate_corpus.py

# Build and run
cd build
cmake .. && cmake --build . --target hex_demo
./hex_demo    # trains both organelles, then plays 100 games
```

Auto-resumes from checkpoints (`hex_planner.ckpt`, `hex_player.ckpt`).

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| **P1** | Connection-aware encoding: "X-group-1 near left edge, 3 cells from right" | Enable path reasoning from text |
| **P1** | Reduce board to 5×5 (25 cells) to fit BLOCK_SIZE=128 | Lower parse errors from truncation |
| **P2** | Increase BLOCK_SIZE to 256+ for 7×7 representation | Reduced truncation |
| **P2** | Increase WARMUP_STEPS to 500 (per TRAINING_STRATEGIES.md) | Better gradient stability |
| **P3** | Graph-aware encoding with bridge detection | Explicit spatial reasoning hints |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
