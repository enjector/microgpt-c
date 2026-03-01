# Hex Multi-Organelle Pipeline

A 92K-parameter Transformer plays Hex as X against a random opponent — **27% win rate on 7×7** (uplifted from 4% via topology features + MCTS corpus) and **32% win rate on 5×5** (further improvement from reduced encoding pressure).

---

## Spear Summary

**Point:** BFS connectivity features + topological Judge + MCTS corpus turned Hex from near-random (4%) to 27% win rate on 7×7 — the Judge alone (no retraining) gave a 6.25× improvement, validating that better filters beat smarter models.

**Picture:** Like giving a blind chess player a map showing which pieces connect rather than just listing coordinates — the model can't see paths in flat text, but structural features make connectivity implicit.

**Proof:** 7×7: 27% win (up from 4%), parse errors dropped 50%→17%. 5×5: 32% win with halved prompt length. Deeper MCTS (500 vs 200 iterations) shows no benefit on 5×5 — search converges early on smaller boards.

**Push:** Virtual connection templates (precomputed bridge patterns) are the next lever. K-9 in the Kanban backlog.

---

## How It Works

```
┌──────────┐  "board+topo features"       ┌──────────┐ "R3C4" ┌──────────┐
│ Planner  │─────────────────────────────▶│  Player  │───────▶│  Judge   │
│ (neural) │  "todo=place,check,place"    │ (neural) │        │ (determ.)│
│  92K par.│                              │  92K par.│        │topo check│
└──────────┘                              └──────────┘        └────┬─────┘
      ▲                                                            │
      │ replan if stalls > 3          ┌──────────┐                │
      └───────────────────────────────│  Kanban  │◀───────────────┘
                                      │  State   │ blocked + history
                                      └──────────┘
```

- **Board** is an NxN string: `X`, `O`, `.` for empty (7×7 = 49 chars, 5×5 = 25 chars)
- **Topo features** in prompt: `xg` (X groups), `xd` (X edge distance), `og`, `od`, `xb` (X bridges)
- **Judge** rejects placements not adjacent to any friendly stone → first connected alternative
- **MCTS corpus** uses UCB1 selection (200 iterations) for higher-quality training data

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 48 |
| N_HEAD | 4 |
| N_LAYER | 3 |
| BLOCK_SIZE | 128 |
| Params/organelle | ~92,000 (Tier 2) |

## Results

### 7×7 Hex (Topology Uplift Progression)

| Variant | Win Rate | Parse Errors | Corpus Size |
|---------|---------|-------------|-------------|
| Original baseline (MC, flat encoding) | 4% | 50% | 4,981 |
| + Judge only (no retrain) | 25% | 50% | 4,981 |
| + Enriched encoding (MC) | 19% | 18% | 10,707 |
| **+ MCTS corpus (best)** | **27%** | **17%** | **13,510** |

### 5×5 Hex Variant

| Metric | Value |
|--------|-------|
| **Games won (X)** | **32 / 100 (32%)** |
| Games lost | 68 |
| Total moves | 2,152 (avg 21.5) |
| Parse errors | 387 |
| Pipeline time | 2.0s |

### Deeper MCTS (500 iterations, 5×5)

No additional benefit — identical corpus generated (same MD5). On a 5×5 board, 200 MCTS iterations already converges. The bottleneck on 5×5 is model capacity and Judge heuristics, not corpus quality.

### Cross-Board Comparison

| Board | State Space | Win Rate | Parse Errors | Key Finding |
|-------|-------------|---------|-------------|-------------|
| 7×7 | ~10^10 | 27% | 17% | Topology features provide 6.75× uplift |
| 5×5 | ~10^7 | 32% | high | Smaller board = less encoding pressure |

## Build & Run

```bash
# Generate 7×7 corpus
cd demos/character-level/hex
python3 generate_corpus.py

# Build and run 7×7
cmake --build build --target hex_demo && ./build/hex_demo

# Build and run 5×5
cmake --build build --target hex5_demo && ./build/hex5_demo
```

`main.c` supports compile-time grid size via `HEX_GRID` define (default: 7). The 5×5 variant uses `HEX_GRID=5`.

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| **P1** | Virtual connection templates (bridge patterns) | Strategic knowledge beyond local adjacency |
| **P2** | BLOCK_SIZE=256 for 7×7 | Eliminate remaining parse errors |
| **P3** | Deeper MCTS on 7×7 (500+ iterations) | May help where search tree is wider |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
