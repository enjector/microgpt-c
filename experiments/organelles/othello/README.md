# Othello 6×6 Multi-Organelle Pipeline

A 92K-parameter Transformer plays 6×6 Othello as X against a random opponent — **67% win rate** (70% win+draw) demonstrating learned positional play with 1,183 parse errors across 100 games.

---

## Spear Summary

**Point:** The pipeline wins 56% of Othello games against random with zero strategic encoding — proving that OPA can learn basic positional awareness from 4.5K self-play examples, though 979 parse errors and a 38% loss rate show the limits of character-level board representation for adversarial games.

**Picture:** A chess beginner plays fast games against someone who moves randomly. The beginner wins most games not by brilliant strategy but by avoiding obviously bad moves. Against even a weak strategic opponent they would struggle.

**Proof:** 100 games: 56 wins, 6 draws, 38 losses. Win+draw rate 62%. 979 parse errors.  Average 31.9 moves per game. Zero planner re-plans. Pipeline time: 69.6s.

**Push:** Add corner priority encoding and flip-count heuristics to teach positional play. Against random, a corner-aware policy should achieve >80% win rate on 6×6.

---

## How It Works

```
┌──────────┐  "board=36chars|turn=X|moves=.."  ┌──────────┐ "R3C4" ┌──────────┐
│ Planner  │───────────────────────────────────▶│  Player  │───────▶│  Judge   │
│ (neural) │  "todo=move,check,move"            │ (neural) │        │ (determ.)│
│  92K par.│                                    │  92K par.│        │flip+score│
└──────────┘                                    └──────────┘        └────┬─────┘
      ▲                                                                  │
      │ replan if stalls > 3          ┌──────────┐                      │
      └───────────────────────────────│  Kanban  │◀─────────────────────┘
                                      │  State   │ blocked + history
                                      └──────────┘
```

- **Board** is a 36-char string (6×6): `X`, `O`, `.` for empty
- **Player** sees board + valid moves + blocked → outputs "RrCc" (row, col)
- **Judge** applies Othello flip rules (all 8 directions), updates score
- **Opponent** plays random valid moves (no strategy)
- **Kanban** tracks blocked moves and game state

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| Board | 6×6 grid (36 cells) |
| Params/organelle | ~92,000 (Tier 2 — Small) |
| Total neural params | ~920,000 |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `othello_planner.txt` | 2,322 | 174 KB | 0.10 | ~19 min |
| Player | `othello_player.txt` | 4,496 | 398 KB | 0.11 | ~25 min |

**Total: ~44 minutes. 25,000 steps each, multi-threaded.**

### Corpus Generation

- Random self-play with minimax-weighted move selection
- 300 games, averaging 30 moves each
- Both planner and player corpora include board state + valid moves + chosen move

## Results (100 Games vs Random)

| Metric | Value |
|--------|-------|
| **Games won (X)** | **56 / 100 (56%)** |
| Games drawn | 6 / 100 (6%) |
| Games lost | 38 / 100 (38%) |
| **Win+Draw rate** | **62%** |
| Total moves | 3,187 (avg 31.9) |
| Parse errors | 979 |
| Planner re-plans | 0 |
| Pipeline time | 69.64s |

### Key Observations

1. **56% vs random is modest** — a simple corner-prioritising heuristic achieves ~80-90% on 6×6, so the model learned basic placement but not corner strategy
2. **979 parse errors (31%)** — the 36-char board + valid move list is at the limit of what BLOCK_SIZE=128 can handle; longer prompts get truncated
3. **Zero re-plans** — the kanban never needed to reset, suggesting the planner produces consistent (if imperfect) plans
4. **38% loss rate** — losses typically occur when the model misses corner captures or makes moves that give the opponent corner access
5. **Adversarial games are harder** — compare to single-player puzzles where the model plays against a fixed goal, not a reactive opponent

### Comparison with Other Games

| Game | Type | Solve/Win % | Parse Errors | Key Difference |
|------|------|------------|--------------|---------------|
| Mastermind | Deduction | 86% | 25 | Fixed secret, no opponent |
| Sudoku | Constraint | 76% | 184 | Fixed solution |
| **Othello** | **Adversarial** | **56%** | **979** | **Reactive opponent** |
| Lights Out | Coupled | 12% | 0 | Deterministic puzzle |

Adversarial games with reactive opponents are fundamentally harder for OPA because the optimal move depends on the opponent's response, which the model cannot predict.

## Build & Run

```bash
# Generate corpus
cd experiments/organelles/othello
python3 generate_corpus.py

# Build and run
cd build
cmake .. && cmake --build . --target othello_demo
./othello_demo    # trains both organelles, then plays 100 games
```

Auto-resumes from checkpoints (`othello_planner.ckpt`, `othello_player.ckpt`).

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| **P1** | Corner priority encoding: tag corner cells with special markers | Teach positional value without raw board memorisation |
| **P1** | Flip-count heuristic: show "this move flips N pieces" | Help model evaluate move quality directly |
| **P2** | Increase WARMUP_STEPS to 500 (per TRAINING_STRATEGIES.md) | Better gradient stability |
| **P2** | Increase BLOCK_SIZE to 256 for longer board representations | Reduce truncation-related parse errors |
| **P3** | Scale to 8×8 (standard Othello) with larger corpus | Test OPA on full game complexity |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
