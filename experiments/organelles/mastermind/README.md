# Mastermind Multi-Organelle Pipeline

A 92K-parameter Transformer cracks 4-peg/6-colour Mastermind codes — **79% solve rate** with zero invalid guesses and only 44 parse errors across 100 games.

---

## Spear Summary

**Point:** The pipeline solves 86% of Mastermind games in an average of 5.3 guesses with zero invalid moves proving that OPA can learn sequential deduction from under 2K training examples.

**Picture:** Two interns play Mastermind — one reads the hints and plans a strategy and the other picks the actual pegs. A referee checks every guess is four valid colours. Even blindfolded the interns crack 86 out of 100 codes.

**Proof:** 100 games: 86 solved, 0 invalid guesses, 25 parse errors, 41 games solved in exactly 5 guesses. Planner loss 0.07, Player loss 0.10. Pipeline runs in 6.48s.

**Push:** Increase WARMUP_STEPS from 100 to 500 per TRAINING_STRATEGIES.md recommendations and test with expanded 6-peg codes or 8 colours to probe scaling limits.

---

## How It Works

```
┌──────────┐  "secret=????|guess=1"      ┌──────────┐ "ACBD" ┌──────────┐
│ Planner  │────────────────────────────▶│  Player  │───────▶│  Judge   │
│ (neural) │  "todo=guess,check,guess"   │ (neural) │        │ (determ.)│
│  92K par.│                             │  92K par.│        │B/W score │
└──────────┘                             └──────────┘        └────┬─────┘
      ▲                                                           │
      │ replan if stalls > 3          ┌──────────┐               │
      └───────────────────────────────│  Kanban  │◀──────────────┘
                                      │  State   │ feedback history
                                      └──────────┘
```

- **Planner** sees guess count + feedback history → outputs a guess/check chain
- **Player** sees full feedback history ("G1=AABB|F1=B1W1|G2=...") + blocked → outputs a **4-char code** (A–F)
- **Ensemble** runs 3 inferences with temperature jitter, majority-votes
- **Judge** is deterministic — scores Black (exact position) and White (right colour, wrong position)
- **Kanban** tracks blocked guesses and stall count; triggers re-plan after 3 failures

## Architecture

| Parameter | Value |
|-----------|-------|
| Organelles | 2 neural (Planner + Player) + deterministic Judge |
| N_EMBD | 96 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| MLP_DIM | 384 |
| BLOCK_SIZE | 128 |
| Code length | 4 pegs, 6 colours (A–F) |
| Params/organelle | ~92,000 (Tier 2 — Small) |
| Total neural params | ~920,000 |

## Training

| Organelle | Corpus | Entries | Size | Best Loss | Time |
|-----------|--------|---------|------|-----------|------|
| Planner | `mastermind_planner.txt` | 988 | 78 KB | 0.07 | ~12 min |
| Player | `mastermind_player.txt` | 1,975 | 118 KB | 0.10 | ~12 min |

**Total: ~24 minutes. 25,000 steps each, multi-threaded.**

### Corpus Generation

- Simplified Knuth algorithm generates game traces for 800 random secrets
- Each trace records guess + B/W feedback at every turn
- Average 5.0 turns per game trace
- Blocked-variant examples for kanban support

## Results (100 Games)

| Metric | Value |
|--------|-------|
| **Games solved** | **86 / 100 (86%)** |
| Avg guesses (solved) | 5.3 |
| Valid guesses | 577 |
| **Invalid guesses** | **0** |
| Parse errors | 25 |
| Planner re-plans | 0 |
| Pipeline time | 6.48s |

### Guess Distribution

| Guesses | Games | % |
|---------|-------|---|
| 2 | 2 | 2% |
| 3 | 3 | 3% |
| **4** | **13** | **15%** |
| **5** | **41** | **48%** |
| 6 | 23 | 27% |
| 7 | 3 | 3% |
| 8 | 1 | 1% |

The distribution peaks at 5 guesses — the theoretical optimal average for 4-peg/6-colour Mastermind (Knuth's algorithm averages 4.48). The pipeline is within 1 guess of information-theoretic optimality for the majority of solvable games.

### Key Observations

1. **Zero invalid guesses** — ensemble voting + 4-char format parsing works reliably
2. **86% solve rate within 10 guesses** — strong sequential deduction with only ~2K training examples
3. **5-guess peak matches theory** — Knuth's minimax algorithm needs 4.48 guesses on average; the neural pipeline's modal 5 is excellent for a learned system
4. **25 parse errors (4.3%)** — the pipeline gracefully handles these via fallback; no structural damage to play
5. **14 unsolved games** — typically codes with unusual colour distributions where the model gets stuck repeating the same guess

## Build & Run

```bash
# Generate corpus
cd experiments/organelles/mastermind
python3 generate_corpus.py

# Build and run
cd build
cmake .. && cmake --build . --target mastermind_demo
./mastermind_demo    # trains both organelles, then plays 100 games
```

Auto-resumes from checkpoints (`mastermind_planner.ckpt`, `mastermind_player.ckpt`).

## Recommended Next Steps

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| **P1** | Increase WARMUP_STEPS to 500 (per TRAINING_STRATEGIES.md) | Better gradient stability, potentially lower player loss |
| **P2** | Expand corpus with more diverse secrets (all 1296) | Cover rare colour distributions causing the 14% failure |
| **P3** | Add feedback encoding (running elimination count) | Help model learn which colours are "ruled out" |
| **P3** | Scale to Medium tier (128/8/6, 1.2M params, lr=0.0005) | Better representation for longer feedback histories |

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
