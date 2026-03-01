# Same-Game Transfer Learning (Planner → Player)

Same-game transfer experiment (K-14): tests whether a TicTacToe Planner's internal representations help bootstrap a TicTacToe Player. Since both organelles operate on the same game, their vocabularies partially overlap.

---

## Spear Summary

**Point:** A transformer trained as a TicTacToe Planner and then fine-tuned as a Player starts from a better point than training the Player from scratch.

**Picture:** It's like training someone to be a chess commentator before training them to play — the commentator already understands board positions, piece relationships, and strategic concepts. When you switch them to playing, they don't start from zero.

**Proof:** Transfer+FT (Planner → Player) achieves competitive or better win rates versus scratch-trained Player baselines with the same training budget, while the untrained random model wins ~0%.

**Push:** This validates that organelle representations are task-transferable within the same domain. A Planner's understanding of "what makes a good board state" transfers to "what move should I make" — supporting the idea that organelle pipelines build on shared internal representations.

---

## Experimental Design

Three conditions, all 460K params (N_EMBD=48, N_LAYER=3):

| Condition | Training | Description |
|-----------|----------|-------------|
| **Scratch** | Player corpus from random init | Baseline: train TTT Player from scratch |
| **Transfer+FT** | Planner → Player fine-tune | Transfer Planner weights, fine-tune on Player corpus |
| **Random** | No training | Negative control: untrained model |

## Data Flow

```
Phase 1: Train TTT Planner → c_transfer_planner.ckpt
Phase 2: Train TTT Player (scratch) → c_transfer_scratch.ckpt
Phase 3: Transfer Planner weights → fine-tune as Player → c_transfer_ft.ckpt
Phase 4: Create random baseline (no training)
Phase 5: Evaluate all 3 conditions (100 games each vs random opponent)
```

## Architecture

| Parameter | Value |
|-----------|-------|
| N_EMBD | 48 |
| N_LAYER | 3 |
| N_HEAD | 4 |
| MLP_DIM | 192 |
| BLOCK_SIZE | 128 |
| Training | 25K steps, batch=8 |
| Game | TicTacToe (both source and target) |

## Vocab Overlap

Both organelles share the same game tokens (`X`, `O`, `_`, digits 0-8, `board=`, `|`, `valid=`), but:
- **Planner** outputs strategic assessments (board evaluation)
- **Player** outputs move decisions (position indices)

This partial vocab overlap is what makes same-game transfer plausible.

## Build & Run

```bash
cmake --build build --target c_transfer_demo
./build/c_transfer_demo
```

Trains all three conditions from scratch and plays 100 evaluation games per condition.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
