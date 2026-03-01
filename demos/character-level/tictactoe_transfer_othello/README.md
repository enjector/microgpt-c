# TicTacToe → Othello Transfer Learning

Cross-game transfer experiment (K-12/K-13): trains a TicTacToe player, then transfers its internal representations to bootstrap an Othello player. Tests whether board-game knowledge generalises across different games.

---

## Spear Summary

**Point:** A transformer trained on TicTacToe and then fine-tuned on Othello learns faster and plays better than one trained on Othello from scratch.

**Picture:** It's like teaching someone checkers before chess — even though the games have different rules, the spatial reasoning transfers. The model learns "how to think about boards" from the simple game, then applies that thinking to the complex one.

**Proof:** Transfer+FT achieves higher win rates against random opponents than scratch-trained Othello with the same number of training steps, while the untrained random baseline wins ~0%.

**Push:** Use this as evidence that organelle internals encode general board-game reasoning, not just game-specific moves. This supports the composability thesis — organelles aren't black boxes, they build transferable representations.

---

## Experimental Design

Four conditions, all 460K params (N_EMBD=48, N_LAYER=3):

| Condition | Source | Target | Description |
|-----------|--------|--------|-------------|
| **TTT Source** | TicTacToe player corpus | — | Phase 1: train source organelle |
| **Othello Scratch** | Othello player corpus | — | Baseline: train from random init |
| **Transfer+FT** | TicTacToe → Othello | Othello player corpus | Transfer TTT weights, fine-tune on Othello |
| **Random** | — | — | Negative control: untrained model |

## Data Flow

```
Phase 1: Train TTT → c_tto_ttt.ckpt
Phase 2: Train Othello (scratch) → c_tto_scratch.ckpt
Phase 3: Transfer TTT weights → fine-tune on Othello → c_tto_transfer.ckpt
Phase 4: Create random baseline (no training)
Phase 5: Evaluate all 4 conditions (100 games each vs random opponent)
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
| Board | 8×8 Othello (target), 3×3 TTT (source) |

## Build & Run

```bash
cmake --build build --target c_tictactoe_transfer_othello
./build/c_tictactoe_transfer_othello
```

Trains all four conditions from scratch and plays 100 evaluation games per condition.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
