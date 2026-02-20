# Pretrained Models

MicroGPT-C ships with pretrained checkpoints ready for **instant inference** — no training required.

---

## Directory Layout

```
models/
  README.md                                 This file

  foundation/                               Base demo checkpoints
    shakespeare.ckpt                        840K params — character-level Shakespeare
    shakespeare.ckpt.log                    Training log (loss curve, timing, samples)
    names.ckpt.log                          Training log for name generation demo

  organelles/                               Organelle game + code checkpoints
    connect4_planner.ckpt  /  .ckpt.log     460K — Connect-4
    connect4_player.ckpt   /  .ckpt.log
    tictactoe_planner.ckpt /  .ckpt.log     460K — Tic-Tac-Toe
    tictactoe_player.ckpt  /  .ckpt.log
    puzzle8_strategist_v3b.ckpt / .ckpt.log 460K — 8-Puzzle (5 organelles)
    puzzle8_mover_v3b.ckpt      / .ckpt.log
    puzzle8_detector_v3b.ckpt   / .ckpt.log
    puzzle8_detour_mover_v3b.ckpt / .ckpt.log
    puzzle8_judge_v3b.ckpt      / .ckpt.log
    pentago_planner.ckpt   /  .ckpt.log      92K — Pentago
    pentago_player.ckpt    /  .ckpt.log
    mastermind_planner.ckpt / .ckpt.log      92K — Mastermind
    mastermind_player.ckpt  / .ckpt.log
    othello_planner.ckpt   /  .ckpt.log      92K — Othello
    othello_player.ckpt    /  .ckpt.log
    hex_planner.ckpt       /  .ckpt.log      92K — Hex
    hex_player.ckpt        /  .ckpt.log
    sudoku_planner.ckpt    /  .ckpt.log     160K — Sudoku
    sudoku_player.ckpt     /  .ckpt.log
    lightsout_planner.ckpt /  .ckpt.log     160K — Lights Out
    lightsout_player.ckpt  /  .ckpt.log
    klotski_planner.ckpt   /  .ckpt.log      30K — Klotski
    klotski_player.ckpt    /  .ckpt.log
    reddonkey_planner.ckpt /  .ckpt.log      30K — Red Donkey
    reddonkey_player.ckpt  /  .ckpt.log
    c_codegen.ckpt         /  .ckpt.log     875K — C code generation
    c_wiringgen.ckpt       /  .ckpt.log     875K — C wiring generation
    c_planner.ckpt         /  .ckpt.log     1.2M — c_compose planner
    c_judge.ckpt           /  .ckpt.log     1.2M — c_compose judge
```

Each `.ckpt` file contains model weights (float32) + Adam optimizer state + training step counter.
Each `.ckpt.log` file contains the full training history (loss per step, timings, architecture).

---

## Available Foundation Models

| Model | Size | Params | Trained On | What It Does |
|-------|------|--------|-----------|-------------|
| `foundation/shakespeare.ckpt` | 9.6 MB | 840K | Complete works of Shakespeare | Character-level text generation |
| `organelles/c_codegen.ckpt` | 20 MB | 875K | 2,081 C functions (math, physics, DSP) | Code retrieval — byte-perfect recall |
| `organelles/c_wiringgen.ckpt` | 19.9 MB | 875K | C function compositions | Multi-function pipeline generation |
| `organelles/c_planner.ckpt` | 13.8 MB | 1.2M | Function composition plans | Planner for c_compose pipeline |
| `organelles/c_judge.ckpt` | 13.8 MB | 1.2M | Plan validation pairs | Judge for c_compose pipeline |

---

## Organelle Game Checkpoints

**31 checkpoints** across 11 games, using 4 parameter tiers based on corpus complexity:

| Tier | Config (EMBD/HEAD/LAYER/MLP) | Params | Games |
|------|------------------------------|--------|-------|
| Micro | 32/4/2/128 | ~30K | Klotski, Red Donkey |
| Small | 48/4/3/192 | ~92K | Mastermind, Pentago, Othello, Hex |
| Standard | 64/4/3/256 | ~160K | Lights Out, Sudoku |
| Legacy | 96/8/4/384 | ~460K | Connect-4, Tic-Tac-Toe, 8-Puzzle |

### Game Results

| Game | Checkpoints | Params | Result |
|------|------------|-------:|-------:|
| **Pentago** | `pentago_planner/player.ckpt` | 92K | **91% win** |
| **Connect-4** | `connect4_planner/player.ckpt` | 460K | **90% win** |
| **Tic-Tac-Toe** | `tictactoe_planner/player.ckpt` | 460K | **90% w+d** |
| **Mastermind** | `mastermind_planner/player.ckpt` | 92K | **79% solve** |
| **Sudoku** | `sudoku_planner/player.ckpt` | 160K | **78% solve** |
| **Othello** | `othello_planner/player.ckpt` | 92K | **67% win** |
| **Klotski** | `klotski_planner/player.ckpt` | 30K | **62% solve** |
| **8-Puzzle** | `puzzle8_*_v3b.ckpt` (×5) | 460K | **60% solve** |
| **Red Donkey** | `reddonkey_planner/player.ckpt` | 30K | **12% solve** |
| **Lights Out** | `lightsout_planner/player.ckpt` | 160K | **10% solve** |
| **Hex** | `hex_planner/player.ckpt` | 92K | **4% win** |

### Usage

Copy checkpoints to the build directory. The demos auto-detect and skip training:

```bash
cp ../models/organelles/connect4_*.ckpt .
./connect4_demo    # skips training, plays 100 games immediately
```

---

## Using a Pretrained Model

### Quickest path: just run the demo

```bash
cd build

# Copy pretrained checkpoint from models/ to build/
cp ../models/foundation/shakespeare.ckpt .

# Run — training is skipped, goes straight to generation
./shakespeare_demo
```

Output:
```
loaded checkpoint 'shakespeare.ckpt' (trained 30000 steps) — skipping training
--- generated Shakespeare (character-level) ---
[sample 1 — seed: 'T']
TSERVICE THAMANT. Why, that command of me such'd good the live of
```

### From your own code

Use `checkpoint_load()` to load a checkpoint and `forward_inference()` to generate tokens:

```c
#include "microgpt.h"

// 1. Set up config matching the checkpoint's architecture
MicrogptConfig cfg = microgpt_default_config();
cfg.n_embd = 128;
cfg.n_head = 8;
cfg.n_layer = 4;
cfg.block_size = 256;

// 2. Load the pretrained checkpoint
scalar_t *m_adam = calloc(nparams, sizeof(scalar_t));
scalar_t *v_adam = calloc(nparams, sizeof(scalar_t));
int step = 0;
Model *model = checkpoint_load("shakespeare.ckpt", vocab_size,
                                &cfg, m_adam, v_adam, &step);

// 3. Generate token by token
scalar_t *logits = malloc(vocab_size * sizeof(scalar_t));
scalar_t **keys = ...; scalar_t **values = ...; size_t *cache_len = ...;

size_t token = seed_token;
for (int i = 0; i < max_tokens; i++) {
    forward_inference(model, token, i, keys, values, cache_len, logits);
    token = sample_token(logits, vocab_size, temperature);
    printf("%c", vocab.chars[token]);
}
```

---

## Architecture Requirements

Each checkpoint requires matching compile-time architecture. The demos handle this automatically via `CMakeLists.txt`.

| Checkpoint | N_EMBD | N_HEAD | N_LAYER | BLOCK_SIZE | MLP_DIM |
|-----------|--------|--------|---------|------------|---------| 
| `shakespeare.ckpt` | 128 | 8 | 4 | 256 | 512 |
| `organelles/c_codegen.ckpt` | 128 | 4 | 4 | 512 | 512 |
| `organelles/c_wiringgen.ckpt` | 128 | 4 | 4 | 512 | 512 |
| `organelles/c_planner.ckpt` / `c_judge.ckpt` | 128 | 8 | 6 | 128 | 512 |
| `organelles/connect4_*.ckpt` | 96 | 8 | 4 | 128 | 384 |
| `organelles/tictactoe_*.ckpt` | 96 | 8 | 4 | 128 | 384 |
| `organelles/puzzle8_*.ckpt` | 96 | 8 | 4 | 128 | 384 |
| `organelles/mastermind_*.ckpt` | 48 | 4 | 3 | 128 | 192 |
| `organelles/pentago_*.ckpt` | 48 | 4 | 3 | 128 | 192 |
| `organelles/othello_*.ckpt` | 48 | 4 | 3 | 128 | 192 |
| `organelles/hex_*.ckpt` | 48 | 4 | 3 | 128 | 192 |
| `organelles/klotski_*.ckpt` | 32 | 4 | 2 | 128 | 128 |
| `organelles/reddonkey_*.ckpt` | 32 | 4 | 2 | 128 | 128 |
| `organelles/lightsout_*.ckpt` | 64 | 4 | 3 | 128 | 256 |
| `organelles/sudoku_*.ckpt` | 64 | 4 | 3 | 128 | 256 |

---

## Training Your Own

Delete the `.ckpt` file and the demo trains from scratch:

```bash
rm shakespeare.ckpt
./shakespeare_demo    # trains fresh
```

---

## Checkpoint Format

```
[4 bytes]   magic number (0x4D475054 — "MGPT")
[4 bytes]   training step count
[N bytes]   model weights (row-major, float32)
[N bytes]   Adam m1 state (same layout)
[N bytes]   Adam m2 state (same layout)
```

File size ≈ `3 × num_params × sizeof(float)`.

---

## License

All pretrained models are derived from MIT-licensed code and public domain / original training data. Released under the **MIT License**. See [DATA_LICENSE.md](../DATA_LICENSE.md) for provenance.
