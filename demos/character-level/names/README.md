# Name Generation Demo

The "Hello World" of MicroGPT-C — a tiny Transformer learns to invent plausible human names from a list of real ones.

---

## Spear Summary

**Point:** This is the simplest possible proof that MicroGPT-C works — train on names and it generates new ones that sound real.

**Picture:** It's like teaching a child to name imaginary friends by reading them the phone book. After enough pages they start inventing names that sound right even though nobody actually has them.

**Proof:** The entire pipeline — load data, build vocab, train, generate — runs in a single 330-line C file with zero external dependencies. That's the baseline everything else in the project is built from.

**Push:** Run this demo first to verify your build is working before trying Shakespeare or the organelle experiments.

---

## How It Works

1. Load `names.txt` — one name per line
2. Build a character-level vocabulary (a–z lowercase + BOS)
3. Train with mini-batch gradient accumulation + Adam
4. Generate new names autoregressively until BOS (used as end-of-sequence)

## Architecture

Uses MicroGPT-C default configuration:

| Parameter | Value |
|-----------|-------|
| N_EMBD | 48 |
| N_HEAD | 4 |
| N_LAYER | 2 |
| BLOCK_SIZE | 64 |
| BATCH_SIZE | 16 |
| Training steps | 10,000 |
| Inference temp | 0.8 |

## Features

- **Single-threaded** — the original, simplest training loop (no threading layer)
- **Round-robin document selection** over shuffled dataset
- **Training log** — appends to `names_demo.training.log`
- **Three-phase structure** — DATA → TRAIN → INFER cleanly separated

## Build & Run

```bash
mkdir build && cd build
cmake .. && cmake --build . --target names_demo
./names_demo    # trains ~10K steps, generates sample names
```

## Corpus

`names.txt` — list of human names, one per line.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
