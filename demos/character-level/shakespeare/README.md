# Shakespeare Character-Level Generation

A ~900K-parameter Transformer learns to write Shakespeare character by character — no tokeniser, no `<unk>` tokens, just raw bytes in and bytes out.

---

## Spear Summary

**Point:** Character-level tokenisation eliminates the `<unk>` problem entirely while teaching the model spelling and punctuation for free.

**Picture:** It's like teaching someone to write by showing them every letter in every word rather than handing them a dictionary with missing pages. Nothing is ever "unknown" — the model sees every character that exists in the text.

**Proof:** The entire vocabulary is just ~80 unique characters. No word is ever out-of-vocabulary because words aren't the unit — individual letters are. The earlier word-level approach flooded output with `<unk>` tokens; this approach produces zero.

**Push:** Use this demo as the starting point for any new text generation experiment. Character-level is the right default for models at this scale.

---

## How It Works

1. Load Shakespeare's complete works as one-line-per-document
2. Build a character vocabulary (~80 unique bytes including letters, punctuation, whitespace)
3. Train a decoder-only Transformer using AdamW optimisation
4. Generate new text from single-character seeds (`T`, `O`, `W`, `M`, `H`)

Each generated character is sampled autoregressively at temperature 0.7 — warm enough for creativity, cool enough for coherence.

## Architecture

| Parameter | Value |
|-----------|-------|
| N_EMBD | 128 |
| N_HEAD | 8 |
| N_LAYER | 4 |
| BLOCK_SIZE | 256 |
| BATCH_SIZE | 16 |
| Training steps | 30,000 |
| Learning rate | 0.001 |
| Generation temp | 0.7 |
| Gen length | 300 chars/sample |
| Samples | 5 |

## Features

- **Multi-threaded training** via portable `microgpt_thread.h` layer
- **Checkpoint save/load** — resumes from `shakespeare.ckpt` on restart
- **Training log** — appends to `shakespeare_demo.training.log`
- **Per-sample inference timing** — reports tokens/second

## Build & Run

```bash
mkdir build && cd build
cmake .. && cmake --build . --target shakespeare_demo
./shakespeare_demo    # trains ~30K steps, then generates 5 samples
```

Auto-resumes from checkpoint if `shakespeare.ckpt` exists.

## Corpus

`shakespeare.txt` — Shakespeare's complete works (~5.4 MB). Each line is one training document.

---

*MicroGPT-C — Enjector Software Ltd. MIT License.*
