# Pretrained Models

MicroGPT-C ships with pretrained checkpoints ready for **instant inference** — no training required.

---

## Available Models

| Model | Size | Params | Trained On | What It Does |
|-------|------|--------|-----------|-------------|
| `shakespeare.ckpt` | 9.6 MB | 840K | Complete works of Shakespeare | Character-level text generation — learns verse, spelling, punctuation |
| `c_codegen.ckpt` | 20 MB | 875K | 2,081 C functions (math, physics, DSP) | Code retrieval — byte-perfect recall from comment prompts |
| `c_wiringgen.ckpt` | 19.9 MB | 875K | C function compositions (wiring patterns) | Multi-function pipeline generation |

Each `.ckpt` file contains:
- Model weights (float32)
- Adam optimizer state (first & second moments)
- Training step counter

The accompanying `.ckpt.log` files contain the full training history (loss per step, timings, and generated samples).

---

## Using a Pretrained Model

### Quickest path: just run the demo

The demos **automatically detect** checkpoint files and skip training:

```bash
cd build

# Copy pretrained checkpoint from models/ to build/
cp ../models/shakespeare.ckpt .

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

Use `model_load()` to load a checkpoint and `forward_inference()` to generate tokens:

```c
#include "microgpt.h"

// 1. Set up config matching the checkpoint's architecture
MicrogptConfig cfg = microgpt_default_config();
cfg.n_embd = 128;
cfg.n_head = 8;
cfg.n_layer = 4;
cfg.block_size = 256;

// 2. Load the pretrained checkpoint
Model *model = model_load("shakespeare.ckpt", vocab_size, &cfg);
if (!model) { /* handle error */ }

// 3. Allocate KV cache and run inference
scalar_t *keys, *values, *logits;
size_t cache_len = 0;
kv_cache_alloc(&cfg, &keys, &values);
logits = malloc(vocab_size * sizeof(scalar_t));

// 4. Generate token by token
size_t token = seed_token;
for (int i = 0; i < max_tokens; i++) {
    forward_inference(model, token, i, &keys, &values,
                      &cache_len, logits);
    token = sample_token(logits, vocab_size, temperature);
    printf("%c", vocab_decode(token));
}
```

---

## Architecture Requirements

Each checkpoint is tied to the architecture it was trained with. You must compile with matching dimensions:

| Checkpoint | N_EMBD | N_HEAD | N_LAYER | BLOCK_SIZE | MLP_DIM |
|-----------|--------|--------|---------|------------|---------|
| `shakespeare.ckpt` | 128 | 8 | 4 | 256 | 512 |
| `c_codegen.ckpt` | 128 | 4 | 4 | 512 | 512 |
| `c_wiringgen.ckpt` | 128 | 4 | 4 | 512 | 512 |

> **Note:** The demos in `CMakeLists.txt` already have these dimensions baked in via `add_demo(... DEFINES ...)`. You only need to worry about this if integrating `model_load()` into your own code.

---

## Training Your Own

Don't want to use the pretrained weights? Simply delete (or don't copy) the `.ckpt` file — the demo will train from scratch:

```bash
# Train fresh — no checkpoint present, so training begins
./shakespeare_demo

# Next run — checkpoint found, training skipped
./shakespeare_demo
```

To retrain from scratch even when a checkpoint exists, delete it first:

```bash
rm shakespeare.ckpt
./shakespeare_demo    # trains again
```

---

## Checkpoint Format

Checkpoints are raw binary files written by `model_save()`:

```
[4 bytes]   magic number (0x4D475054 — "MGPT")
[4 bytes]   training step count
[N bytes]   model weights (row-major, float32)
[N bytes]   Adam m1 state (same layout)
[N bytes]   Adam m2 state (same layout)
```

The file size is approximately `3 × num_params × sizeof(float)` — the factor of 3 accounts for weights + two optimizer moment vectors.

---

## License

All pretrained models are derived from MIT-licensed code and public domain / original training data. They are released under the **MIT License**. See [DATA_LICENSE.md](../DATA_LICENSE.md) for full provenance.
