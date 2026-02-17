# MicroGPT-C

A **zero-dependency, pure C99** GPT implementation — a complete language model you can understand, modify, and embed anywhere.

> **No Python. No PyTorch. No GPU. Just C.**

---

## What Is This?

MicroGPT-C is a **serious, production-quality implementation** of a GPT (Generative Pre-trained Transformer) in plain C99. It faithfully implements the same architecture as [Karpathy's `microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — attention, backpropagation, Adam optimiser — compiled to native code with optional SIMD vectorisation.

It is **not a toy**. While the model architecture is intentionally minimal (suitable for learning and experimentation), the C implementation is robust:

- **Full training pipeline** — forward pass, backward pass, Adam optimiser with cosine LR + warmup
- **39 unit tests** covering every public API function
- **15 performance benchmarks** with measured throughput
- **Two tokenisation strategies** — character-level and word-level (with O(1) hash lookup)
- **INT8 quantisation** support for memory-constrained devices
- **SIMD auto-vectorisation** enabled by default
- **Cross-platform multi-threaded training** (auto-detected threads; pthread on Linux/macOS, Win32 on Windows)

| Use Case | Why MicroGPT-C |
|----------|----------------|
| **Learning** | Read the entire GPT algorithm in ~1,500 lines of commented C |
| **Embedded / edge** | Entire model fits in **< 50 KB** RAM; runs on MCUs without an OS |
| **Research** | Auditable baseline for quantisation, custom layers, or optimiser experiments |
| **Integration** | Drop two files (`microgpt.h` + `microgpt.c`) into any C/C++ project |

---

## Quick Start

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release

# Character-level name generation
./names_demo

# Character-level Shakespeare generation (multi-threaded)
./shakespeare_demo

# Run unit tests (39 tests)
./test_microgpt

# Run benchmarks (15 benchmarks)
./bench_microgpt
```

### Example Output — Name Generation

```
docs:32033 vocab:27 embd:16 heads:4
params:4192
step    1/1000 loss 3.4468
step  500/1000 loss 2.1334
step 1000/1000 loss 2.4463

Train: 0.12s
--- names ---
   1: sarey          6: kenian         11: gamaren
   2: canal          7: zaynna         12: aleliy
   3: aninan         8: sanari         13: rarian
   4: aniel          9: ararinn        14: jaylan
   5: kanana        10: amaria         15: mariah
```

### Example Output — Shakespeare Generation

Trained with `N_EMBD=128 N_LAYER=4 N_HEAD=8` (840K params, ~37 min on 12 threads):

```
--- generated Shakespeare (character-level) ---

[sample — seed: 'O']
O    That may one is the stand's hanget stronger,

[sample — seed: 'W']
Whe win O the Luce of my find lifest a stance,

[sample — seed: 'M']
M    Reposs you find in have to see a man; as all eyes him.

[sample — seed: 'H']
Hand her calas to the satime of she for hear.
```

---

## How It Works

A decoder-only Transformer following the GPT-2 design:

```
Input → Token Embedding + Position Embedding
      → RMSNorm
      → [Self-Attention (multi-head, causal) → Residual] × N_LAYER
      → [RMSNorm → MLP (fc1 → ReLU → fc2, 4× width) → Residual] × N_LAYER
      → Linear (lm_head) → Softmax → next-token probabilities
```

**Training** uses cross-entropy loss with the Adam optimiser and cosine learning-rate schedule (linear warmup → cosine decay). The entire forward + backward pass is implemented manually — no autograd, no computational graph.

All architecture parameters are compile-time configurable:

| Parameter | Default | Override | Effect |
|-----------|---------|----------|--------|
| `N_EMBD` | 32 | `-DN_EMBD=64` | Embedding dimension |
| `N_HEAD` | 4 | `-DN_HEAD=8` | Attention heads |
| `N_LAYER` | 2 | `-DN_LAYER=4` | Transformer blocks |
| `BLOCK_SIZE` | 32 | `-DBLOCK_SIZE=64` | Maximum sequence length |
| `MLP_DIM` | `N_EMBD × 4` | `-DMLP_DIM=256` | MLP hidden dimension |
| `WARMUP_STEPS` | `NUM_STEPS / 10` | `-DWARMUP_STEPS=500` | LR warmup duration |

---

## Using as a Library

### Integration

Add two files to your project — no build system changes needed beyond compiling one extra `.c`:

```c
#include "microgpt.h"
```

### Character-Level Pipeline

Best for short text: names, codes, identifiers.

```c
Docs docs;
load_docs("names.txt", &docs);       // Load line-separated training data

Vocab vocab;
build_vocab(&docs, &vocab);           // Build character vocabulary (auto-sized)

size_t ids[256];
size_t n = tokenize("alice", 5, &vocab, ids, 256);

Model *model = model_create(vocab.vocab_size);
// ... train with forward_backward_one + adam_step ...
// ... generate with forward_inference + sample_token ...
model_free(model);
```

### Word-Level Pipeline

Best for prose, dialogue, poetry. Uses O(1) hash-based `word_to_id` lookup.

```c
size_t len;
char *text = load_file("shakespeare.txt", &len);

WordVocab wv;
build_word_vocab(text, len, 10000, &wv);  // Keep top 10,000 words

size_t ids[8192];
size_t n = tokenize_words(text, len, &wv, ids, 8192);

Model *model = model_create(wv.vocab_size);
// ... train and generate ...
free_word_vocab(&wv);
model_free(model);
```

### Character-Level vs Word-Level — Which to Use?

Both tokenisation strategies are available, but they suit different model scales:

| Factor | Character-level | Word-level |
|--------|----------------|-----------|
| **Vocab size** | ~50–100 tokens | ~5,000–10,000+ tokens |
| **`<unk>` tokens** | **Zero** — every byte is in vocab | Common for rare words |
| **lm_head cost** | ~100 × N_EMBD ≈ tiny | ~10,000 × N_EMBD ≈ dominates model |
| **Training signal** | Every character seen thousands of times | Rare words get few examples |
| **Best for** | Small models (\< 1M params) | Large models (millions of params) |

**Why character-level wins at this scale:** Shakespeare has ~20,000 unique words. Even keeping the top 8,000, the output layer (`lm_head`) alone would consume more parameters than the rest of the model combined. The model can't learn meaningful distinctions between thousands of words, so output floods with `<unk>`. With character-level (~84 tokens), the entire vocabulary fits comfortably and the model masters every symbol.

**Rule of thumb:** Use character-level unless your model has enough capacity (N_EMBD ≥ 256, N_LAYER ≥ 4) to handle a large word vocabulary.

### Training Checkpoints

Save and resume training without losing optimizer momentum:

```c
// Save: model weights + Adam m/v state + step counter
checkpoint_save(model, m_adam, v_adam, step, "checkpoint.bin");

// Resume: restores everything needed to continue training
Model *model = checkpoint_load("checkpoint.bin", vocab_size,
                               m_adam, v_adam, &step);
// Continue training from 'step' onwards — momentum and LR schedule are preserved
```

### Complete Examples

See [`examples/names/main.c`](examples/names/main.c) (character-level) and [`examples/shakespeare/main.c`](examples/shakespeare/main.c) (character-level, multi-threaded) for full working programs.

Detailed guides:
- [Character-level tokenisation](docs/character-level.md)
- [Word-level tokenisation](docs/word-level.md)

---

## Performance

### Character-Level vs Karpathy's microgpt.py

Measured on the **character-level name generation** workload (1,000 training steps, 20 inference samples) — MicroGPT-C vs [Karpathy's `microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95):

| Metric | Python (microgpt.py) | C (fp64) | Speedup |
|--------|--------|----------|---------| 
| **Training time** | ~93 s | **0.09 s** | **~1,000×** |
| **Training throughput** | ~0.1 k tok/s | **~616 k tok/s** | **~6,000×** |
| **Steps/sec** | ~11 | **~10,800** | **~1,000×** |
| **Inference time** | ~0.74 s | **< 1 ms** | **~700×+** |

### Shakespeare Character-Level (N_EMBD=128, N_LAYER=4, vocab≈84)

Multi-threaded training with auto-detected workers (uses all available CPU cores):

| Metric | Value |
|--------|-------|
| **Model parameters** | ~1M |
| **Vocab** | ~84 characters (zero `<unk>`) |
| **Throughput** | ~85 steps/s, ~64k tok/s (12 threads) |

### Benchmarks (N_EMBD=32, N_LAYER=2)

Run `./bench_microgpt` to reproduce on your machine:

| Operation | Latency | Throughput |
|-----------|---------|------------|
| `forward_inference` (1 token) | 0.01 ms | **125k infer/s** |
| `forward_backward_one` (1 pos) | 0.01 ms | **96k fwd+bwd/s** |
| `adam_step` | 0.03 ms | **39k steps/s** |
| `sample_token` (vocab=50) | < 0.01 ms | **4.3M samples/s** |
| `tokenize` (char, 12 chars) | < 0.01 ms | **34.5M tok/s** |
| `tokenize_words` (1KB text) | 0.41 ms | **486k tok/s** |
| `checkpoint_save` + `load` | 0.70 ms | **1,437 roundtrips/s** |
| Full training step (seq=8) | 0.08 ms | **82.6k tok/s** |

> **INT8 quantised build:** ~25% slower training than fp64 on this tiny model, but **~8× smaller** weight storage — ideal for constrained devices.

---

## Performance Optimisations

The engine includes several optimisations for training throughput:

- **Cache-friendly `lin_bwd`** — backward gradient accumulation uses row-major weight traversal, eliminating L1 cache thrashing for large output layers (e.g. lm_head with vocab=10003)
- **Hash-based `word_to_id`** — O(1) DJB2 hash lookup instead of O(n) linear scan across the vocabulary
- **Cosine LR with warmup** — linear warmup for `WARMUP_STEPS` followed by cosine annealing, avoiding premature LR decay
- **`restrict` + vectorisation hints** — C99 `restrict` qualifiers and Clang loop pragmas on all hot-path functions (`lin_fwd`, `lin_bwd`, `rmsnorm_fwd/bwd`) to enable full auto-vectorisation
- **Compiler flags** — `-O3 -march=native -ffast-math -funroll-loops` for Release builds
- **Cross-platform multi-threaded batches** — Shakespeare demo auto-detects CPU count and parallelises batch processing via portable `microgpt_thread.h` (pthread on Linux/macOS, Win32 threads on Windows)

---

## Build Options

### SIMD auto-vectorisation (ON by default)

The compiler targets the best available instruction set (`-march=native` on GCC/Clang, `/arch:AVX2` on MSVC). To disable:

```bash
cmake -DMICROGPT_SIMD=OFF ..
```

### INT8 quantised build

Weights stored as 8-bit integers with per-matrix scales:

```bash
cmake -DQUANTIZATION_INT8=ON ..
```

### Custom architecture (Shakespeare example)

Override compile-time parameters for larger models:

```bash
cmake -DN_EMBD=64 -DN_HEAD=4 -DN_LAYER=2 -DBLOCK_SIZE=32 ..
```

---

## Amalgamated Demo

[`microgpt_amalgamated.c`](microgpt_amalgamated.c) is the **entire GPT** — training, inference, multi-head attention, Adam optimiser — in a single ~50-line C99 file. Just a fun exercise to see how small a working Transformer can be:

```bash
cc -O2 -o microgpt_amalgamated microgpt_amalgamated.c -lm
./microgpt_amalgamated    # expects names.txt in current directory
```

> **Note:** This is a compressed demo for curiosity and portability. For real work — configurable architecture, multi-layer support, word-level tokenisation, INT8 quantisation, threading, checkpoints, tests, and benchmarks — use the full library in `src/`.

---

## Project Layout

```
src/
  microgpt.h              Public API — all functions documented
  microgpt.c              Core engine (~1,900 lines)
  microgpt_thread.h       Portable threading (pthread / Win32)
examples/
  names/main.c            Character-level name generation demo
  shakespeare/main.c      Character-level Shakespeare generation (multi-threaded)
tests/
  test_microgpt.c         Unit tests (39 tests, zero dependencies)
  bench_microgpt.c        Performance benchmarks (15 benchmarks)
tools/
  vocab_analysis.c        Vocabulary coverage analysis utility
docs/
  character-level.md      Character-level tokenisation guide
  word-level.md           Word-level tokenisation guide
microgpt_amalgamated.c    Single-file demo (see above)
CMakeLists.txt            Build system (C99, SIMD default ON)
```

---

## Requirements

- **C99 compiler** (GCC, Clang, MSVC)
- **CMake 3.10+**
- **pthreads** (Linux/macOS) or **Win32 threads** (Windows) — for Shakespeare demo
- No other dependencies

---

## License

MIT — see [LICENSE](LICENSE) and source file headers.

**Author:** Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
