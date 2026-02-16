# MicroGPT-C

A **zero-dependency, pure C99** implementation of a GPT-style character-level language model.

The algorithm faithfully matches [Andrej Karpathy's `microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — same architecture, same training loop, same sampling — but compiles to native code with optional compiler-driven SIMD auto-vectorisation for dramatically faster training and inference.

> **Train a GPT in 20 ms. Generate names in microseconds. No Python. No PyTorch. No GPU.**

---

## What Is This?

MicroGPT-C is a minimal, readable implementation of a GPT (Generative Pre-trained Transformer) — the same family of models behind ChatGPT, but stripped down to its essential algorithm. It trains a tiny character-level language model that learns to generate realistic human names from scratch.

The goal is **education and experimentation**: understand how attention, backpropagation, and the Adam optimiser actually work at the lowest level, without any framework abstractions.

| Audience | Value |
|----------|-------|
| **Students & educators** | Study attention, softmax, Adam, and backprop in readable C — no framework magic |
| **Embedded / edge engineers** | Entire model fits in **< 50 KB** RAM; runs on MCUs with no runtime dependencies |
| **Researchers** | Auditable baseline for quantisation, custom layers, or optimiser experiments |
| **Rapid prototypers** | Train → iterate in milliseconds; test tokenisers, vocabularies, data formats |

---

## Quick Start

```bash
# Linux / macOS
chmod +x build.sh
./build.sh
./build/microgpt
```

```batch
:: Windows
build.bat
build\Release\microgpt.exe
```

The build automatically copies `data/names.txt` next to the executable.

---

## Performance

Measured on the same workload (1,000 training steps, 20 inference samples) — C vs the reference Python:

| Metric | Python | C (fp64) | Speedup |
|--------|--------|----------|---------|
| **Training time** | ~93 s | **0.02 s** | **~4,600×** |
| **Training throughput** | ~0.1 k tok/s | **~289 k tok/s** | **~2,800×** |
| **Steps/sec** | ~11 | **~40,000** | **~3,600×** |
| **Inference time** | ~0.74 s | **< 1 ms** | **~700×+** |
| **Inference rate** | ~27 samples/s | **20,000 samples/s** | **~740×** |
| **Token throughput** | — | **109,000 tok/s** | — |

> **INT8 quantised build:** ~25% slower training than fp64 on this tiny model, but **~8× smaller** weight storage — ideal for constrained devices.

---

## Architecture

A single-layer, decoder-only Transformer following the GPT-2 design:

```
Input → Token Embed + Pos Embed → RMSNorm
  → Self-Attention (4 heads, causal) → Residual
  → RMSNorm → MLP (fc1 → ReLU → fc2, 4× width) → Residual
  → Linear (lm_head) → Softmax → next-token probabilities
```

| Parameter | Value |
|-----------|-------|
| Embedding dim | 16 |
| Attention heads | 4 |
| Layers | 1 |
| Context length | 16 |
| Total parameters | ~4,600 |
| Weight memory (fp64) | ~37 KB |
| Weight memory (INT8) | ~4.6 KB |
| Training memory | ~144 KB |
| Inference memory | < 50 KB |

Training uses the **Adam** optimiser with linear learning-rate decay (configurable in `microgpt.h`).

---

## Build Options

### Build scripts (recommended)

| Platform | Standard | SIMD (faster) |
|----------|----------|---------------|
| Linux/macOS | `./build.sh` | `./build.sh --simd` |
| Windows | `build.bat` | `build.bat simd` |

### SIMD auto-vectorisation

The `--simd` flag enables compiler-driven **auto-vectorisation** of the core dot products, matrix multiplications, and normalisations. On x86-64 the compiler targets the best available instruction set (SSE4, AVX2, etc.) via `-march=native`; on MSVC it enables `/arch:AVX2`. This gives a measurable speed-up on larger models without any hand-written intrinsics — the compiler re-writes the scalar loops into SIMD instructions automatically.

```bash
# Linux / macOS — auto-detect best ISA
./build.sh --simd

# CMake directly
cmake -DMICROGPT_SIMD=ON ..
cmake --build . --config Release
```

### INT8 quantised build

Weights are stored as 8-bit integers with per-matrix scales — the forward pass dequantises on the fly; Adam updates an fp64 master copy and requantises each step. This reduces weight storage by **~8×** (37 KB → 4.6 KB) at a small accuracy/speed trade-off.

| Platform | Standard | SIMD |
|----------|----------|------|
| Linux/macOS | `./build_quantised.sh` | `./build_quantised.sh --simd` |
| Windows | `build_quantised.bat` | `build_quantised.bat simd` |

### CMake directly

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release

# With INT8 quantisation
cmake -DQUANTIZATION_INT8=ON ..

# With SIMD auto-vectorisation
cmake -DMICROGPT_SIMD=ON ..

# Both
cmake -DQUANTIZATION_INT8=ON -DMICROGPT_SIMD=ON ..
```

---

## Project Layout

| Path | Description |
|------|-------------|
| `microgpt.h` | Model config, public API declarations |
| `microgpt.c` | Core engine: model, forward/backward, Adam, data loading |
| `main.c` | Entry point: load data → train → generate samples |
| `data/names.txt` | Training data (one name per line, ~32k names) |
| `CMakeLists.txt` | CMake build (C99, Release, optional SIMD / INT8) |

---

## Requirements

- **C99 compiler** (GCC, Clang, MSVC)
- **CMake 3.10+**
- No other dependencies

---

## License

MIT — see [LICENSE](LICENSE) and source file headers.

**Author:** Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
