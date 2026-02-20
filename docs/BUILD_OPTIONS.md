# Build Options

## SIMD Auto-Vectorisation (ON by default)

The compiler targets the best available instruction set (`-march=native` on GCC/Clang, `/arch:AVX2` on MSVC). To disable:

```bash
cmake -DMICROGPT_SIMD=OFF ..
```

---

## INT8 Quantised Build

Weights stored as 8-bit integers with per-matrix scales:

```bash
cmake -DQUANTIZATION_INT8=ON ..
```

---

## Apple Metal GPU Acceleration (macOS only)

Offloads `lin_fwd`/`lin_bwd` matmuls to Metal compute shaders. Best for larger models (N_EMBD ≥ 512):

```bash
cmake -DMICROGPT_METAL=ON ..
```

> **Note:** For small models (N_EMBD=128), GPU dispatch overhead exceeds compute time. Multi-threaded CPU is faster.

---

## Apple Accelerate / BLAS

Uses `CBLAS_GEMV`/`CBLAS_GER` macros (auto-dispatched to `cblas_dgemv`/`cblas_sgemv` based on `scalar_t`):

```bash
cmake -DMICROGPT_BLAS=ON ..
```

> **Note:** Accelerate's internal threading conflicts with multi-threaded training. Best for single-threaded inference.

---

## Float Precision

Switch all weights, activations, and gradients from `double` to `float` (32-bit). Useful for ARM NEON throughput (4-wide vs 2-wide SIMD) and memory-constrained devices:

```bash
cmake -DMICROGPT_USE_FLOAT=ON ..
```

> **Note:** Optimizer hyperparameters (learning rate, Adam β1/β2/ε) remain `double` for numerical stability. Test tolerances auto-adjust via `SCALAR_TOL`.

---

## Custom Architecture

Each demo uses the `DEFINES` parameter in CMakeLists.txt to set its architecture. For ad-hoc overrides:

```bash
cmake -DN_EMBD=128 -DN_HEAD=8 -DN_LAYER=4 -DBLOCK_SIZE=256 ..
```

> **Note:** All demos already have their optimal architecture sizes baked into `CMakeLists.txt` via `add_demo(... DEFINES ...)`. Manual overrides affect only the default library target.

---

## Architecture Parameters

All architecture parameters are compile-time constants (`#define` macros in `microgpt.h`):

| Parameter | Default | Override | Effect |
|-----------|---------|----------|--------|
| `N_EMBD` | 16 | `-DN_EMBD=128` | Embedding dimension |
| `N_HEAD` | 4 | `-DN_HEAD=8` | Attention heads |
| `N_LAYER` | 1 | `-DN_LAYER=4` | Transformer blocks |
| `BLOCK_SIZE` | 16 | `-DBLOCK_SIZE=256` | Maximum sequence length |
| `MLP_DIM` | 64 | `-DMLP_DIM=512` | MLP hidden dimension |
| `WARMUP_STEPS` | 100 | `-DWARMUP_STEPS=500` | LR warmup duration |
| `scalar_t` | `float` | `-DMICROGPT_USE_FLOAT=OFF` | Switch all weights/activations to `double` |

---

## Platform Accelerators

> **Zero dependencies means zero dependencies.** The core engine (`microgpt.h` + `microgpt.c`) requires nothing beyond a C99 compiler, `libc`, and `libm`. It compiles and runs on any platform — from a Raspberry Pi to a mainframe.
>
> The following are **opt-in platform accelerators** that you enable explicitly via CMake flags. They are never required:
>
> | Accelerator | Flag | Requires | When To Use |
> |------------|------|----------|-------------|
> | Metal GPU | `-DMICROGPT_METAL=ON` | macOS + Apple Silicon | Models with N_EMBD ≥ 512 |
> | BLAS | `-DMICROGPT_BLAS=ON` | Accelerate / OpenBLAS / MKL | Single-threaded inference |
>
> If you don't set these flags, you get pure C99 with compiler auto-vectorisation — which, at current model sizes, is actually the fastest option (see [Optimisation Strategies](foundation/OPTIMISATION_STRATEGIES.md)).
