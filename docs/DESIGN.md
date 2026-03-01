# MicroGPT-C Design

> Key design decisions, patterns, and trade-offs in the MicroGPT-C codebase.

---

## Core Design Principles

1. **Zero dependencies** — No external libraries. Everything compiles with a C99 compiler and libc.
2. **Compile-time architecture** — Model dimensions (`N_EMBD`, `N_HEAD`, `N_LAYER`, `BLOCK_SIZE`, `MLP_DIM`) are compile-time constants via `-D` flags, enabling aggressive compiler optimisation.
3. **Single-file embedding** — The core library is two files: `microgpt.c` + `microgpt.h`. Add to any project with no build system changes.
4. **Checkpoint portability** — Binary checkpoint format (magic + step + weights + Adam state) works across platforms.

## Organelle Architecture Pattern

The **Adaptive Organelle Planner (OPA)** coordinates multiple specialist models:

- **Planner** — Analyses the current state, proposes an action
- **Player** — Executes the planned action
- **OPA Kanban** — Working memory: tracks recent actions, blocked moves, and state history
- **Cycle Detector** — Detects A↔B oscillation and forces alternative moves

Each organelle is a separate, independently trained transformer with its own checkpoint. At runtime, they communicate through text: the planner's output becomes the player's input prompt.

## Tokenisation Strategy

Two tokenisation approaches, chosen per demo at compile time:

| Approach | When to use | Implementation |
|----------|-------------|----------------|
| **Character-level** | Short sequences, small vocab, games | `build_vocab()` — auto-discovers unique chars |
| **Word-level** | Longer text, code generation | `build_word_vocab()` — frequency-based BPE-like |

## Memory Management

- **Stack-allocated scratch buffers** where possible (no malloc in hot paths)
- **KV-cache** for autoregressive inference — avoids recomputing all previous positions
- **Arena pattern** in VM engine — bulk allocate, bulk free

## Threading Model

- **Training** uses `TrainWorker` with pthreads for batch parallelism
- **Inference** is single-threaded (fast enough at small model sizes)
- Each demo spawns workers via `train_worker_create()` / `train_worker_run()`

---

*See also: [ARCHITECTURE.md](ARCHITECTURE.md) for system overview, [FUNCTIONAL_SPEC.md](FUNCTIONAL_SPEC.md) for API reference.*
