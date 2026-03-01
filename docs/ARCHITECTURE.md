# MicroGPT-C Architecture

> System-level overview of the MicroGPT-C codebase — components, data flow, and build structure.

---

## System Overview

MicroGPT-C is a pure-C, dependency-free transformer inference and training engine designed for composable intelligence at the edge. The system consists of three layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Applications                         │
│   demos/  — Shakespeare, games, VM codegen, lottery     │
├─────────────────────────────────────────────────────────┤
│                  Organelle Layer                        │
│   microgpt_organelle.c/h  — OPA pipeline, Kanban,      │
│   cycle detection, multi-organelle coordination         │
├─────────────────────────────────────────────────────────┤
│                  Foundation Layer                        │
│   microgpt.c/h  — Transformer, tokeniser, training,    │
│   inference, checkpoint I/O, KV-cache                   │
└─────────────────────────────────────────────────────────┘
```

## Component Map

| Component | Files | Purpose |
|-----------|-------|---------|
| **Core Transformer** | `src/microgpt.c`, `src/microgpt.h` | Forward/backward pass, attention, Adam optimiser, checkpoint save/load |
| **Organelle Runtime** | `src/microgpt_organelle.c`, `src/microgpt_organelle.h` | Multi-organelle pipeline: OPA Kanban, planner/player pattern, cycle detection |
| **VM Engine** | `src/microgpt_vm.*` | Bytecode compiler + runtime for learned code generation |
| **Foundation Demos** | `demos/character-level/shakespeare/`, `demos/word-level/` | Text generation (char + word level) |
| **Game Demos** | `demos/character-level/{connect4,puzzle8,...}/` | Organelle-based game playing across 11 games |
| **Pretrained Models** | `models/foundation/`, `models/organelles/` | Ready-to-use checkpoints |
| **Book** | `book/` | 16-chapter technical guide + PDF build |

## Data Flow

```
Training data (.txt)
    │
    ▼
tokenize() ──→ token IDs
    │
    ▼
forward_backward_one() ──→ gradients
    │
    ▼
adam_step() ──→ updated weights
    │
    ▼
checkpoint_save() ──→ .ckpt file
```

```
Inference:
checkpoint_load() ──→ Model
    │
    ▼
forward_inference() + KV-cache ──→ logits
    │
    ▼
sample_token() ──→ generated output
```

## Build Structure

```
CMakeLists.txt              Public build (all public demos + tests)
_private/CMakeLists.txt     Full build (public + OpaBoard, tools, ML, market demos)
bootstrap.sh / .bat         Quick public build
_private/bootstrap.sh       Quick full build
```

## Directory Layout

```
microgpt-c/
  src/                      Core library sources
  demos/                    Demo applications
    character-level/        Char-tokenised demos (games, Shakespeare, etc.)
    word-level/             Word-tokenised demos
  models/                   Pretrained checkpoints
    foundation/             Shakespeare models
    organelles/             Game + experiment checkpoints
  book/                     Technical book (chapters, PDF, build script)
  docs/                     Engineering documentation
    ARCHITECTURE.md         This file
    DESIGN.md               Design decisions and patterns
    FUNCTIONAL_SPEC.md      API specification and usage guide
    BUILD_OPTIONS.md        CMake flags and compiler options
    testing/                Test design and validation
    research/               Research notes and experiment analysis
  tests/                    Unit tests, benchmarks, test resources
  tools/                    Standalone analysis utilities
  _private/                 Private extensions (OpaBoard, market demos, research)
```

---

*See also: [DESIGN.md](DESIGN.md) for design patterns, [FUNCTIONAL_SPEC.md](FUNCTIONAL_SPEC.md) for API reference, [BUILD_OPTIONS.md](BUILD_OPTIONS.md) for build configuration.*
