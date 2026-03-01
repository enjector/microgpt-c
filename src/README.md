# Source Code

The MicroGPT-C engine is implemented in pure C99 with zero external dependencies. All source files are in this directory — there are no subdirectories to navigate.

## Core Engine

| File | Lines | Purpose |
|------|------:|---------|
| **microgpt.h** | 1,070 | Single-header API — Transformer architecture, training, inference, tokenisation, checkpointing. Include this one file to use the engine. |
| **microgpt.c** | 2,950 | Implementation — forward/backward pass, attention, Adam optimiser, KV-cache, checkpoint I/O, vocabulary builder, document loader. |

## Organelle Pipeline

| File | Lines | Purpose |
|------|------:|---------|
| **microgpt_organelle.h** | 300 | Organelle pipeline API — Planner→Judge coordination, flat-string protocols, multi-organelle wiring. |
| **microgpt_organelle.c** | 1,400 | Implementation — pipeline execution, rejection sampling, protocol parsing, organelle checkpoint management. |

## Virtual Machine Engine

| File | Lines | Purpose |
|------|------:|---------|
| **microgpt_vm.h** | 1,470 | VM API and data structures — `vm_engine`, `vm_module`, `vm_function`, `vm_variable`, type system, support layer (`sequence`, `cmap`, `queue`, `verb_context`). |
| **microgpt_vm.c** | 3,675 | VM implementation — code generator, 6-pass verifier, stack-based runtime, native function callbacks, IL dump. |
| **microgpt_vm.l** | 130 | Flex lexer definition — tokenises TypeScript-like syntax and LaTeX math into the parser. |
| **microgpt_vm.y** | 680 | Bison grammar definition — parses tokens into AST nodes that drive bytecode generation. |
| **microgpt_vm_parser.l.c** | — | Pre-generated Flex output (committed so builds work without Flex installed). |
| **microgpt_vm_parser.tab.c** | — | Pre-generated Bison output (committed so builds work without Bison ≥ 3.0). |

## Metal Acceleration (Experimental)

| File | Lines | Purpose |
|------|------:|---------|
| **microgpt_metal.h** | 60 | Metal compute API — GPU-accelerated matrix operations for Apple Silicon. |
| **microgpt_metal.m** | 340 | Objective-C bridge — Metal device setup, command buffer management, kernel dispatch. |
| **microgpt_metal.metal** | 80 | GPU shader kernels — matrix multiply and element-wise operations in Metal Shading Language. |

## Architecture

```
microgpt.h / .c                    ← Core Transformer engine
    │
    ├── microgpt_organelle.h / .c  ← Pipeline coordination layer
    │
    ├── microgpt_vm.h / .c         ← VM compiler + runtime
    │     ├── microgpt_vm.l        ← Lexer (Flex)
    │     └── microgpt_vm.y        ← Grammar (Bison)
    │
    └── microgpt_metal.h / .m      ← GPU acceleration (Apple Silicon)
          └── microgpt_metal.metal ← Compute shaders
```

## How to Use

The engine is designed as a single-header library. For most use cases:

```c
#include "microgpt.h"  // Core engine — that's it
```

For organelle pipelines, also include:
```c
#include "microgpt_organelle.h"
```

For VM scripting, also include:
```c
#include "microgpt_vm.h"
```

See `demos/` for working examples and `demos/character-level/` for the full experiment suite.
