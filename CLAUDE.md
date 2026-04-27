# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Pure C99, CMake, no required dependencies beyond `libc`/`libm`. Optional Flex/Bison ≥ 3.0 (pre-generated parser sources are committed as a fallback).

```bash
# Standard build (Release, parallel)
./bootstrap.sh                              # Linux/macOS
bootstrap.bat                               # Windows
# or manually:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 8
```

Binaries land in `build/` on Linux/macOS, `build/Release/` on Windows. Each demo's training/data files are copied next to its binary by `add_demo()` POST_BUILD steps — **always run demos from `build/`**, not the source tree.

### Build flags (cmake `-D...`)

- `MICROGPT_USE_FLOAT=ON|OFF` — float32 (default) vs double64. Hyperparameters stay double for stability; test tolerances auto-adjust via `SCALAR_TOL`.
- `MICROGPT_SIMD=ON` (default) — `-march=native` / `/arch:AVX2`.
- `MICROGPT_BLAS=ON` — Apple Accelerate / OpenBLAS / MKL. Single-threaded only (Accelerate's internal threading conflicts with our pthread training).
- `MICROGPT_METAL=ON` — Apple Metal GPU. Worth it only for `N_EMBD ≥ 512`; for small models, dispatch overhead exceeds compute.
- `MICROGPT_PAGED_KV=ON`, `MICROGPT_HEAD_PARALLEL=ON`, `MICROGPT_ATTN_RES=ON`, `QUANTIZATION_INT8=ON`, `ENABLE_TURBOQUANT=ON`, `ENABLE_ROTORQUANT=ON` — opt-in features.
- **DeepSeek-V4 port stack** (default OFF, all four orthogonal):
  - `MICROGPT_PARTIAL_ROPE=ON` — rotates last `ROPE_DIMS` (default `min(head_dim, 32)`) of every per-head Q/K. Adds a closed-form rotation backward (no new params). Standalone −1.6% PPL on deep configs.
  - `MICROGPT_ATTN_SINK=ON` — adds `exp(ATTN_SINK_LOGIT)` (default −1.0) to softmax denominator per head. Standalone −3.1% PPL on deep configs.
  - `MICROGPT_QK_NORM=ON` — per-head RMSNorm of Q/K before dot product. Real value is high-LR stability; super-additive with sink.
  - `MSA_POOL_MODE=0..3` — pool weighting: 0 mean (default), 1 linear ramp, 2 exp recency, 3 content-aware (recommended).
  - **Recommended stack** for any model with `N_LAYER ≥ 2`, `BLOCK_SIZE ≥ 64`: enable all four — combined effect **−8.7% held-out PPL**, 0 new params, ~1% runtime overhead. Tests `test_microgpt_qk_norm` and `test_microgpt_rope` validate the new backward paths under each flag. See `docs/BUILD_OPTIONS.md` § "DeepSeek-V4 Port Stack" and the `RESEARCH_DEEPSEEK_V4_*.md` paper series in `docs/research/` for measurements.
- **Rope-aware MSA helpers** (when integrating MSA in a demo with `MICROGPT_PARTIAL_ROPE=ON`): replace `msa_pool_chunk` → `msa_pool_chunk_rope`, `msa_expand_context` → `msa_expand_context_rope`, `msa_recency_inject` → `msa_recency_inject_rope`. The rope-aware versions take `start_pos` and `n_head` so they can un-rotate K to position 0 before averaging and re-rotate at injection time. Track an `abs_pos_at_slot0` counter that bumps by `chunk_size` per chunking event. See `msa_infinite_shakespeare_v4` and `bench_microgpt_msa_sliding.c` for worked examples.

## Tests & benchmarks

```bash
# From build/:
./test_microgpt              # core engine
./test_microgpt_msa          # Memory Sparse Attention
./test_microgpt_turboquant   # 4-bit KV compression
./test_microgpt_rotorquant   # rotor quantisation
./test_microgpt_organelle    # organelle pipeline
./test_microgpt_pipeline     # Pipeline IR (graph DAG + verifier + text round-trip + DOT)
./test_microgpt_vm           # VM compiler/runtime (needs resources/vm/, copied automatically)

# Or run all registered tests via CTest:
ctest --test-dir build --output-on-failure

# Benchmarks (no asserts, print timings):
./bench_microgpt  ./bench_microgpt_msa  ./bench_microgpt_turboquant
./bench_microgpt_rotorquant  ./bench_microgpt_vm  ./bench_ssd
```

Tests use a homegrown harness (`tests/test.h` plus inline `TEST(name) { ... }` / `RUN(name)` macros in `test_microgpt.c`). To run a single test: build, then comment out unwanted `RUN(...)` calls in `main()` of the relevant test file, or filter by stdout — there is no `--filter` flag.

CI (`.github/workflows/cmake-multi-platform.yml`) builds and runs `test_microgpt` + `bench_microgpt` on Ubuntu (gcc/clang), macOS (clang), Windows (cl).

## Architecture

Three-layer system, all in `src/`:

```
demos/  — applications (Shakespeare, 11 games, VM codegen, lottery, MSA, TurboQuant)
   │
   ├── microgpt_organelle.c/.h  — OPA Kanban pipeline, planner→player→judge,
   │                              cycle detection, multi-organelle coordination
   │
   ├── microgpt_msa.c/.h        — Memory Sparse Attention (LRU-paged latent storage)
   ├── microgpt_turboquant.c/.h — 4-bit dual-state KV compression
   ├── microgpt_rotorquant.c/.h — rotor-based KV compression
   ├── microgpt_pipeline.c/.h   — Pipeline IR (typed graph DAG, verifier, text round-trip, DOT)
   ├── microgpt_vm.c/.h         — bytecode compiler + runtime (Flex/Bison frontend)
   └── microgpt_metal.{h,m,metal} — optional Apple Metal GPU kernels
   │
   └── microgpt.c/.h            — core: forward/backward, attention, Adam,
                                  KV-cache, tokeniser (char + word level),
                                  checkpoint I/O, TrainWorker pthread harness
```

**`microgpt.h` is a single-header API** (~1k lines) — including it gets you the whole core engine. `microgpt.c` is ~3k lines of implementation. Three more headers layer on top: `microgpt_organelle.h` for pipelines, `microgpt_pipeline.h` for the graph IR (typed dataflow DAG that the Wiring Organelle emits), `microgpt_vm.h` for VM scripting.

### Compile-time architecture (critical)

Model dimensions are **`#define` macros**, not runtime config: `N_EMBD`, `N_HEAD`, `N_LAYER`, `BLOCK_SIZE`, `MLP_DIM`, `NUM_STEPS`, `LEARNING_RATE`, `BATCH_SIZE`, `MAX_VOCAB`, `MAX_DOCS`, `MAX_DOC_LEN`, `GRAD_CLIP`, `LABEL_SMOOTH`, `WARMUP_STEPS`. Each demo passes its own values via `add_demo(... DEFINES ...)` in `CMakeLists.txt`, which causes the constants to constant-fold into matmul loops.

Because demos use different macro values, **each unique combination compiles its own `microgpt_lib_<md5>` static library variant** (see `_microgpt_lib_for_defines()` in `CMakeLists.txt:130`). This is why a small change to `microgpt.c` rebuilds many `.o` files — that's expected, not a bug. Never `#define` these in source files; always pass via the `add_demo(... DEFINES ...)` block.

To add a new demo: create `demos/<category>/<name>/main.c`, then register with `add_demo(NAME ... SOURCE ... [THREADS] [METAL] [BLAS] COPY data.txt DEFINES N_EMBD=... ...)` in the root `CMakeLists.txt`. Don't write your own `add_executable` — `add_demo` handles library variant selection, data file copy-out, threading, BLAS, and Metal shader copy.

### Organelle pipeline pattern

The "intelligence" claim is in the *coordination*: each organelle is its own checkpoint of the same ~30K–460K-param transformer, trained on a different role (planner / player / judge). They communicate via flat pipe-separated text strings — the planner's stdout is fed as the player's prompt. `OpaKanban` is the shared working memory (history, blocked moves, stalls). `OpaCycleDetector` breaks A↔B oscillations. The deterministic C scaffolding (~340 lines) does what gradient descent can't, freeing tiny models to be pattern-matchers.

When changing organelle wire formats, update both the producer's output template *and* the consumer's parser — they're symmetric and there is no schema enforcement.

### Pipeline IR + Wiring Organelle

`microgpt_pipeline.{h,c}` is an orthogonal optional module — no changes to the core engine. It defines a typed graph IR (`Pipeline`, `PipelineNode`, `PipelineEdge`, `PipelineType`, `@graph...@end` text format) with a verifier (cycle/connectivity/type checks) that doubles as a Judge for generated graphs. The **Wiring Organelle** (`demos/wiring_organelle/`) is a 540K-param word-level transformer trained on (prompt, graph) pairs from `tools/pipeline_corpus_gen.c`. Best-of-16 sampling + verify-as-judge + post-parse `pipeline_repair()` + Phase 6 end-to-end execution via `wiring_natives_dispatch()` (40 C-implemented primitives in `wiring_natives.{h,c}`) + Phase 7 reference-answer correctness check (`wiring_references.{h,c}`) gives **75% strict-verify, 40% end-to-end executed, 35% numerically correct on natural-English transfer** (87.5% accuracy among executing graphs). Three layered fallbacks for organelle outputs: `pipeline_parse_text` (strict) → `pipeline_parse_text_tolerant` (auto-promote refs, dedup sigs) → `pipeline_repair` (drop internally-inconsistent nodes/edges, drop unused sig ports). When extending the corpus generator, ensure `pipeline_render_text()` round-trips byte-stably via the canonical Kahn topo sort. Held-out NL prompts in `pipeline_corpus_held_out.txt` are annotated with `# EXPECTED:` (primitive set) and `# REFERENCE:` (reference fn name) — keep both in sync when adding new prompts. See `docs/research/RESEARCH_PIPELINE_IR.md`.

### VM engine

`microgpt_vm.l` (Flex) → `microgpt_vm.y` (Bison ≥ 3.0, uses `%define api.prefix`) → AST → bytecode → 6-pass verifier → stack-based runtime. macOS ships Bison 2.3 which **cannot** parse this grammar; CMake auto-falls back to the committed pre-generated `microgpt_vm_parser.{l,tab}.c`. If you regenerate them, install Bison ≥ 3.0 (`brew install bison && export PATH=/opt/homebrew/opt/bison/bin:$PATH`).

## Code style

- **C99 only** in core engine. C11/C23 features are not allowed in `microgpt.{h,c}`.
- **Zero deps** in core. Platform accelerators (Metal, BLAS, etc.) live behind `#ifdef` guards and are gated by CMake options.
- Use `scalar_t` for weights/activations — never hardcode `float` or `double`. Constants use `(scalar_t)0.5f` style.
- BLAS is dispatched through `CBLAS_GEMV`/`CBLAS_GER` macros that pick `s` vs `d` based on `scalar_t`.
- Optimizer state (Adam β1/β2/ε, LR) stays `double` regardless of `scalar_t`.
- Hot paths avoid `malloc` — prefer stack scratch buffers; the VM uses an arena pattern.
