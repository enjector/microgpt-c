# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research discipline (load-bearing — read this first)

**Do not synthesise across research documents from excerpts.** When asked to interpret, summarise, or build conclusions from any document in `docs/research/`, `docs/organelles/`, `experiments/`, or the sibling repo at `~/dev/projects/microgpt-c/`, you must either:

1. **Read the document end to end** via the Read tool (with explicit `offset` / `limit` pairs covering the full line count if necessary), OR
2. **Explicitly state that you have not done so** and refuse to draw conclusions, OR
3. **List which specific sections you have read in full and which are inferred from excerpts**, and clearly mark inferred-from-excerpts material as **unvalidated** in your output.

The failure mode this rule exists to prevent: reading `head -50` or `head -80` of a document, then writing confident multi-paragraph syntheses, strategic reframes, or experiment proposals based on those excerpts. This produces plausible-sounding output that the user has no way to challenge without reading the documents themselves — which is exactly the work they're delegating to you.

**Specific prohibitions:**

- **Do NOT propose strategic rewrites of `ORGANELLE_STATE.md`, the project README, or any experiment's Section 4 conclusion** without first listing every document you have read in full to support the claim.
- **Do NOT draft new pre-registered experiments (E1N)** that depend on findings from documents you have only sampled. If the new experiment's framing depends on understanding the prior arc, read the prior arc in full first.
- **Do NOT summarise or characterise the Markets arc (`ORGANELLE_MARKETS_V1.md` through `V24+`)** from any subset of those files. The arc is a 24-version progression; cherry-picked endpoints distort it.
- **Do NOT claim the Bonsai engine, Nucleus, or Organelle Lifecycle "does X"** without reading the corresponding `microgpt_bonsai.{h,c}`, `microgpt_nucleus.{h,c}`, `microgpt_organelle_lifecycle.{h,c}` source in full. Header-only reads do not count.

**Required behaviour when uncertain:**

When in doubt about whether you have enough grounding to make a claim, **produce a reading log first**: list every document, with line ranges read, that underpins the claim. If the log is thin, do the reading before continuing.

The user has the right to challenge any synthesis with *"name what you've read in full to support that claim"* and you must answer with a specific list of documents and line ranges — or retract the synthesis.

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

`microgpt_pipeline.{h,c}` is an orthogonal optional module — no changes to the core engine. It defines a typed graph IR (`Pipeline`, `PipelineNode`, `PipelineEdge`, `PipelineType`, `@graph...@end` text format) with a verifier (cycle/connectivity/type checks) that doubles as a Judge for generated graphs. The **Wiring Organelle** (`demos/wiring_organelle/`) is a 540K-param word-level transformer trained on 368 (prompt, graph) pairs from `tools/pipeline_corpus_gen.c`, plus a Phase 15 **planner organelle** (also 540K params, 2000 steps) trained on (prompt, graph_name) pairs that predicts a re-ranking hint. Best-of-16 + verify-as-judge + post-parse `pipeline_repair()` + Phase 6 native dispatch (40 C primitives in `wiring_natives.{h,c}`) + Phase 7 reference-answer suite (`wiring_references.{h,c}`) + Phase 8 multi-input self-consistency vote re-ranking + Phase 11 structural-diversity templates + Phase 12 lexical-anchoring paraphrases + Phase 13 three-bucket held-out-phrase corpus expansion + Phase 15 planner+graded-family-match re-ranking *previously claimed* 75% median / 80% peak at the wiring layer. **Phase 2d leakage audit (`RESEARCH_PIPELINE_IR.md` §38) found that 13 of 20 original held-out prompts appear verbatim in the wiring training corpus** (introduced by Phase 13's lexical-anchoring expansion at `tools/pipeline_corpus_gen.c` lines 1902, 1924, 1950, 1979, 2011, …). The honest restated numbers: **anchor-retrieval mechanism = 🎯 100% (20/20) on the leakage-free Phase 2c paraphrases**; **wiring transformer alone = 35% (7/20) on the same clean set**. The 17-phase corpus-engineering lift from 35→75% was largely the model memorising prompts that Phase 13 explicitly added to training. The Phase 1a/1b/1c manifold-retrieval diagnoses (re-ranking can't help unanimous failures, classification works, generation is the bottleneck) remain valid; Phase 2/2b/2c anchor-retrieval (`wiring_anchor_graphs.{h,c}` + `wiring_geo_classifier.{h,c}`, 20D Geodesic, unique-slot embedder) closes the bottleneck on genuinely novel inputs. CLI flags `--no-anchor` (disable anchor injection) and `--clean-only` (skip first 20 leaked prompts) reproduce the audit. **Phase 3a (`RESEARCH_PIPELINE_IR.md` §40 pre-registration + §41 falsification): a TF-IDF centroid classifier — the simplest learned encoder — scored 4/20 on the adversarial axis-2 stress test vs the pre-registered 12-16/20 prediction, and ~8/20 on the no-regression target vs the pre-registered ≥18/20. Per the pre-registered §40.7 skip condition, Phase 3a-full (EKAN-Network classifier) and Phase 3c (RAG fallback) are cancelled — at the 408-example corpus scale, no learned encoder beats the handcoded keyword bag.** **Phase 3b (`RESEARCH_PIPELINE_IR.md` §42 pre-registration + §43 results) shipped: 6/10 on a 10-prompt multi-stage composition test set, within the pre-registered 5-7/10 target, with 20/20 no-regression on Phase 2c clean paraphrases.** Implementation: `wiring_fragments.{h,c}` decomposes existing anchors into 15 reusable fragments; `wiring_compose_for_prompt()` picks top-2/3 fragments by keyword hits and chains them by output→input linkage. A fidelity-trumps gate (`score += 1000` when composition has the expected primitive set and no anchor does) makes composition dominate on multi-stage prompts without disturbing the single-anchor 100% headline. CLI flags `--composition` (eval against composition test file) and `--no-composition` (disable composition injection). **Arc closed at the architecture level (`RESEARCH_PIPELINE_IR.md` §44):** three of four boundary axes were open and corpus-bounded; axis 3 (multi-stage compositions) closed by Phase 3b at 60%. **Phase 4 (`RESEARCH_PIPELINE_IR.md` §45 pre-registration + §46 results) shipped: `tools/corpus_expand.c` (~370 LOC, deterministic) generates a 4,102-prompt expanded corpus from per-family synonym tables + sentence templates; TF-IDF centroid classifier on this corpus hits 18/20 (90%) on adversarial axis-2 (vs pre-registered 8-12/20 target) and 20/20 no-regression on Phase 2c clean. Phase 4b-full (EKAN-Network) cancelled per the pre-registered §45.2 outcome logic — simpler model exceeded the escalation trigger.** Axes 2 (weak keyword overlap) and 4 (domain-vocabulary drift) are now "soft-closed" within the curator's synonym table range. Remaining frontier is library extension (anchor table size, axis 1) or genuinely-out-of-distribution domain-vocabulary (still bounded by curator labour). Phase 4 (corpus expansion to 5k–50k examples) is the right corrective if scaling beyond 408 is desired. See `RESEARCH_PIPELINE_IR.md` §35–§42 and `RESEARCH_MANIFOLD_LEARNING.md`. The 6-phase diagnostic-prescription arc 8→9→10→11→12→13→15 documented in `docs/research/RESEARCH_PIPELINE_IR.md`: bimodal failure pattern (Phase 8) → capacity overfit (9) → paraphrases flat (10) → topology diversity (11) → lexical anchoring (12, +15pp) → three-bucket expansion (13, +25pp) → multi-organelle planner (15, +5pp moon target). Three layered fallbacks for organelle outputs: `pipeline_parse_text` (strict) → `pipeline_parse_text_tolerant` (auto-promote refs, dedup sigs) → `pipeline_repair` (drop internally-inconsistent nodes/edges, drop unused sig ports). When extending the corpus generator, ensure `pipeline_render_text()` round-trips byte-stably via the canonical Kahn topo sort. Held-out NL prompts in `pipeline_corpus_held_out.txt` are annotated with `# EXPECTED:` (primitive set) and `# REFERENCE:` (reference fn name) — keep both in sync when adding new prompts. See `docs/research/RESEARCH_PIPELINE_IR.md`.

### VM engine

`microgpt_vm.l` (Flex) → `microgpt_vm.y` (Bison ≥ 3.0, uses `%define api.prefix`) → AST → bytecode → 6-pass verifier → stack-based runtime. macOS ships Bison 2.3 which **cannot** parse this grammar; CMake auto-falls back to the committed pre-generated `microgpt_vm_parser.{l,tab}.c`. If you regenerate them, install Bison ≥ 3.0 (`brew install bison && export PATH=/opt/homebrew/opt/bison/bin:$PATH`).

## Code style

- **C99 only** in core engine. C11/C23 features are not allowed in `microgpt.{h,c}`.
- **Zero deps** in core. Platform accelerators (Metal, BLAS, etc.) live behind `#ifdef` guards and are gated by CMake options.
- Use `scalar_t` for weights/activations — never hardcode `float` or `double`. Constants use `(scalar_t)0.5f` style.
- BLAS is dispatched through `CBLAS_GEMV`/`CBLAS_GER` macros that pick `s` vs `d` based on `scalar_t`.
- Optimizer state (Adam β1/β2/ε, LR) stays `double` regardless of `scalar_t`.
- Hot paths avoid `malloc` — prefer stack scratch buffers; the VM uses an arena pattern.
