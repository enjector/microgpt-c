# NFRD — MicroGPT-C Non-Functional Requirements Document

**Document ID:** MGC-NFRD-001
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30

---

## 1. Purpose

This document captures the qualities the MicroGPT-C platform MUST exhibit, beyond the per-feature behaviour in `FRD.md`. Where a non-functional requirement has a measured target, it is expressed as an `SLO-` ID; the SLO carries a measurement methodology and a benchmark file.

NFRs and SLOs in this document are platform-wide. Subsystem-specific SLOs live in the relevant `BS_*.md`.

## 2. Portability and toolchain

| ID | Requirement |
|---|---|
| NFR-001 | The core engine SHALL build on Linux, macOS, and Windows with a C99-compliant compiler (GCC, Clang, MSVC). |
| NFR-002 | The core engine SHALL NOT use C11 or C23 features. |
| NFR-003 | The core engine SHALL depend only on `libc` and `libm`. |
| NFR-004 | The build SHALL succeed without Flex/Bison installed; pre-generated VM parser sources are committed as a fallback. |
| NFR-005 | The build SHALL succeed without Apple Accelerate, OpenBLAS, MKL, or Metal frameworks; these are opt-in via CMake flags. |
| NFR-006 | The build SHALL produce binaries that run on x86-64, ARM64 (Apple Silicon), and ARMv7-A (compatibility target — runtime testing on small SBCs is documented future work, not an SLO). |
| NFR-007 | A CI matrix SHALL exercise `gcc` + `clang` on Ubuntu, `clang` on macOS, and `cl` on Windows for at least `test_microgpt` and `bench_microgpt`. |

## 3. Determinism and reproducibility

| ID | Requirement |
|---|---|
| NFR-010 | Given a fixed RNG seed (`seed_rng(s)`) and a fixed corpus, the platform SHALL produce identical model weights to within `scalar_t` precision after the same number of training steps, on the same compiler / `scalar_t` / feature-flag combination. |
| NFR-011 | A `MICROGPT_USE_FLOAT=ON` build and a `MICROGPT_USE_FLOAT=OFF` build SHALL converge to the same target loss on the names demo (loss ≈ 0.0011 after 100 steps), within their respective scalar tolerances. |
| NFR-012 | Test tolerances SHALL be expressed via a `SCALAR_TOL` macro that auto-adjusts between the float and double builds. |
| NFR-013 | The platform SHALL emit a deterministic configuration banner via `microgpt_print_config(name, cfg)` so that benchmark logs can be matched to the build configuration that produced them. |

## 4. Performance — measured SLOs

The following SLOs are reproducible on the dev reference machine (Apple M2 Max, 12 threads where multi-threading is invoked) using the benchmarks under `tests/bench_*.c`. Each SLO is a *target*; regression below the target is a P1 issue.

### 4.1 Core engine (single-threaded, float32, vocab=50, `N_EMBD=16`, `N_LAYER=1`)

| ID | SLO | Measurement | Benchmark |
|---|---|---|---|
| SLO-CORE-001 | `forward_backward_one`: ≥ 500 K fwd+bwd ops/s (float32). | Median over 1 s warm-up + 5 s measurement. | `bench_microgpt.c::bench_forward_backward` |
| SLO-CORE-002 | `adam_step`: ≥ 600 K steps/s (float32). | As above. | `bench_microgpt.c::bench_adam_step` |
| SLO-CORE-003 | `sample_token`: ≥ 6 M samples/s (float32, vocab=50). | As above. | `bench_microgpt.c::bench_sample_token` |
| SLO-CORE-004 | Full training step (seq=8, float32): ≥ 600 K tok/s. | As above. | `bench_microgpt.c::bench_full_step` |
| SLO-CORE-005 | `forward_inference` single token: ≥ 1.5 M infer/s (float32). | As above. | `bench_microgpt.c::bench_forward_inference` |
| SLO-CORE-006 | Auto-regressive inference (seq=16, float32): ≥ 1 M tok/s. | As above. | `bench_microgpt.c::bench_autoreg` |
| SLO-CORE-007 | `checkpoint_save` + `checkpoint_load` round trip: ≥ 4,500 rt/s (float32). | As above. | `bench_microgpt.c::bench_checkpoint_roundtrip` |

### 4.2 Tokeniser

| ID | SLO | Measurement | Benchmark |
|---|---|---|---|
| SLO-TOK-001 | Character tokenisation: ≥ 30 M tok/s on a 12-character string. | Median over warm-up + measurement. | `bench_microgpt.c::bench_tokenize` |
| SLO-TOK-002 | Word tokenisation: ≥ 800 K tok/s on a 1 KB text. | As above. | `bench_microgpt.c::bench_tokenize_words` |
| SLO-TOK-003 | Word vocabulary build (1 KB text, top-10000): ≥ 200 K builds/s. | As above. | `bench_microgpt.c::bench_word_vocab_build` |

### 4.3 End-to-end demos (Apple M2 Max, multi-thread where indicated)

| ID | SLO | Notes |
|---|---|---|
| SLO-CORE-010 | Names demo: train 1 K steps, ≤ 0.1 s. | Validates the "minute-class" stem-cell training claim. |
| SLO-CORE-011 | Character Shakespeare: ~841 K params, training ≥ 28 K tok/s with 12 threads, inference ≥ 16 K tok/s, training time ~ 14 min for 30 K steps. | `c_shakespeare_demo` |
| SLO-CORE-012 | Word Shakespeare: ~510 K params, training ≥ 12.5 K tok/s with 12 threads, inference ≥ 40 K tok/s, ~ 2 min for 10 K steps. | `w_shakespeare_demo` |
| SLO-CORE-013 | VM dispatch: ≥ 3.7 M ops/s, single-threaded. | `bench_microgpt_vm.c` |
| SLO-CORE-014 | SSD ensemble (5-vote, prefix cache shared): ≥ 1.9× faster than the equivalent independent-vote ensemble. | `bench_ssd.c` |

### 4.4 MSA + KV compression

| ID | SLO | Notes |
|---|---|---|
| SLO-MSA-001 | MSA pool encode at production batch sizes: ≥ 1.3 M encodes/s on M2 Max. | `bench_microgpt_msa.c` |
| SLO-QUANT-001 | TurboQuant 4-bit produces ≥ 8× memory reduction vs raw `scalar_t` KV. | Static, computed from layout. |
| SLO-QUANT-002 | TurboQuant ≥ 25 % inference speedup on demos that integrate it under sufficient context length. | `bench_microgpt_turboquant.c` |

### 4.5 DeepSeek-V4 port stack (deep config)

| ID | SLO | Notes |
|---|---|---|
| SLO-CORE-020 | Combined V4 stack (`MICROGPT_PARTIAL_ROPE` + `MICROGPT_ATTN_SINK` + `MICROGPT_QK_NORM` + `MSA_POOL_MODE=3`): ≥ 8.7 % held-out PPL improvement on the deep benchmark (`N_LAYER ≥ 2`, `BLOCK_SIZE ≥ 64`), with no new parameters and ≤ 2 % runtime overhead. | `RESEARCH_DEEPSEEK_V4_PORTING.md` |
| SLO-CORE-021 | Each individual flag SHALL be measurable in isolation via the `bench_*` targets in `CMakeLists.txt`. | |

### 4.6 Wiring organelle (post-leakage-audit honest baselines)

| ID | SLO | Notes |
|---|---|---|
| SLO-WIRE-001 | Anchor-retrieval mechanism on Phase 2c clean (leakage-free) paraphrases: 100 % (20/20). | `RESEARCH_PIPELINE_IR.md` §41 |
| SLO-WIRE-002 | Wiring transformer alone on the same clean set: ≥ 35 % (7/20). | As above |
| SLO-WIRE-003 | Phase 3b composition multi-stage prompts: ≥ 60 % (6/10). | `RESEARCH_PIPELINE_IR.md` §43 |
| SLO-WIRE-004 | Phase 4 TF-IDF on the expanded ≥ 4,000-prompt corpus, adversarial axis-2: ≥ 90 % (18/20), with 100 % (20/20) no-regression on Phase 2c clean. | `RESEARCH_PIPELINE_IR.md` §46 |
| SLO-WIRE-005 | Phase 5 compositional held-out — pre-registered target was ≥ 50 % (15/30); V1.0.7 (Phase 6b) achieved baseline is **40 % (12/30) verified-and-correct, 100 % (30/30) verified** (lifted from V1.0.5 30 % via four-root-cause fix in `COMPOSITIONAL_GENERATOR_FIX_PLAN.md` v2.0). 40 % is the new SLO baseline; the 50 % target persists in `GAP-WIRE-006`. | `RESEARCH_DISCLOSURE.md` §5.5 |
| SLO-WIRE-006 | Phase 5 compositional with `--use-expected` ranker gate: pre-registered target ≥ 80 % (24/30) — the gate is not yet implemented (would require ranker integration in V1.0.5; postponed alongside the beam widening per `GAP-WIRE-006`). | As above |
| SLO-WIRE-007 | Phase 5 no-regression: 100 % (20/20) on Phase 2c clean must persist when the type-directed search is integrated alongside the existing anchor / transformer paths. (Anchor / transformer paths unchanged in V1.0.5; no-regression upheld.) | As above |

## 5. Memory and footprint

| ID | Requirement |
|---|---|
| NFR-020 | A 30 K-parameter checkpoint SHALL fit in ≤ 360 KB on disk and in working memory. |
| NFR-021 | A 460 K-parameter checkpoint SHALL fit in ≤ 5.4 MB on disk and in working memory. |
| NFR-022 | The smallest demo binary (names demo with default flags) SHALL be ≤ 100 KB on Linux x86-64 release. |
| NFR-023 | The platform SHALL avoid `malloc` in hot inner loops; gradients, KV cache, and matmul scratch buffers SHALL be pre-allocated outside the per-token critical path. |
| NFR-024 | The VM SHALL use an arena allocator for compilation-time scratch state. |
| NFR-025 | When `MICROGPT_PAGED_KV` is enabled, KV memory SHALL be allocated lazily in `KV_PAGE_SIZE × n_embd × scalar_t`-sized pages. |

## 6. Numerical stability

| ID | Requirement |
|---|---|
| NFR-030 | Training SHALL exhibit zero `NaN` instability across the bundled demos: SwiGLU is not used (the engine uses ReLU MLP), but RMSNorm with an epsilon of `1e-5` SHALL be the default normalisation. |
| NFR-031 | Optimiser hyperparameters (`BETA1 = 0.85`, `BETA2 = 0.99`, `EPS_ADAM = 1e-8`, default `LEARNING_RATE = 0.01`) SHALL remain `double` regardless of `scalar_t`. |
| NFR-032 | When `MICROGPT_QK_NORM` is enabled, the engine SHALL remain stable at higher learning rates: at `LR = 0.02` the un-normed baseline diverges to PPL 731; with the flag, PPL stays at 205 (3.6× recovery). This is a *regression* SLO — the gain SHALL persist. |
| NFR-033 | An optional gradient-clipping path SHALL be available via `clip_gradients` and `GRAD_CLIP > 0` for users who push training into instability regimes. |
| NFR-034 | A regression test SHALL exist in `tests/test_microgpt.c` that compares MicroGPT-C output logits against a reference PyTorch implementation; tolerances are auto-adjusted by `scalar_t`. |

## 7. Concurrency

| ID | Requirement |
|---|---|
| NFR-040 | Training SHALL support batch parallelism via the `TrainWorker` + `train_worker_run` harness. |
| NFR-041 | Each training worker SHALL own its own gradient buffer, KV cache, token buffer, and RNG seed; the only shared mutable state SHALL be the model weights, written only between worker phases. |
| NFR-042 | Inference SHALL be single-threaded by default (the small-model regime makes head-parallel dispatch overhead-bound); the optional `MICROGPT_HEAD_PARALLEL` SHALL parallelise attention heads when enabled. |
| NFR-043 | When `MICROGPT_BLAS` is enabled, BLAS SHALL be invoked single-threaded; multi-threaded BLAS conflicts with the pthread training harness and the build SHALL warn against the combination. |

## 8. Security and responsible use

| ID | Requirement |
|---|---|
| NFR-050 | The platform SHALL emit no telemetry by default. |
| NFR-051 | The platform SHALL make no outbound network calls in any default-build configuration. |
| NFR-052 | All training data SHALL remain local; the user explicitly chooses the corpus path passed to `load_docs`, `load_file`, or `opa_load_docs_multiline`. |
| NFR-053 | Documentation SHALL warn that small models trained on narrow corpora inherit corpus biases. |
| NFR-054 | High-confidence model output SHALL NOT be presented as ground truth; the deterministic Judge / verifier path SHALL filter outputs for safety-critical applications. |
| NFR-055 | Threat models (`TSM_*.md`) and per-framework compliance mappings (`COMPLIANCE_*.md`) are deferred to Phase 3 and SHALL be added before any regulated-vertical product ship. |

## 9. Maintainability

| ID | Requirement |
|---|---|
| NFR-060 | The core engine SHALL fit in two files (`microgpt.h`, `microgpt.c`) totaling ≤ 5,000 lines combined. The current size (~3,600 + ~1,200) is the operating budget; growth past 5,000 lines requires a documented justification. |
| NFR-061 | Each subsystem header SHALL declare its public API and document each function with a short prose comment. |
| NFR-062 | Compile-time architecture macros SHALL NEVER be `#define`d in source files; they SHALL be passed via `add_demo(... DEFINES ...)` in CMake. |
| NFR-063 | The build system SHALL avoid duplicate compilation by caching library variants by their compile-define hash (`_microgpt_lib_for_defines`). |
| NFR-064 | Each demo SHOULD register via the `add_demo` CMake helper rather than defining its own `add_executable`. |

## 10. Testing

| ID | Requirement |
|---|---|
| NFR-070 | The platform SHALL ship a homegrown test harness in `tests/test.h` with `TEST(name) { ... }` and `RUN(name)` macros. |
| NFR-071 | Test executables SHALL include at least: `test_microgpt`, `test_microgpt_msa`, `test_microgpt_turboquant`, `test_microgpt_rotorquant`, `test_microgpt_organelle`, `test_microgpt_pipeline`, `test_microgpt_vm`, `test_microgpt_geodesic`, `test_microgpt_vr`, `test_microgpt_ekan`, `test_microgpt_ekan_network`. |
| NFR-072 | All registered tests SHALL be invokable via `ctest --test-dir build --output-on-failure`. |
| NFR-073 | Benchmarks (`bench_*`) SHALL print timings and SHALL NOT contain assertions; they are measurement, not verification. |
| NFR-074 | At least 51/51 unit tests SHALL pass for the pipeline IR module, and 16/16 for the geodesic module, as documented in `STRATEGY_ONE_PAGER.md`. |

## 11. Documentation

| ID | Requirement |
|---|---|
| NFR-080 | The repository SHALL ship a top-level `README.md`, `VISION.md`, `VALUE_PROPOSITION.md`, `ROADMAP.md`, `FAQ.md`, `CONTRIBUTING.md`, `LICENSE`, `DATA_LICENSE.md`. |
| NFR-081 | The `docs/` directory SHALL ship `ARCHITECTURE.md`, `DESIGN.md`, `FUNCTIONAL_SPEC.md`, `BUILD_OPTIONS.md`, `EXTENDING_WIRING_ORGANELLE.md`, `DEPENDENCY_POLICY.md`, productisation sketches, and a `research/` folder with the per-feature research notes. |
| NFR-082 | This corpus (`docs/engineering/CLEAN_ROOM_IMPLEMENTATION/`) is the prescriptive engineering counterpart to the descriptive `docs/` folder. |
| NFR-083 | A 16-chapter book SHALL be published in `book/`, both as a `.md` aggregate and a `.pdf`. |

## 12. Licensing

| ID | Requirement |
|---|---|
| NFR-090 | The platform SHALL be licensed MIT (`LICENSE`). |
| NFR-091 | Training-data licensing SHALL be documented separately in `DATA_LICENSE.md`; data licences (e.g., the Shakespeare corpus public-domain status) SHALL NOT be assumed to follow the source code's MIT licence. |
| NFR-092 | Pretrained model checkpoints SHALL be made available via Git LFS for users who do not wish to retrain from scratch. |

## 13. Cross-references

- `BRD.md` — business requirements that motivate these qualities.
- `FRD.md` — functional surface to which these qualities apply.
- `BS_*.md` — per-subsystem invariants and SLOs in RFC 2119 voice.
- `docs/testing/PERFORMANCE.md` — measurement methodology for the perf SLOs.
- `docs/BUILD_OPTIONS.md` — feature-flag documentation.
- `book.7th/Reversible_Engineering.md` Chapter 6 — Phase 3 NFR derivation under adversarial review.

## 14. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
