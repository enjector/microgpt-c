# MicroGPT-C Project Roadmap

A living roadmap for MicroGPT-C — a zero-dependency C99 GPT engine designed for composable, autonomous edge intelligence. See [VISION.md](VISION.md) for the full "Stem Cell" philosophy and [VALUE_PROPOSITION.md](VALUE_PROPOSITION.md) for the business case.

---

## Spear Summary

**Point:** The engine works — five experiments prove it. Now it needs tooling (CLI, organelle API) to go from research demos to production pipelines.

**Picture:** We've built the LEGO bricks and proved they snap together. The next step is the instruction manual and the box.

**Proof:** Names (trains in < 1s), Shakespeare (840K params), 8-puzzle (96.7% solve, 3-organelle pipeline), Connect-4 (85% wins), C code generation (byte-perfect retrieval of 2,081 functions) — all validated.

**Push:** The Q2 2026 organelle toolkit is the critical next step — it turns five separate `main.c` files into a single `microgpt create/train/infer` CLI.

---

## ✅ Completed (Q1 2026)

### Core Engine
- [x] Full GPT-2 architecture: multi-head attention, RMSNorm, MLP, residual connections
- [x] Manual forward + backward pass (no autograd dependency)
- [x] Adam optimiser with cosine LR schedule + linear warmup
- [x] Two tokenisation strategies: character-level and word-level (O(1) hash lookup)
- [x] INT8 quantisation with per-matrix scales
- [x] Training checkpoints (save/resume with optimizer momentum)
- [x] `scalar_t` precision abstraction — compile-time `double`/`float` switch via `-DMICROGPT_USE_FLOAT=ON`
- [x] Precision-aware math macros (`M_EXP`, `M_LOG`, `M_SQRT`, etc.) and BLAS dispatch (`CBLAS_GEMV`/`CBLAS_GER`)

### Performance & Optimisation
- [x] SIMD auto-vectorisation (`-march=native -ffast-math -funroll-loops`)
- [x] Cache tiling in `lin_fwd`/`lin_bwd` (32×64 panels for L1 residency)
- [x] Apple Accelerate BLAS integration (opt-in via `-DMICROGPT_BLAS=ON`)
- [x] Metal GPU compute shaders for Apple Silicon (opt-in via `-DMICROGPT_METAL=ON`)
- [x] Per-head attention parallelism (opt-in via `-DMICROGPT_HEAD_PARALLEL=ON`)
- [x] Paged KV cache for memory-constrained deployments (opt-in via `-DMICROGPT_PAGED_KV=ON`)
- [x] Cross-platform multi-threaded training via portable `microgpt_thread.h`
- [x] Common training helpers extracted to library (`TrainWorker`, `train_worker_run`, `shuffle_docs`)

### Testing & Documentation
- [x] 44 unit tests covering all public API functions
- [x] 15 performance benchmarks with measured throughput
- [x] Comprehensive README with build instructions, examples, and benchmarks
- [x] [Optimisation Strategies](docs/foundation/OPTIMISATION_STRATEGIES.md) technical white paper (9 strategies documented)
- [x] Character-level and word-level tokenisation guides

### Demos & Experiments
- [x] **names** — character-level name generation (4K params, trains in < 1s)
- [x] **shakespeare** — character-level Shakespeare (840K params, multi-threaded, zero `<unk>`)
- [x] **c_codegen** — C code generation from prompts (875K params, byte-perfect recall of 2,081 functions)
- [x] **c_wiringgen** — C function composition grammar (875K params, training in progress)
- [x] **tic-tac-toe** — 2-organelle pipeline: Planner → Player (82% win+draw vs random)
- [x] **8-puzzle** — 3-organelle pipeline: Planner → Mover → Judge with kanban coordination (96.7% solve rate)
- [x] **Connect-4** — 2-organelle pipeline: Planner → Player (85% wins despite 60% invalid moves)
- [x] Multi-organelle [experiment READMEs](experiments/organelles/) with Spear summaries
- [x] [Design documents](docs/organelles/) — pipeline wire format, kanban planner, CLI vision

---

## 🔜 Short-Term (Q2 2026): Stem Cell Foundation

Focus: make MicroGPT-C production-ready as a **differentiable LEGO block** for edge deployment.

### 1. Organelle Toolkit
- [ ] **`microgpt_organelle.h`** — high-level API for the stem cell lifecycle: `organelle_create()` → `organelle_train()` → `organelle_infer()` → `organelle_confidence()`
- [ ] Built-in confidence scoring via softmax entropy (wraps existing `forward_inference` + softmax)
- [ ] Organelle serialisation: save/load specialised blocks as compact `.bin` files with embedded vocab

### 2. Replay Buffer for On-Device Learning
- [ ] Ring buffer of representative training examples to prevent catastrophic forgetting during incremental updates
- [ ] `organelle_correct(block, input, expected_output)` — single-example correction with automatic replay

### 3. Robustness
- [ ] CI pipeline (GitHub Actions: Linux, macOS, Windows) for both `double` and `float` builds
- [ ] AddressSanitizer / UBSan clean across all demos and tests
- [ ] Equivalence tests: compare logits against Python reference on toy datasets

### 4. BPE Tokenisation
- [ ] Byte Pair Encoding tokeniser for better token efficiency on structured text
- [ ] Compatible with the existing `Vocab` API so organelles can use either char, word, or BPE

---

## 🔮 Medium-Term (Q3–Q4 2026): Composable Intelligence

Focus: enable **organelle chaining** — multiple specialised blocks working together as autonomous pipelines.

### 1. Organelle Chaining Protocol
- [ ] Lightweight IPC for composing blocks: output of one organelle feeds input of the next
- [ ] Pipeline definition format: `validator → formatter → completer`
- [ ] Confidence-gated routing: if block A's confidence drops below threshold, route to block B (fallback) or escalate to human

### 2. Model Distillation Pipeline
- [ ] Tools to generate high-quality training corpora from large cloud models (GPT-4, Claude, etc.)
- [ ] Distillation workflow: cloud model → corpus → `organelle_train()` → compact edge block
- [ ] Evaluation framework: measure distilled block quality against teacher model

### 3. FP16 / BF16 Support
- [ ] Extend `scalar_t` to support `_Float16` / `__bf16` on supported hardware
- [ ] ARM FP16 NEON intrinsics for 8-wide SIMD on Apple Silicon / Cortex-A
- [ ] Metal FP16 shaders (native, no conversion overhead)

### 4. Dynamic Vocabulary
- [ ] Replace `MAX_VOCAB` compile-time limit with realloc-based growth
- [ ] UTF-8 aware tokenisation for non-English corpora
- [ ] Vocabulary merging: combine vocabs from multiple organelles

### 5. Community & Release
- [ ] v1.0 release with changelog and binary releases for major platforms
- [ ] Example organelle library: pre-trained blocks for common tasks (email validation, date parsing, unit conversion)
- [ ] Documentation: "Build Your First Organelle" tutorial

---

## 🔭 Long-Term (2027+): Autonomous Edge Ecosystem

Focus: a world where intelligence is a **composable, self-evolving component** of every device.

### 1. Federated Differentiation
- [ ] Multiple edge devices contribute gradient updates to improve a shared organelle
- [ ] Privacy-preserving: raw data never leaves the device; only aggregated gradients are shared
- [ ] Differential privacy guarantees for gradient aggregation

### 2. Organelle Marketplace
- [ ] Repository of pre-specialised `.bin` organelles indexed by task domain
- [ ] Version control and compatibility metadata for organelle updates
- [ ] Over-the-air organelle updates for deployed edge devices

### 3. Self-Monitoring Organelles
- [ ] Built-in drift detection: organelle monitors its own confidence distribution over time
- [ ] Automatic re-differentiation trigger when confidence degrades below threshold
- [ ] Telemetry-free: all monitoring happens on-device

### 4. Hardware Targets
- [ ] RISC-V embedded support (no FPU fallback for INT8 organelles)
- [ ] ESP32 / STM32 demo: running an organelle on a $5 microcontroller
- [ ] FPGA acceleration for ultra-low-latency inference

---

## Guiding Principles

| Principle | Meaning |
| --- | --- |
| **Zero dependencies** | The core engine must never require external libraries beyond libc + libm |
| **Stem cell, not monolith** | Every feature should make blocks smaller and more focused, not larger and more general |
| **Train where you deploy** | On-device learning is a first-class capability, not an afterthought |
| **Confidence over correctness** | A block that knows when it's wrong is more valuable than one that's always "right" |
| **C99 everywhere** | If it compiles with `cc -std=c99`, it ships |

---

*Roadmap updated February 2026. Priorities are flexible and driven by the [stem cell vision](VISION.md).*
