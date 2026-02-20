# Frequently Asked Questions (FAQ)

- [🧠 Conceptual & Architectural](#-conceptual--architectural) — OPA, stem cells, composable intelligence
- [🛠 Technical Implementation](#-technical-implementation) — tokenisation, pipe-strings, training, organelle library, build config
- [🧪 Experiments & Results](#-experiments--results) — 16 experiments, markets vs lottery, adding your own
- [⚡ Performance & Comparison](#-performance--comparison) — llama.cpp, speed claims, memory, accelerators
- [📚 Getting Started](#-getting-started) — build, learn, use as a library
- [⚖️ Safety & Reliability](#️-safety--reliability) — hallucinations, lottery, randomness
- [🤝 Contributing & Community](#-contributing--community) — how to help, who built this, license

---

## 🧠 Conceptual & Architectural

### What is the "Organelle Pipeline Architecture" (OPA)?

OPA is a coordination framework for Small Language Models (SLMs). Instead of training one large monolith, we train multiple "micro-models" (organelles) specialized for specific roles: **Planners** (strategy), **Workers** (execution), and **Judges** (validation). By piping the output of one organelle into the next, the system architecture compensates for the cognitive limits of individual micro-models. See [ORGANELLE_PIPELINE.md](docs/organelles/ORGANELLE_PIPELINE.md) for the full design.

### Why use Agile/Scrum roles like "Planner" and "Judge"?

Decomposing intelligence into specific roles is a proven pattern in software engineering (Agile) and biology (cellular organelles). In MicroGPT-C, this pattern transforms a probabilistic prediction into a deterministic pipeline. For example, a "Judge" organelle acts as a safety gate, ensuring that a "Player" organelle never makes an illegal move in a logic game.

### What is "Stem Cell Intelligence"?

This is our long-term vision. Rather than shipping hardware with a static, pre-trained AI, we ship a "stem cell" model — a blank, high-efficiency engine. On the edge, this model "differentiates" (specialises) by training on local data and environment-specific tasks, becoming a unique organelle for that specific device. See [VISION.md](VISION.md) for the full whitepaper.

### What is "Composable Intelligence"?

The idea that you don't need one brilliant model — you need several focused ones that work together. A single Connect-4 organelle produces ~50% invalid moves. But wrapped in a Judge + replan loop, the *system* wins 88% of games. The coordination is the intelligence, not any individual model.

---

## 🛠 Technical Implementation

### Why character-level symbols instead of word-level tokens?

For models under 1M parameters, vocabulary overhead is a significant bottleneck. Word-level models waste massive amounts of parameter capacity on embedding tables for natural language. By using **character-level tokens**, we:

1. **Reduce overhead**: Keep the model footprint extremely small.
2. **Master protocols**: Allow models to master structured pipe-strings (e.g., `board=XO_|valid=1,3`) where they would otherwise fail at complex natural language syntax.
3. **Byte-perfect recall**: Achieve higher precision for structured data like C code synthesis.

### How do organelles communicate?

They use a shared **pipe-string protocol**. Instead of passing verbose natural language, organelles pass compact, structured strings.

*Example:*
- **Planner output**: `move=3|logic=defensive`
- **Worker input**: `board=...|planner_hint=defensive`
- **Worker output**: `action=place(3)`

This wire format is documented in [ORGANELLE_PIPELINE.md](docs/organelles/ORGANELLE_PIPELINE.md).

### How does on-device training work?

MicroGPT-C includes a pure C99 implementation of the **Adam optimiser** with **cosine LR scheduling** and **linear warmup**. Because the models are so small (<1M parameters), "differentiation" (training a specialised checkpoint) can happen on a standard CPU in seconds. Multi-threaded gradient accumulation via `microgpt_thread.h` parallelises batch processing across all available cores. This allows true edge learning without cloud dependencies.

### What is the shared organelle library?

[`microgpt_organelle.c|h`](src/microgpt_organelle.h) provides reusable infrastructure so you don't have to write boilerplate for every experiment:

| Component | Purpose |
|-----------|---------|
| `Organelle` struct | Bundles model + vocab + docs into one trainable/inferrable unit |
| `OpaKanban` | Blocked actions, move history, stall tracking |
| `OpaCycleDetector` | A↔B oscillation breaking |
| `organelle_train()` | Full training lifecycle with multi-threaded gradient accumulation |
| `organelle_generate()` | Temperature-controlled inference from prompt |

It eliminates 300–500 lines of boilerplate per demo and enables ensemble voting + valid-move pre-filtering across all game experiments.

### What architecture parameters can I configure?

All architecture parameters are **compile-time constants** set via CMake defines. Each demo has its optimal configuration baked in, but you can override them:

| Parameter | Default | Override | Effect |
|-----------|---------|----------|--------|
| `N_EMBD` | 16 | `-DN_EMBD=128` | Embedding dimension |
| `N_HEAD` | 4 | `-DN_HEAD=8` | Attention heads |
| `N_LAYER` | 1 | `-DN_LAYER=4` | Transformer blocks |
| `BLOCK_SIZE` | 16 | `-DBLOCK_SIZE=256` | Maximum sequence length |
| `scalar_t` | `float` | `-DMICROGPT_USE_FLOAT=OFF` | Switch to `double` precision |

See [BUILD_OPTIONS.md](docs/BUILD_OPTIONS.md) for the full set of options including Metal GPU, BLAS, INT8 quantisation, and SIMD flags.

---

## 🧪 Experiments & Results

### How many experiments have you run?

**Sixteen experiments** across three categories:

| Category | Experiments | Highlights |
|----------|-------------|------------|
| **Logic games** (11) | Tic-Tac-Toe, Connect-4, 8-Puzzle, Sudoku, Mastermind, Othello, Pentago, Hex, Lights Out, Klotski, Red Donkey | 87–91% win rates, zero invalid moves |
| **Code generation** (3) | c_codegen, c_wiringgen, c_compose | 83% exact match, 98% parse rate |
| **Real-world data** (2) | Market regime detection, Lottery prediction | 57% holdout accuracy vs 0.50 entropy floor |

See the full leaderboard in [ORGANELLE_GAMES.md](docs/organelles/ORGANELLE_GAMES.md) and individual experiment READMEs in `experiments/organelles/`.

### What do the market and lottery experiments prove?

They demonstrate that OPA can distinguish **learnable signal from randomness**:

- **Market regime detection** (positive test): A 3-organelle pipeline trained on real cross-asset financial data achieves **57% accuracy on unseen data** — 2.8× the random baseline of 20%. The model reached 0.03–0.06 training loss.
- **Lottery prediction** (negative control): A 2-organelle pipeline trained on EuroMillions draws hit an **entropy floor at ~0.50 loss** — exactly what theory predicts for random data. It learned nothing.

Same engine, same architecture. One learns, one can't. **That's the proof** that the engine has integrity and doesn't hallucinate patterns where none exist.

### Can I add my own game or experiment?

Yes. The shared organelle library makes this straightforward. Each experiment follows the same pattern:

1. Define your game state as a pipe-string (e.g., `board=...|valid=1,3,5`)
2. Generate training data with a random opponent or corpus
3. Train organelles using `organelle_train()`
4. Wire them into a pipeline with `OpaKanban` for coordination

Look at any experiment in `experiments/organelles/` as a template. Simpler games (Tic-Tac-Toe, Pentago) are good starting points.

---

## ⚡ Performance & Comparison

### How does this compare to `llama.cpp`?

`llama.cpp` is the gold standard for running large models (LLMs) with high-level optimisations. **MicroGPT-C** focuses on the "Micro" niche (SLMs). It is a complete lifecycle engine (inference **+ training**) designed for embedding intelligence into constrained environments like microcontrollers where even a 1B parameter model is too large.

| Feature | MicroGPT-C | llama.cpp |
|---------|-----------|-----------|
| Dependencies | **Zero** | cmake + stdlib |
| Training on-device | **Yes** | No |
| Smallest binary | **~50 KB** | ~2 MB |
| Composable models | **Yes (organelles)** | No |

### Is the "1,000× faster than Python" claim accurate?

Yes. This comparison is against the original `microgpt.py` script by Andrej Karpathy. By moving to pure C99, implementing a localised KV cache, and leveraging hardware-specific acceleration (like Metal on macOS), we achieved sub-millisecond inference and significantly faster training iterations.

### What are the memory requirements?

A typical 460K parameter model has a memory footprint of just a few megabytes. You can tune the precision at compile-time using `scalar_t` (`float` vs `double`) to fit your hardware's RAM constraints. INT8 quantisation (`-DQUANTIZATION_INT8=ON`) further reduces this by 4×.

### What hardware accelerators are supported?

The core engine is **pure C99 with zero dependencies** — it compiles anywhere. Optional accelerators are enabled via CMake flags:

| Accelerator | Flag | When to use |
|------------|------|-------------|
| SIMD auto-vectorisation | On by default | Always (compiler handles it) |
| Metal GPU | `-DMICROGPT_METAL=ON` | macOS, models with N_EMBD ≥ 512 |
| Apple Accelerate / BLAS | `-DMICROGPT_BLAS=ON` | Single-threaded inference |
| Multi-threaded training | Built-in | Multi-core CPUs |

For current model sizes, pure C99 with compiler auto-vectorisation is actually the fastest option. See [PERFORMANCE.md](docs/PERFORMANCE.md) for benchmarks.

---

## 📚 Getting Started

### How do I build and run it?

```bash
git clone https://github.com/enjector/microgpt-c.git
cd microgpt-c
mkdir build && cd build
cmake ..
cmake --build . --config Release

# Train a name generator in < 1 second (4K params)
./names_demo

# Train Shakespeare text generation (840K params, multi-threaded)
./shakespeare_demo

# Run a multi-organelle game pipeline (88% win rate)
./connect4_demo
```

Requirements: a **C99 compiler** (GCC, Clang, or MSVC) and **CMake 3.10+**. Nothing else.

### Where can I learn the theory?

The project includes a **14-chapter technical guide** covering everything from transformer fundamentals to organelle pipeline design. Start at [docs/book/0.md](docs/book/0.md). Key documents:

| Document | What it covers |
|----------|---------------|
| [VISION.md](VISION.md) | The stem cell philosophy |
| [VALUE_PROPOSITION.md](VALUE_PROPOSITION.md) | Why this matters, who benefits |
| [Technical guide](docs/book/0.md) | 14 chapters, from basics to advanced |
| [ORGANELLE_PIPELINE.md](docs/organelles/ORGANELLE_PIPELINE.md) | Pipeline wire format design |
| [TRAINING_STRATEGIES.md](docs/foundation/TRAINING_STRATEGIES.md) | LR scheduling, warmup, capacity scaling |

### Can I use MicroGPT-C as a library in my own project?

Yes. Include `microgpt.h`, link against `microgpt.c`, and you have access to the full API: model creation, training, inference, checkpointing, and tokenisation. See [LIBRARY_GUIDE.md](docs/LIBRARY_GUIDE.md) for a worked example.

---

## ⚖️ Safety & Reliability

### How do you handle "hallucinations"?

In the OPA framework, hallucinations are mitigated by the **Judge** organelle. If a Worker proposes an invalid action, the Judge catches it before it is executed. In our game experiments, this coordination resulted in **zero invalid moves** over thousands of test cycles. The kanban replan loop ensures that rejected moves trigger a new attempt, not a crash.

### Does it solve the lottery?

**No.** We used the lottery as a "negative control" experiment. The engine hit a hard entropy floor at ~0.50 loss, proving that it distinguishes between learnable patterns (market regimes, logic games) and true randomness (lottery draws). If it *claimed* to solve the lottery, it would be hallucinating; the fact that it **fails** proves the engine's integrity.

### Can the models learn from random data?

No — and that's by design. The lottery experiment confirms that OPA pipelines **cannot extract signal from noise**. This is a feature, not a bug. It means that when a model *does* learn (e.g., market regime detection at 57% accuracy), you can trust that real patterns exist in the data.

---

## 🤝 Contributing & Community

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Key areas where contributions are welcome:

- **New game experiments** — add your favourite logic puzzle using the organelle library
- **Performance optimisation** — SIMD, cache tuning, platform-specific acceleration
- **Documentation** — tutorials, examples, translations
- **Testing** — expanding the 97 unit tests and 22 benchmarks

### Who built this?

This project was built transparently with human–AI collaboration:

| Role | Member |
|------|--------|
| 🧭 Principal Research Manager | **Ajay Soni** — research direction, validation, decisions |
| 💻 Engineering & Documentation | **Claude** — coding, documentation, junior research |
| 🔬 Senior Research Assistant | **Grok** — in-depth analysis and insights |
| 🎨 Senior Research Assistant | **Gemini** — creative synthesis and validation |
| 📚 Community Education | **NotebookLM** — accessible explanations |

### What license is it under?

**MIT** — see [LICENSE](LICENSE). Training data and pretrained checkpoints have separate licensing documented in [DATA_LICENSE.md](DATA_LICENSE.md).
