# MicroGPT-C

[![Build](https://github.com/enjector/microgpt-c/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/enjector/microgpt-c/actions/workflows/cmake-multi-platform.yml)
[![CodeQL](https://github.com/enjector/microgpt-c/actions/workflows/codeql.yml/badge.svg)](https://github.com/enjector/microgpt-c/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### Tiny specialist models, coordinated by a pipeline, outperform monoliths on focused tasks — using 4,000× fewer resources.

![Composable Intelligence — the four phases of MicroGPT-C: stem cell foundation, targeted differentiation, organelle pipeline coordination, and proven results across logic games and code composition](docs/organelles/images/Composable%20Intelligence%20Small%20AI%20Infographic.jpg)

---

## The Story

This project started as a C port of Andrej Karpathy's [microGPT.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a ~200 line Python GPT that trains a character-level Transformer from scratch. We rewrote it in pure C99 with zero dependencies, and it got **1,000× faster**.

Then we asked a bigger question: **can tiny models actually be intelligent?**

Not by making them bigger — the industry already does that. Instead, by making them **work together**. We took the same ~460K parameter engine and trained it on different tasks: one becomes a planner, another becomes a player, another becomes a judge. Each one starts as the same blank "stem cell" and *differentiates* based on its training data.

We call them **organelles** — like the specialised structures inside a biological cell.

The result surprised us. A single organelle playing Connect-4 wins about 55% of the time. But when a planner and player coordinate through a shared protocol, the system hits **90%** — even though the individual models are still wrong half the time. The pipeline catches the mistakes. **The coordination is the intelligence.**

We've now tested this across [11 logic games](docs/organelles/ORGANELLE_GAMES.md), from Tic-Tac-Toe to Sudoku, with models ranging from 30K to 460K parameters. The pattern holds: right-sized specialists working together consistently outperform a single larger model working alone.

Then we asked: **does it work on real-world data?**

We ran two experiments back-to-back — a [lottery prediction](experiments/organelles/lottery/) pipeline (negative control) and a [market regime detection](experiments/organelles/markets/) pipeline (positive test). The lottery model hit an entropy floor at 0.50 loss — it learned nothing, because lottery draws are random. The market model reached 0.03–0.06 loss and **57% accuracy on unseen data** (2.8× the random baseline) — because cross-asset correlations are real, learnable signal.

Same engine. Same architecture. One learns, one can't. **That's the proof.**

---

## Quick Start

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

# Run a multi-organelle game pipeline (90% win rate)
./connect4_demo
```

All 11 game experiments, 2 real-world data experiments (lottery + markets), 3 pretrained checkpoints, 97 unit tests, and 22 benchmarks are included. See the full list in `experiments/organelles/`.

---

## Explore Further

| Topic | Link |
|-------|------|
| ❓ **FAQ** | [FAQ.md](FAQ.md) |
| 🧬 **The stem cell philosophy** | [VISION.md](VISION.md) |
| 💡 **Why this matters** | [VALUE_PROPOSITION.md](VALUE_PROPOSITION.md) |
| 🗺️ **Roadmap** | [ROADMAP.md](ROADMAP.md) |
| 📖 **Technical guide** (14 chapters) | [docs/book/0.md](docs/book/0.md) |
| 🏆 **Game leaderboard** (11 games) | [ORGANELLE_GAMES.md](docs/organelles/ORGANELLE_GAMES.md) |
| 📈 **Market regime detection** (57% holdout) | [markets/README.md](experiments/organelles/markets/README.md) |
| 🎲 **Lottery experiment** (entropy baseline) | [lottery/README.md](experiments/organelles/lottery/README.md) |
| 🔬 **Pipeline architecture** (white paper) | [ORGANELLE_PIPELINE.md](docs/organelles/ORGANELLE_PIPELINE.md) |
| 📚 **Using as a library** | [LIBRARY_GUIDE.md](docs/LIBRARY_GUIDE.md) |
| ⚡ **Performance & benchmarks** | [PERFORMANCE.md](docs/PERFORMANCE.md) |
| 🔧 **Build options** (Metal, BLAS, INT8, SIMD) | [BUILD_OPTIONS.md](docs/BUILD_OPTIONS.md) |
| 🤝 **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 📋 **Data licensing** | [DATA_LICENSE.md](DATA_LICENSE.md) |

---

## Requirements

- **C99 compiler** (GCC, Clang, MSVC)
- **CMake 3.10+**
- No other dependencies

Optional: [Git LFS](https://git-lfs.github.com/) for pretrained checkpoints (`git lfs pull`).

---

## Responsible Use

MicroGPT-C runs entirely on-device with no telemetry, no cloud calls, and no data collection. Small models trained on narrow corpora inherit the biases of that corpus — be aware of this when deploying. High confidence means the model has seen similar patterns, not that the output is correct. Always validate through deterministic checks (the Judge pattern) or human review for safety-critical applications.

See [CONTRIBUTING.md](CONTRIBUTING.md) for ethics guidelines.

---

## Research Team

This project was built transparently with human–AI collaboration — the same philosophy of coordinated intelligence that MicroGPT-C explores.

| Role | Member |
|------|--------|
| 🧭 Principal Research Manager | **Ajay Soni** — research direction, validation, and decisions |
| 💻 Engineering & Documentation | **Claude** — coding, documentation, and junior research |
| 🔬 Senior Research Assistant | **Grok** — in-depth analysis and insights |
| 🎨 Senior Research Assistant | **Gemini** — creative synthesis and validation |
| 📚 Community Education | **NotebookLM** — accessible explanations and education materials |

---

## License

MIT — see [LICENSE](LICENSE).
