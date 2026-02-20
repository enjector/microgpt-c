# Why MicroGPT-C Matters

> A zero-dependency C99 GPT engine that runs anywhere — from data centres to Raspberry Pis.

---

## The Problem

Modern AI requires:
- **Cloud infrastructure** — GPU clusters, API keys, network connectivity
- **Massive scale** — billions of parameters, terabytes of training data
- **Complex dependencies** — Python ecosystems, CUDA, framework lock-in

This creates a hard floor: if you can't afford the infrastructure, you can't use AI. If you need offline operation, real-time inference, or edge deployment — you're stuck.

## The Proposition

MicroGPT-C proves that **useful AI can be small, fast, and self-contained**.

A single C99 file pair (`microgpt.h` + `microgpt.c`) implements:
- Complete GPT-2 architecture with multi-head attention
- Training with Adam optimiser, cosine LR scheduling, gradient accumulation
- Inference with KV caching at ~16,000 tokens/second
- Character-level and word-level tokenisation
- Model serialisation (save/load checkpoints)
- Optional quantisation (INT8 — 4× memory reduction)

**No dependencies. No frameworks. No cloud. Just C99.**

---

## Who Benefits

### Embedded & IoT Engineers

| Challenge | MicroGPT-C Solution |
|-----------|---------------------|
| No Python runtime available | Pure C99, compiles with any C compiler |
| Memory constrained (< 10 MB) | Sub-1M parameter models fit in < 4 MB |
| No network connectivity | Train and infer entirely on-device |
| Real-time requirements | ~16,000 tok/s inference, no GC pauses |

**Use case:** On-device text prediction, sensor anomaly narration, command parsing.

### Product Owners & Startups

| Challenge | MicroGPT-C Solution |
|-----------|---------------------|
| GPU costs dominate runway | Runs on $5 hardware (Raspberry Pi) |
| Vendor lock-in (OpenAI, Anthropic) | Self-hosted, MIT-licensed, no API keys |
| Privacy regulations (GDPR, HIPAA) | All data stays on-device |
| Latency-sensitive applications | Sub-millisecond inference per token |

**Use case:** Privacy-first AI features, offline-capable products, edge intelligence.

### Researchers & Students

| Challenge | MicroGPT-C Solution |
|-----------|---------------------|
| Black-box frameworks | ~2,700 lines of readable C |
| Reproducibility concerns | Seeded RNG, deterministic training |
| Can't afford GPU time | Train a model in < 60 seconds on CPU |
| Hard to modify architectures | Compile-time macros for everything |

**Use case:** Learning transformers from first principles, rapid architecture experiments.

### AI/ML Engineers

| Challenge | MicroGPT-C Solution |
|-----------|---------------------|
| Deployment complexity | Single binary, no container needed |
| Cross-platform concerns | Builds on Linux, macOS, Windows, ARM |
| Model serving overhead | Direct C API, no REST/gRPC layer needed |
| Integration with existing C/C++ | Header include, link, done |

**Use case:** Embedding AI into existing C/C++ applications, edge model serving.

---

## The Organelle Advantage

MicroGPT-C's **organelle architecture** takes this further. Instead of one large model, you compose specialised micro-models into pipelines:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Planner  │────▶│  Mover   │────▶│  Judge   │
│(30K-160K)│     │(30K-160K)│     │(30K-160K)│
└──────────┘     └──────────┘     └──────────┘
     │                │                │
     ▼                ▼                ▼
  "What to do"    "How to do it"  "Did it work?"
```

**Total: 60K–480K parameters per pipeline. Right-sized by corpus complexity. Zero invalid moves.**

This is proven in fourteen experiments:
- **C Code Composition** — **83% exact match** on function plans, 98% parse rate (1.2M params with LR scheduling)
- **8-Puzzle solver** — **90% solve rate** with 5 organelles (100% easy, 100% medium, 70% hard) and zero parse errors
- **Tic-Tac-Toe** — **87% win+draw** rate with 2 organelles, zero invalid moves
- **Connect-4** — **88% win rate** with 2 organelles, zero invalid moves
- **C Code Generation** — byte-perfect function recall
- **C Wiring Generation** — multi-function composition
- **8 additional games** — Pentago (**91% win**, 92K), Mastermind (**79% solve**, 92K), Sudoku (**78%**, 160K), Othello (**67% win**, 92K), Klotski (**62%**, 30K), Red Donkey (**12%**, 30K), Lights Out (**10%**, 160K), Hex (**4%**, 92K) — right-sized parameters per game

---

## Competitive Positioning

| Feature | MicroGPT-C | llama.cpp | ONNX Runtime | TensorFlow Lite |
|---------|-----------|-----------|-------------|----------------|
| Dependencies | **Zero** | cmake + stdlib | protobuf + stdlib | flatbuffers + stdlib |
| Language | **C99** | C++ | C++ | C++ |
| Training on-device | **Yes** | No | No | Limited |
| Smallest binary | **~50 KB** | ~2 MB | ~5 MB | ~1 MB |
| Compiles everywhere | **Yes** | Most platforms | Most platforms | Most platforms |
| Composable models | **Yes (organelles)** | No | No | No |
| MIT License | **Yes** | Yes | MIT | Apache 2.0 |

MicroGPT-C is not competing with large language models. It occupies a different niche entirely: **the space below 1M parameters where simplicity, portability, and composability matter more than scale.**

---

## The Bottom Line

MicroGPT-C makes AI accessible to anyone with a C compiler.

- **For products:** Ship AI features without cloud costs or vendor lock-in
- **For research:** Understand transformers by reading 2,700 lines of C
- **For edge:** Deploy intelligent agents on hardware that can't run Python
- **For education:** Train a GPT from scratch in under a minute

The organelle architecture shows that **small, specialised models composed into pipelines can solve problems that individual models cannot** — at a fraction of the parameter cost. From game-playing to autonomous code composition, the evidence spans 14 experiments across 11 game domains and 3 code generation tasks.

This is not a toy. This is a foundation.

---

*See [VISION.md](VISION.md) for the full stem cell philosophy, or dive into `experiments/organelles/` for the evidence.*
