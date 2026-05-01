# BRD — MicroGPT-C Business Requirements Document

**Document ID:** MGC-BRD-001
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30
**Authors:** Extracted from `VISION.md`, `VALUE_PROPOSITION.md`, `ROADMAP.md`, `docs/STRATEGY_ONE_PAGER.md`, `docs/PRODUCT_*.md`.
**Owner:** Ajay Soni, Enjector Software Ltd.

---

## 1. Purpose

This document captures **why** MicroGPT-C exists, **who** it serves, and **what business outcomes it must enable**. It is the entry point for executive readers; engineers should read `FRD.md` and `NFRD.md` after this for the operational consequences.

The voice is descriptive. Each distinct business requirement carries a stable `BREQ-NNN` ID for cross-referencing in the rest of the corpus.

## 2. Problem statement

The dominant trajectory of large language models requires cloud infrastructure, multi-billion-parameter models, opaque inference paths, and per-token billing. A meaningful set of decision-making problems cannot live in that posture for one or more of these reasons:

- **Regulatory audit demands explainability** — fraud, finance, defence, healthcare, regulated payments cannot accept opaque outputs.
- **Edge constraints** — embedded devices, IoT, vehicle / vessel / dismounted-infantry platforms have no network connectivity, ≤ 10 MB RAM budgets, or ≤ 5 ms inference latency requirements.
- **Privacy regulations** — GDPR, HIPAA, on-prem-only contractual clauses prohibit the data leaving the device.
- **Cost economics** — at high request volume, per-token billing for an LLM dominates the unit economics; a CPU-resident specialist is effectively free at the margin.
- **Composition and verification** — a domain rule is naturally expressed as a typed DAG, not as a free-text generator. The verifier-as-Judge is the audit surface.

MicroGPT-C is a research-validated platform that is **purpose-built for the verticals where the dominant trajectory does not work**, while remaining honest about where it loses to the dominant trajectory (open-domain question answering, free-text generation across diverse domains).

## 3. Vision

> Intelligence does not need to be big. It needs to be focused. A tiny model trained on one task outperforms a giant model distracted by everything.

Source: `VISION.md` "Spear Summary". The metaphor is biological: a stem cell becomes specialised by encountering its environment (a corpus + a training run), producing a small, high-confidence specialist organelle. Multiple organelles compose into pipelines that solve problems no individual organelle can solve.

## 4. Target audiences

| Audience | What they get | What they pay |
|---|---|---|
| Embedded / IoT engineers | Pure-C99 model that compiles anywhere with `cc`; no Python; no GC; no cloud | None at the margin: zero deps, MIT licence |
| Product owners and startups | Ship AI features without GPU bills, vendor lock-in, or per-token billing | One-time training cost (CPU minutes); ongoing ≈ zero |
| Researchers and students | ~3,600 lines of readable C implementing a full GPT with backprop; reproducible (seeded RNG, deterministic training) | None |
| AI/ML engineers integrating into existing C/C++ apps | Two-file core engine; direct C API — no REST/gRPC server layer | Build-time integration |
| Regulated-industry product teams | Auditable typed pipeline IR, deterministic Judge, on-device deployment, audit-trail primitives | Vertical-specific corpus curation |

## 5. Business requirements

### 5.1 Strategic positioning

| ID | Requirement |
|---|---|
| BREQ-001 | The platform SHALL run entirely on-device with no cloud calls, no telemetry, and no data collection by default. |
| BREQ-002 | The platform SHALL be MIT-licensed in its open-source form so that downstream commercial use is unrestricted. |
| BREQ-003 | The core engine SHALL be portable to any platform with a C99 compiler and a C standard library (`libc`/`libm`), without modification. |
| BREQ-004 | The platform SHALL be small enough to deploy on commodity hardware: a 540K-parameter model SHALL fit in < 5 MB binary. |
| BREQ-005 | The platform SHALL produce **deterministic** output for a given seed: re-running training with the same RNG seed and corpus produces identical model weights to within scalar precision. |
| BREQ-006 | The platform SHALL provide an **explainable** output path: every "decision" made by an organelle pipeline SHALL be traceable to (a) the input prompt, (b) the organelle that produced each intermediate output, (c) the deterministic scaffolding (Kanban / cycle detector / verifier) that filtered the output. |

### 5.2 Demonstrated capability — "what was proven, not promised"

The platform's claim of usefulness rests on a documented, reproducible evidence set. The following are normative business requirements: a clean-room rebuild SHALL be able to reproduce these capabilities and outcomes from this corpus.

| ID | Capability | Evidence threshold |
|---|---|---|
| BREQ-010 | Train a character-level Shakespeare model that emits zero `<unk>` tokens. | 841K-param model, 14 min training (12 threads), ≥ 28K tok/s training, ≥ 16K tok/s inference. |
| BREQ-011 | Train a word-level Shakespeare model that achieves ≥ 2.5× the inference throughput of the character model. | 510K-param model, ~40K tok/s inference. |
| BREQ-012 | Solve 8-puzzle with a multi-organelle pipeline at ≥ 90% solve rate (100 evaluation episodes). | 5 organelles × 460K params; zero invalid moves. |
| BREQ-013 | Win Connect-4 against a random opponent at ≥ 88% (100 evaluation games). | 2 organelles × 460K params. |
| BREQ-014 | Provide a **negative control**: a lottery prediction experiment SHALL fail to learn (entropy floor at ≈ 0.50 loss) on random data. | 2 organelles × 163K params; demonstrates the engine does not hallucinate signal where none exists. |
| BREQ-015 | Achieve ≥ 75% retrieval accuracy on the held-out wiring NL→graph corpus, **after** the leakage-audited honest restatement: anchor-retrieval mechanism = 100% on the 20-prompt clean Phase 2c paraphrases; wiring transformer alone = 35% on the same clean set. | See `docs/research/RESEARCH_PIPELINE_IR.md` §38 (leakage audit) and §41 (Phase 3a falsification). |
| BREQ-016 | Provide compose-from-fragments capability: given a multi-stage NL prompt, the platform SHALL achieve ≥ 60% on a 10-prompt test set. | Phase 3b composition — `RESEARCH_PIPELINE_IR.md` §43. |
| BREQ-017 | After expanding the wiring corpus to ≥ 4,000 examples, a TF-IDF centroid classifier SHALL achieve ≥ 90% on the adversarial axis-2 stress test. | Phase 4 — `RESEARCH_PIPELINE_IR.md` §46. |
| BREQ-018 | Provide a **Memory Sparse Attention (MSA)** path that supports unbounded sequence length via O(1) chunked latent storage with cosine top-K routing. | `RESEARCH_MSA.md`; demos `msa_*`. |
| BREQ-019 | Provide a **TurboQuant** 4-bit dual-state quantiser (MSE codebook + 1-bit QJL residual) that achieves ≥ 8× memory reduction on KV-cache storage with no accuracy regression on the demos that integrate it, and ≥ 1.3 M encodes/s. | `RESEARCH_TURBO_QUANT.md`. |

### 5.3 Productisation strategy

The platform's productisation thesis (per `docs/STRATEGY_ONE_PAGER.md`) identifies three verticals in priority order. These are *strategic* business requirements: each names what the platform must support to ship the corresponding product.

| ID | Vertical | Time-to-revenue | Capability the platform must add |
|---|---|---|---|
| BREQ-020 | **Fraud detection** (mid-tier payment processors, neobanks) | 90-day customer pilot, ~3-month sales cycle | Deterministic typed-DAG verifier (already shipped); streaming ingestion adapter (Kafka or equivalent); compliance audit log surface |
| BREQ-021 | **Finance market regime / risk** (mid-tier asset managers, hedge funds) | 6-month prototype, 12–24-month sales cycle | Probabilistic verifier outputs; time-series primitive library (rolling-stats, change-point, EWMA) |
| BREQ-022 | **Defence digital-twin object tracking** (defence prime partner-led) | 12–18 months | Sensor adapters (vision, RADAR, ADS-B, AIS); multi-object tracking primitives; security accreditation path |

These are stated for completeness; per the strategy one-pager the project MAY accept fraud first and defer the others.

### 5.4 Dependency boundary

| ID | Requirement |
|---|---|
| BREQ-030 | The **core engine** (`microgpt.{h,c}`) SHALL remain pure C99, libc + libm only. |
| BREQ-031 | Platform accelerators (Metal, BLAS, INT8, MSA, TurboQuant, RotorQuant, paged KV, attention residuals, DeepSeek-V4 ports) MAY add link-time dependencies (Apple frameworks, BLAS implementations) but MUST be opt-in via CMake flags and MUST default OFF when in conflict with the zero-deps baseline. |
| BREQ-032 | A **dependency policy** (`docs/DEPENDENCY_POLICY.md`) SHALL govern the addition of new dependencies, with three categories — Allowed (Category A), Conditionally-allowed (Category B), Forbidden (Category C). The categories are listed in §5.4 of that document. |
| BREQ-033 | **Cloud-only ML APIs** (OpenAI, Anthropic, Cohere, etc.) are FORBIDDEN as dependencies of the core engine; they may only enter as customer-explicit opt-in on a per-deployment basis. |

### 5.5 Responsible-use posture

| ID | Requirement |
|---|---|
| BREQ-040 | The platform SHALL emit no telemetry by default. |
| BREQ-041 | All training data SHALL remain on-device by default. The user explicitly chooses the corpus path. |
| BREQ-042 | Documentation SHALL acknowledge that small models trained on narrow corpora inherit corpus biases, and SHALL recommend deterministic-Judge filtering or human review for safety-critical applications. |

## 6. Constraints (business)

| ID | Constraint |
|---|---|
| BCON-001 | The author / IP holder is Ajay Soni / Enjector Software Ltd. (per source-file copyright). |
| BCON-002 | The platform SHALL be released under the MIT licence (see `LICENSE`); training-data licensing is documented separately in `DATA_LICENSE.md`. |
| BCON-003 | The platform SHALL NOT be marketed as higher-accuracy than the latest LLM on open-domain tasks; the strategy explicitly concedes that ground (per `STRATEGY_ONE_PAGER.md` §"What we are not claiming"). |

## 7. Out of scope

The following are explicitly out of scope for this corpus's V1.0:

- Federated training across multiple devices.
- Model distillation pipeline from a cloud teacher.
- Bidirectional / encoder-only architectures (the platform is autoregressive only).
- Real-time fine-tuning beyond the on-device incremental-learning caveat (catastrophic forgetting is documented and not solved).
- A general-purpose tokeniser beyond character / word level (BPE / SentencePiece are recommended future work in `VISION.md` §7).

## 8. Glossary (business-side)

| Term | Meaning |
|---|---|
| Organelle | One trained MicroGPT-C model bundled with its vocabulary and training docs (`Organelle` struct in `microgpt_organelle.h`). |
| OPA / Adaptive Organelle Planner | The deterministic C scaffolding (Kanban + cycle detector + valid-move filter + ensemble vote + judge) that orchestrates organelles into pipelines. |
| Stem cell | Metaphor for an untrained MicroGPT-C model (a "blank" architecture) that becomes specialised by training on a particular corpus. |
| Pipeline IR | The typed-graph representation of a computation; the artefact a Wiring Organelle emits and a verifier checks. |
| Anchor / anchor library | A curated set of canonical pipeline graphs indexed by domain keywords, used at retrieval time as a high-quality fallback to autoregressive generation. |
| Discretisation Wall | The empirically observed barrier between categorical reasoning (where small organelles win) and continuous-valued prediction (where they lose). |
| Negative control | The lottery experiment that demonstrates the platform does not learn structure from random data. |

## 9. Cross-references

- `FRD.md` — functional surface that satisfies these business requirements.
- `NFRD.md` — non-functional qualities (performance SLOs, portability, reproducibility).
- `BS_*.md` — per-subsystem behaviour specs.
- `docs/STRATEGY_ONE_PAGER.md` — the one-page version of the strategic story.
- `docs/PRODUCTIZATION_VERTICALS.md` — the longer-form productisation analysis.
- `docs/DEPENDENCY_POLICY.md` — dependency boundary policy.
- `book/MicroGPT-C_Composable_Intelligence_at_the_Edge.md` — the project book.

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
