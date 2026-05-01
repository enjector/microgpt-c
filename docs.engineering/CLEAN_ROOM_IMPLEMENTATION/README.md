# MicroGPT-C — Clean-Room Implementation Corpus

**Document ID:** MGC-CRI-INDEX
**Version:** 1.0
**Date:** 2026-04-30
**Source repository:** `microgpt-c` @ branch `main`, commit `649f9b3` and 17 ahead of `origin/main`
**Status:** initial extraction (Phase 2: Archaeological Extraction per Reversible Engineering)

---

## Purpose

This corpus is the **typed specification** for the MicroGPT-C platform. It is written so that a competent independent team — human or AI — can rebuild the platform from this corpus alone, without reading the source tree, and produce a behaviourally equivalent system. It is the artefact a regulator, customer security team, or auditor reads to evaluate readiness; it is also the artefact a clean-room reimplementation team works against.

The corpus follows the **Reversible Engineering** methodology described in `book.7th/Reversible_Engineering.md` (Chapter 5: artefact taxonomy; Appendix A: templates; Chapter 7: rebuild test).

## Reading order

| Audience | Read in this order |
|---|---|
| Investor / customer / executive | `BRD.md` → `STRATEGY_ONE_PAGER.md` (in `docs/`) → `NFRD.md` §1 |
| Product manager | `BRD.md` → `FRD.md` → `NFRD.md` |
| Reimplementation engineer | `METHODOLOGY.md` → `FRD.md` → all `BS_*.md` → all `FS_*.md` → all `TDD_*.md` |
| Security / compliance reviewer | `METHODOLOGY.md` → `NFRD.md` → `TRACEABILITY.md` |
| Clean-room reviewer (Rebuild Test, Appendix B) | All `BS_*.md` and `FS_*.md` only — **never** the `TDD_*.md` or source |

## Artefact taxonomy

Every document in this corpus carries a typing prefix as defined by the Reversible Engineering chapter on the artefact taxonomy. Mixing types within a single document is forbidden.

| Prefix | Voice | Purpose |
|---|---|---|
| `BRD` | descriptive | Business Requirements Document — why the platform exists |
| `FRD` | descriptive + REQ-IDs | Functional Requirements Document — what the platform must do |
| `NFRD` | descriptive + NFR-IDs/SLO-IDs | Non-Functional Requirements Document — qualities the platform must exhibit |
| `BS_*` | prescriptive (RFC 2119) | Behaviour Spec — what each subsystem promises to its callers |
| `TDD_*` | descriptive | Technical Design Doc — how each subsystem is built |
| `FS_*` | prescriptive (RFC 2119) | Functional / Format Spec — byte-level wire/file formats |
| `TRACEABILITY.md` | structural | Master index linking BRD → FRD → BS → source → tests |
| `METHODOLOGY.md` | normative | The RFC 2119 distillation of the engineering process |

## Subsystems covered

The MicroGPT-C platform is built from the following subsystems. Each gets its own `BS_*.md` + `TDD_*.md` pair. Formats get `FS_*.md`.

| Subsystem | BS | TDD | FS | Sources of truth |
|---|---|---|---|---|
| Core transformer engine (forward, backward, Adam, KV cache) | `BS_core.md` | `TDD_core.md` | — | `src/microgpt.{h,c}` |
| Tokenisation (character + word) | `BS_tokeniser.md` | `TDD_tokeniser.md` | — | `src/microgpt.{h,c}` |
| Checkpoint serialisation | — | — | `FS_checkpoint.md` | `src/microgpt.c` (`checkpoint_save`, `checkpoint_load`) |
| Organelle pipeline (OPA Kanban + cycle detector + ensemble) | `BS_organelle.md` | `TDD_organelle.md` | `FS_organelle_wire.md` | `src/microgpt_organelle.{h,c}` |
| Memory Sparse Attention (MSA) | `BS_msa.md` | `TDD_msa.md` | — | `src/microgpt_msa.{h,c}` |
| TurboQuant / RotorQuant KV compression | `BS_quant.md` | `TDD_quant.md` | — | `src/microgpt_{turbo,rotor}quant.{h,c}` |
| Pipeline IR (typed graph + verifier + DOT) | `BS_pipeline_ir.md` | `TDD_pipeline_ir.md` | `FS_pipeline_ir_text.md` | `src/microgpt_pipeline.{h,c}` |
| Wiring Organelle (NL → graph) + anchor retrieval | `BS_wiring.md` | `TDD_wiring.md` | — | `demos/wiring_organelle/` |
| Geodesic state-space metrics | `BS_geodesic.md` | `TDD_geodesic.md` | — | `src/microgpt_geodesic.{h,c}` |
| Vietoris-Rips persistent cohomology | `BS_vr.md` | `TDD_vr.md` | — | `src/microgpt_vr.{h,c}` |
| EKAN B-spline edge activations | `BS_ekan.md` | `TDD_ekan.md` | — | `src/microgpt_ekan.h`, `src/microgpt_ekan_network.h` |
| Virtual Machine (compiler + runtime) | `BS_vm.md` | `TDD_vm.md` | `FS_vm_bytecode.md` | `src/microgpt_vm.*` |
| Metal GPU bridge | `BS_metal.md` | `TDD_metal.md` | — | `src/microgpt_metal.{h,m,metal}` |
| Build & feature flag system | — | `TDD_build.md` | — | `CMakeLists.txt`, `docs/BUILD_OPTIONS.md` |

## Out of scope for V1.0 of this corpus

Threat models (`TSM_*.md`), per-framework compliance mappings (`COMPLIANCE_*.md`), formal models (`FORMAL_*.md`), and the FMEA catalogue are deliberately deferred to Phase 3 (Adversarial Hardening) per `book.7th/Reversible_Engineering.md` Chapter 6. They are listed as gaps in `TRACEABILITY.md` with disposition `DEFERRED`.

## How to extend this corpus

1. Pick the subsystem.
2. Read the source files listed above.
3. Open the matching `BS_*.md` and write down what the code promises in RFC 2119 voice. Cite line numbers.
4. Open the matching `TDD_*.md` and write down how the code is built. No RFC 2119 voice.
5. Add new `INV-`, `REQ-`, `ERR-`, `SLO-`, `ACC-` IDs to `TRACEABILITY.md` as you create them. IDs are assigned once and never reused.
6. If a behaviour is unclear from the code, mark it as a gap in `TRACEABILITY.md` rather than guessing.

## Companion artefacts

- `book.7th/Reversible_Engineering.md` — the methodology this corpus follows.
- `book/MicroGPT-C_Composable_Intelligence_at_the_Edge.md` — the project's narrative book.
- `docs/research/ORGANELLE_STATE.md` — **start here** for the recommended reading order across all `docs/research/RESEARCH_ORGANELLE_*.md` topical docs and the Wiring Organelle scaling arc (research synthesis, May 2026; calibrated three-bound claim).
- `docs/research/RESEARCH_*.md` — topical research notes and pre-registered experiment outcomes (these are *experimental* artefacts, not normative).
- `docs/research/wiring_scaling_*.md` — the post-Phase-3 cleanup arc (v1 leakage discovery → v2 clean baseline → Phase 2 sharpening → Phase 3 broad expansion + bigram/char-ngram falsification → post-Phase-3 #1-3 cleanup); consolidated as `RESEARCH_DISCLOSURE.md` §7.
- `RESEARCH_DISCLOSURE.md` — pre-registered cancellations and honest headline restatements (regulator-friendly distillation of the research log).
- `docs/STRATEGY_ONE_PAGER.md` — the executive summary of the productisation strategy.
- `docs/PRODUCT_*.md` — vertical sketches (fraud, finance, defence).
- `docs/DEPENDENCY_POLICY.md` — the dependency-boundary policy (gating decision for Phase 1 productisation; tracked as `GAP-DEP-001`).

## Diagram cross-references

The infographics in `docs/research/images/` illustrate concepts in this corpus:

| Diagram | Illustrates |
|---|---|
| `Composable Intelligence Small AI Infographic.jpg` | The four phases (stem cell → differentiation → organelle pipeline → results) — covered by `BRD.md` §3, `BS_organelle.md` §1 |
| `MicroGPT-C Coordination Flow.png` | Planner → Player → Judge inference protocol — `FS_organelle_wire.md` §2 |
| `MicroGPT-C Stem Cell Differentiation Flow.jpg` | `model_create` → `organelle_train` → checkpoint — `BS_core.md` §3.1, `BS_organelle.md` §3.1 |
| `MicroGPT-C Organelle Game Leaderboard.jpg` | BREQ-012, BREQ-013 evidence |
| `Monolith vs. Organelle Comparison Thesis.jpg` | The Neural Algorithmic Reasoning argument — `BRD.md` §3 |
| `OPA_Biology_Analogy.jpg` (in `docs/research/`) | Stem-cell metaphor — `VISION.md` §1 |
| `OPA.png` | OPA Kanban / cycle detector / valid-move filter — `BS_organelle.md` §2.2-2.4 |
| `Scaling AI for the Edge.jpg`, `MicroGPT-C Edge Deployment Vision.jpg` | NFR-020..025 footprint and edge claims |
| `The Kanban State Machine Infographic.jpg` | `OpaKanban` state machine — `BS_organelle.md` §2.2 |

## Status flags used in this corpus

| Status | Meaning |
|---|---|
| `DRAFT` | Document exists; content has not been independently verified against source |
| `REVIEW` | Document is being reviewed against the code; one or more reviewers have read it |
| `APPROVED` | Document has been verified against source by at least one independent reviewer; the rebuild test (Appendix B of the methodology) has been run successfully |

All documents in this V1.0 corpus are `DRAFT` until a Rebuild-Test review pass is run.
