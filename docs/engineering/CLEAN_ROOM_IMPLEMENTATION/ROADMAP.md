# ROADMAP — MicroGPT-C platform

**Document ID:** MGC-ROADMAP-001
**Version:** 0.1 — STUB
**Date:** 2026-05-01
**Status:** Internal working document. Not a public commitment until customer-facing.

---

## Purpose

A single living view of where the platform is going and in what order. Dates are *targets*, not commitments. Every roadmap row links to a `GAP-*`, an `ADR-NNN`, or a `BREQ-NNN` so it can be traced back to a real driver.

---

## Phase 0 — Research arc closed (DONE, 2026-04-30)

| Item | Source of truth | Status |
|---|---|---|
| Wiring NL → graph honest baseline | `RESEARCH_DISCLOSURE.md` §3 | DONE — 100 %/35 %/60 %/90 % per `SLO-WIRE-001..004` |
| Compositional generator V1 | `GAP-WIRE-005 PARTIALLY-RESOLVED` | DONE at 30 % correct (`SLO-WIRE-005`) |
| Three-bound consolidation | `RESEARCH_DISCLOSURE.md` §7 | DONE; `INV-WIRE-060/061/062` enforce it |
| V1.0 corpus (BRD/FRD/NFRD/BS/TDD/FS/TRACEABILITY) | this directory | DRAFT — `GAP-RE-001` is the single open corpus gap |

---

## Phase 1 — Fraud vertical pilot (TARGET 2026-Q3)

Trigger: customer LOI from a mid-tier payment processor or neobank per `docs/PRODUCT_FRAUD_DETECTION.md`. Until that LOI lands, this phase remains DEFERRED.

| Item | Source of truth | Trigger / dependency |
|---|---|---|
| Adopt dependency-boundary policy (drop "pure C99 zero-deps") | `GAP-DEP-001 DEFERRED` → ADR-001 (to be written) | First PR that needs `librdkafka` |
| Build 20-family fraud anchor library | `docs/PRODUCT_FRAUD_DETECTION.md` §"Concrete anchor library" | LOI |
| Implement 25 fraud primitives in `wiring_natives_fraud.c` | `docs/PRODUCT_FRAUD_DETECTION.md` §"New primitives needed" | After anchor library |
| Hash-chained audit log (real implementation) | `AUDITLOG_SPEC.md` (currently STUB) → fill in | LOI; specifies crypto library choice |
| Threat model for fraud product line | New `TSM_fraud.md` (deferred per `GAP-RE-002`) | First serious customer security review |
| PCI compliance mapping | New `COMPLIANCE_PCI.md` (deferred per `GAP-RE-003`) | First payment-processor customer |
| Wiring binary vote-loop fix (proper) | `GAP-WIRE-003 OPEN` (path 1-3 in `wiring_binary_phase8_regression.md`) | Surfaces if customer needs new families absorbed live; otherwise fine to defer |

---

## Phase 2 — Cross-cutting investments (TARGET 2026-Q4)

Sequence after Phase 1 ships. Each item independently shippable.

| Item | Source of truth | Dependency |
|---|---|---|
| External pretrained semantic embeddings (break the bag-of-features ceiling) | `GAP-WIRE-002 DEFERRED`, `GAP-WIRE-007 RESOLVED` (model-bound finding) | Dependency policy adopted |
| Probabilistic verifier (`pipeline_verify_with_confidence`) | `docs/PRODUCT_FINANCE_RISK.md` §"Probabilistic verifier" | Required for finance vertical |
| Time-series primitive library (`wiring_natives_finance.c`) | `docs/PRODUCT_FINANCE_RISK.md` §"Time-series primitive library" | Finance prototype |
| Walk-forward backtest harness with lookahead-bias enforcement | `docs/PRODUCT_FINANCE_RISK.md` §"Backtesting harness" | Finance prototype |
| Rebuild test of one BS (`GAP-RE-001` first close) | `METHODOLOGY.md` §7 | Independent reviewer availability |

---

## Phase 3 — Finance vertical prototype (TARGET 2027-Q1)

Trigger: finance partner conversation reaches LOI stage per `docs/PRODUCT_FINANCE_RISK.md`. Sales cycle assumption: 12-24 months from first conversation.

(Roadmap rows added when the phase activates.)

---

## Phase 4 — Defence partner-led prototype (TARGET 2027-Q2 onwards)

Trigger: defence prime / system integrator agreement per `docs/PRODUCT_DEFENCE_TRACKING.md`. Until partner conversation lands, this phase is DEFERRED.

(Roadmap rows added when the phase activates.)

---

## Standing engineering hygiene (continuous)

| Item | Source of truth | Cadence |
|---|---|---|
| Run leakage audit on any new held-out before reporting numbers | `INV-WIRE-062` + `tools/scaling_leakage_audit.sh` | Every new corpus |
| Maintain bag-of-features convergence invariant | `INV-WIRE-060` | Every new feature variant |
| Update `TRACEABILITY.md` revision history per gap-fill / corpus pass | `METHODOLOGY.md` §4 | Every corpus pass |
| Keep `book.7th/Reversible_Engineering.md` and corpus aligned | `METHODOLOGY.md` §1 (in scope) | Whenever methodology evolves |

---

## What this roadmap deliberately does NOT contain

- Specific revenue projections (depends on customer access — no honest forecast possible).
- Hiring plan (none of the above is doable as a single-engineer project past the first prototype).
- Open-source vs commercial licensing call (deferred ADR).
- Specific sensor / market-data / prime partner names beyond what is publicly cited in the vertical sketches.
- Anthropic-Claude-API or other external LLM dependency (would be its own ADR if proposed).

## Cross-references

- `docs/STRATEGY_ONE_PAGER.md` §"Recommended sequence" — the executive view of this roadmap.
- `docs/PRODUCTIZATION_VERTICALS.md` — the strategic reasoning behind the phase ordering.
- `docs/PRODUCT_*.md` — per-vertical detailed plans.
- `TRACEABILITY.md` §3 — the gap register driving each phase trigger.
- `ADR_template.md` — the format for any decision that would change this roadmap.

## Revision history

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-01 | Stub. Initial roadmap aligned with `STRATEGY_ONE_PAGER.md` recommended sequence. Phases 0 (DONE) and 1 (TARGET 2026-Q3) populated; Phase 2-4 are placeholders activated when triggers fire. |
