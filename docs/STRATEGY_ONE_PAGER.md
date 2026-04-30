# microGPT-C — strategy one-pager

*For board / investor / customer conversations. Companion to the four detailed sketches in `docs/`. Honest about both edges and gaps.*

## Elevator pitch

**Auditable AI for regulated, edge-deployed, compositional decision problems.** A small, deterministic, on-device alternative to LLMs for verticals where explainability is mandatory and a 540K-parameter specialist beats a 540B-parameter generalist.

## The thesis (in three sentences)

The dominant trajectory of AI is bigger models, more compute, cloud-only, opaque. The verticals that *can't* live there — fraud, finance, defence, anything with regulatory audit, edge constraints, or compositional reasoning — are an under-served market. We've built a research-validated architecture (tiny specialists + typed pipeline + deterministic verifier + on-device deployment) that is *purpose-built* for those verticals.

## What's actually built and proven (not promised)

| Capability | Maturity | Evidence |
|---|---|---|
| Typed pipeline IR with deterministic verifier | Production | 51/51 unit tests; used end-to-end |
| Geodesic state-space metrics | Production | 16/16 unit tests; supports fraud / regime / trajectory |
| Multi-organelle Kanban orchestration | Production | 11 game demos shipped |
| Tiny transformer (~30K–540K params) | Production | Char-level Shakespeare + word-level NL → graph |
| Retrieval over curated library | ~80% on novel paraphrases | Audited, leakage-checked, ceiling honestly documented |
| Edge-deployable footprint | Yes | < 5MB binary, single laptop, < 5ms p99 inference |
| Audit-trail primitives | Built into IR | DOT renderer; pipeline_verify; type-checked DAG |

## Three verticals, ranked by time-to-revenue

### 1. Fraud detection — 90-day customer pilot, ~3-month sales cycle

Cleanest fit. The architecture's strengths (composable typed graphs, deterministic Judge, edge deployment, ≤5ms decisions, analyst-editable rules) line up directly with what mid-tier payment processors and neobanks are spending money to fix. Pitch: comparable accuracy to incumbents (FICO Falcon, NICE Actimize, Featurespace), at a fraction of the cost, with regulator-ready audit trails and rules an analyst can edit without an engineer.

**Investment:** ~3 months, 1-2 engineers. Real risk: dataset access (mitigated by paid POC under NDA).

### 2. Finance market regime / risk detection — 6-month prototype, 12-24-month sales cycle

Plausible second move. Same compositional + auditable + on-prem story, applied to risk reports and regime classification. Requires building probabilistic-verifier extension and time-series primitive library — a real engineering investment but not a research project. Customer profile: mid-tier asset managers, multi-strategy hedge funds, regulators, central counterparties — anyone paying $500k+/yr to MSCI BarraOne / Bloomberg PORT / RiskMetrics for tools they can't extend.

**Investment:** ~5-6 months, 2-3 engineers. Real risk: regulatory approval cycles measured in 6-18 months at every regulated bank.

### 3. Defence digital-twin object tracking — 12-18 months, partner-led

Largest TAM, slowest path. Right architectural fit (compositional, explainable, edge-deployable matches Defence AI Strategy / DoD RAI mandates). Wrong project shape today (no sensor primitives, no multi-object tracking, no security accreditation). Realistic only with a defence prime as a partner from day one. First proof point: civilian AIS (maritime tracking) anomaly demo, then partnership-led port to classified data.

**Investment:** 12-18 months, requires partner. Don't start cold.

## The single gating decision

**Drop "pure C99, zero dependencies" research-project policy** — replace with a thin deliberate dependency boundary documented in `docs/DEPENDENCY_POLICY.md`. Three categories (allowed / conditionally allowed / forbidden) with named libraries in each. Without this, no vertical ships; with it, all three become tractable. Adoption is triggered by the first vertical PR that needs it (probably `librdkafka` for fraud-stream ingestion).

## Recommended sequence

```
Phase 0 (today)        : Strategy + dependency policy approval
Phase 1 (months 1-3)   : Fraud MVP → first paid customer pilot
Phase 2 (months 4-6)   : Cross-cutting (probabilistic verifier, audit log, embeddings)
Phase 3 (months 6-12)  : Finance prototype + first regulatory engagement
Phase 4 (months 12-24) : Defence partner conversations + AIS demo
```

Each phase is independently shippable. Each later phase reuses the earlier phase's infrastructure.

## What we are *not* claiming

- **Not** higher accuracy than the latest LLM on open-domain tasks (we lose, by design)
- **Not** a replacement for incumbent risk / fraud / defence systems (we're a complement that solves the auditability + edge gap)
- **Not** a research breakthrough (the architecture is mostly conventional; the discipline around audit, leakage-checking, and pre-registration is what's distinctive)
- **Not** scalable to 100s of families across all domains without external semantic embeddings (per the Post-Phase-3 ceiling finding — honestly documented, not hidden)

## The honest current state

- Research arc closed; calibrated honest claims documented (`docs/research/wiring_scaling_*.md`)
- 18 commits ahead of `origin/main` containing the full arc + productization sketches
- One known regression in the wiring binary's vote loop (rolled back surgically; real fix documented as future work)
- Three open follow-ups documented but not done (vote-loop fix, external embeddings, independent-curator reproducibility)
- Architecture is proven on toy benchmarks (Shakespeare, 11 games, lottery, Pipeline IR). **Not yet proven on a real customer dataset.** That's exactly what fraud Phase 1 fixes.

## The one ask

**Approve the dependency policy and commit to fraud Phase 1.** Everything else cascades from those two decisions. Without them, the research stays as research and the productization remains aspirational.

## Where to read more

| Want detail on | Read |
|---|---|
| Strategy + cross-vertical view | `docs/PRODUCTIZATION_VERTICALS.md` |
| Fraud MVP plan | `docs/PRODUCT_FRAUD_DETECTION.md` |
| Finance regime / risk plan | `docs/PRODUCT_FINANCE_RISK.md` |
| Defence tracking plan | `docs/PRODUCT_DEFENCE_TRACKING.md` |
| Dependency boundary rules | `docs/DEPENDENCY_POLICY.md` |
| Honest research findings | `docs/research/wiring_scaling_post_phase3.md` |
