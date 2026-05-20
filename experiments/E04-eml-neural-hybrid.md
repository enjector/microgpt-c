# Experiment E04 — End-to-end neural + EML hybrid on a public physics dataset

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** demonstrate the *complementary organelle classes* story (neural for pattern matching, EML for shallow-elementary symbolic) as a worked end-to-end neuro-symbolic system on public data.
**Cost estimate:** ~4-5 weeks (1 wk dataset + 1 wk regime classifier + 1 wk EML organelle library + 1 wk hybrid pipeline + 1 wk measurement).
**Falsification risk:** Medium — depends on whether the chosen target really is a shallow elementary form and whether the regime classifier hits its accuracy floor.

---

## Spear summary

**Point:** OPA today has two organelle classes that exist in separate doc paragraphs: neural (the entire 11-game demo set, the wiring organelle) and EML (the depth-2 noise-robust demo from [`RESEARCH_EML_ORGANELLE.md`](../docs/research/RESEARCH_EML_ORGANELLE.md)). No demo exercises both in one pipeline. This experiment closes that gap.

**Picture:** Pendulum data (or projectile, or Beer-Lambert — pick one in §2.3). Noisy input. A small neural **regime classifier** identifies which physical regime the data sits in (small-angle / large-angle / damped). An **EML organelle** evaluates the closed-form prediction for that regime. The Pipeline IR verifies the type signature and sanity-checks the prediction against bounded physical limits.

**Proof (to be measured):** the hybrid pipeline extrapolates ≥ 50× further outside the training range than a pure neural baseline, with ≥ 99% in-domain accuracy. The audit trail decodes back to a paste-able sympy expression — what the field calls "interpretable AI by construction."

**Push:** A worked end-to-end demo where neural + symbolic genuinely cooperate is rarer than the neuro-symbolic literature implies. The repo already has the components — assembling them is the experiment.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

On a chosen public physical-law dataset (locked in §2.3), the following holds:

> *A hybrid pipeline — `noisy_input → neural_regime_classifier → EML_organelle_per_regime → IR_verifier → prediction` — achieves ≥ 99% accuracy on in-domain test data, extrapolates ≥ 50× further (in the relevant physical variable) than a pure neural baseline of equivalent total parameter count, and produces an audit trail that decodes to a human-readable sympy expression for each prediction.*

### 1.2 Why this matters

The "complementary organelle classes" claim is in the project's narrative ([`RESEARCH_EML_ORGANELLE.md`](../docs/research/RESEARCH_EML_ORGANELLE.md) §"What it's good for") but has no worked demonstration where both classes actually contribute. The two failure modes today:

- **Pure-neural baselines** memorise the training range; extrapolation degrades catastrophically (LinReg max error 1.28 vs EML 2.4e-7 in the existing EML demo).
- **Pure-EML** can't pattern-match which closed-form applies; it needs a regime classifier to dispatch correctly.

The neuro-symbolic literature mostly demonstrates concept-symbol grounding (e.g. CLIP-like attribute extraction). Closed-form extrapolation under regime dispatch is a distinct and under-studied capability. The architecture has both pieces; the experiment proves they compose.

### 1.3 Mechanism

**Phase 1 — Dataset selection and curation (1 week).** Three candidate domains (pick one in §2.3):

| Candidate | Physical law | Regimes | Public data |
|---|---|---|---|
| **Pendulum period** | T = 2π√(L/g) (small angle); T = 4√(L/g)·K(sin(θ/2)) (large angle) | small-angle, large-angle | Synthetic + replicated lab data on Zenodo |
| **Projectile range** | R = (v²sin(2θ))/g (no drag); R ≈ (v cos θ)/k (high drag) | drag-free, low-drag, high-drag | Synthetic + sports-physics datasets |
| **Beer-Lambert absorbance** | A = εcl (linear); A = εcl + βc² (concentrated, deviation) | dilute, concentrated | UCI spectroscopy datasets |

Add Gaussian noise σ ∈ {0.001, 0.01, 0.05, 0.1} to the input variable (matching the EML repo's tested noise levels).

**Phase 2 — Neural regime classifier (1 week).** Train a tiny (~30K-param) transformer on (noisy_input → regime_label) pairs. Output is a one-hot over the regime set. Use the existing `microgpt.{h,c}` engine — no new model code.

**Phase 3 — EML organelle library (1 week).** Train one EML organelle per regime using the offline trainer in the EML research repo. Export each as a frozen `c_<regime>_eml_tree.h` header. The runtime evaluator (`microgpt_eml_eval()`, ~150 LOC) already exists.

**Phase 4 — Hybrid pipeline (1 week).** Wire into `demos/character-level/eml_neural_hybrid/`:

```
                                   Pipeline IR @graph
                                   ─────────────────────────────
                            ┌───▶ [classifier:neural] ───┐
                            │                             ▼
   [noisy_input:float] ─────┤                     [regime:enum]
                            │                             ▼
                            │       ┌──── [eml_small_angle:eml] ────┐
                            │       │                                │
                            └───────┼──── [eml_large_angle:eml] ────┼──▶ [prediction:float]
                                    │                                │
                                    └──── [eml_damped:eml] ─────────┘
                                                  ▼
                                          [verifier:judge]
                                                  ▼
                                          [output:float + audit_trail]
```

The pipeline is expressed as an `@graph...@end` IR document; the verifier checks (a) type-flow validity, (b) regime-classifier output is one-hot, (c) EML organelle output is within physical bounds (e.g. pendulum period ≥ 0, ≤ 100s for plausible Earth-surface lengths).

**Phase 5 — Pure-neural baseline (concurrent).** Train a single neural transformer of equivalent total parameter count (regime classifier params + sum of EML organelle bytes converted equivalently — call it ~100K params total budget) on (noisy_input → prediction) directly. This is the comparison baseline.

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | Hybrid in-domain accuracy ≥ 99% (test MSE ≈ σ² of noise) | < 95% |
| **T2** | Hybrid extrapolation accuracy ≥ 99% on 50× wider input range | < 90% |
| **T3** | Pure-neural in-domain accuracy ≥ 95% (it should learn the training range fine) | < 80% (= baseline is broken; not a fair comparison) |
| **T4** | Pure-neural extrapolation accuracy on 50× wider input range — *prediction* < 50% | ≥ 90% (= neural baseline extrapolates; weakens the story) |
| **T5** | Regime classifier accuracy ≥ 95% on the regime-label task | < 90% (= classifier is the bottleneck; reduce regime granularity) |
| **T6** | Audit trail: every hybrid prediction decodes to a paste-able sympy expression matching the underlying physical law | < 100% (= audit-trail bug) |
| **T7** | End-to-end inference latency ≤ 1 ms p99 on M2 Max | > 10 ms |

The headline survives if **T1, T2, T6 pass** and **T4 confirms < 50%** (i.e. the contrast is real). If T4 falsifies (neural baseline extrapolates), the story narrows to "audit-trail and latency only" rather than "extrapolation also."

### 1.5 Skip rules

- If T5 falls below 90% (classifier bottleneck): reduce regime count to 2 (e.g. drop one of small-angle/large-angle/damped) and re-run.
- If T1 falls below 95% in-domain: the chosen physical law isn't a shallow elementary form OR the EML training failed to find it. Switch dataset (one of the §2.3 alternates) or abandon and document.
- If the audit trail (T6) doesn't decode for any prediction: that's a fundamental EML organelle bug, not an experiment failure — fix in the EML repo and re-run.

### 1.6 Falsification risk: Medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| Chosen physical law not actually depth ≤ 4 in EML | Medium | Pre-verify with EML repo's compiler before committing the dataset |
| Regime classifier bottleneck (T5 < 90%) | Medium | Start with 2 regimes; expand only if accuracy holds |
| Pure-neural baseline overfit & extrapolates well anyway (T4 fails high) | Low for genuinely-physical laws; medium for low-noise datasets | Test at σ = 0.05 minimum |
| Audit-trail format change | Low | The format is frozen in `microgpt_eml.h` |
| Composition with the Pipeline IR introduces hidden coupling | Low | Same IR that the wiring binary uses; well-tested |

### 1.7 What this experiment is NOT testing

- It is **not** testing whether EML scales to deeper closed forms. EML depth ≥ 5 has low recovery rates per [`RESEARCH_EML_ORGANELLE.md`](../docs/research/RESEARCH_EML_ORGANELLE.md); this experiment stays at depth ≤ 4.
- It is **not** testing whether OPA can do general-purpose physics simulation. The architecture handles closed-form lookup + regime dispatch; numerical integration of ODEs is out of scope.
- It is **not** competitive with PINNs or SciML libraries. The contribution is the architectural composition (neural classifier + frozen symbolic + IR verifier all in one auditable pipeline), not state-of-the-art physics.
- It is **not** trying to discover a new physical law. The laws are pre-known; the EML organelle compresses the *recovery* into a tiny deterministic artefact.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| EML organelle today | [`docs/research/RESEARCH_EML_ORGANELLE.md`](../docs/research/RESEARCH_EML_ORGANELLE.md) |
| EML runtime | [`src/microgpt_eml.{h,c}`](../src/microgpt_eml.c) |
| EML training (offline) | `~/dev/research/eml/experiments/` (companion repo) |
| EML demo | [`demos/character-level/eml_organelle/`](../demos/character-level/eml_organelle/) |
| Pipeline IR substrate | [`src/microgpt_pipeline.{h,c}`](../src/microgpt_pipeline.c) |
| "Two complementary classes" framing | [`README.md`](README.md) §"Five genuine research contributions" #3 |

---

## 2. Initial state

### 2.1 What's currently known

- EML organelle at depth 2 recovers the target function exactly (test MSE 1.7e-15 ≈ float32 epsilon squared, extrapolation max error 2.4e-7).
- LinReg-quadratic baseline extrapolation max error: 1.28 (≈ 5 orders of magnitude worse).
- Neural regime classifiers in the OPA games hit > 90% on labelled-state tasks.
- Pipeline IR verifier handles typed-DAG composition at 100% (51/51 tests).

### 2.2 Baselines to beat

| Baseline | Number to beat |
|---|---|
| Pure-neural same-budget baseline in-domain | ≈ 95% (assumed reasonable) |
| Pure-neural same-budget baseline extrapolation | **target < 50%** — this is the differentiator |
| EML-alone (no regime classifier; assume single regime) | Already 100% in-domain on its trained regime; fails on out-of-regime inputs |

### 2.3 Dataset choice — TO BE LOCKED before measurement commit

Recommended: **pendulum period** (T as a function of L and amplitude θ).
- Small-angle regime: T = 2π√(L/g), depth-2 in EML compiled form ✅
- Large-angle regime: T = 4√(L/g)·K(sin(θ/2)) — elliptic integral, depth > 4 ❌ (would need approximation as a depth-≤-4 series expansion; this is a real design call to lock in pre-reg).
- Alternative: dispatch *between* two depth-≤-4 approximations of the elliptic integral, leaving the regime classifier to choose which polynomial branch.

Lock the choice in §2.3 of this doc in the pre-reg commit. **Recommendation: pendulum with two regime approximations.** Fallback: projectile range (drag-free is depth 2, low-drag is depth 4 via series expansion). Beer-Lambert is the cleanest mathematically but the experiment is more interesting with mechanical motion.

### 2.4 Dependencies / blockers

- EML training repo (`~/dev/research/eml/`) — present and working.
- `tools/eml_export.py` — present in this repo.
- Pendulum public dataset — to be sourced (Zenodo or generate synthetically).
- New demo dir `demos/character-level/eml_neural_hybrid/` — to be created.
- `add_demo()` integration in `CMakeLists.txt`.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Dataset locked (incl. noise levels, train/test split, extrapolation range)
- 3.2 Regime classifier training: corpus stats, final accuracy
- 3.3 EML organelles trained: one per regime, recovery success per seed, snap MSE
- 3.4 Hybrid pipeline: `@graph` definition, integration commit
- 3.5 Pure-neural baseline: training, final in-domain, final extrapolation
- 3.6 Audit-trail examples: 3-5 sample predictions with decoded sympy
- 3.7 Latency numbers

---

## 4. Conclusion

**TODO** — fill on measurement commit. Sections to populate:

- 4.1 Verdict per T1-T7 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Headline outcome — extrapolation contrast survives?
- 4.3 Lessons (especially: which design choices on the regime boundary were the load-bearing ones?)
- 4.4 Next moves: (a) extend to a second physical domain to test the architectural-composition story isn't pendulum-specific; (b) feed into [E06](E06-medical-guideline-graphs.md) as the template for "neural dispatch + symbolic prediction" in clinical pathways
- 4.5 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_EML_ORGANELLE.md`)
