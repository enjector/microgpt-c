# Experiment E06 — Medical guideline → typed treatment graph on public data

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** real-world public-data application that exercises the three structural bounds (curator-, model-, domain-bounded) on an audit-mandated domain where reviewers can verify the work.
**Cost estimate:** ~8-10 weeks (2 wk dataset + 2 wk family library + 2 wk integration + 2 wk clinician review + 1-2 wk measurement & writeup).
**Falsification risk:** Medium-high — clinician evaluation is the hardest gate; if clinicians reject the typed-graph framing as clinically meaningless, the experiment falsifies at the framing level, not at the architecture level.

---

## Spear summary

**Point:** The architecture's strongest combined claim — *audit-trail-native, tiny, composable, distinctive-vocabulary* — has a textbook fit with clinical-pathway formalisation. Public datasets exist (SNOMED CT, UMLS, NICE Clinical Knowledge Summaries, CDC public-health guidelines). The clinical-decision-support audit requirement is severe enough that pipelined symbolic verification is genuinely needed, not just nice-to-have.

**Picture:** Take a public clinical guideline (e.g. NICE CKS "asthma in adults: management" or CDC "hypertension management"). Each guideline is a structured document with assessment → diagnosis → first-line treatment → step-up → monitoring stages. Express each stage as a typed graph in Pipeline IR. Train a wiring organelle on the guideline corpus; emit treatment graphs from a natural-language patient scenario; verify against typed-flow rules + a clinician reviewer.

**Proof (to be measured):** on a held-out set of 50 scenarios drawn from guidelines NOT in the training corpus, OPA produces type-valid graphs that a clinician reviewer rates as ≥ 80% **structurally correct** (right stages in the right order; correct primitive nodes), with 100% audit-trail coverage. The headline is **NOT** "clinically correct" — that's an unsafe claim. The headline is "structurally faithful to the published guideline" — a defensible, testable claim.

**Push:** This is the strongest possible demonstration of the architecture's calibrated value: distinctive vocabulary, compositional structure, audit-mandated downstream — all three bounds favourable, with a public dataset and a real expert reviewer.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

On a public clinical guideline corpus (locked in §2.3) and a held-out set of 50 patient-scenario prompts drawn from guidelines NOT in the training set:

> *OPA's wiring pipeline (anchor retrieval + fragment composition + IR verifier) produces typed treatment graphs that match the published guideline's recommended pathway on ≥ 80% of scenarios as judged by a clinician reviewer using a pre-locked rubric, with 100% audit-trail coverage and ≤ 50 ms p99 latency per scenario on M2 Max.*

**Critical disclaimer (load-bearing):** the headline metric is *structural fidelity to the published guideline*, not *clinical correctness for an individual patient*. This is a research experiment, not a deployable clinical tool. Any subsequent productisation requires the full clinical-AI regulatory path (separate from this experiment).

### 1.2 Why this matters

The migrated productisation verticals — fraud, finance risk, defence tracking — went to the private `organelles.bio` repo on 2026-05-01. The public repo needs a real-world public-data demonstration that:

- Exercises the same three-bound favourable conditions (distinctive vocabulary, compositional structure, audit requirement).
- Uses public data so reviewers can replicate.
- Has a credible expert evaluation path (clinician review).
- Doesn't reveal anything proprietary from the private vertical work.

Clinical guidelines hit all four:
- **Distinctive vocabulary** (✅ favourable for bound 2): "first-line ACE inhibitor" / "step-up ICS" / "monitor eGFR every 12 weeks" — about as distinctive as English gets.
- **Compositional structure** (✅ favourable for bound 1): multi-stage protocols are exactly the multi-stage compositional task Phase 3b shipped 6/10 on.
- **Audit requirement** (✅ favourable for the architecture's distinctive value): clinical-decision-support audit is regulated under MHRA / FDA Software-as-Medical-Device; structural traceability is mandatory.
- **Public data** (✅ favourable for reproducibility): NICE CKS is Crown Copyright, openly licensed; CDC guidelines are public domain; SNOMED CT international edition is free for research.

### 1.3 Mechanism

**Phase 1 — Dataset curation (2 weeks).** Source clinical guidelines from at least two of:

| Source | Licence | Volume |
|---|---|---|
| **NICE Clinical Knowledge Summaries** (UK) | Open Government Licence 3.0 | ~400 guidelines |
| **CDC public-health guidelines** (US) | Public domain | ~200 relevant for primary care |
| **WHO essential medicines guidelines** | Open Access | ~30 condition-specific |
| **AHA / ACC cardiology guidelines** | Conditional licence — research use OK | Selected |

Curate to 20 families (matching the wiring v2 library size) covering: asthma, hypertension, type 2 diabetes, depression, hyperlipidemia, anticoagulation, antibiotic stewardship (5 indications), pain management (3 categories), routine immunisation, contraception, menopause HRT, dyspepsia/GORD.

Each family gets:
- 1-2 **reference graphs** (`@graph...@end`) representing the published pathway.
- 3-5 **anchor prompts** (synthetic patient scenarios drawn from the guideline's worked examples).
- 5-10 **paraphrase prompts** (the same scenarios in different wording, for the corpus expansion pass).

Held-out: 50 patient scenarios drawn from guidelines NOT in the family library, balanced across the 20 conditions.

**Phase 2 — Family library + training (2 weeks).** Reuse the existing wiring organelle pattern. Each family becomes an anchor entry in `wiring_anchor_graphs_clinical.c`; synonyms go into `corpus_expand_clinical.c`. The wiring transformer trains on the expanded corpus.

The Pipeline IR's primitive set extends with clinical-specific natives (`prescribe`, `monitor`, `escalate`, `refer`, `defer`) added to `wiring_natives_clinical.{h,c}`. Each native has a type signature: e.g. `prescribe :: (medication, dose, frequency) → script` and verifier rules enforce flow validity.

**Phase 3 — Integration + audit infrastructure (2 weeks).** Build `demos/wiring_organelle_clinical/`. Wire up:
- Standing leakage audit on the held-out vs training corpus.
- DOT renderer for clinician-readable graph traces.
- A simple HTML report generator: scenario → emitted graph → DOT-rendered SVG → reference-graph side-by-side.

**Phase 4 — Clinician review (2 weeks).** Recruit ≥ 1 clinician reviewer (GP, hospital physician, or specialist) familiar with NICE CKS / equivalent guidelines.

Lock the **review rubric** before any clinician sees output:

| Rubric axis | What "correct" means |
|---|---|
| **Stage ordering** | Assessment before diagnosis; diagnosis before treatment; first-line before step-up |
| **Primitive correctness** | The right medications/interventions for the condition, per the source guideline |
| **Type-flow validity** | All edges connect compatible types (e.g. `medication` flows into `prescribe`, not into `refer`) |
| **Audit-trail readability** | Clinician can read the DOT-rendered graph and identify which guideline section it came from |
| **Safety floor** | NO graph emits a contraindicated combination (e.g. ACE inhibitor + ARB; β-blocker + verapamil) |

Each held-out scenario gets a pass/fail per axis. Headline: % of scenarios passing **all 5 axes**.

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | Clinician-rated structural fidelity ≥ 80% (passing all 5 rubric axes) on 50 held-out scenarios | < 60% |
| **T2** | 100% audit-trail coverage (every emitted graph passes `pipeline_verify()`) | < 99% |
| **T3** | Inference latency ≤ 50 ms p99 per scenario on M2 Max | > 500 ms |
| **T4** | Safety-floor metric: 0 contraindicated combinations across 50 scenarios | ≥ 1 contraindicated combination |
| **T5** | Leakage audit (Jaccard < 0.7) passes on held-out vs training corpus | Any held-out prompt at Jaccard ≥ 0.7 → excluded |
| **T6** | Inter-rater agreement (if ≥ 2 clinicians review): κ ≥ 0.6 on the structural-fidelity axis | κ < 0.4 (= rubric is too subjective; revise and re-rate) |
| **T7** | Compositional multi-stage scenarios (step-up therapy, monitoring escalation) ≥ 60% (matching Phase 3b's 6/10 on multi-stage as the architecture-wide ceiling) | < 40% |

**T4 is hard-floor.** If even one contraindicated combination appears in 50 scenarios, the experiment cannot ship as "structurally faithful" because structural fidelity to the guideline *includes* respecting safety constraints encoded in those guidelines. A T4 failure triggers a fix-and-rerun cycle, not a partial-pass narrative.

### 1.5 Skip rules

- If T1 falls below 60%: the family library or wiring transformer is insufficient for clinical compositional tasks. Investigate which condition families fail most; potentially reduce to a 10-family library and re-run on a smaller held-out (does the failure cluster on specific conditions, or is it broad?).
- If T4 fails (contraindicated combination): immediate fix cycle. Add typed-flow rules that the verifier rejects contraindicated edges. Re-run T4 to confirm. **Do not** ship a result that includes contraindicated outputs.
- If T6 fails low (κ < 0.4): the rubric is too subjective. Revise with concrete decision rules, get a second clinician to re-rate, re-run T1.
- If clinician reviewer unavailable for ≥ 4 weeks: pause the experiment; do not substitute a non-clinician reviewer for the headline metric.

### 1.6 Falsification risk: Medium-high

| Risk | Likelihood | Mitigation |
|---|---|---|
| Clinician finds the typed-graph framing clinically meaningless | Medium | Pre-validate the framing with the reviewer on 5 worked examples before locking the rubric |
| Family library too thin for the breadth of clinical scenarios | High at first attempt | Phase 1 budget assumes iterative library refinement; build for the 20 conditions, not all of medicine |
| Wiring transformer hits the calibrated ceiling and clinician requires > 80% | Medium | Pre-reg target is 80%; expand the architecture (E04 EML-style hybrid? structured retrieval?) only after the baseline measures |
| Held-out scenarios overlap training corpus at Jaccard ≥ 0.7 | Medium | Standing leakage audit runs first; excluded prompts are reported transparently |
| Recruiting a clinician reviewer pro bono / paid | High (logistics) | Allocate budget; alternatively partner with a medical-education institution |
| Regulatory misinterpretation (someone treats this as a clinical tool) | Medium | Bold-print disclaimer on every output; "research, not clinical" header on the demo |

### 1.7 What this experiment is NOT testing

- It is **NOT** testing whether OPA is clinically safe to deploy. That requires the full FDA / MHRA Software-as-Medical-Device regulatory path, far beyond this scope.
- It is **NOT** testing whether OPA replaces clinical decision support systems. Existing CDS systems (NICE BNF, Lexicomp, etc.) are mature and integrated; this experiment demonstrates an *architectural pattern*.
- It is **NOT** measuring patient outcomes. Outcome studies require IRB / ethics approval; out of scope.
- It is **NOT** testing whether the wiring transformer can generalise to unseen drug classes. Held-out scenarios stay within the 20 trained conditions; cross-condition generalisation is a separate experiment.
- It is **NOT** competitive with LLM-based clinical scribing (Hippocratic AI, etc.). Different problem class — those are conversational; this is structured-pathway emission.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| Architecture's distinctive value claim | [`ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) §"The current calibrated claim" |
| Multi-stage composition floor | [`RESEARCH_PIPELINE_IR.md`](../docs/research/RESEARCH_PIPELINE_IR.md) §43 (Phase 3b: 6/10 on multi-stage) |
| Vertical work that went private | [`docs/MIGRATED_TO_ORGANELLES_BIO.md`](../docs/MIGRATED_TO_ORGANELLES_BIO.md) |
| Wiring infrastructure being reused | [`demos/wiring_organelle/main.c`](../demos/wiring_organelle/main.c), [`src/wiring_anchor_graphs.c`](../src/wiring_anchor_graphs.c), [`src/corpus_expand.c`](../tools/corpus_expand.c) |
| Substrate for verifier-as-Judge | [`src/microgpt_pipeline.{h,c}`](../src/microgpt_pipeline.c) |
| Pipeline IR library that consumes the output | [E02](E02-pipeline-ir-library.md) |
| Hybrid neural + symbolic pattern that could extend this | [E04](E04-eml-neural-hybrid.md) |

---

## 2. Initial state

### 2.1 What's currently known

- The architecture hits 6/10 (60%) on Phase 3b multi-stage composition with general-vocabulary prompts — the architectural ceiling for compositional tasks.
- Distinctive-vocabulary tasks (e.g. wiring v2 with clearly-distinct primitives) lift this to the 75-80% calibrated band.
- Clinical vocabulary is **strongly distinctive** by design — guidelines use precise drug names, dosages, specific monitoring intervals.
- No prior public demonstration of OPA on clinical pathways.

### 2.2 Baselines to beat

| Baseline | Number |
|---|---|
| Phase 3b general-vocabulary multi-stage | 60% — pre-reg target T1 must exceed this |
| Wiring v2 single-family distinctive-vocab | 80% — aspirational; clinical task is harder (multi-stage) |
| Pure-LLM clinical pathway emission | Unknown — could be high but lacks audit trail; pair with [E01](E01-llm-head-to-head.md) if budget allows |

### 2.3 Dataset choice — TO BE LOCKED before measurement commit

**Recommendation:** NICE CKS subset (Crown Copyright OGL 3.0, 20 selected conditions, structured XML available via OpenAthens or NHS Digital).

**Backup:** CDC primary-care guidelines subset (public domain) if NICE licensing complications arise.

**Locked vocabulary:** SNOMED CT International Edition (free for research) for medication / condition / procedure identifiers in the typed graphs.

### 2.4 Dependencies / blockers

- **Clinician reviewer.** Hard requirement. Allocate budget (~£2,000-£5,000 GBP for ~40 hours of expert time) or partner with a medical education institution.
- **NICE CKS access.** OGL 3.0 covers research reuse; data extraction via NHS Digital or direct scraping.
- **SNOMED CT licence.** Free for research under the SNOMED International affiliate licence — apply early.
- **Ethical considerations.** Even though this is research, not clinical, document via a brief ethics impact statement; not a full IRB submission but a "we are aware of the misuse risks" note.
- **`tools/scaling_leakage_audit.sh` clinical-corpus adapter.** Audit-B Jaccard already domain-agnostic; should work as-is.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Dataset locked: guideline source, condition list, train/held-out split
- 3.2 Clinical family library statistics
- 3.3 Wiring training: corpus size after expansion, training steps, dev held-out score
- 3.4 Standing leakage audit results
- 3.5 Clinician reviewer profile + time spent + per-rubric-axis ratings
- 3.6 Per-scenario results: 50-row table of (scenario, emitted graph, reference graph, pass/fail per axis)
- 3.7 Headline numbers: T1-T7

---

## 4. Conclusion

**TODO** — fill on measurement commit. Sections to populate:

- 4.1 Verdict per T1-T7 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Headline outcome: distinctive-vocab + compositional + audit-mandated triple holds?
- 4.3 Clinician retrospective: what was easy, what was hard, what would they want next?
- 4.4 Comparison to the architecture's three structural bounds — does clinical favourable-conditions lift to upper-band 80%, or stay at compositional 60-70%?
- 4.5 Next moves: (a) demo paper to clinical-informatics venue (AMIA / JAMIA / Studies in Health Technology and Informatics); (b) consider whether this seeds a public-data vertical in the research repo or stays research-only; (c) cross-reference with [E04](E04-eml-neural-hybrid.md) for dose-calculation symbolic primitives (e.g. eGFR-adjusted dosing as a depth-2 EML expression)
- 4.6 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`)
- 4.7 Hard regulatory disclaimer (re-stated for any external reader landing here directly)
