# Experiment E03 — Independent-curator reproducibility of the v2 anchor library

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** the architecture's own hardest unrun falsification test, pre-registered in [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) §2.3 and named in [`ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) "What we still don't know" §3.
**Cost estimate:** ~4-6 weeks (1 wk recruit + 3-5 wk independent work + 1 wk measurement).
**Falsification risk:** **High** — and that's the point. If this experiment falsifies, the result is *more* valuable than if it confirms, because it tells the field something genuinely new about curator-vocabulary specificity in bag-of-features retrieval.

---

## Spear summary

**Point:** Every measurement in the wiring-organelle arc so far has had one author (Ajay Soni) writing both the synonym tables AND the held-out paraphrases. The calibrated 75-80 % ceiling could be *curator-vocabulary-specific* rather than *architectural* — a much weaker claim. The only honest way to know is to put the same task in front of an independent curator and measure their numbers against the architecture, not against the original curator's library.

**Picture:** A second curator (different person, no access to the existing anchor/synonym tables) is given the same 20-family v2 specification (intent description + expected primitives + 1-2 reference graphs per family) and rebuilds the library from scratch using their own vocabulary choices. The same evaluation harness — Phase 2c clean held-out + TF-IDF v2 retrieval + Phase 3b composition — judges the result.

**Proof (to be measured):** the independent curator's library scores 14-18/20 on the v2 held-out (within ±5pp of current 16/20) confirms architectural; < 11/20 (more than 5pp below) falsifies the architectural claim and re-frames the result as curator-specific.

**Push:** This is the test no other small-model paper runs. Whichever way it lands, the project's claims become *more* defensible because the measurement was made.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *A second, independent curator (different person, no access to the existing `wiring_anchor_graphs.{h,c}` / `wiring_fragments.{h,c}` / `wiring_geo_classifier.{h,c}` / `corpus_expand.c` synonym tables) rebuilding the v2 20-family library from scratch achieves a TF-IDF retrieval rate within ±5 pp of the current 16/20 on the v2 held-out evaluation.*

In numbers: 14-18 out of 20 on the v2 held-out is the **pre-registered "ceiling is architectural"** band. Outside that band (in either direction) is informative.

### 1.2 Why this matters

The project's headline claim — a calibrated **three-bound ceiling** at 75-80 % on novel-paraphrase retrieval — rests on the assumption that the *architecture*, not the *author*, sets the ceiling. Three confirming variants (unigram TF-IDF, bigram, char-ngram all hit the same band per [`wiring_scaling_post_phase3.md`](../docs/research/wiring_scaling_post_phase3.md)) support the model-bound claim. But every one of those variants was tuned by the same person who chose the held-out vocabulary.

Without an independent-curator measurement, a reasonable reviewer can ask:
- "Are you measuring an architectural property, or are you measuring your own vocabulary intuitions?"
- "Would a different curator hit a different ceiling — higher or lower — with the same architecture?"
- "Is the 'distinctive vocabulary' bound a real constraint on the architecture, or just on this one curator's synonym intuitions?"

The project has no answer today. This experiment is the only honest one.

### 1.3 Mechanism

**Phase 1 — Recruit and brief (1 week).** Find a second person (call them **Curator B**) with the following profile:
- Software-engineering literate (can write C99 / read existing code patterns).
- No prior involvement in MicroGPT-C wiring corpus work.
- Available for 3-5 weeks of focused work.

Brief Curator B with a sealed package:

| What they get | What they don't get |
|---|---|
| The 20-family **task specification** (intent description per family, expected primitive set, 1-2 reference graphs) | Existing `wiring_anchor_graphs.c` contents, synonym tables, geo classifier weights |
| The Pipeline IR grammar reference | Existing held-out paraphrases (these are the test set; they stay sealed) |
| The Pipeline IR test harness | Existing `corpus_expand.c` output |
| Read access to `microgpt_pipeline.{h,c}` (the verifier they ship into) | Read access to `demos/wiring_organelle/main.c` ranking logic |
| 3-5 weeks of paid time | The pre-registered targets in §1.4 (so they can't optimise for the ceiling) |

**Phase 2 — Independent build (3-5 weeks).** Curator B writes their own:
- `wiring_anchor_graphs_v2_curator_b.c` — anchor library from scratch.
- `corpus_expand_v2_curator_b.c` — their own synonym tables.
- `wiring_geo_classifier_v2_curator_b.c` — their own feature-set curation.

They train the wiring transformer on their corpus using the same hyperparameters. They evaluate against their *own* development held-out (a 20-prompt set they write themselves, distinct from the sealed v2 held-out used for final measurement).

**Phase 3 — Sealed evaluation (1 week).** The original v2 held-out — kept sealed during Phase 2 — is evaluated against Curator B's library. The same `tools/scaling_leakage_audit.sh` (Audit B Jaccard < 0.7) runs first; if any prompt in the original held-out scores ≥ 0.7 Jaccard against Curator B's training corpus, that prompt is excluded and reported as a leakage event.

Cross-leakage audit also runs in the other direction: if Curator B's development held-out (revealed after Phase 2) overlaps the original v2 anchors at Jaccard ≥ 0.7, that's an information-leakage event from the briefing materials — a finding in itself.

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | Curator B's library scores 14-18/20 on v2 sealed held-out (within ±5pp of current 16/20) | Result outside 11-19/20 |
| **T2** | Audit B leakage rate of v2 held-out against Curator B's training corpus = 0 prompts at Jaccard ≥ 0.7 | > 2 prompts (= sealed-corpus breach; redo Phase 2) |
| **T3** | Curator B's library matches the same Phase 3b composition floor (≥ 6/10) on the 10-prompt multi-stage set | < 4/10 |
| **T4** | The curators' synonym table vocabulary **overlap** (set intersection / set union) is < 60% — i.e. they genuinely chose different words | ≥ 80% (= briefing was too prescriptive; vocabulary choices weren't independent) |
| **T5** | Curator B's library uses ≤ 30 families (no model-class change) | > 50 families (= scaled the library to compensate, not architecturally equivalent) |

The headline result is judged on **T1**:

| Outcome on T1 | Interpretation | Claim update |
|---|---|---|
| 14-18/20 (band) | **CONFIRMED architectural** | Headline claim strengthened; `INV-WIRE-061` upgraded to "independently reproduced" |
| 19-20/20 | **EXCEEDED ceiling** | Either Curator B is exceptionally skilled, briefing leaked information, or the ceiling is curator-LOW — a positive falsification |
| 11-13/20 | **PARTIAL miss** | Curator B's library is below band but not catastrophically; investigate vocabulary overlap (T4), report as "ceiling has curator-skill variance ±5pp" |
| < 11/20 | **FALSIFIED architectural** | The ceiling is curator-specific. Substantial rewrite of `RESEARCH_DISCLOSURE.md` §7 and `INV-WIRE-061`. **This is the most valuable possible outcome scientifically** — the field has no prior measurement of this. |

### 1.5 Skip rules

- If T2 fails (sealed-corpus breach): redo Phase 2 with new sealed held-out. Do not retroactively re-seal.
- If T4 fails above 80% (vocabularies too similar): the briefing package was too prescriptive. Re-do Phase 1 with a more abstract task specification (intent descriptions, no reference graphs). Report this as a finding in itself — "the architecture requires X bits of vocabulary information to reproduce" is a new measurement.
- If T5 fails (Curator B used many more families): document as a scaling confound and re-run Phase 2 with a hard cap of 25 families.

### 1.6 Falsification risk: High (intentional)

| Risk | Likelihood | Why this is good or bad |
|---|---|---|
| T1 falsifies at < 11/20 (curator-specific ceiling) | Medium | **Most scientifically valuable outcome.** Forces honest reframing of the calibrated claim. |
| T1 exceeds at 19-20/20 | Low-medium | Suggests information leakage in briefing OR exceptional Curator B skill. Either way: bigger writeup, careful interpretation. |
| Curator B unavailable / drops out mid-experiment | High | The hardest practical risk. Pre-commit budget for 1.5x effort or a second curator candidate. |
| Curator B reads existing `RESEARCH_*.md` / repo despite the brief | High | Sealing the briefing requires real discipline. Sign an honor agreement; verify by post-hoc comparison of vocabulary diversity (T4). |

### 1.7 What this experiment is NOT testing

- It is **not** testing whether the ceiling can be broken — that requires a *model class change* (external pretrained embeddings, per [`ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) §"What we still don't know" #2), not a curator change.
- It is **not** testing whether two curators can pool their libraries productively. Cross-curator fusion is interesting but a separate experiment.
- It is **not** measuring "the best human curator" — Curator B is one data point. Honest interpretation is "the ceiling holds across two curators" or "the ceiling did not hold across two curators," NOT "the ceiling is universal."
- It is **not** about absolute accuracy. The architecture's calibrated band is 75-80 %; both curators landing in that band is what passes, even if one is 16/20 and the other is 14/20.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| Pre-reg origin | [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) §2.3 |
| Hardest-unrun status | [`ORGANELLE_STATE.md`](../docs/research/ORGANELLE_STATE.md) §"What we still don't know" #3 |
| The v2 held-out being sealed | The 20-prompt set in `wiring_scaling_post_phase3.md` and `wiring_scoreboard_tier1.md` |
| Audit infrastructure | [`tools/scaling_leakage_audit.sh`](../tools/scaling_leakage_audit.sh) |
| Calibrated three-bound claim | [`RESEARCH_DISCLOSURE.md`](../docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md) §7, `INV-WIRE-061` |

---

## 2. Initial state

### 2.1 What's currently known

- Single-curator (Ajay) v2 result: 16/20 on the sealed v2 held-out (after subtractive sharpening from 15/20).
- Three feature variants (unigram / bigram / char-ngram TF-IDF) all hit the same ~80% ceiling — argues *model*-bound under one curator's vocabulary.
- No prior measurement exists in the literature on small-model bag-of-features retrieval ceilings being curator-dependent vs architectural.

### 2.2 Baselines

| Curator | Library size | v2 held-out |
|---|---|---|
| Curator A (Ajay, existing) | 20 families, ~340 anchor entries, ~200 synonym pairs | 16/20 |
| Curator B (new, this experiment) | ≤ 30 families | **target 14-18/20** |

### 2.3 Dependencies / blockers

- **Find Curator B.** Hardest dependency. Candidates: a research assistant; an independent ML engineer hired for the contract; a collaborator from a partner lab. Budget: 3-5 weeks at market rate (~£3,000-£8,000 GBP depending on the candidate).
- **Sealed v2 held-out.** Must not appear anywhere in the repo before Curator B finishes. Audit before measurement.
- **Honor agreement.** Curator B agrees in writing not to read `docs/research/RESEARCH_PIPELINE_IR.md`, `wiring_*.md`, the existing anchor source files, or the existing synonym table during Phase 2. Post-hoc audit via T4 vocabulary overlap.
- **Compensation budget.** Allocated separately from the project's research time — needs explicit Enjector approval.

### 2.4 Briefing materials (to be assembled in Phase 1)

| Document | Goes to Curator B | Sealed from Curator B |
|---|---|---|
| `BRIEF_TASK_SPEC_v2.md` | ✅ Family intent descriptions | ❌ No reference vocabulary |
| `BRIEF_IR_GRAMMAR.md` | ✅ Full IR grammar + 5 examples from a *different* domain | ❌ |
| `BRIEF_HARNESS.md` | ✅ Test harness, leakage audit tool | ❌ |
| Existing anchor / synonym / classifier source | ❌ | ✅ Sealed |
| v2 held-out prompts | ❌ | ✅ Sealed until Phase 3 |

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Curator B identity (or anonymous designation), time spent, library statistics
- 3.2 Pre-Phase-3 leakage audit results
- 3.3 Sealed evaluation results — T1, T2, T3 numeric outcomes
- 3.4 Vocabulary-overlap analysis — T4
- 3.5 Family-count comparison — T5
- 3.6 Curator B retrospective: what was hard, what was unexpected, what choices they regretted

---

## 4. Conclusion

**TODO** — fill on measurement commit. Sections to populate:

- 4.1 Verdict per T1-T5 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Headline outcome: CONFIRMED / EXCEEDED / PARTIAL / FALSIFIED architectural
- 4.3 Claim updates: `RESEARCH_DISCLOSURE.md` §7, `INV-WIRE-061`, `ORGANELLE_STATE.md` headline
- 4.4 If FALSIFIED: substantial rewrite of the three-bound framing. Open follow-up: what does a curator-specific ceiling mean for productisation?
- 4.5 If CONFIRMED: this is one of the few independently-reproduced ceiling claims in small-model research. Likely standalone short paper.
- 4.6 Next moves
