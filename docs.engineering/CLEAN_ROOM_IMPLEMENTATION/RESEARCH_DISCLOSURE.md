# RESEARCH_DISCLOSURE — Cancelled phases and honest restatements

**Document ID:** MGC-DISCL-001
**Version:** 1.0
**Status:** APPROVED (records-only document)
**Last updated:** 2026-04-30

---

## 1. Purpose

This document captures, in one place, the experimental phases that were **pre-registered and then cancelled** under their pre-registered skip conditions, plus the **honest restatements** of headline numbers that were originally inflated by training-corpus leakage. It is the audit trail a regulator, customer security team, or independent reviewer can read in 5 minutes to know which claims have been retracted, which were never made, and which were validated under stronger conditions than originally promised.

The full research log lives in `docs/research/RESEARCH_PIPELINE_IR.md`, `RESEARCH_MANIFOLD_LEARNING.md`, and the `wiring_scaling_*.md` family. This file is the regulator-friendly distillation.

## 2. Cancelled phases (pre-registered skip rule honoured)

### 2.1 Phase 3a-full — EKAN-Network classifier

- **Pre-registered hypothesis (`RESEARCH_PIPELINE_IR.md` §40):** A learned EKAN-Network classifier should outperform the handcoded keyword-bag baseline on the adversarial axis-2 stress test. Pre-registered target: 12–16/20.
- **Phase 3a outcome (§41):** A simpler TF-IDF centroid classifier — explicitly the *minimal* learned encoder — scored 4/20 on the adversarial axis-2 set, vs the 12–16/20 prediction.
- **Pre-registered §40.7 skip rule:** "If the simplest learned encoder underperforms the handcoded baseline by more than 4 points on adversarial axis-2, the more complex encoder (EKAN-Network) shall be cancelled."
- **Decision:** **CANCELLED** per the skip rule. The 4/20 result was 8 points below the lower bound of the prediction interval.
- **Implication:** No learned encoder beats the handcoded keyword bag *at the 408-example corpus scale*. Phase 4 (corpus expansion) reopened the question and answered it positively under more data.

### 2.2 Phase 3c — RAG fallback

- **Pre-registered hypothesis (`RESEARCH_PIPELINE_IR.md` §40):** A retrieval-augmented-generation fallback over an external semantic index should bridge the 408-example coverage gap.
- **Pre-registered §40.7 skip rule:** "Phase 3c is conditional on Phase 3a-full shipping. If 3a-full is cancelled, 3c is cancelled."
- **Decision:** **CANCELLED** per the conditional rule.
- **Implication:** Scaling to 100s of families across all domains requires external semantic embeddings (`GAP-WIRE-002`, deferred). Phase 4's corpus expansion is the partial mitigation actually shipped.

### 2.3 Phase 4b-full — EKAN-Network on the expanded corpus

- **Pre-registered hypothesis (`RESEARCH_PIPELINE_IR.md` §45):** With ≥ 4,000 prompts, EKAN-Network should outperform TF-IDF on adversarial axis-2.
- **Pre-registered §45.2 outcome logic:** "If the simpler TF-IDF classifier exceeds the escalation trigger (≥ 16/20 on axis-2), the more complex 4b-full is cancelled."
- **Phase 4 outcome (§46):** TF-IDF scored 18/20 on axis-2 (the SLO-WIRE-004 number).
- **Decision:** **CANCELLED** per the §45.2 logic. Simpler model exceeded the trigger.

## 3. Honest restatements

### 3.1 Wiring NL → graph (the leakage audit)

- **Original claim (pre-Phase 2d):** 75 % median / 80 % peak on the wiring layer over a 20-prompt held-out set, after 17 phases of corpus engineering 8→9→10→11→12→13→15.
- **Phase 2d audit (`RESEARCH_PIPELINE_IR.md` §38):** **13 of the 20 original held-out prompts appeared verbatim in the wiring training corpus.** The leakage was introduced by Phase 13's lexical-anchoring expansion at `tools/pipeline_corpus_gen.c:1902, 1924, 1950, 1979, 2011, …`. The 35→75% lift attributed to the 17-phase corpus engineering was largely the model memorising prompts that Phase 13 had explicitly added to training data.
- **Restated honest headlines:**
  - **Anchor-retrieval mechanism on the leakage-free Phase 2c paraphrases: 100 % (20/20).**
  - **Wiring transformer alone on the same clean set: 35 % (7/20).**
  - **Phase 3b composition multi-stage prompts: 60 % (6/10) on a 10-prompt set with no overlap.**
  - **Phase 4 TF-IDF on the expanded ≥ 4,000-prompt corpus, adversarial axis-2: 90 % (18/20), with 100 % no-regression on the Phase 2c clean set.**
- **Reproducer flags:** `--clean-only` (anchor retrieval), `--no-anchor --clean-only` (wiring-only baseline), `--composition` (multi-stage). For Phase 4: `./corpus_expand pipeline_corpus_phase4_train.txt 42` then `./manifold_tfidf_demo pipeline_corpus_adversarial.txt pipeline_corpus_phase4_train.txt`.
- **Standing audit script:** `tools/scaling_leakage_audit.sh` checks every held-out prompt against the training corpus before any new claim is published. It is the project's standing protection against a recurrence.

### 3.2 What is NOT being claimed

Per the strategic positioning in `BRD.md` and `docs/STRATEGY_ONE_PAGER.md`:

- **Not** higher accuracy than the latest LLM on open-domain tasks (the architecture loses by design on free-text, distribution-wide questions).
- **Not** a replacement for incumbent risk / fraud / defence systems — a complement that addresses the auditability + edge gap.
- **Not** a research breakthrough — the architecture is mostly conventional. The discipline around audit, leakage-checking, and pre-registration is what is distinctive.
- **Not** scalable to 100s of families across all domains without external semantic embeddings (per the cancelled Phase 3c finding).

## 4. Standing protections against recurrence

| Protection | Mechanism | Source of truth |
|---|---|---|
| Pre-registration of skip rules | Every multi-phase research arc names its pre-registered §X.Y outcome rule before measurement. | `RESEARCH_PIPELINE_IR.md` §40, §45 |
| Leakage audit | `tools/scaling_leakage_audit.sh` runs against every held-out set before it is reported. | `tools/scaling_leakage_audit.sh` |
| Honest headline gate | A claim cannot be reported until it cites the leakage-audited held-out set. | This document |
| Cancellation transparency | Every cancelled phase is documented here, not silently dropped. | This document |

## 3. Phase 5 — Compositional generator pre-registration (2026-04-30)

The compositional generator fix (Stream B of `COMPOSITIONAL_GENERATOR_FIX_PLAN.md`) ships a deterministic type-directed search over the primitive manifest in `wiring_primitive_manifest.{h,c}`. To prevent recurrence of the Phase-2 leakage incident, the held-out evaluation is **pre-registered** here before any score is reported.

### 3.1 Held-out test set

`demos/wiring_organelle/pipeline_corpus_compositional_test.txt` — 30 prompts in three axes:

- **Axis 1** (10): compose 2 primitives that DO NOT co-occur in any of the 20 anchor families. Tests "novel pair" composition.
- **Axis 2** (10): compose 3 primitives where at least one is referenced by a synonym never seen in `tools/corpus_expand.c`.
- **Axis 3** (10): compose 2 primitives + an outer transform; tests that the search backtracks under type pressure.

Each prompt is annotated with `# EXPECTED: <comma-separated primitives>` and `# REFERENCE: <function_name>`; the latter resolves through `wiring_references.c::wiring_reference_compute_at` to a deterministic numeric answer for the demo's input sets.

### 3.2 Pre-registered targets

| Mode | Target | Definition |
|---|---|---|
| Default ranking | **≥ 50 % (15/30)** verified-and-correct on the leakage-audited compositional set. | "Verified" = `pipeline_verify` returned `PIPE_OK`. "Correct" = the `pipeline_execute_vm` numeric output matches the reference for at least 3 of the 5 input sets. |
| With `--use-expected` ranker gate | **≥ 80 % (24/30)** verified-and-correct. | Same as above but ranking promotes a candidate whose primitive set is a superset of `# EXPECTED:`. |
| No-regression | **100 % (20/20)** on Phase 2c clean. | Honest baseline — must not regress when the new search is integrated alongside the existing anchor / transformer paths. |

### 3.3 Skip rule

If the achieved score on Axis 1 + Axis 2 falls below 5/20 (25 %) under the default ranker, Stream C is `PARTIALLY-RESOLVED` rather than `RESOLVED`; the actual achieved score is recorded as the new SLO and a `GAP-WIRE-006` is opened to track the next iteration. This rule is in force regardless of how Axis 3 scores.

### 3.4 Standing leakage audit

`tools/scaling_leakage_audit.sh` MUST be run with the new test file as held-out and the entire wiring training corpus (`pipeline_corpus_train.txt`, `pipeline_corpus_planner.txt`, `pipeline_corpus_phase4_train.txt`) as the source. Verbatim or near-duplicate (Jaccard ≥ 0.7) hits MUST be zero before any Phase 5 score is reported.

### 3.5 Outcome (2026-05-01 — first run)

The Phase 5 harness (`build/wiring_phase5_harness`) was run against the leakage-audited 30-prompt held-out. Results:

| Axis | Description | Verified | Correct (≥3/5 input sets) |
|---|---|---|---|
| 1 | novel pair (2 primitives) | 10/10 | 2/10 |
| 2 | synonym stress (3 primitives) | 10/10 | 4/10 |
| 3 | outer transform (composition) | 10/10 | 3/10 |
| **Total** | | **30/30 (100 %)** | **9/30 (30 %)** |

**Verified rate: 100 %** — every prompt produced a graph that passed `pipeline_verify` and ran end-to-end through `pipeline_execute_vm`. The compositional mechanism is sound at the "generates a runnable graph" level.

**Correctness rate: 30 %** — below the §3.2 pre-registered target of 50 %.

#### Disposition (per §3.3 skip rule)

- Axis 1 + Axis 2 combined = 6/20 = 30 %, **above** the 5/20 (25 %) floor.
- Therefore: **`GAP-WIRE-005` is PARTIALLY-RESOLVED, not RESOLVED.**
- The achieved 9/30 (30 %) score becomes the new SLO baseline.
- A fresh `GAP-WIRE-006` is opened in `TRACEABILITY.md` to track the next iteration toward the 50 % target.

#### Honest analysis of the 21 failures

Of the 30 verified graphs, 21 produced numerically-wrong answers (matched < 3 of 5 input sets). Categorising the failure modes by inspecting the harness output:

1. **Wrong primitive ordering** (≈ 10 cases) — the greedy beam=1 search picks the highest-scoring primitive as the *outer* node, but the prompt's semantic ordering is reversed. Example: prompt 1 ("the absolute difference between x and y") produces `abs_val(subtract(x,y))` ordered as `subtract → abs_val` semantically, which is correct mathematically and matches `ref_abs_diff`. But the harness shows match=0/5 — investigation shows the search picked `abs_val` as outer with `subtract` as inner, so wiring is `abs_val(subtract(x,y))` = correct, but the input mapping is wrong: `subtract(x,y)` becomes `subtract(arg_0, arg_1)` in the graph; if `arg_0` and `arg_1` are mapped to the input set's slot 0 and slot 1, the answer should match. The likely cause is that the search wired `abs_val` to a *fresh* signature input rather than `subtract.out` — i.e. the inner-node connection logic mis-binds in some cases.
2. **Synonym mismatch** (≈ 5 cases) — Axis 2 prompts use synonyms not in the manifest's keyword sets (e.g. "after-tax markup" — `apply_tax` matches, but `markup` was de-prioritised by the dedup pass).
3. **Type-mismatch or arity-mismatch** in the input mapping (≈ 6 cases) — e.g. prompt 12 ("after-tax markup at rate r") expects `apply_tax(markup(price, m_rate), t_rate)` with 3 sig inputs, but the search produced `apply_tax(price=arg_0, rate=arg_1) ∘ tax_amount` and never wired markup. The dedup heuristic killed the right primitive.

#### Next steps (recorded as future work, not done now)

Per the methodology's "honest-disclosure-then-iterate" pattern, the V1.0.4 search is **frozen at 30 %** and the gap is documented. Future iterations:

- **Beam width 2–4 instead of greedy beam=1** — explore alternative outer/inner pairings; pick the candidate with the best fidelity score against `# EXPECTED:` annotation when available. Likely lifts axis-1 wrong-ordering failures.
- **Drop the dedup-by-name pass** and instead allow the same primitive on multiple ports when the prompt clearly mentions it twice (e.g. "x squared and y squared"). Likely lifts axis-2 synonym-stress failures.
- **Consult the wiring_geo_classifier for a family-prior** to disambiguate when keyword scores tie at the manifest level.

These changes are pre-registered as Phase 6 in a future revision; the present pass leaves the 30 % SLO as the honest baseline and does not silently re-tune to chase the 50 % target.

## 4. Phase 6 — Compositional search improvements (pre-registration, 2026-05-01)

V1.0.5 left `GAP-WIRE-006` open with a 30 % baseline against a 50 % target. Phase 6 attempts three named improvements **pre-registered** here BEFORE any code change. The result will be measured against the same `pipeline_corpus_compositional_test.txt` 30-prompt held-out, scored by the same `wiring_phase5_harness`, with no re-tuning of the held-out or the reference oracle.

### 4.1 Hypotheses (in falsifiable form)

- **H1 — beam widening.** Replacing the greedy beam=1 outer-pick with a beam-2 search (try the two highest-scoring outer candidates, build both compositions, keep whichever produces a verified graph whose primitive set has greater overlap with the prompt's content nouns) lifts axis-1 wrong-outer-ordering failures. Predicted axis-1 lift: ≥ +2 prompts (2 → 4 of 10).
- **H2 — drop name-dedup pass.** The current de-duplication of inner picks across outer input ports throws away the inner primitive when it gets matched to two ports (e.g. `square` matched on both inputs of `subtract`). Without dedup, both ports get an inner; a verifier-side type-mismatch is auto-recovered by `pipeline_repair`. Predicted axis-1 + axis-2 lift: ≥ +2 prompts.
- **H3 — geo-classifier tie-break.** When the manifest scoring ties (multiple primitives at the same hit count), consult `wiring_geo_predict_top_k(prompt, ...)` for a family hint and prefer the primitive in that family. Predicted axis-2 lift: ≥ +1 prompt.

The three improvements compose orthogonally; each is gated by an independent compile-time toggle so the contribution can be ablated.

### 4.2 Aggregate target

Combined target (all three on): ≥ **15/30 (50 %)** correct — the original §3.2 design goal. **Failure target**: < 12/30 (40 %) — at that point the simple-search hypothesis is falsified; per the methodology, escalation to a learned ranker (Phase 6b) is reconsidered only if a customer signal warrants the cost.

### 4.3 Disposition logic

- If achieved ≥ 50 %: `GAP-WIRE-006` → `RESOLVED`. SLO-WIRE-005 promoted.
- If achieved 40–49 %: `GAP-WIRE-006` stays `PARTIALLY-RESOLVED`. New SLO-WIRE-005 baseline becomes the achieved score. No new gap opened.
- If achieved < 40 %: `GAP-WIRE-006` stays `OPEN`, V1.0.5 30 % baseline persists. Phase 6b deferred indefinitely.

### 4.4 No-regression invariant

The four pre-existing wiring numbers (anchor 100 %, fragment-chain 60 %, transformer-only 35 %, Phase 4 TF-IDF 90 % on adversarial axis-2) MUST persist unchanged. The Phase 6 changes touch only `wiring_compositional_search.c`; no wiring-organelle path is modified.

### 4.5 Outcome (2026-05-01 — Phase 6 first run)

The three improvements landed together (no per-improvement ablation in this pass — the compile-time toggles `WIRING_BEAM`, `WIRING_KEEP_DUPS`, `WIRING_USE_GEO` are all default-ON; ablation deferred unless we revisit). Re-running the harness:

| Axis | V1.0.5 | V1.0.6 | Δ |
|---|---:|---:|---:|
| 1 (novel pair) | 2/10 | 3/10 | +1 |
| 2 (synonym stress) | 4/10 | 4/10 | 0 |
| 3 (outer transform) | 3/10 | 2/10 | −1 |
| **Total correct** | **9/30** | **9/30** | **0** |
| Verified | 30/30 | 30/30 | 0 |

**Aggregate target was ≥ 50 % (15/30); achieved 30 % (9/30). Failure target was < 40 % (12/30); achieved 30 %.** The simple-search hypothesis is **falsified** under the §4.3 disposition logic.

Per-prompt observation: the `WIRING_KEEP_DUPS=1` change produced visible duplication in many graphs (e.g. `[abs_val, abs_val, subtract]` instead of `[abs_val, subtract]`). Because the VM ABI assigns the native's return value only to the first output port and the dispatcher binds the duplicate to a separate downstream slot, the duplicate is computed but its result is sometimes mis-routed — net zero on the score. Two prompts changed sign:

- **Gain** (axis 1, prompt 7 `ref_sigmoid_double_x`): 0/5 → 5/5. The keep-dups change exposed a `[double_val, sigmoid]` pairing that was previously discarded by the dedup.
- **Gain** (axis 2, prompt 17 `ref_gcd_sq_diff`): 1/5 → 4/5 (CORRECT). The beam widening picked `gcd` as outer where V1.0.5 picked `subtract`.
- **Loss** (axis 3, prompt 21 `ref_double_gcd_sq_y`): originally `[double_val, square]`-only mis-fired all 5; under Phase 6 it picks `[square, square, gcd]` — gcd now appears but the wrong square is wired to gcd's first arg (matches 1/5 instead of 0/5; below the 3/5 correctness threshold).
- **Loss** (axis 3, prompt 24 `ref_relu_gcd_x_y`): unchanged at CORRECT — actually preserved; the loss elsewhere is prompt 21's near-miss not promoting plus prompt 27 `ref_discount_ke` slipping out.

#### Disposition

Per §4.3:
- 9/30 = 30 % is **below** the 40 % failure threshold.
- `GAP-WIRE-006` remains **OPEN**. The V1.0.5 30 % baseline persists.
- Phase 6b (learned ranker / beam widening to ≥ 4 / external semantic embeddings) is **deferred indefinitely**, opened only on customer signal.

The Phase 6 changes are kept in the source — they don't regress correctness in the aggregate, they expose a structural failure mode (duplicate-inner-node misrouting) that is genuinely informative, and they leave the search trivially toggleable for ablation. The `--use-expected` ranker gate (`SLO-WIRE-006`) was not implemented — without ground-truth at inference time it offers no production value, and the failure analysis suggests the bottleneck is graph-structure routing, not ranking.

#### What Phase 6 actually proved

- **Beam widening alone ≠ correctness.** The greedy outer pick is not the dominant failure mode at this corpus scale.
- **Keep-dups creates new failure modes** as fast as it fixes old ones. The right move is probably the opposite — a stricter argument-binding step that maps each prompt content noun to a unique input slot, regardless of which inner is selected.
- **Geo-classifier prior** in the form of a +1 score bump did not produce a measurable axis-2 lift on this set. Its 12-D family table was tuned for the legacy Phase-13-leaked anchor set, not for the Phase 5 prompts.

These observations are recorded for the next iteration; no further code changes are made in this pass.

## 5. Phase 6b — Argument-binder + manifest-driven geo (pre-registration, 2026-05-01)

V1.0.6 falsified the simple-search hypothesis (`§4.5`). The harness output localised four root causes; the V2.0 fix plan in `COMPOSITIONAL_GENERATOR_FIX_PLAN.md` attacks them with three new streams. As before, the plan is **pre-registered here BEFORE any code change**.

### 5.1 Hypotheses (in falsifiable form)

- **H4 — Argument-to-port binder.** Replacing the positional `arg_0..arg_N` cursor with a prompt-noun → port-name binder lifts axis-1 from 2/10 (V1.0.5) / 3/10 (V1.0.6) to ≥ 5/10. Predicted dominant lift.
- **H5 — Repeated-noun unification.** When the same prompt noun is bound to two ports, both ports share the same signature input, eliminating the V1.0.6 duplicate-inner misrouting failure mode (R2). Predicted: prompts 1, 6, 8, 9 (axis 1) and prompts 11, 12 (axis 2) move from match=0/5 to match≥3/5.
- **H6 — `INPUT_ORDER:` annotation.** A per-prompt input-order annotation in the held-out lets the harness remap inputs by noun rather than by position, so a topologically-correct graph with reordered signature inputs scores correctly. Predicted: 2 additional prompts in axis 3 promote.
- **H7 — Manifest-driven geo classifier.** Re-tuning the family table on the 36-primitive manifest converts the V1.0.6 +1 score-bump from a no-op into a meaningful disambiguator. Predicted axis-2 lift: ≥ 5/10 (vs 4/10 in V1.0.5/V1.0.6).

### 5.2 Aggregate target

Combined target (all three streams on): ≥ **15/30 (50 %)** — the original `SLO-WIRE-005` design goal. **Failure target**: < 12/30 (40 %). Same disposition logic as `§4.3`.

### 5.3 Disposition (Phase 6b)

- ≥ 50 %: `GAP-WIRE-005` and `GAP-WIRE-006` → `RESOLVED`. SLO-WIRE-005 promoted to the achieved score.
- 40–49 %: `GAP-WIRE-005` and `GAP-WIRE-006` stay `PARTIALLY-RESOLVED` with achieved score as new SLO baseline.
- < 40 %: gaps stay `OPEN`, V1.0.5 30 % baseline persists, Phase 6c (learned argument binder) deferred indefinitely.

### 5.4 No-regression invariant

The four pre-existing wiring numbers (anchor 100 %, fragment-chain 60 %, transformer-only 35 %, Phase 4 TF-IDF 90 %) MUST persist. Phase 6b changes touch only `wiring_compositional_search.c`, `wiring_arg_binder.{h,c}` (new), `wiring_primitive_manifest.{h,c}` (extension), `wiring_geo_classifier.{h,c}` (extension), `pipeline_corpus_compositional_test.txt` (annotation only), and `wiring_phase5_harness.c`. No wiring-organelle path is modified.

### 5.5 Outcome (to be filled when results land)

To be populated after re-running `wiring_phase5_harness` post-implementation.

## 7. Three-bound consolidation — TF-IDF retrieval ceiling (V1.0.7, 2026-05-01)

This section consolidates the post-Phase-3 cleanup arc (`docs/research/wiring_scaling_post_phase3.md`, `wiring_scaling_curve_phase3.md`, `wiring_scaling_v3_deep_negative.md`) into a single normative restatement. **Nothing is being newly pre-registered here** — every datapoint in this section is an outcome of an already-pre-registered experiment, restated as a closed-form structural bound.

### 7.1 The three structural bounds (each independently confirmed)

| Bound | What it is | Evidence | Implication |
|---|---|---|---|
| **Curator-bounded** | TF-IDF retrieval is bounded by the curator's synonym vocabulary; held-out paraphrases that share no surface words with training cannot be matched | v1 / v2 leakage audit and the falsified 1:1 claim (`RESEARCH_DISCLOSURE.md` §3.1); standing protection is `tools/scaling_leakage_audit.sh` (Audit B) | Held-out sets MUST be vocabulary-disjoint and audited BEFORE measurement; standard procedure recorded in `INV-WIRE-062` |
| **Model-bounded** | The retrieval ceiling is a property of the bag-of-features classifier *family*, not of unigram features specifically | Three feature variants (unigram / word-bigram / char-trigram) all converge within ±1/20 on the v2 vocabulary-disjoint set: 16/15/15 (`wiring_scaling_post_phase3.md` 4-cell table); recorded as `INV-WIRE-060` | Breaking past this ceiling requires a different model class (external pretrained semantic embeddings — see `GAP-WIRE-002`, gated by `GAP-DEP-001`); recorded as `SLO-WIRE-010` |
| **Domain-bounded** | The achievable ceiling depends on whether the family's distinctive nouns are unique vs share generic English vocabulary with other families | v3 (chemistry / time / conversions / combinatorics) lean: 3/20; v3 deep: 0/20 (`wiring_scaling_v3_deep_negative.md`); v2 (math / physics / finance) deep: 16/20 | Vertical productisation should target distinctive-noun domains where the upper bound is achievable; recorded as `INV-WIRE-061` and informs `docs/PRODUCT_FRAUD_DETECTION.md` (high-distinctiveness fraud nouns) |

### 7.2 What is NOT being added to the claim catalogue

Per the methodology, restated bounds are not new pre-registrations and do not earn new SLO targets beyond the calibrated ones. Specifically:

- **Not** a claim that bigrams or char n-grams "fail" — they perform as expected for bag-of-features methods; the convergence at the same ceiling IS the finding.
- **Not** a claim that v3 is unfixable — it is documented as fixable via either external embeddings (model class change) or vertical-domain restriction.
- **Not** a re-opening of any previously-cancelled phase. Phase 3a-full (EKAN-Network) and Phase 3c (RAG fallback) remain `CANCELLED`.

### 7.3 Standing protections (additive to §4)

| Protection | Mechanism | Source of truth |
|---|---|---|
| Bag-of-features convergence invariant | `INV-WIRE-060`: any new feature variant tested on a vocabulary-disjoint held-out MUST converge within ±1 prompt of unigram baseline. A diverging variant invalidates the calibrated ceiling and triggers re-measurement. | `BS_wiring.md` §4 |
| Distinctive-noun bound disclosure | `INV-WIRE-061`: any retrieval claim on a domain MUST disclose whether the family vocabulary is distinctive-noun or generic-English-vocabulary; the latter implies a ≤ 20 % ceiling regardless of curation depth. | `BS_wiring.md` §4 |
| Pre-measurement leakage audit | `INV-WIRE-062`: every new held-out set MUST run `tools/scaling_leakage_audit.sh` (Audit B Jaccard ≥ 0.7 ≤ 1/N) before any retrieval number is reported. | `BS_wiring.md` §4 + `tools/scaling_leakage_audit.sh` |

### 7.4 Cross-references for this section

- `docs/research/wiring_scaling_post_phase3.md` — full 4-cell measurement table and bigram/char-ngram derivation.
- `docs/research/wiring_scaling_v3_deep_negative.md` — root-cause analysis of why deep synonyms HURT v3.
- `docs/research/ORGANELLE_STATE.md` — synthesis treating the three bounds as a single calibrated claim.
- `GAP-WIRE-007` (RESOLVED) — bag-of-features ceiling.
- `GAP-WIRE-008` (RESOLVED) — domain-bounded ceiling.

## 6. Cross-references

- `BRD.md` BREQ-015..017 — cite restated headlines.
- `BS_wiring.md` INV-WIRE-040..041, SLO-WIRE-001..004.
- `docs/research/RESEARCH_PIPELINE_IR.md` — full development log.
- `docs/research/wiring_scaling_post_phase3.md` — honest scaling-curve closure.
- `docs/STRATEGY_ONE_PAGER.md` "What we are not claiming".

## 6. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction from `RESEARCH_PIPELINE_IR.md` and the strategy one-pager. |
| 1.1 | 2026-05-01 | Added §3 (Phase 5 compositional generator pre-reg), §4 (Phase 6 simple-search falsification + per-axis sign analysis), §5 (Phase 6b argument-binder pre-reg). |
| 1.2 | 2026-05-01 | Added §7 three-bound consolidation. Restates the post-Phase-3 cleanup arc as three structural bounds (curator-, model-, domain-bounded) with cross-references to `INV-WIRE-060/061/062`, `SLO-WIRE-008/009/010`, and `GAP-WIRE-007/008` (both `RESOLVED`). No new pre-registration; consolidates already-measured outcomes. |
