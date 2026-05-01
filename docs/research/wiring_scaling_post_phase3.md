# Post-Phase 3 cleanup + extensions — final consolidated findings

Branch: `investigation/post-phase3-cleanup-and-extensions`. Three numbered moves, all merged into a single arc.

## #1 — Wiring binary Phase 8 regression: FIXED

Removed the 20 Phase 1-4 family entries (slots 20-39, geometry/physics/statistics/math chains) from `wiring_anchor_graphs.c` and `wiring_geo_classifier.c`. They remain in `corpus_expand.c` synonym tables and `tfidf_main.c` `FAMILY_NAMES` so the TF-IDF measurement infrastructure is intact.

| Eval | Before #1 | After #1 |
|---|---|---|
| Phase 8 (correct on all 5 inputs) — `--clean-only` | **3/20** | **20/20** ✓ |
| HEADLINE strict-verified | 20/20 | 20/20 |
| Phase 3b composition | 7/10 | 7/10 |
| TF-IDF v2 retrieval | 16/20 | 16/20 |
| TF-IDF no-regression | 39/40 | 39/40 |
| TF-IDF adversarial | 17/20 | 17/20 |

**Status: shipped on main.** The "real" fix (vote-loop scoring change in `demos/wiring_organelle/main.c`) is documented in `wiring_binary_phase8_regression.md` as paths 1-3 but not done. Path 4 (rollback) was the cheapest legitimate restoration.

## #2 — v3 deep synonyms: HURT, did NOT help

Phase 3's caveat was "v3 used lean synonyms — a fair v3-with-deep-synonyms run is missing." #2 ran that test.

Expanded each of 20 v3 families' synonym tables from ~6 entries (3 templates) to ~12 entries (5 templates), audit-clean before measurement.

| Eval | v3 lean | v3 deep |
|---|---|---|
| **TF-IDF Top-1 on v3** | **3/20** | **0/20** (regressed) |
| v2 | 16/20 | 16/20 |
| Adversarial | 17/20 | 18/20 |

Root cause: deep templates introduced 5 generic English glue phrases per family ("expressed by", "formed by", "computed as", "scaled by"). Across 20 families, those words now appear with high frequency → low IDF → no discriminative weight. Family centroids collapsed toward each other.

**Sharpened §44.5 claim:** synonym depth helps only when each new entry is *domain-distinctive*. v2's 80% rate depended on uniqueness of pricing/math/finance/physics nouns (interest, factorial, fibonacci, sigmoid, momentum). v3's chemistry/time/conversions families share generic English vocabulary (count, value, total, scale, conversion) that doesn't naturally distinguish — independent of curation depth.

**Status: code reverted to v3 lean (3/20). Deep version preserved as negative-result tier in `wiring_scaling_v3_deep_negative.md`.**

## #3 — Char n-gram TF-IDF: ceiling not moved (triple confirmation)

Implemented `--char-ngram` flag in `tfidf_main.c`: tokenizer emits length-3 char trigrams from each word (prefixed `_t_` to avoid namespace collision with unigrams/bigrams).

4-cell measurement, three feature variants:

| Eval | Unigram | Bigram | Char-ngram |
|---|---|---|---|
| v2 (deep distinctive synonyms) | 16/20 (80%) | 15/20 | 15/20 |
| v3 (lean generic-English synonyms) | 3/20 (15%) | 2/20 | 3/20 |
| Phase 2c no-regression | 39/40 | 39/40 | 39/40 |
| Adversarial axis-2 | 17/20 (85%) | 18/20 | 18/20 (+1) |

**Triple confirmation: bag-of-features TF-IDF hits the same ceiling regardless of unigram/bigram/char-ngram features.**

The pattern is consistent: every feature variant gives a small +1 lift on adversarial (which uses synonym-shuffled paraphrases of *existing* families — feature variants can pick up partial-overlap signal there) but holds or hurts on v2/v3 (which use vocabulary-disjoint paraphrases — no feature variant can bridge zero-overlap held-outs without seeing similar surface forms in training).

**Word2vec was the next escalation candidate** ("if char n-grams don't help, train minimal C skipgram on the existing 12k corpus"). Decision: not implemented. Rationale: with char n-grams as a third bag-of-features variant showing the same ceiling, word2vec on a 12k-token corpus (orders of magnitude smaller than typical word2vec training) would likely produce noisy vectors that wouldn't break the pattern either. ~4 hours of implementation for almost-certain confirmation of the same finding.

**The honest model-bound conclusion: breaking past 75-80% on novel-paraphrase tests requires a model trained on far more data than the project corpus** — pretrained sentence embeddings (sentence-transformers, GloVe, etc.) or LLM-based retrieval. That's a real-dependency engineering investment, not a feature-tuning experiment.

## Cumulative arc — what we know now

The five-tier scaling arc, in honest cumulative form:

| Tier | What was claimed | Honest reality |
|---|---|---|
| Phase 1-4 (v1) | 1:1 scaling, 20/20 | Inflated by curator-self-overlap (Audit B = 19/20 ≥ 0.7) |
| Phase 2 v2 clean | 75% baseline | Honest. v2 = 15/20 with disjoint-vocab discipline |
| Phase 2 subtractive | 80% (16/20) | Honest +1 via removing generic clamp words from competing families |
| Phase 3 broad expansion | "75-80% should hold across new domains" | False — v3 hit 15% with same disjoint-vocab discipline |
| Phase 3 bigrams + post-Phase-3 char-ngrams | "ceiling might be unigram-specific" | False — three feature variants all hit the same ceiling |
| Post-Phase-3 #2 v3 deep | "deeper curation should help" | False — generic English glue dilutes centroids; v3 dropped to 0/20 |

**Final §44.5 honest claim:**

> **TF-IDF + curator's synonyms achieves ~75-80% retrieval on novel-paraphrase tests *only* when the family's distinctive nouns are genuinely unique to that family AND the held-out paraphrases use enough surface vocabulary overlap with the synonyms. For domains where family concepts share generic English vocabulary, the ceiling drops sharply (15% on v3) and is independent of curation depth or n-gram feature choice. Breaking past this ceiling requires a different model class (semantic embeddings trained on external corpora, or LLM-based retrieval) — not more curation, deeper synonyms, or richer bag-of-features.**

For deployment scoping: a 100-family library projects to ~75-80 retrievable *if* the families are in domains with distinctive nouns (math, physics, finance), and ~15-20 retrievable *if* the families are in domains with generic English nouns (chemistry concepts, time spans, conversions). The architecture is **domain-bounded** in addition to curator-bounded and model-bounded.

## Three known follow-ups still open

1. **Real fix for the wiring binary vote loop** (path 1-3 in `wiring_binary_phase8_regression.md`) — would let new families coexist with existing ones in the wiring binary. Currently rolled back.
2. **External-embedding scaling experiment** — would formally close the model-bound question. Requires breaking "pure C99, zero deps" project policy (sentence-transformers + ONNX runtime in C, or fastText vectors).
3. **Independent-curator reproducibility** — every measurement so far has had me as both curator (synonym tables) and held-out author. A second person should rebuild v2's family library from scratch with their own synonyms and held-outs. Would test whether the 75-80% ceiling is curator-vocabulary-specific.

None of these is a research breakthrough; all are engineering investments with bounded scope.

---

## Compositional held-out arc — separate from the retrieval arc above

The retrieval ceiling above (75-80 %) is for **paraphrase classification** — picking the right anchor family for a given prompt. There is a separate **compositional generation** arc that asks a different question: given a prompt that names primitives the curator did NOT pre-anchor, can the search machinery emit a verified `@graph` that runs end-to-end through the VM?

This second arc is documented in `RESEARCH_PIPELINE_IR.md` §47 and `RESEARCH_DISCLOSURE.md` §3, §4, §5, §6, §8. The honest cumulative trajectory:

| Iteration | Mechanism | Score | Disposition |
|---|---|---:|---|
| V1.0.4 (Phase 4 mechanism shipped) | Type-directed search over 36-prim manifest | (untested) | Mechanism only |
| V1.0.5 (first measurement) | Honest first run | 9/30 = 30 % | Pre-registered ≥ 50 % FALSIFIED — methodology saved a confirmation-bias trap |
| V1.0.6 (Phase 6 simple-search) | Beam-2, drop-name-dedup, geo-prior | 9/30 = 30 % | All three variations measurably no-op or harmful |
| V1.0.8 (Phase 6b argument binder) | Repeated-noun unification + outer-port-keyword inheritance | 12/30 = 40 % | PARTIALLY-RESOLVED |
| V1.0.9 (Phase 6c branched-mining) | Per-primitive synonym lift + symmetric tie-break + disabled legacy substring bump | 16/30 = 53 % | Original SLO-WIRE-005 design goal met |
| **V1.1.0 (Phase 6d port-aware inner picker)** | H8 noun-aware picker + dedup fallback + symmetry-aware override + single-letter filter | **19/30 = 63 %** | PARTIALLY-RESOLVED (60–69 % band); 70 % aggregate target NOT met; H9 not implemented |

`GAP-WIRE-010` (binder positional scoping, ~+13 pp predicted) is the highest-leverage residual fix, customer-signal-gated.

This compositional headline (63 %) is now the **second SLO-track** alongside the retrieval ceiling — both honest, both leakage-audited, both falsifiable.

---

## Phase 7 — OPA Adaptive-Depth (V1.2.0, primitives shipped, measurement deferred)

Orthogonal to the wiring layer entirely: V1.2.0 ships the API primitives (`OpaActHalting`, `OpaFrozenInput`) for the four pre-registered mechanisms in `OPA_ADAPTIVE_DEPTH_ROADMAP.md` v1.0. M1 (ACT halting) and M2 (frozen input) have full APIs + 6 unit tests; M3 (loop-index step token) and M4 (depth extrapolation) are explicitly deferred. Demo integration of M1/M2 is also deferred per the roadmap's customer-signal trigger. The §9.7 measurement section in `RESEARCH_DISCLOSURE.md` stays empty until Phase 7b lands. V1.0 demo numbers (90 % 8-puzzle, 88 % Connect-4) remain the published baseline.
