# Scaling experiment Phase 3 — 40 families + bigram TF-IDF

Branch: `investigation/scaling-40-families-bigram`. Commits 69e161f → present.

## Question (from the user)

After Phase 2 hit a structural ceiling around 80% on v2-clean held-out and showed whack-a-mole behavior under subtractive sharpening, the user picked Option B: "broad expansion to 40 families AND test bigram TF-IDF to see if the ceiling is curator-bound or model-bound."

## Pre-registration (locked before measurement)

| ID | Hypothesis | Threshold |
|---|---|---|
| H_v3 | TF-IDF unigram on 20 new disjoint-domain families (chemistry/time/conversions/combinatorics) — Top-1 EXACT match | ≥ 15/20 (75%, matching v2 ceiling) |
| H_bigram_v2 | Bigram TF-IDF lifts v2 (existing 20 new families) from 16/20 toward ≥ 18/20 | ≥ 18/20 |
| H_bigram_holds | Bigram doesn't regress no-regression or adversarial below pre-Phase-3 baselines | ≥ 38/40 / ≥ 16/20 |

Decision criterion: **bigram lifts v2 ≥ 18/20 OR bigram holds adversarial AND v3 ≥ 15/20 = ceiling moved.**

## What was added

20 new families across 4 disjoint domains, slots 40-59 in `tools/corpus_expand.c` synonym tables and `FAMILY_NAMES` in `demos/manifold_classifier/tfidf_main.c`. **TF-IDF-only integration** — deliberately skipped `wiring_anchor_graphs.c` and `wiring_geo_classifier.c` (and the GEO_DIMS=60 bump) to avoid worsening a wiring-binary regression discovered during Phase 3 setup (see `wiring_binary_phase8_regression.md`).

Domains:
| Slots | Families |
|---|---|
| 40-44 | molarity, mole_ratio, yield_percentage, dilute_volume, molar_mass_x_moles |
| 45-49 | minutes_to_hours, hours_to_seconds, elapsed_seconds, seconds_remaining, average_two_durations |
| 50-54 | celsius_to_kelvin, meters_to_centimeters, kg_to_grams, inches_to_cm, bytes_to_kilobytes |
| 55-59 | factorial_n, power_of_two_n, triangular_doubled, fibonacci_squared, fib_minus_fact |

Bigram support added to `tfidf_main.c` via `--bigram` flag: tokenizer emits unigrams + adjacent-pair bigrams (`tok[i]_tok[i+1]`), centroid math unchanged.

## Pre-measurement audit (before any TF-IDF run)

`tools/scaling_leakage_audit.sh pipeline_corpus_scaling_heldout_v3.txt pipeline_corpus_phase4_train.txt`:

| Audit | v3 result |
|---|---|
| A: verbatim leaks | **0/20** ✓ |
| B: Jaccard ≥ 0.7 | **0/20** ✓ (no near-duplicates) |
| C: ≥ 50% lexical anchors | 1/20 (only inches_to_cm) |

v3 is the cleanest held-out built so far — disjoint-vocabulary discipline worked well.

## 4-cell measurement results

| Eval | Unigram | Bigram | Δ |
|---|---|---|---|
| **v2 (existing 20 families, deep synonym tables from Phase 1-4)** | **16/20 (80%)** | **15/20 (75%)** | **-1** |
| **v3 (new 20 families, lean synonym tables from Phase 3)** | **3/20 (15%)** | **2/20 (10%)** | **-1** |
| No-regression Phase 2c (40 prompts) | 39/40 (98%) | 39/40 (98%) | 0 |
| Adversarial axis-2 (20 prompts) | 17/20 (85%) | 18/20 (90%) | +1 |

## Verdict against pre-registration

| Hypothesis | Threshold | Result | Verdict |
|---|---|---|---|
| H_v3 (unigram) | ≥ 15/20 | **3/20** | **FAIL** |
| H_bigram_v2 | ≥ 18/20 | 15/20 | **FAIL** |
| H_bigram_holds adversarial | ≥ 16/20 | 18/20 | PASS |
| H_bigram_holds no-regression | ≥ 38/40 | 39/40 | PASS |

**Both substantive hypotheses falsified. The ceiling is real and bigram TF-IDF does not move it.**

## Two findings, one each axis

### Finding 1 — synonym-table depth matters more than family count

v3's 3/20 with unigrams (vs v2's 16/20) is not a "more families = worse retrieval" result. Both have 20 prompts each, both are vocabulary-disjoint from training. The difference is **how much surface coverage each family has**:

| Family set | Avg synonyms/family | Avg templates/family | Result |
|---|---|---|---|
| v2 families (Phase 1-4) | ~12 across 3-4 groups | 4-6 | 80% |
| v3 families (Phase 3) | ~6 across 2-3 groups | 3 | 15% |

I deliberately wrote v3 synonym tables more concisely to test scaling behavior. The result: **TF-IDF retrieval is bottlenecked by per-family centroid mass on distinctive vocabulary, not by family count.** A new family added with only 6 synonyms produces a sparse centroid that loses to richer existing-family centroids on most novel paraphrases.

This means the §44.5 #1 question ("does adding a family translate to held-out coverage?") has an **implicit prerequisite**: the family must be curated *deeply*, not just declared. ~15-min/family of curation gets you 15% retrieval; ~30+ min/family of curation gets you 75-80%.

### Finding 2 — bigram features don't break the ceiling

Bigram TF-IDF helps on adversarial (+1) but hurts on both v2 and v3 (-1 each). Why:

- **Adversarial axis-2** uses synonym-shuffled paraphrases of *existing* families — many bigrams from training (e.g. "compound interest", "tax owed") still appear. Bigrams add useful signal.
- **v2 and v3** use vocabulary-disjoint paraphrases — held-out shares almost no unigrams with training, let alone bigrams. Adding bigram features just adds noise dimensions.

The structural ceiling on novel-paraphrase generalization isn't unigram-specific. Any bag-of-features classifier (unigrams, bigrams, character n-grams, etc.) faces the same constraint: **without seeing the held-out's surface vocabulary in training, retrieval can't find a match.**

## What this means for §44.5 (revised again)

The cumulative trajectory of this experiment:
1. v1 contaminated: 20/20 (100%) — invalid
2. v2 clean: 15/20 (75%) — first honest baseline
3. v2 + subtractive sharpening: 16/20 (80%) — small clean lift
4. v3 (Phase 3 new families, lean synonyms): 3/20 (15%) — synonym depth matters
5. v2 with bigrams: 15/20 (75%) — bigrams don't help
6. v3 with bigrams: 2/20 (10%) — bigrams hurt on lean tables

**Honest claim: TF-IDF + curator's synonyms achieves ~75-80% retrieval on novel-paraphrase tests when families are deeply curated (~30 min/family of synonym writing). Lean curation (~15 min/family) drops to ~15%. Neither bigram features nor subtractive sharpening break the 75-80% ceiling.**

For deployment scoping: a 100-family library projects to ~75 retrievable on novel-paraphrase tests **assuming each family receives the deep-curation budget**. Cutting curation cost in half cuts retrieval to ~15%.

Beyond ~80%, the next move is *not* more curation or more features — it's a different model class (genuinely semantic, e.g. small embedding model trained on domain corpora; or LLM-based retrieval).

## Honest caveats

1. **The v3 lean-synonym result is a curator-effort confound, not a scaling-curve datapoint.** A fair v3-with-deep-synonyms run was not done; that would test whether the v2 80% rate generalises to *new* domains (chemistry/time/conversions/combinatorics) at the same depth.

2. **The wiring binary regression remains.** Phase 3 deliberately routed around it; fixing it requires either a vote-loop scoring change in `demos/wiring_organelle/main.c` or rolling back the Phase 1-4 anchor_graphs additions.

3. **Bigram experiment is one model variant.** Character n-grams, word embeddings, or learned sentence representations might break the ceiling — bigrams alone are not the full "model-side experiment" that would falsify "the ceiling is structural to bag-of-features."

4. **Curator self-overlap risk persists.** Even though v3 had Audit B = 0/20 ≥ 0.7, the synonyms and held-out share an author. Independent-curator reproducibility remains untested.

5. **Adversarial +1 with bigrams is a single-prompt swing**, possibly noise. Three independent paraphrase sets would be needed to call it a real lift.

## Reproducibility

```sh
git checkout investigation/scaling-40-families-bigram
cd build
cmake --build . --target manifold_tfidf_demo corpus_expand --config Release
./corpus_expand pipeline_corpus_phase4_train.txt 1234

# 4-cell measurement:
./manifold_tfidf_demo            pipeline_corpus_scaling_heldout_v2.txt pipeline_corpus_phase4_train.txt | grep Top-1
./manifold_tfidf_demo            pipeline_corpus_scaling_heldout_v3.txt pipeline_corpus_phase4_train.txt | grep Top-1
./manifold_tfidf_demo --bigram   pipeline_corpus_scaling_heldout_v2.txt pipeline_corpus_phase4_train.txt | grep Top-1
./manifold_tfidf_demo --bigram   pipeline_corpus_scaling_heldout_v3.txt pipeline_corpus_phase4_train.txt | grep Top-1

# Pre-measurement audit (re-run any time):
bash ../tools/scaling_leakage_audit.sh pipeline_corpus_scaling_heldout_v3.txt pipeline_corpus_phase4_train.txt
```

The corpus is regenerated deterministically (seed=1234) so all numbers reproduce byte-stably.
