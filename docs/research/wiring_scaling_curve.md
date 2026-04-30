# Scaling-curve experiment — does linear-effort family expansion translate 1:1 to held-out coverage?

Branch: `investigation/scaling-curve-experiment`. Commits 1363753 → 4b83db1.

## The question

`docs/research/RESEARCH_PIPELINE_IR.md` §44.5 left exactly one open *engineering scope* question after the research arc closed: **how big to grow the curated library, and does the cost actually scale linearly with capability?** The earlier planning conversation framed it concretely: if a curator spends 30 min/family, do held-out paraphrases of that family get retrieved correctly?

This experiment answers that with measured data.

## Pre-registered hypotheses (locked before any new family was added)

| ID | Hypothesis | Threshold |
|---|---|---|
| H_main | TF-IDF Top-1 EXACT match on a 20-prompt held-out set (one paraphrase per new family, leakage-guarded) | ≥ 18/20 |
| H_no_regression | TF-IDF on the existing 40-prompt no-regression set holds steady | ≥ 38/40 |
| H_adversarial_floor | TF-IDF adversarial axis-2 stays within tolerance of the prior 18/20 baseline | ≥ 16/20 |

Skip conditions:
- If the first 5-family batch hit < 4/5 on its own held-out subset, halt and root-cause before continuing.
- If GEO_DIMS bump broke any existing test, halt and redesign the embedder.

Falsification clause: any of the three H_* failing = scaling is NOT 1:1; document the negative result honestly.

## What was added

20 new families across 4 disjoint domains, each using only existing primitives in `wiring_natives.c`:

| Domain | Families | Slots |
|---|---|---|
| Geometry | circle_area_ratio, square_of_sum, triangle_area, rectangle_perimeter, hypotenuse_squared | 20-24 |
| Physics | kinetic_energy_clamped, momentum, work_done, power_clamped, harmonic_sum | 25-29 |
| Statistics | variance_two, abs_z_score, range_two, midpoint_clamped, mse_simple | 30-34 |
| Math chains | lerp_clamped, cube_then_clamp, gcd_with_offset, harmonic_clamped, percentage_of_average | 35-39 |

Each family received entries in three places:
1. `demos/wiring_organelle/wiring_anchor_graphs.c` — canonical `@graph` DAG
2. `demos/wiring_organelle/wiring_geo_classifier.c` — slot + keywords
3. `tools/corpus_expand.c` — synonym groups + sentence templates

Plus a held-out paraphrase per family in `demos/wiring_organelle/pipeline_corpus_scaling_heldout.txt`, with each prompt added to `corpus_expand.c FORBIDDEN[]` to prevent training-on-test leakage.

Prerequisite: `GEO_DIMS` bumped 20 → 40 in `src/microgpt_geodesic.h` (16/16 geodesic tests still pass).

## Scaling curve

| Batch | Families curated | Corpus prompts | Vocab | Scaling held-out (Top-1 EXACT) | Per-family hit rate |
|---|---|---|---|---|---|
| 0 (baseline) | 0 | 4,102 | 341 | **0/20** (0%) | — |
| 1 (geometry) | 5 | 5,007 | 393 | **5/20** (25%) | 5/5 = 100% |
| 2 (physics) | 10 | 5,973 | 424 | **10/20** (50%) | 5/5 = 100% |
| 3 (statistics) | 15 | 6,874 | 447 | **15/20** (75%) | 5/5 = 100% |
| 4 (math chains) | 20 | 8,124 | 467 | **20/20** (100%) | 5/5 = 100% |

**Slope: exactly +1 hit per +1 family curated.** Every single curated family was retrieved correctly on its held-out paraphrase. No false negatives, no batch-level rate decay.

## No-regression bookkeeping

| Eval | Pre-experiment | Post-experiment | Pre-reg threshold | Verdict |
|---|---|---|---|---|
| Existing 40-prompt no-regression (Phase 2c clean originals + paraphrases) | 40/40 (100%) | 39/40 (98%) | ≥ 38/40 | PASS — single cross-family interference (`gross_minus_tax` → `apply_tax` on prompt #38) appeared in Batch 1 and persisted. Within tolerance. |
| TF-IDF adversarial axis-2 | 18/20 (90%) | 17/20 (85%) | ≥ 16/20 | PASS — single drop (one prompt that was previously borderline became misclassified after corpus growth shifted IDF weights). Within tolerance. |
| Geodesic test suite (`test_microgpt_geodesic`) | 16/16 | 16/16 | no regression | PASS |
| Wiring Phase 2c clean (`--clean-only` end-to-end) | 20/20 strict-verified, 20/20 correct on all inputs | unchanged after FAMILY_NAMES extension | no regression | PASS |

## Verdict against the pre-registration

| Hypothesis | Threshold | Result | Verdict |
|---|---|---|---|
| H_main | ≥ 18/20 | **20/20** | **PASS** (exceeded — no false negatives) |
| H_no_regression | ≥ 38/40 | 39/40 | **PASS** |
| H_adversarial_floor | ≥ 16/20 | 17/20 | **PASS** |

**Conclusion: linear-effort family expansion translates 1:1 to held-out TF-IDF retrieval coverage in the 0-to-20 family expansion regime tested.** Every curated family was retrievable; no batch-level decay; the no-regression and adversarial-floor budgets were both honoured.

## Honest caveats

1. **Curated synonyms test the curator, not the system.** The held-out paraphrases were written by me before any family was added, but my synonym tables were also written by me. There's a risk that the synonym tables and held-out paraphrases share my idiomatic vocabulary in ways a different curator wouldn't reproduce. The TF-IDF hit-rate is bounded by the *intersection* of curator-vocabulary and held-out-vocabulary. A second, independent curator should rebuild Batch 1 from scratch to verify reproducibility.

2. **The 1:1 slope held in this 0-20 regime; nothing here predicts 20-40 or 40-100.** The §44.3 axis-2 (weak keyword overlap) and axis-4 (domain-vocabulary drift) frontiers were "soft-closed" at 20 families per Phase 4. Adding 20 more pushed cross-family interference from 0 to 1 mismatched prompt on the no-regression set. Extrapolating: each additional 20-family batch may cost ~1 cross-family mismatch. At 100 families that's ~5 mismatches — soft scaling, not catastrophic, but non-zero.

3. **TF-IDF measures retrieval, not execution.** The wiring binary's end-to-end correctness depends on more than retrieval: graph parsing, primitive availability, reference-function comparison. The end-to-end metric for the new families would require writing per-family reference functions in `wiring_references.c`. That's an additional curator surface this experiment did NOT measure.

4. **The "30 min/family" cost is the human-curator estimate.** The actual machine-time cost (3-file edits + corpus regen + measurement) was negligible. The bottleneck in real deployment is curator judgement (which families to add, how to write synonym tables that generalise), not the typing.

5. **Curated families overlap with existing 20 in primitive composition.** All 20 new families use existing primitives in `wiring_natives.c` (no new natives added). The experiment validated *vocabulary-level* scaling, not *primitive-level*. Adding families that require new primitives (e.g., `circumference` requires a `pi` constant the natives don't have) is a separate axis.

## Where this leaves the §44.5 question

§44.5's "how big does the curated library grow?" question was framed as "an engineering decision, scales with intended deployment scope." This experiment converts that hand-wave into a measured rate: **+1 retrieval hit per +1 curated family for at least the first 20 expansions, with the cross-family interference budget burning at ~1/20 prompts per batch.**

For deployment scoping:
- A 50-family library: predicted ~50/50 retrievable, with ~2-3 cross-family mismatches on the no-regression set. Likely fine for most domains.
- A 100-family library: predicted ~100/100 retrievable, with ~5 cross-family mismatches. Worth re-measuring at the 50-family checkpoint to confirm the slope holds.
- Beyond 100: extrapolation gets thin. A genuine cross-talk knee may emerge as the synonym tables grow into each other's vocabulary.

The architecture handled the bump cleanly. No research breakthrough needed — just the curator's hand and the existing TF-IDF + corpus-expansion pipeline.

## Reproducibility

```sh
git checkout investigation/scaling-curve-experiment
./bootstrap.sh
cd build

# Reproduce the curve at any batch by checking out that batch's commit:
git checkout 5444fa2  # Batch 1 (geometry only)
git checkout 96fe5a1  # Batch 2 (physics added)
git checkout 77ad867  # Batch 3 (statistics added)
git checkout 4b83db1  # Batch 4 (final, all 20)

# Then for any checkout:
cmake --build . --target wiring_organelle_demo manifold_tfidf_demo corpus_expand --config Release
./corpus_expand pipeline_corpus_phase4_train.txt 1234   # regenerate corpus
./manifold_tfidf_demo pipeline_corpus_scaling_heldout.txt pipeline_corpus_phase4_train.txt | tail -3
```

The corpus is regenerated deterministically (seed=1234) so the curve reproduces byte-stably.
