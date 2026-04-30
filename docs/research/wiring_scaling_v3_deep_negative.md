# #2 result — v3 with v2-depth synonym tables: 3/20 → 0/20 (HURT, not helped)

## Setup

The Phase 3 report flagged a confound: v3 (3/20) used lean synonym tables (~6 entries/family, 3 templates), v2 (16/20) used deep tables (~12 entries/family, 4-6 templates). The Phase 3 caveat said "a fair v3-with-deep-synonyms run was not done; that would test whether the v2 80% rate generalises to *new* domains at v2 depth."

#2 ran that test.

## Procedure

For each of 20 v3 families (chemistry / time / conversions / combinatorics):
- Synonym groups expanded from 2-3 → 3-4
- Synonyms per group expanded from 2-4 → 5-7
- Templates expanded from 3 → 5

Audit invariant maintained — held-out vocabulary not added to synonyms (Audit B = 0/20 ≥ 0.7 confirmed before measurement).

## Result

| Eval | Lean (3 templates, ~6 syns) | Deep (5 templates, ~12 syns) |
|---|---|---|
| **v3 retrieval (TF-IDF Top-1)** | **3/20 (15%)** | **0/20 (0%)** |
| v2 retrieval | 16/20 | 16/20 (held) |
| Phase 2c no-regression | 39/40 | 39/40 (held) |
| Adversarial axis-2 | 17/20 | **18/20** (+1) |

**The deep version made v3 worse, not better.**

## Diagnosis

The deep templates added 5 generic glue phrases per family ("%0% expressed by %1% per %2%", "the %0% formed by %1% scaled by %2%", "%0% computed as %1% scaled by %2%", "%0% expressed as %1% with %2%", "the %0% formed when %1% sits inside %2%"). Across 20 families, that's 100 new prompt patterns each containing English glue words like *expressed*, *formed*, *scaled*, *computed*, *given*, *via*.

These words now appear with high frequency across MANY family centroids → low IDF weight → no discriminative contribution. Worse, the GLUE words started showing up in the held-out paraphrases too (which use phrases like "derived from", "computed at"), so they became active centroid contributors but with the WRONG family.

Net effect: deep curation diluted distinctive vocabulary and amplified shared generic vocabulary. Family centroids collapsed toward each other.

## Implication for the original §44.5 question

The Phase 3 finding ("synonym depth matters") needs revision: **synonym depth helps only when each new synonym is *domain-distinctive*. Adding synonym variants that share generic English vocabulary across families HURTS retrieval, not helps.**

The v2 success at 80% depended on something Phase 3-deep didn't replicate: not just "more synonyms" but **"more synonyms each unique to one family."** v2's pricing/finance/math/physics families used domain-distinctive nouns (interest, factorial, gcd, fibonacci, sigmoid, momentum) that show up in few other families. v3's chemistry/time/conversions/combinatorics families used more generic concept words (count, value, total, scale, conversion) that don't naturally distinguish.

This means the §44.5 cumulative claim narrows further:

> **TF-IDF + curator's synonyms achieves ~75-80% retrieval on novel-paraphrase tests when each family's distinctive vocabulary is genuinely unique to that family. For domains where family concepts share generic English vocabulary (chemistry concentration vs molarity vs ratio; time durations vs intervals vs spans), the ceiling drops sharply — independent of curation depth.**

This is a more honest claim than either "1:1 scaling" or "75% scaling holds across domains."

## Decision

Code reverted to v3 lean (3/20 baseline). The deep version is documented here as a negative-result tier, not shipped.

## Reproducibility

```sh
# v3 lean baseline (current main):
./manifold_tfidf_demo pipeline_corpus_scaling_heldout_v3.txt pipeline_corpus_phase4_train.txt | grep Top-1
# → 3/20 (15%)

# v3 deep (negative result, see git history of investigation/post-phase3-cleanup-and-extensions):
git show <commit-of-this-doc>~1:tools/corpus_expand.c | head -1000  # or check this branch
```
