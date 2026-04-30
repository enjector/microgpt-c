# Wiring binary Phase 8 regression — discovered during Phase 3 setup

## Summary

The wiring binary's `correct on all 5 inputs` metric (Phase 8 — end-to-end execution correctness) regressed from **20/20 → 3/20** on the `--clean-only` Phase 2c held-out somewhere between commits `1363753` (Tier 1, last verified clean) and `5178a8b` (Phase 2 subtractive sharpening, current main).

The regression has been **latent for the entire scaling-curve experiment** because the eval harness (`tools/wiring_eval_all.sh`) tracks `[HEADLINE] strict-verified` (which stayed at 20/20 — graphs are well-formed and pass the verifier) and only added `correct on all inputs` reporting at Tier 0; per-batch monitoring during Phase 1-4 didn't re-check it.

## Root cause

The 20 new families added to `wiring_anchor_graphs.c` and `wiring_geo_classifier.c` between commits `5444fa2` and `4b83db1` injected new anchor candidates into the wiring binary's vote loop. The vote loop's scoring doesn't sufficiently discriminate between same-arity candidates from different families, so an existing-family prompt now sometimes resolves to a new-family graph that *executes successfully* but on the wrong semantics.

Reproducer:
```sh
./wiring_organelle_demo --clean-only 2>&1 | sed -n '/^\[21\]/,/^\[22\]/p'
```

```
[21] // bmi of weight and height clipped to a healthy lo hi range
    EXPECTED: bmi clamp
    PLANNER:  bmi_classified
    well=Y parse=Y verify=Y fidelity=n exec=Y correct=n correct_all=n votes=16/16 cands=7
    EXEC [74 52 13 208 25]
    REF  [11 10 5 20 8]  (0/5 match)
    --- best output ---
@graph hypotenuse_squared
  : in a -> int
  : in b -> int
  ...
```

`hypotenuse_squared` (slot 24, added by Phase 1) is winning the vote over `bmi_clamped` (slot 0, original) for the BMI prompt. EXEC values [74, 52, 13, 208, 25] = `square(weight) + square(height)` for each test input vector — a successful execution of the wrong family's graph.

The same pattern repeats across most of the original 20 held-out prompts.

## Scope

- **Affects**: wiring binary `--clean-only` Phase 8 (correct on all inputs)
- **Does NOT affect**: TF-IDF demo (independent of geo classifier / anchor_graphs)
- **Does NOT affect**: `--no-anchor --clean-only` (wiring transformer alone, still 7/20)
- **Does NOT affect**: `--composition` test set (different prompts, different anchor-vote dynamics)
- **Does NOT affect**: HEADLINE strict-verified (graphs are well-formed)
- **Does NOT affect**: TF-IDF no-regression (40/40 → 39/40 from Batch 1, unchanged since)

## Status

Not addressed in this branch. Phase 3 (40-family TF-IDF expansion) deliberately routes around it by skipping anchor_graphs.c / geo_classifier.c additions for the new 20 families — TF-IDF measurement doesn't need them.

## Fix paths (for a future session)

1. **Vote-loop scoring fix**: tighten the wiring binary's candidate-scoring in `demos/wiring_organelle/main.c` to prefer the geo classifier's top-1 prediction over runners-up unless the verifier strictly rejects it. Today the loop seems to evaluate all candidates equally, letting same-arity graphs steal votes.

2. **Geo classifier top-K reduction**: pass fewer candidates from `wiring_geo_classify_topk()` (currently returns top-K with K likely ≥3-5). Reducing K to 1-2 would prevent unrelated families from entering the vote.

3. **Per-family arity-or-keyword guard**: filter anchor candidates by basic prompt-keyword overlap before voting. If no keyword from the prompt matches a candidate family's geo classifier keywords, drop it.

4. **Roll back**: remove the 20 new families from `wiring_anchor_graphs.c` and `wiring_geo_classifier.c` (keep them in `corpus_expand.c` and `FAMILY_NAMES` for TF-IDF). Restores Phase 8 = 20/20 at the cost of losing wiring binary support for the 20 new families.

Path 4 is the cheapest. Paths 1-3 are a real engineering investment in the wiring binary's vote loop.
