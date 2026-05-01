# Compositional Generator — Phase 6d Plan

**Document ID:** MGC-PLAN-COMP-6D-001
**Version:** 1.0
**Status:** PRE-REGISTERED (not scheduled — opens on customer signal)
**Date:** 2026-05-01
**Predecessor:** `COMPOSITIONAL_GENERATOR_FIX_PLAN.md` v2.0 (Phase 6b/6c, COMPLETE in V1.0.8/V1.0.9)
**Tracks:** `GAP-WIRE-009` (OPEN, P2)
**Author:** Claude (clean-room corpus author)

---

## Context

V1.0.9 (Phase 6c) lifted the compositional baseline to **15/30 = 50 %**, meeting the original `SLO-WIRE-005` design goal pre-registered in `RESEARCH_DISCLOSURE.md` §3.2. The §6.3 plan set a stretch full-RESOLUTION gate at 60 % which was not reached. Per the methodology's "honest-disclosure-first, do not silently re-tune to chase the target" rule, V1.0.9 is **the published baseline**; the 50 % achievement is the work being done.

This document records what Phase 6d **would** do if a customer signal warranted opening it. It is pre-registered now so a future iteration (or a customer commitment) can pick it up without re-deriving the failure-mode analysis.

## Failure-mode analysis (from `RESEARCH_DISCLOSURE.md` §6.5)

Of the 15 wrong-answer prompts in V1.0.9, two structural patterns dominate:

### Pattern A — 3+ arity outers + duplicate inner primitives (12/15 failures)

Examples:

- prompt 2 `the maximum of x squared and y` → `[square, square, max_two]` — should be `max_two(square(x), y)` not `max_two(square(x), square(y))`. The search picks `square` for both ports of `max_two` because both port slots match `square`'s output type and the same prompt-noun analysis.
- prompt 3 `the average of doubled x and y` → `[double_val, double_val, average_two]` — same pattern. Should be `average_two(double_val(x), y)`.
- prompts 10, 11, 14, 16, 21, 23, 25, 27, 28, 29, 30 — all duplicate-inner variants.

**Root cause:** `discover_inner_picks` in `wiring_compositional_search.c` looks up the highest-scoring primitive per outer-port-input-type-match without considering whether the prompt-noun expected at THAT port is one the inner primitive can consume. Both `max_two`'s port `a` and port `b` see `square` as the best INT-output candidate, so both get `square`.

### Pattern B — Wrong number of nodes (3/15 failures)

Examples:

- prompt 19 `harmonic-sum of the absolute fibonacci of n` → `[add, harmonic_n]`. Should be `harmonic_n(abs_val(fibonacci(n)))` (3 primitives). Search picks 2.
- prompt 20 `future-value of the present-value of x at rate r over n periods` → `[divide, divide, divide, future_value]`. Should be `future_value(present_value(x, r, n), r, n)` (2 primitives, recursing). Search misses the recursion.
- prompt 25 `sigmoid of the difference of x and tripled y` → `[subtract, sigmoid]`. Missing `triple_val`. Should be `sigmoid(subtract(x, triple_val(y)))` (3 primitives).

**Root cause:** `discover_inner_picks` recurses one level deep only. Inner primitives don't get their own inner-pick pass. Anything requiring depth ≥ 2 falls through.

## Pre-registered hypotheses

- **H8 — Per-port noun-aware inner picker.** Replace `pick_best_primitive` in `discover_inner_picks` with a function that takes the OUTER's port-keyword set as a constraint, plus a "noun budget" (the prompt nouns not yet consumed by another port). Pick the inner whose own keywords + the prompt-noun-it-would-consume best match the outer port's expected noun. Predicted lift on Pattern A: 8/12 prompts → axis-1/-3 +6.
- **H9 — Recursive inner picks.** Allow `discover_inner_picks` to recurse one more level (depth=2). Gate the recursion on score: only recurse when the inner's own keywords have a score-tied OR superior alternative reachable through one more primitive. Predicted lift on Pattern B: 2/3 prompts → axis-2 +1, axis-3 +1.

## Pre-registered targets (Phase 6d)

| Source | Predicted lift | Cumulative |
|---|---:|---:|
| V1.0.9 baseline | — | 15/30 (50 %) |
| H8 per-port inner picker | +6 prompts | 21/30 (70 %) |
| H9 depth-2 recursion | +2 prompts | 23/30 (77 %) |

Aggregate target: **≥ 21/30 (70 %)**. **Failure target**: < 18/30 (60 %). Same disposition logic as §5.3 / §6.3:

- ≥ 70 %: `GAP-WIRE-005`, `GAP-WIRE-006`, `GAP-WIRE-009` → RESOLVED. SLO-WIRE-005 promoted.
- 60–69 %: PARTIALLY-RESOLVED with achieved score as new SLO baseline.
- < 60 %: H8/H9 falsified. V1.0.9 50 % baseline persists. Phase 6e (manifold learning, per `RESEARCH_MANIFOLD_LEARNING.md`) reconsidered only on a stronger customer signal.

## Implementation sketch (executable when scheduled)

### Stream H1 — per-port noun-aware inner picker

**File**: `demos/wiring_organelle/wiring_compositional_search.c`

Refactor `discover_inner_picks(manifest, n_manifest, outer_idx, prompt_lc, inner_picks_out)` into:

```c
static void discover_inner_picks_v2(
    const WiringPrimitive *manifest, int n_manifest,
    int outer_idx, const char *prompt_lc,
    const char *original_prompt,
    int *inner_picks_out,
    int *consumed_token_idx /* WIRING_PRIM_MAX_INPUTS slots */);
```

Per outer input port `ip`:
1. Determine the **expected noun** for this port — first match of any token in `prompt_lc` against `outer->port_keywords[ip]` (or fallback to `outer->input_names[ip]`), among tokens NOT already in `consumed_token_idx`.
2. Search the manifest for primitives whose:
   - output type matches `outer->input_types[ip]`,
   - keyword set has a positive score on `prompt_lc`,
   - port_keyword set or port name accepts the expected noun.
3. Highest-scoring among those is the inner. If none qualify, fall through to the current "highest-scoring INT-output" path (preserving V1.0.9 behaviour as a fallback).
4. Mark the consumed token in `consumed_token_idx` so subsequent ports don't re-consume it.

This eliminates the duplicate-inner pattern: once "x" is consumed by port 0's inner, port 1 must find a different noun (typically "y") for its inner pick.

### Stream H2 — depth-2 inner recursion

In the picked-inner loop, after each inner is selected, run a one-shot `discover_inner_picks_v2` against the *inner* primitive (with the rest of the prompt's unconsumed nouns). If the recursive pick has a strictly positive score AND its primitive's keywords appear in the prompt strictly to the right of the outer's keyword (semantic nesting), accept it as a depth-2 inner.

Ablation guard: gate via compile-time `WIRING_INNER_DEPTH=2` (default 1 to preserve V1.0.9 behaviour for ablation).

### Stream H3 — coverage scoring update

Update the coverage heuristic in `wiring_compositional_search`:

```c
coverage = score_primitive(outer, prompt_lc)
         + sum(port_kw_hits across ports)
         + sum(score_primitive(distinct inners))
         + sum(score_primitive(distinct depth-2 inners))     // H2
         + (n_consumed_tokens / total_content_tokens) * 5    // H1 reward
```

The token-coverage bonus rewards graphs that consume more of the prompt's content nouns — discouraging single-input outers that ignore most of the prompt.

## No-regression invariants

The Phase 6d changes are gated by `WIRING_INNER_DEPTH` and `WIRING_PORT_AWARE_INNER` compile-time flags, default OFF, so V1.0.9 behaviour is bit-identical when the flags are off. Pre-existing wiring numbers (anchor 100 %, fragment 60 %, transformer 35 %, TF-IDF 90 %) untouched. ctest 15/15 must persist.

## Standing leakage discipline

`tools/scaling_leakage_audit.sh` MUST be re-run on `pipeline_corpus_compositional_test.txt` after Phase 6d implementation. The held-out is unchanged in this plan; only the search algorithm changes. Zero verbatim, < 0.7 Jaccard.

## What this plan deliberately does NOT do

- **Does not train any new model.** Same constraint as Phase 6c.
- **Does not change the held-out file.** No new prompts, no annotation changes.
- **Does not reach for external embeddings.** Per the cancelled Phase 3c finding, that's `GAP-WIRE-002` and is Phase 7 scope.
- **Does not propose a Phase 6e** unless H8/H9 fail. Avoid stacking unfounded escalations.

## Trigger criteria

Phase 6d is opened when ALL of the following are true:

1. A customer or stakeholder has explicitly asked for compositional accuracy > 50 % on a vertical-relevant test set.
2. The vertical's prompts have at least 3+ arity outer primitives (otherwise H8 has nothing to fix).
3. The engineering cost (estimated 1-2 weeks for H1+H2+H3 plus ablation validation) is approved against an alternative use of the same engineering capacity.

In the absence of all three, Phase 6d stays pre-registered but not scheduled. The V1.0.9 50 % baseline is the published number.

## Cross-references

- `RESEARCH_DISCLOSURE.md` §6.5 — Phase 6c outcome and per-prompt failure analysis.
- `TRACEABILITY.md` `GAP-WIRE-009` — the gap this plan tracks.
- `BS_wiring.md` §1.1 — current scope-of-compositionality note citing this plan.
- `COMPOSITIONAL_GENERATOR_FIX_PLAN.md` — predecessor (Phases 6b/6c).
- `book.7th/Reversible_Engineering.md` Chapter 6.5 — methodology for pre-registered plans.

## Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-01 | Initial pre-registration after Phase 6c achieved the original 50 % design goal. |
