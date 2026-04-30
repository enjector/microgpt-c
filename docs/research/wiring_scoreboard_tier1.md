# Wiring scoreboard — Tier 1 (§43.6 free wins)

Reproduced via `tools/wiring_eval_all.sh` from `build/` after rebuilding `wiring_organelle_demo` with the Tier 1 changes.

## Diff vs Tier 0

| Eval | Tier 0 | Tier 1 | Δ |
|---|---|---|---|
| Phase 2c clean — strict-verified | 20/20 | 20/20 | 0 |
| Phase 2c clean — correct on all inputs | 20/20 | 20/20 | 0 |
| **Phase 3b composition — correct on all inputs** | **6/10** | **7/10** | **+1** |
| Phase 3b composition — strict-verified | 10/10 | 10/10 | 0 |
| Wiring transformer alone — correct on all inputs | 7/20 | 7/20 | 0 |
| TF-IDF adversarial axis-2 | 18/20 | 18/20 | 0 |
| TF-IDF no-regression | 40/40 | 40/40 | 0 |

**No-regression target met** (Phase 2c clean held 20/20). Phase 3b composition moved from 6/10 toward the §43.6 8-9/10 range; exactly one prompt flipped (#9), as predicted by the diagnosis in `wiring_scoreboard_tier0.md`.

## Tier 1 changes shipped

1. **`after`-connective handling in `wiring_compose_for_prompt()`** (`demos/wiring_organelle/wiring_fragments.c` +20 LoC, just before the `qsort(... cmp_by_pos_asc)` call). When the prompt contains ` after `, fragments whose first matched keyword sits to the right of " after " are pulled to the front of the chain (their `first_pos` is decremented by 100000), preserving relative order among the pulled fragments. Targets the chain-direction issue in §43.4 prompt #9 (`"the discounted price after markup"`).

2. **Three new fragments appended to `FRAGMENTS[]`**: `divide_step` (primitive `divide`, 2 args), `square_step` (primitive `square`, 1 arg), `lerp_step` (primitive `lerp`, 3 args). All three primitives already exist in `wiring_natives.c` (lines 32, 36, 55). `MAX_FRAGMENTS` bumped 16 → 24 to accommodate growth headroom (current count = 18).

## What still fails on the composition test

3 of 10 composition prompts still miss the `correct on all inputs` metric. All three are out of §43.6 Tier-1 scope (confirmed in Tier 0 diagnosis):

| # | Prompt | Root cause | Fix path |
|---|---|---|---|
| 2 | "compound balance bounded between lo and hi" | Numerical edge case at small periods (per §43.4) | Not a composer/keyword issue |
| 8 | "compound interest as a percentage of principal" | Composer creates fresh inputs for `subtract.y` and `percentage.whole` instead of binding both back to the original `principal` input | Composer architectural change (input deduplication / argument-tying); beyond §43.6 |
| 10 | "fibonacci of n times factorial of n bounded by lo hi" | `multiply_step` keyword-matches alongside `fib_fact_mul_step` (which already includes multiply), producing a redundant extra node | Anti-coverage detection: suppress overlapping fragments; beyond §43.6 |

## Tier 0 + Tier 1 status: complete

The plan in `~/.claude/plans/could-you-create-a-wild-pizza.md` called for stopping after Tier 1 to look at the data before committing to Tier 2 (library doubling). Tier 1 produced the predicted +1 Phase 3b lift with zero regressions, exhausting every §43.6 item that does not require an architectural change.

**Recommended re-decision point:** the next 1-2 prompts of composition score live behind the §43.6-out-of-scope items above (composer-architectural, not curator-bounded). Tier 2's library-doubling work would lift coverage on *new* families (axis 1) but would not move the composition score. If composition is the priority, the next tier should target the composer (#8 input-binding, #10 anti-coverage) rather than library size. If breadth is the priority, Tier 2 as planned remains the right move.
