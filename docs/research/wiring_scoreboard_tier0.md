# Wiring scoreboard — Tier 0 baseline

Reproduced via `tools/wiring_eval_all.sh` from `build/` on commit `2c487f4` (2026-04-30).

| Eval | Score | Command |
|---|---|---|
| Phase 2c clean (anchor) | **HEADLINE (strict-verified)**: 20/20 (100%)<br>correct on all inputs: 20/20 (100%) | `./wiring_organelle_demo --clean-only` |
| Phase 3b composition | **HEADLINE (strict-verified)**: 10/10 (100%)<br>correct on all inputs: 6/10 (60%) | `./wiring_organelle_demo --composition` |
| Wiring transformer alone | **HEADLINE (strict-verified)**: 18/20 (90%)<br>correct on all inputs: 7/20 (35%) | `./wiring_organelle_demo --no-anchor --clean-only` |
| TF-IDF adversarial axis-2 | **Top-1 EXACT**: 18/20 (90%) | `./manifold_tfidf_demo pipeline_corpus_adversarial.txt pipeline_corpus_phase4_train.txt` |
| TF-IDF no-regression (Phase 2c clean) | **Top-1 EXACT**: 40/40 (100%) | `./manifold_tfidf_demo pipeline_corpus_held_out.txt pipeline_corpus_phase4_train.txt` |

## Reconciliation with §44.2

Every §44.2 headline reproduces:

| §44.2 number | Tier 0 reading | Match |
|---|---|---|
| Phase 2c clean — Anchor: 20/20 | correct on all inputs: 20/20 | ✓ |
| Phase 3b composition: 6/10 | correct on all inputs: 6/10 | ✓ |
| Wiring transformer alone: 7/20 (35%) | correct on all inputs: 7/20 | ✓ |
| TF-IDF adversarial: 18/20 | Top-1 EXACT: 18/20 | ✓ |
| TF-IDF no-regression: 20/20 | Top-1 EXACT: 40/40 (held-out file = 20 originals + 20 paraphrases) | ✓ (rate is 100%) |

Note that the per-binary `[HEADLINE]` line in the wiring binary tracks `strict-verified` (graph well-formed and passes the verifier), which can be 100% even when end-to-end correctness is lower. **§44.2's reported numbers correspond to the `correct on all 5 inputs` line (Phase 8 in the wiring binary), not the `[HEADLINE]` line.** The harness reports both so the §43.6 backlog can target the right metric.

## Tier 1 backlog (§43.6)

Diagnosis of the four failing composition prompts (§43.4 in `RESEARCH_PIPELINE_IR.md`):

| Prompt | §43.4 root cause | Fixable in Tier 1? |
|---|---|---|
| #2 "compound balance bounded between lo and hi" | Numerical edge case at small periods; fragments picked correctly | No — not a keyword/composer issue |
| #8 "compound interest as a percentage of principal" | 3-fragment chain; subtract's `y` arg should bind back to compound's `principal`, but the composer creates a fresh independent input | No — composer architectural limitation, beyond §43.6 scope |
| #9 "the discounted price after markup" | Position-based ordering picks discount→markup; "after" connective inverts the chain | **Yes** — implement `after`-connective reordering |
| #10 "fibonacci of n times factorial of n bounded by lo hi" | `multiply_step` fires alongside `fib_fact_mul_step` (which already includes multiply) — extra redundant node | Out of §43.6 scope (anti-coverage detection) |

**Tier 1 commits:**
1. Add `after`-connective handling to `wiring_compose_for_prompt()` — moves fragments mentioned after "after" to the front of the chain, preserving relative order. Targets #9 and any other "after"-clause prompt that may have been silently failing.
2. Append `divide_step`, `square_step`, `lerp_step` to the fragment table — primitives already exist in `wiring_natives.c` lines 204/208/223. No score movement expected on Tier 1's 10-prompt test (none of the prompts use these), but enables future 2-3-stage chains.
3. Skip the keyword-bag iteration (task #4) — diagnosis above shows #2 and #8 are not keyword issues.
