# Compositional Generator Fix Plan

**Document ID:** MGC-PLAN-COMP-001
**Version:** 2.0 — Phase 6b
**Status:** APPROVED (planning artefact — implementation pending)
**Date:** 2026-05-01
**Supersedes:** v1.0 (2026-04-30) — original three-stream plan, now COMPLETE in V1.0.4–V1.0.6.
**Author:** Claude (clean-room corpus author)

---

## Context

V1.0.4 (Stream A) closed `GAP-PIPE-003` — `pipeline_execute_vm` now actually runs. V1.0.4 (Stream B) closed `GAP-WIRE-005` provisionally with a deterministic type-directed search; V1.0.5 measured the honest baseline against a leakage-audited 30-prompt held-out and recorded **30/30 verified (100 %), 9/30 correct (30 %)** — below the §3.2 pre-registered 50 % target. V1.0.6 (Phase 6) implemented three pre-registered improvements (beam=2, keep-dups, geo-tie-break) and produced the **same 9/30 (30 %) total**, falsifying the simple-search hypothesis under the §4.3 disposition rule.

The platform **runs** novel compositions but does not **reliably get them right**. The Phase 5/6 measurement runs (recorded in `RESEARCH_DISCLOSURE.md` §3.5 and §4.5) localised four concrete root causes, none of which is "the search picks the wrong outer primitive". The dominant failure mode is **argument-to-port routing** — the search treats inputs as positional placeholders (`arg_0..arg_N` filled in node-add order) rather than as named prompt nouns.

This v2.0 plan attacks the four root causes head-on. The intended outcome: the V1.0.5 30 % SLO baseline lifts to **≥ 50 % verified-and-correct** on the same `pipeline_corpus_compositional_test.txt` 30-prompt held-out, **without re-tuning** the held-out, the reference oracle, or the leakage-audit script.

---

## Root causes (as observed in the harness output)

1. **R1 — Argument-to-port routing.** The search has no notion of "this prompt noun maps to this input port"; it allocates fresh `arg_0..arg_N` signature inputs in the order inner nodes are added. Example: `subtract(x, y)` then becomes `subtract(arg_0, arg_1)` regardless of whether the prompt said "x and y" or "y from x". Symptom: verified graph + numerically-wrong answer (axis 1 prompts 1, 6, 8, 9; axis 3 prompts 25, 26, 28). Dominant failure mode of the V1.0.5 / V1.0.6 30 % score.

2. **R2 — Duplicate-inner misrouting** (Phase 6 surfacing). With `WIRING_KEEP_DUPS=1`, the same primitive picked for two outer ports gets *two distinct* inner nodes, each wired to a *distinct* fresh signature input. The graph verifies, runs, and produces the wrong answer because `arg_0` and `arg_1` aren't the prompt's `x` and `y`. Visible in axis-1 prompts 1, 6, 8, 9 (each shows `[X, X, Y]` triples).

3. **R3 — Geo-classifier prior keyed on the wrong vocabulary.** `wiring_geo_classifier.c:40` `FAMILIES[]` table was tuned for the legacy Phase-13-leaked anchor families, not for the 36-primitive manifest. The +1 score-bump on family-name substring match produced zero axis-2 lift in V1.0.6.

4. **R4 — Reference-oracle input-ordering coupling.** A handful of "wrong" results are the search producing a topologically right graph whose input order doesn't match the reference's `S[0..N]` mapping. Latent under R1 — fixing R1 partially fixes R4.

---

## Approach

Three coupled work-streams, **pre-registered before any code change** under the same methodology that V1.0.4–V1.0.6 followed. Each stream is independently shippable; the real value lands when all three are on.

### Stream D — Argument-to-port binder (closes R1, partially R4)

The dominant change. Replace the positional `arg_0..arg_N` cursor with a deliberate **prompt-noun → port-name** binder that resolves prompt content words to the input ports of the search's chosen primitives.

**Mechanism:**

1. Add to `wiring_primitive_manifest.{h,c}` a per-primitive **port-noun table**: each input port gets a small set of NL nouns the prompt is likely to assign to that port. E.g. `bmi`'s port `weight` gets `{"weight", "mass", "kg"}`, port `height` gets `{"height", "cm", "tall"}`. Per-port keyword sets exist for ~half the manifest already (in the input port names themselves); the manifest just needs a `keyword_set_per_port[]` field.
2. Add to `wiring_compositional_search.c` a new `bind_args_to_ports()` pass that runs **after** the search picks its outer + inner primitives and **before** the graph is constructed. The pass walks the prompt left-to-right, identifies content nouns ("x", "y", "price", "rate"…), and assigns each to the highest-scoring still-unbound port across all selected primitives. Unbound ports fall through to the existing fresh-signature-input fallback.
3. The binder also unifies "the same prompt noun mentioned twice" — if "x" appears in two ports, both ports get the same signature input `x`, eliminating the R2 duplicate-inner misrouting structurally (same arg → same input → no aliasing).

**Files:**
- Modify: `demos/wiring_organelle/wiring_primitive_manifest.{h,c}` — add `keyword_set_per_port[]` field and populate for all 36 primitives.
- Modify: `demos/wiring_organelle/wiring_compositional_search.c` — replace `sig_cursor` accounting with the binder pass.
- New: `demos/wiring_organelle/wiring_arg_binder.{h,c}` — the binder lives in its own file (~200 LOC) so it is unit-testable in isolation.
- Tests in `tests/test_microgpt_wiring_compositional.c` — new `argbind_unique_nouns_unify`, `argbind_falls_back_to_arg_n`, `argbind_respects_per_port_keywords`.

**Pre-registered effect on R1:** the V1.0.5 axis-1 score (2/10) lifts to ≥ 5/10. The duplicate-inner misrouting failures of V1.0.6 disappear because shared-noun detection unifies the wiring at the binder pass.

### Stream E — Reference-oracle alignment (closes R4)

The held-out's `# REFERENCE: <fn>` annotation maps a prompt to a reference function `ref_<name>(const int64_t *S)` that reads inputs in *its own* convention (`S[0]=x, S[1]=y, …`). If the binder produces a different signature-input order, the harness scores against the wrong reference inputs.

**Mechanism:**

1. Extend the held-out file format with a per-prompt **input-order annotation** `# INPUT_ORDER: x,y,…`. The annotation is the noun-sequence the reference function expects on `S[]`.
2. Modify `wiring_phase5_harness.c` to read `INPUT_ORDER:` and, for each prompt, *re-shuffle* the harness's input vector so that signature input named `arg_*` corresponding to noun `x` reads `S[0]`, `y` reads `S[1]`, etc. The harness becomes a noun-keyed input mapper rather than a positional one.
3. Add `INPUT_ORDER:` to all 30 lines of `pipeline_corpus_compositional_test.txt` (one new line per prompt).

**Files:**
- Modify: `demos/wiring_organelle/pipeline_corpus_compositional_test.txt` — add 30 `# INPUT_ORDER:` lines.
- Modify: `demos/wiring_organelle/wiring_phase5_harness.c` — parse and apply the annotation.

**Pre-registered effect on R4:** any axis-* prompt that V1.0.6 verified-but-mis-scored as 0/5 because of input-order coupling now scores correctly. Estimated lift: +2 prompts (specifically axis-3 prompts 23, 28).

### Stream F — Re-tune the geo-classifier on the manifest (closes R3)

The geo-classifier table was tuned for legacy Phase-13 anchors. Re-tune it on the 36-primitive manifest: each primitive name is its own "family" and each port-noun is one of its anchor keywords. The 40-D Geodesic gives every primitive its own axis (no slot collisions; per `INV-WIRE-010`).

**Mechanism:**

1. Add a `wiring_geo_manifest_init()` that, at startup, registers each manifest entry as a Geodesic family with axis = manifest-index and anchor keywords = the union of the primitive's keyword set + per-port keywords. This replaces the static `FAMILIES[]` table at `wiring_geo_classifier.c:40` with a programmatically-built table at first use.
2. The compositional search's `WIRING_USE_GEO=1` tie-break now becomes a meaningful signal: when two manifest scores tie at the outer pick, the geo-classifier votes for the primitive whose family axis has minimum geodesic distance to the prompt's embedding.

**Files:**
- Modify: `demos/wiring_organelle/wiring_geo_classifier.{h,c}` — new init function; FAMILIES table optional (legacy fallback).
- Modify: `demos/wiring_organelle/wiring_compositional_search.c` — call `wiring_geo_manifest_init` once per process.

**Pre-registered effect on R3:** the V1.0.6 axis-2 score (4/10, unchanged from V1.0.5) lifts to ≥ 5/10 because the geo-prior now actually disambiguates primitives.

---

## Pre-registered targets (Phase 6b)

Per `RESEARCH_DISCLOSURE.md` §4.3 disposition logic, applied to the same 30-prompt held-out:

| Stream | Axis | V1.0.5 / V1.0.6 baseline | Predicted (Phase 6b) |
|---|---|---:|---:|
| D + E | 1 (novel pair) | 2/10 / 3/10 | **≥ 5/10** |
| D + E + F | 2 (synonym stress) | 4/10 / 4/10 | **≥ 5/10** |
| D | 3 (outer transform) | 3/10 / 2/10 | **≥ 5/10** |
| | **Total** | **9/30 (30 %)** | **≥ 15/30 (50 %)** |

The 50 % target is the original `SLO-WIRE-005` design goal restored. **Failure target**: < 15/30. At that point the simple-binder hypothesis is falsified and Phase 6c (learned argument binder, requires training data) is reconsidered only on customer signal.

### Disposition

- **Achieved ≥ 50 %:** `GAP-WIRE-006` → `RESOLVED`. `GAP-WIRE-005` → `RESOLVED`. NFRD SLO-WIRE-005 promoted from 30 % baseline to ≥ 50 %.
- **Achieved 40–49 %:** `GAP-WIRE-006` stays `PARTIALLY-RESOLVED` with the achieved score as the new baseline. No new gap. SLO-WIRE-005 updated.
- **Achieved < 40 %:** `GAP-WIRE-006` stays `OPEN`. V1.0.5 30 % baseline persists. Phase 6c deferred indefinitely.

The discipline of "**do not silently re-tune** the held-out, the reference oracle, or the leakage-audit script" is preserved. Stream E adds a single new annotation to the held-out (`INPUT_ORDER:`), but the prompts and reference functions remain byte-identical.

---

## Critical files

| File | Stream | Change |
|---|---|---|
| `demos/wiring_organelle/wiring_primitive_manifest.{h,c}` | D | Add `keyword_set_per_port[]` field + populate 36 entries × ports |
| `demos/wiring_organelle/wiring_arg_binder.{h,c}` | D | NEW — prompt-noun → port-name binder (~200 LOC) |
| `demos/wiring_organelle/wiring_compositional_search.c` | D + F | Replace `sig_cursor` with binder; call manifest-init once |
| `demos/wiring_organelle/wiring_geo_classifier.{h,c}` | F | Add `wiring_geo_manifest_init`; programmatic family table |
| `demos/wiring_organelle/pipeline_corpus_compositional_test.txt` | E | Add 30 `# INPUT_ORDER:` lines |
| `demos/wiring_organelle/wiring_phase5_harness.c` | E | Parse and apply `INPUT_ORDER:` |
| `tests/test_microgpt_wiring_compositional.c` | D | 3 new binder tests |
| `CMakeLists.txt` | D | Add `wiring_arg_binder.c` to harness + test target sources |
| `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md` | all | New §5 Phase 6b pre-registration; §5.5 outcome (post-run) |
| `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/BS_wiring.md` | all | §1.1 update on positive outcome; new INV-WIRE-052 (binder) |
| `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/FRD.md` | all | New REQ-WIRE-013 (binder), REQ-WIRE-014 (input-order annotation) |
| `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/NFRD.md` | all | SLO-WIRE-005 update to achieved (post-run) |
| `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/TDD_wiring.md` | D + F | New §7 binder, §8 manifest-driven geo classifier |
| `docs/engineering/CLEAN_ROOM_IMPLEMENTATION/TRACEABILITY.md` | all | Promote `GAP-WIRE-005` and `GAP-WIRE-006` per §4.3 outcome |

---

## Reuse (existing code that the new path leans on)

- `pipeline_create / _add_node / _connect / _set_signature / _verify` — entire IR construction surface unchanged.
- `pipeline_execute_vm` — unchanged; the binder produces graphs with the same shape, just with named `arg_*` slots.
- `wiring_natives.{h,c}` and the harness's VM-shim layer — unchanged.
- `wiring_geo_classifier`'s 40-D Geodesic embedder + `geo_compute_tension` — unchanged; only the family table is rebuilt.
- `wiring_reference_compute_at(name, set_idx, &out)` — unchanged; Stream E remaps inputs *before* calling, not after.
- `tools/scaling_leakage_audit.sh` — unchanged; re-run on the modified held-out as a no-op (only annotations change, not prompts).

---

## Verification plan (end-to-end)

```
# 1. Build
./bootstrap.sh

# 2. Unit tests — all must pass
ctest --test-dir build --output-on-failure
# Expect 15/15 + 3 new binder tests = 18/18 in test_microgpt_wiring_compositional

# 3. No-regression on existing wiring numbers
./build/wiring_organelle_demo --clean-only          # still 100% (20/20)
./build/wiring_organelle_demo --no-anchor --clean-only   # still 35% (7/20)
./build/wiring_organelle_demo --composition         # still 60% (6/10)

# 4. Stream E leakage no-op check
bash tools/scaling_leakage_audit.sh \
     pipeline_corpus_compositional_test.txt \
     pipeline_corpus_phase4_train.txt
# Must remain 0 verbatim leaks, max Jaccard < 0.7.
# Adding INPUT_ORDER: lines does not change leak count
# (annotation lines start with '#' and are skipped by the audit).

# 5. Phase 6b end-to-end harness
./build/wiring_phase5_harness pipeline_corpus_compositional_test.txt
# Pre-registered target: ≥ 15/30 (50%) verified-and-correct.

# 6. Ablation toggles (optional but recommended for the disclosure write-up)
./build/wiring_phase5_harness --no-binder           # Stream D off
./build/wiring_phase5_harness --no-input-order      # Stream E off
./build/wiring_phase5_harness --no-geo              # Stream F off
# Each ablation should drop ≥ 2 prompts vs the all-on combination.

# 7. Spec consistency
grep -rn "GAP-WIRE-005\|GAP-WIRE-006\|SLO-WIRE-005" \
     docs/engineering/CLEAN_ROOM_IMPLEMENTATION/
# Both gaps should reflect the §4.3 outcome consistently.
```

**Pass criteria:**

- Every `ctest` test passes.
- The three pre-existing wiring numbers do not regress.
- The leakage audit remains clean (0 verbatim, max Jaccard < 0.7).
- Phase 6b harness reaches the pre-registered ≥ 50 % target. If it doesn't, the achieved score is recorded honestly in `RESEARCH_DISCLOSURE.md` §5.5 and the §4.3 disposition rule is applied — **the score is not silently re-tuned**.
- Each ablation toggle drops ≥ 2 prompts (validates that each stream is contributing, not just one of them carrying the lift).
- `TRACEABILITY.md` revision history adds a V1.0.7 entry; gap states updated per the disposition rule.
- A regulator following BRD → BS_wiring §1.1 → SLO-WIRE-005..007 → RESEARCH_DISCLOSURE.md §5 → harness output → reference fn → source code can verify every claim end-to-end.

---

## Sequencing

Stream D first — without the binder there is nothing to test. Stream E is a 30-line edit + harness change; can land same day. Stream F third because its lift is conditional on Stream D producing different outer/inner picks at tie-break time.

If Stream D alone reaches ≥ 50 %, Streams E and F are still landed (they are not regressions and they make the failure analysis cleaner) but the pre-registered prediction is updated honestly: "Stream D was sufficient; E and F were marginal".

If the combined 6b run lands at 40–49 %, **do not re-tune**. Record the score, promote SLO-WIRE-005 to the achieved baseline, leave the gap PARTIALLY-RESOLVED. The 50 % target persists for a future Phase 6c.

If the combined 6b run lands at < 40 %, the simple-binder hypothesis is falsified. `GAP-WIRE-006` stays OPEN, V1.0.5 30 % baseline persists, Phase 6c (learned argument binder, requires training data) is reconsidered only on a customer signal that warrants the engineering investment — same disposition logic as V1.0.6.

---

## What this plan deliberately does NOT do

- **Does not replace the wiring transformer or train any new model.** The fix lives entirely in the deterministic search + binder + classifier-tuning path. A learned argument binder (Phase 6c) requires training data and is deferred.
- **Does not change the held-out file's prompts or reference functions.** Only adds annotations.
- **Does not widen the manifold-retrieval anchor library** beyond the 36 primitives currently in the manifest. Domain-vocabulary expansion is `GAP-WIRE-002` and is Phase-7 scope.
- **Does not implement the `--use-expected` ranker gate** (`SLO-WIRE-006`). The Phase 6 analysis identified routing, not ranking, as the bottleneck.
- **Does not introduce external dependencies.** Aligns with the dependency-policy gate (`GAP-DEP-001`); the binder is pure C99 + libc.

---

## Cross-references

- `TRACEABILITY.md` — `GAP-WIRE-005`, `GAP-WIRE-006` (the gaps targeted by this plan).
- `BS_wiring.md` §1.1 — current four-mechanism scope.
- `RESEARCH_DISCLOSURE.md` §3.5 (V1.0.5 baseline), §4.5 (V1.0.6 falsification), §5 (Phase 6b pre-registration to be added by this plan).
- `book.7th/Reversible_Engineering.md` Chapter 6.5 — "Findings That Become Commits" — the discipline this plan operationalises.
- Plan v1.0 source: `/Users/user/.claude/plans/wise-cooking-gadget.md` (the predecessor; its content is now COMPLETE in V1.0.4–V1.0.6).

## Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial plan; three streams (A: VM dispatch, B: search, C: held-out). All three landed in V1.0.4–V1.0.5. |
| 2.0 | 2026-05-01 | Phase 6b plan supersedes v1.0. Targets the four root causes the V1.0.5 + V1.0.6 measurement runs localised: argument-to-port routing (R1), duplicate-inner misrouting (R2), wrong-vocabulary geo prior (R3), reference-oracle ordering coupling (R4). Three streams (D: binder, E: input-order annotation, F: manifest-driven geo). Pre-registered target ≥ 50 % verified-and-correct on the same 30-prompt held-out, with the same §4.3 disposition rules. |
