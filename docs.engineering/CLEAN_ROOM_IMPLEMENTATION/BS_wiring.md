# BS_wiring — Behaviour Specification (Wiring Organelle + Anchor Retrieval)

**Document ID:** BS-WIRE-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

Behavioural contract of the `wiring_organelle_demo` (NL → @graph) including the wiring transformer, the anchor library, the Geodesic and TF-IDF classifiers, the composition fallback, the planner organelle (Phase 15), and the CLI flag matrix that reproduces the leakage-audited honest baselines.

## 1.1 Scope of compositionality (load-bearing — read first)

The wiring layer composes novel graphs through **four** complementary mechanisms, in order of decreasing prior probability of matching a prompt:

1. **Anchor retrieval** over the 20-entry curated library (100 % on Phase 2c clean — `SLO-WIRE-001`).
2. **Fragment-chaining** over the same library for multi-stage prompts (60 % on a 10-prompt test set — `SLO-WIRE-003`).
3. **Type-directed compositional search** (V1.0.4 mechanism + V1.0.7 Phase-6b improvements: argument-to-port binder + per-port keyword manifest + `INPUT_ORDER:` annotation + manifest-driven prior + earliest-keyword-position tie-break). **V1.0.7 achieved baseline: 100 % verified, 40 % correct** on the leakage-audited 30-prompt compositional set (`SLO-WIRE-005`). Lifted from V1.0.5 30 % via the four-root-cause fix in `COMPOSITIONAL_GENERATOR_FIX_PLAN.md` v2.0; the 50 % design target persists in `GAP-WIRE-006` (PARTIALLY-RESOLVED). Per-prompt analysis in `RESEARCH_DISCLOSURE.md` §5.5.
4. **Wiring transformer fallback** when (1)–(3) miss (35 % on Phase 2c clean — `SLO-WIRE-002`).

The V1.0 corpus's "no generative compositionality" caveat is **superseded** by mechanism (3): the platform now has a path to genuinely-novel compositions through type-directed search over the primitive manifest. Honest disclosure of mechanisms (1)–(4)'s respective coverage remains in `RESEARCH_DISCLOSURE.md`.

**End-to-end execution** of a generated graph through the shipped VM is now supported via `pipeline_execute_vm` (V1.0.4 — `GAP-PIPE-003` RESOLVED). Demos / tests link `src/microgpt_pipeline_vm.c` + the VM library; the host registers natives via `vm_engine_register_fn` and the dispatcher resolves them via `vm_engine_find_fn`.

What downstream products SHOULD do (per `docs/STRATEGY_ONE_PAGER.md` §"What we are not claiming"):

- Treat anchor retrieval over a curated library as the front-line mechanism for prompts in well-curated domains.
- Curate the anchor library per vertical (the "20-family fraud anchor library" in `docs/PRODUCT_FRAUD_DETECTION.md` is the worked example).
- Use the V1.0.4 compositional search for prompts whose primitive set is in the manifest but the combination is not in any anchor.
- Plan for external semantic embeddings if the vertical needs broad-domain coverage beyond the manifest's 36 primitives (per the Post-Phase-3 #3 finding).

## 2. Type contracts

### 2.1 `WiringAnchor`

**Invariants:**
- INV-WIRE-001: An anchor SHALL carry a `family_name` (unique within the library), a `keyword_set[]` (used by classifiers), and a `graph_text` payload (a verified `@graph`).
- INV-WIRE-002: For every anchor, `pipeline_verify(pipeline_parse_text(anchor.graph_text))` MUST succeed (the anchor library is a curated set of valid graphs).

### 2.2 Geodesic classifier slot table

**Invariants:**
- INV-WIRE-010: Each anchor family is assigned a unique axis in 20-D state space (slot collisions would cap classification accuracy at 80 %).
- INV-WIRE-011: The Geodesic classifier's `GEO_DIMS` is set to 40 in this fork (was 12, then 20). The slot count SHALL be ≥ the number of held-out template families.

### 2.3 TF-IDF classifier (Phase 4)

**Invariants:**
- INV-WIRE-020: The classifier is trained on a deterministically generated corpus of ≥ 4,000 prompts (`tools/corpus_expand`).
- INV-WIRE-021: Classification is centroid cosine over TF-IDF vectors; argmax → predicted family.

## 3. Operation contracts

### 3.1 Demo execution flow

`wiring_organelle_demo` SHALL, for each held-out prompt:

1. Run the anchor classifier (Geodesic top-1 by default, TF-IDF when invoked through `manifold_tfidf_demo`).
2. Run the wiring transformer with best-of-N (typically N=16) using prefix-cache sharing.
3. For each candidate, attempt `pipeline_parse_text_tolerant` → `pipeline_repair` → `pipeline_verify`.
4. Re-rank verified candidates by:
   - Match against the planner organelle's hint family.
   - Composition fragment overlap for multi-stage prompts.
   - The "fidelity-trumps" gate (+1000) for compositions matching expected primitive sets.
5. Return the top-ranked verified candidate, or a NULL graph + error if none verify.

### 3.2 CLI flags

INV-WIRE-030: The demo SHALL recognise the flags `--clean-only`, `--no-anchor`, `--composition`, `--no-composition` and SHALL reproduce the documented honest baselines:

| Flag combination | Expected outcome |
|---|---|
| `--clean-only` (default anchor + composition) | 100 % (20/20) anchor-retrieval on the leakage-free Phase 2c set |
| `--no-anchor --clean-only` | 35 % (7/20) wiring transformer alone |
| `--composition` | 60 % (6/10) on `pipeline_corpus_composition_test.txt` |

INV-WIRE-031: `manifold_tfidf_demo` invoked with the Phase 4 expanded corpus SHALL achieve ≥ 90 % (18/20) on the adversarial axis-2 stress test and 100 % (20/20) no-regression on Phase 2c clean.

### 3.3 Native primitives

`wiring_natives.{h,c}` SHALL register ≥ 40 primitive functions consumed by the dispatch path. Each primitive's contract is its own (caller-defined) signature; the wiring layer is agnostic to the implementation.

### 3.4 Reference answers

`wiring_references.{h,c}` SHALL define a reference-answer suite for the held-out NL prompts; the demo SHALL match generated graphs against the reference set for accuracy reporting.

## 4. Invariants table

| ID | Invariant |
|---|---|
| INV-WIRE-001..002 | Anchor library is curated; every anchor verifies. |
| INV-WIRE-010..011 | Geodesic slot uniqueness; `GEO_DIMS ≥ family count`. |
| INV-WIRE-020..021 | TF-IDF classifier trained on the expanded ≥ 4,000-prompt corpus. |
| INV-WIRE-030..031 | CLI flag matrix reproduces leakage-audited baselines. |
| INV-WIRE-040 | Phase 3a-full (EKAN-Network classifier) and Phase 3c (RAG fallback) are CANCELLED per the §40.7 pre-registered skip rule (the simpler TF-IDF classifier exceeded the escalation trigger at the 408-example corpus scale). |
| INV-WIRE-041 | The wiring transformer alone is **not** load-bearing — it scores ≈ 35 % on the clean set; the anchor library is what reaches 100 %. A clean-room rebuild MUST include the anchor library to reproduce the headline. |
| INV-WIRE-042 | (V1.0.4 SUPERSEDED) Compositionality is now achieved via four mechanisms — anchor retrieval, fragment chaining, type-directed search, and wiring transformer — listed in §1.1 in order of decreasing prior probability. The previous "no generative compositionality" caveat no longer applies; mechanism (3) provides a genuine generative path. |
| INV-WIRE-043 | (V1.0.4 SUPERSEDED) Executing a generated graph end-to-end through the shipped VM IS supported via `pipeline_execute_vm` (RESOLVES `GAP-PIPE-003`). Demos / tests link `src/microgpt_pipeline_vm.c` + the VM library; the host registers natives via `vm_engine_register_fn`. |
| INV-WIRE-050 | Every graph returned by `wiring_compositional_search` SHALL pass `pipeline_verify` before being returned to the caller. The function returns NULL if no verified graph can be synthesised. |
| INV-WIRE-051 | The compositional search is deterministic — same prompt + same manifest produces the same graph. (Greedy beam=1 in V1.0.4; beam ≤ 2 with `WIRING_BEAM=2` from V1.0.6.) |
| INV-WIRE-060 | The TF-IDF retrieval ceiling is a *bag-of-features* property, not a unigram artefact. Three feature variants (unigram, word-bigram, character-trigram) MUST converge within ±1 prompt on any vocabulary-disjoint held-out test. Breaking this convergence would invalidate the calibrated 75-80 % ceiling claim and require re-measurement. |
| INV-WIRE-061 | The TF-IDF retrieval ceiling is *domain-bounded*: novel-paraphrase retrieval on families with distinctive nouns (math, physics, finance, fraud-domain vocabulary) MUST hit ≥ 75 %; on families with generic English vocabulary (chemistry concepts, time spans, conversions) it MAY drop to ≤ 20 % even with deep curation. This is a curator-vocabulary structural bound, not an implementation defect. |
| INV-WIRE-062 | A clean-room rebuild MUST run `tools/scaling_leakage_audit.sh` (Audit B Jaccard ≥ 0.7 ≤ 1/N) on every new held-out test set BEFORE reporting any retrieval number. This is the standing protection against curator self-overlap (the V1.0 leakage incident). |

## 5. Errors

The demo's exit code is non-zero if any held-out prompt fails to produce a verified graph.

## 6. Performance SLOs

Reference machine in `NFRD.md` §4. Methodology and pre-registered baselines in `RESEARCH_PIPELINE_IR.md`.

| ID | Measured target |
|---|---|
| SLO-WIRE-001 | Anchor-retrieval on Phase 2c clean: 100 % (20/20) — see `NFRD.md` §4.6 |
| SLO-WIRE-002 | Wiring transformer alone on Phase 2c clean: ≥ 35 % (7/20) — see `NFRD.md` §4.6 |
| SLO-WIRE-003 | Phase 3b composition: ≥ 60 % (6/10) — see `NFRD.md` §4.6 |
| SLO-WIRE-004 | Phase 4 TF-IDF on adversarial axis-2: ≥ 90 % (18/20), 100 % no-regression — see `NFRD.md` §4.6 |
| SLO-WIRE-005 | Compositional search on the 30-prompt leakage-audited held-out: ≥ 100 % verified (`pipeline_verify` pass), achieved baseline ≥ 30 % correct (9/30) — V1.0.5/V1.0.6 honest outcome; design target 50 % deferred to Phase 6b+ (`GAP-WIRE-006 OPEN`). |
| SLO-WIRE-006 | Compositional search verification rate (graphs that pass the verifier even if numerically wrong): ≥ 100 % (30/30) — V1.0.5. The 100 %/30 % gap is the surface area of the residual research question. |
| SLO-WIRE-007 | Compositional search reproducibility: same prompt + same manifest produces the same graph; ≥ 100 % byte-stable across runs (greedy beam=1 in V1.0.4; beam ≤ 2 from V1.0.6). |
| SLO-WIRE-008 | TF-IDF retrieval on novel-paraphrase held-out (vocabulary-disjoint, leakage-audited via `tools/scaling_leakage_audit.sh`): ≥ 75 % (15/20) on v2 lean baseline; ≥ 80 % (16/20) achieved with V1.0.7 subtractive sharpening of generic clamp vocabulary. **Calibrated ceiling, post-Phase-3:** 75-80 % under the *distinctive-noun* condition; drops to ≤ 15 % when family concepts share generic English vocabulary (chemistry / time / conversions). See `RESEARCH_DISCLOSURE.md` §5 for the three structural bounds. |
| SLO-WIRE-009 | TF-IDF no-regression on Phase 2c clean held-out + paraphrases: ≥ 38/40 (95 %) — current achieved 39/40 (98 %) per V1.0.7 measurement. |
| SLO-WIRE-010 | Bag-of-features ceiling invariance: unigram, bigram, and character-trigram TF-IDF all converge to within ±1/20 on the v2 novel-paraphrase test (16/15/15 respectively). Confirms the ~75-80 % ceiling is **model-bound to bag-of-features**, not unigram-specific. Breaking past this ceiling requires either external pretrained embeddings (`GAP-WIRE-002`, `GAP-DEP-001`) or restriction to distinctive-noun domains. |

## 7. Scenarios

### SCN-WIRE-001: Held-out prompt → anchor

User prompt "find transactions exceeding the cardholder's velocity baseline" hits the Geodesic classifier's `velocity_spike_24h` slot; the anchor's @graph is returned without ever invoking the wiring transformer.

### SCN-WIRE-002: Multi-stage composition

Prompt "compute the rolling z-score then flag when above threshold" has no single anchor; `wiring_compose_for_prompt` chains `rolling_zscore → flag_if_above` from fragments, producing a verified two-node graph.

### SCN-WIRE-003: Reproducing the honest headline

`./wiring_organelle_demo --clean-only` reports 20/20. `./wiring_organelle_demo --no-anchor --clean-only` reports 7/20. The two numbers together are the methodology's honest restated result.

## 8. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-WIRE-001 | SLO-WIRE-001 | `wiring_organelle_demo --clean-only` |
| ACC-WIRE-002 | SLO-WIRE-002 | `wiring_organelle_demo --no-anchor --clean-only` |
| ACC-WIRE-003 | SLO-WIRE-003 | `wiring_organelle_demo --composition` |
| ACC-WIRE-004 | SLO-WIRE-004 | `manifold_tfidf_demo` against Phase 4 corpus |
| ACC-WIRE-005 | SLO-WIRE-005, 006, 007 | `wiring_phase5_harness` against `pipeline_corpus_compositional_test.txt` |
| ACC-WIRE-006 | SLO-WIRE-008 | `manifold_tfidf_demo pipeline_corpus_scaling_heldout_v2.txt pipeline_corpus_phase4_train.txt` |
| ACC-WIRE-007 | SLO-WIRE-009 | `manifold_tfidf_demo pipeline_corpus_held_out.txt pipeline_corpus_phase4_train.txt` (no-regression check) |
| ACC-WIRE-008 | SLO-WIRE-010 | Same as ACC-WIRE-006 with `--bigram` and `--char-ngram` flags; results converge within ±1/20 |

## 9. Cross-references

- **TDD:** `TDD_wiring.md`
- **Pipeline IR:** `BS_pipeline_ir.md`
- **Geodesic:** `BS_geodesic.md`
- **Source:** `demos/wiring_organelle/`, `demos/manifold_classifier/`, `tools/corpus_expand.c`
- **Research log:** `docs/research/RESEARCH_PIPELINE_IR.md`, `RESEARCH_WIRING_ORGANELLE_PAPER.md`, `RESEARCH_MANIFOLD_LEARNING.md`

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
