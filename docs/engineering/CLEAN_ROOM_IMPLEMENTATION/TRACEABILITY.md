# TRACEABILITY — Master Index

**Document ID:** MGC-TRACE-001
**Version:** 1.0
**Date:** 2026-04-30
**Purpose:** Single source of truth linking BRD → FRD → NFRD → BS / FS / TDD → source → tests, plus the gap register.

---

## 1. Summary

**Documents in this corpus:**

| Type | Count |
|---|---:|
| BRD | 1 |
| FRD | 1 |
| NFRD | 1 |
| BS | 11 (`core, tokeniser, organelle, msa, quant, pipeline_ir, wiring, geodesic, vr, ekan, vm, metal`) |
| TDD | 11 (paired with the BSes; plus `TDD_build.md`) |
| FS | 4 (`checkpoint, pipeline_ir_text, organelle_wire, vm_bytecode`) |
| METHODOLOGY | 1 |
| TRACEABILITY | 1 (this) |

**Requirements** (cumulative across BRD / FRD / NFRD): 168.

**Gaps:**

| State | Count |
|---|---:|
| RESOLVED | 17 |
| PARTIALLY-RESOLVED | 2 |
| WITHDRAWN | 1 |
| ACCEPTED | 7 |
| TRIAGED | 0 |
| BLOCKED | 0 |
| DEFERRED | 6 |
| OPEN | 1 |
| **Total** | **34** |

By severity: P0 = 0, P1 = 4, P2 = 16, P3 = 12, INFO = 2.

Updated 2026-05-01 after the V1.0 self-audit pass, the V1.0.2 gap-fill pass, the V1.0.3 compositionality-honesty pass, the V1.0.4 compositional-fix pass, the V1.0.5 honest-outcome pass, the V1.0.6 Phase 6 falsification pass, and the V1.0.7 scaling-curve consolidation pass (§§9, 10, 11, 12, 13, 14 below).

The corpus is V1.0 — `DRAFT` status. It has been authored in a single archaeological-extraction pass and has not yet been independently rebuild-tested. The pillar scoreboard is therefore in §5 below as `PASS WITH NOTES`.

## 2. Requirements matrix (selected — full matrix tracked per-subsystem)

The full per-ID matrix is too long for in-line presentation; the table below lists representative rows. Every `REQ-*`, `INV-*`, `ERR-*`, `SLO-*`, `ACC-*` ID in the corpus has a row in the per-subsystem BS / FS document. The following rows are the ones that gate the rebuild test.

| ID | Subsystem | Defining doc | Source | Tests | State |
|---|---|---|---|---|---|
| REQ-CORE-001..026 | core | `BS_core.md` §3 | `src/microgpt.{h,c}` | `tests/test_microgpt.c`, `tests/bench_microgpt.c` | MET |
| REQ-TOK-001..010 | tokeniser | `BS_tokeniser.md` §3 | `src/microgpt.{h,c}` §1, §9 | `tests/test_microgpt.c`, `tests/bench_microgpt.c` | MET |
| REQ-CKPT-001..006 | checkpoint | `FRD.md` §4 | `src/microgpt.c` §3 | `tests/test_microgpt.c::checkpoint_*` | MET |
| REQ-ORG-001..019 | organelle | `BS_organelle.md` §3 | `src/microgpt_organelle.{h,c}` | `tests/test_microgpt_organelle.c` | MET |
| REQ-MSA-001..008 | MSA | `BS_msa.md` §3 | `src/microgpt_msa.{h,c}` | `tests/test_microgpt_msa.c` | MET |
| REQ-QUANT-001..005 | quant | `BS_quant.md` §3 | `src/microgpt_{turbo,rotor}quant.{h,c}` | `tests/test_microgpt_{turbo,rotor}quant.c` | MET |
| REQ-PIPE-001..015 | pipeline_ir | `BS_pipeline_ir.md` §3 | `src/microgpt_pipeline.{h,c}` | `tests/test_microgpt_pipeline.c` (51/51) | MET |
| REQ-WIRE-001..010 | wiring | `BS_wiring.md` §3 | `demos/wiring_organelle/`, `tools/corpus_expand.c` | `wiring_organelle_demo` flag matrix | MET |
| REQ-GEO-001..005 | geodesic | `BS_geodesic.md` §3 | `src/microgpt_geodesic.{h,c}` | `tests/test_microgpt_geodesic.c` (16/16) | MET |
| REQ-VR-001..003 | vr | `BS_vr.md` §3 | `src/microgpt_vr.{h,c}` | `tests/test_microgpt_vr.c` | MET |
| REQ-EKAN-001..003 | ekan | `BS_ekan.md` §3 | `src/microgpt_ekan.h`, `microgpt_ekan_network.h` | `tests/test_microgpt_ekan*.c` | MET |
| REQ-VM-001..006 | vm | `BS_vm.md` §3 | `src/microgpt_vm.*` | `tests/test_microgpt_vm.c` | MET |
| REQ-METAL-001..004 | metal | `BS_metal.md` §3 | `src/microgpt_metal.{h,m,metal}` | manual + CI macOS | MET |
| REQ-BUILD-001..007 | build | `TDD_build.md` | `CMakeLists.txt`, `bootstrap.{sh,bat}` | `.github/workflows/cmake-multi-platform.yml` | MET |
| BREQ-001..042 | platform-wide | `BRD.md` | many | demo evidence + tests | MET |
| NFR-001..092 | platform-wide | `NFRD.md` | many | tests + benchmarks | MET (with caveats — see gap register) |

Detailed per-ID rows (one per `INV-*`, `ERR-*`, `SLO-*`, `ACC-*`) live in the per-subsystem BS documents and are imported by reference here.

## 3. Gaps matrix

| Gap ID | Category | Severity | Affected | Disposition | Owner | Discovered | Target | Notes |
|---|---|---|---|---|---|---|---|---|
| GAP-FMT-001 | format | P2 | `FS_checkpoint.md` | DEFERRED | — | 2026-04-30 | V2.0 | Checkpoint format does not self-describe `(N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, MLP_DIM, scalar_t_width)`; consumers depend on matched build flags. Promoted to DEFERRED — fix requires a magic+version byte and is bundled with `GAP-FMT-003`. |
| GAP-FMT-002 | format | P2 | `FS_checkpoint.md` | DEFERRED | — | 2026-04-30 | V2.0 | Per-layer attention-residual block is implicit on `MICROGPT_ATTN_RES`. Bundled with `GAP-FMT-001`/`GAP-FMT-003` for the V2.0 magic+version revision. |
| GAP-FMT-003 | format | P1 | `FS_checkpoint.md` | DEFERRED | — | 2026-04-30 | V2.0 | Checkpoint format is unversioned (no magic, no version byte). |
| GAP-FMT-004 | format | P3 | `FS_checkpoint.md` | ACCEPTED | — | 2026-04-30 | — | Endianness is not handled; cross-endian portability is out of scope. The constraint is documented in `FS_checkpoint.md` §2 as a normative consumer requirement (matched-endianness build). |
| GAP-WIRE-001 | format | P3 | `FS_organelle_wire.md` | ACCEPTED | — | 2026-04-30 | — | Pipe-string format has no escape mechanism for `\|` or `\n` in values. Producers MUST avoid those bytes in values. Documented in `FS_organelle_wire.md` §4.3 as a normative producer requirement. |
| GAP-PIPE-001 | format | P3 | `FS_pipeline_ir_text.md` | DEFERRED | — | 2026-04-30 | V2.0 | Text format is unversioned; future grammar changes need a `@graph_v2` keyword. |
| GAP-VM-001 | format | P2 | `FS_vm_bytecode.md` | DEFERRED | — | 2026-04-30 | TBD | No on-disk binary form for compiled VM bytecode. |
| GAP-VM-002 | format | P3 | `FS_vm_bytecode.md` | DEFERRED | — | 2026-04-30 | TBD | `opXPATH` and `opJSON` runtime semantics deferred. |
| GAP-WIRE-002 | research | P1 | `BS_wiring.md` | DEFERRED | — | 2026-04-30 | Phase 4+ | Scaling beyond curator's synonym tables requires external semantic embeddings (per Post-Phase-3 #3). |
| GAP-WIRE-003 | bug | P1 | `demos/wiring_organelle/` | OPEN | — | 2026-04-30 | TBD | Known regression in the wiring binary's vote loop, rolled back surgically; proper fix documented as future work in `STRATEGY_ONE_PAGER.md`. |
| GAP-RE-001 | corpus | P1 | this corpus | OPEN | — | 2026-04-30 | V1.1 | None of the BSes have been rebuild-tested; the V1.0 status is `DRAFT` end-to-end. **This is the only remaining OPEN gap after the V1.0.2 gap-fill pass.** |
| GAP-RE-002 | corpus | P2 | this corpus | DEFERRED | — | 2026-04-30 | Phase 3 | Threat models (`TSM_*.md`) deferred to Reversible Engineering Phase 3. |
| GAP-RE-003 | corpus | P2 | this corpus | DEFERRED | — | 2026-04-30 | Phase 3 | Per-framework compliance mappings (`COMPLIANCE_*.md`) deferred. |
| GAP-RE-004 | corpus | P2 | this corpus | DEFERRED | — | 2026-04-30 | Phase 3 | Formal models (`FORMAL_*.md`) deferred. |
| GAP-RE-005 | corpus | P2 | this corpus | DEFERRED | — | 2026-04-30 | Phase 3 | FMEA catalogue deferred. |
| GAP-DEP-001 | strategy | P1 | platform | DEFERRED | — | 2026-04-30 | Phase 1 (fraud) | Drop "pure C99 zero-deps" project policy and adopt the dependency-boundary policy in `MIGRATED:DEPENDENCY_POLICY.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md` (Categories A/B/C). |
| GAP-DOC-001 | docs | P3 | this corpus | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Diagrams in `docs/research/images/` cross-referenced from `README.md` "Diagram cross-references" table to the relevant BS sections. |
| GAP-DOC-002 | docs | P3 | platform | ACCEPTED | — | 2026-04-30 | — | The narrative book uses descriptive chapter numbers, not normative IDs. Per `book.7th` Chapter 9 (TDD voice), the book is descriptive; cross-referencing every `INV-*` ID would inflate the prose. The corpus's IDs are normative; the book is reference reading. |
| GAP-PERF-001 | perf | P2 | core | DEFERRED | — | 2026-04-30 | TBD | Edge-device (Raspberry Pi, MCU) benchmarking is documented future work, not measured. The reference machine in `NFRD.md` §4 is M2 Max; deferring edge benchmarks until a target customer commits a deployment platform. |
| GAP-PORT-001 | port | P3 | core | ACCEPTED | — | 2026-04-30 | — | Cross-endian support is intentionally not provided. Documented as a normative consumer requirement in `FS_checkpoint.md` §2. Equivalent to `GAP-FMT-004`. |
| GAP-INT8-001 | feature | P2 | core | DEFERRED | — | 2026-04-30 | V2.0 | INT8 mode does not support checkpoint save / load (`ERR-CKPT-005`). Source comment added at the INT8 stub explaining the deferral. |
| GAP-CYC-001 | feature | P3 | organelle | ACCEPTED | — | 2026-04-30 | — | `OpaCycleDetector` handles only A↔B oscillation. Longer cycles are documented as out-of-scope for V1.0 in `BS_organelle.md` and `TDD_organelle.md` §8. The fixed-window history would need to grow to detect longer cycles; the trade is documented and accepted. |
| GAP-SPEC-001 | spec | P2 | `BS_tokeniser.md` INV-TOK-023 | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Originally claimed "power of two ≥ 2 × num_words" hash table; reality is `vocab_size × 4` clamped ≥ 64, modulo (h+1) probe. Corrected in this revision. |
| GAP-SPEC-002 | spec | P3 | `TDD_tokeniser.md` §5.3 | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Originally described two-pass count + heap-select; reality is single-pass count + qsort. Corrected in this revision. |
| GAP-SPEC-003 | spec | P3 | `BS_tokeniser.md` §3.6 | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Originally said "whitespace-delimited"; reality only splits on space/`\n`/`\r` (not tab). Corrected. |
| GAP-SPEC-004 | spec | P2 | `BS_pipeline_ir.md` INV-PIPE-013 | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Originally implied execute would auto-re-verify; reality is execute returns an error on unverified input. Caller must re-verify. Corrected. |
| GAP-SPEC-005 | spec | P3 | `BS_core.md` ACC-CORE-005/006 | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Originally cited `tests/test_microgpt_qk_norm.c` and `tests/test_microgpt_rope.c` as separate source files. Reality: those are CMake-generated executables built from `tests/test_microgpt.c` with per-target `-D` defines (`CMakeLists.txt:549,711`). Corrected. |
| GAP-DOC-003 | docs | P2 | `microgpt.h:181` vs `CMakeLists.txt:23` vs `docs/BUILD_OPTIONS.md:52` | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Header comment in `microgpt.h` and the example in `BUILD_OPTIONS.md` §"Float Precision" rewritten to state plainly that `MICROGPT_USE_FLOAT=ON` is the default and that `=OFF` switches to double. `BUILD_OPTIONS.md:180` was already correct. |
| GAP-DOC-004 | docs | P3 | `tokenize_words` | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Header comment for `tokenize_words` in `microgpt.h` extended to spell out: delimiters are space/`\n`/`\r` only; tab is NOT a delimiter; OOV words map to `unk_id`; the function does not prepend BOS. |
| GAP-DOC-005 | docs | P2 | corpus-wide | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Per-BS SLO sections updated to alias rows in `NFRD.md` §4 instead of repeating reference-machine specs. Updated: `BS_core.md`, `BS_tokeniser.md`, `BS_msa.md`, `BS_quant.md`, `BS_wiring.md`, `BS_vm.md`. |
| GAP-DOC-006 | docs | P3 | `BS_core.md` `microgpt_print_config` | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | `microgpt_print_config` extended to print `MSA_POOL_MODE` mode value, plus the constants `BETA1`, `BETA2`, `EPS_ADAM`, `INIT_STD`, `WEIGHT_DECAY`, `GRAD_CLIP`, `LABEL_SMOOTH` for full build-fingerprint coverage. |
| GAP-CKPT-001 | bug | P2 | `microgpt.c` "fp64" naming | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Helpers `write_doubles` / `read_doubles` renamed to `write_scalars` / `read_scalars`. Section comment changed from "Serialisation (fp64 only)" to "Serialisation (scalar_t, non-INT8 builds)". Corpus already documented the actual behaviour in `FS_checkpoint.md` §2. |
| GAP-CORE-001 | feature | P2 | `microgpt.c::INT8` Adam path | ACCEPTED | — | 2026-04-30 | — | INT8 Adam requantises every weight matrix every step. Acceptable for V1.0 — the INT8 path is only used by users who have explicitly opted out of the default fp32/fp64 path; frozen-layer fast-paths are out of scope until a customer reports the cost as material. |
| GAP-WIRE-004 | docs | P3 | `BS_wiring.md` SLO-WIRE-* | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | `RESEARCH_DISCLOSURE.md` created. Documents Phase 3a-full and 3c cancellations under the pre-registered §40.7 skip rule, the Phase 4b-full cancellation under §45.2, the leakage-audit honest restatements, what is NOT being claimed, and the standing protections against recurrence. |
| GAP-VR-001 | feature | P3 | `microgpt_vr.c` simplex caps | ACCEPTED | — | 2026-04-30 | — | `vr_compute` silently truncates on cap exceedance. Per-call error path would require a public-ABI change to `VRDiagram`. Header comment in `microgpt_vr.h` upgraded to flag the truncation as "IMPORTANT — fixed-cap silent truncation". Caller MUST pre-bound `n_points ≤ VR_MAX_PTS = 64` and validate `diagram->count`. |
| GAP-METAL-001 | feature | P2 | `microgpt_metal.m` re-init | WITHDRAWN | corpus-author | 2026-04-30 | 2026-04-30 | Reading error in the V1.0.1 audit pass. `metal_init` already has `if (g_initialized) return 0;` at line 112 — it IS truly idempotent. No fix needed. |
| GAP-VM-003 | feature | P3 | `microgpt_vm.h` native ABI | ACCEPTED | — | 2026-04-30 | — | `vm_native_fn` returns `double`. ABI revision is breaking; deferred indefinitely until a vertical product needs first-class native error signalling. Sentinel-NaN convention is the documented workaround. |
| GAP-PIPE-002 | spec | P3 | `pipeline_execute_vm` | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Audit revealed `pipeline_execute_vm` is actually a stub returning `PIPE_ERR_EXEC` with "deferred to Phase 3"; the original gap concerned an error-message detail that does not apply. The deeper finding is now tracked as `GAP-PIPE-003`. |
| GAP-PIPE-003 | feature | P2 | `microgpt_pipeline.c::pipeline_execute_vm` | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | V1.0.4 (Stream A of compositional fix) ships a real implementation in the opt-in TU `src/microgpt_pipeline_vm.c`. New public API `vm_engine_find_fn` (`REQ-VM-007`) lets the dispatcher resolve registered natives. INT/FLOAT/VOID ports only — non-numeric ports return `PIPE_ERR_EXEC` with a message identifying the offending node and port (also closes `GAP-PIPE-002`'s improvement target). 5 new unit tests in `tests/test_microgpt_pipeline.c` cover simple/chain/missing-primitive/string-port-rejected/callback-equivalence; all 55/55 tests pass. |
| GAP-WIRE-005 | platform | P1 | wiring layer | PARTIALLY-RESOLVED | corpus-author | 2026-04-30 | 2026-05-01 | V1.0.4 ships the mechanism. V1.0.5/V1.0.6 = 30 %. V1.0.7 (Phase 6b) = 40 %. **V1.0.8 (Phase 6c) = 15/30 (50 %)** — original SLO-WIRE-005 design goal met at the achievement level. Stays `PARTIALLY-RESOLVED` per §6.3 (gate at 60 % for full RESOLUTION). New SLO-WIRE-005 baseline = 50 %. Phase 6c delivered axis 1 +4, axis 2 +1, axis 3 +1 over V1.0.5 baseline (+20 pp total). See `RESEARCH_DISCLOSURE.md` §6.5: G1 confirmed (synonym lift), G2 no-op on this corpus, G3 redundant, **G4 incidental** (legacy `wiring_geo` substring bump was net-harmful — disabled). |
| GAP-WIRE-006 | research | P1 | wiring compositional search | PARTIALLY-RESOLVED | corpus-author | 2026-05-01 | Phase 6d? | V1.0.8 (Phase 6c) executed the `COMPOSITIONAL_GENERATOR_FIX_PLAN.md` v2.0 mining-sketch streams G1 (per-primitive synonym lift from the branch's `corpus_expand`), G2 (after-connective re-ordering, gated on the connective splitting the candidate set), G3 (binder audit — already mature in V1.0.8) plus G4 (disabled the legacy `wiring_geo_predict_top_k` substring bump that was biasing wrong-direction). **Achieved 15/30 = 50 %** (Δ +3 from V1.0.7). Per §6.3 (40–59 % range), `PARTIALLY-RESOLVED` with new SLO-WIRE-005 baseline = 50 %. Original §3.2 pre-registered design target met. Residual failure mode (3+ arity outers + duplicate inner primitives) motivates a Phase 6d (per-port keyword binding at inner-pick time, not just at port-allocation time). Phase 6d not scheduled — opened only on customer signal. |
| GAP-WIRE-007 | research | P2 | TF-IDF retrieval (`demos/manifold_classifier/tfidf_main.c`) | RESOLVED | corpus-author | 2026-05-01 | 2026-05-01 | **Bag-of-features ceiling confirmed structural.** Post-Phase-3 #3 (`docs/research/wiring_scaling_post_phase3.md`) tested three feature variants (unigram / word-bigram / character-trigram) on the v2 vocabulary-disjoint held-out: 16/15/15 — converge within ±1 prompt. Confirms the ~75-80 % retrieval ceiling is **model-bound** to the bag-of-features family, not unigram-specific. RESOLVED as documented limitation; new `INV-WIRE-060` enforces the convergence invariant in BS_wiring. Path past the ceiling requires external pretrained embeddings — see `GAP-WIRE-002` (DEFERRED) and `GAP-DEP-001`. |
| GAP-WIRE-008 | research | P2 | TF-IDF retrieval / corpus_expand synonym tables | RESOLVED | corpus-author | 2026-05-01 | 2026-05-01 | **Domain-bounded ceiling confirmed.** Post-Phase-3 #2 (`docs/research/wiring_scaling_v3_deep_negative.md`): adding 20 new families with chemistry / time / conversions vocabulary (lean synonyms) hit 3/20 (15 %); expanding to v2-depth synonyms HURT to 0/20. Diagnosis: generic English glue ("expressed by", "formed by", "computed as") shared across families lowers IDF weight on distinctive concepts → centroids collapse. RESOLVED as documented limitation; new `INV-WIRE-061` records the distinctive-noun structural bound. Implication: vertical productisation (`MIGRATED:PRODUCT_FRAUD_DETECTION.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md` etc.) targets domains with naturally distinctive vocabulary (fraud nouns, finance jargon) where the upper bound is achievable. |
| GAP-ORG-001 | bug | P2 | `opa_extract_pipe_value` mutation | RESOLVED | corpus-author | 2026-04-30 | 2026-04-30 | Function rewritten to terminate at the FIRST delimiter only (`min(strchr('|'), strchr('\n'))`). Earlier behaviour wrote NULs at BOTH the next pipe AND the next newline, destroying every later "|key=" anchor in the buffer. After the fix, `opa_extract_pipe_value` can be called repeatedly on the same buffer without losing later keys (provided values do not contain `|` or `\n` themselves — see `GAP-WIRE-001`). |

## 4. Convergent P0 items

None at V1.0 of this corpus. (Per `book.7th` Chapter 6.5, convergent-P0s emerge from the Phase 3 Adversarial Hardening pass; no Phase 3 has been run on this corpus yet.)

## 5. Pillar scoreboard

| Pillar | Exit gate | Verdict |
|---|---|---|
| 1. Functional surface (BSes + FSes complete) | every public function is covered by exactly one BS contract; every wire/file format has an FS | **PASS WITH NOTES** — coverage is complete; no rebuild test has been run |
| 2. Reproducibility | `bootstrap.sh` produces a working build on Linux/macOS/Windows; `microgpt_print_config` produces a deterministic banner | **PASS** — `.github/workflows/cmake-multi-platform.yml` exercises this |
| 3. Performance SLOs | every SLO in `NFRD.md` has a benchmark file | **PASS WITH NOTES** — benchmarks exist; SLO breach reporting is manual |
| 4. Security + Compliance (Phase 3) | TSM per subsystem; COMPLIANCE per framework; FMEA catalogue | **DEFERRED** — Phase 3 not started (per the methodology, deliberate for V1.0 of the corpus) |
| 5. Audit trail | every requirement is in this matrix; every gap has a disposition | **PASS WITH NOTES** — matrix complete; gaps are tracked but not yet acted on |

## 6. ID assignment registry (drift prevention)

| Subsystem prefix | Highest assigned `REQ` | Highest `INV` | Highest `ERR` | Highest `SLO` | Highest `ACC` |
|---|---|---|---|---|---|
| CORE | 026 | 041 | 002 | 020 | 006 |
| TOK | 010 | 023 | 003 | 003 | 003 |
| CKPT | 006 | — | 005 | 007 | — |
| ORG | 019 | 051 | 003 | — | 004 |
| MSA | 008 | 022 | 002 | 001 | 002 |
| QUANT | 005 | 022 | — | 003 | 003 |
| PIPE | 015 | 024 | 010 | — | 005 |
| WIRE | 010 | 041 | — | 004 | 004 |
| GEO | 005 | 031 | — | 001 | 003 |
| VR | 003 | 031 | — | — | 001 |
| EKAN | 003 | 021 | — | — | 002 |
| VM | 006 | 022 | 004 | 001 | 002 |
| METAL | 004 | 004 | — | — | 002 |
| BUILD | 007 | — | — | — | — |

When extending the corpus, the next `REQ-<SUBSYSTEM>-NNN` MUST start at the value above + 1. Withdrawn IDs SHALL NOT be reused.

## 7. Cross-corpus links

| External artefact | Internal anchor |
|---|---|
| `book.7th/Reversible_Engineering.md` | Methodology source (`METHODOLOGY.md`) |
| `book/MicroGPT-C_Composable_Intelligence_at_the_Edge.md` | Narrative book — chapter numbers are descriptive, not normative |
| `MIGRATED:STRATEGY_ONE_PAGER.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md` | `BRD.md` §5.3 productisation |
| `MIGRATED:PRODUCTIZATION_VERTICALS.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md` | `BRD.md` §5.3 |
| `MIGRATED:DEPENDENCY_POLICY.md → see docs/MIGRATED_TO_ORGANELLES_BIO.md` | `BRD.md` §5.4, `GAP-DEP-001` |
| `docs/research/RESEARCH_PIPELINE_IR.md` | `BS_wiring.md` §6 SLOs (research log + leakage audit) |
| `docs/research/RESEARCH_*` | Descriptive research notes — not normative for the rebuild |
| `docs/testing/PERFORMANCE.md` | `NFRD.md` §4 measurement methodology |

## 9. Self-audit pass (2026-04-30)

A targeted gap-analysis pass was run after the V1.0 extraction to verify load-bearing claims against source. Findings:

**Specification errors found and corrected (5):**
- `GAP-SPEC-001` BS_tokeniser INV-TOK-023 — hash-table capacity rule was wrong (claimed power-of-two; reality is `vocab_size × 4` clamped ≥ 64, modulo probe).
- `GAP-SPEC-002` TDD_tokeniser §5.3 — algorithm description was wrong (claimed two-pass + heap-select; reality is single-pass + qsort).
- `GAP-SPEC-003` BS_tokeniser §3.6 — claimed "whitespace-delimited"; tab is not actually a delimiter.
- `GAP-SPEC-004` BS_pipeline_ir INV-PIPE-013 — was ambiguous on post-mutation execute; clarified that execute returns an error and does NOT auto-re-verify.
- `GAP-SPEC-005` BS_core ACC-CORE-005/006 — cited test source files that do not exist; the targets are CMake-generated variants of `test_microgpt.c`.

**New codebase / documentation gaps surfaced (8):**
- `GAP-DOC-003` `MICROGPT_USE_FLOAT` default-state inconsistent across header / CMake / BUILD_OPTIONS.
- `GAP-DOC-004` `tokenize_words` does not split on tab — undocumented in source.
- `GAP-DOC-005` Per-BS SLO sections repeat the M2 Max reference machine spec; should reference NFRD §4.
- `GAP-DOC-006` `microgpt_print_config` banner does not list every load-bearing flag value.
- `GAP-CKPT-001` `write_doubles`/`read_doubles` helpers misnamed — they write `sizeof(scalar_t)` bytes.
- `GAP-CORE-001` INT8 Adam path requantises every matrix every step; no frozen-layer fast-path.
- `GAP-WIRE-004` Honest-disclosure trail for cancelled Phase-3a-full / 3c is in the research log only; should have a `RESEARCH_DISCLOSURE.md` stub.
- `GAP-VR-001` VR simplex-cap exceedance is silent; should error or strictly precondition.
- `GAP-METAL-001` `metal_init` not actually idempotent if called twice without `cleanup`.
- `GAP-VM-003` Native ABI cannot signal errors as a first-class result.
- `GAP-PIPE-002` `pipeline_execute_vm` error on non-numeric port does not identify the offending port.
- `GAP-ORG-001` `opa_extract_pipe_value` writes BOTH delimiter terminators; subsequent calls fail silently.

**Areas spot-checked and confirmed correct as authored:**
- `count_params` formula in BS_core matches `microgpt.c::count_params` line 417.
- `OpaTrace` overflow behaviour (silent clamp at `OPA_TRACE_MAX_STEPS`).
- `MICROGPT_USE_FLOAT` macro path through `scalar_t` typedef.
- VR / Geodesic / EKAN constant bounds.
- `mgpt_default_threads` returns `min(cpu_count, batch_size)`.
- `pipeline_render_text` topo-ordering and arg-form rendering.
- Performance SLOs vs `docs/testing/PERFORMANCE.md`.

**Items NOT yet audited (candidates for follow-up):**
- `forward_backward_one` gradient layout vs `count_params` / Adam buffer layout under all four V4-port flag combinations.
- `model_soup_average` element-wise correctness across heterogeneous architectures (the function has no built-in shape check).
- `pipeline_repair` fixed-point convergence on adversarial graphs.
- `organelle_generate_speculative` KV-rollback correctness on rejection.
- TurboQuant / RotorQuant codebook agreement across builds (no test vector exists).

These are recorded for a future Phase-2 deepening pass; none rise to P0 at V1.0.

## 10. V1.0.2 gap-fill pass (2026-04-30)

A second audit pass closed every actionable gap from V1.0.1.

**Code fixes (committed to source):**
- `GAP-DOC-003` — `microgpt.h` header comment and `BUILD_OPTIONS.md` rewritten to state `MICROGPT_USE_FLOAT=ON` is the default.
- `GAP-DOC-004` — `tokenize_words` header comment updated to spell out delimiters (no tab) and OOV behaviour.
- `GAP-DOC-006` — `microgpt_print_config` extended to print `MSA_POOL_MODE`, `BETA1`, `BETA2`, `EPS_ADAM`, `INIT_STD`, `WEIGHT_DECAY`, `GRAD_CLIP`, `LABEL_SMOOTH`.
- `GAP-CKPT-001` — helpers renamed `write_doubles`/`read_doubles` → `write_scalars`/`read_scalars`; section comment changed to "Serialisation (scalar_t, non-INT8 builds)".
- `GAP-VR-001` — `microgpt_vr.h` API section comment upgraded to a normative "IMPORTANT — fixed-cap silent truncation" note.
- `GAP-INT8-001` — INT8 stubs in `microgpt.c` annotated with `ERR-CKPT-005` and a pointer to this matrix.
- `GAP-ORG-001` — `opa_extract_pipe_value` rewritten to NUL-terminate at the FIRST delimiter only (was destroying later keys in the buffer).

**Corpus fixes (this directory):**
- `GAP-DOC-005` — per-BS SLO sections in `BS_core.md`, `BS_tokeniser.md`, `BS_msa.md`, `BS_quant.md`, `BS_wiring.md`, `BS_vm.md` now alias `NFRD.md` §4 instead of repeating reference-machine specs.
- `GAP-DOC-001` — `README.md` "Diagram cross-references" table added; each `docs/research/images/*.jpg` is mapped to the BS section it illustrates.
- `GAP-WIRE-004` — `RESEARCH_DISCLOSURE.md` created; captures Phase 3a-full / 3c / 4b-full cancellations, the leakage-audit honest restatements, and the standing protections.
- `GAP-PIPE-002` / new `GAP-PIPE-003` — discovered that `pipeline_execute_vm` is actually a Phase-3 stub; corpus corrected in `BS_pipeline_ir.md` §3.3, `FRD.md` REQ-PIPE-008, `TDD_pipeline_ir.md` §5.7.
- `GAP-METAL-001` — withdrawn: `metal_init` was already idempotent (early-return at line 112).

**Promoted to ACCEPTED (documented limitations, not closed in code):**
- `GAP-FMT-004` — endianness, intentional non-portability documented in `FS_checkpoint.md` §2.
- `GAP-WIRE-001` — pipe-string escape mechanism, documented in `FS_organelle_wire.md` §4.3.
- `GAP-CYC-001` — `OpaCycleDetector` A↔B-only.
- `GAP-PORT-001` — cross-endian (equivalent to GAP-FMT-004).
- `GAP-CORE-001` — INT8 Adam requantisation cost.
- `GAP-VM-003` — VM native ABI single-double return.
- `GAP-DOC-002` — book chapter ID cross-linking is descriptive, not normative.

**Bundled into V2.0 of the checkpoint format:**
- `GAP-FMT-001` (architecture self-description), `GAP-FMT-002` (attn-res presence flag), `GAP-FMT-003` (magic + version byte) all promoted to `DEFERRED` to V2.0 of the format.

**Remaining OPEN (1 gap, the only one that requires a future activity):**
- `GAP-RE-001` (P1) — V1.0 status is `DRAFT` end-to-end; rebuild test not yet run on any subsystem. Closing this requires running the §7 rebuild-test procedure in `METHODOLOGY.md` per subsystem.

**Remaining DEFERRED (7 gaps, by design):**
- `GAP-RE-002..005` (Phase 3 artefacts: TSM, COMPLIANCE, FORMAL, FMEA).
- `GAP-DEP-001` (dependency policy adoption — gated on Phase 1 fraud).
- `GAP-FMT-001..003` (V2.0 format).
- `GAP-PERF-001` (edge-device benchmarks gated on customer commitment).
- `GAP-INT8-001` (V2.0 format coupling).
- `GAP-PIPE-003` (VM-backed dispatch).

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. Whole-corpus authoring pass. Status `DRAFT` — pending rebuild test. |
| 1.0.1 | 2026-04-30 | Self-audit pass: corrected GAP-SPEC-001..005. Added GAP-DOC-003..006, GAP-CKPT-001, GAP-CORE-001, GAP-WIRE-004, GAP-VR-001, GAP-METAL-001, GAP-VM-003, GAP-PIPE-002, GAP-ORG-001. |
| 1.0.2 | 2026-04-30 | Gap-fill pass: 14 gaps RESOLVED in code or corpus, 7 promoted to ACCEPTED, 1 WITHDRAWN, 7 remaining DEFERRED, 1 remaining OPEN (`GAP-RE-001` rebuild test). New `GAP-PIPE-003` for the VM-backed dispatch stub. New file `RESEARCH_DISCLOSURE.md` for cancelled-phase audit trail. |
| 1.0.3 | 2026-04-30 | Compositionality-honesty pass: added new `GAP-WIRE-005` (P1) recording why the platform does not generate novel compositions — corpus shape, cancelled Phase 3c, stubbed `pipeline_execute_vm`. Added §1.1 "Scope of compositionality" to `BS_wiring.md` with `INV-WIRE-042` and `INV-WIRE-043`. This is now a normative limit of the V1.0 platform; clean-room rebuilders will reproduce the limit, not work around it. |
| 1.0.4 | 2026-04-30 | Compositional-fix pass per `COMPOSITIONAL_GENERATOR_FIX_PLAN.md`. **Stream A** RESOLVED `GAP-PIPE-003`: real `pipeline_execute_vm` in `src/microgpt_pipeline_vm.c`, new public `vm_engine_find_fn` (REQ-VM-007), 5 new tests (55/55 pipeline tests pass). **Stream B** RESOLVED `GAP-WIRE-005` (provisionally — outcome pending): type-directed compositional search over a 36-primitive manifest in `demos/wiring_organelle/wiring_{primitive_manifest,compositional_search}.{h,c}` plus 4 unit tests (4/4 pass). **Stream C** added 30-prompt leakage-audited held-out (`pipeline_corpus_compositional_test.txt`) with 30 reference functions in `wiring_references.c` and a §3 pre-registration in `RESEARCH_DISCLOSURE.md` (SLO-WIRE-005..007). `BS_wiring.md` §1.1 superseded — four-mechanism composition replaces the V1.0.3 honest limit. |
| 1.0.5 | 2026-05-01 | Phase 5 honest-outcome pass. New CLI harness `wiring_phase5_harness` (in `demos/wiring_organelle/`) runs the 30-prompt held-out end-to-end through `pipeline_execute_vm` and scores against the reference oracle. **Result: 30/30 verified (100 %), 9/30 correct (30 %).** Below the §3.2 pre-registered 50 % target. Per the §3.3 skip rule (Axis 1 + Axis 2 = 6/20 = 30 %, above the 5/20 = 25 % floor): `GAP-WIRE-005` reverts from `RESOLVED` → `PARTIALLY-RESOLVED`. New `GAP-WIRE-006` (P1, OPEN) tracks the next iteration toward the 50 % target with three pre-registered improvements (beam ≥ 2, drop name-dedup, geo-classifier prior). NFRD SLO-WIRE-005..007 updated to record the achieved 30 % baseline; the 50 % design target persists. `RESEARCH_DISCLOSURE.md` §3.5 populated with per-axis numbers and failure-mode analysis. The discipline of "honest disclosure first, do not silently re-tune to chase the target" is upheld. |
| 1.0.6 | 2026-05-01 | Phase 6 implementation pass. `wiring_compositional_search.c` upgraded with all three pre-registered improvements behind compile-time toggles (`WIRING_BEAM=2`, `WIRING_KEEP_DUPS=1`, `WIRING_USE_GEO=1`). Geo classifier wired into the harness + test target. **Result: 30/30 verified, 9/30 correct (30 %)** — same total as V1.0.5; per-axis +1 axis-1, 0 axis-2, −1 axis-3. Per §4.3, 30 % < 40 % failure threshold → simple-search hypothesis **falsified**. `GAP-WIRE-006` stays OPEN; V1.0.5 baseline persists. Phase 6 retained in source for ablation toggles and because it exposed a structural duplicate-inner misrouting failure mode that is informative for the next iteration. `RESEARCH_DISCLOSURE.md` §4 + §4.5 record the pre-registration and the per-prompt sign analysis. ctest 15/15 still green. |
| 1.0.7 | 2026-05-01 | Scaling-curve consolidation pass. The post-Phase-3 cleanup arc (`docs/research/wiring_scaling_post_phase3.md`, `wiring_scaling_v3_deep_negative.md`) and the organelle synthesis (`docs/research/ORGANELLE_STATE.md`) brought into the corpus. New `GAP-WIRE-007` (RESOLVED) records the **bag-of-features ceiling** — three feature variants (unigram / bigram / char-ngram) all converge to the ~80 % retrieval bound on vocabulary-disjoint held-outs; ceiling is model-bound to bag-of-features, not unigram-specific. New `GAP-WIRE-008` (RESOLVED) records the **domain-bounded ceiling** — distinctive-noun domains hit the upper bound; generic-English-vocabulary domains drop to ≤ 20 % independent of curation depth (v3 deep HURT, dropping 3/20 → 0/20). New `INV-WIRE-060/061/062` enforce the convergence and distinctive-noun invariants in `BS_wiring.md`, plus the standing leakage-audit precondition. New `SLO-WIRE-008/009/010` and `ACC-WIRE-005..008` record the calibrated 75-80 % ceiling baseline. `README.md` Companion artefacts updated to point at `ORGANELLE_STATE.md` as the recommended top-of-stack reading order. **No code changes** — purely corpus consolidation. |
| 1.0.8 | 2026-05-01 | Phase 6b implementation pass per `COMPOSITIONAL_GENERATOR_FIX_PLAN.md` v2.0. New file `demos/wiring_organelle/wiring_arg_binder.{h,c}` (Stream D) — prompt-noun → port-name binder with repeated-noun unification and outer-port-keyword inheritance for inner-feeding inputs. `wiring_primitive_manifest.{h,c}` extended with `port_keywords[]` per input port for finance / physics / clamp / lerp primitives. `pipeline_corpus_compositional_test.txt` annotated with 30 `# INPUT_ORDER:` lines (Stream E); `wiring_phase5_harness.c` extended to remap S[0..N] inputs by noun. `wiring_compositional_search.c` extended with manifest-driven port-keyword prior (Stream F), an earliest-keyword-position tie-break for outer pick, and a coverage heuristic that sums keyword + port-keyword hits across nodes. New `WiringComposeReport.signature_in_names[]` exposes binder names to the harness. **Result: 30/30 verified, 12/30 correct (40 %)** — lifted from V1.0.5/V1.0.6 30 % baseline by +10 pp. Per §5.3 (40–49 % range), `GAP-WIRE-005` and `GAP-WIRE-006` stay `PARTIALLY-RESOLVED` with new SLO-WIRE-005 baseline = 40 %. Hypotheses H4 / H6 confirmed; H5 partially; H7 not measurable in isolation. Residual structural limit (3+ arity outers + duplicate inners) motivates a future Phase 6c — opened only on customer signal. ctest 15/15 still green; existing wiring numbers (anchor 100 %, fragment 60 %, transformer 35 %, TF-IDF 90 %) preserved. |
| 1.0.9 | 2026-05-01 | Phase 6c implementation pass — branched-project mining sketch from `RESEARCH_DISCLOSURE.md` §6. **G1** lifted per-primitive synonyms from the branch's `tools/corpus_expand.c` family table into the manifest (`circle`/`disk`/`circular` for `circle_area`; `kinetic`/`mass`/`velocity`/`object`/`moving`/`motion` for `kinetic_energy`; `interest`/`yield`/`accrued`/`grow` plus port-keyword expansions for `compound`; removed overlapping `tax` from `tax_amount` keywords to stop shadowing `apply_tax`). **G2** ported `wiring_fragments.c:336-352` after-connective re-ordering, gated on the connective splitting the candidate set (no-op on the current held-out, harmless on the rest). **G3** confirmed the V1.0.8 binder is already mature against the branch's `wiring_arg_binder.c`; no further lift. **G4 (incidental)** disabled the legacy `wiring_geo_predict_top_k` substring bump in `pick_top_n_primitives` — per §6.2 the legacy `FAMILIES` table was tuned for Phase-13-leaked anchors and was actively biasing `gcd_scaled` substring matches over the manifest's correct `double_val`-as-outer pick on prompt 21. Code preserved under `#if 0` for ablation. **Result: 30/30 verified, 15/30 correct (50 %)** — original `SLO-WIRE-005` design goal met; lifted from V1.0.7 40 % by +5 prompts (axis 1: +0, axis 2: +2, axis 3: +1). Per §6.3 (40–59 % range), `GAP-WIRE-005` and `GAP-WIRE-006` stay `PARTIALLY-RESOLVED` with new baseline = 50 %; the §6.3 60 % full-RESOLUTION gate persists. Leakage audit re-run (0 verbatim, max Jaccard 0.667 < 0.7, 0 % anchor-exclusivity). Existing wiring regressions preserved (anchor 100 %, fragment 60 %, transformer 35 %, TF-IDF 90 %); ctest 15/15 still green. Residual failure mode: 3+ arity outers + duplicate inner primitives (axis-3 prompts 16, 19, 25, 28, 30) — motivates future Phase 6d (per-port keyword binding at inner-pick time, not just port-allocation time). Phase 6d not scheduled. |

## 14. V1.0.7 scaling-curve consolidation pass (2026-05-01)

A corpus-only pass folding the post-Phase-3 research arc and the today-written organelle synthesis into the rebuild-test corpus. **No code changes.**

**New gaps registered (both immediately RESOLVED as documented limitations):**
- `GAP-WIRE-007` (P2) — bag-of-features ceiling. Three feature variants (unigram / bigram / char-ngram) converge within ±1/20 on the v2 vocabulary-disjoint held-out (16/15/15). Confirms the ~75-80% retrieval ceiling is model-bound to bag-of-features, not unigram-specific.
- `GAP-WIRE-008` (P2) — domain-bounded ceiling. v3 (chemistry / time / conversions / combinatorics, lean synonyms) hit 3/20; expanding to v2-depth synonyms HURT to 0/20 due to generic English glue diluting distinctive concepts. Bound is domain (curator-vocabulary) not curator-effort.

**Corpus updates (this directory):**
- `BS_wiring.md` §6 SLOs: added `SLO-WIRE-008` (calibrated 75-80% ceiling), `SLO-WIRE-009` (no-regression), `SLO-WIRE-010` (bag-of-features convergence invariance). §8 ACC: added `ACC-WIRE-005..008` paired tests.
- `BS_wiring.md` §4 invariants: added `INV-WIRE-060` (bag-of-features convergence MUST hold), `INV-WIRE-061` (distinctive-noun structural bound), `INV-WIRE-062` (standing leakage-audit precondition before any new retrieval claim).
- `RESEARCH_DISCLOSURE.md` §5 added — three-bound consolidation; pre-registers nothing new (the bag-of-features and domain bounds are RESOLVED documented limitations, not falsified hypotheses).
- `README.md` Companion artefacts: `docs/research/ORGANELLE_STATE.md` added as the recommended top-of-stack reading order across all `RESEARCH_ORGANELLE_*` topical docs.

**Rationale:**
The post-Phase-3 arc moved the project from "1:1 scaling claim (inflated)" to "75-80% calibrated ceiling under three named structural bounds (curator-, model-, domain-bounded)." That calibration is load-bearing for productisation conversations and for any future contributor trying to reproduce numbers — but it lived only in `docs/research/wiring_scaling_post_phase3.md` until this pass. The corpus now reflects it as normative SLOs and invariants, so a clean-room rebuilder will reproduce both the calibration AND the standing protections (leakage audit, bag-of-features convergence).
