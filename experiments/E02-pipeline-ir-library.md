# Experiment E02 — Promote Pipeline IR + verifier to a standalone C99 library

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** widen the audience for the project's most distinctive component — extract it from OPA's gravitational well so it can serve any neurosymbolic system.
**Cost estimate:** ~3-4 weeks (1 wk extraction + 1 wk API hardening + 1 wk demo bindings + 1 wk packaging/docs).
**Falsification risk:** Low (the extraction is mostly mechanical; the falsification surface is whether the library *gets used* externally — but that's measured over months, not in the experiment itself).

---

## Spear summary

**Point:** Pipeline IR + parser + tolerant-parser + repair + verifier + DOT renderer is OPA's most novel and most reusable component, and it has zero dependencies on the transformer engine. As a standalone library `libpipeline_ir`, it can serve as the deterministic post-hoc Judge for *any* graph-emitting system — including frontier LLMs.

**Picture:** Today, `microgpt_pipeline.{h,c}` lives inside `src/`, depends on nothing beyond `<stdlib.h>` and `<string.h>`, and is statically linked into the wiring binary. Extract it into `libs/pipeline_ir/`, add a clean C99 ABI, package it for CMake `FetchContent` / `find_package` / vcpkg / Conan, write three minimal example projects (LLM bridge, custom-generator bridge, hand-written bridge), and the project gains a publishable, reusable artefact.

**Proof (to be measured):** all 51 existing pipeline IR unit tests pass against the extracted library unchanged; a new end-to-end test exercises the library against LLM-emitted `@graph...@end` text via a 50-line bridge; documentation includes a one-page "use it from your project" guide.

**Push:** A standalone Pipeline IR library is the single highest-leverage architectural move because it converts a buried implementation detail into a reusable artefact that other research groups can adopt without buying into OPA's transformer engine or its tiny-specialist thesis.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

The Pipeline IR component is *fully* separable from OPA. Concretely:

> *Extracting `microgpt_pipeline.{h,c}` + the 40 native primitives in `wiring_natives.{h,c}` (renamed `pipeline_ir_natives.{h,c}`) + the reference suite scaffolding from `wiring_references.{h,c}` (renamed `pipeline_ir_reference_runner.{h,c}`) into `libs/pipeline_ir/` with a clean public C99 ABI does not regress any existing test, expands binary-level reuse to any C-callable language (Rust, Python, Go, Swift via FFI), and provides a 50-LOC bridge surface for accepting LLM-emitted graph text.*

### 1.2 Why this matters

Today the Pipeline IR is **structurally separable but socially buried**. Anyone wanting to use it has to:

1. Clone the entire MicroGPT-C repo (~15 MB).
2. Build the entire engine (CMake + ~3 minutes).
3. Reverse-engineer the API from the wiring binary's usage.
4. Buy into the tiny-specialist thesis to justify the architectural commitment.

That gate is unnecessary. The IR is its own contribution, separable from the model classes that happen to emit text into it. By promoting it:

- The "deterministic post-hoc Judge for LLM tool calls" framing becomes actionable for *anyone* — no need to adopt OPA.
- The methodology-paper case in [E05](E05-prereg-methodology-public.md) gets a reference implementation to point at.
- The head-to-head experiment in [E01](E01-llm-head-to-head.md) becomes easier because System B's LLM bridge is one of three example projects, not a custom one-off.
- The medical-guideline-graphs work in [E06](E06-medical-guideline-graphs.md) gains a clean integration surface.

### 1.3 Mechanism

**Phase 1 — Extraction (1 week).**

```
libs/pipeline_ir/
├── include/pipeline_ir/
│   ├── pipeline_ir.h          # main public ABI (was microgpt_pipeline.h)
│   ├── natives.h              # built-in primitive registry (was wiring_natives.h)
│   └── reference_runner.h     # ACC harness (was wiring_references.h)
├── src/
│   ├── pipeline_ir.c
│   ├── natives.c
│   └── reference_runner.c
├── tests/
│   ├── test_parse.c
│   ├── test_verify.c
│   ├── test_repair.c
│   └── test_dot.c
├── examples/
│   ├── llm_bridge/            # accept @graph from LLM stdin → verify → JSON status
│   ├── custom_generator/      # hand-written graphs → verify
│   └── audit_visualiser/      # graph → DOT → SVG
├── CMakeLists.txt
├── pipeline_ir-config.cmake.in
├── README.md
└── LICENSE                    # MIT, same as parent
```

**Phase 2 — API hardening (1 week).** Audit every public symbol for:
- ABI versioning: `PIPELINE_IR_API_VERSION_MAJOR`, `_MINOR`, `_PATCH`.
- Error reporting: replace any `printf` / `fprintf(stderr, ...)` in library code with a returned error code + optional user-supplied logger callback.
- Allocation: replace any internal `malloc` with a `pipeline_ir_allocator_t` indirection so embedded users can plug in their own.
- Threading: document re-entrancy assumptions; add a thread-safety statement per symbol.
- Const-correctness sweep on all input pointers.

**Phase 3 — Demo bindings (1 week).** Three example projects in `examples/`:

1. **`llm_bridge/`** — Reads `@graph...@end` from stdin (assumed LLM output), runs `pipeline_parse_text_tolerant()` → `pipeline_repair()` → `pipeline_verify()`, emits JSON `{verdict, errors[], dot_url}` to stdout. ~50 LOC.
2. **`custom_generator/`** — Hand-written graph (C struct literal) → verify → DOT → SVG. Shows the "no model required" mode for unit-test-as-judge style usage. ~60 LOC.
3. **`audit_visualiser/`** — Read graph from file → emit DOT → invoke `dot -Tsvg` → output SVG. ~40 LOC.

**Phase 4 — Packaging (1 week).** CMake config, `find_package(pipeline_ir)` and `FetchContent_Declare()` support; vcpkg port file; Conan recipe; one-page "use in your project" guide; release as `v0.1.0` with semver commitment.

**Backward compatibility for OPA.** The original `src/microgpt_pipeline.{h,c}` becomes a thin shim that re-exports the library's symbols under the legacy names; or, cleaner, OPA's CMake adds `pipeline_ir` via `FetchContent` and updates includes to `<pipeline_ir/pipeline_ir.h>`. Existing OPA tests must pass unchanged.

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | All 51 existing pipeline IR unit tests pass against the extracted library | < 51 (= a real semantic regression introduced by extraction) |
| **T2** | All OPA tests (`test_microgpt`, `test_microgpt_msa`, …) still pass via the shim/FetchContent path | < 100% (= integration bug) |
| **T3** | `libpipeline_ir.a` size ≤ 200 KB stripped on `-O2` | > 500 KB (= bloat from extraction) |
| **T4** | LLM bridge example accepts a `@graph...@end` string and returns verdict in ≤ 5 ms p99 on M2 Max | > 50 ms (= path-length regression) |
| **T5** | Public ABI documented per-symbol with stability classification (stable / experimental / internal) | < 95% coverage (= docs gap) |
| **T6** | New end-to-end test: a Claude Sonnet `@graph` emission flows through the LLM bridge and is correctly classified pass/fail | < 100% on a curated 10-prompt sanity set |

### 1.5 Skip rules

- If T1 falls below 51 (semantic regression): **do not ship**. Diagnose, fix, re-pre-register the affected sub-experiment.
- If T3 exceeds 500 KB: investigate whether the natives registry needs to become a separate optional library (`libpipeline_ir_natives`); the core IR + verifier should be << 100 KB.
- If T5 is < 95%: documentation gap → extend Phase 2 by one week; do not release v0.1.0 until met.

### 1.6 Falsification risk: Low

| Risk | Likelihood | Mitigation |
|---|---|---|
| OPA tests break after extraction | Medium | Shim layer + FetchContent dual path; CI runs both for one release cycle |
| ABI churn risk (someone adopts v0.1.0 then we break it) | Medium | Semver from day one; deprecation cycle (≥ 2 minor versions) for any removal |
| Native primitives have hidden OPA-specific assumptions | Low-medium | Audit each of the 40 primitives in `wiring_natives.c` during Phase 1 |
| Build system fragility on non-Apple platforms | Low | CI already covers Ubuntu (gcc/clang), macOS (clang), Windows (cl) per `.github/workflows/cmake-multi-platform.yml`; extend to the new library |

### 1.7 What this experiment is NOT testing

- It is **not** testing whether external users adopt the library. Adoption is a multi-month signal, not a 4-week measurement.
- It is **not** testing whether the IR grammar evolves. Conditional graphs (Experiment 5.1 in `RESEARCH_OPA_DIRECTIONS.md`) and recursive sub-graphs (5.2) are separate experiments.
- It is **not** testing whether the verifier's pass/fail surface gets extended to probabilistic verdicts (Experiment 3.1 in `RESEARCH_OPA_DIRECTIONS.md` → finance vertical) — orthogonal.
- It is **not** a productisation move. The library is MIT-licensed and research-positioned; productisation lives in `organelles.bio`.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| The IR component being extracted | [`src/microgpt_pipeline.{h,c}`](../src/microgpt_pipeline.c) |
| Primitives that move with it | [`src/wiring_natives.{h,c}`](../src/wiring_natives.c), [`src/wiring_references.{h,c}`](../src/wiring_references.c) |
| Existing 51 IR tests that gate the extraction | `tests/test_microgpt_pipeline.c` |
| Direction parent | [`RESEARCH_OPA_DIRECTIONS.md`](../docs/research/RESEARCH_OPA_DIRECTIONS.md) §"Pipeline IR extensions" — separation is a prerequisite to §5.1 and §5.2 |
| LLM head-to-head that uses the library | [E01](E01-llm-head-to-head.md) System B bridge |

---

## 2. Initial state

### 2.1 What's currently known

- `microgpt_pipeline.c` is ~1,200 LOC, zero dependencies beyond `libc`.
- 51 unit tests pass on every CI platform.
- `pipeline_render_text()` round-trips byte-stably via Kahn topo sort (`INV-PIPE-002`).
- `pipeline_verify()` returns `PIPE_OK` / `PIPE_ERR_*` codes; no probabilistic surface yet.
- DOT renderer is feature-complete and audit-passing on every existing graph.

### 2.2 Dependencies the library will have

- C99 standard library only (`<stdlib.h>`, `<string.h>`, `<stdio.h>`, `<stdbool.h>`).
- No external libs.
- No CMake `find_package(...)` dependencies.
- Optional `dot` (Graphviz) at *consumer* runtime for SVG rendering of DOT output — not a build dependency.

### 2.3 What's blocking

Nothing — every prerequisite exists. The work is mechanical.

### 2.4 Baselines

- Status quo: 0 external adopters (library doesn't exist as a separable artefact).
- Comparable artefacts in the neurosymbolic-verifier space: none with the audit-DOT + tolerant-parse + repair triple. Closest is Microsoft's [LMQL grammar layer](https://lmql.ai/) — but that targets prompt-side grammars, not post-hoc graph verification.

---

## 3. Implementation + results

**TODO** — fill on measurement commit. Sections to populate:

- 3.1 Extraction commit hash + diff stats
- 3.2 Test pass matrix (T1, T2 across all three CI platforms)
- 3.3 Library size + symbol count
- 3.4 Example project working screenshots / SVG renders
- 3.5 Reproduction: `git clone … && cd libs/pipeline_ir && cmake … && ctest`

---

## 4. Conclusion

**TODO** — fill on measurement commit. Sections to populate:

- 4.1 Verdict per T1-T6 (PASS / FAIL)
- 4.2 ABI v0.1.0 released? Y/N
- 4.3 Lessons (especially: which native primitives turned out to have hidden OPA-coupling?)
- 4.4 Next moves: announce on Hacker News / r/MachineLearning / Twitter; submit to neurosymbolic AI venues as a tool paper; integrate into [E01](E01-llm-head-to-head.md) and [E06](E06-medical-guideline-graphs.md)
- 4.5 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`, `RESEARCH_DISCLOSURE.md`)
