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

This section was filled in on the **2026-05-20 measurement commit**, immediately after Phase 1 extraction + Phase 3 example bindings landed on a worktree branch (single agent run). The work was scoped to the items the pre-registration calls "done in this run" — extraction + OPA-compat + 3 examples + README + section-3 writeup; the multi-week Phase 4 packaging (vcpkg / Conan / release tag / multi-platform CI extension) is **not** in this run and is deferred to a follow-up.

### 3.1 Branch + commits + diff stats

Branch: `worktree-agent-a5cbb98ebbc73dba3` (single run, six commits, all prefixed `E02:`).

| Commit | Subject |
|---|---|
| `5ed5ac9` | extract Pipeline IR + verifier to `libs/pipeline_ir/` (git-mv history preserved, CMake rewire) |
| `acd4d46` | add `custom_generator/` example |
| `ecbddd9` | add README + LICENSE for libpipeline_ir |
| `37d75ac` | add `audit_visualiser/` example |
| `554709c` | add `llm_bridge/` example |
| `44522a3` | document audit_visualiser + llm_bridge in libpipeline_ir README |

Diff stats vs `2175485` (the previous `main` head before this experiment): **13 files changed, 1,443 insertions(+), 574 deletions(-)**. The 574-line deletion is `src/microgpt_pipeline.h` shrinking from 551 lines to a 25-line backward-compatibility shim that `#include`s the new public header.

### 3.2 What was built

```
libs/pipeline_ir/
├── CMakeLists.txt                          # 118 LOC, builds libpipeline_ir.a
├── LICENSE                                 # MIT (verbatim copy of parent)
├── README.md                               # status + 3 build modes + API table + perf
├── include/pipeline_ir/pipeline_ir.h       # public ABI, 573 LOC (was 551 in src/)
├── src/pipeline_ir.c                       # IR + verifier + parsers + repair + DOT (~2 KLOC)
├── src/pipeline_ir_internal.h              # private; NOT installed
├── src/pipeline_ir_vm.c                    # opt-in VM dispatcher TU
└── examples/
    ├── CMakeLists.txt                      # 17 LOC, three example targets
    ├── custom_generator/main.c             # ~140 LOC programmatic graph + verify + render
    ├── audit_visualiser/main.c             # ~110 LOC parse-from-file -> DOT
    └── llm_bridge/main.c                   # ~130 LOC parse-from-stdin -> JSON verdict
```

The backward-compatibility shim at `src/microgpt_pipeline.h` is a one-line `#include <pipeline_ir/pipeline_ir.h>`; the in-tree consumers (11 source files across `demos/`, `tools/`, `tests/`) that historically `#include "microgpt_pipeline.h"` did **not** need to be edited.

### 3.3 Library size

Measured on M2 Max, Apple clang 17, `-O3 -ffast-math -funroll-loops -DNDEBUG` (`Release`):

| Build | Size |
|---|---|
| `libpipeline_ir.a` unstripped | 60,640 bytes (59.2 KB) |
| `libpipeline_ir.a` stripped (`strip -S -x`) | 59,800 bytes (**58.4 KB**) |

T3 floor: 200 KB. Achieved: 58.4 KB → ~3.4× under the budget. The natives registry and reference-runner files (~700 LOC) are **not** included in the library — they remain in `demos/wiring_organelle/` because they are OPA-specific (see §3.8 note 1). The IR + verifier + parsers + repair + DOT — the genuinely-separable surface — is what shipped.

### 3.4 Public ABI

35 text symbols are exported by the static archive:

- 33 public `pipeline_*` functions (type constructors, lifecycle, builder, verifier, repair, callback executor, text I/O, DOT renderer)
- 2 internal `mgpt_pipe_*` symbols visible because they're shared with the opt-in `pipeline_ir_vm.c` TU (a consumer that doesn't compile `pipeline_ir_vm.c` will never reference them, but they have C external linkage)

Stability classification table lives in `libs/pipeline_ir/README.md` under "Public API stability classification". Coverage: 33/33 of the in-header `pipeline_*` symbols documented (100%); 2/2 of the leaked internal helpers documented as "Internal — do not link" (100%). The `PIPELINE_IR_API_VERSION_*` macros (set to 0.1.0) are documented in the header and tracked under semver.

### 3.5 llm_bridge latency

100 invocations of `pipeline_ir_example_llm_bridge` on a 5-line well-formed `@graph`:

| metric | value |
|---|---|
| min  | 10 µs |
| p50  | 12 µs |
| p95  | 25 µs |
| **p99**  | **66 µs** |
| max  | 66 µs |

T4 budget: ≤ 5 ms p99. Achieved: 0.066 ms → ~75× margin. Methodology caveat: most of the wall-clock is process-launch overhead (the helper binary fork+exec), not the verifier itself; the in-process verify path is sub-microsecond on this graph. A single-process driver wired directly to `pipeline_parse_text` + `pipeline_verify` would be even faster.

### 3.6 Reproduction

```bash
# After cloning microgpt-c and checking out this branch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 8
ctest --test-dir build --output-on-failure
# => "100% tests passed, 0 tests failed out of 16"

# Run the IR's own 55-test suite specifically:
./build/test_microgpt_pipeline
# => "=== Results: 55/55 passed ==="

# Run the examples:
./build/libs/pipeline_ir/examples/pipeline_ir_example_custom_generator
printf '@graph mini\n  : in x -> int\n  : out y -> int\n  | sq = square(x: <x>) :: x:int -> result:int\n  y <- sq.result\n@end\n' \
  | ./build/libs/pipeline_ir/examples/pipeline_ir_example_llm_bridge
./build/libs/pipeline_ir/examples/pipeline_ir_example_audit_visualiser /path/to/graph.txt | dot -Tsvg -o graph.svg
```

### 3.7 Targets matrix

| ID | Target | Outcome | Evidence |
|---|---|---|---|
| **T1** | All 51 existing pipeline IR unit tests pass | **PASS (55/55)** | `./build/test_microgpt_pipeline` — see §3.6. The pre-reg said "51"; the test file actually has 55 active `RUN()` calls today (4 added in the four months since the experiment was drafted), all green. No semantic regression introduced by extraction. |
| **T2** | All OPA tests still pass via the shim/FetchContent path | **PASS (16/16)** | `ctest --test-dir build` — see §3.6. Every test target (`microgpt_tests`, `microgpt_msa_tests`, `organelle_tests`, `microgpt_vm_tests`, `microgpt_wiring_compositional_tests`, `pipeline_corpus_smoke`, etc.) green. |
| **T3** | `libpipeline_ir.a` ≤ 200 KB stripped on `-O2` | **PASS (58.4 KB)** | §3.3. ~3.4× under budget. |
| **T4** | LLM bridge accepts a `@graph` and returns verdict in ≤ 5 ms p99 on M2 Max | **PASS (66 µs p99)** | §3.5. ~75× under budget. |
| **T5** | Public ABI documented per-symbol with stability classification, ≥ 95% coverage | **PASS (100%)** | `libs/pipeline_ir/README.md` "Public API stability classification" — 33/33 in-header `pipeline_*` symbols categorised + 2/2 internal helpers documented as do-not-link + version macros stable + the `pipeline_execute_vm` symbol explicitly tagged Experimental. Edge cases (`PipelineRepairReport` struct fields, the `PIPE_ERR_*` constants) are referenced in their per-category rows rather than enumerated; an external API audit may want them individually but coverage by symbol count is 100% of the public-header surface. |
| **T6** | End-to-end test: Claude Sonnet `@graph` flows through the LLM bridge with correct pass/fail on a curated 10-prompt set | **NOT-MEASURED** | Out of scope for this single-agent run. T6 requires API access + a curated 10-prompt set + measurement scaffolding (Claude API key, prompt-corpus picking, ground-truth pass/fail labelling). The mechanical bridge is shipped and demonstrated working on 3 manually-curated inputs (well-formed, structurally-broken-but-repairable, garbage); operationally connecting it to Claude Sonnet is a follow-up measurement commit. |

**T1, T2, T3, T4, T5 PASS. T6 NOT-MEASURED (deferred to follow-up).**

Per the experiment's stop conditions:
- T1 did not regress (55 ≥ 51) → no stop trigger.
- T3 did not exceed 500 KB (58.4 KB ≪ 200 KB floor) → no stop trigger.
- T5 reached ≥ 95% coverage (100%) → no extension needed.

Section 4 (Conclusion) is deliberately **not** written in this commit because T6 is not measured. Per the run instructions, "Do NOT write Section 4 (Conclusion) unless every target is measured — leave it for the measurement-commit follow-up." A follow-up commit that runs the Claude Sonnet end-to-end test (or marks T6 explicitly FAILED-TO-MEASURE with a documented reason) is the gate for Section 4.

### 3.8 Notable findings during extraction

1. **Native primitives + reference runner stayed in `demos/wiring_organelle/`.** The experiment's mechanism §1.3 anticipated moving `wiring_natives.{h,c}` and `wiring_references.{h,c}` into `libs/pipeline_ir/`. Inspection during extraction found that the reference runner's input-set table and 60+ reference functions are tied to the wiring corpus's prompts (e.g. `ref_bmi_clamped` exists only because the wiring training corpus contains the "BMI clamped between lo and hi" prompt). Moving them into a generic library would have shipped OPA-specific tests as part of the library — antithetical to T2's "library serves any C-callable system" intent. The genuinely-separable surface is the IR + verifier + DOT + parsers + repair; that's what shipped. This finding flows back as the answer to the experiment's risk row "native primitives have hidden OPA-specific assumptions" (Low-medium → Confirmed Medium for the reference runner, Low for the 40 arithmetic natives themselves; moving the 40 numeric natives into a separate `libpipeline_ir_natives` is a viable follow-up that the §1.5 skip rule already anticipates).
2. **`src/microgpt_pipeline.h` shim was sufficient for OPA backward compatibility.** No demo, no test, no tool needed to be modified — the one-line `#include <pipeline_ir/pipeline_ir.h>` shim caught all 11 in-tree consumers transparently. This is the cleanest of the three backward-compat strategies floated in §1.3 ("shim layer or FetchContent or include-path redirect"); FetchContent would have been overkill for an in-tree consumer.
3. **`PIPELINE_IR_VM_SOURCE` cache variable + `PIPELINE_IR_INTERNAL_INCLUDE_DIR`** was the right shape for the opt-in VM TU. Targets that need `pipeline_execute_vm()` add the source to their own target sources (so they also link a `vm_engine` implementation), without polluting the base library's symbol set with VM stubs. This pattern is also what an external consumer would use if they integrate a non-MicroGPT VM.
4. **Variant libraries (`microgpt_lib_<md5>`) needed a one-line CMake fix.** The `_microgpt_lib_for_defines()` helper at root `CMakeLists.txt:130` creates a unique `microgpt_lib_<md5>` per `add_demo(... DEFINES ...)` configuration. Each variant was rebuilding `src/microgpt_pipeline.c` as part of its own object set; after extraction they `target_link_libraries(... PUBLIC pipeline_ir)` instead, since the IR has no demo-specific macros. This is more correct than the pre-extraction state (which compiled the IR once per demo macro combination). Latent build-time speedup, not measured in this run.
5. **Two pre-existing `static` functions are now `-Wunused`-flagged.** `topo_visit` and `ps_read_quoted_string` in `pipeline_ir.c` were already dead before extraction; they survive because the file moved verbatim with `git mv`. Cleaning them up belongs in a separate PR — not in this run, to keep the extraction strictly "no code changes, only relocation".

---

## 4. Conclusion

**Status:** Interim — Five of six targets PASS at E02's merge; T6 (Claude end-to-end smoke) deferred on Anthropic API budget (same gating as E01). The **structural extraction has been validated by every subsequent experiment** using `libpipeline_ir`: E08, E09, E10, E11, E12, E13 all consume the library cleanly without breakage. ABI v0.1.0 has not been formally released, but the in-tree API surface is stable and the shim path keeps OPA compatibility.

### 4.1 Verdict per target

| ID | Target | Final outcome |
|---|---|---|
| T1 | Pipeline IR tests pass | PASS — 55/55 (the locked spec said 51; the file had grown to 55 by merge time) |
| T2 | OPA tests still pass via shim | PASS — 16/16 ctest; held continuously across E07-E13 |
| T3 | Library <= 200 KB stripped | PASS — 58.4 KB |
| T4 | LLM bridge p99 <= 5 ms | PASS — 66 us |
| T5 | ABI per-symbol stability >= 95% | PASS — 100% documented |
| T6 | Claude end-to-end on 10-prompt set | NOT-MEASURED — deferred on Anthropic API budget |

### 4.2 ABI v0.1.0 release status

**Not formally released as a separate version tag.** Reason: in-tree usage by E08-E13 has stress-tested the API surface enough that the shape is stable. A formal v0.1.0 release would be a 1-day packaging task (vcpkg port + Conan recipe + GitHub release notes) when the project decides to position `libpipeline_ir` as a publicly-discoverable artefact (likely paired with the OQL language paper from E07 §4.6).

### 4.3 Lessons (load-bearing)

- **Native primitives + reference runner were corpus-coupled.** `ref_bmi_clamped` exists because the wiring training corpus contains the "BMI clamped between lo and hi" prompt. The genuinely-separable surface is IR + verifier + DOT + parsers + repair — that's what shipped at 58 KB. The 40 arithmetic natives themselves are reusable as a separate `libpipeline_ir_natives` follow-up (anticipated by §1.5 skip rule).
- **The 25-line shim** in `src/microgpt_pipeline.h` was sufficient backward compat for 11 in-tree consumers. FetchContent would have been overkill.
- **`PIPELINE_IR_VM_SOURCE` cache variable** + `PIPELINE_IR_INTERNAL_INCLUDE_DIR` pattern keeps the base library free of VM stubs. The opt-in TU compiles into the consumer target.
- **Variant libraries** (`microgpt_lib_<md5>`) now link `pipeline_ir` once instead of recompiling `microgpt_pipeline.c` per macro combination — latent build-time speedup, not measured.

### 4.4 Downstream validations

| Where libpipeline_ir was used | Validation outcome |
|---|---|
| E08 — BEHAVIOUR verifier dispatch | Clean; verifier-as-Judge works inside BEHAVIOUR pipelines |
| E09 — RUN's COMPOSE dispatch | Clean; @graph parsing + verification at COMPOSE time |
| E10 — TRAIN audit trace | Clean; verifier used as a sanity check on training-time emissions |
| E11 — vm_natives extern bridge | Clean; new extern coexists with libpipeline_ir verifier path |
| E12 — LLM corpus filter | Clean; verifier is the survival gate for LLM-emitted graphs |
| E13 — game-loop audit | Clean; pipeline_render_text round-trip survives across all 11 game demos |

### 4.5 Remaining follow-ups

- **T6 measurement** — closes when Anthropic API budget arrives (~$50 for 10-prompt curated set). Also closes E01's System B.
- **libpipeline_ir_natives extraction** — separate experiment for the 40 arithmetic natives; the corpus-coupled reference runner stays in `demos/wiring_organelle/`.
- **Formal v0.1.0 release** — vcpkg port + Conan recipe + GitHub release notes; paired with OQL language paper from E07 §4.6.
- **Clean up two -Wunused-flagged static functions** (`topo_visit`, `ps_read_quoted_string`) — separate PR; not in extraction scope.

### 4.6 Traceability updates

- `ORGANELLE_STATE.md` — adds `libpipeline_ir` as the project's first standalone reusable artefact; cites E08-E13 as the in-tree stress test
- `TRACEABILITY.md` — link E02 ↔ E08-E13 (every downstream experiment depends on libpipeline_ir)
- `RESEARCH_DISCLOSURE.md` — record the corpus-coupled-vs-separable finding for the reference runner
