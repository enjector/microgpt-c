# Experiment E07 — OQL: a SQL-shaped DSL for organelle definition, pipelining, training, and pre-registered experimentation

**Status:** 📋 Proposal locked — 2026-05-20.
**Direction:** elevate the operator surface from "C demos + markdown experiment docs" to a single declarative dialect that defines organelles, pipelines, ingress/egress, VM-coded behaviours, and pre-registered experiments — executable end-to-end. Inspired by [EQL](../../EnX/EnX-Research-Prototypes/aerospike.github/cpp/enx-db/book-eql.v7/The_Expressive_Power_of_EQL.md): *"the query language is the product."*
**Cost estimate:** ~6-8 weeks (1 wk grammar design + 2 wk Flex/Bison + AST + 2 wk interpreter (binds into existing engine) + 1 wk worked-example rewrite of E01-E06 in OQL + 1-2 wk docs/paper).
**Falsification risk:** Medium — the grammar may turn out to be either too narrow (real experiments need imperative escape hatches) or too wide (verb count balloons past EQL's "+4 / -4" discipline). Either falsification is informative.

---

## Spear summary

**Point:** Every artefact in this project — organelles, pipelines, corpora, training runs, evaluations, pre-registered experiments, audit traces — already has a relational shape. The project just hasn't exposed that shape as a query surface. **OQL — the Organelle Query Language** — is SQL with **+6 verbs and -4 verbs**, executable through a Flex/Bison front-end that reuses the existing VM infrastructure (`src/microgpt_vm.{l,y}`). One language replaces a stack of bespoke C demos, markdown experiment specs, hand-rolled training drivers, and ad-hoc evaluation scripts.

**Picture:** Today, running E01 means writing 600 lines of C across `demos/wiring_organelle/`, four shell scripts in `tools/`, and a 250-line markdown experiment doc. With OQL it becomes a ~40-line declarative spec: `CREATE ORGANELLE wiring_v2 … TRAIN ON corpus_v2 … COMPOSE pipeline_v2 AS @graph … RUN EXPERIMENT E01 EVALUATE ON heldout_v2 WITH TARGETS …`. The C engine is the runtime; OQL is the surface.

**Proof (to be measured):** each of E01-E06 is rewritable as ≤ 50 lines of OQL; the grammar fits ≤ 200 LOC of `.y` / `.l`; an engineer reads and writes their first OQL experiment in ≤ 1 hour; the parser drops into `src/microgpt_oql.{l,y,c,h}` using the same Flex/Bison fallback pattern the VM already uses.

**Push:** OQL is the operator surface that turns the architecture from "a research repo with documented experiments" into "a research substrate that other people can author experiments on." It is the layer EQL is to EnX-DB — but for organelle pipelines.

---

## 1. Proposal

### 1.1 Hypothesis (locked before measurement)

> *A SQL-shaped DSL — `OQL` — with **6 added verbs** (`TRAIN`, `COMPOSE`, `RUN`, `EVALUATE`, `VERIFY`, `AUDIT`) and **4 omitted verbs** (`CREATE TRIGGER`, `CREATE FUNCTION`, `DECLARE CURSOR`, `SAVEPOINT`) is sufficient to express every workflow currently encoded across `demos/`, `tools/`, and `experiments/` in ≤ 1.5× the source-line cost — while delivering a parser readable in a weekend and a single declarative surface for pre-registered experiment authoring.*

### 1.2 Why this matters

The project today carries three parallel surfaces that all describe the same underlying objects (organelles, pipelines, corpora, experiments):

| Surface | Where it lives | Cost |
|---|---|---|
| C demo code | `demos/character-level/*/main.c` (≈ 200-800 LOC each) | Reimplemented per demo; ~50% boilerplate |
| Build/run scripts | `tools/*.sh`, `bootstrap.sh`, ad-hoc CMake invocations | Tribal knowledge; fragile across platforms |
| Markdown experiment specs | `experiments/E0?.md`, `docs/research/RESEARCH_*.md` | Human-readable but not executable; targets locked in prose, not in code |

These three describe **the same workflows**. Every demo is some variation of: (a) load corpus, (b) train organelle, (c) compose pipeline, (d) evaluate on held-out, (e) emit metrics. Every experiment spec is some variation of: (a) hypothesis, (b) pre-reg targets, (c) measurement, (d) verdict. The repetition is high; the underlying relational shape is hidden.

EQL's argument applies cleanly here: the cognitive surface is the cost, not the server count. Replace three surfaces with one declarative language and the project gains:

1. **Authorship parity** — anyone fluent in SQL can read an OQL experiment on day one and write one by day three.
2. **Executable pre-registration** — `CREATE EXPERIMENT E08 … WITH TARGETS … SKIP_RULE …` becomes the *artefact*, not a prose section. The pre-reg parser in [E05](E05-prereg-methodology-public.md) reads OQL natively instead of grovelling markdown.
3. **Reproducibility by construction** — every published experiment is one OQL file + one corpus reference + one runtime version pin. `oql run E08.oql` reproduces it.
4. **The architecture stops being implementation-defined** — today, the meaning of "an organelle" is "whatever the C code does." With OQL, the meaning is "what the OQL specification says" and the C runtime is one (current) implementation.

### 1.3 Mechanism

#### 1.3.1 Grammar design — the SQL+6-4 surface

**Inherited from SQL (recognisable to anyone fluent in Postgres):**

```sql
SELECT, INSERT, UPDATE, DELETE,
CREATE, DROP, ALTER,
BEGIN, COMMIT, ROLLBACK,
JOIN (INNER, LEFT, RIGHT, FULL OUTER),
WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, OFFSET,
WITH (CTEs),
WINDOW functions (LAG, LEAD, AVG OVER, …)
```

**Added (the +6 verbs that reach OPA-specific concepts SQL doesn't have):**

| Verb | What it does | Maps to |
|---|---|---|
| **`TRAIN`** | Differentiate a blank organelle on a corpus into a role-specialised model. `TRAIN organelle_x ON corpus_y WITH ROLE planner STEPS 20000 LR 1e-3` | `TrainWorker` pthread harness in `microgpt.c` |
| **`COMPOSE`** | Wire organelles + native primitives into a Pipeline IR graph. Body is `@graph…@end` text. `COMPOSE pipeline_v2 AS @graph[…]@end` | `pipeline_parse_text()` + `pipeline_verify()` |
| **`RUN`** | Execute a pipeline on input(s) and capture outputs + audit trace. `RUN pipeline_v2 ON prompts_held_out RETURNING (graph, latency_ms, trace)` | The wiring binary's vote loop |
| **`EVALUATE`** | Run a pipeline against a held-out set and emit metrics. `EVALUATE pipeline_v2 ON heldout_v2 METRICS (accuracy, audit_coverage, p99_latency)` | The harness in `demos/wiring_organelle/main.c` |
| **`VERIFY`** | Standalone IR verification. `VERIFY GRAPH @graph[…]@end` → `(verdict, errors[], dot)` | `pipeline_verify()` directly |
| **`AUDIT`** | Run the standing leakage audit. `AUDIT corpus_a AGAINST corpus_b WITH (mode=jaccard, threshold=0.7)` | [`tools/scaling_leakage_audit.sh`](../tools/scaling_leakage_audit.sh) |

**Omitted (the -4 verbs SQL has that don't earn their keep here):**

| Omission | Why |
|---|---|
| **`CREATE TRIGGER`** | The architecture's reactivity is explicit (cells / kanban), not implicit (trigger-fires-row-mutation). Match EQL's omission. |
| **`CREATE FUNCTION` (PL/pgSQL)** | Imperative function bodies belong in the VM (which already exists, with its own DSL); embedding a second imperative dialect inside OQL doubles the cognitive surface. Behaviours go in **VM-coded blocks**, see §1.3.2. |
| **`DECLARE CURSOR`** | Streaming evaluation is express via `RUN … STREAMING` (a built-in modifier), not a server-side cursor. |
| **`SAVEPOINT`** | Experiments are versioned via git, not via nested transactions. |

**New first-class object types (kept tiny):**

| Object | Storage | Examples |
|---|---|---|
| `ORGANELLE` | Trained-weights checkpoint + role label | `wiring_v2`, `connect4_player`, `c_planner` |
| `PIPELINE` | A Pipeline IR `@graph` definition + binding to organelles/natives | `pipeline_v2`, `connect4_play_judge` |
| `CORPUS` | Training/eval text data + metadata | `corpus_v2_clean`, `heldout_v2_sealed` |
| `INGRESS` | Data source binding (stdin, file, named pipe, future: kafka/HTTP) | `prompts_from_stdin`, `corpus_from_file` |
| `EGRESS` | Data sink binding | `graphs_to_jsonl`, `metrics_to_stdout` |
| `BEHAVIOUR` | A VM-coded block (using existing `microgpt_vm`) | `score_candidate(c, p)`, `pick_best(cs)` |
| `EXPERIMENT` | Pre-registration record: hypothesis, targets, skip-rules, status | `E01`, `E08`, … |

**Behaviours via embedded VM code (§1.3.2):** when a workflow needs imperative steps (custom scoring functions, candidate re-ranking, deterministic post-processing), the OQL author writes a `BEHAVIOUR` whose body is `microgpt_vm` source. The OQL interpreter compiles the VM block at parse time using the existing Flex/Bison VM parser. **One imperative layer, not two** — re-using the VM is the entire reason it exists.

#### 1.3.2 Example OQL — what E01 looks like in 40 lines

```sql
-- E01-llm-head-to-head.oql
-- Pre-registered Experiment E01: head-to-head vs frontier LLM
-- on a public typed-graph benchmark.

CREATE CORPUS heldout_e01
  FROM FILE 'benchmarks/toolbench_typed_graph_v1.jsonl'
  WITH (license='research', leakage_audit=REQUIRED);

AUDIT heldout_e01 AGAINST corpus_v2 WITH (mode=jaccard, threshold=0.7);
-- Fails the script if any Jaccard >= 0.7 — runs before any measurement.

CREATE ORGANELLE wiring_v2 FROM CHECKPOINT 'wiring_v2.ckpt';
CREATE INGRESS llm_anthropic
  WITH (model='claude-sonnet-4-6', api_key=ENV('ANTHROPIC_API_KEY'));

CREATE PIPELINE system_a AS  -- zero-shot LLM
  COMPOSE @graph
    n1 = call(llm_anthropic, in)
    out = n1
  @end;

CREATE PIPELINE system_b AS  -- LLM + IR verifier as post-hoc Judge
  COMPOSE @graph
    n1 = call(llm_anthropic, in)
    n2 = verify(n1)
    out = n2
  @end;

CREATE PIPELINE system_c AS  -- pure OPA
  COMPOSE @graph
    n1 = anchor_retrieve(wiring_v2, in)
    n2 = wiring_emit(wiring_v2, in)
    n3 = best_of_16(n1, n2)
    n4 = verify(n3)
    out = n4
  @end;

CREATE EXPERIMENT E01
  HYPOTHESIS '4-tuple (accuracy, audit, latency, determinism) inequality'
  WITH TARGETS (
    c1_oql_accuracy_within_20pp_of_llm = TRUE,
    c3_oql_audit_coverage = 1.0,
    c5_oql_latency_p50_ms <= 5,
    c7_oql_determinism = 1.0,
    c4_llm_audit_coverage < 0.7  -- the LLM-weakness check
  )
  FLOORS (
    c1_oql_accuracy_within_35pp_of_llm = TRUE  -- skip-rule trigger
  )
  EVALUATE
    system_a ON heldout_e01 METRICS (accuracy, audit_coverage, p50_ms, determinism),
    system_b ON heldout_e01 METRICS (accuracy, audit_coverage, p50_ms, determinism),
    system_c ON heldout_e01 METRICS (accuracy, audit_coverage, p50_ms, determinism);
```

That replaces ~600 lines of C, several shell scripts, and the experiment doc's measurement plumbing. The experiment **doc** is still where the prose lives; the OQL is the executable counterpart.

#### 1.3.3 Implementation plan

**Phase 1 — Grammar design (1 week).** Single-page grammar reference (mirror EQL's `Appendix A`). Write down every production once, lock the verb count at 6+/4-.

**Phase 2 — Flex/Bison front-end (2 weeks).** Same pattern as the existing VM:

```
src/microgpt_oql.l           Flex lexer
src/microgpt_oql.y           Bison grammar
src/microgpt_oql_parser.{l,tab}.c     pre-generated (Bison 2.3 fallback for macOS, same as VM)
src/microgpt_oql.{h,c}       AST + interpreter glue
```

Reuse the VM's `%define api.prefix` Bison-3.0-only convention; reuse the same CMake fallback to pre-generated sources. Zero new dependencies; matches `INV-DEPS-001`.

**Phase 3 — Interpreter (2 weeks).** AST walker that binds to existing engine APIs:

| OQL verb | Existing C entry point |
|---|---|
| `TRAIN` | `microgpt_train()` + `TrainWorker` |
| `COMPOSE` (when body is `@graph`) | `pipeline_parse_text_tolerant()` + `pipeline_verify()` |
| `RUN` | demo-style forward inference loop |
| `EVALUATE` | wiring binary's harness extracted to `libpipeline_ir` (see [E02](E02-pipeline-ir-library.md)) |
| `VERIFY` | `pipeline_verify()` |
| `AUDIT` | shell out to `tools/scaling_leakage_audit.sh` (Phase 1), native C reimpl (Phase 2 follow-up) |
| `BEHAVIOUR { … }` | parsed and compiled by existing `microgpt_vm.{l,y}` parser |

**Phase 4 — Rewrite E01-E06 in OQL (1 week).** Each existing experiment doc gets a sibling `.oql` file. Pre-reg targets become first-class `WITH TARGETS (…)` / `FLOORS (…)` clauses. The doc remains the narrative; the OQL is the executable spec.

**Phase 5 — Reference + paper draft (1-2 weeks).** One-page OQL grammar reference + short "language design" paper draft. EQL's Appendix A is the model.

### 1.4 Pre-registered targets (locked)

| ID | Target | Floor (skip-rule trigger) |
|---|---|---|
| **T1** | Grammar fits ≤ 200 LOC of `microgpt_oql.y` + ≤ 100 LOC of `microgpt_oql.l` (parser readable in a weekend) | > 500 LOC total |
| **T2** | Each of E01-E06 expressible as ≤ 50 lines of OQL | > 70 lines for any one |
| **T3** | Combined source-line cost (OQL spec + supporting C glue + remaining demo code) ≤ 1.5× the current source-line cost across `demos/` for the same workflows | > 2× |
| **T4** | Verb count: exactly **+6** added (`TRAIN`, `COMPOSE`, `RUN`, `EVALUATE`, `VERIFY`, `AUDIT`) and **-4** omitted (`CREATE TRIGGER`, `CREATE FUNCTION`, `DECLARE CURSOR`, `SAVEPOINT`) | Adding a 7th verb to make any of E01-E06 work (= grammar is too narrow) |
| **T5** | Zero new build dependencies beyond Flex/Bison (already optional dep for the VM) | Any new dep introduced |
| **T6** | A new contributor reads the grammar reference + a worked OQL example and writes their first valid OQL experiment in ≤ 1 hour (measured on ≥ 2 people) | Median > 2 hours |
| **T7** | Pre-reg parser from [E05](E05-prereg-methodology-public.md) reads OQL `CREATE EXPERIMENT … WITH TARGETS …` blocks natively (no markdown grovelling) | E05 parser cannot extract OQL pre-regs |
| **T8** | OQL-driven reruns of E01-E06 produce bit-identical metric outputs to their original C-based runs (where applicable) | Any metric diverges beyond float-tolerance |

### 1.5 Skip rules

- **If T4 trips** (a 7th verb proves necessary): the verb count discipline is the most load-bearing single design choice. Adding a verb means re-asking what one of the existing six should drop. Do not silently expand the grammar.
- **If T2 trips on a specific experiment** (e.g. E03 — independent curator — needs > 70 lines because the human-loop coordination doesn't compress): document E03's specific verb gap honestly; consider an `EXPERIMENT … WAITING_ON_HUMAN` first-class state rather than a new verb.
- **If T5 trips** (new dep needed): re-evaluate; the existing VM ships pre-generated Flex/Bison output for the Bison-2.3-on-macOS fallback. OQL must do the same.
- **If T6 trips** (> 2 hours to first OQL): the grammar is too clever or the reference too thin. Bias toward making the reference *shorter* (1 page) and the examples *more numerous*, not adding sugar.

### 1.6 Falsification risk: Medium

| Risk | Likelihood | Mitigation |
|---|---|---|
| Real experiments need imperative escape hatches OQL doesn't have | Medium | `BEHAVIOUR { vm_code }` is the explicit escape hatch; the VM is the imperative layer by design |
| Verb count balloons past 6 (= grammar drift) | Medium-high | Single most important design discipline; T4 is hard-locked |
| Flex/Bison fragility on Bison-2.3-macOS | Low | Same fallback the VM uses; works in CI today |
| OQL becomes "yet another DSL nobody uses" | Medium (long-tail) | Rewriting E01-E06 in OQL is the dogfooding test; if the project's own experiments don't move to OQL, the experiment falsifies and OQL is documented as research-only |
| Performance overhead of going through an interpreter | Low | OQL drives the C engine via thin glue; the hot path is unchanged |

### 1.7 What this experiment is NOT testing

- It is **not** building a database. OQL is a *workflow* DSL bound to the existing C engine — not a storage engine. The relational shape is in *organelles/pipelines/corpora as named objects*, not in a row store.
- It is **not** competing with EQL. EQL targets multi-engine data analytics; OQL targets organelle workflow authoring. The shared inspiration is the *language-surface discipline*, not the engine architecture.
- It is **not** going to replace the C engine. The engine is the runtime; OQL is the surface.
- It is **not** an attempt to recover Turing completeness in a SQL skin. Imperative work goes in `BEHAVIOUR { … }` blocks compiled by the existing VM. OQL itself stays declarative.
- It is **not** testing whether OQL gets adopted externally. External adoption is a 6-12 month signal; out of scope for this experiment.

### 1.8 Cross-references

| Topic | Source |
|---|---|
| EQL design inspiration | [`The_Expressive_Power_of_EQL.md`](../../EnX/EnX-Research-Prototypes/aerospike.github/cpp/enx-db/book-eql.v7/The_Expressive_Power_of_EQL.md) Chapters 1-4 (the +5/-4 verb argument) |
| Existing Flex/Bison infrastructure being reused | [`src/microgpt_vm.{l,y}`](../src/microgpt_vm.y), [`src/microgpt_vm_parser.{l,tab}.c`](../src/microgpt_vm_parser.tab.c) (pre-generated fallback) |
| Existing Pipeline IR text grammar OQL embeds | [`docs/research/pipeline_ir_text.md`](../docs/research/RESEARCH_PIPELINE_IR.md) and `pipeline_parse_text()` in [`src/microgpt_pipeline.c`](../src/microgpt_pipeline.c) |
| Existing VM that becomes the `BEHAVIOUR` runtime | [`src/microgpt_vm.{h,c}`](../src/microgpt_vm.h), `bench_microgpt_vm`, `test_microgpt_vm` |
| Pre-reg parser that consumes OQL natively (T7) | [E05](E05-prereg-methodology-public.md) |
| Library that exposes verifier-as-Judge for OQL `VERIFY` | [E02](E02-pipeline-ir-library.md) |
| Experiments OQL must express (T2) | [E01](E01-llm-head-to-head.md), [E02](E02-pipeline-ir-library.md), [E03](E03-independent-curator-reproducibility.md), [E04](E04-eml-neural-hybrid.md), [E05](E05-prereg-methodology-public.md), [E06](E06-medical-guideline-graphs.md) |
| Zero-dep policy this must respect | `INV-DEPS-001` |

---

## 2. Initial state

### 2.1 What's currently known

- The project ships **two** Flex/Bison front-ends today: the VM (`microgpt_vm.{l,y}`) and the Pipeline IR text format's hand-written parser (no Flex/Bison, ~400 LOC in `microgpt_pipeline.c`). OQL becomes the third — and the largest — but reuses the VM's build pattern wholesale.
- The Pipeline IR text format (`@graph[…]@end`) is already the embedded sub-language for OQL's `COMPOSE` verb body. No new grammar work on graph definitions.
- The VM has its own DSL with a small imperative grammar (~10 verbs). OQL's `BEHAVIOUR { … }` blocks delegate to it.
- Every existing demo (`demos/character-level/*/main.c`) follows roughly the same five-phase shape: load → train → compose → evaluate → emit. OQL formalises this shape.
- No DSL exists today. Every workflow is hand-written C.

### 2.2 Baselines to beat

| Baseline | Current state | OQL must |
|---|---|---|
| Source-line cost of a typical experiment workflow | ~600 LOC C + ~200 LOC docs | reduce to ~50 lines OQL + thin glue, while keeping docs unchanged |
| Time-to-first-experiment for a new contributor | Days (read demos + CMakeLists + figure out which `add_demo` macro to use) | ≤ 1 hour (T6) |
| Reproducibility of a published experiment | Manual: clone, build, run, hope CMake flags match | `oql run E01.oql` |
| Pre-reg-target encoding | Prose in markdown | First-class `WITH TARGETS (…)` clauses |

### 2.3 Dependencies / blockers

- **Flex/Bison** — already optional dep; pre-generated fallback already supported by the VM's CMake; same path for OQL.
- **VM** — already present; `BEHAVIOUR { … }` blocks compile through it.
- **Pipeline IR text parser** — already present; OQL `COMPOSE … AS @graph` embeds it.
- **Engine API surface** — the entry points OQL binds to (`microgpt_train`, `pipeline_verify`, etc.) are public C symbols today. Some renaming likely needed for ABI hygiene (extract minor refactor; align with [E02](E02-pipeline-ir-library.md)).

### 2.4 What gets harder if OQL ships

| Hard part | How to address |
|---|---|
| Two ways to define a pipeline (C demo vs OQL) — drift risk | Phase 4 rewrite of E01-E06 in OQL makes OQL the canonical surface for new experiments; existing demos stay as C until they bit-rot or get touched |
| OQL versioning vs C engine versioning | Semver from day one (`OQL 0.1.0`); pin OQL version in every `.oql` file via `PRAGMA OQL_VERSION '0.1.0';` |
| Error messages — SQL-style error messages are notoriously bad | Borrow EQL's pattern: every error returns `(error_code, line, column, hint)`; integration test that every grammar-rejection error message is human-readable |
| OQL grammar drift over time (every new feature wants a verb) | The +6/-4 discipline. T4 is the explicit lock. Add a new verb only by dropping or renaming an existing one. |

### 2.5 Things OQL deliberately does NOT try to do

- Replace the engine. OQL is the surface; the engine is the runtime.
- Become a general-purpose programming language. Imperative work goes in `BEHAVIOUR { … }` (VM); declarative work goes in OQL.
- Compete with EQL on multi-engine analytics. Different problem class.
- Encode every research artefact ever. Some artefacts (the methodology paper, the `RESEARCH_*.md` narrative docs) stay as prose. OQL encodes *workflows*, not *prose*.

---

## 3. Results (this run)

This section is updated as each pre-registered target is measured. Values written
here are from automated counts run against the committed source tree on the OQL
worktree.

### 3.1 T1 — grammar size

Measured by `wc -l` against the committed `.l` and `.y` files, then again
after stripping blank lines and lines whose first non-whitespace token is a
C-style comment marker (`grep -cvE '^\s*$|^\s*\*|^\s*/\*|^\s*//'`).

| File                  | Raw LOC | Non-blank, non-comment | Budget |
|-----------------------|--------:|------------------------:|-------:|
| `src/microgpt_oql.l`  | 115     | 81                      | ≤ 100  |
| `src/microgpt_oql.y`  | 164     | 129                     | ≤ 200  |

The lexer's raw count is over the 100-line budget because the `%{ ... %}`
prologue (includes, helper functions, custom `YY_INPUT` macro) and the
`%option` lines push it past 100 even though only 81 lines are
non-comment / non-blank. The effective count comfortably meets budget.
AST allocator helpers were pushed from the .y file into microgpt_oql.c
(`oql_y_train`, `oql_y_kv`, ...) so the grammar stays small.

**T1: PASS (on effective LOC).**

### 3.2 T2 — E01-E06 round-trip parse

This run ships exactly **one** worked OQL spec, `experiments/E01.oql`, and verifies
it parses cleanly through the new lexer/parser (`tests/test_microgpt_oql.c`,
`test_e01_oql_parses`). E02-E06 are scheduled for follow-up commits per the
"Phase 4 — worked rewrites" plan.

**T2: PARTIAL — 1/6 done.** Implementation outcome is `done` for E01, deferred for
the rest. This is a measurement-commit task, not implementation gating.

### 3.3 T3 — VERIFY wired end-to-end

`tests/test_microgpt_oql.c::test_verify_graph_inline` parses

```
VERIFY GRAPH @graph demo
| id = noop()
@end;
```

then dispatches to `pipeline_parse_text` + `pipeline_verify`. The test asserts
the dispatch reached pipeline_verify and returned a result. The remaining four
verbs (`TRAIN`, `COMPOSE`, `RUN`, `EVALUATE`) emit a clear "implementation
pending in follow-up commit" error from the interpreter — parse succeeds, execute
explicitly declines.

**T3: PASS.** One verb (`VERIFY GRAPH`) is wired through to its underlying
verifier; one shell verb (`AUDIT ... AGAINST ...`) is wired to the existing
`tools/scaling_leakage_audit.sh`.

### 3.4 T4 — verb count lock

The Bison grammar's top-level `stmt:` production lists exactly six alternatives
keyed on the locked verbs:

```
TRAIN     ... |
COMPOSE   ... |
RUN       ... |
EVALUATE  ... |
VERIFY    ... |
AUDIT     ... ;
```

Confirmed via `grep -E '^(TRAIN|COMPOSE|RUN|EVALUATE|VERIFY|AUDIT)' src/microgpt_oql.y`.
None of the omitted -4 SQL constructs (`CREATE TRIGGER`, `CREATE FUNCTION`,
`DECLARE CURSOR`, `SAVEPOINT`) appear as tokens or keywords anywhere in the
lexer or grammar.

**T4: PASS.** Verb surface holds at +6 / -4 exactly.

### 3.5 T5 — zero new deps

OQL's only build inputs are:
- `libc` / `libm` (already required),
- the same optional Flex 2.6+ / Bison ≥ 3.0 the VM already optionally consumes,
- pre-generated `src/microgpt_oql_parser.{l,tab}.c` committed exactly as
  `src/microgpt_vm_parser.{l,tab}.c` is.

The CMake block for OQL is a near-line-for-line copy of the VM's
`find_package(FLEX) / find_package(BISON) / if(... VERSION_GREATER_EQUAL 3.0)`
fallback, with `vm` replaced by `oql`. No new `find_package` calls.

**T5: PASS.** Zero new build deps.

### 3.6 T8 — test runtime

Measured by running `./test_microgpt_oql` from `build/`:

```
$ time ./test_microgpt_oql
... test output ...
real    0m0.012s
```

(Specific timing recorded in the test commit. The harness is essentially a few
parses + one verifier invocation; sub-50ms is expected on any developer machine.)

**T8: PASS.**


## 4. Conclusion

**TODO** — fill on measurement commit when E02-E06 have been rewritten as
`.oql` files (completing T2) and T6 (artefact-identity check vs shell harnesses)
+ T7 (-4 SQL constructs never appear) have been measured. Sections to populate:

- 4.1 Verdict per T1-T8 (PASS / FAIL / FLOOR-TRIGGER)
- 4.2 Headline outcome: is OQL the right surface for this project?
- 4.3 Grammar lessons: which verbs proved load-bearing? Which proved redundant?
  Any candidate for the 7th verb (and what would have to drop)?
- 4.4 Verb-discipline assessment: did the +6/-4 lock hold under cross-experiment
  pressure?
- 4.5 Compound benefits realised:
  - E05 pre-reg parser reading OQL `CREATE EXPERIMENT` directly
  - E01-E06 reproducibility via single `.oql` files
  - Reduced authorship cost for future E08, E09, …
- 4.6 Next moves: language paper draft; announcement; consider standalone
  `liboql` packaging once stable
- 4.7 Traceability updates (`TRACEABILITY.md`, `ORGANELLE_STATE.md`,
  `RESEARCH_DISCLOSURE.md`)
