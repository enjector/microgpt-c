# Pipeline IR — Phase 1: Graph-based Call & Data Flow

> A SysML-flavoured graph IR for representing computations as directed graphs of typed nodes and dataflow edges. Designed as the target output of a future "Wiring Organelle" — a tiny model that emits graph constructions instead of free-form code, with verification (type-check + cycle-check + connectivity-check) acting as a deterministic Judge before execution. **Phase 1 ships the IR, verifier, text round-trip, DOT renderer, and a callback-based executor — all in pure C99 with libc + libm only.**

**Reference:** Conversation transcript proposal for "graph based call and data flow pipeline system that can perform data transformation and call composition, also like SysML model"; the project's standing thesis from `demos/README.md` that *single micro-models are retrieval systems, not generators* and *composition belongs in the pipeline, not the model*.

**Status:** Phase 1 shipped. `src/microgpt_pipeline.{h,c}` (~1900 LOC), `tests/test_microgpt_pipeline.c` (24 tests, all passing), CMake wiring complete. All 61 core unit tests + 3 MSA tests + 24 pipeline tests pass on a fresh build. No source changes outside the new module.

---

## 1. Spear Summary

**Point:** A directed graph IR of typed nodes and edges replaces the current "model emits free-form code text" composition strategy with "model emits graph constructions verified before execution." Construction is a small finite vocabulary (`add_node`, `connect`, `set_signature`, `set_config`); validity is checked locally and incrementally; invalid graphs are rejected before they can produce wrong outputs. This matches the project's **composition-in-the-pipeline-not-the-model** thesis at the IR level.

**Picture:** Today, the VM Codegen organelle generates `function fib(n: number): number { ... }` token-by-token, where the entire token sequence has to be syntactically valid for the output to compile. Every position is a precise next-token prediction problem in a vocabulary of ~1000 word tokens. With a graph IR, the model instead emits decisions like "add a `mean()` node with input port `x` connected to `signal.out`" — a smaller, more structured search space. Type-checked edges are the equivalent of the Connect-4 Judge that took win-rate from 55% to 88% by catching invalid moves before they ruined the game.

**Proof:** The Phase 1 deliverable is not yet measured against the wiring problem (Phase 3 work) — but the IR itself is verified end-to-end:

| Property | Status |
|---|---|
| Construction API (add_node, connect, set_signature, configs) | ✅ 4 tests |
| Type system (constructors, equality, ANY-polymorphism, deep clone, format) | ✅ 7 tests |
| Verifier (cycles, dangling ports, type mismatch, signature) | ✅ 7 tests |
| Executor (callback dispatch, multi-node chains, signature-output propagation) | ✅ 2 tests |
| Text round-trip (render → parse → structurally-equivalent graph) | ✅ 2 tests |
| DOT renderer (GraphViz output) | ✅ 1 smoke test |
| Last-error reporting | ✅ 1 test |
| **All 24 pipeline tests pass** | ✅ |
| **No regressions in existing 64 tests** | ✅ |

**Push:** Phase 1 lands the IR. Phase 2 will lower leaf primitives to VM calls so existing `vm_codegen` corpus functions become first-class graph nodes. Phase 3 trains a Wiring Organelle on `(prompt, graph-text)` pairs, with incremental verification rejecting invalid construction operations the way `OpaCycleDetector` rejects A↔B oscillations in the existing organelle pipeline. Phase 4 adds SysML multi-view rendering and the headline benchmark.

---

## 2. Mechanism

### 2.1 What the IR represents

A `Pipeline` is a named directed graph with:
- **Nodes** (`PipelineNode`): a unique id, a primitive name (resolved at execute time) **or** a nested subgraph for recursive composition, typed input/output **ports**, and optional **config** values (SysML "value properties" — constants set per instance, distinct from dataflow inputs).
- **Edges** (`PipelineEdge`): from one node's output port to another node's input port. Edges carry typed values; the type is derived from the source port at construction and verified equal to the destination port at `pipeline_verify` time.
- **Signature**: the graph's external I/O contract — proxy ports through which the graph can be wired into a parent graph as a single node. Recursive composition is the rule, not an exception: any verified pipeline becomes a usable subgraph.

### 2.2 Type system

Structural, with one polymorphic placeholder:

```
void  int  float  string                       — scalars
list[T]                                         — homogeneous list of T
tensor[T, dim_0, dim_1, …]                      — typed tensor with shape
record{name: T, age: int, …}                    — struct-like field map
any                                              — polymorphic, matches anything
```

`ANY` is the parser's friend — the round-trip path uses it for ports whose concrete types weren't recovered from the textual form (Phase 1 doesn't yet preserve port types in the canonical text output; that's deliberate to keep parsing minimal). At `verify` time, `ANY` matches anything, so an `ANY`-port graph still passes type-check structurally.

Tensor wildcard dimensions (`-1`) match any concrete dim — useful for "any batch size" specifications. RECORD fields use ordered structural equality (same field names in same order with matching types).

### 2.3 Verification (the deterministic Judge)

`pipeline_verify(p)` runs eight checks in order, returning a specific error code naming the offending element:

1. **Unique node ids** — duplicates rejected (`PIPE_ERR_DUP_NODE_ID`).
2. **Edge endpoints reference existing nodes/ports** (`PIPE_ERR_UNKNOWN_PORT`).
3. **Type matching on every edge** (`PIPE_ERR_TYPE_MISMATCH`) — error message includes both source and destination type printouts.
4. **Every input port has exactly one incoming edge** (`PIPE_ERR_DANGLING_PORT`) — both under-connected and over-connected ports rejected.
5. **Every signature output has exactly one feeder** (`PIPE_ERR_BAD_SIGNATURE`).
6. **Every signature input is used at least once** — unused signature inputs are a smell, rejected for hygiene.
7. **Topological sort with cycle detection** (DFS-based, three-coloured node states) — cycles rejected with `PIPE_ERR_CYCLE` naming the entry node.
8. **Cache `exec_order`** on success — subsequent verify calls are idempotent until mutation invalidates.

Error messages reference offending elements by id/name, not by index, so the Wiring Organelle (Phase 3) gets actionable feedback when it emits invalid construction operations. This is the key alignment with the planner-player-judge organelle pattern: invalid moves produce specific, structured rejections.

### 2.4 Execution

Phase 1 ships a **callback-based** executor — the host registers a `PipelineDispatchFn` that maps primitive names to native functions. The executor walks `exec_order`, gathers each node's inputs from incoming edges, invokes dispatch (or recurses into a subgraph), and propagates outputs along outgoing edges. Signature inputs are pre-loaded into the edges that consume them; signature outputs are pulled from the edges that feed them.

Subgraph nodes recurse via the same `pipeline_execute` call — composition is structural, not a separate code path.

Phase 2 will add a VM-backed dispatch that automatically lowers `primitive` to a `vm_call`, eliminating the need for hosts to register per-primitive callbacks. For Phase 1, the callback design is the right minimum viable surface: it lets the pipeline be embedded in any C program with the host's own primitives today.

### 2.5 Text format

A line-based, deterministic, round-trip-safe textual representation. The renderer always emits node lines in topological order, so the same verified graph produces the same text every time. Format:

```
@graph zscore_rolling
  : in signal -> tensor[float, *]
  : out result -> tensor[float, *]
  | n1 = rolling_mean(x: <signal>)            # node | id = primitive(port: src.port, …)
  | n2 = stddev(x: <signal>)
  | n3 = subtract(a: <signal>, b: n1.out)
  | n4 = divide(a: n3.out, b: n2.out)
  result <- n4.out                            # output binding
@end
```

The grammar is small — fewer than 30 tokens excluding identifiers and primitive names — designed to be tractable output for a small organelle model. Each `|` line is a complete node-construction operation; each `<-` line is a single binding.

### 2.6 DOT renderer

`pipeline_render_dot()` emits GraphViz format with port-aware record-shaped boxes (input ports on top, output ports on bottom), labelled edges showing the type carried, and signature ports as ellipses on the boundary. `dot -Tsvg foo.dot > foo.svg` produces a clean visual that matches a SysML internal block diagram.

---

## 3. Worked Examples

### 3.1 Simple add graph

```c
Pipeline *p = pipeline_create("add_graph");

/* Signature: in a:int, in b:int, out y:int */
const char *sig_in_names[]  = {"a", "b"};
PipelineType *sig_in_types[] = {pipeline_type_int(), pipeline_type_int()};
const char *sig_out_names[] = {"y"};
PipelineType *sig_out_types[] = {pipeline_type_int()};
pipeline_set_signature(p, 2, sig_in_names, sig_in_types,
                          1, sig_out_names, sig_out_types);

/* Node: add(x, y) -> out */
const char *in_names[]   = {"x", "y"};
PipelineType *in_types[] = {pipeline_type_int(), pipeline_type_int()};
const char *out_names[]  = {"out"};
PipelineType *out_types[] = {pipeline_type_int()};
pipeline_add_node(p, "n_add", "add",
                  2, in_names, in_types,
                  1, out_names, out_types);

pipeline_connect_signature_in(p, "a", "n_add", "x");
pipeline_connect_signature_in(p, "b", "n_add", "y");
pipeline_connect_signature_out(p, "n_add", "out", "y");

if (pipeline_verify(p) != PIPE_OK) {
    fprintf(stderr, "verify failed: %s\n", pipeline_last_error());
}

PipelineValue inputs[2] = {0}, outputs[1] = {0};
inputs[0].v.i = 3; inputs[1].v.i = 4;
pipeline_execute(p, inputs, outputs, dispatch_fn, NULL);
/* outputs[0].v.i == 7 */
```

Rendered text:

```
@graph add_graph
  : in a -> int
  : in b -> int
  : out y -> int
  | n_add = add(x: <a>, y: <b>)
  y <- n_add.out
@end
```

### 3.2 Multi-node arithmetic chain (`(a + b) * c`, then negated)

```c
Pipeline *p = pipeline_create("chain3");
/* Signature: in a, b, c → out y, all int */

pipeline_add_node(p, "n1", "add", …);   /* (a, b) → out */
pipeline_add_node(p, "n2", "mul", …);   /* (n1.out, c) → out */
pipeline_add_node(p, "n3", "neg", …);   /* (n2.out) → out */

pipeline_connect_signature_in(p, "a", "n1", "x");
pipeline_connect_signature_in(p, "b", "n1", "y");
pipeline_connect      (p, "n1", "out", "n2", "x");
pipeline_connect_signature_in(p, "c", "n2", "y");
pipeline_connect      (p, "n2", "out", "n3", "x");
pipeline_connect_signature_out(p, "n3", "out", "y");

pipeline_verify(p);  /* exec_order: [n1, n2, n3] */

/* a=2, b=3, c=5 → (2+3)*5 = 25, neg = -25 */
```

This is the test in `tests/test_microgpt_pipeline.c` `execute_chain_three_nodes`. The exec_order is automatically computed by topological sort; the host dispatcher only needs to know how to compute single primitives — the IR handles all wiring.

---

## 4. API Reference (summary — see microgpt_pipeline.h for full doc-comments)

```c
/* Type constructors */
PipelineType *pipeline_type_void/int/float/string/any(void);
PipelineType *pipeline_type_list(PipelineType *element_type);
PipelineType *pipeline_type_tensor(PipelineType *element_type,
                                   int n_dims, const int *dims);
PipelineType *pipeline_type_record(int n_fields,
                                   const char **field_names,
                                   PipelineType **field_types);
PipelineType *pipeline_type_clone(const PipelineType *t);
void          pipeline_type_free(PipelineType *t);
int           pipeline_type_equal(const PipelineType *a, const PipelineType *b);
int           pipeline_type_format(const PipelineType *t, char *buf, size_t buf_size);

/* Pipeline lifecycle */
Pipeline *pipeline_create(const char *name);
void      pipeline_free(Pipeline *p);

/* Construction */
int  pipeline_add_node(Pipeline *p, const char *id, const char *primitive,
                       int n_in,  const char **in_names,  PipelineType **in_types,
                       int n_out, const char **out_names, PipelineType **out_types);
int  pipeline_add_subgraph(Pipeline *p, const char *id, Pipeline *subgraph);
int  pipeline_connect(Pipeline *p,
                      const char *src_id, const char *src_port,
                      const char *dst_id, const char *dst_port);
int  pipeline_set_signature(Pipeline *p,
                            int n_in,  const char **in_names,  PipelineType **in_types,
                            int n_out, const char **out_names, PipelineType **out_types);
int  pipeline_connect_signature_in (Pipeline *p, const char *sig_in_name,
                                    const char *dst_id, const char *dst_port);
int  pipeline_connect_signature_out(Pipeline *p, const char *src_id,
                                    const char *src_port, const char *sig_out_name);
int  pipeline_node_set_config_int   (Pipeline *p, const char *id, const char *k, int64_t v);
int  pipeline_node_set_config_float (Pipeline *p, const char *id, const char *k, double v);
int  pipeline_node_set_config_string(Pipeline *p, const char *id, const char *k, const char *v);

/* Verify, execute */
int         pipeline_verify(Pipeline *p);
const char *pipeline_last_error(void);
int         pipeline_execute(const Pipeline *p,
                             const PipelineValue *inputs,
                             PipelineValue *outputs,
                             PipelineDispatchFn dispatch, void *user_data);
void        pipeline_value_clear(PipelineValue *val);

/* Render */
char *pipeline_render_text(const Pipeline *p);
Pipeline *pipeline_parse_text(const char *src);
char *pipeline_render_dot(const Pipeline *p);
```

**Error codes:** `PIPE_OK` (0), `PIPE_ERR_DUP_NODE_ID`, `PIPE_ERR_UNKNOWN_NODE`, `PIPE_ERR_UNKNOWN_PORT`, `PIPE_ERR_DANGLING_PORT`, `PIPE_ERR_TYPE_MISMATCH`, `PIPE_ERR_CYCLE`, `PIPE_ERR_BAD_SIGNATURE`, `PIPE_ERR_OOM`, `PIPE_ERR_PARSE`, `PIPE_ERR_EXEC`. Use `pipeline_last_error()` for human-readable messages.

---

## 5. Test Suite

`tests/test_microgpt_pipeline.c` — 24 tests across 7 sections, all passing on a fresh build:

```
[Pipeline IR — Type system]            (7 tests)
  type_constructors_basic              PASS
  type_equal_basic                     PASS
  type_any_matches_anything            PASS
  type_list_recursive                  PASS
  type_tensor_dims                     PASS
  type_clone_deep_copy                 PASS
  type_format_pretty_print             PASS

[Pipeline IR — Construction]           (4 tests)
  pipeline_create_and_free             PASS
  pipeline_add_node_basic              PASS
  pipeline_add_node_duplicate_id_rejected   PASS
  pipeline_connect_unknown_node_rejected    PASS

[Pipeline IR — Verification]           (7 tests)
  verify_simple_graph_passes           PASS
  verify_dangling_input_port_rejected  PASS
  verify_type_mismatch_rejected        PASS
  verify_cycle_rejected                PASS
  verify_signature_output_unconnected_rejected   PASS
  verify_signature_input_unused_rejected         PASS
  verify_topological_order_correct     PASS

[Pipeline IR — Execution]              (2 tests)
  execute_simple_add                   PASS
  execute_chain_three_nodes            PASS

[Pipeline IR — Text round-trip]        (2 tests)
  text_render_basic_does_not_crash     PASS
  text_round_trip_structural           PASS

[Pipeline IR — DOT renderer]           (1 test)
  dot_render_smoke                     PASS

[Pipeline IR — Error reporting]        (1 test)
  last_error_set_on_failure            PASS

=== Results: 24/24 passed ===
```

Plus: existing `test_microgpt` (61/61) and `test_microgpt_msa` (3/3) continue to pass — no regressions.

---

## 6. Phasing

This paper documents **Phase 1**. The full plan:

| Phase | Deliverable | Status |
|---|---|---|
| **1** | **IR + verifier + text round-trip + DOT + callback executor + tests** | **✅ Shipped (this paper)** |
| 2 | VM-backed dispatch — leaf primitives lower automatically to vm_call | Pending |
| 3 | Wiring Organelle: train on (prompt, graph-text) pairs, incrementally verify partial graphs | Pending |
| 4 | SysML multi-view rendering (block, internal block, activity, parametric) + headline benchmark | Pending |

Phase 1 is genuinely useful on its own — an embeddable graph IR with type-checked composition, a deterministic Judge, and visualisation support. Even if Phase 3's wiring-organelle hypothesis fails empirically, Phase 1 is reusable infrastructure: any future organelle that wants to emit verifiable compositions can target this IR.

---

## 7. Limitations and Future Work

1. **Phase 1 parser does not preserve port types.** The textual form emitted by `render_text` doesn't yet include port type annotations on node lines — only on the signature. The parser therefore reconstructs nodes with `ANY`-typed ports, which means the round-trip is **structurally** correct (same nodes, same edges, same primitives) but loses concrete port-type fidelity. Type fidelity will be added in Phase 2 when the textual form gains a node-type annotation syntax.

2. **Output binding name preservation in parsed graphs.** The Phase-1 parser uses positional matching of `<-` bindings to `signature_out` ports rather than name-resolved matching. Sufficient for canonical round-trip; brittle if a hand-edited file shuffles binding lines. Will be hardened in Phase 2.

3. **No VM lowering yet.** Leaf primitives are dispatched via host callback. Phase 2 adds a `vm_engine *` field to a new `pipeline_execute_vm()` variant that auto-resolves primitive names to `vm_module_compile`+`vm_call`.

4. **No multi-output primitive support tested under text format.** All test fixtures use single-output nodes. Multi-output should work mechanically (the data structure supports it; renderer emits `node.port` for each binding), but isn't covered by tests yet.

5. **No incremental verification for partial graphs.** Phase 3 needs `pipeline_verify_partial()` — accept a graph that's missing some signature-output bindings but check what's present. Not in Phase 1.

6. **No serialisation other than text + DOT.** SysML XMI fragments would be Phase 4 work for users who want round-trip with SysML tools. Not in Phase 1.

7. **Single-threaded executor.** Topologically-independent nodes could in principle be executed in parallel; Phase 1 doesn't bother. Worth doing in Phase 2 for graphs with multiple wide layers.

---

## 8. References

- Conversation transcript: user proposal for "graph based call and data flow pipeline system that can perform data transformation and call composition, also like SysML model" — the direct prompt for this work.
- Project's standing thesis: [`demos/README.md`](../../demos/README.md) — *"single micro-models are retrieval systems not generators... pipelines of specialised micro-models achieve what no single model can"*. This Phase 1 IR provides the structural pipeline that thesis requires.
- Composition prior art in this codebase:
  - [`RESEARCH_GENERALISATION.md`](RESEARCH_GENERALISATION.md) — VM generalisation research.
  - [`RESEARCH_ORGANELLE_PIPELINE.md`](RESEARCH_ORGANELLE_PIPELINE.md) — wire format design for organelle composition.
  - [`RESEARCH_ORGANELLE_PLANNER.md`](RESEARCH_ORGANELLE_PLANNER.md) — Kanban coordination protocol.
- SysML reference: OMG Systems Modeling Language v2 — for the multi-view abstraction (block, internal block, activity, parametric). Phase 1 doesn't yet implement multi-view rendering, but the data model is structured to support it cleanly.
- Implementation:
  - `src/microgpt_pipeline.h` — full public API with detailed doc-comments.
  - `src/microgpt_pipeline.c` — ~1900 LOC implementation.
  - `tests/test_microgpt_pipeline.c` — 24 tests covering the surface above.
  - `CMakeLists.txt` — `microgpt_pipeline.c` added to `microgpt_lib` and to the `_microgpt_lib_for_defines` factory; `test_microgpt_pipeline` target with CTest registration.

---

## 9. Closing Remark

The IR ships. 24 tests pass. The header has detailed doc-comments. The DOT renderer makes graphs human-readable. The text format is small enough for a tiny model to emit.

The real test of this work happens in Phase 3, when the Wiring Organelle attempts to generalise from a corpus of (prompt, graph) pairs to novel compositions. That experiment will succeed or fail on its own merits — but whatever the outcome, the IR is reusable infrastructure for any future composition strategy. The project's central thesis (composition in the pipeline, not the model) finally has a literal pipeline data structure to live in, with a verifier acting as the Judge.

If Phase 3 succeeds, the V4-port papers' standing claim — that microGPT-C demonstrates "tiny specialists, coordinated by a pipeline, outperform single models on focused tasks" — extends from games and Shakespeare to *programmable composition*. That's the bet this Phase 1 enables.

---

*Phase 1 ships. The IR is real, verified, and tested. Phase 2 begins where the test suite ends.*
