# TDD_pipeline_ir — Technical Design Document

**Document ID:** TDD-PIPE-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_pipeline_ir.md`
**Sources:** `src/microgpt_pipeline.{h,c}`

## 1. Overview

The Pipeline IR module is an orthogonal optional layer over the core engine — it requires no changes to `microgpt.c` and adds no runtime cost when not used. It defines a typed-graph IR (`Pipeline`, `PipelineNode`, `PipelineEdge`, `PipelineType`, `PipelineValue`, `PipelineConfig`) plus a verifier, a tolerant parser, a graph repair pass, a callback-based executor, a VM-backed dispatcher, a text serialiser, and a DOT visualiser.

The IR is the canonical artefact a future "Wiring Organelle" emits: instead of free-form code, the model produces typed graph constructions which the verifier checks before execution. The verifier is the *deterministic Judge* that protects the system from low-confidence model outputs.

## 2. Architecture

```
   NL prompt           (wiring organelle / human author)
         │
         ▼
    @graph text  ──pipeline_parse_text(_tolerant)──►  Pipeline (unverified)
                                                         │
                                                ┌────────┴───────────┐
                                                ▼                    ▼
                                       pipeline_repair          pipeline_verify
                                          (drop deads)         (full check + topo sort)
                                                                       │
                                                                       ▼
                                                            verified Pipeline
                                                                       │
                              ┌────────────────────────────────────────┤
                              ▼                                        ▼
                    pipeline_render_text                  pipeline_render_dot
                    (canonical, topo-ordered)             (GraphViz visualisation)
                              │                                        │
                              ▼                                        ▼
                       text round-trip                          *.svg via `dot`

   (separately) verified Pipeline ── pipeline_execute(dispatch, user_data) ──► outputs
                                  ── pipeline_execute_vm(vm)                ──► outputs
```

## 3. Data flow — verifier

`pipeline_verify(p)` runs eight checks in order, on first call only (subsequent calls are idempotent):

1. Every node id is unique.
2. Every edge endpoint references an existing node and port.
3. Every input port has exactly one incoming edge OR is connected to a signature-input.
4. Every signature-input is connected to at least one node port.
5. Every signature-output is connected to exactly one node port.
6. Edge types match (src port type == dst port type, modulo `ANY`).
7. Graph is acyclic (DFS-based cycle detection).
8. Topological sort populates `p->exec_order`.

On success, `verified = 1` and the cached `exec_order` makes subsequent executions O(n_nodes). Mutating the graph after verify resets `verified` to 0.

`pipeline_verify_partial(p, &missing)` runs the same checks but treats dangling input ports, unconnected signature outputs, and unused signature inputs as *recoverable warnings* — useful for incremental construction by the wiring organelle.

`pipeline_repair(p, &report)` iteratively drops nodes whose input ports cannot be satisfied (their sources don't exist or were killed in a prior round), then drops edges touching dead nodes, then disconnects signature outputs whose sources became dead. Repair is a pure subtraction pass — it MUST NOT add nodes, edges, or signature ports.

## 4. Key data structures

### 4.1 `PipelineType`

```c
typedef struct PipelineType {
  PipelineTypeKind kind;
  struct PipelineType *element_type;  /* LIST/TENSOR */
  int n_dims; int *dims;              /* TENSOR; -1 = wildcard */
  PipelineRecordField *fields; int n_fields; /* RECORD */
} PipelineType;
```

`PipelineTypeKind` is one of `VOID, INT, FLOAT, STRING, LIST, TENSOR, RECORD, ANY`. Composite types own their child types via `pipeline_type_clone` / `_free`. Equality (`pipeline_type_equal`) is structural; `ANY` matches anything; tensor wildcard `-1` matches any concrete dim.

### 4.2 `PipelinePort`

Each port carries a name, type, and a list of edges. Input ports must end up with exactly one incoming edge after verify; output ports may have any number of outgoing edges (fan-out).

### 4.3 `PipelineEdge`

Endpoints by node + port *index* (not name) for O(1) traversal. The edge's type is an alias of the source port's type after verify; equality with the destination port's type is what gets checked.

### 4.4 `PipelineNode`

Either a leaf (`primitive` non-NULL, `subgraph` NULL) or a composition (`subgraph` non-NULL). The two are mutually exclusive. Leaves are dispatched by name through the host's `PipelineDispatchFn` (or via the VM's `vm_engine_register_fn`); compositions execute their subgraph recursively via `pipeline_execute`.

`PipelineConfig` is a name + tagged-union for INT / FLOAT / STRING node parameters (SysML "value properties").

### 4.5 `Pipeline`

The graph itself: nodes, edges, signature-in/out ports, cached `exec_order`. The signature defines the graph-as-block I/O contract, which is what makes a graph composable as a child node in a parent graph.

## 5. Algorithms

### 5.1 Cycle detection (verifier check 7)

DFS with three-colour (white / grey / black) marking. White = unvisited, grey = on the current DFS path, black = fully explored. A grey-targeting edge is a back edge → cycle.

### 5.2 Topological sort (check 8)

Reverse-postorder of the DFS produces a topological order; `p->exec_order` is the index sequence.

### 5.3 Text serialiser

`pipeline_render_text(p)` walks `exec_order` (or insertion order for unverified graphs), emitting the canonical form documented in `FS_pipeline_ir_text.md`. The serialiser:

- Emits all signature inputs first, then all signature outputs.
- For each node, writes `| id = primitive(arg_list)` then optionally a `:: type-annotation` and `# config` block.
- For each input port, looks up the incoming edge via `find_incoming_edge(p, idx, ip)` and renders `port_name: <source>`.
- After all nodes, emits one `out_name <- node.port` line per connected signature output.
- Terminates with `@end\n`.

### 5.4 Strict parser

`pipeline_parse_text(src)` is a hand-rolled recursive-descent parser keyed off the grammar in `FS_pipeline_ir_text.md` §3. Lexer state is tracked in `PState { src; cur; line; col; }`. Errors populate a thread-local message buffer accessible through `pipeline_last_error()`.

### 5.5 Tolerant parser

`pipeline_parse_text_tolerant(src)` re-runs the strict parser with three repair hooks:

1. Duplicate `: in <name> -> <type>` lines silently dedup.
2. `<sig_name>` arg references with no declared signature input auto-promote.
3. `<sig_name> <- node.port` bindings with no declared signature output auto-promote.

The repaired parse may still fail subsequent `pipeline_verify` (type mismatches, cycles); the tolerant parser only addresses *declaration / reference* mismatches.

### 5.6 DOT renderer

`pipeline_render_dot(p)` emits GraphViz DOT format: each node is a record-shaped box with input ports on top and output ports on bottom; edges are labelled with their type; signature ports are ellipses at the boundary. Pipe through `dot -Tsvg foo.dot > foo.svg` to render.

### 5.7 Executor

`pipeline_execute(p, inputs, outputs, dispatch, user_data)` walks `exec_order`:

```
for idx in exec_order:
    node = p->nodes[idx]
    materialise input PipelineValues from incoming edges
    if node->subgraph:
        recursive call to pipeline_execute(subgraph, ...)
    else:
        dispatch(node->primitive, node->config, inputs, outputs, user_data)
    propagate output PipelineValues along outgoing edges
materialise final outputs from edges into the caller's signature_out array
```

`pipeline_execute_vm(p, vm, inputs, outputs)` (V1.0.4 — `GAP-PIPE-003` RESOLVED) resolves each leaf node's `primitive` to a `vm_native_fn` registered via `vm_engine_register_fn` (looked up through the new public `vm_engine_find_fn`) and dispatches it. The VM's native ABI is `double(int argc, const double *argv)`, so this path is restricted to INT / FLOAT / VOID-typed pipelines; ports of type STRING / LIST / TENSOR / RECORD cause `PIPE_ERR_EXEC` with a message that identifies the offending node and port name.

Implementation lives in the opt-in TU `src/microgpt_pipeline_vm.c`, which links the VM library. The core `microgpt_lib.a` does NOT link the VM — demos / tests that call `pipeline_execute_vm` add `microgpt_pipeline_vm.c` plus `microgpt_vm.c` (and the parser sources `microgpt_vm_parser.l.c` / `.tab.c`) to their target. This split keeps the core lib's footprint minimal for the many demos that don't use VM dispatch.

The two TUs share a small private header `src/microgpt_pipeline_internal.h` that exposes `mgpt_pipe_set_err` and `mgpt_pipe_find_incoming_edge` (thin wrappers around the file-static helpers in `pipeline.c`). That header is NOT part of the public API.

## 6. Concurrency model

A `Pipeline *` is single-owner — construction, verify, mutation are not thread-safe. Once verified, the graph is logically immutable and `pipeline_execute` is safe to call concurrently with the same dispatch function so long as the dispatch function itself is thread-safe with respect to `user_data`.

`pipeline_last_error()` is thread-local — each thread has its own error buffer.

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Construction vs verification | Two phases | One-pass | Construction order is unknown to the wiring organelle; deferring all checks to a verify pass means the model can emit graphs in any order without the IR rejecting work-in-progress states. |
| Tolerant parser | Three named repairs | Forgive everything / forgive nothing | The three repairs match the most common organelle generation incoherences observed in `RESEARCH_PIPELINE_IR.md`; they are local enough to be auditable. |
| Text format ordering | Topological order for verified graphs | Insertion order | Topological order makes the canonical form a deterministic fingerprint — same graph → same string. |
| Type-annotation suffix | Optional, only when non-`ANY` types present | Always emit | Round-trips structurally even for ANY-only graphs; suffix is ergonomic only when types matter. |
| VM backend | Convenience API for INT/FLOAT pipelines | Full reflection of VM types | The VM's native ABI predates the Pipeline IR; bridging string/list/tensor/record types into the VM is a Phase-3 effort and out of scope for V1.0. |

## 8. Known limitations

- The VM-backed dispatcher (`pipeline_execute_vm`) is INT/FLOAT-only.
- The DOT renderer does not lay out subgraphs as nested clusters; a composition node renders as a single record box. Visualising the inner graph requires recursing manually.
- The error buffer is thread-local but globally singleton per thread — concurrent error retrievals from independent verify operations on different graphs will overwrite each other.
- No on-disk binary form for compiled pipelines (the format is text only); a future binary form is left as future work.

## 9. References

- `docs/research/RESEARCH_PIPELINE_IR.md` — 17-phase development log.
- `docs/research/RESEARCH_WIRING_ORGANELLE_PAPER.md` — formal v2.0 paper.
- SysML proxy ports as the inspiration for graph-level signatures.

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
