# BS_pipeline_ir — Behaviour Specification (Pipeline IR)

**Document ID:** BS-PIPE-001
**Version:** 1.0
**Status:** DRAFT

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Scope

Behavioural contract of the Pipeline IR: the type system (`PipelineType` and constructors), graph construction (`pipeline_create`, `_add_node`, `_add_subgraph`, `_connect`, `_set_signature`, `_node_set_config_*`), verification (`pipeline_verify`, `_verify_partial`, `_repair`), execution (`pipeline_execute`, `_execute_vm`), text serialisation (`pipeline_render_text`, `_parse_text`, `_parse_text_tolerant`), and DOT rendering (`pipeline_render_dot`).

## 2. Type contracts

### 2.1 `PipelineType`

**Invariants:**
- INV-PIPE-001: A type is one of `VOID, INT, FLOAT, STRING, LIST, TENSOR, RECORD, ANY`.
- INV-PIPE-002: A `LIST` type owns its `element_type`; a `TENSOR` type owns its `element_type` and `dims[]`; a `RECORD` type owns its `fields[]`. `pipeline_type_free` recursively frees these.
- INV-PIPE-003: `pipeline_type_clone` produces a deep copy.
- INV-PIPE-004: `pipeline_type_equal` is structural; `ANY` matches any type; tensor wildcard `-1` matches any concrete dim.

### 2.2 `PipelineNode`, `PipelineEdge`, `Pipeline`

**Invariants:**
- INV-PIPE-010: A node is either a leaf (`primitive` non-NULL, `subgraph` NULL) or a composition (`subgraph` non-NULL, `primitive` NULL); never both.
- INV-PIPE-011: An edge identifies endpoints by node + port index, not name (resolved at construction time).
- INV-PIPE-012: After `pipeline_verify` returns 0, `verified == 1` and `exec_order[]` holds a topological ordering.
- INV-PIPE-013: Mutating a graph after verify resets `verified` to 0. `pipeline_execute` and `pipeline_execute_vm` SHALL refuse an unverified graph (returning a negative error code with `pipeline_last_error()` set to "graph not verified"); they do NOT auto-re-verify. The caller MUST invoke `pipeline_verify(p)` again before re-executing.

## 3. Operation contracts

### 3.1 Construction primitives

`pipeline_create(name)` returns an empty unverified pipeline. `pipeline_free(p)` frees the graph and all owned types.

`pipeline_add_node(p, id, primitive, n_in, in_names, in_types, n_out, out_names, out_types)` adds a leaf; returns the new node index, or -1 on duplicate id / OOM. Each `PipelineType *` argument is OWNED by the node from this call onwards (the caller relinquishes ownership).

`pipeline_add_subgraph(p, id, subgraph)` adds a composition node and takes ownership of `subgraph`; the subgraph MUST be verified.

`pipeline_connect(p, src_id, src_port, dst_id, dst_port)` resolves names to indices and records the edge; type compatibility is deferred to verify.

`pipeline_set_signature(p, n_in, in_names, in_types, n_out, out_names, out_types)` replaces the signature, taking ownership of types.

`pipeline_connect_signature_in / _out` connect a signature port to a node port.

`pipeline_node_set_config_int / _float / _string` set a config K/V on a named node.

### 3.2 Verifier

`pipeline_verify(p)` runs the eight checks listed in `TDD_pipeline_ir.md` § 3 and returns `PIPE_OK` (0) on success or a negative `PIPE_ERR_*` code on failure.

`pipeline_verify_partial(p, &missing)` performs the same checks but treats:
- Dangling input ports as recoverable warnings.
- Unconnected signature outputs as recoverable warnings.
- Unused signature inputs as recoverable warnings.

The count of "missing" elements is written to `*missing` (may be NULL). `*missing == 0` is equivalent to a strict verify pass.

`pipeline_repair(p, &report)` iteratively drops nodes whose input ports cannot be satisfied; never adds. Writes a `PipelineRepairReport` summarising what was removed.

### 3.3 Executor

`pipeline_execute(p, inputs, outputs, dispatch, user_data)`:

**Preconditions:** `p` verified; `inputs` length == `p->n_sig_in`; `outputs` capacity ≥ `p->n_sig_out`.

**Postconditions:** Walks `exec_order`, materialises input values, calls `dispatch` for each leaf or recursively executes each subgraph, propagates outputs along edges. Writes the final signature outputs into `outputs[]`. Returns 0 on success or a negative `PIPE_ERR_*` code.

`pipeline_execute_vm(p, vm, inputs, outputs)` resolves each leaf node's `primitive` string via `vm_engine_find_fn(vm, name)` (added in V1.0.4 — see `BS_vm.md` `REQ-VM-007`), marshals each `PipelineValue` to a `double[]`, and calls the resolved `vm_native_fn`. Subgraph nodes recurse via the same function. INT/FLOAT/VOID port types only — STRING/LIST/TENSOR/RECORD ports cause `PIPE_ERR_EXEC` with a message identifying the offending node and port. A missing native returns `PIPE_ERR_EXEC` with a message naming the missing primitive and its containing node. Implementation lives in the opt-in TU `src/microgpt_pipeline_vm.c` (linked alongside `microgpt_vm.c`); `microgpt_lib` itself is unchanged. See `GAP-PIPE-003` (RESOLVED in V1.0.4).

### 3.4 Text serialisation

`pipeline_render_text(p)` returns a heap-allocated canonical string per the grammar in `FS_pipeline_ir_text.md`. Returns NULL on error.

`pipeline_parse_text(src)` is the strict inverse for verified-graph round-trips. Returns a NEW unverified pipeline (caller owns), NULL on parse error.

`pipeline_parse_text_tolerant(src)` applies the three named repairs (dedup signature inputs; auto-promote referenced inputs; auto-promote referenced outputs).

`pipeline_render_dot(p)` returns a heap-allocated GraphViz string.

`pipeline_last_error()` returns a pointer to a thread-local static buffer holding the most recent error message; never NULL.

## 4. Invariants table

| ID | Invariant |
|---|---|
| INV-PIPE-001..004 | Type system properties. |
| INV-PIPE-010..013 | Node / edge / pipeline structure. |
| INV-PIPE-020 | After `pipeline_verify` returns 0, executing the graph multiple times is deterministic (modulo dispatch-side state). |
| INV-PIPE-021 | The strict text parser is the right inverse of the renderer for verified graphs (round-trip property). |
| INV-PIPE-022 | The tolerant parser is a *superset* of the strict grammar — strict-parsable input is always tolerant-parsable to the same graph. |
| INV-PIPE-023 | `pipeline_repair` is monotone subtractive: `nodes_dropped + edges_dropped + sig_outs_disconnected + sig_ins_dropped + sig_outs_dropped` is non-negative; the residual graph is a subgraph of the input. |
| INV-PIPE-024 | `pipeline_execute_vm` is restricted to INT/FLOAT ports. |

## 5. Errors

| ID | Code | Symbol | Conditions |
|---|---:|---|---|
| ERR-PIPE-001 | -1 | `PIPE_ERR_DUP_NODE_ID` | Two nodes share an id |
| ERR-PIPE-002 | -2 | `PIPE_ERR_UNKNOWN_NODE` | Edge endpoint references missing node |
| ERR-PIPE-003 | -3 | `PIPE_ERR_UNKNOWN_PORT` | Edge endpoint references missing port |
| ERR-PIPE-004 | -4 | `PIPE_ERR_DANGLING_PORT` | Input port has no incoming edge (strict only) |
| ERR-PIPE-005 | -5 | `PIPE_ERR_TYPE_MISMATCH` | Edge type mismatch |
| ERR-PIPE-006 | -6 | `PIPE_ERR_CYCLE` | Graph contains a cycle |
| ERR-PIPE-007 | -7 | `PIPE_ERR_BAD_SIGNATURE` | Signature input/output mismatch |
| ERR-PIPE-008 | -8 | `PIPE_ERR_OOM` | Allocation failure |
| ERR-PIPE-009 | -9 | `PIPE_ERR_PARSE` | Parser error |
| ERR-PIPE-010 | -10 | `PIPE_ERR_EXEC` | Dispatch failure |

## 6. Concurrency

Construction is single-threaded per pipeline. Once verified, `pipeline_execute` is safe to call concurrently with the same dispatch function so long as the dispatch is itself thread-safe. `pipeline_last_error` is thread-local.

## 7. Scenarios

### SCN-PIPE-001: Build, verify, execute

A demo constructs a 5-node DAG with `pipeline_create / _add_node / _connect / _set_signature`, calls `pipeline_verify`, then `pipeline_execute` with a dispatch function that interprets the primitive names as standard arithmetic. Outputs are populated.

### SCN-PIPE-002: Round-trip

A test calls `pipeline_render_text(p)` on a verified graph, parses the result back with `pipeline_parse_text`, re-verifies, and confirms the second render is byte-equal to the first.

### SCN-PIPE-003: Tolerant parse + repair

A wiring-organelle output omits a `: in x -> int` declaration but uses `<x>` in a node argument. `pipeline_parse_text_tolerant` auto-promotes; `pipeline_repair` then drops a dead-end node whose source no longer exists; the residual verifies.

## 8. Acceptance criteria

| ID | Verifies | Test |
|---|---|---|
| ACC-PIPE-001 | INV-PIPE-001..004 | `tests/test_microgpt_pipeline.c::test_types` |
| ACC-PIPE-002 | INV-PIPE-010..013, 020 | `tests/test_microgpt_pipeline.c::test_verify_topo` |
| ACC-PIPE-003 | INV-PIPE-021..022 | `tests/test_microgpt_pipeline.c::test_roundtrip` |
| ACC-PIPE-004 | INV-PIPE-023 | `tests/test_microgpt_pipeline.c::test_repair` |
| ACC-PIPE-005 | All `PIPE_ERR_*` paths | `tests/test_microgpt_pipeline.c::test_errors` |

## 9. Cross-references

- **TDD:** `TDD_pipeline_ir.md`
- **FS:** `FS_pipeline_ir_text.md`
- **Source:** `src/microgpt_pipeline.{h,c}`
- **Tests:** `tests/test_microgpt_pipeline.c` (51/51)
- **Downstream:** `BS_wiring.md`, `pipeline_execute_vm` integration with `BS_vm.md`

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
