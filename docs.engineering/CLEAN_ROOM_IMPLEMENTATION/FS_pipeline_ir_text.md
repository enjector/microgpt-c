# FS_pipeline_ir_text — Functional / Format Specification

**Document ID:** FS-PIPE-001
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30
**Source of truth:** `src/microgpt_pipeline.c` — `pipeline_render_text`, `pipeline_parse_text`, `pipeline_parse_text_tolerant`. Header documentation in `src/microgpt_pipeline.h` lines 474–525.

---

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Format overview

The Pipeline IR text format is a deterministic, human-readable serialisation of a typed pipeline graph (`Pipeline` struct in `microgpt_pipeline.h`). It is the wire format between:

- The **Wiring Organelle** (or any future model that emits graphs) — the producer.
- The **Pipeline IR runtime** (`pipeline_parse_text`, `pipeline_verify`, `pipeline_execute`) — the consumer.

For any **verified** pipeline `p`, the round-trip property holds:

```
pipeline_parse_text(pipeline_render_text(p))   →   a graph that re-renders to the same string
```

This is the load-bearing invariant for the format: the canonical text form is the unique fingerprint of a graph's structure (modulo non-essential whitespace).

## 2. Lexical structure

A graph is a sequence of UTF-8 lines. Tokens within a line are separated by ASCII whitespace (space, tab). Newlines are significant — they terminate node lines and signature lines.

| Token class | Recognition rule |
|---|---|
| Identifier | `[A-Za-z_][A-Za-z0-9_]*` (and a leading `-` accepted only for negative tensor dimensions like `-1`) |
| Integer literal | A sequence of digits with an optional leading `-` |
| Float literal | A `printf("%g", ...)`-formatted decimal |
| Quoted string | `"..."` — backslash escapes are NOT recognised in V1.0 |
| Keyword | `@graph`, `@end`, `@subgraph`, `in`, `out`, `void`, `int`, `float`, `string`, `list`, `tensor`, `record`, `any` |
| Punctuation | `: -> | = , ( ) # < > <- :: { } [ ]` |
| Comment | (none — V1.0 has no comment syntax) |

A reader MUST permit any amount of leading or trailing horizontal whitespace on each line. A reader MUST NOT permit multi-line statements (every node and signature declaration occupies exactly one line).

## 3. Grammar

The informal grammar (matching the implementation):

```
graph        ::= '@graph' IDENT NEWLINE
                 sig_lines
                 node_lines
                 binding_lines?
                 '@end' NEWLINE?

sig_lines    ::= ( ':' 'in'  IDENT '->' type NEWLINE
                 | ':' 'out' IDENT '->' type NEWLINE )*

node_lines   ::= ( '|' IDENT '=' (IDENT | '@subgraph') '(' arg_list ')'
                       type_annotation?
                       opt_config?
                       NEWLINE )*

arg_list     ::= ( arg ( ',' arg )* )?

arg          ::= IDENT ':' arg_value

arg_value    ::= IDENT '.' IDENT          /* node-id . output-port */
              |  '<' IDENT '>'            /* graph signature input by name */
              |  '<unconnected>'          /* placeholder for partial graphs */

type_annotation
             ::= '::' port_type ( ',' port_type )* '->' port_type ( ',' port_type )*

port_type    ::= IDENT ':' type

opt_config   ::= '#' kv ( ',' kv )*

kv           ::= IDENT '=' (INT | FLOAT | QUOTED_STRING)

binding_lines::= ( IDENT '<-' IDENT '.' IDENT NEWLINE )*

type         ::= 'void' | 'int' | 'float' | 'string' | 'any'
              |  'list'   '[' type ']'
              |  'tensor' '[' type ',' INT ( ',' INT )* ']'
              |  'record' '{' IDENT ':' type ( ',' IDENT ':' type )* '}'
```

### 3.1 Graph header

The first non-empty line of the format MUST be `@graph <name>`, where `<name>` is an identifier. The matching close marker is `@end` on its own line. Anything between is the body.

### 3.2 Signature lines

Signature lines start with `:` after optional indentation. They declare the graph's external I/O contract (`signature_in[]`, `signature_out[]`) and MUST appear before any node line. The form is:

```
  : in <name> -> <type>
  : out <name> -> <type>
```

The implementation does not require all signature inputs before signature outputs (the parser accepts them interleaved); the renderer always emits inputs first then outputs.

### 3.3 Node lines

Node lines start with `|`. The form is:

```
  | <id> = <primitive>(<arg_list>) [:: <type-annotation>] [# <config-list>]
```

Where:

- `<id>` is a unique identifier within the graph.
- `<primitive>` is the name a host-side `PipelineDispatchFn` resolver maps to a function pointer. The reserved value `@subgraph` indicates a composition node (the actual subgraph is taken ownership of via `pipeline_add_subgraph` and is not serialised inline).
- `<arg_list>` consists of `port_name : <source>` entries, comma-separated. `<source>` is either `<other_node>.<out_port>` or `<sig_input_name>` (graph signature input).
- The optional `::` type-annotation suffix lists each input port and each output port with its concrete type; this is emitted only if at least one port is non-`any`.
- The optional `# kv = value, kv = value` config block contains node configuration (`pipeline_node_set_config_*`). Strings are quoted (`"..."`); ints and floats are bare literals.

The renderer always emits node lines in **topological order** (from `exec_order` after a successful `pipeline_verify`). This ordering is part of the canonical form.

### 3.4 Output binding lines

After the last node line, signature outputs are bound via:

```
  <sig_output_name> <- <node_id>.<output_port>
```

These lines appear in the order of `signature_out[]`. Only signature outputs that have an incoming edge are emitted.

### 3.5 Termination

The graph terminates with `@end` on its own line. Anything after `@end` is ignored by the parser.

## 4. Type grammar

The type grammar is identical to the formatter's output (`pipeline_type_format`):

| Source form | Meaning |
|---|---|
| `void` | The "no value" type (control-only edges) |
| `int` | 64-bit signed integer (`int64_t` at runtime) |
| `float` | IEEE-754 double (`double` at runtime) |
| `string` | Heap-allocated NUL-terminated C string |
| `any` | Polymorphic placeholder; matches anything during verify |
| `list[<elem>]` | Homogeneous list of `<elem>` |
| `tensor[<elem>, d0, d1, ..., dk]` | Tensor with element type `<elem>` and shape `[d0, d1, ..., dk]`; a `-1` dimension matches any concrete dim |
| `record{<name>: <type>, ...}` | Named tuple of fields |

`pipeline_type_equal` treats `any` as compatible with any other type and treats wildcard `-1` tensor dims as compatible with concrete dims.

## 5. Rendering rules

`pipeline_render_text(p)` SHALL produce output that satisfies:

1. The first line is `@graph <name>\n`.
2. All signature inputs are rendered before all signature outputs, each on its own line.
3. Nodes are rendered in topological order if the graph is verified, else in insertion order.
4. Each input port is rendered as `<port_name>: <source>`. An unconnected port renders as `<port_name>: <unconnected>`.
5. The type-annotation suffix `:: ...` is emitted iff at least one input or output port has a type other than `any`.
6. Config blocks emit ints with `%lld`, floats with `%g`, strings inside `"..."`.
7. Output bindings appear after all node lines, only for connected signature outputs.
8. The graph terminates with `@end\n`.

## 6. Parsing rules

`pipeline_parse_text(src)` SHALL implement the strict reverse of the renderer. For any string `s` produced by `pipeline_render_text` on a verified graph, `pipeline_parse_text(s)` SHALL succeed and SHALL produce an equivalent unverified graph. Rendering that graph (after a fresh `pipeline_verify`) SHALL yield `s` again.

`pipeline_parse_text_tolerant(src)` SHALL accept the same grammar plus three named repairs:

| Repair | Trigger | Action |
|---|---|---|
| Repair 1 | A `: in <name> -> <type>` line repeats a previously declared name | Drop the duplicate (silent) |
| Repair 2 | A node argument references `<sig_name>` that has no `: in <sig_name>` declaration | Auto-promote a matching `: in <sig_name> -> int` signature input |
| Repair 3 | An output binding `<sig_name> <- <node>.<port>` references a signature output not declared in any `: out <sig_name>` line | Auto-promote a matching `: out <sig_name> -> int` signature output |

The tolerant parser MUST NOT silently fix type mismatches, cycles, or undefined node references; those remain hard errors.

A consumer MAY also call `pipeline_repair(p, &report)` after parsing to drop dead-end fragments before verification (see `BS_pipeline_ir.md` and the `PipelineRepairReport` struct).

## 7. Error codes

Parser failures populate a thread-local error buffer accessible via `pipeline_last_error()`. The function returns `NULL` on parse failure.

| ID | Source code | Conditions |
|---|---|---|
| ERR-PIPE-001 | `PIPE_ERR_PARSE` (-9) | Generic parse error — malformed token, unknown keyword, missing `@end`, unterminated string |
| ERR-PIPE-002 | `PIPE_ERR_DUP_NODE_ID` (-1) | Two node lines share the same `<id>` |
| ERR-PIPE-003 | `PIPE_ERR_UNKNOWN_NODE` (-2) | Edge endpoint references a node id that does not exist |
| ERR-PIPE-004 | `PIPE_ERR_UNKNOWN_PORT` (-3) | Edge endpoint references a port name that does not exist on its node |
| ERR-PIPE-005 | `PIPE_ERR_DANGLING_PORT` (-4) | An input port has no incoming edge after parsing (strict only — tolerant parser allows this) |
| ERR-PIPE-006 | `PIPE_ERR_TYPE_MISMATCH` (-5) | The annotated type on an edge endpoint disagrees with the source's port type |
| ERR-PIPE-007 | `PIPE_ERR_CYCLE` (-6) | The connected component contains a cycle (detected by `pipeline_verify`) |
| ERR-PIPE-008 | `PIPE_ERR_BAD_SIGNATURE` (-7) | A signature input is unused, or a signature output is referenced but not declared (strict only) |
| ERR-PIPE-009 | `PIPE_ERR_OOM` (-8) | Allocation failure during construction |
| ERR-PIPE-010 | `PIPE_ERR_EXEC` (-10) | A primitive name in a leaf node has no resolver in the `PipelineDispatchFn` |

## 8. Normative example

A two-stage pipeline that classifies a transaction by computing its z-score against a baseline and flagging it. Verified, then rendered:

```
@graph velocity_spike_24h
  : in transactions -> list[record{ts: int, amount: float}]
  : in baseline -> record{mean: float, std: float}
  : out flag -> int
  | window = txn_count_window(in: <transactions>) :: in:list[record{ts:int, amount:float}] -> out:int
  | score  = compare_to_baseline(count: window.out, profile: <baseline>) :: count:int, profile:record{mean:float, std:float} -> out:float
  | flag_n = flag(score: score.out) :: score:float -> out:int #threshold=3.0
  flag <- flag_n.out
@end
```

Round-trip property: parsing this string produces a `Pipeline *` whose subsequent `pipeline_render_text` call returns a string identical to this one (after `pipeline_verify`).

## 9. Versioning

The format is **unversioned** in V1.0. The grammar is fixed at the version of `microgpt_pipeline.c` shipping with this corpus. Future grammar changes SHALL be additive (new optional clauses, new types) within version 1.x; breaking changes (different keyword set, different separator structure) SHALL bump to 2.0 and MUST be marked by a `@graph_v2` opening keyword.

The current grammar accepts the lenient subset documented under `pipeline_parse_text_tolerant` and rejects everything else with a single error code (`PIPE_ERR_PARSE`).

## 10. Test vectors

`tests/test_microgpt_pipeline.c` exercises 51 unit tests including round-trip tests on:

- Empty graphs (no nodes, signature only).
- Single-node graphs with all type variants.
- Multi-node DAGs with type annotations.
- Graphs with config (int / float / string).
- Graphs with subgraphs.
- Graphs with tensor and record types (including wildcard dims).
- Tolerant-parser repairs 1, 2, and 3.

## 11. Cross-references

- `BS_pipeline_ir.md` for the type-system invariants and the verifier's behavioural contract.
- `TDD_pipeline_ir.md` for the implementation strategy.
- `BS_wiring.md` for the model that produces these strings as output.
- `FRD.md` REQ-PIPE-009 .. REQ-PIPE-014.

## 12. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
