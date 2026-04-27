/*
 * MicroGPT-C — Graph-based Call & Data Flow Pipeline IR
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 *
 * A SysML-flavoured graph IR for representing computations as
 * directed graphs of typed nodes and dataflow edges. Designed to be
 * the target output of a future "Wiring Organelle" — a tiny model
 * that emits graph constructions instead of free-form code, with
 * verification (type-check + cycle-check + connectivity-check) acting
 * as a deterministic Judge before execution.
 *
 * Phase 1 of the multi-phase plan in RESEARCH_PIPELINE_IR.md:
 *   - Phase 1 (this header + microgpt_pipeline.c): IR + verifier +
 *     text round-trip + DOT renderer + callback-based executor.
 *   - Phase 2: VM lowering (each leaf node compiles to a VM call).
 *   - Phase 3: Wiring Organelle trains on (prompt, graph-text) pairs.
 *   - Phase 4: SysML multi-view rendering + benchmark.
 *
 * Design tenets:
 *   1. Pure C99, libc + libm only — same as the rest of the engine.
 *   2. Orthogonal to the existing engine — no changes to microgpt.h
 *      or the VM. Sit alongside as an optional module.
 *   3. Construction is a small, finite vocabulary of operations
 *      (add_node, connect, set_signature) — designed to be tractable
 *      output for a small organelle model.
 *   4. Verification is local: every error has a specific node/edge
 *      to point at. The model receives actionable feedback.
 *   5. Round-trip-safe text format: serialise ↔ parse is the
 *      identity for any verified graph.
 */

#ifndef MICROGPT_PIPELINE_H
#define MICROGPT_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  Type system
 * ============================================================
 *
 * Edges in the pipeline carry typed values. The type system is
 * structural (two types are equal if their structures match), with
 * a polymorphic ANY for staging during construction.
 *
 * Composite types (LIST, TENSOR, RECORD) own their child types and
 * are deep-copied / deep-freed by the helpers below.
 */

typedef enum {
    PIPE_T_VOID,    /* the "no value" type, e.g. for control-only flows */
    PIPE_T_INT,
    PIPE_T_FLOAT,
    PIPE_T_STRING,
    PIPE_T_LIST,    /* element_type = type of list elements (homogeneous) */
    PIPE_T_TENSOR,  /* element_type = scalar type, dims = shape (n_dims long) */
    PIPE_T_RECORD,  /* fields[] = named typed fields (struct-like) */
    PIPE_T_ANY      /* polymorphic placeholder; matches anything during verify */
} PipelineTypeKind;

typedef struct PipelineRecordField {
    char *name;
    struct PipelineType *type;  /* owned */
} PipelineRecordField;

typedef struct PipelineType {
    PipelineTypeKind kind;
    /* For LIST and TENSOR: */
    struct PipelineType *element_type;  /* owned, NULL when not applicable */
    /* For TENSOR: */
    int n_dims;
    int *dims;                          /* heap, length n_dims; -1 = wildcard dim */
    /* For RECORD: */
    PipelineRecordField *fields;        /* heap, length n_fields */
    int n_fields;
} PipelineType;

/* Type constructors (caller owns return value, free with pipeline_type_free). */
PipelineType *pipeline_type_void(void);
PipelineType *pipeline_type_int(void);
PipelineType *pipeline_type_float(void);
PipelineType *pipeline_type_string(void);
PipelineType *pipeline_type_any(void);
PipelineType *pipeline_type_list(PipelineType *element_type);  /* takes ownership */
PipelineType *pipeline_type_tensor(PipelineType *element_type, /* takes ownership */
                                   int n_dims, const int *dims);
PipelineType *pipeline_type_record(int n_fields,
                                   const char **field_names,
                                   PipelineType **field_types); /* takes ownership of each */

/* Deep clone — caller owns the returned type. */
PipelineType *pipeline_type_clone(const PipelineType *t);

/* Free a type tree (recurses into element_type and fields). */
void pipeline_type_free(PipelineType *t);

/* Structural equality. ANY matches everything. Tensor wildcards (-1)
 * match any concrete dim. Returns 1 if equal/compatible, 0 otherwise. */
int pipeline_type_equal(const PipelineType *a, const PipelineType *b);

/* Pretty-print a type into caller-provided buffer. Returns bytes written
 * (excluding NUL), or -1 on error. Format examples:
 *   void  int  float  string  list[float]  tensor[float, 3, 224, 224]
 *   record{name: string, age: int}  any */
int pipeline_type_format(const PipelineType *t, char *buf, size_t buf_size);

/* ============================================================
 *  Ports and Edges
 * ============================================================ */

typedef struct PipelinePort {
    char *name;                /* port name within its node, e.g. "in", "out" */
    PipelineType *type;        /* owned */
    /* Set after pipeline_verify(): which edges connect to this port.
     * For input ports: must be exactly one (= incoming dataflow).
     * For output ports: zero-or-more (= fanout). */
    struct PipelineEdge **edges;
    int n_edges;
    int edges_cap;             /* internal */
} PipelinePort;

typedef struct PipelineEdge {
    /* Endpoints by node + port index (NOT name) for O(1) traversal. */
    int src_node_idx;
    int src_port_idx;
    int dst_node_idx;
    int dst_port_idx;
    /* Type carried on the wire — derived from src port at construction time;
     * verified equal to dst port type by pipeline_verify(). */
    PipelineType *type;        /* not owned (alias of src port's type after verify) */
} PipelineEdge;

/* ============================================================
 *  Nodes
 * ============================================================
 *
 * A node has a unique id within its parent graph, references either a
 * leaf primitive (looked up at execute time) or a nested subgraph
 * (recursive composition), and exposes typed input/output ports.
 *
 * Config parameters (SysML "value properties") are constants the
 * organelle can set per-instance — distinct from dataflow inputs.
 */

typedef enum {
    PIPE_CFG_INT,
    PIPE_CFG_FLOAT,
    PIPE_CFG_STRING
} PipelineConfigKind;

typedef struct PipelineConfig {
    char *name;
    PipelineConfigKind kind;
    union {
        int64_t i;
        double f;
        char *s;        /* heap-allocated, owned */
    } v;
} PipelineConfig;

typedef struct PipelineNode {
    char *id;                       /* unique within parent graph */
    char *primitive;                /* primitive name (resolved at execute time) */
    struct Pipeline *subgraph;      /* NULL = leaf, non-NULL = composition */
    PipelinePort *inputs;
    int n_inputs;
    PipelinePort *outputs;
    int n_outputs;
    PipelineConfig *config;
    int n_config;
} PipelineNode;

/* ============================================================
 *  Pipeline (the graph itself)
 * ============================================================
 *
 * A graph is a named collection of nodes, edges, and a signature —
 * the graph's external I/O contract. The signature is what makes a
 * graph *itself* usable as a node in a parent graph (compositional
 * recursion).
 *
 * Construction and verification are separate phases:
 *   1. Builder API populates nodes/edges/signature in any order.
 *   2. pipeline_verify() runs all integrity checks and caches
 *      exec_order on success. Subsequent calls are idempotent.
 *   3. Mutating the graph after verify resets the verified flag.
 */

typedef struct Pipeline {
    char *name;
    PipelineNode **nodes;
    size_t n_nodes;
    size_t nodes_cap;
    PipelineEdge **edges;
    size_t n_edges;
    size_t edges_cap;
    /* Signature: graph-level inputs and outputs, like SysML proxy ports. */
    PipelinePort *signature_in;
    int n_sig_in;
    PipelinePort *signature_out;
    int n_sig_out;
    /* Cached topological order (node indices) after pipeline_verify(). */
    int *exec_order;
    int verified;
} Pipeline;

/* ============================================================
 *  Construction API
 * ============================================================ */

Pipeline *pipeline_create(const char *name);
void      pipeline_free(Pipeline *p);

/* Add a leaf node. primitive is the name a host-side callback resolves
 * (see pipeline_execute). Inputs/outputs are arrays of length n_in/n_out.
 * Each PipelineType* is taken ownership of by the node and freed with
 * pipeline_free.
 *
 * Returns the new node index (>= 0) or -1 on error (duplicate id, OOM). */
int pipeline_add_node(Pipeline *p, const char *id, const char *primitive,
                      int n_in,  const char **in_names,  PipelineType **in_types,
                      int n_out, const char **out_names, PipelineType **out_types);

/* Add a composition node — primitive == NULL, subgraph is taken ownership of
 * (the parent will free it). Subgraph's signature defines this node's port
 * types (deep-copied at insertion time). Subgraph must be verified first.
 *
 * Returns the new node index or -1 on error. */
int pipeline_add_subgraph(Pipeline *p, const char *id, Pipeline *subgraph);

/* Connect output port of src node to input port of dst node. Resolves
 * id strings to indices internally. Validates port existence; defers
 * type-checking to pipeline_verify().
 *
 * Returns 0 on success, -1 on error (unknown id, unknown port, OOM). */
int pipeline_connect(Pipeline *p,
                     const char *src_id, const char *src_port,
                     const char *dst_id, const char *dst_port);

/* Set graph signature — these are the external I/O ports of the
 * pipeline-as-a-block. Replaces any prior signature. Type ownership
 * transferred. Returns 0 on success, -1 on OOM. */
int pipeline_set_signature(Pipeline *p,
                           int n_in,  const char **in_names,  PipelineType **in_types,
                           int n_out, const char **out_names, PipelineType **out_types);

/* Connect an external signature input (treated as if it were a virtual
 * source node) to a node's input port. Use signature_in_name as the
 * "external" port name on the source side. Symmetric for outputs. */
int pipeline_connect_signature_in(Pipeline *p, const char *sig_in_name,
                                  const char *dst_id, const char *dst_port);
int pipeline_connect_signature_out(Pipeline *p, const char *src_id,
                                   const char *src_port, const char *sig_out_name);

/* Set a config value on a node (parameterisation). kind must match the
 * declared config slot's kind once such a slot is added (Phase 1 supports
 * untyped key-value bag — config is stored as-given without schema check). */
int pipeline_node_set_config_int(Pipeline *p, const char *node_id,
                                 const char *key, int64_t value);
int pipeline_node_set_config_float(Pipeline *p, const char *node_id,
                                   const char *key, double value);
int pipeline_node_set_config_string(Pipeline *p, const char *node_id,
                                    const char *key, const char *value);

/* ============================================================
 *  Verification
 * ============================================================
 *
 * pipeline_verify() runs ALL of the following in order:
 *   1. Every node id is unique.
 *   2. Every edge endpoint references an existing node and port.
 *   3. Every input port has exactly one incoming edge OR is connected
 *      to a signature-input. (No dangling input ports.)
 *   4. Every signature-input is connected to at least one node port.
 *   5. Every signature-output is connected to exactly one node port.
 *   6. Edge types match (src port type == dst port type, modulo ANY).
 *   7. Graph is acyclic (DFS-based cycle detection).
 *   8. Topologically-sorts nodes into exec_order.
 *
 * Returns 0 on success (verified=1, exec_order populated), or a
 * negative error code on failure. Use pipeline_last_error() to fetch
 * a human-readable message naming the offending node/edge/port.
 */

#define PIPE_OK                  0
#define PIPE_ERR_DUP_NODE_ID    -1
#define PIPE_ERR_UNKNOWN_NODE   -2
#define PIPE_ERR_UNKNOWN_PORT   -3
#define PIPE_ERR_DANGLING_PORT  -4
#define PIPE_ERR_TYPE_MISMATCH  -5
#define PIPE_ERR_CYCLE          -6
#define PIPE_ERR_BAD_SIGNATURE  -7
#define PIPE_ERR_OOM            -8
#define PIPE_ERR_PARSE          -9
#define PIPE_ERR_EXEC          -10

int pipeline_verify(Pipeline *p);

/* ============================================================
 *  Partial verification (Phase 2 — for incremental construction)
 * ============================================================
 *
 * pipeline_verify_partial() runs the same checks as pipeline_verify()
 * but treats "still incomplete" conditions as recoverable warnings
 * rather than hard errors:
 *
 *   - Dangling input ports are allowed (they will be wired later).
 *   - Unconnected signature outputs are allowed.
 *   - Unused signature inputs are allowed.
 *
 * Hard errors (still rejected):
 *   - Duplicate node ids.
 *   - Edge endpoints referencing nonexistent nodes/ports.
 *   - Type mismatches on connected edges.
 *   - Cycles among connected nodes.
 *
 * On success returns PIPE_OK and writes the count of "missing"
 * elements (dangling ports + unconnected sig outputs + unused sig
 * inputs) into *missing_out (may be NULL). The caller can decide
 * whether the graph is complete enough to execute (typically
 * complete iff missing == 0, equivalent to pipeline_verify success).
 *
 * Designed for the future Wiring Organelle — it constructs a graph
 * one node at a time and gets actionable feedback after each step.
 */
int pipeline_verify_partial(Pipeline *p, int *missing_out);

/* Returns a pointer to a static buffer holding the most recent error
 * message produced by any pipeline_* function in this thread. Buffer
 * is overwritten by subsequent calls. Never returns NULL. */
const char *pipeline_last_error(void);

/* ============================================================
 *  Execution (callback-based)
 * ============================================================
 *
 * Phase 1 ships a host-callback executor — the host registers a
 * resolver that maps primitive names to function pointers, plus a
 * dispatch function that invokes the resolved primitive with input
 * values and writes output values.
 *
 * Phase 2 will add a VM-backed executor that lowers leaf primitives
 * to vm_module_compile + vm_call calls automatically.
 *
 * Values are tagged unions matching PipelineType. The executor walks
 * exec_order, materialises input values from incoming edges, calls the
 * primitive, captures output values, and propagates them along
 * outgoing edges.
 */

typedef struct PipelineValue {
    PipelineType *type;        /* not owned — alias of port type */
    union {
        int64_t i;
        double f;
        char *s;               /* heap, owned */
        struct {
            void *data;        /* opaque host pointer for tensors / lists / records */
            void (*free_fn)(void *);  /* called by pipeline_value_clear */
        } opaque;
    } v;
} PipelineValue;

void pipeline_value_clear(PipelineValue *val);

/* Host-supplied dispatcher.
 *   primitive : node->primitive string
 *   config    : node->config array (length n_config)
 *   inputs    : input values, length = n_inputs
 *   outputs   : caller-provided buffer, length = n_outputs; dispatcher
 *               populates each value (the type field is pre-populated).
 *   user_data : pass-through from pipeline_execute call site.
 * Return 0 on success, non-zero on error. */
typedef int (*PipelineDispatchFn)(const char *primitive,
                                  const PipelineConfig *config, int n_config,
                                  const PipelineValue *inputs, int n_inputs,
                                  PipelineValue *outputs, int n_outputs,
                                  void *user_data);

/* Execute a verified pipeline. inputs[] length must equal n_sig_in;
 * outputs[] is caller-provided buffer of length n_sig_out (each
 * value's type field is set on success).
 *
 * Returns 0 on success, negative error code otherwise. Subgraphs are
 * recursively executed via this same function. */
int pipeline_execute(const Pipeline *p,
                     const PipelineValue *inputs,
                     PipelineValue *outputs,
                     PipelineDispatchFn dispatch, void *user_data);

/* ============================================================
 *  VM-backed dispatch (Phase 2 — convenience integration)
 * ============================================================
 *
 * Executes a pipeline by resolving each leaf node's primitive name to
 * a registered native function in the supplied vm_engine, then calling
 * it via the engine's dispatch path.
 *
 * Phase 2 limitation: vm_native_fn takes (int argc, const double *argv)
 * and returns double — so this entry point only works for pipelines
 * whose leaf nodes use INT/FLOAT-typed ports exclusively. Pipelines
 * with STRING/LIST/TENSOR/RECORD ports must use the callback-based
 * pipeline_execute().
 *
 * The host is responsible for:
 *   - Calling vm_engine_register_fn() for every primitive name that
 *     appears in the pipeline (before pipeline_execute_vm).
 *   - Disposing the engine when done.
 *
 * Returns 0 on success or a negative PIPE_ERR_* code. If a primitive
 * is missing from the engine, returns PIPE_ERR_EXEC and
 * pipeline_last_error() identifies the missing primitive.
 */

/* Forward declaration to avoid including microgpt_vm.h in this header. */
typedef struct vm_engine_t vm_engine;

int pipeline_execute_vm(const Pipeline *p,
                        vm_engine *vm,
                        const PipelineValue *inputs,
                        PipelineValue *outputs);

/* ============================================================
 *  Text serialisation (round-trip safe for verified graphs)
 * ============================================================
 *
 * Format (informal grammar):
 *   graph     ::= '@graph' IDENT NEWLINE sig_lines node_lines '@end'
 *   sig_lines ::= ( ':' 'in' '->' type | ':' 'out' '->' type | ':' name '->' type )*
 *   node_lines::= ( '|' node_id '=' primitive '(' arg_list ')' opt_config? )*
 *   arg_list  ::= ( arg ( ',' arg )* )?
 *   arg       ::= IDENT                       (positional input bound by name)
 *               | name ':' IDENT              (named input)
 *               | <ext_in_name>               (graph signature input)
 *   opt_config::= '#' key '=' value (',' key '=' value)*
 *   final     ::= ( ext_out_name '<-' node_id '.' port_name )+
 *
 * The serializer always emits node lines in topological order, so the
 * text form is a deterministic canonical representation.
 *
 * Returns a heap-allocated string the caller must free(), or NULL on error.
 */
char *pipeline_render_text(const Pipeline *p);

/* Parse the textual form. Returns a NEW unverified Pipeline (caller owns).
 * On parse error, returns NULL and pipeline_last_error() describes the
 * problem with line/column where possible. */
Pipeline *pipeline_parse_text(const char *src);

/* ============================================================
 *  DOT renderer (visualisation)
 * ============================================================
 *
 * Emits GraphViz DOT format. Each node becomes a record-shaped box
 * with input ports on the top and output ports on the bottom. Edges
 * are labelled with their type. Signature ports are rendered as
 * ellipses at the boundary.
 *
 * Returns a heap-allocated string the caller must free(), or NULL on error.
 * Pipe through `dot -Tsvg foo.dot > foo.svg` to render.
 */
char *pipeline_render_dot(const Pipeline *p);

#ifdef __cplusplus
}
#endif

#endif /* MICROGPT_PIPELINE_H */
