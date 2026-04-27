/*
 * MicroGPT-C — Graph Pipeline IR — Implementation
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 *
 * Pure C99, libc + libm only. Single TU. Public API in microgpt_pipeline.h.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <ctype.h>

/* ============================================================
 *  Error reporting
 * ============================================================ */

static char g_pipeline_err[512] = "";

static void set_err(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(g_pipeline_err, sizeof(g_pipeline_err), fmt, ap);
    va_end(ap);
}

const char *pipeline_last_error(void) {
    return g_pipeline_err;
}

/* ============================================================
 *  Type system
 * ============================================================ */

static PipelineType *type_alloc(PipelineTypeKind kind) {
    PipelineType *t = (PipelineType *)calloc(1, sizeof(PipelineType));
    if (!t) { set_err("OOM in type_alloc"); return NULL; }
    t->kind = kind;
    return t;
}

PipelineType *pipeline_type_void(void)   { return type_alloc(PIPE_T_VOID); }
PipelineType *pipeline_type_int(void)    { return type_alloc(PIPE_T_INT); }
PipelineType *pipeline_type_float(void)  { return type_alloc(PIPE_T_FLOAT); }
PipelineType *pipeline_type_string(void) { return type_alloc(PIPE_T_STRING); }
PipelineType *pipeline_type_any(void)    { return type_alloc(PIPE_T_ANY); }

PipelineType *pipeline_type_list(PipelineType *element_type) {
    PipelineType *t = type_alloc(PIPE_T_LIST);
    if (!t) { pipeline_type_free(element_type); return NULL; }
    t->element_type = element_type;
    return t;
}

PipelineType *pipeline_type_tensor(PipelineType *element_type,
                                   int n_dims, const int *dims) {
    PipelineType *t = type_alloc(PIPE_T_TENSOR);
    if (!t) { pipeline_type_free(element_type); return NULL; }
    t->element_type = element_type;
    t->n_dims = n_dims;
    if (n_dims > 0) {
        t->dims = (int *)malloc(sizeof(int) * (size_t)n_dims);
        if (!t->dims) { pipeline_type_free(t); set_err("OOM in tensor dims"); return NULL; }
        memcpy(t->dims, dims, sizeof(int) * (size_t)n_dims);
    }
    return t;
}

PipelineType *pipeline_type_record(int n_fields,
                                   const char **field_names,
                                   PipelineType **field_types) {
    PipelineType *t = type_alloc(PIPE_T_RECORD);
    if (!t) {
        for (int i = 0; i < n_fields; i++) pipeline_type_free(field_types[i]);
        return NULL;
    }
    t->n_fields = n_fields;
    if (n_fields > 0) {
        t->fields = (PipelineRecordField *)calloc((size_t)n_fields, sizeof(PipelineRecordField));
        if (!t->fields) {
            for (int i = 0; i < n_fields; i++) pipeline_type_free(field_types[i]);
            free(t);
            set_err("OOM in record fields");
            return NULL;
        }
        for (int i = 0; i < n_fields; i++) {
            t->fields[i].name = field_names[i] ? strdup(field_names[i]) : NULL;
            t->fields[i].type = field_types[i];  /* take ownership */
        }
    }
    return t;
}

PipelineType *pipeline_type_clone(const PipelineType *t) {
    if (!t) return NULL;
    PipelineType *c = type_alloc(t->kind);
    if (!c) return NULL;
    if (t->element_type) {
        c->element_type = pipeline_type_clone(t->element_type);
        if (!c->element_type) { pipeline_type_free(c); return NULL; }
    }
    if (t->n_dims > 0 && t->dims) {
        c->n_dims = t->n_dims;
        c->dims = (int *)malloc(sizeof(int) * (size_t)t->n_dims);
        if (!c->dims) { pipeline_type_free(c); return NULL; }
        memcpy(c->dims, t->dims, sizeof(int) * (size_t)t->n_dims);
    }
    if (t->n_fields > 0 && t->fields) {
        c->n_fields = t->n_fields;
        c->fields = (PipelineRecordField *)calloc((size_t)t->n_fields, sizeof(PipelineRecordField));
        if (!c->fields) { pipeline_type_free(c); return NULL; }
        for (int i = 0; i < t->n_fields; i++) {
            c->fields[i].name = t->fields[i].name ? strdup(t->fields[i].name) : NULL;
            c->fields[i].type = pipeline_type_clone(t->fields[i].type);
        }
    }
    return c;
}

void pipeline_type_free(PipelineType *t) {
    if (!t) return;
    if (t->element_type) pipeline_type_free(t->element_type);
    if (t->dims) free(t->dims);
    if (t->fields) {
        for (int i = 0; i < t->n_fields; i++) {
            if (t->fields[i].name) free(t->fields[i].name);
            if (t->fields[i].type) pipeline_type_free(t->fields[i].type);
        }
        free(t->fields);
    }
    free(t);
}

int pipeline_type_equal(const PipelineType *a, const PipelineType *b) {
    if (!a || !b) return 0;
    if (a->kind == PIPE_T_ANY || b->kind == PIPE_T_ANY) return 1;
    if (a->kind != b->kind) return 0;
    switch (a->kind) {
    case PIPE_T_VOID: case PIPE_T_INT: case PIPE_T_FLOAT:
    case PIPE_T_STRING: case PIPE_T_ANY:
        return 1;
    case PIPE_T_LIST:
        return pipeline_type_equal(a->element_type, b->element_type);
    case PIPE_T_TENSOR:
        if (!pipeline_type_equal(a->element_type, b->element_type)) return 0;
        if (a->n_dims != b->n_dims) return 0;
        for (int i = 0; i < a->n_dims; i++) {
            if (a->dims[i] == -1 || b->dims[i] == -1) continue;  /* wildcard */
            if (a->dims[i] != b->dims[i]) return 0;
        }
        return 1;
    case PIPE_T_RECORD:
        if (a->n_fields != b->n_fields) return 0;
        for (int i = 0; i < a->n_fields; i++) {
            if (!a->fields[i].name || !b->fields[i].name) return 0;
            if (strcmp(a->fields[i].name, b->fields[i].name) != 0) return 0;
            if (!pipeline_type_equal(a->fields[i].type, b->fields[i].type)) return 0;
        }
        return 1;
    }
    return 0;
}

int pipeline_type_format(const PipelineType *t, char *buf, size_t buf_size) {
    if (!t || !buf || buf_size == 0) return -1;
    switch (t->kind) {
    case PIPE_T_VOID:   return snprintf(buf, buf_size, "void");
    case PIPE_T_INT:    return snprintf(buf, buf_size, "int");
    case PIPE_T_FLOAT:  return snprintf(buf, buf_size, "float");
    case PIPE_T_STRING: return snprintf(buf, buf_size, "string");
    case PIPE_T_ANY:    return snprintf(buf, buf_size, "any");
    case PIPE_T_LIST: {
        char inner[256] = "";
        pipeline_type_format(t->element_type, inner, sizeof(inner));
        return snprintf(buf, buf_size, "list[%s]", inner);
    }
    case PIPE_T_TENSOR: {
        char inner[128] = "";
        pipeline_type_format(t->element_type, inner, sizeof(inner));
        size_t off = (size_t)snprintf(buf, buf_size, "tensor[%s", inner);
        for (int i = 0; i < t->n_dims && off < buf_size; i++) {
            if (t->dims[i] == -1)
                off += (size_t)snprintf(buf + off, buf_size - off, ", *");
            else
                off += (size_t)snprintf(buf + off, buf_size - off, ", %d", t->dims[i]);
        }
        if (off < buf_size) off += (size_t)snprintf(buf + off, buf_size - off, "]");
        return (int)off;
    }
    case PIPE_T_RECORD: {
        size_t off = (size_t)snprintf(buf, buf_size, "record{");
        for (int i = 0; i < t->n_fields && off < buf_size; i++) {
            char fbuf[128] = "";
            pipeline_type_format(t->fields[i].type, fbuf, sizeof(fbuf));
            off += (size_t)snprintf(buf + off, buf_size - off, "%s%s: %s",
                                    i ? ", " : "",
                                    t->fields[i].name ? t->fields[i].name : "?",
                                    fbuf);
        }
        if (off < buf_size) off += (size_t)snprintf(buf + off, buf_size - off, "}");
        return (int)off;
    }
    }
    return -1;
}

/* ============================================================
 *  Pipeline construction
 * ============================================================ */

Pipeline *pipeline_create(const char *name) {
    Pipeline *p = (Pipeline *)calloc(1, sizeof(Pipeline));
    if (!p) { set_err("OOM in pipeline_create"); return NULL; }
    p->name = name ? strdup(name) : strdup("graph");
    if (!p->name) { free(p); set_err("OOM in pipeline_create"); return NULL; }
    return p;
}

static void port_free_internal(PipelinePort *port) {
    if (!port) return;
    if (port->name) free(port->name);
    if (port->type) pipeline_type_free(port->type);
    if (port->edges) free(port->edges);
}

static void node_free(PipelineNode *n) {
    if (!n) return;
    if (n->id) free(n->id);
    if (n->primitive) free(n->primitive);
    if (n->subgraph) pipeline_free(n->subgraph);
    for (int i = 0; i < n->n_inputs; i++) port_free_internal(&n->inputs[i]);
    for (int i = 0; i < n->n_outputs; i++) port_free_internal(&n->outputs[i]);
    if (n->inputs) free(n->inputs);
    if (n->outputs) free(n->outputs);
    if (n->config) {
        for (int i = 0; i < n->n_config; i++) {
            if (n->config[i].name) free(n->config[i].name);
            if (n->config[i].kind == PIPE_CFG_STRING && n->config[i].v.s)
                free(n->config[i].v.s);
        }
        free(n->config);
    }
    free(n);
}

void pipeline_free(Pipeline *p) {
    if (!p) return;
    if (p->name) free(p->name);
    if (p->nodes) {
        for (size_t i = 0; i < p->n_nodes; i++) node_free(p->nodes[i]);
        free(p->nodes);
    }
    if (p->edges) {
        for (size_t i = 0; i < p->n_edges; i++) free(p->edges[i]);
        free(p->edges);
    }
    if (p->signature_in) {
        for (int i = 0; i < p->n_sig_in; i++) port_free_internal(&p->signature_in[i]);
        free(p->signature_in);
    }
    if (p->signature_out) {
        for (int i = 0; i < p->n_sig_out; i++) port_free_internal(&p->signature_out[i]);
        free(p->signature_out);
    }
    if (p->exec_order) free(p->exec_order);
    free(p);
}

/* Find node by id; returns index or -1. */
static int find_node(const Pipeline *p, const char *id) {
    if (!id) return -1;
    for (size_t i = 0; i < p->n_nodes; i++) {
        if (p->nodes[i] && p->nodes[i]->id && strcmp(p->nodes[i]->id, id) == 0)
            return (int)i;
    }
    return -1;
}

/* Find port by name on a node side. side: 0=input, 1=output. */
static int find_port(const PipelineNode *n, const char *port_name, int side) {
    if (!n || !port_name) return -1;
    PipelinePort *ports = side ? n->outputs : n->inputs;
    int count = side ? n->n_outputs : n->n_inputs;
    for (int i = 0; i < count; i++) {
        if (ports[i].name && strcmp(ports[i].name, port_name) == 0) return i;
    }
    return -1;
}

static int find_signature_port(const Pipeline *p, const char *name, int side) {
    PipelinePort *ports = side ? p->signature_out : p->signature_in;
    int count = side ? p->n_sig_out : p->n_sig_in;
    for (int i = 0; i < count; i++) {
        if (ports[i].name && strcmp(ports[i].name, name) == 0) return i;
    }
    return -1;
}

/* Reset verification flag — call whenever the graph is mutated. */
static void invalidate(Pipeline *p) {
    p->verified = 0;
    if (p->exec_order) { free(p->exec_order); p->exec_order = NULL; }
}

static int build_ports(PipelinePort **out, int *out_count,
                       int n, const char **names, PipelineType **types) {
    if (n <= 0) { *out = NULL; *out_count = 0; return 0; }
    PipelinePort *ports = (PipelinePort *)calloc((size_t)n, sizeof(PipelinePort));
    if (!ports) { set_err("OOM in build_ports"); return -1; }
    for (int i = 0; i < n; i++) {
        ports[i].name = names && names[i] ? strdup(names[i]) : NULL;
        ports[i].type = types ? types[i] : NULL;  /* take ownership */
    }
    *out = ports; *out_count = n;
    return 0;
}

int pipeline_add_node(Pipeline *p, const char *id, const char *primitive,
                      int n_in,  const char **in_names,  PipelineType **in_types,
                      int n_out, const char **out_names, PipelineType **out_types) {
    if (!p || !id) { set_err("pipeline_add_node: null arg"); return PIPE_ERR_OOM; }
    if (find_node(p, id) >= 0) {
        set_err("pipeline_add_node: duplicate id '%s'", id);
        /* free any provided types since we're not taking them */
        for (int i = 0; i < n_in; i++) pipeline_type_free(in_types[i]);
        for (int i = 0; i < n_out; i++) pipeline_type_free(out_types[i]);
        return PIPE_ERR_DUP_NODE_ID;
    }
    PipelineNode *n = (PipelineNode *)calloc(1, sizeof(PipelineNode));
    if (!n) { set_err("OOM"); return PIPE_ERR_OOM; }
    n->id = strdup(id);
    n->primitive = primitive ? strdup(primitive) : NULL;
    if (build_ports(&n->inputs,  &n->n_inputs,  n_in,  in_names,  in_types) < 0 ||
        build_ports(&n->outputs, &n->n_outputs, n_out, out_names, out_types) < 0) {
        node_free(n);
        return PIPE_ERR_OOM;
    }
    /* grow nodes array */
    if (p->n_nodes + 1 > p->nodes_cap) {
        size_t newcap = p->nodes_cap ? p->nodes_cap * 2 : 8;
        PipelineNode **arr = (PipelineNode **)realloc(p->nodes, sizeof(PipelineNode *) * newcap);
        if (!arr) { node_free(n); set_err("OOM"); return PIPE_ERR_OOM; }
        p->nodes = arr; p->nodes_cap = newcap;
    }
    p->nodes[p->n_nodes++] = n;
    invalidate(p);
    return (int)(p->n_nodes - 1);
}

int pipeline_add_subgraph(Pipeline *p, const char *id, Pipeline *subgraph) {
    if (!p || !id || !subgraph) { set_err("pipeline_add_subgraph: null arg"); return PIPE_ERR_OOM; }
    if (!subgraph->verified) {
        set_err("pipeline_add_subgraph: subgraph '%s' not verified", subgraph->name ? subgraph->name : "?");
        return PIPE_ERR_BAD_SIGNATURE;
    }
    /* Materialise input/output port names + types from subgraph signature. */
    const char **in_names  = (const char **)calloc((size_t)subgraph->n_sig_in,  sizeof(char *));
    const char **out_names = (const char **)calloc((size_t)subgraph->n_sig_out, sizeof(char *));
    PipelineType **in_types  = (PipelineType **)calloc((size_t)subgraph->n_sig_in,  sizeof(PipelineType *));
    PipelineType **out_types = (PipelineType **)calloc((size_t)subgraph->n_sig_out, sizeof(PipelineType *));
    if (!in_names || !out_names || !in_types || !out_types) {
        free((void *)in_names); free((void *)out_names);
        free(in_types); free(out_types);
        set_err("OOM"); return PIPE_ERR_OOM;
    }
    for (int i = 0; i < subgraph->n_sig_in; i++) {
        in_names[i] = subgraph->signature_in[i].name;
        in_types[i] = pipeline_type_clone(subgraph->signature_in[i].type);
    }
    for (int i = 0; i < subgraph->n_sig_out; i++) {
        out_names[i] = subgraph->signature_out[i].name;
        out_types[i] = pipeline_type_clone(subgraph->signature_out[i].type);
    }
    int idx = pipeline_add_node(p, id, NULL,
                                subgraph->n_sig_in, in_names, in_types,
                                subgraph->n_sig_out, out_names, out_types);
    free((void *)in_names); free((void *)out_names);
    free(in_types); free(out_types);
    if (idx < 0) { return idx; }
    p->nodes[idx]->subgraph = subgraph;  /* take ownership */
    invalidate(p);
    return idx;
}

int pipeline_set_signature(Pipeline *p,
                           int n_in,  const char **in_names,  PipelineType **in_types,
                           int n_out, const char **out_names, PipelineType **out_types) {
    if (!p) { set_err("pipeline_set_signature: null arg"); return PIPE_ERR_OOM; }
    /* free old */
    if (p->signature_in) {
        for (int i = 0; i < p->n_sig_in; i++) port_free_internal(&p->signature_in[i]);
        free(p->signature_in); p->signature_in = NULL; p->n_sig_in = 0;
    }
    if (p->signature_out) {
        for (int i = 0; i < p->n_sig_out; i++) port_free_internal(&p->signature_out[i]);
        free(p->signature_out); p->signature_out = NULL; p->n_sig_out = 0;
    }
    if (build_ports(&p->signature_in,  &p->n_sig_in,  n_in,  in_names,  in_types) < 0 ||
        build_ports(&p->signature_out, &p->n_sig_out, n_out, out_names, out_types) < 0) {
        return PIPE_ERR_OOM;
    }
    invalidate(p);
    return PIPE_OK;
}

/* Internal: connect two endpoints (resolved indices), record the edge. */
static int connect_internal(Pipeline *p, int src_node_idx, int src_port_idx,
                            int dst_node_idx, int dst_port_idx) {
    PipelineEdge *e = (PipelineEdge *)calloc(1, sizeof(PipelineEdge));
    if (!e) { set_err("OOM"); return PIPE_ERR_OOM; }
    e->src_node_idx = src_node_idx;
    e->src_port_idx = src_port_idx;
    e->dst_node_idx = dst_node_idx;
    e->dst_port_idx = dst_port_idx;
    if (p->n_edges + 1 > p->edges_cap) {
        size_t newcap = p->edges_cap ? p->edges_cap * 2 : 16;
        PipelineEdge **arr = (PipelineEdge **)realloc(p->edges, sizeof(PipelineEdge *) * newcap);
        if (!arr) { free(e); set_err("OOM"); return PIPE_ERR_OOM; }
        p->edges = arr; p->edges_cap = newcap;
    }
    p->edges[p->n_edges++] = e;
    invalidate(p);
    return PIPE_OK;
}

int pipeline_connect(Pipeline *p,
                     const char *src_id, const char *src_port,
                     const char *dst_id, const char *dst_port) {
    int s = find_node(p, src_id);
    int d = find_node(p, dst_id);
    if (s < 0) { set_err("pipeline_connect: unknown src node '%s'", src_id); return PIPE_ERR_UNKNOWN_NODE; }
    if (d < 0) { set_err("pipeline_connect: unknown dst node '%s'", dst_id); return PIPE_ERR_UNKNOWN_NODE; }
    int sp = find_port(p->nodes[s], src_port, 1);
    int dp = find_port(p->nodes[d], dst_port, 0);
    if (sp < 0) { set_err("pipeline_connect: unknown output port '%s' on '%s'", src_port, src_id); return PIPE_ERR_UNKNOWN_PORT; }
    if (dp < 0) { set_err("pipeline_connect: unknown input port '%s' on '%s'", dst_port, dst_id); return PIPE_ERR_UNKNOWN_PORT; }
    return connect_internal(p, s, sp, d, dp);
}

/* Signature-input/output edges use sentinel node index -1 (input side)
 * or -2 (output side), with the *port* index being the sig port index. */
#define SIG_IN_NODE  (-1)
#define SIG_OUT_NODE (-2)

int pipeline_connect_signature_in(Pipeline *p, const char *sig_in_name,
                                  const char *dst_id, const char *dst_port) {
    int sig_idx = find_signature_port(p, sig_in_name, 0);
    if (sig_idx < 0) { set_err("pipeline_connect_signature_in: unknown signature input '%s'", sig_in_name); return PIPE_ERR_UNKNOWN_PORT; }
    int d = find_node(p, dst_id);
    if (d < 0) { set_err("pipeline_connect_signature_in: unknown node '%s'", dst_id); return PIPE_ERR_UNKNOWN_NODE; }
    int dp = find_port(p->nodes[d], dst_port, 0);
    if (dp < 0) { set_err("pipeline_connect_signature_in: unknown input port '%s' on '%s'", dst_port, dst_id); return PIPE_ERR_UNKNOWN_PORT; }
    return connect_internal(p, SIG_IN_NODE, sig_idx, d, dp);
}

int pipeline_connect_signature_out(Pipeline *p, const char *src_id,
                                   const char *src_port, const char *sig_out_name) {
    int sig_idx = find_signature_port(p, sig_out_name, 1);
    if (sig_idx < 0) { set_err("pipeline_connect_signature_out: unknown signature output '%s'", sig_out_name); return PIPE_ERR_UNKNOWN_PORT; }
    int s = find_node(p, src_id);
    if (s < 0) { set_err("pipeline_connect_signature_out: unknown node '%s'", src_id); return PIPE_ERR_UNKNOWN_NODE; }
    int sp = find_port(p->nodes[s], src_port, 1);
    if (sp < 0) { set_err("pipeline_connect_signature_out: unknown output port '%s' on '%s'", src_port, src_id); return PIPE_ERR_UNKNOWN_PORT; }
    return connect_internal(p, s, sp, SIG_OUT_NODE, sig_idx);
}

/* Config setters — phase-1 stores as opaque key/value bag (no schema). */
static PipelineConfig *node_config_grow(PipelineNode *n, const char *key, PipelineConfigKind kind) {
    /* Replace existing key if present. */
    for (int i = 0; i < n->n_config; i++) {
        if (n->config[i].name && strcmp(n->config[i].name, key) == 0) {
            if (n->config[i].kind == PIPE_CFG_STRING && n->config[i].v.s) free(n->config[i].v.s);
            n->config[i].kind = kind;
            return &n->config[i];
        }
    }
    PipelineConfig *arr = (PipelineConfig *)realloc(n->config, sizeof(PipelineConfig) * (size_t)(n->n_config + 1));
    if (!arr) { set_err("OOM"); return NULL; }
    n->config = arr;
    PipelineConfig *slot = &n->config[n->n_config++];
    memset(slot, 0, sizeof(*slot));
    slot->name = strdup(key);
    slot->kind = kind;
    return slot;
}

int pipeline_node_set_config_int(Pipeline *p, const char *node_id, const char *key, int64_t value) {
    int idx = find_node(p, node_id);
    if (idx < 0) { set_err("set_config_int: unknown node '%s'", node_id); return PIPE_ERR_UNKNOWN_NODE; }
    PipelineConfig *c = node_config_grow(p->nodes[idx], key, PIPE_CFG_INT);
    if (!c) return PIPE_ERR_OOM;
    c->v.i = value;
    return PIPE_OK;
}
int pipeline_node_set_config_float(Pipeline *p, const char *node_id, const char *key, double value) {
    int idx = find_node(p, node_id);
    if (idx < 0) { set_err("set_config_float: unknown node '%s'", node_id); return PIPE_ERR_UNKNOWN_NODE; }
    PipelineConfig *c = node_config_grow(p->nodes[idx], key, PIPE_CFG_FLOAT);
    if (!c) return PIPE_ERR_OOM;
    c->v.f = value;
    return PIPE_OK;
}
int pipeline_node_set_config_string(Pipeline *p, const char *node_id, const char *key, const char *value) {
    int idx = find_node(p, node_id);
    if (idx < 0) { set_err("set_config_string: unknown node '%s'", node_id); return PIPE_ERR_UNKNOWN_NODE; }
    PipelineConfig *c = node_config_grow(p->nodes[idx], key, PIPE_CFG_STRING);
    if (!c) return PIPE_ERR_OOM;
    c->v.s = value ? strdup(value) : NULL;
    return PIPE_OK;
}

/* ============================================================
 *  Verification
 * ============================================================ */

/* Get the type carried on an edge's source side — for normal nodes
 * it's the node's output port type; for SIG_IN_NODE it's the signature
 * input's type. */
static const PipelineType *edge_src_type(const Pipeline *p, const PipelineEdge *e) {
    if (e->src_node_idx == SIG_IN_NODE) {
        if (e->src_port_idx < 0 || e->src_port_idx >= p->n_sig_in) return NULL;
        return p->signature_in[e->src_port_idx].type;
    }
    if (e->src_node_idx < 0 || (size_t)e->src_node_idx >= p->n_nodes) return NULL;
    PipelineNode *n = p->nodes[e->src_node_idx];
    if (e->src_port_idx < 0 || e->src_port_idx >= n->n_outputs) return NULL;
    return n->outputs[e->src_port_idx].type;
}

static const PipelineType *edge_dst_type(const Pipeline *p, const PipelineEdge *e) {
    if (e->dst_node_idx == SIG_OUT_NODE) {
        if (e->dst_port_idx < 0 || e->dst_port_idx >= p->n_sig_out) return NULL;
        return p->signature_out[e->dst_port_idx].type;
    }
    if (e->dst_node_idx < 0 || (size_t)e->dst_node_idx >= p->n_nodes) return NULL;
    PipelineNode *n = p->nodes[e->dst_node_idx];
    if (e->dst_port_idx < 0 || e->dst_port_idx >= n->n_inputs) return NULL;
    return n->inputs[e->dst_port_idx].type;
}

/* DFS-based topological sort with cycle detection. */
static int topo_visit(const Pipeline *p, int node_idx, char *state, int *order, int *order_pos) {
    if (state[node_idx] == 2) return 0;          /* finished */
    if (state[node_idx] == 1) return PIPE_ERR_CYCLE;  /* in stack: cycle */
    state[node_idx] = 1;
    /* Walk outgoing edges, recurse into successors. */
    for (size_t e = 0; e < p->n_edges; e++) {
        const PipelineEdge *edge = p->edges[e];
        if (edge->src_node_idx != node_idx) continue;
        if (edge->dst_node_idx == SIG_OUT_NODE) continue;
        int rc = topo_visit(p, edge->dst_node_idx, state, order, order_pos);
        if (rc != 0) return rc;
    }
    state[node_idx] = 2;
    order[(*order_pos)++] = node_idx;
    return 0;
}

/* Internal: shared verify body with a `strict` flag. When strict==0
 * (partial mode), dangling ports and signature unbinding are tolerated
 * and counted into *missing rather than rejected. */
static int verify_impl(Pipeline *p, int strict, int *missing_out) {
    int missing = 0;
    if (!p) { set_err("verify: null"); return PIPE_ERR_OOM; }

    /* 1. Unique node ids. */
    for (size_t i = 0; i < p->n_nodes; i++) {
        for (size_t j = i + 1; j < p->n_nodes; j++) {
            if (strcmp(p->nodes[i]->id, p->nodes[j]->id) == 0) {
                set_err("verify: duplicate node id '%s'", p->nodes[i]->id);
                return PIPE_ERR_DUP_NODE_ID;
            }
        }
    }

    /* 2. Edge endpoints valid. */
    for (size_t i = 0; i < p->n_edges; i++) {
        const PipelineEdge *e = p->edges[i];
        if (!edge_src_type(p, e)) {
            set_err("verify: edge %zu has invalid source node/port (%d.%d)", i, e->src_node_idx, e->src_port_idx);
            return PIPE_ERR_UNKNOWN_PORT;
        }
        if (!edge_dst_type(p, e)) {
            set_err("verify: edge %zu has invalid dest node/port (%d.%d)", i, e->dst_node_idx, e->dst_port_idx);
            return PIPE_ERR_UNKNOWN_PORT;
        }
    }

    /* 3. Type matching on every edge. */
    for (size_t i = 0; i < p->n_edges; i++) {
        const PipelineEdge *e = p->edges[i];
        const PipelineType *st = edge_src_type(p, e);
        const PipelineType *dt = edge_dst_type(p, e);
        if (!pipeline_type_equal(st, dt)) {
            char sb[128] = "?", db[128] = "?";
            pipeline_type_format(st, sb, sizeof(sb));
            pipeline_type_format(dt, db, sizeof(db));
            set_err("verify: edge %zu type mismatch: %s -> %s", i, sb, db);
            return PIPE_ERR_TYPE_MISMATCH;
        }
    }

    /* 4. Every node input port has exactly one incoming edge (strict). */
    for (size_t n = 0; n < p->n_nodes; n++) {
        for (int ip = 0; ip < p->nodes[n]->n_inputs; ip++) {
            int incoming = 0;
            for (size_t e = 0; e < p->n_edges; e++) {
                const PipelineEdge *edge = p->edges[e];
                if (edge->dst_node_idx == (int)n && edge->dst_port_idx == ip) incoming++;
            }
            if (incoming == 0) {
                if (strict) {
                    set_err("verify: node '%s' input port '%s' has 0 incoming edges (need 1)",
                            p->nodes[n]->id,
                            p->nodes[n]->inputs[ip].name ? p->nodes[n]->inputs[ip].name : "?");
                    return PIPE_ERR_DANGLING_PORT;
                }
                missing++;
            } else if (incoming > 1) {
                set_err("verify: node '%s' input port '%s' has %d incoming edges (need 1)",
                        p->nodes[n]->id,
                        p->nodes[n]->inputs[ip].name ? p->nodes[n]->inputs[ip].name : "?",
                        incoming);
                return PIPE_ERR_DANGLING_PORT;
            }
        }
    }

    /* 5. Signature outputs (strict: exactly 1; partial: tolerate 0). */
    for (int so = 0; so < p->n_sig_out; so++) {
        int incoming = 0;
        for (size_t e = 0; e < p->n_edges; e++) {
            const PipelineEdge *edge = p->edges[e];
            if (edge->dst_node_idx == SIG_OUT_NODE && edge->dst_port_idx == so) incoming++;
        }
        if (incoming == 0) {
            if (strict) {
                set_err("verify: signature output '%s' has 0 incoming edges (need 1)",
                        p->signature_out[so].name ? p->signature_out[so].name : "?");
                return PIPE_ERR_BAD_SIGNATURE;
            }
            missing++;
        } else if (incoming > 1) {
            set_err("verify: signature output '%s' has %d incoming edges (need 1)",
                    p->signature_out[so].name ? p->signature_out[so].name : "?", incoming);
            return PIPE_ERR_BAD_SIGNATURE;
        }
    }

    /* 6. Signature inputs (strict: must be used; partial: tolerate). */
    for (int si = 0; si < p->n_sig_in; si++) {
        int outgoing = 0;
        for (size_t e = 0; e < p->n_edges; e++) {
            const PipelineEdge *edge = p->edges[e];
            if (edge->src_node_idx == SIG_IN_NODE && edge->src_port_idx == si) outgoing++;
        }
        if (outgoing < 1) {
            if (strict) {
                set_err("verify: signature input '%s' is unused (must connect to >=1 port)",
                        p->signature_in[si].name ? p->signature_in[si].name : "?");
                return PIPE_ERR_BAD_SIGNATURE;
            }
            missing++;
        }
    }

    /* 7. Topological sort + cycle check.
     *
     * Use Kahn's algorithm with a lexicographic-id tiebreaker so the
     * output order is canonical: any two equivalent DAGs (same edges,
     * any insertion order) produce the same exec_order. This is
     * required for round-trip byte-equality of the rendered text. */
    if (p->exec_order) { free(p->exec_order); p->exec_order = NULL; }
    if (p->n_nodes == 0) { p->verified = strict ? 1 : 0; if (missing_out) *missing_out = missing; return PIPE_OK; }
    p->exec_order = (int *)calloc(p->n_nodes, sizeof(int));
    if (!p->exec_order) { set_err("OOM"); return PIPE_ERR_OOM; }
    int *indegree = (int *)calloc(p->n_nodes, sizeof(int));
    if (!indegree) { free(p->exec_order); p->exec_order = NULL; set_err("OOM"); return PIPE_ERR_OOM; }
    for (size_t e = 0; e < p->n_edges; e++) {
        const PipelineEdge *edge = p->edges[e];
        if (edge->src_node_idx >= 0 && edge->dst_node_idx >= 0) {
            indegree[edge->dst_node_idx]++;
        }
    }
    int emitted = 0;
    while (emitted < (int)p->n_nodes) {
        /* Pick the lexicographically smallest node with in-degree 0
         * that hasn't been emitted yet. */
        int pick = -1;
        for (size_t i = 0; i < p->n_nodes; i++) {
            if (indegree[i] != 0) continue;     /* already emitted (-1) or has deps */
            if (pick < 0) { pick = (int)i; continue; }
            if (strcmp(p->nodes[i]->id, p->nodes[pick]->id) < 0) pick = (int)i;
        }
        if (pick < 0) {
            free(indegree);
            free(p->exec_order); p->exec_order = NULL;
            set_err("verify: cycle detected (no node with in-degree 0 remaining)");
            return PIPE_ERR_CYCLE;
        }
        p->exec_order[emitted++] = pick;
        indegree[pick] = -1;                    /* mark emitted */
        for (size_t e = 0; e < p->n_edges; e++) {
            const PipelineEdge *edge = p->edges[e];
            if (edge->src_node_idx == pick && edge->dst_node_idx >= 0) {
                indegree[edge->dst_node_idx]--;
            }
        }
    }
    free(indegree);

    /* Only mark verified=1 in strict mode (partial graphs can't safely
     * be executed even if no hard errors). */
    if (strict) p->verified = 1;
    if (missing_out) *missing_out = missing;
    return PIPE_OK;
}

int pipeline_verify_partial(Pipeline *p, int *missing_out) {
    return verify_impl(p, /*strict=*/0, missing_out);
}

int pipeline_verify(Pipeline *p) {
    /* Strict mode of the shared verify_impl. */
    return verify_impl(p, /*strict=*/1, NULL);
}

/* ============================================================
 *  Execution
 * ============================================================ */

void pipeline_value_clear(PipelineValue *val) {
    if (!val || !val->type) return;
    switch (val->type->kind) {
    case PIPE_T_STRING:
        if (val->v.s) { free(val->v.s); val->v.s = NULL; }
        break;
    case PIPE_T_LIST: case PIPE_T_TENSOR: case PIPE_T_RECORD:
        if (val->v.opaque.data && val->v.opaque.free_fn) {
            val->v.opaque.free_fn(val->v.opaque.data);
            val->v.opaque.data = NULL;
        }
        break;
    default: break;
    }
}

/* Find the edge that feeds a given (dst_node, dst_port). Returns -1 if none. */
static int find_incoming_edge(const Pipeline *p, int dst_node, int dst_port) {
    for (size_t i = 0; i < p->n_edges; i++) {
        const PipelineEdge *e = p->edges[i];
        if (e->dst_node_idx == dst_node && e->dst_port_idx == dst_port) return (int)i;
    }
    return -1;
}

int pipeline_execute(const Pipeline *p,
                     const PipelineValue *inputs,
                     PipelineValue *outputs,
                     PipelineDispatchFn dispatch, void *user_data) {
    if (!p || !p->verified) {
        set_err("pipeline_execute: graph not verified");
        return PIPE_ERR_EXEC;
    }
    /* Materialise per-edge value storage. Each edge holds the value
     * produced by its source side; consumers read it. */
    PipelineValue *edge_values = (PipelineValue *)calloc(p->n_edges + 1, sizeof(PipelineValue));
    if (!edge_values) { set_err("OOM"); return PIPE_ERR_OOM; }

    /* Pre-fill edge values from signature inputs. */
    for (size_t e = 0; e < p->n_edges; e++) {
        const PipelineEdge *edge = p->edges[e];
        if (edge->src_node_idx == SIG_IN_NODE) {
            int si = edge->src_port_idx;
            edge_values[e] = inputs[si];
            edge_values[e].type = p->signature_in[si].type;
        }
    }

    int rc = PIPE_OK;
    /* Walk nodes in topological order. */
    for (size_t k = 0; k < p->n_nodes; k++) {
        int n_idx = p->exec_order[k];
        PipelineNode *node = p->nodes[n_idx];
        /* Gather inputs by walking the unique incoming edge per input port. */
        PipelineValue *in_vals = (PipelineValue *)calloc((size_t)(node->n_inputs > 0 ? node->n_inputs : 1), sizeof(PipelineValue));
        PipelineValue *out_vals = (PipelineValue *)calloc((size_t)(node->n_outputs > 0 ? node->n_outputs : 1), sizeof(PipelineValue));
        if (!in_vals || !out_vals) { free(in_vals); free(out_vals); rc = PIPE_ERR_OOM; goto cleanup; }
        for (int ip = 0; ip < node->n_inputs; ip++) {
            int eidx = find_incoming_edge(p, n_idx, ip);
            if (eidx < 0) { free(in_vals); free(out_vals); set_err("execute: missing edge for input port"); rc = PIPE_ERR_EXEC; goto cleanup; }
            in_vals[ip] = edge_values[eidx];
            in_vals[ip].type = node->inputs[ip].type;
        }
        /* Pre-set output value type slots. */
        for (int op = 0; op < node->n_outputs; op++) {
            out_vals[op].type = node->outputs[op].type;
        }
        /* Dispatch — leaf primitive via callback, OR recurse into subgraph. */
        if (node->subgraph) {
            int sub_rc = pipeline_execute(node->subgraph, in_vals, out_vals, dispatch, user_data);
            if (sub_rc != 0) { free(in_vals); free(out_vals); rc = sub_rc; goto cleanup; }
        } else {
            int dr = dispatch(node->primitive,
                              node->config, node->n_config,
                              in_vals, node->n_inputs,
                              out_vals, node->n_outputs,
                              user_data);
            if (dr != 0) { free(in_vals); free(out_vals); set_err("execute: dispatch returned %d for primitive '%s'", dr, node->primitive ? node->primitive : "?"); rc = PIPE_ERR_EXEC; goto cleanup; }
        }
        /* Propagate outputs along outgoing edges. */
        for (size_t e = 0; e < p->n_edges; e++) {
            const PipelineEdge *edge = p->edges[e];
            if (edge->src_node_idx != n_idx) continue;
            edge_values[e] = out_vals[edge->src_port_idx];
        }
        free(in_vals);
        free(out_vals);
    }

    /* Pull final outputs from edges feeding signature_out. */
    for (int so = 0; so < p->n_sig_out; so++) {
        int eidx = -1;
        for (size_t e = 0; e < p->n_edges; e++) {
            if (p->edges[e]->dst_node_idx == SIG_OUT_NODE && p->edges[e]->dst_port_idx == so) {
                eidx = (int)e; break;
            }
        }
        if (eidx < 0) { set_err("execute: signature output '%s' unconnected", p->signature_out[so].name ? p->signature_out[so].name : "?"); rc = PIPE_ERR_EXEC; goto cleanup; }
        outputs[so] = edge_values[eidx];
        outputs[so].type = p->signature_out[so].type;
    }

cleanup:
    free(edge_values);
    return rc;
}

/* ============================================================
 *  VM-backed dispatch (Phase 2 — API surface, deferred dispatch)
 * ============================================================
 *
 * The header declares pipeline_execute_vm(). The intended semantics
 * are: resolve each leaf primitive name to a registered vm_native_fn
 * in the supplied vm_engine, marshal PipelineValue ↔ double, dispatch.
 *
 * Phase 2 ships the API surface (so dependent code can compile and
 * tests can assert on the error path) but the internal dispatch is
 * deferred to Phase 3 because the public vm_engine API does not
 * provide:
 *   1. A way to enumerate or look up registered native functions
 *      from C — vm_engine_t's native_fns[] table is private to
 *      microgpt_vm.c.
 *   2. A way to invoke a registered native function with C-side
 *      arguments — vm_engine_run() takes only a fn_name and returns
 *      via the engine's result slot; arguments must come from a
 *      preloaded VM script.
 *
 * Working around this either requires (a) extending microgpt_vm.h
 * with an exported lookup-and-call API (cleanest) or (b) synthesising
 * a per-pipeline VM script that calls the registered fns in
 * topological order and runs it via vm_engine_run() (messier, but
 * keeps microgpt_vm unmodified).
 *
 * Phase 3 will choose between (a) and (b) based on whether the
 * Wiring Organelle work pushes for changes in microgpt_vm.h anyway.
 * Until then, callers should use pipeline_execute() with their own
 * (name, fn) lookup table — which is exactly what the VM dispatcher
 * would do internally.
 */
int pipeline_execute_vm(const Pipeline *p,
                        vm_engine *vm,
                        const PipelineValue *inputs,
                        PipelineValue *outputs) {
    (void)inputs; (void)outputs;
    if (!p) {
        set_err("pipeline_execute_vm: null pipeline");
        return PIPE_ERR_EXEC;
    }
    if (!vm) {
        set_err("pipeline_execute_vm: null vm_engine");
        return PIPE_ERR_EXEC;
    }
    set_err("pipeline_execute_vm: dispatch deferred to Phase 3 — "
            "the public vm_engine API doesn't expose native-fn lookup-and-call. "
            "Use pipeline_execute() with a host-supplied dispatcher that calls "
            "your fn pointers directly. See microgpt_pipeline.c §VM-backed dispatch "
            "for the design notes.");
    return PIPE_ERR_EXEC;
}

/* ============================================================
 *  Text serialisation
 * ============================================================ */

/* Simple growable string buffer. */
typedef struct { char *buf; size_t len; size_t cap; } SBuf;

static void sb_init(SBuf *s) { s->buf = NULL; s->len = 0; s->cap = 0; }
static void sb_free(SBuf *s) { if (s->buf) free(s->buf); s->buf = NULL; }
static int sb_reserve(SBuf *s, size_t need) {
    if (s->len + need + 1 <= s->cap) return 0;
    size_t newcap = s->cap ? s->cap * 2 : 256;
    while (newcap < s->len + need + 1) newcap *= 2;
    char *p = (char *)realloc(s->buf, newcap);
    if (!p) { set_err("OOM in sb_reserve"); return -1; }
    s->buf = p; s->cap = newcap;
    return 0;
}
static int sb_append(SBuf *s, const char *str) {
    size_t l = strlen(str);
    if (sb_reserve(s, l) < 0) return -1;
    memcpy(s->buf + s->len, str, l);
    s->len += l; s->buf[s->len] = '\0';
    return 0;
}
static int sb_appendf(SBuf *s, const char *fmt, ...) {
    char tmp[512];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return -1;
    if ((size_t)n < sizeof(tmp)) return sb_append(s, tmp);
    /* fallback for very long strings */
    char *big = (char *)malloc((size_t)n + 1);
    if (!big) { set_err("OOM"); return -1; }
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    int rc = sb_append(s, big);
    free(big);
    return rc;
}

static int render_type(SBuf *out, const PipelineType *t) {
    char tb[256] = "";
    pipeline_type_format(t, tb, sizeof(tb));
    return sb_append(out, tb);
}

char *pipeline_render_text(const Pipeline *p) {
    if (!p) return NULL;
    SBuf out; sb_init(&out);
    if (sb_appendf(&out, "@graph %s\n", p->name ? p->name : "graph") < 0) goto err;
    /* Signature inputs. */
    for (int i = 0; i < p->n_sig_in; i++) {
        if (sb_appendf(&out, "  : in %s -> ", p->signature_in[i].name ? p->signature_in[i].name : "?") < 0) goto err;
        if (render_type(&out, p->signature_in[i].type) < 0) goto err;
        if (sb_append(&out, "\n") < 0) goto err;
    }
    for (int i = 0; i < p->n_sig_out; i++) {
        if (sb_appendf(&out, "  : out %s -> ", p->signature_out[i].name ? p->signature_out[i].name : "?") < 0) goto err;
        if (render_type(&out, p->signature_out[i].type) < 0) goto err;
        if (sb_append(&out, "\n") < 0) goto err;
    }
    /* Nodes — emit in topological order if verified, else insertion order. */
    int *order = p->verified && p->exec_order ? p->exec_order : NULL;
    size_t count = p->n_nodes;
    for (size_t k = 0; k < count; k++) {
        size_t idx = order ? (size_t)order[k] : k;
        const PipelineNode *node = p->nodes[idx];
        if (sb_appendf(&out, "  | %s = %s(", node->id,
                       node->primitive ? node->primitive : "@subgraph") < 0) goto err;
        /* For each input port, find the source endpoint and render `port: src.port` form. */
        for (int ip = 0; ip < node->n_inputs; ip++) {
            int eidx = find_incoming_edge(p, (int)idx, ip);
            if (sb_appendf(&out, "%s%s: ", ip ? ", " : "",
                           node->inputs[ip].name ? node->inputs[ip].name : "?") < 0) goto err;
            if (eidx < 0) { if (sb_append(&out, "<unconnected>") < 0) goto err; continue; }
            const PipelineEdge *e = p->edges[eidx];
            if (e->src_node_idx == SIG_IN_NODE) {
                if (sb_appendf(&out, "<%s>",
                               p->signature_in[e->src_port_idx].name ?
                               p->signature_in[e->src_port_idx].name : "?") < 0) goto err;
            } else {
                if (sb_appendf(&out, "%s.%s",
                               p->nodes[e->src_node_idx]->id,
                               p->nodes[e->src_node_idx]->outputs[e->src_port_idx].name ?
                               p->nodes[e->src_node_idx]->outputs[e->src_port_idx].name : "?") < 0) goto err;
            }
        }
        if (sb_append(&out, ")") < 0) goto err;
        /* Phase 2: Type annotation suffix `:: in_port:type, ... -> out_port:type, ...`
         * Emitted only if any port has a non-ANY type (round-trip preserves
         * concrete types for verified graphs; ANY-only graphs round-trip
         * structurally without the suffix). */
        {
            int needs_annot = 0;
            for (int ip = 0; ip < node->n_inputs && !needs_annot; ip++)
                if (node->inputs[ip].type && node->inputs[ip].type->kind != PIPE_T_ANY)
                    needs_annot = 1;
            for (int op = 0; op < node->n_outputs && !needs_annot; op++)
                if (node->outputs[op].type && node->outputs[op].type->kind != PIPE_T_ANY)
                    needs_annot = 1;
            if (needs_annot) {
                if (sb_append(&out, " :: ") < 0) goto err;
                for (int ip = 0; ip < node->n_inputs; ip++) {
                    char tb[128] = "";
                    pipeline_type_format(node->inputs[ip].type, tb, sizeof(tb));
                    if (sb_appendf(&out, "%s%s:%s", ip ? ", " : "",
                                   node->inputs[ip].name ? node->inputs[ip].name : "?",
                                   tb) < 0) goto err;
                }
                if (sb_append(&out, " -> ") < 0) goto err;
                for (int op = 0; op < node->n_outputs; op++) {
                    char tb[128] = "";
                    pipeline_type_format(node->outputs[op].type, tb, sizeof(tb));
                    if (sb_appendf(&out, "%s%s:%s", op ? ", " : "",
                                   node->outputs[op].name ? node->outputs[op].name : "?",
                                   tb) < 0) goto err;
                }
            }
        }
        /* Config. */
        if (node->n_config > 0) {
            if (sb_append(&out, " #") < 0) goto err;
            for (int c = 0; c < node->n_config; c++) {
                const char *prefix = c ? ", " : " ";
                switch (node->config[c].kind) {
                case PIPE_CFG_INT:
                    if (sb_appendf(&out, "%s%s=%lld", prefix, node->config[c].name,
                                   (long long)node->config[c].v.i) < 0) goto err;
                    break;
                case PIPE_CFG_FLOAT:
                    if (sb_appendf(&out, "%s%s=%g", prefix, node->config[c].name,
                                   node->config[c].v.f) < 0) goto err;
                    break;
                case PIPE_CFG_STRING:
                    if (sb_appendf(&out, "%s%s=\"%s\"", prefix, node->config[c].name,
                                   node->config[c].v.s ? node->config[c].v.s : "") < 0) goto err;
                    break;
                }
            }
        }
        if (sb_append(&out, "\n") < 0) goto err;
    }
    /* Output bindings. */
    for (int so = 0; so < p->n_sig_out; so++) {
        int eidx = -1;
        for (size_t e = 0; e < p->n_edges; e++) {
            if (p->edges[e]->dst_node_idx == SIG_OUT_NODE && p->edges[e]->dst_port_idx == so) {
                eidx = (int)e; break;
            }
        }
        if (eidx >= 0) {
            const PipelineEdge *edge = p->edges[eidx];
            const char *src_id = edge->src_node_idx == SIG_IN_NODE ? "<sig>" :
                                 p->nodes[edge->src_node_idx]->id;
            const char *src_port = edge->src_node_idx == SIG_IN_NODE ?
                p->signature_in[edge->src_port_idx].name :
                p->nodes[edge->src_node_idx]->outputs[edge->src_port_idx].name;
            if (sb_appendf(&out, "  %s <- %s.%s\n",
                           p->signature_out[so].name ? p->signature_out[so].name : "?",
                           src_id, src_port ? src_port : "?") < 0) goto err;
        }
    }
    if (sb_append(&out, "@end\n") < 0) goto err;
    return out.buf;
err:
    sb_free(&out);
    return NULL;
}

/* ============================================================
 *  Text parser (recursive descent)
 *
 *  Phase-1 parser handles the canonical form emitted by render_text.
 *  It is deliberately minimal — accepts what we emit, rejects free-form
 *  variations. Quoted strings, identifiers, integers and floats only.
 *
 *  Robustness goal: round-trip property — render(parse(s)) yields a
 *  string equivalent to s for any verified graph that was rendered.
 * ============================================================ */

typedef struct { const char *src; const char *cur; int line; int col; } PState;

static void ps_advance(PState *ps, int n) {
    for (int i = 0; i < n && *ps->cur; i++) {
        if (*ps->cur == '\n') { ps->line++; ps->col = 1; } else { ps->col++; }
        ps->cur++;
    }
}
static void ps_skip_ws(PState *ps) {
    while (*ps->cur == ' ' || *ps->cur == '\t') ps_advance(ps, 1);
}
static void ps_skip_ws_nl(PState *ps) {
    while (*ps->cur == ' ' || *ps->cur == '\t' || *ps->cur == '\n' || *ps->cur == '\r')
        ps_advance(ps, 1);
}
static int ps_match_kw(PState *ps, const char *kw) {
    ps_skip_ws(ps);
    size_t l = strlen(kw);
    if (strncmp(ps->cur, kw, l) == 0) { ps_advance(ps, (int)l); return 1; }
    return 0;
}
static int ps_eat(PState *ps, char c) {
    ps_skip_ws(ps);
    if (*ps->cur == c) { ps_advance(ps, 1); return 1; }
    return 0;
}

static char *ps_read_ident(PState *ps) {
    ps_skip_ws(ps);
    const char *start = ps->cur;
    /* Allow `-` only as a leading sign character (for negative tensor
     * dimensions like `-1`). `.` is a separator between node id and port
     * name and must NOT be consumed by the identifier reader. */
    if (*ps->cur == '-') ps->cur++;
    while (isalnum((unsigned char)*ps->cur) || *ps->cur == '_') ps->cur++;
    if (ps->cur == start) return NULL;
    size_t l = (size_t)(ps->cur - start);
    char *id = (char *)malloc(l + 1);
    if (!id) return NULL;
    memcpy(id, start, l); id[l] = '\0';
    /* Adjust col tracking. */
    ps->col += (int)l;
    return id;
}

static char *ps_read_quoted_string(PState *ps) {
    ps_skip_ws(ps);
    if (*ps->cur != '"') return NULL;
    ps_advance(ps, 1);
    const char *start = ps->cur;
    while (*ps->cur && *ps->cur != '"') ps->cur++;
    if (*ps->cur != '"') { set_err("parse: unterminated string at line %d", ps->line); return NULL; }
    size_t l = (size_t)(ps->cur - start);
    char *s = (char *)malloc(l + 1);
    if (!s) return NULL;
    memcpy(s, start, l); s[l] = '\0';
    ps_advance(ps, 1);
    ps->col += (int)l;
    return s;
}

/* Read a type. Recursive for list[] / tensor[] / record{}. */
static PipelineType *ps_read_type(PState *ps) {
    ps_skip_ws(ps);
    char *kw = ps_read_ident(ps);
    if (!kw) { set_err("parse: expected type at line %d:%d", ps->line, ps->col); return NULL; }
    PipelineType *t = NULL;
    if (strcmp(kw, "void") == 0) t = pipeline_type_void();
    else if (strcmp(kw, "int") == 0) t = pipeline_type_int();
    else if (strcmp(kw, "float") == 0) t = pipeline_type_float();
    else if (strcmp(kw, "string") == 0) t = pipeline_type_string();
    else if (strcmp(kw, "any") == 0) t = pipeline_type_any();
    else if (strcmp(kw, "list") == 0) {
        free(kw);
        if (!ps_eat(ps, '[')) { set_err("parse: list expects '['"); return NULL; }
        PipelineType *inner = ps_read_type(ps);
        if (!inner) return NULL;
        if (!ps_eat(ps, ']')) { pipeline_type_free(inner); set_err("parse: list missing ']'"); return NULL; }
        return pipeline_type_list(inner);
    }
    else if (strcmp(kw, "tensor") == 0) {
        free(kw);
        if (!ps_eat(ps, '[')) { set_err("parse: tensor expects '['"); return NULL; }
        PipelineType *inner = ps_read_type(ps);
        if (!inner) return NULL;
        int dims[16]; int n_dims = 0;
        while (ps_eat(ps, ',')) {
            ps_skip_ws(ps);
            if (n_dims >= 16) { pipeline_type_free(inner); set_err("parse: too many tensor dims"); return NULL; }
            if (*ps->cur == '*') { ps_advance(ps, 1); dims[n_dims++] = -1; }
            else {
                char *num = ps_read_ident(ps);
                if (!num) { pipeline_type_free(inner); set_err("parse: tensor dim"); return NULL; }
                dims[n_dims++] = atoi(num); free(num);
            }
        }
        if (!ps_eat(ps, ']')) { pipeline_type_free(inner); set_err("parse: tensor missing ']'"); return NULL; }
        return pipeline_type_tensor(inner, n_dims, dims);
    }
    else if (strcmp(kw, "record") == 0) {
        free(kw);
        if (!ps_eat(ps, '{')) { set_err("parse: record expects '{'"); return NULL; }
        const char *fnames[32]; PipelineType *ftypes[32]; int n_f = 0;
        while (!ps_eat(ps, '}')) {
            if (n_f > 0 && !ps_eat(ps, ',')) { set_err("parse: record field separator"); break; }
            ps_skip_ws(ps);
            char *fname = ps_read_ident(ps);
            if (!fname) { set_err("parse: record field name"); break; }
            if (!ps_eat(ps, ':')) { free(fname); set_err("parse: record ':' expected"); break; }
            PipelineType *ft = ps_read_type(ps);
            if (!ft) { free(fname); break; }
            if (n_f >= 32) { free(fname); pipeline_type_free(ft); set_err("parse: too many record fields"); break; }
            fnames[n_f] = fname;  /* leak the strdup'd ident — record will copy. */
            ftypes[n_f] = ft; n_f++;
        }
        PipelineType *rec = pipeline_type_record(n_f, fnames, ftypes);
        for (int i = 0; i < n_f; i++) free((char *)fnames[i]);
        return rec;
    }
    if (kw) free(kw);
    return t;
}

Pipeline *pipeline_parse_text(const char *src) {
    if (!src) return NULL;
    PState ps = { src, src, 1, 1 };
    ps_skip_ws_nl(&ps);
    if (!ps_match_kw(&ps, "@graph")) { set_err("parse: expected '@graph'"); return NULL; }
    char *name = ps_read_ident(&ps);
    if (!name) { set_err("parse: expected graph name"); return NULL; }
    Pipeline *p = pipeline_create(name);
    free(name);
    if (!p) return NULL;
    ps_skip_ws_nl(&ps);

    /* Collect signature in/out, then nodes, then output bindings.
     * The format we emit puts ":" lines first, "|" lines next, "<-" last. */
    const char *sig_in_names[64];  PipelineType *sig_in_types[64];  int n_sig_in = 0;
    const char *sig_out_names[64]; PipelineType *sig_out_types[64]; int n_sig_out = 0;

    /* Phase 1: signature lines. */
    while (*ps.cur && ps_match_kw(&ps, ":")) {
        ps_skip_ws(&ps);
        char *kind = ps_read_ident(&ps);
        if (!kind) {
            /* Could be malformed input or just end-of-signature. Stop
             * gracefully rather than goto fail, which leaks the signature
             * name/type arrays. */
            set_err("parse: signature kind expected");
            break;
        }
        char *port_name = ps_read_ident(&ps);
        if (!port_name) { free(kind); set_err("parse: signature port name"); break; }
        if (!ps_match_kw(&ps, "->")) { free(kind); free(port_name); set_err("parse: signature '->' expected"); break; }
        PipelineType *t = ps_read_type(&ps);
        if (!t) { free(kind); free(port_name); break; }
        if (strcmp(kind, "in") == 0) {
            if (n_sig_in >= 64) { free(kind); free(port_name); pipeline_type_free(t); set_err("parse: too many sig inputs"); goto fail; }
            sig_in_names[n_sig_in] = port_name;
            sig_in_types[n_sig_in] = t;
            n_sig_in++;
        } else if (strcmp(kind, "out") == 0) {
            if (n_sig_out >= 64) { free(kind); free(port_name); pipeline_type_free(t); set_err("parse: too many sig outputs"); goto fail; }
            sig_out_names[n_sig_out] = port_name;
            sig_out_types[n_sig_out] = t;
            n_sig_out++;
        } else {
            free(kind); free(port_name); pipeline_type_free(t);
            set_err("parse: unknown signature kind"); goto fail;
        }
        free(kind);
        ps_skip_ws_nl(&ps);
    }
    if (pipeline_set_signature(p, n_sig_in, sig_in_names, sig_in_types,
                               n_sig_out, sig_out_names, sig_out_types) != 0) {
        for (int i = 0; i < n_sig_in; i++) free((char *)sig_in_names[i]);
        for (int i = 0; i < n_sig_out; i++) free((char *)sig_out_names[i]);
        goto fail;
    }
    for (int i = 0; i < n_sig_in; i++) free((char *)sig_in_names[i]);
    for (int i = 0; i < n_sig_out; i++) free((char *)sig_out_names[i]);

    /* Phase 2: node lines. We collect them, defer connections to phase 3
     * because edges may reference forward declarations. We don't have port
     * type information yet — give nodes ANY-typed ports until output
     * bindings tell us better. For Phase-1, the parse target is canonical
     * round-tripping, so port types are recovered from the connect lines'
     * source ports as the graph is rebuilt. */
    /* Phase 2: ParsedNode now optionally carries port-type annotations
     * recovered from the `::` suffix. If absent, ports default to ANY
     * (Phase-1 behaviour). Multi-output nodes are also supported. */
    typedef struct {
        char *id; char *prim;
        char **in_names; char **in_src_node; char **in_src_port; int n_in;
        PipelineType **in_types;            /* NULL or array of length n_in */
        char **out_names;                   /* NULL or array of length n_out */
        PipelineType **out_types;           /* NULL or array of length n_out */
        int n_out;
    } ParsedNode;
    ParsedNode *pn = NULL; int n_pn = 0; int cap_pn = 0;

    while (*ps.cur && *ps.cur != '@') {
        ps_skip_ws_nl(&ps);
        if (!*ps.cur || *ps.cur == '@') break;
        if (*ps.cur == '|') {
            ps_advance(&ps, 1);
            ps_skip_ws(&ps);
            char *id = ps_read_ident(&ps);
            if (!id) { set_err("parse: node id"); goto fail2; }
            if (!ps_eat(&ps, '=')) { free(id); set_err("parse: '=' after node id"); goto fail2; }
            char *prim = ps_read_ident(&ps);
            if (!prim) { free(id); set_err("parse: node primitive"); goto fail2; }
            if (!ps_eat(&ps, '(')) { free(id); free(prim); set_err("parse: '(' after primitive"); goto fail2; }
            if (n_pn >= cap_pn) { cap_pn = cap_pn ? cap_pn * 2 : 16; pn = (ParsedNode *)realloc(pn, sizeof(ParsedNode) * (size_t)cap_pn); }
            ParsedNode *cur = &pn[n_pn++];
            memset(cur, 0, sizeof(*cur));
            cur->id = id; cur->prim = prim;
            int cap_in = 8;
            cur->in_names = (char **)calloc((size_t)cap_in, sizeof(char *));
            cur->in_src_node = (char **)calloc((size_t)cap_in, sizeof(char *));
            cur->in_src_port = (char **)calloc((size_t)cap_in, sizeof(char *));
            ps_skip_ws(&ps);
            while (*ps.cur != ')') {
                if (cur->n_in > 0 && !ps_eat(&ps, ',')) break;
                ps_skip_ws(&ps);
                char *pname = ps_read_ident(&ps);
                if (!pname) break;
                if (!ps_eat(&ps, ':')) { free(pname); break; }
                ps_skip_ws(&ps);
                char *src_node = NULL, *src_port = NULL;
                if (*ps.cur == '<') {
                    ps_advance(&ps, 1);
                    src_node = strdup("<sig>");
                    src_port = ps_read_ident(&ps);
                    ps_eat(&ps, '>');
                } else {
                    src_node = ps_read_ident(&ps);
                    if (!src_node) { free(pname); break; }
                    ps_eat(&ps, '.');
                    src_port = ps_read_ident(&ps);
                }
                /* Defensive: if src_port is NULL (malformed input — e.g.
                 * `<` followed by non-ident, or `node.` followed by non-
                 * ident), bail out of the arg loop cleanly. */
                if (!src_port) {
                    free(pname); free(src_node);
                    break;
                }
                if (cur->n_in >= cap_in) {
                    cap_in *= 2;
                    cur->in_names = (char **)realloc(cur->in_names, sizeof(char *) * (size_t)cap_in);
                    cur->in_src_node = (char **)realloc(cur->in_src_node, sizeof(char *) * (size_t)cap_in);
                    cur->in_src_port = (char **)realloc(cur->in_src_port, sizeof(char *) * (size_t)cap_in);
                }
                cur->in_names[cur->n_in] = pname;
                cur->in_src_node[cur->n_in] = src_node;
                cur->in_src_port[cur->n_in] = src_port;
                cur->n_in++;
                ps_skip_ws(&ps);
            }
            ps_eat(&ps, ')');
            /* Phase 2: optional `:: in:type, ... -> out:type, ...` suffix. */
            ps_skip_ws(&ps);
            if (ps_match_kw(&ps, "::")) {
                /* Inputs */
                int cap_t = 8;
                cur->in_types  = (PipelineType **)calloc((size_t)cap_t, sizeof(PipelineType *));
                cur->out_names = (char **)calloc((size_t)cap_t, sizeof(char *));
                cur->out_types = (PipelineType **)calloc((size_t)cap_t, sizeof(PipelineType *));
                int n_in_t = 0;
                ps_skip_ws(&ps);
                while (*ps.cur != '-' && *ps.cur && *ps.cur != '\n') {
                    if (n_in_t > 0 && !ps_eat(&ps, ',')) break;
                    ps_skip_ws(&ps);
                    char *pn_name = ps_read_ident(&ps);
                    if (!pn_name) break;
                    if (!ps_eat(&ps, ':')) { free(pn_name); break; }
                    PipelineType *pt = ps_read_type(&ps);
                    if (!pt) { free(pn_name); break; }
                    /* Match by name to the existing in_names entry. */
                    int matched = 0;
                    for (int k = 0; k < cur->n_in; k++) {
                        if (strcmp(cur->in_names[k], pn_name) == 0) {
                            if (n_in_t >= cap_t) {
                                cap_t *= 2;
                                cur->in_types = (PipelineType **)realloc(cur->in_types, sizeof(PipelineType *) * (size_t)cap_t);
                            }
                            /* Store at slot k, padding NULLs as needed. */
                            while (n_in_t <= k) { cur->in_types[n_in_t++] = NULL; }
                            cur->in_types[k] = pt;
                            matched = 1;
                            break;
                        }
                    }
                    if (!matched) pipeline_type_free(pt);
                    free(pn_name);
                    ps_skip_ws(&ps);
                }
                /* "->" separator */
                if (ps_match_kw(&ps, "->")) {
                    int n_out_t = 0;
                    ps_skip_ws(&ps);
                    while (*ps.cur && *ps.cur != '\n' && *ps.cur != '#') {
                        if (n_out_t > 0 && !ps_eat(&ps, ',')) break;
                        ps_skip_ws(&ps);
                        char *pn_name = ps_read_ident(&ps);
                        if (!pn_name) break;
                        if (!ps_eat(&ps, ':')) { free(pn_name); break; }
                        PipelineType *pt = ps_read_type(&ps);
                        if (!pt) { free(pn_name); break; }
                        if (n_out_t >= cap_t) {
                            cap_t *= 2;
                            cur->out_names = (char **)realloc(cur->out_names, sizeof(char *) * (size_t)cap_t);
                            cur->out_types = (PipelineType **)realloc(cur->out_types, sizeof(PipelineType *) * (size_t)cap_t);
                        }
                        cur->out_names[n_out_t] = pn_name;
                        cur->out_types[n_out_t] = pt;
                        n_out_t++;
                        ps_skip_ws(&ps);
                    }
                    cur->n_out = n_out_t;
                }
            }
            /* Skip optional config (Phase 1: ignore values for round-trip). */
            ps_skip_ws(&ps);
            if (*ps.cur == '#') {
                while (*ps.cur && *ps.cur != '\n') ps_advance(&ps, 1);
            }
            ps_skip_ws_nl(&ps);
            continue;
        }
        /* Output binding lines: name <- node.port */
        char *bind_name = ps_read_ident(&ps);
        if (!bind_name) break;
        if (!ps_match_kw(&ps, "<-")) { free(bind_name); break; }
        ps_skip_ws(&ps);
        char *src_node = ps_read_ident(&ps);
        ps_eat(&ps, '.');
        char *src_port = ps_read_ident(&ps);
        /* Defensive: malformed binding line. Skip this entry rather than
         * stash NULL pointers into the wiring loop's arrays. */
        if (!src_node || !src_port) {
            free(bind_name);
            if (src_node) free(src_node);
            if (src_port) free(src_port);
            ps_skip_ws_nl(&ps);
            continue;
        }
        /* Defer wiring until after nodes exist. We'll re-find bind_name in signature_out. */
        /* Add a parsed-node-style entry for the binding so we can wire after. */
        int sig_idx = find_signature_port(p, bind_name, 1);
        if (sig_idx >= 0 && src_node && src_port) {
            /* Defer until all nodes inserted — store as outputs to wire below. */
        }
        free(bind_name);
        /* Save src_node/src_port for second pass. We reuse pn list for simplicity:
         * a "binding" entry is encoded as a ParsedNode with id="<bind>" and prim
         * holding the binding name. Quick & dirty for Phase 1. */
        if (n_pn >= cap_pn) { cap_pn = cap_pn ? cap_pn * 2 : 16; pn = (ParsedNode *)realloc(pn, sizeof(ParsedNode) * (size_t)cap_pn); }
        ParsedNode *cur = &pn[n_pn++];
        memset(cur, 0, sizeof(*cur));
        cur->id = strdup("<bind>");
        cur->prim = NULL;  /* nothing to look up */
        cur->in_names = (char **)calloc(1, sizeof(char *));
        cur->in_src_node = (char **)calloc(1, sizeof(char *));
        cur->in_src_port = (char **)calloc(1, sizeof(char *));
        cur->in_names[0] = strdup("<bind>");  /* placeholder */
        cur->in_src_node[0] = src_node;
        cur->in_src_port[0] = src_port;
        cur->n_in = 1;
        ps_skip_ws_nl(&ps);
    }

    /* @end */
    ps_match_kw(&ps, "@end");

    /* Build nodes. Phase 2: use parsed `::` types when present, else
     * fall back to ANY (Phase-1 behaviour). */
    for (int i = 0; i < n_pn; i++) {
        ParsedNode *cur = &pn[i];
        if (!cur->prim) continue;  /* skip binding entries */
        const char **in_names = (const char **)calloc((size_t)(cur->n_in > 0 ? cur->n_in : 1), sizeof(char *));
        PipelineType **in_types = (PipelineType **)calloc((size_t)(cur->n_in > 0 ? cur->n_in : 1), sizeof(PipelineType *));
        for (int k = 0; k < cur->n_in; k++) {
            in_names[k] = cur->in_names[k];
            if (cur->in_types && cur->in_types[k])
                in_types[k] = cur->in_types[k];      /* take ownership */
            else
                in_types[k] = pipeline_type_any();
        }
        /* Output ports. If parsed `::` provided them, use those; else
         * fall back to a single ANY-typed "out" port. */
        const char **out_names_arr;
        PipelineType **out_types_arr;
        int n_out;
        if (cur->n_out > 0) {
            out_names_arr = (const char **)calloc((size_t)cur->n_out, sizeof(char *));
            out_types_arr = (PipelineType **)calloc((size_t)cur->n_out, sizeof(PipelineType *));
            for (int k = 0; k < cur->n_out; k++) {
                out_names_arr[k] = cur->out_names[k];
                out_types_arr[k] = cur->out_types[k]; /* take ownership */
            }
            n_out = cur->n_out;
        } else {
            out_names_arr = (const char **)malloc(sizeof(char *));
            out_types_arr = (PipelineType **)malloc(sizeof(PipelineType *));
            out_names_arr[0] = "out";
            out_types_arr[0] = pipeline_type_any();
            n_out = 1;
        }
        int add_rc = pipeline_add_node(p, cur->id, cur->prim,
                                       cur->n_in, in_names, in_types,
                                       n_out, out_names_arr, out_types_arr);
        free((void *)in_names); free(in_types);
        free((void *)out_names_arr); free(out_types_arr);
        /* In BOTH cases (success and failure), pipeline_add_node has either
         * taken ownership of the type pointers OR has freed them itself
         * (on duplicate-id / OOM). Either way, the parser must NOT free
         * them again in the cleanup path — null them out unconditionally. */
        if (cur->in_types) {
            for (int k = 0; k < cur->n_in; k++) cur->in_types[k] = NULL;
        }
        if (cur->out_types) {
            for (int k = 0; k < cur->n_out; k++) cur->out_types[k] = NULL;
        }
        if (add_rc < 0) {
            /* Soft-fail: skip this node, continue parsing remaining ones.
             * The model may have emitted duplicate node ids — well-formed
             * but invalid. Verification will report the underlying issue. */
            continue;
        }
    }

    /* Wire edges. */
    for (int i = 0; i < n_pn; i++) {
        ParsedNode *cur = &pn[i];
        if (!cur->prim) continue;
        for (int k = 0; k < cur->n_in; k++) {
            const char *sn = cur->in_src_node[k];
            const char *sp = cur->in_src_port[k];
            if (!sn || !sp) continue;
            if (strcmp(sn, "<sig>") == 0) {
                pipeline_connect_signature_in(p, sp, cur->id, cur->in_names[k]);
            } else {
                pipeline_connect(p, sn, sp, cur->id, cur->in_names[k]);
            }
        }
    }
    /* Wire output bindings. */
    for (int i = 0; i < n_pn; i++) {
        ParsedNode *cur = &pn[i];
        if (cur->prim) continue;
        if (cur->n_in != 1 || !cur->in_src_node[0]) continue;
        /* "id" was "<bind>" but we stashed the actual bind name elsewhere — Phase 1
         * iterates every signature_out and connects to the first matching unconnected
         * one. For our round-trip canonical form this produces correct output ordering. */
        /* TODO: store bind_name properly; for Phase-1 the simpler version is to rely on
         * insertion order matching signature_out order. */
        /* Connect the kth unconnected signature_out. */
        for (int so = 0; so < p->n_sig_out; so++) {
            int already = 0;
            for (size_t e = 0; e < p->n_edges; e++) {
                if (p->edges[e]->dst_node_idx == SIG_OUT_NODE && p->edges[e]->dst_port_idx == so) { already = 1; break; }
            }
            if (already) continue;
            pipeline_connect_signature_out(p, cur->in_src_node[0], cur->in_src_port[0],
                                           p->signature_out[so].name);
            break;
        }
    }

    /* Cleanup parsed nodes. Free any types still owned (NULL'd if transferred). */
    for (int i = 0; i < n_pn; i++) {
        free(pn[i].id);
        if (pn[i].prim) free(pn[i].prim);
        for (int k = 0; k < pn[i].n_in; k++) {
            free(pn[i].in_names[k]);
            free(pn[i].in_src_node[k]);
            free(pn[i].in_src_port[k]);
        }
        free(pn[i].in_names); free(pn[i].in_src_node); free(pn[i].in_src_port);
        if (pn[i].in_types) {
            for (int k = 0; k < pn[i].n_in; k++)
                if (pn[i].in_types[k]) pipeline_type_free(pn[i].in_types[k]);
            free(pn[i].in_types);
        }
        if (pn[i].out_names) {
            for (int k = 0; k < pn[i].n_out; k++)
                if (pn[i].out_names[k]) free(pn[i].out_names[k]);
            free(pn[i].out_names);
        }
        if (pn[i].out_types) {
            for (int k = 0; k < pn[i].n_out; k++)
                if (pn[i].out_types[k]) pipeline_type_free(pn[i].out_types[k]);
            free(pn[i].out_types);
        }
    }
    free(pn);
    return p;

fail2:
    for (int i = 0; i < n_pn; i++) {
        free(pn[i].id);
        if (pn[i].prim) free(pn[i].prim);
        for (int k = 0; k < pn[i].n_in; k++) {
            free(pn[i].in_names[k]);
            free(pn[i].in_src_node[k]);
            free(pn[i].in_src_port[k]);
        }
        free(pn[i].in_names); free(pn[i].in_src_node); free(pn[i].in_src_port);
        if (pn[i].in_types) {
            for (int k = 0; k < pn[i].n_in; k++)
                if (pn[i].in_types[k]) pipeline_type_free(pn[i].in_types[k]);
            free(pn[i].in_types);
        }
        if (pn[i].out_names) {
            for (int k = 0; k < pn[i].n_out; k++)
                if (pn[i].out_names[k]) free(pn[i].out_names[k]);
            free(pn[i].out_names);
        }
        if (pn[i].out_types) {
            for (int k = 0; k < pn[i].n_out; k++)
                if (pn[i].out_types[k]) pipeline_type_free(pn[i].out_types[k]);
            free(pn[i].out_types);
        }
    }
    free(pn);
fail:
    pipeline_free(p);
    return NULL;
}

/* ============================================================
 *  DOT renderer
 * ============================================================ */

char *pipeline_render_dot(const Pipeline *p) {
    if (!p) return NULL;
    SBuf out; sb_init(&out);
    if (sb_appendf(&out, "digraph %s {\n", p->name ? p->name : "graph") < 0) goto err;
    if (sb_append(&out, "  rankdir=TB;\n") < 0) goto err;
    if (sb_append(&out, "  node [shape=record, fontname=\"Helvetica\"];\n") < 0) goto err;
    /* Signature inputs as ellipses on top. */
    for (int i = 0; i < p->n_sig_in; i++) {
        char tb[128] = ""; pipeline_type_format(p->signature_in[i].type, tb, sizeof(tb));
        if (sb_appendf(&out, "  sig_in_%d [shape=ellipse, label=\"%s\\n%s\", style=filled, fillcolor=\"#cdeefd\"];\n",
                       i, p->signature_in[i].name ? p->signature_in[i].name : "?", tb) < 0) goto err;
    }
    /* Nodes as records with input/output ports. */
    for (size_t n = 0; n < p->n_nodes; n++) {
        const PipelineNode *node = p->nodes[n];
        if (sb_appendf(&out, "  node_%zu [label=\"{ {", n) < 0) goto err;
        for (int ip = 0; ip < node->n_inputs; ip++) {
            if (sb_appendf(&out, "%s<in%d> %s", ip ? " | " : "", ip,
                           node->inputs[ip].name ? node->inputs[ip].name : "?") < 0) goto err;
        }
        if (sb_appendf(&out, "} | %s\\n%s | {",
                       node->id, node->primitive ? node->primitive : "@subgraph") < 0) goto err;
        for (int op = 0; op < node->n_outputs; op++) {
            if (sb_appendf(&out, "%s<out%d> %s", op ? " | " : "", op,
                           node->outputs[op].name ? node->outputs[op].name : "?") < 0) goto err;
        }
        if (sb_append(&out, "} }\"];\n") < 0) goto err;
    }
    /* Signature outputs as ellipses on bottom. */
    for (int i = 0; i < p->n_sig_out; i++) {
        char tb[128] = ""; pipeline_type_format(p->signature_out[i].type, tb, sizeof(tb));
        if (sb_appendf(&out, "  sig_out_%d [shape=ellipse, label=\"%s\\n%s\", style=filled, fillcolor=\"#fdcdcd\"];\n",
                       i, p->signature_out[i].name ? p->signature_out[i].name : "?", tb) < 0) goto err;
    }
    /* Edges. */
    for (size_t e = 0; e < p->n_edges; e++) {
        const PipelineEdge *edge = p->edges[e];
        char src_buf[64], dst_buf[64], label[128] = "";
        if (edge->src_node_idx == SIG_IN_NODE)
            snprintf(src_buf, sizeof(src_buf), "sig_in_%d", edge->src_port_idx);
        else
            snprintf(src_buf, sizeof(src_buf), "node_%d:out%d", edge->src_node_idx, edge->src_port_idx);
        if (edge->dst_node_idx == SIG_OUT_NODE)
            snprintf(dst_buf, sizeof(dst_buf), "sig_out_%d", edge->dst_port_idx);
        else
            snprintf(dst_buf, sizeof(dst_buf), "node_%d:in%d", edge->dst_node_idx, edge->dst_port_idx);
        const PipelineType *st = edge_src_type(p, edge);
        if (st) pipeline_type_format(st, label, sizeof(label));
        if (sb_appendf(&out, "  %s -> %s [label=\"%s\", fontsize=10];\n", src_buf, dst_buf, label) < 0) goto err;
    }
    if (sb_append(&out, "}\n") < 0) goto err;
    return out.buf;
err:
    sb_free(&out);
    return NULL;
}
