/*
 * MicroGPT-C — Pipeline IR Test Suite
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Minimal test harness (matches the engine's existing TEST() pattern) ---- */

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

/* Per-test failure flag — set by ASSERT-failed paths so the runner
 * knows not to report PASS afterward. */
static int g_current_test_failed = 0;

#define TEST(name)                                                             \
  static void test_##name(void);                                               \
  static void run_##name(void) {                                               \
    g_tests_run++;                                                             \
    g_current_test_failed = 0;                                                 \
    printf("  %-60s ", #name);                                                 \
    fflush(stdout);                                                            \
    test_##name();                                                             \
    if (!g_current_test_failed) {                                              \
      printf("PASS\n"); g_tests_passed++;                                      \
    }                                                                          \
    fflush(stdout);                                                            \
  }                                                                            \
  static void test_##name(void)

#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL\n    %s:%d  Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
      g_tests_failed++;                                                        \
      g_current_test_failed = 1;                                               \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_STREQ(a, b) ASSERT((a) && (b) && strcmp((a), (b)) == 0)

#define RUN(name) run_##name()

/* ============================================================
 *  Type system
 * ============================================================ */

TEST(type_constructors_basic) {
    PipelineType *t = pipeline_type_int();
    ASSERT(t); ASSERT_EQ(t->kind, PIPE_T_INT);
    pipeline_type_free(t);
}

TEST(type_equal_basic) {
    PipelineType *a = pipeline_type_int();
    PipelineType *b = pipeline_type_int();
    PipelineType *c = pipeline_type_float();
    ASSERT(pipeline_type_equal(a, b));
    ASSERT(!pipeline_type_equal(a, c));
    pipeline_type_free(a); pipeline_type_free(b); pipeline_type_free(c);
}

TEST(type_any_matches_anything) {
    PipelineType *any = pipeline_type_any();
    PipelineType *intt = pipeline_type_int();
    PipelineType *str  = pipeline_type_string();
    ASSERT(pipeline_type_equal(any, intt));
    ASSERT(pipeline_type_equal(intt, any));
    ASSERT(pipeline_type_equal(any, str));
    pipeline_type_free(any); pipeline_type_free(intt); pipeline_type_free(str);
}

TEST(type_list_recursive) {
    PipelineType *a = pipeline_type_list(pipeline_type_int());
    PipelineType *b = pipeline_type_list(pipeline_type_int());
    PipelineType *c = pipeline_type_list(pipeline_type_float());
    ASSERT(pipeline_type_equal(a, b));
    ASSERT(!pipeline_type_equal(a, c));
    pipeline_type_free(a); pipeline_type_free(b); pipeline_type_free(c);
}

TEST(type_tensor_dims) {
    int d1[] = {3, 224, 224};
    int d2[] = {3, 224, 224};
    int d3[] = {1, 224, 224};
    int d4[] = {-1, 224, 224};  /* wildcard */
    PipelineType *a = pipeline_type_tensor(pipeline_type_float(), 3, d1);
    PipelineType *b = pipeline_type_tensor(pipeline_type_float(), 3, d2);
    PipelineType *c = pipeline_type_tensor(pipeline_type_float(), 3, d3);
    PipelineType *w = pipeline_type_tensor(pipeline_type_float(), 3, d4);
    ASSERT(pipeline_type_equal(a, b));
    ASSERT(!pipeline_type_equal(a, c));
    ASSERT(pipeline_type_equal(a, w));   /* wildcard matches concrete */
    ASSERT(pipeline_type_equal(c, w));
    pipeline_type_free(a); pipeline_type_free(b); pipeline_type_free(c); pipeline_type_free(w);
}

TEST(type_clone_deep_copy) {
    int dims[] = {10, 10};
    PipelineType *t = pipeline_type_tensor(pipeline_type_float(), 2, dims);
    PipelineType *c = pipeline_type_clone(t);
    ASSERT(pipeline_type_equal(t, c));
    /* Mutate clone — original unchanged. */
    c->dims[0] = 5;
    ASSERT(t->dims[0] == 10);
    pipeline_type_free(t); pipeline_type_free(c);
}

TEST(type_format_pretty_print) {
    char buf[256];
    PipelineType *t = pipeline_type_int();
    pipeline_type_format(t, buf, sizeof(buf));
    ASSERT_STREQ(buf, "int");
    pipeline_type_free(t);

    t = pipeline_type_list(pipeline_type_float());
    pipeline_type_format(t, buf, sizeof(buf));
    ASSERT_STREQ(buf, "list[float]");
    pipeline_type_free(t);

    int dims[] = {3};
    t = pipeline_type_tensor(pipeline_type_float(), 1, dims);
    pipeline_type_format(t, buf, sizeof(buf));
    ASSERT_STREQ(buf, "tensor[float, 3]");
    pipeline_type_free(t);
}

/* ============================================================
 *  Pipeline construction
 * ============================================================ */

TEST(pipeline_create_and_free) {
    Pipeline *p = pipeline_create("test_graph");
    ASSERT(p);
    ASSERT_STREQ(p->name, "test_graph");
    ASSERT_EQ(p->n_nodes, (size_t)0);
    pipeline_free(p);
}

TEST(pipeline_add_node_basic) {
    Pipeline *p = pipeline_create("g");
    const char *in_names[]  = {"x", "y"};
    PipelineType *in_types[] = {pipeline_type_int(), pipeline_type_int()};
    const char *out_names[] = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    int idx = pipeline_add_node(p, "n1", "add",
                                2, in_names, in_types,
                                1, out_names, out_types);
    ASSERT(idx >= 0);
    ASSERT_EQ(p->n_nodes, (size_t)1);
    ASSERT_STREQ(p->nodes[0]->id, "n1");
    pipeline_free(p);
}

TEST(pipeline_add_node_duplicate_id_rejected) {
    Pipeline *p = pipeline_create("g");
    const char *in_names[]  = {"x"};
    PipelineType *in_types[]  = {pipeline_type_int()};
    const char *out_names[] = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    PipelineType *in_types2[]  = {pipeline_type_int()};
    PipelineType *out_types2[] = {pipeline_type_int()};
    ASSERT(pipeline_add_node(p, "n1", "f", 1, in_names, in_types, 1, out_names, out_types) >= 0);
    int rc = pipeline_add_node(p, "n1", "g", 1, in_names, in_types2, 1, out_names, out_types2);
    ASSERT_EQ(rc, PIPE_ERR_DUP_NODE_ID);
    pipeline_free(p);
}

TEST(pipeline_connect_unknown_node_rejected) {
    Pipeline *p = pipeline_create("g");
    int rc = pipeline_connect(p, "missing", "out", "alsomissing", "in");
    ASSERT_EQ(rc, PIPE_ERR_UNKNOWN_NODE);
    pipeline_free(p);
}

/* ============================================================
 *  Verification
 * ============================================================ */

/* Helper: build a simple "y = a + b" graph. */
static Pipeline *build_add_graph(void) {
    Pipeline *p = pipeline_create("add_graph");

    /* Signature: in a:int, in b:int, out y:int */
    const char *sig_in_names[]  = {"a", "b"};
    PipelineType *sig_in_types[] = {pipeline_type_int(), pipeline_type_int()};
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 2, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    /* Node: add(x, y) -> out */
    const char *in_names[]   = {"x", "y"};
    PipelineType *in_types[] = {pipeline_type_int(), pipeline_type_int()};
    const char *out_names[]  = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    pipeline_add_node(p, "n_add", "add", 2, in_names, in_types, 1, out_names, out_types);

    /* Wire signature to node, node to signature output. */
    pipeline_connect_signature_in(p, "a", "n_add", "x");
    pipeline_connect_signature_in(p, "b", "n_add", "y");
    pipeline_connect_signature_out(p, "n_add", "out", "y");
    return p;
}

TEST(verify_simple_graph_passes) {
    Pipeline *p = build_add_graph();
    int rc = pipeline_verify(p);
    if (rc != 0) printf("\n    err: %s ", pipeline_last_error());
    ASSERT_EQ(rc, PIPE_OK);
    ASSERT_EQ(p->verified, 1);
    ASSERT(p->exec_order != NULL);
    pipeline_free(p);
}

TEST(verify_dangling_input_port_rejected) {
    Pipeline *p = pipeline_create("g");
    const char *sig_in_names[]  = {"a"};
    PipelineType *sig_in_types[] = {pipeline_type_int()};
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 1, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    const char *in_names[]   = {"x", "y"};   /* two inputs */
    PipelineType *in_types[] = {pipeline_type_int(), pipeline_type_int()};
    const char *out_names[]  = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    pipeline_add_node(p, "n_add", "add", 2, in_names, in_types, 1, out_names, out_types);
    /* Only connect one input — the other is dangling. */
    pipeline_connect_signature_in(p, "a", "n_add", "x");
    pipeline_connect_signature_out(p, "n_add", "out", "y");

    int rc = pipeline_verify(p);
    ASSERT_EQ(rc, PIPE_ERR_DANGLING_PORT);
    pipeline_free(p);
}

TEST(verify_type_mismatch_rejected) {
    Pipeline *p = pipeline_create("g");
    const char *sig_in_names[]  = {"a"};
    PipelineType *sig_in_types[] = {pipeline_type_string()};   /* string in */
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 1, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    const char *in_names[]   = {"x"};
    PipelineType *in_types[] = {pipeline_type_int()};   /* int port */
    const char *out_names[]  = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    pipeline_add_node(p, "n", "f", 1, in_names, in_types, 1, out_names, out_types);
    pipeline_connect_signature_in(p, "a", "n", "x");   /* string -> int :( */
    pipeline_connect_signature_out(p, "n", "out", "y");

    int rc = pipeline_verify(p);
    ASSERT_EQ(rc, PIPE_ERR_TYPE_MISMATCH);
    pipeline_free(p);
}

TEST(verify_cycle_rejected) {
    Pipeline *p = pipeline_create("g");
    const char *sig_in_names[]  = {"a"};
    PipelineType *sig_in_types[] = {pipeline_type_int()};
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 1, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    const char *in_names[]   = {"x", "y"};
    PipelineType *in_types1[] = {pipeline_type_int(), pipeline_type_int()};
    const char *out_names[]  = {"out"};
    PipelineType *out_types1[] = {pipeline_type_int()};
    pipeline_add_node(p, "n1", "f", 2, in_names, in_types1, 1, out_names, out_types1);

    PipelineType *in_types2[] = {pipeline_type_int(), pipeline_type_int()};
    PipelineType *out_types2[] = {pipeline_type_int()};
    pipeline_add_node(p, "n2", "g", 2, in_names, in_types2, 1, out_names, out_types2);

    pipeline_connect_signature_in(p, "a", "n1", "x");
    pipeline_connect(p, "n1", "out", "n2", "x");
    pipeline_connect(p, "n2", "out", "n1", "y");   /* cycle: n1 -> n2 -> n1 */
    pipeline_connect_signature_in(p, "a", "n2", "y");
    pipeline_connect_signature_out(p, "n2", "out", "y");

    int rc = pipeline_verify(p);
    ASSERT_EQ(rc, PIPE_ERR_CYCLE);
    pipeline_free(p);
}

TEST(verify_signature_output_unconnected_rejected) {
    Pipeline *p = pipeline_create("g");
    const char *sig_in_names[]  = {"a"};
    PipelineType *sig_in_types[] = {pipeline_type_int()};
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 1, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    const char *in_names[]   = {"x"};
    PipelineType *in_types[] = {pipeline_type_int()};
    const char *out_names[]  = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    pipeline_add_node(p, "n", "f", 1, in_names, in_types, 1, out_names, out_types);
    pipeline_connect_signature_in(p, "a", "n", "x");
    /* No connect_signature_out — y is dangling. */

    int rc = pipeline_verify(p);
    ASSERT_EQ(rc, PIPE_ERR_BAD_SIGNATURE);
    pipeline_free(p);
}

TEST(verify_signature_input_unused_rejected) {
    Pipeline *p = pipeline_create("g");
    const char *sig_in_names[]  = {"a", "unused"};
    PipelineType *sig_in_types[] = {pipeline_type_int(), pipeline_type_int()};
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 2, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    const char *in_names[]   = {"x"};
    PipelineType *in_types[] = {pipeline_type_int()};
    const char *out_names[]  = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    pipeline_add_node(p, "n", "f", 1, in_names, in_types, 1, out_names, out_types);
    pipeline_connect_signature_in(p, "a", "n", "x");
    /* "unused" never connected. */
    pipeline_connect_signature_out(p, "n", "out", "y");
    int rc = pipeline_verify(p);
    ASSERT_EQ(rc, PIPE_ERR_BAD_SIGNATURE);
    pipeline_free(p);
}

TEST(verify_topological_order_correct) {
    /* Build chain: a -> n1 -> n2 -> y */
    Pipeline *p = pipeline_create("chain");
    const char *sig_in_names[]  = {"a"};
    PipelineType *sig_in_types[] = {pipeline_type_int()};
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 1, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    const char *in1[]  = {"x"};   PipelineType *it1[]  = {pipeline_type_int()};
    const char *out1[] = {"out"}; PipelineType *ot1[]  = {pipeline_type_int()};
    pipeline_add_node(p, "n1", "f1", 1, in1, it1, 1, out1, ot1);

    const char *in2[]  = {"x"};   PipelineType *it2[]  = {pipeline_type_int()};
    const char *out2[] = {"out"}; PipelineType *ot2[]  = {pipeline_type_int()};
    pipeline_add_node(p, "n2", "f2", 1, in2, it2, 1, out2, ot2);

    pipeline_connect_signature_in(p, "a", "n1", "x");
    pipeline_connect(p, "n1", "out", "n2", "x");
    pipeline_connect_signature_out(p, "n2", "out", "y");
    int rc = pipeline_verify(p);
    if (rc != 0) printf("\n    err: %s ", pipeline_last_error());
    ASSERT_EQ(rc, PIPE_OK);
    ASSERT(p->exec_order[0] == 0);  /* n1 first */
    ASSERT(p->exec_order[1] == 1);  /* n2 second */
    pipeline_free(p);
}

/* ============================================================
 *  Execution
 * ============================================================ */

/* Test dispatcher: handles a few primitives for executor tests. */
static int test_dispatch(const char *primitive,
                         const PipelineConfig *config, int n_config,
                         const PipelineValue *inputs, int n_inputs,
                         PipelineValue *outputs, int n_outputs,
                         void *user_data) {
    (void)config; (void)n_config; (void)user_data;
    if (strcmp(primitive, "add") == 0) {
        if (n_inputs != 2 || n_outputs != 1) return -1;
        outputs[0].v.i = inputs[0].v.i + inputs[1].v.i;
        return 0;
    }
    if (strcmp(primitive, "mul") == 0) {
        if (n_inputs != 2 || n_outputs != 1) return -1;
        outputs[0].v.i = inputs[0].v.i * inputs[1].v.i;
        return 0;
    }
    if (strcmp(primitive, "neg") == 0) {
        if (n_inputs != 1 || n_outputs != 1) return -1;
        outputs[0].v.i = -inputs[0].v.i;
        return 0;
    }
    return -1;
}

TEST(execute_simple_add) {
    Pipeline *p = build_add_graph();
    ASSERT_EQ(pipeline_verify(p), PIPE_OK);

    PipelineValue inputs[2]  = {0};
    PipelineValue outputs[1] = {0};
    inputs[0].v.i = 3;
    inputs[1].v.i = 4;
    int rc = pipeline_execute(p, inputs, outputs, test_dispatch, NULL);
    if (rc != 0) printf("\n    err: %s ", pipeline_last_error());
    ASSERT_EQ(rc, PIPE_OK);
    ASSERT_EQ((int)outputs[0].v.i, 7);
    pipeline_free(p);
}

TEST(execute_chain_three_nodes) {
    /* (a + b) * c, then negated.
     * Signature: in a, in b, in c → out y
     * n1: add(a, b)  n2: mul(n1, c)  n3: neg(n2)  → y
     */
    Pipeline *p = pipeline_create("chain3");
    const char *sin_names[]  = {"a", "b", "c"};
    PipelineType *sin_types[] = {pipeline_type_int(), pipeline_type_int(), pipeline_type_int()};
    const char *sout_names[] = {"y"};
    PipelineType *sout_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 3, sin_names, sin_types, 1, sout_names, sout_types);

    const char *in2[]  = {"x", "y"}; const char *out1[] = {"out"};
    PipelineType *it_a[] = {pipeline_type_int(), pipeline_type_int()};
    PipelineType *ot_a[] = {pipeline_type_int()};
    pipeline_add_node(p, "n1", "add", 2, in2, it_a, 1, out1, ot_a);

    PipelineType *it_b[] = {pipeline_type_int(), pipeline_type_int()};
    PipelineType *ot_b[] = {pipeline_type_int()};
    pipeline_add_node(p, "n2", "mul", 2, in2, it_b, 1, out1, ot_b);

    const char *in1[] = {"x"};
    PipelineType *it_c[] = {pipeline_type_int()};
    PipelineType *ot_c[] = {pipeline_type_int()};
    pipeline_add_node(p, "n3", "neg", 1, in1, it_c, 1, out1, ot_c);

    pipeline_connect_signature_in(p, "a", "n1", "x");
    pipeline_connect_signature_in(p, "b", "n1", "y");
    pipeline_connect(p, "n1", "out", "n2", "x");
    pipeline_connect_signature_in(p, "c", "n2", "y");
    pipeline_connect(p, "n2", "out", "n3", "x");
    pipeline_connect_signature_out(p, "n3", "out", "y");

    int rc = pipeline_verify(p);
    if (rc != 0) printf("\n    err: %s ", pipeline_last_error());
    ASSERT_EQ(rc, PIPE_OK);

    PipelineValue inputs[3] = {0}, outputs[1] = {0};
    inputs[0].v.i = 2; inputs[1].v.i = 3; inputs[2].v.i = 5;
    rc = pipeline_execute(p, inputs, outputs, test_dispatch, NULL);
    ASSERT_EQ(rc, PIPE_OK);
    /* (2 + 3) * 5 = 25, neg = -25 */
    ASSERT_EQ((int)outputs[0].v.i, -25);
    pipeline_free(p);
}

/* ============================================================
 *  Text round-trip
 * ============================================================ */

TEST(text_render_basic_does_not_crash) {
    Pipeline *p = build_add_graph();
    pipeline_verify(p);
    char *txt = pipeline_render_text(p);
    ASSERT(txt != NULL);
    ASSERT(strstr(txt, "@graph") != NULL);
    ASSERT(strstr(txt, "@end") != NULL);
    ASSERT(strstr(txt, "n_add") != NULL);
    free(txt);
    pipeline_free(p);
}

TEST(text_round_trip_structural) {
    /* Render → parse → verify the parsed graph has the same structure. */
    Pipeline *p = build_add_graph();
    pipeline_verify(p);
    char *txt = pipeline_render_text(p);
    ASSERT(txt != NULL);

    Pipeline *p2 = pipeline_parse_text(txt);
    if (!p2) printf("\n    parse err: %s ", pipeline_last_error());
    ASSERT(p2 != NULL);
    ASSERT_EQ(p2->n_nodes, p->n_nodes);
    ASSERT_EQ(p2->n_sig_in, p->n_sig_in);
    ASSERT_EQ(p2->n_sig_out, p->n_sig_out);
    ASSERT_STREQ(p2->nodes[0]->id, "n_add");
    ASSERT_STREQ(p2->nodes[0]->primitive, "add");

    free(txt);
    pipeline_free(p);
    pipeline_free(p2);
}

/* ============================================================
 *  DOT renderer
 * ============================================================ */

TEST(dot_render_smoke) {
    Pipeline *p = build_add_graph();
    pipeline_verify(p);
    char *dot = pipeline_render_dot(p);
    ASSERT(dot != NULL);
    ASSERT(strstr(dot, "digraph") != NULL);
    ASSERT(strstr(dot, "node_0") != NULL);
    ASSERT(strstr(dot, "sig_in_0") != NULL);
    ASSERT(strstr(dot, "sig_out_0") != NULL);
    ASSERT(strstr(dot, "->") != NULL);
    free(dot);
    pipeline_free(p);
}

/* ============================================================
 *  Phase 2 — Typed round-trip
 * ============================================================ */

TEST(text_round_trip_preserves_int_types) {
    Pipeline *p = build_add_graph();
    pipeline_verify(p);
    char *txt = pipeline_render_text(p);
    ASSERT(txt != NULL);
    ASSERT(strstr(txt, "::") != NULL);  /* annotation present */
    ASSERT(strstr(txt, "x:int") != NULL || strstr(txt, "y:int") != NULL);

    Pipeline *p2 = pipeline_parse_text(txt);
    if (!p2) printf("\n    parse err: %s ", pipeline_last_error());
    ASSERT(p2 != NULL);
    /* Verify port types are int, not ANY. */
    ASSERT_EQ(p2->nodes[0]->n_inputs, 2);
    ASSERT(p2->nodes[0]->inputs[0].type != NULL);
    ASSERT_EQ(p2->nodes[0]->inputs[0].type->kind, PIPE_T_INT);
    ASSERT_EQ(p2->nodes[0]->inputs[1].type->kind, PIPE_T_INT);
    ASSERT_EQ(p2->nodes[0]->outputs[0].type->kind, PIPE_T_INT);

    /* Reparsed graph also verifies. */
    int rc = pipeline_verify(p2);
    if (rc != 0) printf("\n    re-verify err: %s ", pipeline_last_error());
    ASSERT_EQ(rc, PIPE_OK);

    free(txt);
    pipeline_free(p);
    pipeline_free(p2);
}

TEST(text_round_trip_preserves_complex_types) {
    /* Build a graph with float + tensor types so the type annotations
     * exercise multiple code paths. */
    Pipeline *p = pipeline_create("typed_chain");
    int dims[] = {-1};   /* tensor[float, *] */
    const char *sin_names[]  = {"signal"};
    PipelineType *sin_types[] = {pipeline_type_tensor(pipeline_type_float(), 1, dims)};
    const char *sout_names[] = {"result"};
    PipelineType *sout_types[] = {pipeline_type_tensor(pipeline_type_float(), 1, dims)};
    pipeline_set_signature(p, 1, sin_names, sin_types, 1, sout_names, sout_types);

    int dims2[] = {-1};
    const char *in_names[]  = {"x"};
    PipelineType *in_types[] = {pipeline_type_tensor(pipeline_type_float(), 1, dims2)};
    int dims3[] = {-1};
    const char *out_names[] = {"out"};
    PipelineType *out_types[] = {pipeline_type_tensor(pipeline_type_float(), 1, dims3)};
    pipeline_add_node(p, "norm", "normalize", 1, in_names, in_types, 1, out_names, out_types);

    pipeline_connect_signature_in(p, "signal", "norm", "x");
    pipeline_connect_signature_out(p, "norm", "out", "result");
    ASSERT_EQ(pipeline_verify(p), PIPE_OK);

    char *txt = pipeline_render_text(p);
    ASSERT(txt != NULL);
    ASSERT(strstr(txt, "tensor[float") != NULL);

    Pipeline *p2 = pipeline_parse_text(txt);
    ASSERT(p2 != NULL);
    /* Reparsed graph verifies — proves type fidelity is good enough
     * to pass the type-equality check on connected edges. */
    int rc = pipeline_verify(p2);
    if (rc != 0) printf("\n    re-verify err: %s ", pipeline_last_error());
    ASSERT_EQ(rc, PIPE_OK);

    free(txt);
    pipeline_free(p);
    pipeline_free(p2);
}

/* ============================================================
 *  Phase 2 — Partial verification
 * ============================================================ */

TEST(verify_partial_accepts_dangling_input_port) {
    /* Same scenario as verify_dangling_input_port_rejected — but
     * pipeline_verify_partial() should accept it with missing > 0. */
    Pipeline *p = pipeline_create("g");
    const char *sig_in_names[]  = {"a"};
    PipelineType *sig_in_types[] = {pipeline_type_int()};
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 1, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    const char *in_names[]   = {"x", "y"};
    PipelineType *in_types[] = {pipeline_type_int(), pipeline_type_int()};
    const char *out_names[]  = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    pipeline_add_node(p, "n_add", "add", 2, in_names, in_types, 1, out_names, out_types);
    pipeline_connect_signature_in(p, "a", "n_add", "x");
    pipeline_connect_signature_out(p, "n_add", "out", "y");
    /* "y" input is dangling. */

    int missing = 0;
    int rc = pipeline_verify_partial(p, &missing);
    ASSERT_EQ(rc, PIPE_OK);
    ASSERT_EQ(missing, 1);                   /* the dangling y port */
    ASSERT_EQ(p->verified, 0);               /* not safe to execute */
    pipeline_free(p);
}

TEST(verify_partial_still_rejects_type_mismatch) {
    /* Type mismatch is a hard error even in partial mode. */
    Pipeline *p = pipeline_create("g");
    const char *sig_in_names[]  = {"a"};
    PipelineType *sig_in_types[] = {pipeline_type_string()};
    const char *sig_out_names[] = {"y"};
    PipelineType *sig_out_types[] = {pipeline_type_int()};
    pipeline_set_signature(p, 1, sig_in_names, sig_in_types, 1, sig_out_names, sig_out_types);

    const char *in_names[]   = {"x"};
    PipelineType *in_types[] = {pipeline_type_int()};
    const char *out_names[]  = {"out"};
    PipelineType *out_types[] = {pipeline_type_int()};
    pipeline_add_node(p, "n", "f", 1, in_names, in_types, 1, out_names, out_types);
    pipeline_connect_signature_in(p, "a", "n", "x");
    pipeline_connect_signature_out(p, "n", "out", "y");

    int missing = 0;
    int rc = pipeline_verify_partial(p, &missing);
    ASSERT_EQ(rc, PIPE_ERR_TYPE_MISMATCH);
    pipeline_free(p);
}

TEST(verify_partial_complete_graph_zero_missing) {
    Pipeline *p = build_add_graph();
    int missing = 999;
    int rc = pipeline_verify_partial(p, &missing);
    ASSERT_EQ(rc, PIPE_OK);
    ASSERT_EQ(missing, 0);
    /* But the verified flag is NOT set by partial — strict verify is required. */
    ASSERT_EQ(p->verified, 0);
    rc = pipeline_verify(p);
    ASSERT_EQ(rc, PIPE_OK);
    ASSERT_EQ(p->verified, 1);
    pipeline_free(p);
}

/* ============================================================
 *  Phase 2 — VM dispatch (deferred — verify error path)
 * ============================================================ */

TEST(execute_vm_returns_deferred_error) {
    /* Phase 2 ships pipeline_execute_vm() as an API surface; the actual
     * dispatch is deferred to Phase 3 because vm_engine doesn't expose
     * a public lookup-and-call for registered native fns. The function
     * must return PIPE_ERR_EXEC with a clear message that explains what
     * to do instead. */
    Pipeline *p = build_add_graph();
    pipeline_verify(p);
    /* Pass a non-NULL but bogus vm_engine pointer — Phase 2 stub doesn't
     * dereference it, just checks the args. */
    PipelineValue inputs[2] = {0};
    PipelineValue outputs[1] = {0};
    int rc = pipeline_execute_vm(p, (vm_engine *)0xdeadbeef, inputs, outputs);
    ASSERT_EQ(rc, PIPE_ERR_EXEC);
    const char *msg = pipeline_last_error();
    ASSERT(msg != NULL);
    ASSERT(strstr(msg, "Phase 3") != NULL);
    pipeline_free(p);
}

TEST(execute_vm_null_args_rejected) {
    int rc = pipeline_execute_vm(NULL, NULL, NULL, NULL);
    ASSERT_EQ(rc, PIPE_ERR_EXEC);
    Pipeline *p = build_add_graph();
    pipeline_verify(p);
    rc = pipeline_execute_vm(p, NULL, NULL, NULL);
    ASSERT_EQ(rc, PIPE_ERR_EXEC);
    const char *msg = pipeline_last_error();
    ASSERT(strstr(msg, "vm_engine") != NULL || strstr(msg, "null") != NULL);
    pipeline_free(p);
}

/* ============================================================
 *  Phase 3a — Corpus integrity
 *
 * Each test builds a graph that mirrors a hand-curated example from
 * tools/pipeline_corpus_gen.c, then asserts the full corpus round-trip:
 *   build → verify → render → parse → strict-verify → re-render
 *   → byte-equal first render.
 *
 * Catches regressions in any of: builder API, verifier, renderer,
 * parser, type round-trip, or topological sort determinism.
 * ============================================================ */

/* Helper: assert the full render→parse→re-render→byte-equal cycle. */
static int corpus_round_trip_check(Pipeline *p, char **first_render_out) {
    if (!p) return 0;
    if (pipeline_verify(p) != PIPE_OK) {
        printf("\n    verify err: %s ", pipeline_last_error());
        return 0;
    }
    char *r1 = pipeline_render_text(p);
    if (!r1) return 0;
    Pipeline *p2 = pipeline_parse_text(r1);
    if (!p2) { printf("\n    parse err: %s ", pipeline_last_error()); free(r1); return 0; }
    if (pipeline_verify(p2) != PIPE_OK) {
        printf("\n    re-verify err: %s ", pipeline_last_error());
        free(r1); pipeline_free(p2); return 0;
    }
    char *r2 = pipeline_render_text(p2);
    if (!r2) { free(r1); pipeline_free(p2); return 0; }
    int eq = strcmp(r1, r2) == 0;
    if (!eq) {
        printf("\n    --- first render ---\n%s", r1);
        printf("    --- second render ---\n%s", r2);
    }
    if (first_render_out) *first_render_out = r1; else free(r1);
    free(r2);
    pipeline_free(p2);
    return eq;
}

/* Helper: 1-input 1-output int signature. */
static void csig_int(Pipeline *p, const char *in_name, const char *out_name) {
    const char *in_names[]  = { in_name };
    PipelineType *in_types[] = { pipeline_type_int() };
    const char *out_names[] = { out_name };
    PipelineType *out_types[] = { pipeline_type_int() };
    pipeline_set_signature(p, 1, in_names, in_types, 1, out_names, out_types);
}
static void csig_int2(Pipeline *p, const char *a, const char *b, const char *o) {
    const char *in_names[]  = { a, b };
    PipelineType *in_types[] = { pipeline_type_int(), pipeline_type_int() };
    const char *out_names[] = { o };
    PipelineType *out_types[] = { pipeline_type_int() };
    pipeline_set_signature(p, 2, in_names, in_types, 1, out_names, out_types);
}
static void cnode_2in(Pipeline *p, const char *id, const char *prim) {
    const char *in_names[]  = { "x", "y" };
    PipelineType *in_types[] = { pipeline_type_int(), pipeline_type_int() };
    const char *out_names[] = { "out" };
    PipelineType *out_types[] = { pipeline_type_int() };
    pipeline_add_node(p, id, prim, 2, in_names, in_types, 1, out_names, out_types);
}
static void cnode_1in(Pipeline *p, const char *id, const char *prim) {
    const char *in_names[]  = { "x" };
    PipelineType *in_types[] = { pipeline_type_int() };
    const char *out_names[] = { "out" };
    PipelineType *out_types[] = { pipeline_type_int() };
    pipeline_add_node(p, id, prim, 1, in_names, in_types, 1, out_names, out_types);
}

TEST(corpus_ex_add) {
    Pipeline *p = pipeline_create("ex_add");
    csig_int2(p, "a", "b", "y");
    cnode_2in(p, "n", "add");
    pipeline_connect_signature_in(p, "a", "n", "x");
    pipeline_connect_signature_in(p, "b", "n", "y");
    pipeline_connect_signature_out(p, "n", "out", "y");
    char *txt = NULL;
    int ok = corpus_round_trip_check(p, &txt);
    ASSERT(ok);
    ASSERT(strstr(txt, "@graph ex_add") != NULL);
    ASSERT(strstr(txt, "add(x: <a>, y: <b>)") != NULL);
    free(txt);
    pipeline_free(p);
}

TEST(corpus_ex_negate_chain) {
    /* add_then_negate(a, b) — 2-node chain, exercises node-to-node edge. */
    Pipeline *p = pipeline_create("ex_chain");
    csig_int2(p, "a", "b", "y");
    cnode_2in(p, "sum", "add");
    cnode_1in(p, "neg", "negate");
    pipeline_connect_signature_in(p, "a", "sum", "x");
    pipeline_connect_signature_in(p, "b", "sum", "y");
    pipeline_connect(p, "sum", "out", "neg", "x");
    pipeline_connect_signature_out(p, "neg", "out", "y");
    char *txt = NULL;
    int ok = corpus_round_trip_check(p, &txt);
    ASSERT(ok);
    ASSERT(strstr(txt, "negate(x: sum.out)") != NULL);
    free(txt);
    pipeline_free(p);
}

TEST(corpus_ex_multi_node_tree) {
    /* square_sum(a, b) — 3 nodes, two parallel siblings then a join.
     * Tests deterministic topo order across rebuilds. */
    Pipeline *p = pipeline_create("ex_tree");
    csig_int2(p, "a", "b", "y");
    cnode_2in(p, "sq_a", "multiply");
    cnode_2in(p, "sq_b", "multiply");
    cnode_2in(p, "sum",  "add");
    pipeline_connect_signature_in(p, "a", "sq_a", "x");
    pipeline_connect_signature_in(p, "a", "sq_a", "y");
    pipeline_connect_signature_in(p, "b", "sq_b", "x");
    pipeline_connect_signature_in(p, "b", "sq_b", "y");
    pipeline_connect(p, "sq_a", "out", "sum", "x");
    pipeline_connect(p, "sq_b", "out", "sum", "y");
    pipeline_connect_signature_out(p, "sum", "out", "y");
    char *txt = NULL;
    int ok = corpus_round_trip_check(p, &txt);
    ASSERT(ok);
    /* Topological order must place both squares before the sum. */
    const char *p_sq_a = strstr(txt, "sq_a");
    const char *p_sq_b = strstr(txt, "sq_b");
    const char *p_sum  = strstr(txt, "| sum =");
    ASSERT(p_sq_a && p_sq_b && p_sum);
    ASSERT(p_sum > p_sq_a && p_sum > p_sq_b);
    free(txt);
    pipeline_free(p);
}

TEST(corpus_ex_5_nodes) {
    /* distance_squared(a1, a2, b1, b2) — 5 nodes, deeper graph. */
    Pipeline *p = pipeline_create("ex_dist2");
    const char *in_names[]  = { "a1", "a2", "b1", "b2" };
    PipelineType *in_types[] = { pipeline_type_int(), pipeline_type_int(),
                                 pipeline_type_int(), pipeline_type_int() };
    const char *out_names[] = { "y" };
    PipelineType *out_types[] = { pipeline_type_int() };
    pipeline_set_signature(p, 4, in_names, in_types, 1, out_names, out_types);
    cnode_2in(p, "dx",  "subtract");
    cnode_2in(p, "dy",  "subtract");
    cnode_2in(p, "dx2", "multiply");
    cnode_2in(p, "dy2", "multiply");
    cnode_2in(p, "sum", "add");
    pipeline_connect_signature_in(p, "a1", "dx", "x");
    pipeline_connect_signature_in(p, "b1", "dx", "y");
    pipeline_connect_signature_in(p, "a2", "dy", "x");
    pipeline_connect_signature_in(p, "b2", "dy", "y");
    pipeline_connect(p, "dx", "out", "dx2", "x");
    pipeline_connect(p, "dx", "out", "dx2", "y");
    pipeline_connect(p, "dy", "out", "dy2", "x");
    pipeline_connect(p, "dy", "out", "dy2", "y");
    pipeline_connect(p, "dx2", "out", "sum", "x");
    pipeline_connect(p, "dy2", "out", "sum", "y");
    pipeline_connect_signature_out(p, "sum", "out", "y");
    char *txt = NULL;
    int ok = corpus_round_trip_check(p, &txt);
    ASSERT(ok);
    ASSERT(strstr(txt, "ex_dist2") != NULL);
    free(txt);
    pipeline_free(p);
}

TEST(corpus_round_trip_byte_equal_iterated) {
    /* Build → render → parse → render → parse → render. The third
     * render must equal the second (and the second must equal the
     * first). Iterates the round-trip to surface cumulative drift. */
    Pipeline *p = pipeline_create("ex_iter");
    csig_int2(p, "a", "b", "y");
    cnode_2in(p, "sum", "add");
    cnode_1in(p, "neg", "negate");
    pipeline_connect_signature_in(p, "a", "sum", "x");
    pipeline_connect_signature_in(p, "b", "sum", "y");
    pipeline_connect(p, "sum", "out", "neg", "x");
    pipeline_connect_signature_out(p, "neg", "out", "y");
    ASSERT_EQ(pipeline_verify(p), PIPE_OK);

    char *r1 = pipeline_render_text(p);
    Pipeline *p2 = pipeline_parse_text(r1);
    ASSERT(p2 != NULL);
    ASSERT_EQ(pipeline_verify(p2), PIPE_OK);
    char *r2 = pipeline_render_text(p2);
    Pipeline *p3 = pipeline_parse_text(r2);
    ASSERT(p3 != NULL);
    ASSERT_EQ(pipeline_verify(p3), PIPE_OK);
    char *r3 = pipeline_render_text(p3);

    ASSERT(strcmp(r1, r2) == 0);
    ASSERT(strcmp(r2, r3) == 0);

    free(r1); free(r2); free(r3);
    pipeline_free(p); pipeline_free(p2); pipeline_free(p3);
}

/* ============================================================
 *  Phase 3d — Parser robustness (fuzz)
 *
 * The Phase 3c demo surfaced a parser segfault when the model emitted
 * structurally-plausible but malformed text. Phase 3d hardens the
 * parser against arbitrary token streams. These fuzz tests exercise
 * pathological inputs and assert the parser never crashes — either
 * returns NULL or returns a Pipeline that frees cleanly.
 * ============================================================ */

/* Tiny xorshift PRNG (deterministic across runs given the same seed). */
static uint32_t fuzz_state = 0xC0FFEE42u;
static uint32_t fuzz_rand(void) {
    uint32_t x = fuzz_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (fuzz_state = x);
}

TEST(parser_fuzz_empty_string) {
    Pipeline *p = pipeline_parse_text("");
    /* Either NULL or a valid empty graph. Either is fine — must not crash. */
    if (p) pipeline_free(p);
}

TEST(parser_fuzz_garbage) {
    const char *garbage[] = {
        "not a graph",
        "@graph",
        "@graph foo",
        "@graph foo @end",
        "@graph foo : in",
        "@graph foo : in x ->",
        "@graph foo : in x -> int : out y -> int |",
        "@graph foo : in x -> int : out y -> int | n =",
        "@graph foo : in x -> int : out y -> int | n = add(",
        "@graph foo : in x -> int : out y -> int | n = add(x:",
        "@graph foo : in x -> int : out y -> int | n = add(x: <",
        "@graph foo : in x -> int : out y -> int | n = add(x: <a",
        "@graph foo : in x -> int : out y -> int | n = add(x: <a>) y <-",
        "@graph foo : in x -> int : out y -> int | n = add(x: <a>) y <- n",
        "@graph foo : in x -> int : out y -> int | n = add(x: <a>) y <- n.",
        /* Pathologically truncated mid-token: */
        "@graph foo : in x -> int : out y -> int | n = add(x: <a>, y: <a>) :: x:int, y:int -> out:int\ny <- n",
    };
    for (size_t i = 0; i < sizeof(garbage) / sizeof(garbage[0]); i++) {
        Pipeline *p = pipeline_parse_text(garbage[i]);
        if (p) pipeline_free(p);
    }
}

TEST(parser_fuzz_random_truncation) {
    /* Build a known-good graph, render it, then fuzz: try every prefix
     * of the rendered text and assert no crash. */
    Pipeline *src = build_add_graph();
    pipeline_verify(src);
    char *txt = pipeline_render_text(src);
    ASSERT(txt != NULL);
    size_t txt_len = strlen(txt);
    char *prefix = (char *)malloc(txt_len + 1);
    for (size_t cut = 0; cut <= txt_len; cut++) {
        memcpy(prefix, txt, cut);
        prefix[cut] = '\0';
        Pipeline *p = pipeline_parse_text(prefix);
        if (p) pipeline_free(p);
    }
    free(prefix);
    free(txt);
    pipeline_free(src);
}

TEST(parser_fuzz_random_byte_mutation) {
    /* Build a known-good graph, render, then mutate single random bytes
     * and assert no crash on parse. 200 iterations, deterministic seed. */
    Pipeline *src = build_add_graph();
    pipeline_verify(src);
    char *txt = pipeline_render_text(src);
    size_t txt_len = strlen(txt);
    char *mutated = (char *)malloc(txt_len + 1);
    fuzz_state = 0xC0FFEE42u;
    for (int iter = 0; iter < 200; iter++) {
        memcpy(mutated, txt, txt_len + 1);
        size_t pos = fuzz_rand() % txt_len;
        char new_ch = (char)(32 + (fuzz_rand() % 95));   /* printable ASCII */
        mutated[pos] = new_ch;
        Pipeline *p = pipeline_parse_text(mutated);
        if (p) pipeline_free(p);
    }
    free(mutated);
    free(txt);
    pipeline_free(src);
}

TEST(parser_fuzz_random_bytes) {
    /* Pure random bytes — must not crash. */
    char buf[256];
    fuzz_state = 0xDEADBEEFu;
    for (int iter = 0; iter < 100; iter++) {
        size_t len = (size_t)(fuzz_rand() % (sizeof(buf) - 1));
        for (size_t i = 0; i < len; i++) {
            buf[i] = (char)(32 + (fuzz_rand() % 95));   /* printable */
        }
        buf[len] = '\0';
        Pipeline *p = pipeline_parse_text(buf);
        if (p) pipeline_free(p);
    }
}

/* The original Phase 3c-crashing input, lightly redacted.
 * If this passes without crashing, Phase 3d's primary goal is met. */
TEST(parser_fuzz_phase3c_crash_input) {
    const char *malformed =
        "@graph distance_squared_4d\n"
        ": in a1 -> int\n"
        ": in b1 -> int\n"
        ": out y -> int\n"
        "| d1 = subtract(x: <a1>, y: <b1>) :: x:int, y:int -> out:int\n"
        "| sq2 = multiply(x: d2.out, y: d2.out) :: x:int, y:int\n"   /* mid-line cut */
        "  ---\n";
    Pipeline *p = pipeline_parse_text(malformed);
    if (p) pipeline_free(p);
}

/* ============================================================
 *  Last-error reporting
 * ============================================================ */

TEST(last_error_set_on_failure) {
    Pipeline *p = pipeline_create("g");
    int rc = pipeline_connect(p, "missing", "out", "missing2", "in");
    ASSERT_EQ(rc, PIPE_ERR_UNKNOWN_NODE);
    const char *msg = pipeline_last_error();
    ASSERT(msg != NULL);
    ASSERT(strstr(msg, "missing") != NULL);
    pipeline_free(p);
}

/* ---- Main ---- */

int main(void) {
    printf("[Pipeline IR — Type system]\n");
    RUN(type_constructors_basic);
    RUN(type_equal_basic);
    RUN(type_any_matches_anything);
    RUN(type_list_recursive);
    RUN(type_tensor_dims);
    RUN(type_clone_deep_copy);
    RUN(type_format_pretty_print);

    printf("\n[Pipeline IR — Construction]\n");
    RUN(pipeline_create_and_free);
    RUN(pipeline_add_node_basic);
    RUN(pipeline_add_node_duplicate_id_rejected);
    RUN(pipeline_connect_unknown_node_rejected);

    printf("\n[Pipeline IR — Verification]\n");
    RUN(verify_simple_graph_passes);
    RUN(verify_dangling_input_port_rejected);
    RUN(verify_type_mismatch_rejected);
    RUN(verify_cycle_rejected);
    RUN(verify_signature_output_unconnected_rejected);
    RUN(verify_signature_input_unused_rejected);
    RUN(verify_topological_order_correct);

    printf("\n[Pipeline IR — Execution]\n");
    RUN(execute_simple_add);
    RUN(execute_chain_three_nodes);

    printf("\n[Pipeline IR — Text round-trip]\n");
    RUN(text_render_basic_does_not_crash);
    RUN(text_round_trip_structural);

    printf("\n[Pipeline IR — DOT renderer]\n");
    RUN(dot_render_smoke);

    printf("\n[Pipeline IR — Phase 2 typed round-trip]\n");
    RUN(text_round_trip_preserves_int_types);
    RUN(text_round_trip_preserves_complex_types);

    printf("\n[Pipeline IR — Phase 2 partial verify]\n");
    RUN(verify_partial_accepts_dangling_input_port);
    RUN(verify_partial_still_rejects_type_mismatch);
    RUN(verify_partial_complete_graph_zero_missing);

    printf("\n[Pipeline IR — Phase 2 VM dispatch (deferred)]\n");
    RUN(execute_vm_returns_deferred_error);
    RUN(execute_vm_null_args_rejected);

    printf("\n[Pipeline IR — Phase 3a corpus integrity]\n");
    RUN(corpus_ex_add);
    RUN(corpus_ex_negate_chain);
    RUN(corpus_ex_multi_node_tree);
    RUN(corpus_ex_5_nodes);
    RUN(corpus_round_trip_byte_equal_iterated);

    printf("\n[Pipeline IR — Phase 3d parser fuzz]\n");
    RUN(parser_fuzz_empty_string);
    RUN(parser_fuzz_garbage);
    RUN(parser_fuzz_random_truncation);
    RUN(parser_fuzz_random_byte_mutation);
    RUN(parser_fuzz_random_bytes);
    RUN(parser_fuzz_phase3c_crash_input);

    printf("\n[Pipeline IR — Error reporting]\n");
    RUN(last_error_set_on_failure);

    printf("\n=== Results: %d/%d passed ===\n", g_tests_passed, g_tests_run);
    return g_tests_failed == 0 ? 0 : 1;
}
