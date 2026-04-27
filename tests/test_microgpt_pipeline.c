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

#define TEST(name)                                                             \
  static void test_##name(void);                                               \
  static void run_##name(void) {                                               \
    g_tests_run++;                                                             \
    printf("  %-60s ", #name);                                                 \
    fflush(stdout);                                                            \
    test_##name();                                                             \
    printf("PASS\n");                                                          \
    fflush(stdout);                                                            \
    g_tests_passed++;                                                          \
  }                                                                            \
  static void test_##name(void)

#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL\n    %s:%d  Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
      g_tests_failed++;                                                        \
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

    printf("\n[Pipeline IR — Error reporting]\n");
    RUN(last_error_set_on_failure);

    printf("\n=== Results: %d/%d passed ===\n", g_tests_passed, g_tests_run);
    return g_tests_failed == 0 ? 0 : 1;
}
