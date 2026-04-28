/*
 * MicroGPT-C — Pipeline IR Corpus Generator (Phase 3b: templated)
 *
 * Builds ~115 (prompt, graph) examples programmatically via the
 * Pipeline IR API across 10 parametric template families plus the
 * original hand-curated set. All examples verify; round-trip is
 * byte-stable thanks to the canonical Kahn topo sort.
 *
 * Usage:
 *   ./pipeline_corpus_gen [output_file]            # train+val combined
 *   ./pipeline_corpus_gen train.txt val.txt        # 90/10 split
 *
 * In split mode, every Nth example (N=10) goes to val.txt; the rest
 * go to train.txt. Deterministic split.
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ============================================================
 *  Builder helpers
 * ============================================================ */

static PipelineType *T_int(void) { return pipeline_type_int(); }

/* Set a 1-output, n-input int signature with names from the supplied array. */
static void sig_n_in_1_out(Pipeline *p, int n_in, const char **in_names, const char *out_name) {
    PipelineType **in_types = (PipelineType **)calloc((size_t)n_in, sizeof(PipelineType *));
    for (int i = 0; i < n_in; i++) in_types[i] = T_int();
    const char *out_names[] = { out_name };
    PipelineType *out_types[] = { T_int() };
    pipeline_set_signature(p, n_in, in_names, in_types, 1, out_names, out_types);
    free(in_types);
}

/* Add a binary node ("x", "y" → "out"), all int. */
static void node_2in(Pipeline *p, const char *id, const char *prim) {
    const char *in_names[]  = { "x", "y" };
    PipelineType *in_types[] = { T_int(), T_int() };
    const char *out_names[] = { "out" };
    PipelineType *out_types[] = { T_int() };
    pipeline_add_node(p, id, prim, 2, in_names, in_types, 1, out_names, out_types);
}

/* Add a unary node ("x" → "out"), all int. */
static void node_1in(Pipeline *p, const char *id, const char *prim) {
    const char *in_names[]  = { "x" };
    PipelineType *in_types[] = { T_int() };
    const char *out_names[] = { "out" };
    PipelineType *out_types[] = { T_int() };
    pipeline_add_node(p, id, prim, 1, in_names, in_types, 1, out_names, out_types);
}

/* Add an N-input node with caller-chosen port names; single "out" port, all int.
 * Used for domain primitives like clamp(x, lo, hi), compound(principal, rate, periods). */
static void node_named(Pipeline *p, const char *id, const char *prim,
                       int n_in, const char **in_names) {
    PipelineType **in_types = (PipelineType **)calloc((size_t)n_in, sizeof(PipelineType *));
    for (int i = 0; i < n_in; i++) in_types[i] = T_int();
    const char *out_names[] = { "out" };
    PipelineType *out_types[] = { T_int() };
    pipeline_add_node(p, id, prim, n_in, in_names, in_types, 1, out_names, out_types);
    free(in_types);
}

/* ============================================================
 *  Template family 1: chain(prim, n)
 *    y = x_1 prim x_2 prim ... prim x_n   (left-folded binary chain)
 * ============================================================ */
static Pipeline *tpl_chain(const char *prim, int n) {
    char name[64]; snprintf(name, sizeof(name), "chain_%s_%d", prim, n);
    Pipeline *p = pipeline_create(name);
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < n; i++) {
        snprintf(in_buf[i], sizeof(in_buf[i]), "x%d", i + 1);
        in_names[i] = in_buf[i];
    }
    sig_n_in_1_out(p, n, in_names, "y");
    /* Build n-1 binary nodes folding left. */
    char node_id[16];
    for (int i = 1; i < n; i++) {
        snprintf(node_id, sizeof(node_id), "n%d", i);
        node_2in(p, node_id, prim);
        if (i == 1) {
            pipeline_connect_signature_in(p, in_names[0], node_id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "n%d", i - 1);
            pipeline_connect(p, prev, "out", node_id, "x");
        }
        pipeline_connect_signature_in(p, in_names[i], node_id, "y");
    }
    char last[16]; snprintf(last, sizeof(last), "n%d", n - 1);
    pipeline_connect_signature_out(p, last, "out", "y");
    return p;
}

/* ============================================================
 *  Template family 2: fanout_combine(unary, binary, n)
 *    y = unary(x_1) binary unary(x_2) binary ... binary unary(x_n)
 * ============================================================ */
static Pipeline *tpl_fanout_combine(const char *unary, const char *binary, int n) {
    char name[80]; snprintf(name, sizeof(name), "fanout_%s_%s_%d", unary, binary, n);
    Pipeline *p = pipeline_create(name);
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < n; i++) {
        snprintf(in_buf[i], sizeof(in_buf[i]), "x%d", i + 1);
        in_names[i] = in_buf[i];
    }
    sig_n_in_1_out(p, n, in_names, "y");
    /* Per-input unary nodes. */
    char unary_id[16];
    for (int i = 0; i < n; i++) {
        snprintf(unary_id, sizeof(unary_id), "u%d", i + 1);
        node_1in(p, unary_id, unary);
        pipeline_connect_signature_in(p, in_names[i], unary_id, "x");
    }
    /* Left-fold binary chain over the unary outputs. */
    for (int i = 1; i < n; i++) {
        char node_id[16]; snprintf(node_id, sizeof(node_id), "b%d", i);
        node_2in(p, node_id, binary);
        if (i == 1) {
            pipeline_connect(p, "u1", "out", node_id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "b%d", i - 1);
            pipeline_connect(p, prev, "out", node_id, "x");
        }
        char src[16]; snprintf(src, sizeof(src), "u%d", i + 1);
        pipeline_connect(p, src, "out", node_id, "y");
    }
    if (n == 1) {
        /* trivial: no binary; just route u1 through. */
        pipeline_connect_signature_out(p, "u1", "out", "y");
    } else {
        char last[16]; snprintf(last, sizeof(last), "b%d", n - 1);
        pipeline_connect_signature_out(p, last, "out", "y");
    }
    return p;
}

/* ============================================================
 *  Template family 3: polynomial(degree d)
 *    y = a_0 + a_1*x + a_2*x^2 + ... + a_d*x^d
 *  Inputs: a_0..a_d, x.    Total: d+2 inputs.
 *  Build x^k iteratively, multiply by a_k, sum all.
 * ============================================================ */
static Pipeline *tpl_polynomial(int degree) {
    char name[64]; snprintf(name, sizeof(name), "polynomial_d%d", degree);
    Pipeline *p = pipeline_create(name);
    int n_in = degree + 2;  /* a_0..a_d, x */
    char in_buf[16][8];
    const char *in_names[16];
    for (int k = 0; k <= degree; k++) {
        snprintf(in_buf[k], sizeof(in_buf[k]), "a%d", k);
        in_names[k] = in_buf[k];
    }
    snprintf(in_buf[degree + 1], sizeof(in_buf[degree + 1]), "x");
    in_names[degree + 1] = in_buf[degree + 1];
    sig_n_in_1_out(p, n_in, in_names, "y");

    /* Build x^1, x^2, ..., x^d iteratively. */
    /* x^k = x^{k-1} * x */
    for (int k = 2; k <= degree; k++) {
        char id[16]; snprintf(id, sizeof(id), "xp%d", k);
        node_2in(p, id, "multiply");
        if (k == 2) {
            pipeline_connect_signature_in(p, "x", id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "xp%d", k - 1);
            pipeline_connect(p, prev, "out", id, "x");
        }
        pipeline_connect_signature_in(p, "x", id, "y");
    }
    /* Build a_k * x^k for k >= 1. */
    /* For k=1: a_1 * x. For k>=2: a_k * xp_k. */
    for (int k = 1; k <= degree; k++) {
        char id[16]; snprintf(id, sizeof(id), "term%d", k);
        node_2in(p, id, "multiply");
        char a_name[8]; snprintf(a_name, sizeof(a_name), "a%d", k);
        pipeline_connect_signature_in(p, a_name, id, "x");
        if (k == 1) {
            pipeline_connect_signature_in(p, "x", id, "y");
        } else {
            char xp[16]; snprintf(xp, sizeof(xp), "xp%d", k);
            pipeline_connect(p, xp, "out", id, "y");
        }
    }
    /* Sum: a_0 + term_1 + ... + term_d. */
    /* sum_1 = a_0 + term_1, sum_2 = sum_1 + term_2, ... */
    if (degree == 0) {
        /* Degenerate: y = a_0. We don't support this — tpl_polynomial assumes degree >= 1. */
        pipeline_free(p);
        return NULL;
    }
    for (int k = 1; k <= degree; k++) {
        char id[16]; snprintf(id, sizeof(id), "sum%d", k);
        node_2in(p, id, "add");
        if (k == 1) {
            pipeline_connect_signature_in(p, "a0", id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "sum%d", k - 1);
            pipeline_connect(p, prev, "out", id, "x");
        }
        char term[16]; snprintf(term, sizeof(term), "term%d", k);
        pipeline_connect(p, term, "out", id, "y");
    }
    char last[16]; snprintf(last, sizeof(last), "sum%d", degree);
    pipeline_connect_signature_out(p, last, "out", "y");
    return p;
}

/* ============================================================
 *  Template family 4: distance_squared_nd(dim)
 *    y = sum_i (a_i - b_i)^2
 * ============================================================ */
static Pipeline *tpl_distance_squared(int dim) {
    char name[64]; snprintf(name, sizeof(name), "distance_squared_%dd", dim);
    Pipeline *p = pipeline_create(name);
    int n_in = 2 * dim;
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < dim; i++) {
        snprintf(in_buf[2 * i],     sizeof(in_buf[0]), "a%d", i + 1);
        snprintf(in_buf[2 * i + 1], sizeof(in_buf[0]), "b%d", i + 1);
        in_names[2 * i]     = in_buf[2 * i];
        in_names[2 * i + 1] = in_buf[2 * i + 1];
    }
    sig_n_in_1_out(p, n_in, in_names, "y");
    /* d_i = a_i - b_i; sq_i = d_i * d_i. */
    for (int i = 0; i < dim; i++) {
        char d_id[16];   snprintf(d_id,   sizeof(d_id),   "d%d", i + 1);
        char sq_id[16];  snprintf(sq_id,  sizeof(sq_id),  "sq%d", i + 1);
        node_2in(p, d_id, "subtract");
        node_2in(p, sq_id, "multiply");
        char a[8], b[8]; snprintf(a, sizeof(a), "a%d", i + 1); snprintf(b, sizeof(b), "b%d", i + 1);
        pipeline_connect_signature_in(p, a, d_id, "x");
        pipeline_connect_signature_in(p, b, d_id, "y");
        pipeline_connect(p, d_id, "out", sq_id, "x");
        pipeline_connect(p, d_id, "out", sq_id, "y");
    }
    /* Sum of sq_i. */
    if (dim == 1) {
        pipeline_connect_signature_out(p, "sq1", "out", "y");
    } else {
        for (int i = 1; i < dim; i++) {
            char id[16]; snprintf(id, sizeof(id), "sum%d", i);
            node_2in(p, id, "add");
            if (i == 1) {
                pipeline_connect(p, "sq1", "out", id, "x");
            } else {
                char prev[16]; snprintf(prev, sizeof(prev), "sum%d", i - 1);
                pipeline_connect(p, prev, "out", id, "x");
            }
            char sq[16]; snprintf(sq, sizeof(sq), "sq%d", i + 1);
            pipeline_connect(p, sq, "out", id, "y");
        }
        char last[16]; snprintf(last, sizeof(last), "sum%d", dim - 1);
        pipeline_connect_signature_out(p, last, "out", "y");
    }
    return p;
}

/* ============================================================
 *  Template family 5: dot_product_nd(dim)
 *    y = sum_i a_i * b_i
 * ============================================================ */
static Pipeline *tpl_dot_product(int dim) {
    char name[64]; snprintf(name, sizeof(name), "dot_product_%dd", dim);
    Pipeline *p = pipeline_create(name);
    int n_in = 2 * dim;
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < dim; i++) {
        snprintf(in_buf[2 * i],     sizeof(in_buf[0]), "a%d", i + 1);
        snprintf(in_buf[2 * i + 1], sizeof(in_buf[0]), "b%d", i + 1);
        in_names[2 * i]     = in_buf[2 * i];
        in_names[2 * i + 1] = in_buf[2 * i + 1];
    }
    sig_n_in_1_out(p, n_in, in_names, "y");
    for (int i = 0; i < dim; i++) {
        char id[16]; snprintf(id, sizeof(id), "p%d", i + 1);
        node_2in(p, id, "multiply");
        char a[8], b[8]; snprintf(a, sizeof(a), "a%d", i + 1); snprintf(b, sizeof(b), "b%d", i + 1);
        pipeline_connect_signature_in(p, a, id, "x");
        pipeline_connect_signature_in(p, b, id, "y");
    }
    if (dim == 1) {
        pipeline_connect_signature_out(p, "p1", "out", "y");
    } else {
        for (int i = 1; i < dim; i++) {
            char id[16]; snprintf(id, sizeof(id), "sum%d", i);
            node_2in(p, id, "add");
            if (i == 1) {
                pipeline_connect(p, "p1", "out", id, "x");
            } else {
                char prev[16]; snprintf(prev, sizeof(prev), "sum%d", i - 1);
                pipeline_connect(p, prev, "out", id, "x");
            }
            char p_id[16]; snprintf(p_id, sizeof(p_id), "p%d", i + 1);
            pipeline_connect(p, p_id, "out", id, "y");
        }
        char last[16]; snprintf(last, sizeof(last), "sum%d", dim - 1);
        pipeline_connect_signature_out(p, last, "out", "y");
    }
    return p;
}

/* ============================================================
 *  Template family 6: mean_n(n) — sum / n_constant
 *    y = (x_1 + ... + x_n) / n
 *  Uses a `divide` primitive with config "by" = n.
 * ============================================================ */
static Pipeline *tpl_mean(int n) {
    char name[64]; snprintf(name, sizeof(name), "mean_%d", n);
    Pipeline *p = pipeline_create(name);
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < n; i++) {
        snprintf(in_buf[i], sizeof(in_buf[i]), "x%d", i + 1);
        in_names[i] = in_buf[i];
    }
    sig_n_in_1_out(p, n, in_names, "y");
    /* Sum chain. */
    for (int i = 1; i < n; i++) {
        char id[16]; snprintf(id, sizeof(id), "s%d", i);
        node_2in(p, id, "add");
        if (i == 1) {
            pipeline_connect_signature_in(p, in_names[0], id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "s%d", i - 1);
            pipeline_connect(p, prev, "out", id, "x");
        }
        pipeline_connect_signature_in(p, in_names[i], id, "y");
    }
    /* Final divide by n. */
    char last[16]; snprintf(last, sizeof(last), "s%d", n - 1);
    /* divide_by_const: 1-input node with config "by". */
    const char *div_in_names[]  = { "x" };
    PipelineType *div_in_types[] = { T_int() };
    const char *div_out_names[] = { "out" };
    PipelineType *div_out_types[] = { T_int() };
    pipeline_add_node(p, "div", "divide_by_const",
                      1, div_in_names, div_in_types,
                      1, div_out_names, div_out_types);
    pipeline_node_set_config_int(p, "div", "by", (int64_t)n);
    pipeline_connect(p, last, "out", "div", "x");
    pipeline_connect_signature_out(p, "div", "out", "y");
    return p;
}

/* ============================================================
 *  Template family 7: weighted_combine_n(n)
 *    y = w_1*x_1 + w_2*x_2 + ... + w_n*x_n
 * ============================================================ */
static Pipeline *tpl_weighted_combine(int n) {
    char name[64]; snprintf(name, sizeof(name), "weighted_combine_%d", n);
    Pipeline *p = pipeline_create(name);
    int n_in = 2 * n;  /* w_i, x_i */
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < n; i++) {
        snprintf(in_buf[2 * i],     sizeof(in_buf[0]), "w%d", i + 1);
        snprintf(in_buf[2 * i + 1], sizeof(in_buf[0]), "x%d", i + 1);
        in_names[2 * i]     = in_buf[2 * i];
        in_names[2 * i + 1] = in_buf[2 * i + 1];
    }
    sig_n_in_1_out(p, n_in, in_names, "y");
    for (int i = 0; i < n; i++) {
        char id[16]; snprintf(id, sizeof(id), "wx%d", i + 1);
        node_2in(p, id, "multiply");
        char w[8], x[8]; snprintf(w, sizeof(w), "w%d", i + 1); snprintf(x, sizeof(x), "x%d", i + 1);
        pipeline_connect_signature_in(p, w, id, "x");
        pipeline_connect_signature_in(p, x, id, "y");
    }
    if (n == 1) {
        pipeline_connect_signature_out(p, "wx1", "out", "y");
    } else {
        for (int i = 1; i < n; i++) {
            char id[16]; snprintf(id, sizeof(id), "sum%d", i);
            node_2in(p, id, "add");
            if (i == 1) {
                pipeline_connect(p, "wx1", "out", id, "x");
            } else {
                char prev[16]; snprintf(prev, sizeof(prev), "sum%d", i - 1);
                pipeline_connect(p, prev, "out", id, "x");
            }
            char wx[16]; snprintf(wx, sizeof(wx), "wx%d", i + 1);
            pipeline_connect(p, wx, "out", id, "y");
        }
        char last[16]; snprintf(last, sizeof(last), "sum%d", n - 1);
        pipeline_connect_signature_out(p, last, "out", "y");
    }
    return p;
}

/* ============================================================
 *  Template family 8: axpy_then_op(post, depth)
 *    Stack of `depth` axpys (each y_i = a*x_i + y_{i-1}), then post-op.
 * ============================================================ */
static Pipeline *tpl_axpy_chain(const char *post, int depth) {
    char name[64]; snprintf(name, sizeof(name), "axpy_then_%s_%d", post, depth);
    Pipeline *p = pipeline_create(name);
    int n_in = 2 + depth;  /* a, y_in (initial), x_1..x_depth */
    char in_buf[16][8];
    const char *in_names[16];
    snprintf(in_buf[0], sizeof(in_buf[0]), "a");      in_names[0] = in_buf[0];
    snprintf(in_buf[1], sizeof(in_buf[0]), "y_in");   in_names[1] = in_buf[1];
    for (int i = 0; i < depth; i++) {
        snprintf(in_buf[2 + i], sizeof(in_buf[0]), "x%d", i + 1);
        in_names[2 + i] = in_buf[2 + i];
    }
    sig_n_in_1_out(p, n_in, in_names, "y");
    /* Build per-layer: ax_i = a*x_i, then sum_i = ax_i + previous-sum. */
    for (int i = 0; i < depth; i++) {
        char ax[16];   snprintf(ax,   sizeof(ax),   "ax%d", i + 1);
        char sum[16];  snprintf(sum,  sizeof(sum),  "s%d", i + 1);
        node_2in(p, ax, "multiply");
        node_2in(p, sum, "add");
        pipeline_connect_signature_in(p, "a", ax, "x");
        char x[8]; snprintf(x, sizeof(x), "x%d", i + 1);
        pipeline_connect_signature_in(p, x, ax, "y");
        pipeline_connect(p, ax, "out", sum, "x");
        if (i == 0) {
            pipeline_connect_signature_in(p, "y_in", sum, "y");
        } else {
            char prev_sum[16]; snprintf(prev_sum, sizeof(prev_sum), "s%d", i);
            pipeline_connect(p, prev_sum, "out", sum, "y");
        }
    }
    /* Post-op (unary). */
    node_1in(p, "post", post);
    char last[16]; snprintf(last, sizeof(last), "s%d", depth);
    pipeline_connect(p, last, "out", "post", "x");
    pipeline_connect_signature_out(p, "post", "out", "y");
    return p;
}

/* ============================================================
 *  Template family 9: lerp_n(n) — chained linear interpolations
 *    Successively lerp between waypoints w_1, w_2, ..., w_n with t.
 * ============================================================ */
static Pipeline *tpl_lerp(int n) {
    char name[64]; snprintf(name, sizeof(name), "lerp_%d", n);
    Pipeline *p = pipeline_create(name);
    /* Inputs: w_1..w_n, t.  Output: y. */
    int n_in = n + 1;
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < n; i++) {
        snprintf(in_buf[i], sizeof(in_buf[i]), "w%d", i + 1);
        in_names[i] = in_buf[i];
    }
    snprintf(in_buf[n], sizeof(in_buf[n]), "t");
    in_names[n] = in_buf[n];
    sig_n_in_1_out(p, n_in, in_names, "y");
    /* lerp(a, b, t) = a + (b-a)*t. Cascade: first = lerp(w_1, w_2, t),
     * then = lerp(first, w_3, t), etc. */
    for (int i = 1; i < n; i++) {
        char diff[16]; snprintf(diff, sizeof(diff), "d%d", i);
        char scale[16]; snprintf(scale, sizeof(scale), "s%d", i);
        char res[16]; snprintf(res, sizeof(res), "r%d", i);
        node_2in(p, diff, "subtract");
        node_2in(p, scale, "multiply");
        node_2in(p, res, "add");
        char w_curr[8]; snprintf(w_curr, sizeof(w_curr), "w%d", i + 1);
        if (i == 1) {
            pipeline_connect_signature_in(p, w_curr, diff, "x");
            pipeline_connect_signature_in(p, "w1", diff, "y");
        } else {
            char prev_res[16]; snprintf(prev_res, sizeof(prev_res), "r%d", i - 1);
            pipeline_connect_signature_in(p, w_curr, diff, "x");
            pipeline_connect(p, prev_res, "out", diff, "y");
        }
        pipeline_connect(p, diff, "out", scale, "x");
        pipeline_connect_signature_in(p, "t", scale, "y");
        if (i == 1) {
            pipeline_connect_signature_in(p, "w1", res, "x");
        } else {
            char prev_res[16]; snprintf(prev_res, sizeof(prev_res), "r%d", i - 1);
            pipeline_connect(p, prev_res, "out", res, "x");
        }
        pipeline_connect(p, scale, "out", res, "y");
    }
    char last[16]; snprintf(last, sizeof(last), "r%d", n - 1);
    pipeline_connect_signature_out(p, last, "out", "y");
    return p;
}

/* ============================================================
 *  Template family 10: range_n(n) = max(x_1..x_n) - min(x_1..x_n)
 * ============================================================ */
static Pipeline *tpl_range(int n) {
    char name[64]; snprintf(name, sizeof(name), "range_%d", n);
    Pipeline *p = pipeline_create(name);
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < n; i++) {
        snprintf(in_buf[i], sizeof(in_buf[i]), "x%d", i + 1);
        in_names[i] = in_buf[i];
    }
    sig_n_in_1_out(p, n, in_names, "y");
    /* Max chain. */
    for (int i = 1; i < n; i++) {
        char id[16]; snprintf(id, sizeof(id), "mx%d", i);
        node_2in(p, id, "max");
        if (i == 1) {
            pipeline_connect_signature_in(p, in_names[0], id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "mx%d", i - 1);
            pipeline_connect(p, prev, "out", id, "x");
        }
        pipeline_connect_signature_in(p, in_names[i], id, "y");
    }
    /* Min chain. */
    for (int i = 1; i < n; i++) {
        char id[16]; snprintf(id, sizeof(id), "mn%d", i);
        node_2in(p, id, "min");
        if (i == 1) {
            pipeline_connect_signature_in(p, in_names[0], id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "mn%d", i - 1);
            pipeline_connect(p, prev, "out", id, "x");
        }
        pipeline_connect_signature_in(p, in_names[i], id, "y");
    }
    /* Subtract. */
    char mx_last[16]; snprintf(mx_last, sizeof(mx_last), "mx%d", n - 1);
    char mn_last[16]; snprintf(mn_last, sizeof(mn_last), "mn%d", n - 1);
    node_2in(p, "diff", "subtract");
    pipeline_connect(p, mx_last, "out", "diff", "x");
    pipeline_connect(p, mn_last, "out", "diff", "y");
    pipeline_connect_signature_out(p, "diff", "out", "y");
    return p;
}

/* ============================================================
 *  Phase 4 — Real-primitive seed graphs.
 *
 *  Each builder mirrors one of the already-composed end-of-file
 *  functions in demos/word-level/vm_codegen/w_vm_functions.txt.
 *  Together they teach the organelle that real primitive names
 *  (compound, bmi, gcd, sigmoid, clamp, ...) are valid node ops.
 * ============================================================ */

/* sum_results(a, b) = double_val(a) + triple_val(b) */
static Pipeline *seed_sum_results(void) {
    Pipeline *p = pipeline_create("sum_results");
    const char *ins[] = { "a", "b" };
    sig_n_in_1_out(p, 2, ins, "y");
    node_1in(p, "x", "double_val");
    node_1in(p, "y_n", "triple_val");
    node_2in(p, "s", "add");
    pipeline_connect_signature_in(p, "a", "x", "x");
    pipeline_connect_signature_in(p, "b", "y_n", "x");
    pipeline_connect(p, "x", "out", "s", "x");
    pipeline_connect(p, "y_n", "out", "s", "y");
    pipeline_connect_signature_out(p, "s", "out", "y");
    return p;
}

/* compound_interest(principal, rate, years) = compound(p,r,y) - principal */
static Pipeline *seed_compound_interest(void) {
    Pipeline *p = pipeline_create("compound_interest");
    const char *ins[] = { "principal", "rate", "years" };
    sig_n_in_1_out(p, 3, ins, "y");
    const char *cnames[] = { "principal", "rate", "periods" };
    node_named(p, "amount", "compound", 3, cnames);
    node_2in(p, "diff", "subtract");
    pipeline_connect_signature_in(p, "principal", "amount", "principal");
    pipeline_connect_signature_in(p, "rate",      "amount", "rate");
    pipeline_connect_signature_in(p, "years",     "amount", "periods");
    pipeline_connect(p, "amount", "out", "diff", "x");
    pipeline_connect_signature_in(p, "principal", "diff", "y");
    pipeline_connect_signature_out(p, "diff", "out", "y");
    return p;
}

/* analyze_two_points(a, b) = distance_1d(a,b) + midpoint(a,b) */
static Pipeline *seed_analyze_two_points(void) {
    Pipeline *p = pipeline_create("analyze_two_points");
    const char *ins[] = { "a", "b" };
    sig_n_in_1_out(p, 2, ins, "y");
    const char *abnames[] = { "a", "b" };
    node_named(p, "dist", "distance_1d", 2, abnames);
    node_named(p, "mid",  "midpoint",    2, abnames);
    node_2in(p, "s", "add");
    pipeline_connect_signature_in(p, "a", "dist", "a");
    pipeline_connect_signature_in(p, "b", "dist", "b");
    pipeline_connect_signature_in(p, "a", "mid", "a");
    pipeline_connect_signature_in(p, "b", "mid", "b");
    pipeline_connect(p, "dist", "out", "s", "x");
    pipeline_connect(p, "mid",  "out", "s", "y");
    pipeline_connect_signature_out(p, "s", "out", "y");
    return p;
}

/* clamped_average(a, b, lo, hi) = clamp(average_two(a,b), lo, hi) */
static Pipeline *seed_clamped_average(void) {
    Pipeline *p = pipeline_create("clamped_average");
    const char *ins[] = { "a", "b", "lo", "hi" };
    sig_n_in_1_out(p, 4, ins, "y");
    const char *abnames[] = { "a", "b" };
    node_named(p, "avg", "average_two", 2, abnames);
    const char *cnames[] = { "x", "lo", "hi" };
    node_named(p, "c", "clamp", 3, cnames);
    pipeline_connect_signature_in(p, "a", "avg", "a");
    pipeline_connect_signature_in(p, "b", "avg", "b");
    pipeline_connect(p, "avg", "out", "c", "x");
    pipeline_connect_signature_in(p, "lo", "c", "lo");
    pipeline_connect_signature_in(p, "hi", "c", "hi");
    pipeline_connect_signature_out(p, "c", "out", "y");
    return p;
}

/* abs_difference(a, b) = abs_val(a - b) */
static Pipeline *seed_abs_difference(void) {
    Pipeline *p = pipeline_create("abs_difference");
    const char *ins[] = { "a", "b" };
    sig_n_in_1_out(p, 2, ins, "y");
    node_2in(p, "d", "subtract");
    node_1in(p, "a_v", "abs_val");
    pipeline_connect_signature_in(p, "a", "d", "x");
    pipeline_connect_signature_in(p, "b", "d", "y");
    pipeline_connect(p, "d", "out", "a_v", "x");
    pipeline_connect_signature_out(p, "a_v", "out", "y");
    return p;
}

/* discounted_tax(price, discount_rate, tax_rate) =
 *   tax_amount(discount(price, discount_rate), tax_rate)              */
static Pipeline *seed_discounted_tax(void) {
    Pipeline *p = pipeline_create("discounted_tax");
    const char *ins[] = { "price", "discount_rate", "tax_rate" };
    sig_n_in_1_out(p, 3, ins, "y");
    const char *dnames[] = { "price", "rate" };
    node_named(p, "disc", "discount", 2, dnames);
    const char *tnames[] = { "amount", "rate" };
    node_named(p, "tax",  "tax_amount", 2, tnames);
    pipeline_connect_signature_in(p, "price",         "disc", "price");
    pipeline_connect_signature_in(p, "discount_rate", "disc", "rate");
    pipeline_connect(p, "disc", "out", "tax", "amount");
    pipeline_connect_signature_in(p, "tax_rate",      "tax",  "rate");
    pipeline_connect_signature_out(p, "tax", "out", "y");
    return p;
}

/* total_with_tax(price, tax_rate) = price + tax_amount(price, tax_rate) */
static Pipeline *seed_total_with_tax(void) {
    Pipeline *p = pipeline_create("total_with_tax");
    const char *ins[] = { "price", "tax_rate" };
    sig_n_in_1_out(p, 2, ins, "y");
    const char *tnames[] = { "amount", "rate" };
    node_named(p, "tax", "tax_amount", 2, tnames);
    node_2in(p, "tot", "add");
    pipeline_connect_signature_in(p, "price",    "tax", "amount");
    pipeline_connect_signature_in(p, "tax_rate", "tax", "rate");
    pipeline_connect_signature_in(p, "price",    "tot", "x");
    pipeline_connect(p, "tax", "out", "tot", "y");
    pipeline_connect_signature_out(p, "tot", "out", "y");
    return p;
}

/* net_pay(gross, tax_rate) = apply_tax(gross, tax_rate)               */
static Pipeline *seed_net_pay(void) {
    Pipeline *p = pipeline_create("net_pay");
    const char *ins[] = { "gross", "tax_rate" };
    sig_n_in_1_out(p, 2, ins, "y");
    const char *anames[] = { "amount", "rate" };
    node_named(p, "net", "apply_tax", 2, anames);
    pipeline_connect_signature_in(p, "gross",    "net", "amount");
    pipeline_connect_signature_in(p, "tax_rate", "net", "rate");
    pipeline_connect_signature_out(p, "net", "out", "y");
    return p;
}

/* savings_rate(income, expenses) = percentage(income - expenses, income) */
static Pipeline *seed_savings_rate(void) {
    Pipeline *p = pipeline_create("savings_rate");
    const char *ins[] = { "income", "expenses" };
    sig_n_in_1_out(p, 2, ins, "y");
    node_2in(p, "saved", "subtract");
    const char *pnames[] = { "part", "whole" };
    node_named(p, "rate", "percentage", 2, pnames);
    pipeline_connect_signature_in(p, "income",   "saved", "x");
    pipeline_connect_signature_in(p, "expenses", "saved", "y");
    pipeline_connect(p, "saved", "out", "rate", "part");
    pipeline_connect_signature_in(p, "income",   "rate",  "whole");
    pipeline_connect_signature_out(p, "rate", "out", "y");
    return p;
}

/* fib_fact_product(n) = fibonacci(n) * factorial(n)                   */
static Pipeline *seed_fib_fact_product(void) {
    Pipeline *p = pipeline_create("fib_fact_product");
    const char *ins[] = { "n" };
    sig_n_in_1_out(p, 1, ins, "y");
    node_1in(p, "fib",  "fibonacci");
    node_1in(p, "fact", "factorial");
    node_2in(p, "prod", "multiply");
    pipeline_connect_signature_in(p, "n", "fib",  "x");
    pipeline_connect_signature_in(p, "n", "fact", "x");
    pipeline_connect(p, "fib",  "out", "prod", "x");
    pipeline_connect(p, "fact", "out", "prod", "y");
    pipeline_connect_signature_out(p, "prod", "out", "y");
    return p;
}

/* net_present_value(cashflow, rate, years) = present_value(future_value(cashflow,r,y), r, y) */
static Pipeline *seed_net_present_value(void) {
    Pipeline *p = pipeline_create("net_present_value");
    const char *ins[] = { "cashflow", "rate", "years" };
    sig_n_in_1_out(p, 3, ins, "y");
    const char *fnames[] = { "present", "rate", "periods" };
    node_named(p, "fv", "future_value", 3, fnames);
    const char *pnames[] = { "future", "rate", "periods" };
    node_named(p, "pv", "present_value", 3, pnames);
    pipeline_connect_signature_in(p, "cashflow", "fv", "present");
    pipeline_connect_signature_in(p, "rate",     "fv", "rate");
    pipeline_connect_signature_in(p, "years",    "fv", "periods");
    pipeline_connect(p, "fv", "out", "pv", "future");
    pipeline_connect_signature_in(p, "rate",     "pv", "rate");
    pipeline_connect_signature_in(p, "years",    "pv", "periods");
    pipeline_connect_signature_out(p, "pv", "out", "y");
    return p;
}

/* clamped_sigmoid(x, lo, hi) = clamp(sigmoid(x), lo, hi)              */
static Pipeline *seed_clamped_sigmoid(void) {
    Pipeline *p = pipeline_create("clamped_sigmoid");
    const char *ins[] = { "x", "lo", "hi" };
    sig_n_in_1_out(p, 3, ins, "y");
    node_1in(p, "s", "sigmoid");
    const char *cnames[] = { "x", "lo", "hi" };
    node_named(p, "c", "clamp", 3, cnames);
    pipeline_connect_signature_in(p, "x", "s", "x");
    pipeline_connect(p, "s", "out", "c", "x");
    pipeline_connect_signature_in(p, "lo", "c", "lo");
    pipeline_connect_signature_in(p, "hi", "c", "hi");
    pipeline_connect_signature_out(p, "c", "out", "y");
    return p;
}

/* scaled_relu(x, scale) = relu(x) * scale                             */
static Pipeline *seed_scaled_relu(void) {
    Pipeline *p = pipeline_create("scaled_relu");
    const char *ins[] = { "x", "scale" };
    sig_n_in_1_out(p, 2, ins, "y");
    node_1in(p, "r", "relu");
    node_2in(p, "m", "multiply");
    pipeline_connect_signature_in(p, "x", "r", "x");
    pipeline_connect(p, "r", "out", "m", "x");
    pipeline_connect_signature_in(p, "scale", "m", "y");
    pipeline_connect_signature_out(p, "m", "out", "y");
    return p;
}

/* gcd_product(a, b) = gcd(a, b) * a * b                                */
static Pipeline *seed_gcd_product(void) {
    Pipeline *p = pipeline_create("gcd_product");
    const char *ins[] = { "a", "b" };
    sig_n_in_1_out(p, 2, ins, "y");
    const char *abnames[] = { "a", "b" };
    node_named(p, "g", "gcd", 2, abnames);
    node_2in(p, "ga", "multiply");
    node_2in(p, "gab", "multiply");
    pipeline_connect_signature_in(p, "a", "g", "a");
    pipeline_connect_signature_in(p, "b", "g", "b");
    pipeline_connect(p, "g", "out", "ga", "x");
    pipeline_connect_signature_in(p, "a", "ga", "y");
    pipeline_connect(p, "ga", "out", "gab", "x");
    pipeline_connect_signature_in(p, "b", "gab", "y");
    pipeline_connect_signature_out(p, "gab", "out", "y");
    return p;
}

/* bmi_classified(weight, height, lo, hi) = clamp(bmi(weight, height), lo, hi)
 *  (a hand-rolled instance — also covered parametrically by tpl_bmi_classified) */
static Pipeline *seed_bmi_classified(void) {
    Pipeline *p = pipeline_create("bmi_classified");
    const char *ins[] = { "weight", "height", "lo", "hi" };
    sig_n_in_1_out(p, 4, ins, "y");
    const char *bnames[] = { "weight", "height" };
    node_named(p, "b", "bmi", 2, bnames);
    const char *cnames[] = { "x", "lo", "hi" };
    node_named(p, "c", "clamp", 3, cnames);
    pipeline_connect_signature_in(p, "weight", "b", "weight");
    pipeline_connect_signature_in(p, "height", "b", "height");
    pipeline_connect(p, "b", "out", "c", "x");
    pipeline_connect_signature_in(p, "lo", "c", "lo");
    pipeline_connect_signature_in(p, "hi", "c", "hi");
    pipeline_connect_signature_out(p, "c", "out", "y");
    return p;
}

/* Wrapper-with-no-args adapter for seeds. */
typedef Pipeline *(*SeedFn)(void);
static Pipeline *w_seed(void *ctx) { return ((SeedFn)ctx)(); }

/* ============================================================
 *  Phase 4 — Templated families using REAL primitives
 * ============================================================ */

/* tpl_clamped_op(unary_op): clamp(unary_op(x), lo, hi). */
static Pipeline *tpl_clamped_op(const char *unary_op) {
    char name[64]; snprintf(name, sizeof(name), "clamped_%s", unary_op);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "x", "lo", "hi" };
    sig_n_in_1_out(p, 3, ins, "y");
    node_1in(p, "u", unary_op);
    const char *cnames[] = { "x", "lo", "hi" };
    node_named(p, "c", "clamp", 3, cnames);
    pipeline_connect_signature_in(p, "x", "u", "x");
    pipeline_connect(p, "u", "out", "c", "x");
    pipeline_connect_signature_in(p, "lo", "c", "lo");
    pipeline_connect_signature_in(p, "hi", "c", "hi");
    pipeline_connect_signature_out(p, "c", "out", "y");
    return p;
}

/* tpl_taxed_total(qty_first): price * qty (or qty * price), then add tax_amount. */
static Pipeline *tpl_taxed_total(int qty_first) {
    char name[64]; snprintf(name, sizeof(name), "taxed_total_%d", qty_first);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "price", "qty", "tax_rate" };
    sig_n_in_1_out(p, 3, ins, "y");
    node_2in(p, "subtotal", "multiply");
    const char *tnames[] = { "amount", "rate" };
    node_named(p, "tax", "tax_amount", 2, tnames);
    node_2in(p, "tot", "add");
    if (qty_first) {
        pipeline_connect_signature_in(p, "qty",   "subtotal", "x");
        pipeline_connect_signature_in(p, "price", "subtotal", "y");
    } else {
        pipeline_connect_signature_in(p, "price", "subtotal", "x");
        pipeline_connect_signature_in(p, "qty",   "subtotal", "y");
    }
    pipeline_connect(p, "subtotal", "out", "tax", "amount");
    pipeline_connect_signature_in(p, "tax_rate", "tax", "rate");
    pipeline_connect(p, "subtotal", "out", "tot", "x");
    pipeline_connect(p, "tax",      "out", "tot", "y");
    pipeline_connect_signature_out(p, "tot", "out", "y");
    return p;
}

/* tpl_savings_pipeline(n_expenses):
 *   sum n expense terms, subtract from income, then percentage of income. */
static Pipeline *tpl_savings_pipeline(int n_expenses) {
    char name[64]; snprintf(name, sizeof(name), "savings_pipeline_%d", n_expenses);
    Pipeline *p = pipeline_create(name);
    int n_in = 1 + n_expenses;
    char in_buf[8][16];
    const char *in_names[8];
    snprintf(in_buf[0], sizeof(in_buf[0]), "income"); in_names[0] = in_buf[0];
    for (int i = 0; i < n_expenses; i++) {
        snprintf(in_buf[1 + i], sizeof(in_buf[0]), "exp%d", i + 1);
        in_names[1 + i] = in_buf[1 + i];
    }
    sig_n_in_1_out(p, n_in, in_names, "y");
    /* Sum expenses left-to-right. */
    for (int i = 1; i < n_expenses; i++) {
        char id[16]; snprintf(id, sizeof(id), "se%d", i);
        node_2in(p, id, "add");
        if (i == 1) {
            pipeline_connect_signature_in(p, "exp1", id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "se%d", i - 1);
            pipeline_connect(p, prev, "out", id, "x");
        }
        char e[16]; snprintf(e, sizeof(e), "exp%d", i + 1);
        pipeline_connect_signature_in(p, e, id, "y");
    }
    /* total_exp source */
    char total_id[16];
    if (n_expenses == 1) {
        snprintf(total_id, sizeof(total_id), "exp1");  /* sig ref */
    } else {
        snprintf(total_id, sizeof(total_id), "se%d", n_expenses - 1);
    }
    /* saved = income - total_exp */
    node_2in(p, "saved", "subtract");
    pipeline_connect_signature_in(p, "income", "saved", "x");
    if (n_expenses == 1) {
        pipeline_connect_signature_in(p, "exp1", "saved", "y");
    } else {
        pipeline_connect(p, total_id, "out", "saved", "y");
    }
    /* rate = percentage(saved, income) */
    const char *pnames[] = { "part", "whole" };
    node_named(p, "rate", "percentage", 2, pnames);
    pipeline_connect(p, "saved", "out", "rate", "part");
    pipeline_connect_signature_in(p, "income", "rate", "whole");
    pipeline_connect_signature_out(p, "rate", "out", "y");
    return p;
}

/* tpl_compound_chain(periods):  compound interest earned = compound(p,r,n) - p. */
static Pipeline *tpl_compound_chain(int periods) {
    (void)periods; /* periods is conceptual — we still emit one compound + one subtract */
    char name[64]; snprintf(name, sizeof(name), "compound_chain_%d", periods);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "principal", "rate", "years" };
    sig_n_in_1_out(p, 3, ins, "y");
    const char *cnames[] = { "principal", "rate", "periods" };
    node_named(p, "amt", "compound", 3, cnames);
    node_2in(p, "earned", "subtract");
    pipeline_connect_signature_in(p, "principal", "amt", "principal");
    pipeline_connect_signature_in(p, "rate",      "amt", "rate");
    pipeline_connect_signature_in(p, "years",     "amt", "periods");
    pipeline_connect(p, "amt", "out", "earned", "x");
    pipeline_connect_signature_in(p, "principal", "earned", "y");
    pipeline_connect_signature_out(p, "earned", "out", "y");
    return p;
}

/* tpl_gcd_chain(depth):  gcd(...gcd(a,b)... ) of `depth+1` inputs, then * by another input. */
static Pipeline *tpl_gcd_chain(int depth) {
    char name[64]; snprintf(name, sizeof(name), "gcd_chain_%d", depth);
    Pipeline *p = pipeline_create(name);
    int n_in = depth + 2;  /* x_1..x_{depth+1}, k */
    char in_buf[8][8];
    const char *in_names[8];
    for (int i = 0; i <= depth; i++) {
        snprintf(in_buf[i], sizeof(in_buf[0]), "x%d", i + 1);
        in_names[i] = in_buf[i];
    }
    snprintf(in_buf[depth + 1], sizeof(in_buf[0]), "k");
    in_names[depth + 1] = in_buf[depth + 1];
    sig_n_in_1_out(p, n_in, in_names, "y");
    /* gcd chain. */
    const char *gn[] = { "a", "b" };
    for (int i = 1; i <= depth; i++) {
        char id[16]; snprintf(id, sizeof(id), "g%d", i);
        node_named(p, id, "gcd", 2, gn);
        if (i == 1) {
            pipeline_connect_signature_in(p, "x1", id, "a");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "g%d", i - 1);
            pipeline_connect(p, prev, "out", id, "a");
        }
        char x[8]; snprintf(x, sizeof(x), "x%d", i + 1);
        pipeline_connect_signature_in(p, x, id, "b");
    }
    /* Multiply gcd by k. */
    node_2in(p, "m", "multiply");
    char last[16]; snprintf(last, sizeof(last), "g%d", depth);
    pipeline_connect(p, last, "out", "m", "x");
    pipeline_connect_signature_in(p, "k", "m", "y");
    pipeline_connect_signature_out(p, "m", "out", "y");
    return p;
}

/* tpl_fib_fact_blend(combine_op): fibonacci(n) op factorial(n). */
static Pipeline *tpl_fib_fact_blend(const char *op) {
    char name[64]; snprintf(name, sizeof(name), "fib_fact_%s", op);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "n" };
    sig_n_in_1_out(p, 1, ins, "y");
    node_1in(p, "fib",  "fibonacci");
    node_1in(p, "fact", "factorial");
    int two_input = 1;
    /* All ops we use here are 2-input binary primitives. */
    if (two_input) {
        node_2in(p, "blend", op);
        pipeline_connect_signature_in(p, "n", "fib",  "x");
        pipeline_connect_signature_in(p, "n", "fact", "x");
        pipeline_connect(p, "fib",  "out", "blend", "x");
        pipeline_connect(p, "fact", "out", "blend", "y");
        pipeline_connect_signature_out(p, "blend", "out", "y");
    }
    return p;
}

/* tpl_bmi_classified(class_kind):  clamp(bmi(weight, height), lo, hi).
 * class_kind toggles whether output is post-clamp or post-clamp+normalize. */
static Pipeline *tpl_bmi_classified(int normalize) {
    char name[64]; snprintf(name, sizeof(name), "bmi_classified_%d", normalize);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "weight", "height", "lo", "hi" };
    sig_n_in_1_out(p, 4, ins, "y");
    const char *bn[] = { "weight", "height" };
    node_named(p, "b", "bmi", 2, bn);
    const char *cn[] = { "x", "lo", "hi" };
    node_named(p, "c", "clamp", 3, cn);
    pipeline_connect_signature_in(p, "weight", "b", "weight");
    pipeline_connect_signature_in(p, "height", "b", "height");
    pipeline_connect(p, "b", "out", "c", "x");
    pipeline_connect_signature_in(p, "lo", "c", "lo");
    pipeline_connect_signature_in(p, "hi", "c", "hi");
    if (normalize) {
        node_1in(p, "n", "sigmoid");
        pipeline_connect(p, "c", "out", "n", "x");
        pipeline_connect_signature_out(p, "n", "out", "y");
    } else {
        pipeline_connect_signature_out(p, "c", "out", "y");
    }
    return p;
}

/* tpl_pv_npv_chain(years_param):  net_present_value-style pipeline. */
static Pipeline *tpl_pv_npv_chain(int variant) {
    char name[64]; snprintf(name, sizeof(name), "pv_npv_chain_%d", variant);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "cashflow", "rate", "years" };
    sig_n_in_1_out(p, 3, ins, "y");
    const char *fn[] = { "present", "rate", "periods" };
    node_named(p, "fv", "future_value", 3, fn);
    const char *pn[] = { "future", "rate", "periods" };
    node_named(p, "pv", "present_value", 3, pn);
    pipeline_connect_signature_in(p, "cashflow", "fv", "present");
    pipeline_connect_signature_in(p, "rate",     "fv", "rate");
    pipeline_connect_signature_in(p, "years",    "fv", "periods");
    pipeline_connect(p, "fv", "out", "pv", "future");
    pipeline_connect_signature_in(p, "rate",     "pv", "rate");
    pipeline_connect_signature_in(p, "years",    "pv", "periods");
    if (variant == 0) {
        pipeline_connect_signature_out(p, "pv", "out", "y");
    } else {
        /* variant 1: subtract original cashflow to get net. */
        node_2in(p, "net", "subtract");
        pipeline_connect(p, "pv", "out", "net", "x");
        pipeline_connect_signature_in(p, "cashflow", "net", "y");
        pipeline_connect_signature_out(p, "net", "out", "y");
    }
    return p;
}

/* tpl_distance_metrics(dim):  sum of distance_1d(a_i, b_i) for i in 1..dim,
 * then take square (Euclidean-squared style on absolute distances). */
static Pipeline *tpl_distance_metrics(int dim) {
    char name[64]; snprintf(name, sizeof(name), "distance_metrics_%d", dim);
    Pipeline *p = pipeline_create(name);
    int n_in = 2 * dim;
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < dim; i++) {
        snprintf(in_buf[2 * i],     sizeof(in_buf[0]), "a%d", i + 1);
        snprintf(in_buf[2 * i + 1], sizeof(in_buf[0]), "b%d", i + 1);
        in_names[2 * i]     = in_buf[2 * i];
        in_names[2 * i + 1] = in_buf[2 * i + 1];
    }
    sig_n_in_1_out(p, n_in, in_names, "y");
    const char *abn[] = { "a", "b" };
    for (int i = 0; i < dim; i++) {
        char d[16]; snprintf(d, sizeof(d), "d%d", i + 1);
        node_named(p, d, "distance_1d", 2, abn);
        char a[8], b[8];
        snprintf(a, sizeof(a), "a%d", i + 1);
        snprintf(b, sizeof(b), "b%d", i + 1);
        pipeline_connect_signature_in(p, a, d, "a");
        pipeline_connect_signature_in(p, b, d, "b");
    }
    if (dim == 1) {
        node_1in(p, "sq", "square");
        pipeline_connect(p, "d1", "out", "sq", "x");
        pipeline_connect_signature_out(p, "sq", "out", "y");
    } else {
        for (int i = 1; i < dim; i++) {
            char id[16]; snprintf(id, sizeof(id), "s%d", i);
            node_2in(p, id, "add");
            if (i == 1) {
                pipeline_connect(p, "d1", "out", id, "x");
            } else {
                char prev[16]; snprintf(prev, sizeof(prev), "s%d", i - 1);
                pipeline_connect(p, prev, "out", id, "x");
            }
            char d_next[16]; snprintf(d_next, sizeof(d_next), "d%d", i + 1);
            pipeline_connect(p, d_next, "out", id, "y");
        }
        char last[16]; snprintf(last, sizeof(last), "s%d", dim - 1);
        node_1in(p, "sq", "square");
        pipeline_connect(p, last, "out", "sq", "x");
        pipeline_connect_signature_out(p, "sq", "out", "y");
    }
    return p;
}

/* tpl_weighted_real(n):  weighted_avg-style chain over n (value, weight) pairs. */
static Pipeline *tpl_weighted_real(int n) {
    char name[64]; snprintf(name, sizeof(name), "weighted_real_%d", n);
    Pipeline *p = pipeline_create(name);
    int n_in = 2 * n;
    char in_buf[16][8];
    const char *in_names[16];
    for (int i = 0; i < n; i++) {
        snprintf(in_buf[2 * i],     sizeof(in_buf[0]), "v%d", i + 1);
        snprintf(in_buf[2 * i + 1], sizeof(in_buf[0]), "w%d", i + 1);
        in_names[2 * i]     = in_buf[2 * i];
        in_names[2 * i + 1] = in_buf[2 * i + 1];
    }
    sig_n_in_1_out(p, n_in, in_names, "y");
    /* Multiply each value by its weight. */
    for (int i = 0; i < n; i++) {
        char id[16]; snprintf(id, sizeof(id), "vw%d", i + 1);
        node_2in(p, id, "multiply");
        char v[8], w[8];
        snprintf(v, sizeof(v), "v%d", i + 1);
        snprintf(w, sizeof(w), "w%d", i + 1);
        pipeline_connect_signature_in(p, v, id, "x");
        pipeline_connect_signature_in(p, w, id, "y");
    }
    /* Sum products. */
    if (n == 1) {
        pipeline_connect_signature_out(p, "vw1", "out", "y");
        return p;
    }
    for (int i = 1; i < n; i++) {
        char id[16]; snprintf(id, sizeof(id), "ss%d", i);
        node_2in(p, id, "add");
        if (i == 1) {
            pipeline_connect(p, "vw1", "out", id, "x");
        } else {
            char prev[16]; snprintf(prev, sizeof(prev), "ss%d", i - 1);
            pipeline_connect(p, prev, "out", id, "x");
        }
        char vw[16]; snprintf(vw, sizeof(vw), "vw%d", i + 1);
        pipeline_connect(p, vw, "out", id, "y");
    }
    char last_sum[16]; snprintf(last_sum, sizeof(last_sum), "ss%d", n - 1);
    /* Sum of weights (denominator) — simple add chain. */
    if (n == 2) {
        node_2in(p, "wsum", "add");
        pipeline_connect_signature_in(p, "w1", "wsum", "x");
        pipeline_connect_signature_in(p, "w2", "wsum", "y");
    } else {
        for (int i = 1; i < n; i++) {
            char id[16]; snprintf(id, sizeof(id), "ws%d", i);
            node_2in(p, id, "add");
            if (i == 1) {
                pipeline_connect_signature_in(p, "w1", id, "x");
            } else {
                char prev[16]; snprintf(prev, sizeof(prev), "ws%d", i - 1);
                pipeline_connect(p, prev, "out", id, "x");
            }
            char w[8]; snprintf(w, sizeof(w), "w%d", i + 1);
            pipeline_connect_signature_in(p, w, id, "y");
        }
    }
    /* Final divide via percentage(part, whole) for naturalness. */
    const char *pn[] = { "part", "whole" };
    node_named(p, "wavg", "percentage", 2, pn);
    pipeline_connect(p, last_sum, "out", "wavg", "part");
    if (n == 2) {
        pipeline_connect(p, "wsum", "out", "wavg", "whole");
    } else {
        char wlast[16]; snprintf(wlast, sizeof(wlast), "ws%d", n - 1);
        pipeline_connect(p, wlast, "out", "wavg", "whole");
    }
    pipeline_connect_signature_out(p, "wavg", "out", "y");
    return p;
}

/* ============================================================
 *  Phase 11 — Structural-diversity templates.
 *
 *  The bimodal-failure pattern from Phase 8 + the negative results
 *  of Phases 9 (capacity) and 10 (arg-order paraphrases) say the
 *  remaining 12 wrong prompts need new graph topologies — not more
 *  paraphrases of existing graphs. Each family below introduces a
 *  composition shape the corpus didn't previously cover.
 * ============================================================ */

/* tpl_fib_fact_op(op): fibonacci(n) op factorial(n). Targets #7, #17.
 * Differs from tpl_fib_fact_blend: uses single sig input `n` shared
 * by both branches; emphasises the chained-then-combined topology. */
static Pipeline *tpl_fib_fact_op(const char *op) {
    char name[64]; snprintf(name, sizeof(name), "fib_fact_op_%s", op);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "n" };
    sig_n_in_1_out(p, 1, ins, "y");
    node_1in(p, "fib",  "fibonacci");
    node_1in(p, "fact", "factorial");
    node_2in(p, "out_op", op);
    pipeline_connect_signature_in(p, "n", "fib",  "x");
    pipeline_connect_signature_in(p, "n", "fact", "x");
    pipeline_connect(p, "fib",  "out", "out_op", "x");
    pipeline_connect(p, "fact", "out", "out_op", "y");
    pipeline_connect_signature_out(p, "out_op", "out", "y");
    return p;
}

/* tpl_distance_midpoint(combine_op): distance_1d(a,b) op midpoint(a,b).
 * Targets #15 ("distance between two readings combined with their midpoint"). */
static Pipeline *tpl_distance_midpoint(const char *op) {
    char name[64]; snprintf(name, sizeof(name), "distance_midpoint_%s", op);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "a", "b" };
    sig_n_in_1_out(p, 2, ins, "y");
    const char *abnames[] = { "a", "b" };
    node_named(p, "dist", "distance_1d", 2, abnames);
    node_named(p, "mid",  "midpoint",    2, abnames);
    node_2in(p, "comb",  op);
    pipeline_connect_signature_in(p, "a", "dist", "a");
    pipeline_connect_signature_in(p, "b", "dist", "b");
    pipeline_connect_signature_in(p, "a", "mid",  "a");
    pipeline_connect_signature_in(p, "b", "mid",  "b");
    pipeline_connect(p, "dist", "out", "comb", "x");
    pipeline_connect(p, "mid",  "out", "comb", "y");
    pipeline_connect_signature_out(p, "comb", "out", "y");
    return p;
}

/* tpl_apply_tax_chain(extra_op): apply_tax(gross, rate) extra_op constant.
 * Targets #6 ("take home pay from gross income at federal tax rate").
 * extra_op ∈ {add, subtract, multiply}. Adds a 3rd input via a constant-
 * like injection that the model has to wire through.                  */
static Pipeline *tpl_apply_tax_chain(const char *extra_op) {
    char name[64]; snprintf(name, sizeof(name), "apply_tax_chain_%s", extra_op);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "gross", "rate", "delta" };
    sig_n_in_1_out(p, 3, ins, "y");
    const char *anames[] = { "amount", "rate" };
    node_named(p, "net", "apply_tax", 2, anames);
    node_2in(p, "adj", extra_op);
    pipeline_connect_signature_in(p, "gross", "net", "amount");
    pipeline_connect_signature_in(p, "rate",  "net", "rate");
    pipeline_connect(p, "net", "out", "adj", "x");
    pipeline_connect_signature_in(p, "delta", "adj", "y");
    pipeline_connect_signature_out(p, "adj", "out", "y");
    return p;
}

/* tpl_clamped_unary_then_op(unary, op): clamp(unary(x), lo, hi) op extra.
 * Targets #4 (sigmoid+clamp) and #20 (sigmoid+clamp variants). */
static Pipeline *tpl_clamped_unary_then_op(const char *unary, const char *op) {
    char name[80]; snprintf(name, sizeof(name), "clamped_%s_%s", unary, op);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "x", "lo", "hi", "k" };
    sig_n_in_1_out(p, 4, ins, "y");
    node_1in(p, "u", unary);
    const char *cnames[] = { "x", "lo", "hi" };
    node_named(p, "c", "clamp", 3, cnames);
    node_2in(p, "post", op);
    pipeline_connect_signature_in(p, "x", "u", "x");
    pipeline_connect(p, "u", "out", "c", "x");
    pipeline_connect_signature_in(p, "lo", "c", "lo");
    pipeline_connect_signature_in(p, "hi", "c", "hi");
    pipeline_connect(p, "c", "out", "post", "x");
    pipeline_connect_signature_in(p, "k", "post", "y");
    pipeline_connect_signature_out(p, "post", "out", "y");
    return p;
}

/* tpl_compound_then(op): compound(P, r, n) op P. Targets variants of #2, #19. */
static Pipeline *tpl_compound_then(const char *op) {
    char name[64]; snprintf(name, sizeof(name), "compound_then_%s", op);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "principal", "rate", "years" };
    sig_n_in_1_out(p, 3, ins, "y");
    const char *cnames[] = { "principal", "rate", "periods" };
    node_named(p, "amt", "compound", 3, cnames);
    node_2in(p, "comb", op);
    pipeline_connect_signature_in(p, "principal", "amt", "principal");
    pipeline_connect_signature_in(p, "rate",      "amt", "rate");
    pipeline_connect_signature_in(p, "years",     "amt", "periods");
    pipeline_connect(p, "amt", "out", "comb", "x");
    pipeline_connect_signature_in(p, "principal", "comb", "y");
    pipeline_connect_signature_out(p, "comb", "out", "y");
    return p;
}

static Pipeline *w_fib_fact_op(void *ctx)         { return tpl_fib_fact_op((const char *)ctx); }
static Pipeline *w_distance_midpoint(void *ctx)   { return tpl_distance_midpoint((const char *)ctx); }
static Pipeline *w_apply_tax_chain(void *ctx)     { return tpl_apply_tax_chain((const char *)ctx); }
static Pipeline *w_clamped_unary_then_op(void *ctx) {
    const char *u = ((const char **)ctx)[0];
    const char *op = ((const char **)ctx)[1];
    return tpl_clamped_unary_then_op(u, op);
}
static Pipeline *w_compound_then(void *ctx)       { return tpl_compound_then((const char *)ctx); }

/* tpl_micro_unary(prim): single-node unary primitive graph.
 *   y = prim(x)                                                       */
static Pipeline *tpl_micro_unary(const char *prim) {
    char name[64]; snprintf(name, sizeof(name), "micro_%s", prim);
    Pipeline *p = pipeline_create(name);
    const char *ins[] = { "x" };
    sig_n_in_1_out(p, 1, ins, "y");
    node_1in(p, "n", prim);
    pipeline_connect_signature_in(p, "x", "n", "x");
    pipeline_connect_signature_out(p, "n", "out", "y");
    return p;
}

/* tpl_micro_binary(prim, in_names_kind):
 *   y = prim(x, y)  with caller-chosen port names.
 * port_kind 0: {x, y}, 1: {a, b}, 2: {amount, rate}, 3: {part, whole},
 * 4: {weight, height}, 5: {price, rate}, 6: {cost, rate},
 * 7: {base, exp}, 8: {m, v}                                           */
static Pipeline *tpl_micro_binary(const char *prim, int port_kind) {
    char name[80]; snprintf(name, sizeof(name), "micro_%s_%d", prim, port_kind);
    Pipeline *p = pipeline_create(name);
    const char *port_pairs[][2] = {
        {"x", "y"}, {"a", "b"}, {"amount", "rate"}, {"part", "whole"},
        {"weight", "height"}, {"price", "rate"}, {"cost", "rate"},
        {"base", "exp"}, {"m", "v"}
    };
    const char *p1 = port_pairs[port_kind][0];
    const char *p2 = port_pairs[port_kind][1];
    const char *ins[] = { p1, p2 };
    sig_n_in_1_out(p, 2, ins, "y");
    if (port_kind == 0) {
        node_2in(p, "n", prim);
        pipeline_connect_signature_in(p, p1, "n", "x");
        pipeline_connect_signature_in(p, p2, "n", "y");
    } else {
        const char *node_in_names[] = { p1, p2 };
        node_named(p, "n", prim, 2, node_in_names);
        pipeline_connect_signature_in(p, p1, "n", p1);
        pipeline_connect_signature_in(p, p2, "n", p2);
    }
    pipeline_connect_signature_out(p, "n", "out", "y");
    return p;
}

/* tpl_micro_ternary(prim, port_kind):
 *   y = prim(a, b, c)                                                  */
static Pipeline *tpl_micro_ternary(const char *prim, int port_kind) {
    char name[80]; snprintf(name, sizeof(name), "micro_%s_%d", prim, port_kind);
    Pipeline *p = pipeline_create(name);
    const char *port_triples[][3] = {
        {"x", "lo", "hi"},                  /* clamp */
        {"a", "b", "t"},                    /* lerp */
        {"principal", "rate", "periods"},   /* compound */
        {"present", "rate", "periods"},     /* future_value */
        {"future", "rate", "periods"}       /* present_value */
    };
    const char *p1 = port_triples[port_kind][0];
    const char *p2 = port_triples[port_kind][1];
    const char *p3 = port_triples[port_kind][2];
    const char *ins[] = { p1, p2, p3 };
    sig_n_in_1_out(p, 3, ins, "y");
    const char *node_ins[] = { p1, p2, p3 };
    node_named(p, "n", prim, 3, node_ins);
    pipeline_connect_signature_in(p, p1, "n", p1);
    pipeline_connect_signature_in(p, p2, "n", p2);
    pipeline_connect_signature_in(p, p3, "n", p3);
    pipeline_connect_signature_out(p, "n", "out", "y");
    return p;
}

/* Micro template wrapper-context structs (heap-stable for ADD3). */
typedef struct { const char *prim; int port_kind; } MicroBinCtx;
typedef struct { const char *prim; int port_kind; } MicroTerCtx;

static Pipeline *w_micro_unary(void *ctx) { return tpl_micro_unary((const char *)ctx); }
static Pipeline *w_micro_binary(void *ctx) {
    MicroBinCtx *m = (MicroBinCtx *)ctx; return tpl_micro_binary(m->prim, m->port_kind);
}
static Pipeline *w_micro_ternary(void *ctx) {
    MicroTerCtx *m = (MicroTerCtx *)ctx; return tpl_micro_ternary(m->prim, m->port_kind);
}

/* Wrappers for the Phase 4 templates. */
static Pipeline *w_clamped_op(void *ctx) { return tpl_clamped_op((const char *)ctx); }
static Pipeline *w_taxed_total(void *ctx) { return tpl_taxed_total((int)(intptr_t)ctx); }
static Pipeline *w_savings_pipeline(void *ctx) { return tpl_savings_pipeline((int)(intptr_t)ctx); }
static Pipeline *w_compound_chain(void *ctx) { return tpl_compound_chain((int)(intptr_t)ctx); }
static Pipeline *w_gcd_chain(void *ctx) { return tpl_gcd_chain((int)(intptr_t)ctx); }
static Pipeline *w_fib_fact_blend(void *ctx) { return tpl_fib_fact_blend((const char *)ctx); }
static Pipeline *w_bmi_classified(void *ctx) { return tpl_bmi_classified((int)(intptr_t)ctx); }
static Pipeline *w_pv_npv_chain(void *ctx) { return tpl_pv_npv_chain((int)(intptr_t)ctx); }
static Pipeline *w_distance_metrics(void *ctx) { return tpl_distance_metrics((int)(intptr_t)ctx); }
static Pipeline *w_weighted_real(void *ctx) { return tpl_weighted_real((int)(intptr_t)ctx); }

/* ============================================================
 *  Catalog of all examples (templates expanded)
 * ============================================================ */

typedef struct {
    char *prompt;          /* heap; freed at end */
    Pipeline *(*build)(void *ctx);
    void *ctx_a;           /* small int parameters; cast as needed */
    void *ctx_b;
    void *ctx_c;
} CorpusEntry;

/* Wrappers so each template can be called via a single-pointer build fn. */
static Pipeline *w_chain(void *ctx) {
    const char *prim = ((const char **)ctx)[0];
    int n = (int)(intptr_t)((void **)ctx)[1];
    return tpl_chain(prim, n);
}
static Pipeline *w_fanout_combine(void *ctx) {
    const char *u = ((const char **)ctx)[0];
    const char *b = ((const char **)ctx)[1];
    int n = (int)(intptr_t)((void **)ctx)[2];
    return tpl_fanout_combine(u, b, n);
}
static Pipeline *w_polynomial(void *ctx) { int d = (int)(intptr_t)ctx; return tpl_polynomial(d); }
static Pipeline *w_distance(void *ctx) { int d = (int)(intptr_t)ctx; return tpl_distance_squared(d); }
static Pipeline *w_dot(void *ctx) { int d = (int)(intptr_t)ctx; return tpl_dot_product(d); }
static Pipeline *w_mean(void *ctx) { int n = (int)(intptr_t)ctx; return tpl_mean(n); }
static Pipeline *w_weighted(void *ctx) { int n = (int)(intptr_t)ctx; return tpl_weighted_combine(n); }
static Pipeline *w_axpy(void *ctx) {
    const char *post = ((const char **)ctx)[0];
    int depth = (int)(intptr_t)((void **)ctx)[1];
    return tpl_axpy_chain(post, depth);
}
static Pipeline *w_lerp(void *ctx) { int n = (int)(intptr_t)ctx; return tpl_lerp(n); }
static Pipeline *w_range(void *ctx) { int n = (int)(intptr_t)ctx; return tpl_range(n); }

/* Build the entire corpus. Each entry's prompt is heap-allocated; caller frees. */
static CorpusEntry *build_catalog(int *out_count) {
    int cap = 256;
    CorpusEntry *cat = (CorpusEntry *)calloc((size_t)cap, sizeof(CorpusEntry));
    int n = 0;
    /* Macro to add an entry; ctx fields are stuffed with stable pointers
     * via small heap allocations rather than stack pointers. */
#define ADD3(prompt_str, fn, a, b, c) do {                                     \
    cat[n].prompt = strdup(prompt_str);                                        \
    cat[n].build = (fn);                                                       \
    cat[n].ctx_a = (a); cat[n].ctx_b = (b); cat[n].ctx_c = (c);                \
    n++;                                                                       \
} while (0)

    /* --- Family 1: chain(prim, n) ---  prim ∈ {add, multiply, max, min}, n ∈ {2..8} */
    static const char *prim_set[] = { "add", "multiply", "max", "min" };
    for (int p = 0; p < 4; p++) {
        for (int nn = 2; nn <= 8; nn++) {
            char prompt[128];
            snprintf(prompt, sizeof(prompt),
                     "// %s of %d integers", prim_set[p], nn);
            const char **ctx = (const char **)calloc(2, sizeof(void *));
            ctx[0] = prim_set[p];
            ((void **)ctx)[1] = (void *)(intptr_t)nn;
            ADD3(prompt, w_chain, (void *)ctx, NULL, NULL);
        }
    }

    /* --- Family 2: fanout_combine(unary, binary, n) --- */
    static const char *unaries[] = { "negate", "abs" };
    static const char *binaries[] = { "add", "multiply" };
    for (int u = 0; u < 2; u++) {
        for (int b = 0; b < 2; b++) {
            for (int nn = 2; nn <= 4; nn++) {
                char prompt[160];
                snprintf(prompt, sizeof(prompt),
                         "// %s each of %d inputs then %s them",
                         unaries[u], nn, binaries[b]);
                const char **ctx = (const char **)calloc(3, sizeof(void *));
                ctx[0] = unaries[u]; ctx[1] = binaries[b];
                ((void **)ctx)[2] = (void *)(intptr_t)nn;
                ADD3(prompt, w_fanout_combine, (void *)ctx, NULL, NULL);
            }
        }
    }

    /* --- Family 3: polynomial(d) --- d ∈ {1..7} */
    for (int d = 1; d <= 7; d++) {
        char prompt[128];
        snprintf(prompt, sizeof(prompt),
                 "// polynomial of degree %d evaluated at x", d);
        ADD3(prompt, w_polynomial, (void *)(intptr_t)d, NULL, NULL);
    }

    /* --- Family 4: distance_squared(dim) --- dim ∈ {1..6} */
    for (int d = 1; d <= 6; d++) {
        char prompt[128];
        snprintf(prompt, sizeof(prompt),
                 "// squared euclidean distance in %d dimensions", d);
        ADD3(prompt, w_distance, (void *)(intptr_t)d, NULL, NULL);
    }

    /* --- Family 5: dot_product(dim) --- dim ∈ {2..8} */
    for (int d = 2; d <= 8; d++) {
        char prompt[128];
        snprintf(prompt, sizeof(prompt),
                 "// dot product of two %d-dimensional vectors", d);
        ADD3(prompt, w_dot, (void *)(intptr_t)d, NULL, NULL);
    }

    /* --- Family 6: mean_n(n) --- n ∈ {2..8} */
    for (int nn = 2; nn <= 8; nn++) {
        char prompt[128];
        snprintf(prompt, sizeof(prompt), "// arithmetic mean of %d integers", nn);
        ADD3(prompt, w_mean, (void *)(intptr_t)nn, NULL, NULL);
    }

    /* --- Family 7: weighted_combine(n) --- n ∈ {2..6} */
    for (int nn = 2; nn <= 6; nn++) {
        char prompt[128];
        snprintf(prompt, sizeof(prompt),
                 "// weighted sum of %d (weight, value) pairs", nn);
        ADD3(prompt, w_weighted, (void *)(intptr_t)nn, NULL, NULL);
    }

    /* --- Family 8: axpy_then_op(post, depth) --- */
    static const char *post_ops[] = { "negate", "abs" };
    for (int p = 0; p < 2; p++) {
        for (int dep = 1; dep <= 3; dep++) {
            char prompt[160];
            snprintf(prompt, sizeof(prompt),
                     "// %d-stage axpy chain a*x+y then %s the result",
                     dep, post_ops[p]);
            const char **ctx = (const char **)calloc(2, sizeof(void *));
            ctx[0] = post_ops[p];
            ((void **)ctx)[1] = (void *)(intptr_t)dep;
            ADD3(prompt, w_axpy, (void *)ctx, NULL, NULL);
        }
    }

    /* --- Family 9: lerp_n(n) --- n ∈ {2..4} */
    for (int nn = 2; nn <= 4; nn++) {
        char prompt[128];
        snprintf(prompt, sizeof(prompt),
                 "// chained linear interpolation across %d waypoints", nn);
        ADD3(prompt, w_lerp, (void *)(intptr_t)nn, NULL, NULL);
    }

    /* --- Family 10: range_n(n) --- n ∈ {2..5} */
    for (int nn = 2; nn <= 5; nn++) {
        char prompt[128];
        snprintf(prompt, sizeof(prompt),
                 "// range (max minus min) of %d integers", nn);
        ADD3(prompt, w_range, (void *)(intptr_t)nn, NULL, NULL);
    }

    /* ============================================================
     *  Phase 4 — Real-primitive seed graphs (paraphrased prompts).
     *  Each seed graph appears with 3 different natural-English prompts
     *  so the model learns to map domain wording → primitive composition.
     * ============================================================ */
#define ADD_SEED3(p1, p2, p3, fn) do {                                     \
    ADD3(p1, w_seed, (void *)(SeedFn)(fn), NULL, NULL);                    \
    ADD3(p2, w_seed, (void *)(SeedFn)(fn), NULL, NULL);                    \
    ADD3(p3, w_seed, (void *)(SeedFn)(fn), NULL, NULL);                    \
} while (0)

    ADD_SEED3(
        "// double the first value triple the second and add",
        "// sum doubled a and tripled b",
        "// add double of a to triple of b",
        seed_sum_results);

    ADD_SEED3(
        "// compute compound interest earned on principal at rate over years",
        "// money compounded at interest rate over years minus principal",
        "// interest amount after compound growth of principal",
        seed_compound_interest);

    ADD_SEED3(
        "// distance between two points plus their midpoint",
        "// add one dimensional distance and midpoint of a and b",
        "// combine distance and midpoint of two values",
        seed_analyze_two_points);

    ADD_SEED3(
        "// average two numbers then clamp between bounds",
        "// clamped mean of a and b within lo and hi",
        "// take average of a and b and limit to range",
        seed_clamped_average);

    ADD_SEED3(
        "// absolute difference of a and b",
        "// magnitude of a minus b",
        "// distance from a to b without sign",
        seed_abs_difference);

    ADD_SEED3(
        "// tax on price after applying a discount",
        "// discounted then taxed amount",
        "// compute tax amount on price after a discount rate",
        seed_discounted_tax);

    ADD_SEED3(
        "// total cost of price including tax",
        "// price plus tax amount on price",
        "// gross total after adding sales tax",
        seed_total_with_tax);

    ADD_SEED3(
        "// take home pay from gross income at tax rate",
        "// net income after applying tax rate",
        "// post tax pay given gross and rate",
        seed_net_pay);

    ADD_SEED3(
        "// savings rate as percentage of income",
        "// fraction saved out of income after expenses",
        "// percentage of income left after expenses",
        seed_savings_rate);

    ADD_SEED3(
        "// fibonacci of n times factorial of n",
        "// product of fibonacci and factorial of n",
        "// multiply fib n by fact n",
        seed_fib_fact_product);

    ADD_SEED3(
        "// net present value of cashflow at rate over years",
        "// present value of future value of cashflow",
        "// discount the future value of a cashflow back to present",
        seed_net_present_value);

    ADD_SEED3(
        "// sigmoid of x clamped within lo and hi",
        "// bounded sigmoid output between lo and hi",
        "// clip sigmoid x to range lo hi",
        seed_clamped_sigmoid);

    ADD_SEED3(
        "// relu of x scaled by a factor",
        "// rectified linear unit times scale",
        "// scaled rectified output of x",
        seed_scaled_relu);

    ADD_SEED3(
        "// gcd of a and b multiplied by both a and b",
        "// product of gcd a b times a times b",
        "// multiply greatest common divisor by a and b",
        seed_gcd_product);

    ADD_SEED3(
        "// bmi of weight and height clamped between lo and hi",
        "// body mass index limited within bounds",
        "// classify bmi by clamping into a range",
        seed_bmi_classified);

#undef ADD_SEED3

    /* ============================================================
     *  Phase 4 — Real-primitive parametric families.
     * ============================================================ */

    /* tpl_clamped_op(unary_op): unary then clamp within bounds. */
    static const char *clamp_ops[] = {
        "sigmoid", "relu", "abs_val", "square", "double_val", "triple_val"
    };
    for (size_t op = 0; op < sizeof(clamp_ops) / sizeof(clamp_ops[0]); op++) {
        char prompt[160];
        snprintf(prompt, sizeof(prompt),
                 "// apply %s to x then clamp result within lo and hi",
                 clamp_ops[op]);
        ADD3(prompt, w_clamped_op, (void *)clamp_ops[op], NULL, NULL);
    }

    /* tpl_taxed_total: invoice total = price * qty + tax. */
    for (int qf = 0; qf < 2; qf++) {
        char prompt[160];
        snprintf(prompt, sizeof(prompt),
                 "// invoice total of %s times %s plus tax amount at rate",
                 qf ? "quantity" : "price",
                 qf ? "price" : "quantity");
        ADD3(prompt, w_taxed_total, (void *)(intptr_t)qf, NULL, NULL);
        char prompt2[160];
        snprintf(prompt2, sizeof(prompt2),
                 "// %sgross billing including sales tax on units sold",
                 qf ? "" : "");
        ADD3(prompt2, w_taxed_total, (void *)(intptr_t)qf, NULL, NULL);
    }

    /* tpl_savings_pipeline(n_expenses): sum n expenses, subtract from income, percent. */
    for (int n_e = 1; n_e <= 4; n_e++) {
        char prompt[160];
        snprintf(prompt, sizeof(prompt),
                 "// savings rate after subtracting %d expense items from income",
                 n_e);
        ADD3(prompt, w_savings_pipeline, (void *)(intptr_t)n_e, NULL, NULL);
    }

    /* tpl_compound_chain(periods): interest earned. */
    for (int per = 2; per <= 6; per++) {
        char prompt[160];
        snprintf(prompt, sizeof(prompt),
                 "// interest earned on principal compounded at rate for %d years",
                 per);
        ADD3(prompt, w_compound_chain, (void *)(intptr_t)per, NULL, NULL);
    }

    /* tpl_gcd_chain(depth): gcd chain * k. */
    for (int dep = 1; dep <= 4; dep++) {
        char prompt[160];
        snprintf(prompt, sizeof(prompt),
                 "// gcd of %d integers multiplied by k",
                 dep + 1);
        ADD3(prompt, w_gcd_chain, (void *)(intptr_t)dep, NULL, NULL);
    }

    /* tpl_fib_fact_blend(op): combine fibonacci(n) and factorial(n). */
    static const char *blend_ops[] = { "add", "multiply", "max", "min" };
    for (size_t op = 0; op < sizeof(blend_ops) / sizeof(blend_ops[0]); op++) {
        char prompt[160];
        snprintf(prompt, sizeof(prompt),
                 "// %s of fibonacci of n and factorial of n",
                 blend_ops[op]);
        ADD3(prompt, w_fib_fact_blend, (void *)blend_ops[op], NULL, NULL);
    }

    /* tpl_bmi_classified(normalize). */
    for (int nor = 0; nor <= 1; nor++) {
        char prompt[160];
        if (nor) {
            snprintf(prompt, sizeof(prompt),
                     "// bmi from weight and height clamped to bounds and sigmoid normalized");
        } else {
            snprintf(prompt, sizeof(prompt),
                     "// bmi from weight and height clamped to lo and hi range");
        }
        ADD3(prompt, w_bmi_classified, (void *)(intptr_t)nor, NULL, NULL);
    }

    /* tpl_pv_npv_chain(variant). */
    for (int var = 0; var <= 1; var++) {
        char prompt[160];
        if (var) {
            snprintf(prompt, sizeof(prompt),
                     "// net of present value of future value minus original cashflow");
        } else {
            snprintf(prompt, sizeof(prompt),
                     "// present value of future value of cashflow at rate over years");
        }
        ADD3(prompt, w_pv_npv_chain, (void *)(intptr_t)var, NULL, NULL);
    }

    /* tpl_distance_metrics(dim). */
    for (int d = 1; d <= 4; d++) {
        char prompt[160];
        snprintf(prompt, sizeof(prompt),
                 "// squared sum of one dimensional distances across %d coordinate pairs",
                 d);
        ADD3(prompt, w_distance_metrics, (void *)(intptr_t)d, NULL, NULL);
    }

    /* tpl_weighted_real(n). */
    for (int nn = 2; nn <= 5; nn++) {
        char prompt[160];
        snprintf(prompt, sizeof(prompt),
                 "// weighted average of %d value weight pairs",
                 nn);
        ADD3(prompt, w_weighted_real, (void *)(intptr_t)nn, NULL, NULL);
    }

    /* ============================================================
     *  Phase 11 — Structural-diversity templates.
     *
     *  Five new graph topologies the corpus didn't previously cover.
     *  Each targets one of the robustly-wrong held-out cases from
     *  Phase 8/9/10 by introducing the specific composition shape the
     *  model has to produce. Paraphrases of *new* graphs (not new
     *  paraphrases of existing graphs).
     * ============================================================ */

    /* tpl_fib_fact_op(op) — fibonacci(n) op factorial(n). */
    static const char *fib_fact_ops[] = { "add", "multiply", "max", "min", "subtract" };
    static const char *fib_fact_phrases[] = {
        "fibonacci of n combined with factorial of n by",
        "fib of n and fact of n then take",
        "blend fibonacci and factorial of n with",
        "fibonacci then factorial then",
    };
    for (size_t op_i = 0; op_i < sizeof(fib_fact_ops) / sizeof(fib_fact_ops[0]); op_i++) {
        for (size_t ph = 0; ph < sizeof(fib_fact_phrases) / sizeof(fib_fact_phrases[0]); ph++) {
            char prompt[160];
            snprintf(prompt, sizeof(prompt), "// %s %s",
                     fib_fact_phrases[ph], fib_fact_ops[op_i]);
            ADD3(prompt, w_fib_fact_op, (void *)fib_fact_ops[op_i], NULL, NULL);
        }
    }

    /* tpl_distance_midpoint(op) — distance_1d(a,b) op midpoint(a,b). */
    static const char *dm_ops[] = { "add", "multiply", "subtract" };
    static const char *dm_phrases[] = {
        "distance between two readings combined with their midpoint by",
        "one dimensional distance and midpoint of a and b then",
        "distance plus midpoint composed with",
    };
    for (size_t op_i = 0; op_i < sizeof(dm_ops) / sizeof(dm_ops[0]); op_i++) {
        for (size_t ph = 0; ph < sizeof(dm_phrases) / sizeof(dm_phrases[0]); ph++) {
            char prompt[160];
            snprintf(prompt, sizeof(prompt), "// %s %s",
                     dm_phrases[ph], dm_ops[op_i]);
            ADD3(prompt, w_distance_midpoint, (void *)dm_ops[op_i], NULL, NULL);
        }
    }

    /* tpl_apply_tax_chain(extra) — apply_tax then add/sub/mul a delta. */
    static const char *atc_ops[] = { "add", "subtract", "multiply" };
    static const char *atc_phrases[] = {
        "take home pay from gross at rate then adjust by delta with",
        "net pay after tax then composed with delta by",
        "apply tax to gross at rate then take home with delta and",
    };
    for (size_t op_i = 0; op_i < sizeof(atc_ops) / sizeof(atc_ops[0]); op_i++) {
        for (size_t ph = 0; ph < sizeof(atc_phrases) / sizeof(atc_phrases[0]); ph++) {
            char prompt[160];
            snprintf(prompt, sizeof(prompt), "// %s %s",
                     atc_phrases[ph], atc_ops[op_i]);
            ADD3(prompt, w_apply_tax_chain, (void *)atc_ops[op_i], NULL, NULL);
        }
    }

    /* tpl_clamped_unary_then_op(unary, op) — unary → clamp → op. */
    static const char *cuto_unaries[] = { "sigmoid", "relu", "abs_val" };
    static const char *cuto_ops[] = { "add", "multiply" };
    for (size_t u = 0; u < sizeof(cuto_unaries) / sizeof(cuto_unaries[0]); u++) {
        for (size_t op_i = 0; op_i < sizeof(cuto_ops) / sizeof(cuto_ops[0]); op_i++) {
            char prompt[160];
            snprintf(prompt, sizeof(prompt),
                     "// %s of x clamped between lo and hi then %s with k",
                     cuto_unaries[u], cuto_ops[op_i]);
            const char **ctx = (const char **)calloc(2, sizeof(const char *));
            ctx[0] = cuto_unaries[u]; ctx[1] = cuto_ops[op_i];
            ADD3(prompt, w_clamped_unary_then_op, (void *)ctx, NULL, NULL);
        }
    }

    /* tpl_compound_then(op) — compound(P, r, n) op P. */
    static const char *ct_ops[] = { "subtract", "add", "multiply", "divide" };
    static const char *ct_phrases[] = {
        "compound principal at rate over years then",
        "amount after compound growth of principal then",
        "final balance after compound then",
    };
    for (size_t op_i = 0; op_i < sizeof(ct_ops) / sizeof(ct_ops[0]); op_i++) {
        for (size_t ph = 0; ph < sizeof(ct_phrases) / sizeof(ct_phrases[0]); ph++) {
            char prompt[160];
            snprintf(prompt, sizeof(prompt), "// %s %s with original principal",
                     ct_phrases[ph], ct_ops[op_i]);
            ADD3(prompt, w_compound_then, (void *)ct_ops[op_i], NULL, NULL);
        }
    }

    /* ============================================================
     *  Phase 12 — Lexical anchoring for primitive selection.
     *
     *  Phase 11 broke the topology barrier (model emits 3-node fib+
     *  fact+combine graphs) but defaulted to wrong combiner choices
     *  ("min" for "multiplied by"). The training prompts used bare
     *  verbs ("multiply", "add"); held-out uses inflected forms
     *  ("multiplied by", "by adding"). These paraphrases lock the
     *  inflected forms to the correct primitive.
     * ============================================================ */

    /* fib_fact_op with held-out exact verb forms. */
    ADD3("// fibonacci of n multiplied by factorial of n",
         w_fib_fact_op, (void *)"multiply", NULL, NULL);
    ADD3("// fibonacci of n times factorial of n",
         w_fib_fact_op, (void *)"multiply", NULL, NULL);
    ADD3("// product of fibonacci and factorial of n",
         w_fib_fact_op, (void *)"multiply", NULL, NULL);
    ADD3("// multiply fibonacci of n by factorial of n",
         w_fib_fact_op, (void *)"multiply", NULL, NULL);

    ADD3("// fibonacci of n combined with factorial of n by adding",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// fibonacci of n added to factorial of n",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// sum of fibonacci of n and factorial of n",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// fibonacci plus factorial of n",
         w_fib_fact_op, (void *)"add", NULL, NULL);

    /* distance_midpoint with held-out exact verb forms. */
    ADD3("// distance between two readings combined with their midpoint",
         w_distance_midpoint, (void *)"add", NULL, NULL);
    ADD3("// distance plus midpoint of a and b",
         w_distance_midpoint, (void *)"add", NULL, NULL);
    ADD3("// add distance of a and b to their midpoint",
         w_distance_midpoint, (void *)"add", NULL, NULL);
    ADD3("// distance combined with midpoint by adding",
         w_distance_midpoint, (void *)"add", NULL, NULL);

    /* apply_tax_chain with subtract for "reduced by" / "minus". */
    ADD3("// take home pay from gross income at federal tax rate then minus delta",
         w_apply_tax_chain, (void *)"subtract", NULL, NULL);
    ADD3("// gross income reduced by tax liability and delta",
         w_apply_tax_chain, (void *)"subtract", NULL, NULL);

    /* compound_then with subtract for "minus original" / "interest earned". */
    ADD3("// interest earned by subtracting principal from compounded value",
         w_compound_then, (void *)"subtract", NULL, NULL);
    ADD3("// compound minus original principal yields interest",
         w_compound_then, (void *)"subtract", NULL, NULL);

    /* ============================================================
     *  Phase 13 — Held-out lexical bridge.
     *
     *  ~25 paraphrases targeting the 3 failure buckets identified in
     *  Phase 12's per-prompt analysis: gerund anchoring (#17), novel
     *  vocabulary (#1, #14, #20), and exact-phrase coverage (#2, #4,
     *  #5, #6, #12). All anchored to existing seed graphs / templates.
     *  No new templates, natives, or references.
     * ============================================================ */

    /* --- Bucket A: gerund anchoring for #17 ("by adding"). ---
     * Phase 12 added 1 "by adding" paraphrase; bring total to 4 to match
     * the count of "multiply"-form examples (which #7 nailed). */
    ADD3("// fibonacci of n combined with factorial of n by adding them",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// adding fibonacci of n and factorial of n",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// fib of n with fact of n by adding the results",
         w_fib_fact_op, (void *)"add", NULL, NULL);

    /* --- Bucket B: novel vocabulary bridges for #1, #14, #20. --- */

    /* #1 "body mass index ... limit it inside lo and hi" — seed_bmi_classified */
    ADD3("// body mass index from weight and height limit it inside lo and hi bounds",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);
    ADD3("// body mass index computed from weight and height kept inside lo hi",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);
    ADD3("// bmi from weight and height limit inside lo and hi",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);

    /* #14 "axes" + "squared" — tpl_distance_metrics(2). */
    ADD3("// total of distances across two coordinate axes squared",
         w_distance_metrics, (void *)(intptr_t)2, NULL, NULL);
    ADD3("// sum of distances across two axes then squared",
         w_distance_metrics, (void *)(intptr_t)2, NULL, NULL);
    ADD3("// total distance over two coordinate pairs squared",
         w_distance_metrics, (void *)(intptr_t)2, NULL, NULL);

    /* #20 "normalised by clamping" + "bounded range" — seed_clamped_sigmoid */
    ADD3("// sigmoid of x normalised by clamping into a bounded range",
         w_seed, (void *)(SeedFn)seed_clamped_sigmoid, NULL, NULL);
    ADD3("// sigmoid output normalised via clamp inside lo and hi",
         w_seed, (void *)(SeedFn)seed_clamped_sigmoid, NULL, NULL);
    ADD3("// normalise sigmoid of x by clamping to bounds",
         w_seed, (void *)(SeedFn)seed_clamped_sigmoid, NULL, NULL);

    /* --- Bucket C: held-out exact-phrase paraphrases. --- */

    /* #2 "interest gained on an investment" — seed_compound_interest. */
    ADD3("// interest gained on an investment when principal compounds at rate over years",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);
    ADD3("// interest gained on principal compounded at rate r over n years",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);

    /* #4 "limit the output of a sigmoid neuron" — seed_clamped_sigmoid. */
    ADD3("// limit the output of a sigmoid neuron to a low high range",
         w_seed, (void *)(SeedFn)seed_clamped_sigmoid, NULL, NULL);
    ADD3("// limit sigmoid output of x to lo high range",
         w_seed, (void *)(SeedFn)seed_clamped_sigmoid, NULL, NULL);

    /* #5 "gcd ... scaled by a coefficient k" — tpl_gcd_chain(1). */
    ADD3("// greatest common divisor of two numbers scaled by a coefficient k",
         w_gcd_chain, (void *)(intptr_t)1, NULL, NULL);
    ADD3("// gcd of two numbers times a coefficient k",
         w_gcd_chain, (void *)(intptr_t)1, NULL, NULL);

    /* #6 "take home pay from gross income" — seed_net_pay (apply_tax). */
    ADD3("// take home pay from gross income at federal tax rate",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// take home pay equals gross minus federal tax",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// federal take home pay from gross at rate",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);

    /* #12 "tax due on a price after a discount" — seed_discounted_tax. */
    ADD3("// tax due on a price after a discount has been applied",
         w_seed, (void *)(SeedFn)seed_discounted_tax, NULL, NULL);
    ADD3("// tax due after price has been discounted",
         w_seed, (void *)(SeedFn)seed_discounted_tax, NULL, NULL);
    ADD3("// discounted price then tax on it",
         w_seed, (void *)(SeedFn)seed_discounted_tax, NULL, NULL);

    /* ============================================================
     *  Phase 14 — Aggressive oversampling for the last 5 wrong prompts.
     *
     *  Phase 13 lifted the headline 50% → 75% but 5 prompts still fail:
     *  #1, #2, #6, #17 (#3 skipped — reference mismatch). Phase 14
     *  adds ~30 more paraphrases at 5-6× density per failing prompt to
     *  see if lexical anchoring still scales linearly past Phase 13's
     *  saturation point.
     * ============================================================ */

    /* #17 — fib_fact_op add: 5 more "adding" / "by adding" gerund forms
     * to overweight against the dominant subtract co-occurrence. */
    ADD3("// add fibonacci of n and factorial of n together",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// fibonacci and factorial of n added",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// adding the fib of n result and the fact of n result",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// fib n plus fact n by adding",
         w_fib_fact_op, (void *)"add", NULL, NULL);
    ADD3("// addition of fibonacci of n and factorial of n",
         w_fib_fact_op, (void *)"add", NULL, NULL);

    /* #1 — bmi_classified: 5 more "body mass index" + "limit it inside"
     * variations to break the mode collapse. */
    ADD3("// compute body mass index from weight and height limit it inside lo and hi",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);
    ADD3("// body mass index limit it inside bounds lo hi",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);
    ADD3("// the body mass index from weight and height limit it inside lo and hi bounds",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);
    ADD3("// body mass index of weight and height limit inside lo and hi",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);
    ADD3("// derive body mass index limit it inside",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);

    /* #2 — compound_interest: 5 more "interest gained" + "compounds at"
     * variations to drown out competing compound-related templates. */
    ADD3("// interest gained on investment when principal compounds at rate over years",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);
    ADD3("// interest gained when principal compounds at rate r over n years",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);
    ADD3("// the interest gained on a principal that compounds at rate over years",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);
    ADD3("// interest gained on principal compounding at rate r over years",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);
    ADD3("// principal compounds at rate over years interest gained",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);

    /* #6 — net_pay: 5 more "take home pay" + "federal tax rate" anchors.
     * Phase 13 already added 3; bring total to 8 to overweight against
     * the model's tendency to emit percentage-style graphs. */
    ADD3("// take home pay from gross income at federal tax rate is apply tax",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// federal take home pay from gross at tax rate via apply tax",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// take home pay equals apply tax of gross at federal rate",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// gross income at federal tax rate yields take home pay",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// from gross income at federal tax rate the take home pay",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);

    /* ============================================================
     *  Phase 4 — Micro examples (single-primitive 1-node graphs).
     *
     *  Each primitive gets several minimal 1-node examples so the
     *  organelle gets a strong prior on the syntax of every primitive
     *  it needs to assemble. Without these, rare primitives like
     *  fibonacci/factorial/apply_tax appear too few times to be
     *  reliably emitted in held-out generation.
     * ============================================================ */

    /* Unary primitives — each with multiple paraphrased prompts. */
    struct { const char *prim; const char *p[3]; } unary_set[] = {
        {"sigmoid",     {"// apply sigmoid to x", "// sigmoid activation of x", "// squash x with sigmoid"}},
        {"relu",        {"// apply relu to x", "// rectified linear of x", "// relu activation"}},
        {"abs_val",     {"// absolute value of x", "// magnitude of x", "// abs of x"}},
        {"square",      {"// square of x", "// x squared", "// raise x to the second power"}},
        {"cube",        {"// cube of x", "// x cubed", "// x to the third"}},
        {"double_val",  {"// double the value x", "// twice x", "// 2 times x"}},
        {"triple_val",  {"// triple the value x", "// 3 times x", "// thrice x"}},
        {"negate",      {"// negate x", "// minus x", "// flip the sign of x"}},
        {"factorial",   {"// factorial of n", "// n factorial", "// compute n bang"}},
        {"fibonacci",   {"// fibonacci of n", "// nth fibonacci number", "// fib n"}},
        {"harmonic_n",  {"// harmonic series sum of n terms", "// nth harmonic number", "// sum of 1 over k for k up to n"}},
        {"circle_area", {"// area of circle with radius r", "// circle area from radius", "// pi r squared"}}
    };
    for (size_t u = 0; u < sizeof(unary_set) / sizeof(unary_set[0]); u++) {
        for (int k = 0; k < 3; k++) {
            ADD3(unary_set[u].p[k], w_micro_unary, (void *)unary_set[u].prim, NULL, NULL);
        }
    }

    /* Binary primitives — port_kind chooses naming convention. */
    struct { const char *prim; int kind; const char *p[3]; } binary_set[] = {
        {"add",          0, {"// add x and y", "// sum of x and y", "// x plus y"}},
        {"subtract",     0, {"// subtract y from x", "// x minus y", "// difference between x and y"}},
        {"multiply",     0, {"// multiply x and y", "// product of x and y", "// x times y"}},
        {"min_two",      1, {"// minimum of a and b", "// smaller of a and b", "// min a b"}},
        {"max_two",      1, {"// maximum of a and b", "// larger of a and b", "// max a b"}},
        {"average_two",  1, {"// average of a and b", "// mean of a and b", "// midpoint between a and b"}},
        {"distance_1d",  1, {"// distance from a to b", "// one dimensional distance a b", "// absolute gap between a and b"}},
        {"midpoint",     1, {"// midpoint of a and b", "// halfway between a and b", "// center of segment a b"}},
        {"gcd",          1, {"// gcd of a and b", "// greatest common divisor of a and b", "// largest divisor of both a and b"}},
        {"mse",          1, {"// mean squared error of a and b", "// squared error a b", "// quadratic loss between a and b"}},
        {"tax_amount",   2, {"// tax amount on a given amount at rate", "// compute tax due", "// amount times tax rate"}},
        {"apply_tax",    2, {"// take home after tax on amount at rate", "// net of amount after tax", "// post tax remainder"}},
        {"percentage",   3, {"// percentage of part out of whole", "// part as a fraction of whole times 100", "// part divided by whole as percent"}},
        {"bmi",          4, {"// bmi from weight and height", "// body mass index", "// weight over height squared"}},
        {"discount",     5, {"// discounted price after rate", "// price after applying discount rate", "// reduced price"}},
        {"markup",       6, {"// price after markup on cost", "// cost plus markup rate", "// retail price from cost"}},
        {"power",        7, {"// base to the power exp", "// exponentiation base exp", "// raise base to exp"}},
        {"kinetic_energy", 8, {"// kinetic energy from mass m and velocity v", "// half m v squared", "// energy of motion"}}
    };
    for (size_t b = 0; b < sizeof(binary_set) / sizeof(binary_set[0]); b++) {
        for (int k = 0; k < 3; k++) {
            MicroBinCtx *ctx = (MicroBinCtx *)calloc(1, sizeof(MicroBinCtx));
            ctx->prim = binary_set[b].prim;
            ctx->port_kind = binary_set[b].kind;
            ADD3(binary_set[b].p[k], w_micro_binary, (void *)ctx, NULL, NULL);
        }
    }

    /* Ternary primitives. */
    struct { const char *prim; int kind; const char *p[3]; } ternary_set[] = {
        {"clamp",         0, {"// clamp x between lo and hi", "// limit x to range lo hi", "// bound x within lo and hi"}},
        {"lerp",          1, {"// lerp from a to b at t", "// linear interpolation a b t", "// blend a and b by t"}},
        {"compound",      2, {"// compound principal at rate over periods", "// compounded amount", "// compound growth of principal"}},
        {"future_value",  3, {"// future value of present at rate over periods", "// fv compounded forward", "// future worth of present"}},
        {"present_value", 4, {"// present value of future at rate over periods", "// pv discounted backward", "// present worth of future"}}
    };
    for (size_t t = 0; t < sizeof(ternary_set) / sizeof(ternary_set[0]); t++) {
        for (int k = 0; k < 3; k++) {
            MicroTerCtx *ctx = (MicroTerCtx *)calloc(1, sizeof(MicroTerCtx));
            ctx->prim = ternary_set[t].prim;
            ctx->port_kind = ternary_set[t].kind;
            ADD3(ternary_set[t].p[k], w_micro_ternary, (void *)ctx, NULL, NULL);
        }
    }

    /* ============================================================
     *  Phase 4 — Vocabulary-bridging paraphrases.
     *
     *  Bridge held-out NL phrasing patterns to seen primitives. Each
     *  entry pairs an unusual surface form with the core seed graph
     *  it should activate. Without these, novel words like
     *  "body mass index", "magnitude", "scaled by", "rectified output",
     *  "limit", "bounded between" map to <unk> at inference time.
     * ============================================================ */
    /* Bridge prompts for seed graphs. */
    ADD3("// compute body mass index then limit inside lo and hi bounds",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);
    ADD3("// body mass index of weight and height bounded between minimum and maximum",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);
    ADD3("// determine bmi and restrict to range lo and hi",
         w_seed, (void *)(SeedFn)seed_bmi_classified, NULL, NULL);

    ADD3("// interest gained on an investment when principal compounds at rate over years",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);
    ADD3("// final balance after compound growth minus the original principal",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);
    ADD3("// total return after compound growth subtract original",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);

    ADD3("// limit the output of sigmoid to lo and hi range",
         w_seed, (void *)(SeedFn)seed_clamped_sigmoid, NULL, NULL);
    ADD3("// sigmoid of x normalised by clamping into bounded range",
         w_seed, (void *)(SeedFn)seed_clamped_sigmoid, NULL, NULL);
    ADD3("// bounded sigmoid neuron output",
         w_seed, (void *)(SeedFn)seed_clamped_sigmoid, NULL, NULL);

    ADD3("// take home pay from gross income at tax rate",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// gross income reduced by tax liability",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// post tax pay from gross at federal rate",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);

    ADD3("// magnitude of difference between two forecasts",
         w_seed, (void *)(SeedFn)seed_abs_difference, NULL, NULL);
    ADD3("// absolute gap between a and b",
         w_seed, (void *)(SeedFn)seed_abs_difference, NULL, NULL);

    ADD3("// rectified output of x scaled by a gain factor",
         w_seed, (void *)(SeedFn)seed_scaled_relu, NULL, NULL);
    ADD3("// relu of x times a scale coefficient",
         w_seed, (void *)(SeedFn)seed_scaled_relu, NULL, NULL);

    ADD3("// greatest common divisor of a b and the result times k",
         w_seed, (void *)(SeedFn)seed_gcd_product, NULL, NULL);
    ADD3("// gcd of a b scaled by both a and b",
         w_seed, (void *)(SeedFn)seed_gcd_product, NULL, NULL);

    ADD3("// average a and b bounded between minimum and maximum",
         w_seed, (void *)(SeedFn)seed_clamped_average, NULL, NULL);
    ADD3("// mean of a and b limited within lo and hi",
         w_seed, (void *)(SeedFn)seed_clamped_average, NULL, NULL);

    ADD3("// future cashflow discounted back to its present worth",
         w_seed, (void *)(SeedFn)seed_net_present_value, NULL, NULL);
    ADD3("// net present value of cashflow at rate over years",
         w_seed, (void *)(SeedFn)seed_net_present_value, NULL, NULL);

    ADD3("// fibonacci of n times factorial of n",
         w_seed, (void *)(SeedFn)seed_fib_fact_product, NULL, NULL);
    ADD3("// product of fib n and fact n",
         w_seed, (void *)(SeedFn)seed_fib_fact_product, NULL, NULL);

    /* Bridge prompts that name primitives explicitly to anchor them. */
    ADD3("// invoice total of price times quantity plus tax amount at rate",
         w_taxed_total, (void *)(intptr_t)0, NULL, NULL);
    ADD3("// gross billing including sales tax on units sold",
         w_taxed_total, (void *)(intptr_t)1, NULL, NULL);

    ADD3("// fraction of income saved after subtracting expenses",
         w_savings_pipeline, (void *)(intptr_t)2, NULL, NULL);
    ADD3("// percentage of income remaining after deducting expenses",
         w_savings_pipeline, (void *)(intptr_t)3, NULL, NULL);

    /* Phase 10: argument-order disambiguation. Each pair of paraphrases
     * locks one drift-prone primitive into ONE specific arg-order
     * interpretation by naming the args' roles in the prompt. Targets
     * the savings_rate (#13) and take_home_pay (#6) drift cases. */

    /* percentage(part, whole) — first arg is the numerator. */
    ADD3("// percentage of saved out of income",
         w_seed, (void *)(SeedFn)seed_savings_rate, NULL, NULL);
    ADD3("// what fraction of income did we save",
         w_seed, (void *)(SeedFn)seed_savings_rate, NULL, NULL);
    ADD3("// saved as a percentage of total income",
         w_seed, (void *)(SeedFn)seed_savings_rate, NULL, NULL);
    ADD3("// take saved divided by income times one hundred",
         w_seed, (void *)(SeedFn)seed_savings_rate, NULL, NULL);

    /* apply_tax(amount, rate) — first arg is the gross amount. */
    ADD3("// take home pay equals apply_tax of gross at rate",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// net pay from gross income at federal rate",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// pay after federal tax",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);
    ADD3("// post tax net of gross at rate",
         w_seed, (void *)(SeedFn)seed_net_pay, NULL, NULL);

    /* compound(principal, rate, periods) — order locked by paraphrase. */
    ADD3("// principal at rate over years compounded then minus original",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);
    ADD3("// compound principal by rate for years yields total return",
         w_seed, (void *)(SeedFn)seed_compound_interest, NULL, NULL);

#undef ADD3
    *out_count = n;
    return cat;
}

/* ============================================================
 *  Vocabulary analysis
 * ============================================================ */

/* Count unique whitespace-separated tokens in a buffer. Simple O(n*v)
 * but fine for our corpus sizes. */
static int count_unique_tokens(const char *buf, int *char_count_out) {
    char **vocab = NULL; int vsize = 0; int vcap = 0;
    int chars = 0;
    const char *p = buf;
    while (*p) {
        chars++;
        /* skip whitespace */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) { chars++; p++; }
        if (!*p) break;
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
        size_t l = (size_t)(p - start);
        if (l == 0) continue;
        char tmp[128];
        if (l >= sizeof(tmp)) l = sizeof(tmp) - 1;
        memcpy(tmp, start, l); tmp[l] = '\0';
        int found = 0;
        for (int i = 0; i < vsize; i++) {
            if (strcmp(vocab[i], tmp) == 0) { found = 1; break; }
        }
        if (!found) {
            if (vsize == vcap) { vcap = vcap ? vcap * 2 : 64; vocab = (char **)realloc(vocab, sizeof(char *) * (size_t)vcap); }
            vocab[vsize++] = strdup(tmp);
        }
    }
    for (int i = 0; i < vsize; i++) free(vocab[i]);
    free(vocab);
    if (char_count_out) *char_count_out = chars;
    return vsize;
}

/* ============================================================
 *  Main
 * ============================================================ */

int main(int argc, char **argv) {
    FILE *out_train = stdout;
    FILE *out_val = NULL;
    int split = 0;
    if (argc == 2) {
        out_train = fopen(argv[1], "w");
        if (!out_train) { perror(argv[1]); return 1; }
    } else if (argc == 3) {
        out_train = fopen(argv[1], "w");
        out_val   = fopen(argv[2], "w");
        if (!out_train || !out_val) { perror("open"); return 1; }
        split = 1;
    }

    int n = 0;
    CorpusEntry *cat = build_catalog(&n);

    int ok_count = 0, fail_count = 0;
    int train_count = 0, val_count = 0;

    /* Buffer for vocab counting — accumulate all corpus text. */
    char *all_text = (char *)calloc(1024 * 1024, 1);
    size_t all_pos = 0; size_t all_cap = 1024 * 1024;

    fprintf(out_train, "# Pipeline IR — templated corpus (Phase 3b)\n");
    fprintf(out_train, "# %d examples; format: prompt comment + @graph...@end + --- separator\n\n", n);
    if (out_val) {
        fprintf(out_val, "# Pipeline IR — templated corpus (Phase 3b) — held-out validation split\n");
        fprintf(out_val, "# every 10th example reserved for validation\n\n");
    }

    for (int i = 0; i < n; i++) {
        Pipeline *p = cat[i].build(cat[i].ctx_a ? cat[i].ctx_a : (void *)0);
        if (!p) { fprintf(stderr, "[%d] %s: build returned NULL\n", i, cat[i].prompt); fail_count++; continue; }
        if (pipeline_verify(p) != PIPE_OK) {
            fprintf(stderr, "[%d] %s: verify failed: %s\n", i, cat[i].prompt, pipeline_last_error());
            pipeline_free(p);
            fail_count++;
            continue;
        }
        char *txt = pipeline_render_text(p);
        if (!txt) { pipeline_free(p); fail_count++; continue; }
        FILE *target = (split && (i % 10 == 9)) ? out_val : out_train;
        if (target == out_val) val_count++; else train_count++;
        fprintf(target, "%s\n%s---\n\n", cat[i].prompt, txt);

        /* Append to vocab buffer (grow as needed). */
        size_t need = strlen(cat[i].prompt) + strlen(txt) + 8;
        if (all_pos + need >= all_cap) {
            all_cap *= 2;
            all_text = (char *)realloc(all_text, all_cap);
        }
        all_pos += (size_t)snprintf(all_text + all_pos, all_cap - all_pos,
                                    "%s\n%s\n", cat[i].prompt, txt);

        free(txt);
        pipeline_free(p);
        ok_count++;
    }

    int chars = 0;
    int vocab = count_unique_tokens(all_text, &chars);
    free(all_text);

    fprintf(stderr, "\nGenerated %d / %d examples", ok_count, n);
    if (fail_count) fprintf(stderr, " (%d failed)", fail_count);
    if (split) fprintf(stderr, " | train=%d, val=%d", train_count, val_count);
    fprintf(stderr, "\nUnique whitespace-tokens: %d  |  Total characters: %d\n", vocab, chars);

    /* Free catalog. ctx_a is sometimes a heap array (multi-param families)
     * and sometimes an intptr_t-cast int (single-param families). Tracking
     * which is which adds complexity for ~1KB of bounded leak — process is
     * about to exit. Free only the prompt strings. */
    for (int i = 0; i < n; i++) {
        free(cat[i].prompt);
    }
    free(cat);

    if (out_train != stdout) fclose(out_train);
    if (out_val) fclose(out_val);
    return fail_count == 0 ? 0 : 1;
}
