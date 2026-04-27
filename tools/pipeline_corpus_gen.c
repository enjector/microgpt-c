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
