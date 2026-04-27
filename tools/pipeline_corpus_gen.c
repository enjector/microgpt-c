/*
 * MicroGPT-C — Pipeline IR Corpus Generator (Phase 3a)
 *
 * Hand-curated (prompt, graph-text) example pairs for the future
 * Wiring Organelle's training corpus. Each example is built
 * programmatically via the Pipeline IR API, verified, and rendered
 * to text. The generator's output is the corpus.
 *
 * Why hand-curated: the existing 1597-function VM corpus is
 * imperative (loops, conditionals, state) and doesn't decompose
 * cleanly to a pure-dataflow IR. Phase 3a's goal is methodology
 * proof — show that *some* well-formed dataflow graphs can be
 * generated and round-trip cleanly. Phase 3b will scale up.
 *
 * Build:  cmake --build build --target pipeline_corpus_gen
 * Run:    ./build/pipeline_corpus_gen [output_file]
 *         (stdout if no arg)
 *
 * Output format:
 *   # prompt: <natural-language description>
 *   <graph text emitted by pipeline_render_text>
 *   ---
 *   (repeat per example)
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 *  Helpers
 * ============================================================ */

static int sig_int(Pipeline *p, const char *in_name, const char *out_name) {
    /* 1-input, 1-output, both int. Convenience wrapper. */
    const char *in_names[]  = { in_name };
    PipelineType *in_types[] = { pipeline_type_int() };
    const char *out_names[] = { out_name };
    PipelineType *out_types[] = { pipeline_type_int() };
    return pipeline_set_signature(p, 1, in_names, in_types, 1, out_names, out_types);
}

static int sig_int2(Pipeline *p, const char *a, const char *b, const char *out_name) {
    const char *in_names[]  = { a, b };
    PipelineType *in_types[] = { pipeline_type_int(), pipeline_type_int() };
    const char *out_names[] = { out_name };
    PipelineType *out_types[] = { pipeline_type_int() };
    return pipeline_set_signature(p, 2, in_names, in_types, 1, out_names, out_types);
}

static int sig_int3(Pipeline *p, const char *a, const char *b, const char *c, const char *out_name) {
    const char *in_names[]  = { a, b, c };
    PipelineType *in_types[] = { pipeline_type_int(), pipeline_type_int(), pipeline_type_int() };
    const char *out_names[] = { out_name };
    PipelineType *out_types[] = { pipeline_type_int() };
    return pipeline_set_signature(p, 3, in_names, in_types, 1, out_names, out_types);
}

static int sig_int4(Pipeline *p, const char *a, const char *b, const char *c, const char *d, const char *out_name) {
    const char *in_names[]  = { a, b, c, d };
    PipelineType *in_types[] = { pipeline_type_int(), pipeline_type_int(),
                                 pipeline_type_int(), pipeline_type_int() };
    const char *out_names[] = { out_name };
    PipelineType *out_types[] = { pipeline_type_int() };
    return pipeline_set_signature(p, 4, in_names, in_types, 1, out_names, out_types);
}

static int add_node_2in_1out(Pipeline *p, const char *id, const char *primitive,
                             const char *in1, const char *in2) {
    const char *in_names[]  = { in1, in2 };
    PipelineType *in_types[] = { pipeline_type_int(), pipeline_type_int() };
    const char *out_names[] = { "out" };
    PipelineType *out_types[] = { pipeline_type_int() };
    return pipeline_add_node(p, id, primitive,
                             2, in_names, in_types,
                             1, out_names, out_types);
}

static int add_node_1in_1out(Pipeline *p, const char *id, const char *primitive,
                             const char *in1) {
    const char *in_names[]  = { in1 };
    PipelineType *in_types[] = { pipeline_type_int() };
    const char *out_names[] = { "out" };
    PipelineType *out_types[] = { pipeline_type_int() };
    return pipeline_add_node(p, id, primitive,
                             1, in_names, in_types,
                             1, out_names, out_types);
}

/* ============================================================
 *  Hand-curated examples
 * ============================================================ */

/* 1. add(a, b) → y */
static Pipeline *ex_add(void) {
    Pipeline *p = pipeline_create("ex_add");
    sig_int2(p, "a", "b", "y");
    add_node_2in_1out(p, "n", "add", "x", "y");
    pipeline_connect_signature_in(p, "a", "n", "x");
    pipeline_connect_signature_in(p, "b", "n", "y");
    pipeline_connect_signature_out(p, "n", "out", "y");
    return p;
}

/* 2. multiply(a, b) → y */
static Pipeline *ex_multiply(void) {
    Pipeline *p = pipeline_create("ex_multiply");
    sig_int2(p, "a", "b", "y");
    add_node_2in_1out(p, "n", "multiply", "x", "y");
    pipeline_connect_signature_in(p, "a", "n", "x");
    pipeline_connect_signature_in(p, "b", "n", "y");
    pipeline_connect_signature_out(p, "n", "out", "y");
    return p;
}

/* 3. negate(x) → y */
static Pipeline *ex_negate(void) {
    Pipeline *p = pipeline_create("ex_negate");
    sig_int(p, "x", "y");
    add_node_1in_1out(p, "n", "negate", "x");
    pipeline_connect_signature_in(p, "x", "n", "x");
    pipeline_connect_signature_out(p, "n", "out", "y");
    return p;
}

/* 4. abs_val(x) → y */
static Pipeline *ex_abs_val(void) {
    Pipeline *p = pipeline_create("ex_abs_val");
    sig_int(p, "x", "y");
    add_node_1in_1out(p, "n", "abs", "x");
    pipeline_connect_signature_in(p, "x", "n", "x");
    pipeline_connect_signature_out(p, "n", "out", "y");
    return p;
}

/* 5. add_then_negate(a, b) → y     (negate(a + b))   — 2-node chain */
static Pipeline *ex_add_then_negate(void) {
    Pipeline *p = pipeline_create("ex_add_then_negate");
    sig_int2(p, "a", "b", "y");
    add_node_2in_1out(p, "sum", "add", "x", "y");
    add_node_1in_1out(p, "neg", "negate", "x");
    pipeline_connect_signature_in(p, "a", "sum", "x");
    pipeline_connect_signature_in(p, "b", "sum", "y");
    pipeline_connect(p, "sum", "out", "neg", "x");
    pipeline_connect_signature_out(p, "neg", "out", "y");
    return p;
}

/* 6. square_sum(a, b) → y        (a*a + b*b)   — 3-node tree */
static Pipeline *ex_square_sum(void) {
    Pipeline *p = pipeline_create("ex_square_sum");
    sig_int2(p, "a", "b", "y");
    add_node_2in_1out(p, "sq_a", "multiply", "x", "y");
    add_node_2in_1out(p, "sq_b", "multiply", "x", "y");
    add_node_2in_1out(p, "sum", "add", "x", "y");
    pipeline_connect_signature_in(p, "a", "sq_a", "x");
    pipeline_connect_signature_in(p, "a", "sq_a", "y");
    pipeline_connect_signature_in(p, "b", "sq_b", "x");
    pipeline_connect_signature_in(p, "b", "sq_b", "y");
    pipeline_connect(p, "sq_a", "out", "sum", "x");
    pipeline_connect(p, "sq_b", "out", "sum", "y");
    pipeline_connect_signature_out(p, "sum", "out", "y");
    return p;
}

/* 7. axpy(a, x, y_in) → y          (a*x + y_in)   — 2-node chain */
static Pipeline *ex_axpy(void) {
    Pipeline *p = pipeline_create("ex_axpy");
    sig_int3(p, "a", "x", "y_in", "y");
    add_node_2in_1out(p, "ax", "multiply", "x", "y");
    add_node_2in_1out(p, "sum", "add", "x", "y");
    pipeline_connect_signature_in(p, "a", "ax", "x");
    pipeline_connect_signature_in(p, "x", "ax", "y");
    pipeline_connect(p, "ax", "out", "sum", "x");
    pipeline_connect_signature_in(p, "y_in", "sum", "y");
    pipeline_connect_signature_out(p, "sum", "out", "y");
    return p;
}

/* 8. polynomial_2(a, x, b) → y     (a*x^2 + b)   — 3-node chain */
static Pipeline *ex_polynomial_2(void) {
    Pipeline *p = pipeline_create("ex_polynomial_2");
    sig_int3(p, "a", "x", "b", "y");
    /* a * x = ax */
    add_node_2in_1out(p, "ax", "multiply", "x", "y");
    /* ax * x = ax2 (using x = a*x output piped through, NOT x squared, avoiding extra node) */
    /* Actually for clarity: a*x*x uses two multiplies. */
    add_node_2in_1out(p, "ax2", "multiply", "x", "y");
    /* ax2 + b = result */
    add_node_2in_1out(p, "result", "add", "x", "y");
    pipeline_connect_signature_in(p, "a", "ax", "x");
    pipeline_connect_signature_in(p, "x", "ax", "y");
    pipeline_connect(p, "ax", "out", "ax2", "x");
    pipeline_connect_signature_in(p, "x", "ax2", "y");
    pipeline_connect(p, "ax2", "out", "result", "x");
    pipeline_connect_signature_in(p, "b", "result", "y");
    pipeline_connect_signature_out(p, "result", "out", "y");
    return p;
}

/* 9. distance_squared(a1, a2, b1, b2) → y   ((a1-b1)^2 + (a2-b2)^2) — 5-node */
static Pipeline *ex_distance_squared(void) {
    Pipeline *p = pipeline_create("ex_distance_squared");
    sig_int4(p, "a1", "a2", "b1", "b2", "y");
    add_node_2in_1out(p, "dx",  "subtract", "x", "y");
    add_node_2in_1out(p, "dy",  "subtract", "x", "y");
    add_node_2in_1out(p, "dx2", "multiply", "x", "y");
    add_node_2in_1out(p, "dy2", "multiply", "x", "y");
    add_node_2in_1out(p, "sum", "add",      "x", "y");
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
    return p;
}

/* 10. linear_interp(a, b, t) → y    (a + (b-a)*t)  — 3-node chain */
static Pipeline *ex_linear_interp(void) {
    Pipeline *p = pipeline_create("ex_linear_interp");
    sig_int3(p, "a", "b", "t", "y");
    add_node_2in_1out(p, "diff",  "subtract", "x", "y");   /* b - a */
    add_node_2in_1out(p, "scale", "multiply", "x", "y");   /* diff * t */
    add_node_2in_1out(p, "result", "add",     "x", "y");   /* a + scale */
    pipeline_connect_signature_in(p, "b", "diff", "x");
    pipeline_connect_signature_in(p, "a", "diff", "y");
    pipeline_connect(p, "diff", "out", "scale", "x");
    pipeline_connect_signature_in(p, "t", "scale", "y");
    pipeline_connect_signature_in(p, "a", "result", "x");
    pipeline_connect(p, "scale", "out", "result", "y");
    pipeline_connect_signature_out(p, "result", "out", "y");
    return p;
}

/* ============================================================
 *  Catalog
 * ============================================================ */

typedef struct {
    const char *prompt;
    Pipeline *(*build)(void);
} Example;

static const Example g_examples[] = {
    { "// add two integers",                                    ex_add },
    { "// multiply two integers",                               ex_multiply },
    { "// negate an integer",                                   ex_negate },
    { "// absolute value of an integer",                        ex_abs_val },
    { "// negate the sum of two integers",                      ex_add_then_negate },
    { "// sum of squares of two integers",                      ex_square_sum },
    { "// axpy: a*x + y_in",                                    ex_axpy },
    { "// degree-2 polynomial a*x*x + b",                       ex_polynomial_2 },
    { "// squared distance between (a1,a2) and (b1,b2)",        ex_distance_squared },
    { "// linear interpolation a + (b-a)*t",                    ex_linear_interp },
};
static const int N_EXAMPLES = (int)(sizeof(g_examples) / sizeof(g_examples[0]));

/* ============================================================
 *  Main
 * ============================================================ */

int main(int argc, char **argv) {
    FILE *out = stdout;
    if (argc > 1) {
        out = fopen(argv[1], "w");
        if (!out) {
            fprintf(stderr, "failed to open %s for writing\n", argv[1]);
            return 1;
        }
    }

    int ok_count = 0;
    int fail_count = 0;

    fprintf(out, "# Pipeline IR — hand-curated training corpus (Phase 3a)\n");
    fprintf(out, "# %d examples; format: prompt comment + @graph...@end + --- separator\n\n", N_EXAMPLES);

    for (int i = 0; i < N_EXAMPLES; i++) {
        Pipeline *p = g_examples[i].build();
        if (!p) {
            fprintf(stderr, "[%d] build returned NULL\n", i);
            fail_count++;
            continue;
        }
        if (pipeline_verify(p) != PIPE_OK) {
            fprintf(stderr, "[%d] %s: verify failed: %s\n", i, g_examples[i].prompt,
                    pipeline_last_error());
            pipeline_free(p);
            fail_count++;
            continue;
        }
        char *txt = pipeline_render_text(p);
        if (!txt) {
            fprintf(stderr, "[%d] render returned NULL\n", i);
            pipeline_free(p);
            fail_count++;
            continue;
        }
        fprintf(out, "%s\n", g_examples[i].prompt);
        fprintf(out, "%s", txt);
        fprintf(out, "---\n\n");
        free(txt);
        pipeline_free(p);
        ok_count++;
    }

    fprintf(stderr, "\nGenerated %d / %d examples successfully", ok_count, N_EXAMPLES);
    if (fail_count > 0) fprintf(stderr, " (%d failed)", fail_count);
    fprintf(stderr, "\n");

    if (out != stdout) fclose(out);
    return fail_count == 0 ? 0 : 1;
}
