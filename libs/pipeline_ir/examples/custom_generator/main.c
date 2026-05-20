/*
 * libpipeline_ir example: custom_generator
 *
 * Demonstrates the "no model required" use of the Pipeline IR — a
 * host program builds a graph directly from a hand-written C struct,
 * verifies it, renders both the canonical text format and DOT, and
 * prints the result to stdout.  This is the canonical pattern for
 * "unit-test-as-judge" style usage: any C-callable program (test
 * runner, code generator, AOT compiler, LLM wrapper, …) can emit a
 * graph and ask the library to certify it before execution.
 *
 * Build (inside the parent repo):
 *     cmake --build build --target pipeline_ir_example_custom_generator
 *     ./build/libs/pipeline_ir/examples/custom_generator/pipeline_ir_example_custom_generator
 *
 * The example uses ONLY the public ABI in <pipeline_ir/pipeline_ir.h>.
 * No engine, no VM, no transformer, no allocator override.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <pipeline_ir/pipeline_ir.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Construct a tiny three-node pipeline:
 *
 *     (sig.in x:int) ──> [add] ──> [square] ──> (sig.out y:int)
 *                          ^
 *                          │
 *     (sig.in y:int) ──────┘
 *
 * Semantics (executed externally): y_out = (x_in + y_in)^2
 *
 * The example does NOT execute the graph — it stops at verify() +
 * render to demonstrate the deterministic-Judge surface alone.  Use
 * pipeline_execute() with a PipelineDispatchFn (host-supplied) for an
 * end-to-end demo. */
static Pipeline *build_demo_graph(void) {
    Pipeline *p = pipeline_create("demo_add_then_square");
    if (!p) return NULL;

    /* Graph signature: two int inputs, one int output. */
    const char *sig_in_names[]  = { "x", "y" };
    PipelineType *sig_in_types[] = {
        pipeline_type_int(), pipeline_type_int()
    };
    const char *sig_out_names[] = { "result" };
    PipelineType *sig_out_types[] = { pipeline_type_int() };
    if (pipeline_set_signature(p, 2, sig_in_names, sig_in_types,
                                  1, sig_out_names, sig_out_types) != 0) {
        fprintf(stderr, "set_signature failed: %s\n", pipeline_last_error());
        pipeline_free(p);
        return NULL;
    }

    /* Leaf node 1: add(a:int, b:int) -> sum:int */
    {
        const char *in_names[] = { "a", "b" };
        PipelineType *in_types[] = {
            pipeline_type_int(), pipeline_type_int()
        };
        const char *out_names[] = { "sum" };
        PipelineType *out_types[] = { pipeline_type_int() };
        if (pipeline_add_node(p, "add1", "add",
                              2, in_names, in_types,
                              1, out_names, out_types) < 0) {
            fprintf(stderr, "add_node(add1) failed: %s\n",
                    pipeline_last_error());
            pipeline_free(p);
            return NULL;
        }
    }

    /* Leaf node 2: square(x:int) -> result:int */
    {
        const char *in_names[] = { "x" };
        PipelineType *in_types[] = { pipeline_type_int() };
        const char *out_names[] = { "result" };
        PipelineType *out_types[] = { pipeline_type_int() };
        if (pipeline_add_node(p, "sq1", "square",
                              1, in_names, in_types,
                              1, out_names, out_types) < 0) {
            fprintf(stderr, "add_node(sq1) failed: %s\n",
                    pipeline_last_error());
            pipeline_free(p);
            return NULL;
        }
    }

    /* Wire: x -> add1.a, y -> add1.b, add1.sum -> sq1.x, sq1.result -> sig.out result */
    if (pipeline_connect_signature_in(p, "x", "add1", "a") != 0 ||
        pipeline_connect_signature_in(p, "y", "add1", "b") != 0 ||
        pipeline_connect(p, "add1", "sum", "sq1", "x") != 0 ||
        pipeline_connect_signature_out(p, "sq1", "result", "result") != 0) {
        fprintf(stderr, "connect failed: %s\n", pipeline_last_error());
        pipeline_free(p);
        return NULL;
    }

    return p;
}

int main(void) {
    printf("# libpipeline_ir example: custom_generator\n");
    printf("# ABI version: %d.%d.%d\n\n",
           PIPELINE_IR_API_VERSION_MAJOR,
           PIPELINE_IR_API_VERSION_MINOR,
           PIPELINE_IR_API_VERSION_PATCH);

    Pipeline *p = build_demo_graph();
    if (!p) return 1;

    int rc = pipeline_verify(p);
    if (rc != PIPE_OK) {
        fprintf(stderr, "verify FAILED (%d): %s\n", rc, pipeline_last_error());
        pipeline_free(p);
        return 2;
    }
    printf("verify: PASS\n\n");

    char *txt = pipeline_render_text(p);
    if (!txt) {
        fprintf(stderr, "render_text returned NULL: %s\n",
                pipeline_last_error());
        pipeline_free(p);
        return 3;
    }
    printf("# canonical @graph form (round-trip-safe):\n%s\n", txt);
    free(txt);

    char *dot = pipeline_render_dot(p);
    if (!dot) {
        fprintf(stderr, "render_dot returned NULL: %s\n",
                pipeline_last_error());
        pipeline_free(p);
        return 4;
    }
    printf("# GraphViz DOT (pipe through `dot -Tsvg` to render):\n%s\n", dot);
    free(dot);

    pipeline_free(p);
    return 0;
}
