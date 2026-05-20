/*
 * libpipeline_ir example: audit_visualiser
 *
 * Read a Pipeline IR @graph...@end text fragment from the file named
 * on the command line (or stdin if no path is given), verify it, and
 * emit the GraphViz DOT representation to stdout.
 *
 *   ./audit_visualiser graph.txt | dot -Tsvg -o graph.svg
 *
 * Failure modes are mapped to non-zero exit codes so the program is
 * a usable building block in audit pipelines:
 *
 *   0 — graph parsed and verified; DOT printed
 *   1 — could not read input file
 *   2 — parse failed (tolerant parse also rejected)
 *   3 — verify failed after repair
 *   4 — render_dot returned NULL
 *
 * The verifier output (success or the first error) is printed on
 * stderr regardless so an automated audit job sees both the verdict
 * and the trace.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <pipeline_ir/pipeline_ir.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(FILE *f) {
    /* Read until EOF; grow buffer geometrically.  Caller frees. */
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        buf[len++] = (char)c;
    }
    buf[len] = '\0';
    return buf;
}

int main(int argc, char **argv) {
    FILE *in = stdin;
    const char *src_path = "<stdin>";
    if (argc > 1) {
        in = fopen(argv[1], "rb");
        if (!in) {
            fprintf(stderr, "audit_visualiser: cannot open '%s'\n", argv[1]);
            return 1;
        }
        src_path = argv[1];
    }
    char *src = slurp(in);
    if (in != stdin) fclose(in);
    if (!src) {
        fprintf(stderr, "audit_visualiser: OOM reading input\n");
        return 1;
    }

    Pipeline *p = pipeline_parse_text(src);
    if (!p) {
        /* Strict parse rejected — try the tolerant path which
         * dedupes duplicate sig inputs and auto-promotes referenced-
         * but-undeclared signature ports. */
        p = pipeline_parse_text_tolerant(src);
    }
    free(src);
    if (!p) {
        fprintf(stderr, "audit_visualiser: parse failed (%s): %s\n",
                src_path, pipeline_last_error());
        return 2;
    }

    PipelineRepairReport rep = {0};
    pipeline_repair(p, &rep);
    if (rep.nodes_dropped || rep.edges_dropped ||
        rep.sig_outs_disconnected || rep.sig_ins_dropped ||
        rep.sig_outs_dropped) {
        fprintf(stderr,
                "audit_visualiser: repair pass dropped %d nodes, %d edges, "
                "%d sig_outs_disconnected, %d sig_ins_dropped, %d sig_outs_dropped\n",
                rep.nodes_dropped, rep.edges_dropped,
                rep.sig_outs_disconnected, rep.sig_ins_dropped,
                rep.sig_outs_dropped);
    }

    int rc = pipeline_verify(p);
    if (rc != PIPE_OK) {
        fprintf(stderr, "audit_visualiser: VERIFY FAILED (%d): %s\n",
                rc, pipeline_last_error());
        pipeline_free(p);
        return 3;
    }
    fprintf(stderr, "audit_visualiser: VERIFY PASSED\n");

    char *dot = pipeline_render_dot(p);
    if (!dot) {
        fprintf(stderr, "audit_visualiser: render_dot failed: %s\n",
                pipeline_last_error());
        pipeline_free(p);
        return 4;
    }
    fputs(dot, stdout);
    free(dot);
    pipeline_free(p);
    return 0;
}
