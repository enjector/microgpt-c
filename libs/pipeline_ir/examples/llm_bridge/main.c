/*
 * libpipeline_ir example: llm_bridge
 *
 * Bridges a Pipeline IR @graph...@end emission from a frontier LLM
 * (assumed already extracted from the model's response and supplied
 * on stdin) into a single-line JSON verdict suitable for piping into
 * a host orchestrator.
 *
 *   { "verdict": "PASS" | "FAIL",
 *     "stage":   "strict-parse" | "tolerant-parse" | "verify",
 *     "error":   "<pipeline_last_error()>" | null,
 *     "elapsed_us": <integer>,
 *     "repair": {
 *       "nodes_dropped": N, "edges_dropped": N,
 *       "sig_outs_disconnected": N, "sig_ins_dropped": N,
 *       "sig_outs_dropped": N
 *     }
 *   }
 *
 * No transformer engine, no VM — this is the canonical "use the IR
 * as a deterministic Judge for LLM tool calls" pattern that
 * Experiment E02's spear summary calls out.  Roughly 80 lines of C.
 *
 * Build (inside parent repo):
 *     cmake --build build --target pipeline_ir_example_llm_bridge
 *     echo "$LLM_GRAPH" | ./.../pipeline_ir_example_llm_bridge
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <pipeline_ir/pipeline_ir.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char *slurp_stdin(void) {
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    int c;
    while ((c = fgetc(stdin)) != EOF) {
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

/* Print a JSON string literal, escaping the four ASCII control cases
 * we care about plus '"' and '\\'.  No unicode-escape pass — error
 * messages are ASCII. */
static void print_json_string(FILE *out, const char *s) {
    fputc('"', out);
    if (s) {
        for (const char *p = s; *p; p++) {
            unsigned char c = (unsigned char)*p;
            switch (c) {
            case '"':  fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\n': fputs("\\n",  out); break;
            case '\r': fputs("\\r",  out); break;
            case '\t': fputs("\\t",  out); break;
            default:
                if (c < 0x20)
                    fprintf(out, "\\u%04x", c);
                else
                    fputc((char)c, out);
            }
        }
    }
    fputc('"', out);
}

static long long now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + (long long)ts.tv_nsec / 1000LL;
}

int main(void) {
    char *src = slurp_stdin();
    if (!src) {
        fprintf(stderr, "llm_bridge: OOM reading stdin\n");
        return 1;
    }

    long long t0 = now_us();
    const char *stage = "strict-parse";
    const char *err = NULL;
    Pipeline *p = pipeline_parse_text(src);
    if (!p) {
        err = pipeline_last_error();
        stage = "tolerant-parse";
        p = pipeline_parse_text_tolerant(src);
    }
    free(src);

    PipelineRepairReport rep = {0};
    int verdict_pass = 0;
    if (p) {
        pipeline_repair(p, &rep);
        stage = "verify";
        int rc = pipeline_verify(p);
        if (rc == PIPE_OK) {
            verdict_pass = 1;
            err = NULL;
        } else {
            err = pipeline_last_error();
        }
    }
    long long elapsed_us = now_us() - t0;

    fputs("{\"verdict\":", stdout);
    fputs(verdict_pass ? "\"PASS\"" : "\"FAIL\"", stdout);
    fputs(",\"stage\":", stdout); print_json_string(stdout, stage);
    fputs(",\"error\":", stdout);
    if (err) print_json_string(stdout, err);
    else     fputs("null", stdout);
    fprintf(stdout, ",\"elapsed_us\":%lld", elapsed_us);
    fprintf(stdout,
            ",\"repair\":{\"nodes_dropped\":%d,\"edges_dropped\":%d,"
            "\"sig_outs_disconnected\":%d,\"sig_ins_dropped\":%d,"
            "\"sig_outs_dropped\":%d}",
            rep.nodes_dropped, rep.edges_dropped,
            rep.sig_outs_disconnected, rep.sig_ins_dropped,
            rep.sig_outs_dropped);
    fputs("}\n", stdout);

    if (p) pipeline_free(p);
    return verdict_pass ? 0 : 2;
}
