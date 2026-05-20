/*
 * tools/oql_runner.c  —  command-line entry point for `oql run <file.oql>`
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * E09 Phase 3 CLI.  Reads a .oql file from argv[2] (or stdin if argv[2] == "-")
 * and executes it via oql_execute_with_runtime.  Prints the per-RUN metric
 * row that the runtime records on rt->last_*.
 *
 * Usage:
 *   oql run experiments/connect4.oql
 *   oql run path/to/script.oql
 *   echo "VERIFY GRAPH @graph demo @end;" | oql run -
 *
 * Verbs supported (everything except TRAIN):
 *   VERIFY GRAPH / AUDIT / CREATE BEHAVIOUR / CREATE ORGANELLE / COMPOSE / RUN
 *
 * TRAIN remains honestly stubbed per E09 T6 (returns OQL_ERR_NOT_IMPLEMENTED).
 */

#include "microgpt_oql.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *path) {
    FILE *f = (strcmp(path, "-") == 0) ? stdin : fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "oql: cannot open '%s'\n", path);
        return NULL;
    }
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { if (f != stdin) fclose(f); return NULL; }
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *r = (char *)realloc(buf, cap);
            if (!r) { free(buf); if (f != stdin) fclose(f); return NULL; }
            buf = r;
        }
        buf[len++] = (char)c;
    }
    buf[len] = '\0';
    if (f != stdin) fclose(f);
    return buf;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s run <file.oql>          execute an OQL script\n"
        "  %s run -                   read from stdin\n"
        "\n"
        "E09 — wires the RUN / COMPOSE / CREATE ORGANELLE verbs end-to-end.\n"
        "TRAIN remains stubbed (T6 hard-lock); see experiments/E09-oql-runtime-wiring.md.\n",
        argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc < 3 || strcmp(argv[1], "run") != 0) {
        usage(argv[0]);
        return 1;
    }
    char *src = slurp(argv[2]);
    if (!src) return 2;

    OqlScript *script = oql_parse(src);
    free(src);
    if (!script) { fprintf(stderr, "oql: out of memory\n"); return 3; }
    if (script->error) {
        fprintf(stderr, "oql: %s\n", script->error);
        oql_script_free(script);
        return 4;
    }

    OqlRuntime rt;
    oql_runtime_init(&rt);

    int failed_idx = 0;
    oql_status st = oql_execute_with_runtime(script, &rt, stdout, &failed_idx);

    /* Print final summary line if any RUN was issued (rt.last_games_played > 0). */
    if (rt.last_games_played > 0) {
        double win_rate = 100.0 * (double)rt.last_wins
                                 / (double)rt.last_games_played;
        printf("\n--- OQL RUN summary ---\n");
        printf("games:        %d\n", rt.last_games_played);
        printf("wins:         %d (%.1f%%)\n", rt.last_wins, win_rate);
        printf("draws:        %d\n", rt.last_draws);
        printf("losses:       %d\n", rt.last_losses);
        printf("p99 latency:  %.2f ms\n", rt.last_p99_ms);
        printf("audit rows:   %d\n", rt.last_audit_rows);
        printf("total time:   %.2f s\n", rt.last_total_seconds);
    }

    int rc = (st == OQL_OK) ? 0 : 5;
    if (st != OQL_OK) {
        fprintf(stderr, "\noql: execute failed (status=%d, stmt=%d)\n",
                (int)st, failed_idx);
    }
    oql_runtime_dispose(&rt);
    oql_script_free(script);
    return rc;
}
