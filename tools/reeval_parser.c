/*
 * MicroGPT-C — Phase 5a parser re-eval tool.
 *
 * Reads the held-out section of a wiring_organelle_demo log, extracts
 * each "best output" @graph block, and reports how many parse + verify
 * under (a) the strict parser, (b) the tolerant parser. Same model, same
 * generations — measures only the parser-side gain from Phase 5a.
 *
 * Usage:
 *   reeval_parser <wiring_log> [--expected-file <held_out.txt>]
 *
 * Format expected (matches demo output):
 *   [N] // <prompt>
 *       EXPECTED: <prims>
 *       well=Y parse=Y verify=Y fidelity=Y votes=K/16
 *       --- best output ---
 *       <graph lines>
 *       ---
 */

#define _CRT_SECURE_NO_WARNINGS 1
#include "microgpt_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BLOCK 8192

static int graph_uses_all(const Pipeline *p, const char *expected) {
    if (!expected || !*expected) return 1;
    char buf[256];
    strncpy(buf, expected, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *cur = buf;
    while (*cur) {
        while (*cur == ' ' || *cur == '\t') cur++;
        if (!*cur) break;
        char *end = cur;
        while (*end && *end != ' ' && *end != '\t') end++;
        char saved = *end;
        *end = '\0';
        int found = 0;
        for (size_t i = 0; i < p->n_nodes; i++) {
            if (p->nodes[i]->primitive && strcmp(p->nodes[i]->primitive, cur) == 0) { found = 1; break; }
        }
        if (!found) return 0;
        *end = saved;
        cur = (saved == '\0') ? end : end + 1;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <wiring_log>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }

    char line[4096];
    char prompt[1024] = "";
    char expected[256] = "";
    char block[MAX_BLOCK];
    size_t blen = 0;
    int in_block = 0;
    int n = 0;
    int strict_parse = 0, strict_verify = 0, strict_fidelity = 0;
    int tol_parse = 0, tol_verify = 0, tol_fidelity = 0;
    int rep_parse = 0, rep_verify = 0, rep_fidelity = 0;

    while (fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);
        while (l > 0 && (line[l-1] == '\n' || line[l-1] == '\r')) line[--l] = '\0';

        if (strncmp(line, "[", 1) == 0 && strstr(line, "] // ")) {
            const char *p = strstr(line, "// ");
            if (p) { strncpy(prompt, p, sizeof(prompt) - 1); prompt[sizeof(prompt) - 1] = '\0'; }
            expected[0] = '\0';
            in_block = 0;
            blen = 0;
            continue;
        }
        const char *exp = strstr(line, "EXPECTED:");
        if (exp) {
            exp += strlen("EXPECTED:");
            while (*exp == ' ' || *exp == '\t') exp++;
            strncpy(expected, exp, sizeof(expected) - 1);
            expected[sizeof(expected) - 1] = '\0';
            continue;
        }
        if (strstr(line, "--- best output ---")) {
            in_block = 1;
            blen = 0;
            block[0] = '\0';
            continue;
        }
        if (in_block) {
            /* End markers: indented "    ---" or trailing blank line. */
            const char *trim = line;
            while (*trim == ' ' || *trim == '\t') trim++;
            if (strcmp(trim, "---") == 0) {
                /* Process accumulated block. */
                int s_p = 0, s_v = 0, s_f = 0, t_p = 0, t_v = 0, t_f = 0, r_p = 0, r_v = 0, r_f = 0;
                if (blen > 0) {
                    /* All three columns require n_nodes > 0 for a verify to
                     * count as useful (an empty-but-vacuously-verifying graph
                     * is structurally valid but produces nothing). */
                    Pipeline *p1 = pipeline_parse_text(block);
                    if (p1) {
                        s_p = 1;
                        if (pipeline_verify(p1) == PIPE_OK && p1->n_nodes > 0) {
                            s_v = 1;
                            s_f = graph_uses_all(p1, expected);
                        }
                        pipeline_free(p1);
                    }
                    Pipeline *p2 = pipeline_parse_text(block);
                    if (!p2) p2 = pipeline_parse_text_tolerant(block);
                    if (p2) {
                        t_p = 1;
                        if (pipeline_verify(p2) == PIPE_OK && p2->n_nodes > 0) {
                            t_v = 1;
                            t_f = graph_uses_all(p2, expected);
                        }
                        pipeline_free(p2);
                    }
                    /* Phase 5b: tolerant + repair. Requires n_nodes > 0
                     * after repair — an empty residual verifies vacuously
                     * but isn't useful, so we don't count it. */
                    Pipeline *p3 = pipeline_parse_text(block);
                    if (!p3) p3 = pipeline_parse_text_tolerant(block);
                    if (p3) {
                        r_p = 1;
                        int v0 = (pipeline_verify(p3) == PIPE_OK) && p3->n_nodes > 0;
                        if (!v0) {
                            PipelineRepairReport rep = {0};
                            if (pipeline_repair(p3, &rep) == PIPE_OK
                                && p3->n_nodes > 0
                                && pipeline_verify(p3) == PIPE_OK) {
                                v0 = 1;
                            }
                        }
                        if (v0) {
                            r_v = 1;
                            r_f = graph_uses_all(p3, expected);
                        }
                        pipeline_free(p3);
                    }
                }
                strict_parse += s_p; strict_verify += s_v; strict_fidelity += s_f;
                tol_parse    += t_p; tol_verify    += t_v; tol_fidelity    += t_f;
                rep_parse    += r_p; rep_verify    += r_v; rep_fidelity    += r_f;
                printf("[%2d] strict=%c%c%c  tolerant=%c%c%c  repair=%c%c%c  %s\n",
                       n + 1,
                       s_p ? 'P' : '.', s_v ? 'V' : '.', s_f ? 'F' : '.',
                       t_p ? 'P' : '.', t_v ? 'V' : '.', t_f ? 'F' : '.',
                       r_p ? 'P' : '.', r_v ? 'V' : '.', r_f ? 'F' : '.',
                       prompt);
                n++;
                in_block = 0;
                blen = 0;
                continue;
            }
            size_t need = strlen(line) + 2;
            if (blen + need < sizeof(block)) {
                size_t cl = strlen(line);
                memcpy(block + blen, line, cl);
                block[blen + cl] = '\n';
                blen += cl + 1;
                block[blen] = '\0';
            }
        }
    }
    fclose(f);

    printf("\n=== Re-eval summary on %d held-out examples ===\n", n);
    printf("                      strict   tolerant   repair\n");
    printf("  parsed      :       %2d/%2d    %2d/%2d     %2d/%2d\n", strict_parse, n, tol_parse, n, rep_parse, n);
    printf("  verified    :       %2d/%2d    %2d/%2d     %2d/%2d\n", strict_verify, n, tol_verify, n, rep_verify, n);
    printf("  fidelity    :       %2d/%2d    %2d/%2d     %2d/%2d\n", strict_fidelity, n, tol_fidelity, n, rep_fidelity, n);
    printf("\n");
    if (n > 0) {
        printf("  verify pct  :       %3.0f%%      %3.0f%%      %3.0f%%   (repair delta vs strict %+.0fpp)\n",
               100.0 * strict_verify / n, 100.0 * tol_verify / n, 100.0 * rep_verify / n,
               100.0 * (rep_verify - strict_verify) / n);
        printf("  fidelity pct:       %3.0f%%      %3.0f%%      %3.0f%%   (repair delta vs strict %+.0fpp)\n",
               100.0 * strict_fidelity / n, 100.0 * tol_fidelity / n, 100.0 * rep_fidelity / n,
               100.0 * (rep_fidelity - strict_fidelity) / n);
    }
    return 0;
}
