/*
 * tools/e15_generate.c — Experiment E15 oracle corpus generator.
 *
 * Reads an OQL script, finds each CREATE CORPUS … FROM ORACLE …
 * statement and:
 *   1. invokes the oracle binary via oracle_emit (popen + cache);
 *   2. parses the JSON-line stream into (state, solution) pairs;
 *   3. (optional) audits each emitted state against a held-out
 *      corpus using the bigram-Jaccard;
 *   4. writes survivors to the `output` file in the OQL training
 *      format: one line per pair, "<state>\t<solution>\n".
 *
 * Reports: T2 yield, T3 leakage count, T6 cache-hit, T7 wall-clock.
 *
 * Usage:
 *   e15_generate <script.oql> [--audit-against <held_out.tsv>] [--verbose]
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#define _POSIX_C_SOURCE 200809L

#include "microgpt_oql.h"
#include "oracle_corpus_source.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char *slurp_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    if (len_out) *len_out = got;
    return buf;
}

/* Map an OQL "oracle_path" (typically the .c source path) to the
 * actually-invokable binary (build/<basename-without-ext>).  Example:
 * "tools/puzzle15_a_star.c" -> "build/puzzle15_a_star". */
static char *oracle_path_to_binary(const char *path) {
    if (!path) return NULL;
    const char *slash = strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    const char *dot = strrchr(base, '.');
    size_t blen = dot ? (size_t)(dot - base) : strlen(base);
    /* Heuristic: try "build/<base>" first; the caller can override
     * with an absolute path in the OQL `oracle_path` directly. */
    if (path[0] == '/' || strncmp(path, "./", 2) == 0) {
        return strdup(path);
    }
    char *out = (char *)malloc(blen + 16);
    if (!out) return NULL;
    snprintf(out, blen + 16, "build/%.*s", (int)blen, base);
    return out;
}

/* Slurp a held-out tsv (one state per line, "state\t..." or just
 * "state").  Returns a heap array of state strings (caller frees the
 * array AND each element). */
static int load_heldout_states(const char *path,
                               char ***out_states, size_t *out_n) {
    *out_states = NULL;
    *out_n = 0;
    size_t blen = 0;
    char *buf = slurp_file(path, &blen);
    if (!buf) return -1;
    /* Count lines. */
    size_t cap = 64;
    char **arr = (char **)malloc(cap * sizeof(char *));
    if (!arr) { free(buf); return -1; }
    size_t n = 0;
    char *line = buf, *end = buf + blen;
    while (line < end) {
        char *nl = memchr(line, '\n', (size_t)(end - line));
        if (nl) *nl = '\0';
        /* Trim trailing CR. */
        size_t llen = strlen(line);
        if (llen > 0 && line[llen - 1] == '\r') { line[--llen] = '\0'; }
        /* Take the first tab-separated field as the state. */
        char *tab = strchr(line, '\t');
        if (tab) *tab = '\0';
        if (*line) {
            if (n >= cap) {
                cap *= 2;
                char **n2 = (char **)realloc(arr, cap * sizeof(char *));
                if (!n2) { /* leak the partial buf; abort */ free(buf); free(arr); return -1; }
                arr = n2;
            }
            arr[n++] = strdup(line);
        }
        line = nl ? nl + 1 : end;
    }
    free(buf);
    *out_states = arr;
    *out_n = n;
    return 0;
}

static void free_states(char **arr, size_t n) {
    if (!arr) return;
    for (size_t i = 0; i < n; i++) free(arr[i]);
    free(arr);
}

static const char *kv_get_or(const OqlKV *kv, const char *key, const char *def) {
    const char *v = oql_kv_get(kv, key);
    return v ? v : def;
}

static int int_or(const OqlKV *kv, const char *key, int def) {
    const char *v = oql_kv_get(kv, key);
    return v ? atoi(v) : def;
}

static void print_stats(FILE *out, const char *name,
                        long emitted, long survivors, long audit_drops,
                        const OracleEmitStats *stats) {
    fprintf(out,
        "[e15] corpus %s:\n"
        "  emitted   = %ld\n"
        "  survivors = %ld (yield = %.1f%%)\n"
        "  audit drops (T3 leakage) = %ld\n"
        "  cache hit (T6) = %s\n"
        "  bytes received = %ld\n"
        "  wall (this corpus, fresh) = %.2fs\n",
        name, emitted, survivors,
        emitted > 0 ? 100.0 * (double)survivors / (double)emitted : 0.0,
        audit_drops,
        stats ? (stats->cache_hit ? "YES" : "no") : "?",
        stats ? stats->bytes_received : 0L,
        stats ? stats->wall_seconds : 0.0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: e15_generate <script.oql> "
                        "[--audit-against <heldout.tsv>] [--verbose]\n");
        return 2;
    }
    const char *script_path = argv[1];
    const char *audit_path = NULL;
    int verbose = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--audit-against") && i + 1 < argc) {
            audit_path = argv[++i];
        } else if (!strcmp(argv[i], "--verbose")) {
            verbose = 1;
        } else {
            fprintf(stderr, "e15_generate: unknown arg '%s'\n", argv[i]);
            return 2;
        }
    }

    size_t script_len = 0;
    char *script_src = slurp_file(script_path, &script_len);
    if (!script_src) {
        fprintf(stderr, "e15_generate: cannot read %s\n", script_path);
        return 1;
    }
    OqlScript *script = oql_parse(script_src);
    if (script->error) {
        fprintf(stderr, "e15_generate: parse error: %s\n", script->error);
        oql_script_free(script);
        free(script_src);
        return 1;
    }
    free(script_src);

    /* Load held-out states for the leakage audit (optional). */
    char **heldout = NULL;
    size_t n_heldout = 0;
    if (audit_path) {
        if (load_heldout_states(audit_path, &heldout, &n_heldout) != 0) {
            fprintf(stderr, "e15_generate: failed to read %s\n", audit_path);
            oql_script_free(script);
            return 1;
        }
        fprintf(stdout, "[e15] audit corpus: %zu held-out states from %s\n",
                n_heldout, audit_path);
    }

    clock_t total_t0 = clock();
    int n_corpora = 0;
    long total_emitted = 0, total_survivors = 0, total_drops = 0;

    for (const OqlStmt *s = script->head; s; s = s->next) {
        if (s->verb != OQL_VERB_CREATE_CORPUS_ORACLE) continue;
        const OqlCreateCorpusOracle *cc = &s->u.create_corpus_oracle;
        n_corpora++;

        char *binary = oracle_path_to_binary(cc->oracle_path);
        const char *output_path = kv_get_or(cc->with_kv, "output", NULL);
        if (!output_path) {
            fprintf(stderr, "e15_generate: corpus '%s' missing WITH output=...\n",
                    cc->name);
            free(binary);
            continue;
        }
        OracleSource src;
        memset(&src, 0, sizeof(src));
        src.oracle_binary = binary;
        src.cache_dir = kv_get_or(cc->with_kv, "cache", ".oql_oracle_cache");
        src.seed = int_or(cc->with_kv, "seed", 1337);
        src.count = int_or(cc->with_kv, "count", 1000);
        src.difficulty = kv_get_or(cc->with_kv, "difficulty", "mixed");
        src.verbose = verbose;

        fprintf(stdout,
            "[e15] generating corpus '%s' via '%s' "
            "(count=%d, seed=%d, difficulty=%s)\n",
            cc->name, binary, src.count, src.seed, src.difficulty);

        char *buf = NULL; size_t blen = 0;
        OracleEmitStats stats;
        memset(&stats, 0, sizeof(stats));
        FILE *log = verbose ? stderr : NULL;
        if (oracle_emit(&src, &buf, &blen, &stats, log) != 0) {
            fprintf(stderr, "e15_generate: oracle invocation failed for '%s'\n",
                    cc->name);
            free(binary);
            continue;
        }

        OraclePair *pairs = NULL;
        size_t n_pairs = 0;
        if (oracle_parse_jsonl(buf, blen, &pairs, &n_pairs) != 0) {
            fprintf(stderr, "e15_generate: JSON-line parse failed for '%s'\n",
                    cc->name);
            free(buf); free(binary);
            continue;
        }

        /* Open output, audit each pair, write survivors. */
        FILE *of = fopen(output_path, "wb");
        if (!of) {
            fprintf(stderr, "e15_generate: cannot open output '%s': %s\n",
                    output_path, strerror(errno));
            free(pairs); free(buf); free(binary);
            continue;
        }
        long survivors = 0, drops = 0;
        for (size_t i = 0; i < n_pairs; i++) {
            int leaked = 0;
            if (heldout && n_heldout > 0) {
                /* Verbatim equality FIRST (cheap, fast). */
                for (size_t h = 0; h < n_heldout; h++) {
                    if (strcmp(pairs[i].state, heldout[h]) == 0) {
                        leaked = 1; break;
                    }
                }
                /* Then bigram-Jaccard for near-duplicates. */
                if (!leaked) {
                    for (size_t h = 0; h < n_heldout; h++) {
                        double j = oracle_jaccard_state(pairs[i].state,
                                                        heldout[h]);
                        if (j >= 0.7) { leaked = 1; break; }
                    }
                }
            }
            if (leaked) { drops++; continue; }
            fprintf(of, "%s\t%s\n", pairs[i].state, pairs[i].solution);
            survivors++;
        }
        fclose(of);

        print_stats(stdout, cc->name, (long)n_pairs, survivors, drops,
                    &stats);
        fprintf(stdout, "  output -> %s (%ld lines)\n",
                output_path, survivors);

        total_emitted += (long)n_pairs;
        total_survivors += survivors;
        total_drops += drops;

        free(pairs);
        free(buf);
        free(binary);
    }

    free_states(heldout, n_heldout);

    double el = (double)(clock() - total_t0) / CLOCKS_PER_SEC;
    fprintf(stdout,
        "\n[e15] SUMMARY:\n"
        "  corpora generated: %d\n"
        "  total emitted:     %ld\n"
        "  total survivors:   %ld\n"
        "  total audit drops: %ld\n"
        "  wall-clock total:  %.2fs\n",
        n_corpora, total_emitted, total_survivors, total_drops, el);

    oql_script_free(script);
    if (n_corpora == 0) {
        fprintf(stderr, "e15_generate: no FROM ORACLE statements in %s\n",
                script_path);
        return 1;
    }
    return 0;
}
