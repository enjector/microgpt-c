/*
 * tools/e12_generate.c — Experiment E12 corpus generator.
 *
 * Reads an OQL script, finds the CREATE CORPUS … FROM LLM …
 * statement (one or more), and for each:
 *   1. health-checks the LM Studio endpoint
 *   2. emits `count` (prompt, graph) candidates from the LLM
 *   3. for each candidate: parse via pipeline_parse_text_tolerant ->
 *      pipeline_repair -> pipeline_verify (if VERIFY_VIA set)
 *   4. for each survivor: Jaccard audit against held-out corpus
 *      (if AUDIT_AGAINST set)
 *   5. write survivors to the `output` file (or default).
 *
 * Reports: T2 yield, T3 leakage count, T6 cache-hit rate, T7 wall-clock.
 *
 * T8 hard-lock: curl is the only new dep (universal).
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#define _POSIX_C_SOURCE 200809L

#include "llm_corpus_source.h"
#include "microgpt_oql.h"
#include "pipeline_ir/pipeline_ir.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char *slurp_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    if (len_out) *len_out = got;
    return buf;
}

/* ============================================================
 *  Extract one (prompt, graph) pair from an LLM response.
 *
 *  Tolerant parser: looks for a "// <prompt-text>" line followed
 *  somewhere by an "@graph ... @end" block.  Multiple candidates
 *  per response are split.
 * ============================================================ */

typedef struct {
    char *prompt;       /* heap-owned */
    char *graph;        /* heap-owned, contains @graph...@end */
} LlmPair;

/* Find the next "// " line + graph block in `body` after position *pos.
 * On success, fills *out and advances *pos.  Returns 1 if a pair was
 * found, 0 if not (end of input). */
static int extract_pair(const char *body, size_t *pos, LlmPair *out) {
    const char *p = body + *pos;
    while (*p) {
        /* Find "// " prompt line. */
        const char *prompt_start = strstr(p, "// ");
        if (!prompt_start) return 0;
        /* End of prompt = first newline. */
        const char *prompt_end = strchr(prompt_start, '\n');
        if (!prompt_end) prompt_end = prompt_start + strlen(prompt_start);
        /* Find @graph after the prompt line. */
        const char *graph_start = strstr(prompt_end, "@graph");
        if (!graph_start) return 0;
        const char *graph_end = strstr(graph_start, "@end");
        if (!graph_end) return 0;
        graph_end += 4; /* include "@end" */
        /* Build the pair. */
        size_t plen = (size_t)(prompt_end - (prompt_start + 3));
        size_t glen = (size_t)(graph_end - graph_start);
        out->prompt = (char *)malloc(plen + 1);
        out->graph = (char *)malloc(glen + 1);
        if (!out->prompt || !out->graph) {
            free(out->prompt); free(out->graph);
            return 0;
        }
        memcpy(out->prompt, prompt_start + 3, plen);
        out->prompt[plen] = '\0';
        memcpy(out->graph, graph_start, glen);
        out->graph[glen] = '\0';
        /* Strip trailing whitespace from prompt. */
        while (plen > 0 && isspace((unsigned char)out->prompt[plen - 1])) {
            out->prompt[--plen] = '\0';
        }
        *pos = (size_t)(graph_end - body);
        return 1;
    }
    return 0;
}

/* Look for the audit held-out's prompt list and load into an array. */
static int load_heldout_prompts(const char *path, char ***prompts_out) {
    *prompts_out = NULL;
    size_t flen = 0;
    char *body = slurp_file(path, &flen);
    if (!body) return 0;
    /* Parse: lines starting "// " are prompts (the rest are headers,
     * graph blocks, comments). */
    int cap = 64, n = 0;
    char **prompts = (char **)calloc((size_t)cap, sizeof(char *));
    char *line = body;
    while (line && *line) {
        char *eol = strchr(line, '\n');
        size_t llen = eol ? (size_t)(eol - line) : strlen(line);
        if (llen > 3 && line[0] == '/' && line[1] == '/' && line[2] == ' ') {
            if (n >= cap) {
                cap *= 2;
                char **np = (char **)realloc(prompts, (size_t)cap * sizeof(char *));
                if (!np) break;
                prompts = np;
            }
            prompts[n] = (char *)malloc(llen - 3 + 1);
            if (prompts[n]) {
                memcpy(prompts[n], line + 3, llen - 3);
                prompts[n][llen - 3] = '\0';
                n++;
            }
        }
        if (!eol) break;
        line = eol + 1;
    }
    free(body);
    *prompts_out = prompts;
    return n;
}

/* ============================================================
 *  Filter loop for one CREATE CORPUS … FROM LLM statement.
 * ============================================================ */

typedef struct {
    int emissions;
    int parse_failures;   /* pipeline_parse_text_tolerant fail */
    int verify_failures;  /* pipeline_verify fail after repair */
    int audit_failures;   /* Jaccard ≥ threshold against held-out */
    int survivors;
    int cache_hits;
    int curl_attempts;    /* total */
} FilterStats;

static double get_double_kv(const OqlKV *kv, const char *key, double dflt) {
    const char *v = oql_kv_get(kv, key);
    return v ? atof(v) : dflt;
}
static int get_int_kv(const OqlKV *kv, const char *key, int dflt) {
    const char *v = oql_kv_get(kv, key);
    return v ? atoi(v) : dflt;
}
static const char *get_str_kv(const OqlKV *kv, const char *key,
                              const char *dflt) {
    const char *v = oql_kv_get(kv, key);
    return v ? v : dflt;
}

static int run_one_llm_corpus(const OqlCreateCorpusLlm *cc,
                              FILE *log, FilterStats *st) {
    memset(st, 0, sizeof(*st));

    /* Resolve KV defaults. */
    int  count       = get_int_kv(cc->with_kv, "count", 1000);
    int  seed        = get_int_kv(cc->with_kv, "seed", 1337);
    int  max_retries = get_int_kv(cc->with_kv, "max_retries", 3);
    const char *cache = get_str_kv(cc->with_kv, "cache", ".oql_llm_cache");
    const char *output = get_str_kv(cc->with_kv, "output", NULL);
    if (!output) {
        fprintf(log, "ERROR: no output= in WITH clause; nothing to write to.\n");
        return -1;
    }
    double threshold = 0.7;
    const char *audit_path = NULL;
    if (cc->audit_held_out) {
        /* Map held-out NAME to its file path.  For now we accept a
         * convention: the named held-out is the v2 sealed file under
         * demos/wiring_organelle/. */
        if (strcmp(cc->audit_held_out, "scaling_heldout_v2") == 0 ||
            strcmp(cc->audit_held_out, "heldout_v2") == 0 ||
            strcmp(cc->audit_held_out, "held_out_v2") == 0) {
            audit_path = "../demos/wiring_organelle/pipeline_corpus_scaling_heldout_v2.txt";
        }
        if (cc->audit_with) {
            threshold = get_double_kv(cc->audit_with, "threshold", 0.7);
        }
    }

    /* Load held-out prompts (for audit). */
    char **heldout = NULL;
    int n_heldout = 0;
    if (audit_path) {
        n_heldout = load_heldout_prompts(audit_path, &heldout);
        fprintf(log, "[audit] loaded %d held-out prompts from %s "
                "(threshold=%.2f)\n",
                n_heldout, audit_path, threshold);
    }

    /* Init LLM source. */
    LlmSource src;
    memset(&src, 0, sizeof(src));
    src.model_id     = cc->model_id;
    src.endpoint_url = cc->endpoint_url;
    src.cache_dir    = cache;
    src.seed         = seed;
    src.max_retries  = max_retries;
    src.verbose      = 0;

    /* Health check up-front (T1 prerequisite). */
    fprintf(log, "[health] checking endpoint %s for model '%s'...\n",
            src.endpoint_url ? src.endpoint_url : "http://127.0.0.1:1234",
            src.model_id);
    if (llm_health_check(&src, log) != 0) {
        fprintf(log, "FATAL: LM Studio endpoint unreachable or model missing — "
                "STOP per E12 §1.5 skip rule.\n");
        for (int i = 0; i < n_heldout; i++) free(heldout[i]);
        free(heldout);
        return -1;
    }

    /* Open output. */
    FILE *out = fopen(output, "wb");
    if (!out) {
        fprintf(log, "ERROR: cannot open output '%s' for write\n", output);
        for (int i = 0; i < n_heldout; i++) free(heldout[i]);
        free(heldout);
        return -1;
    }
    fprintf(out, "# E12 LLM-generated wiring corpus\n");
    fprintf(out, "# model=%s endpoint=%s seed=%d count_target=%d\n",
            src.model_id,
            src.endpoint_url ? src.endpoint_url : "http://127.0.0.1:1234",
            seed, count);

    /* Generate. */
    time_t t0 = time(NULL);
    int target = count;
    int max_attempts = count * max_retries;
    int attempts = 0;
    while (st->survivors < target && attempts < max_attempts) {
        attempts++;
        /* Build a seed-perturbed prompt — chain attempt index into the
         * prompt so cache keys differ across attempts.  This is how we
         * get N distinct emissions from one PROMPT template. */
        size_t plen = strlen(cc->prompt) + 64;
        char *prompt_n = (char *)malloc(plen);
        snprintf(prompt_n, plen, "%s\n\n[Example #%d, seed=%d]",
                 cc->prompt, attempts, seed);
        /* Perturb the source seed too. */
        LlmSource ps = src;
        ps.seed = seed + attempts;

        char *body = NULL;
        size_t blen = 0;
        LlmEmitStats es;
        if (llm_emit(&ps, prompt_n, &body, &blen, &es, log) != 0) {
            free(prompt_n);
            st->emissions++;
            continue;
        }
        free(prompt_n);
        st->emissions++;
        st->curl_attempts += es.curl_attempts;
        if (es.cache_hit) st->cache_hits++;

        /* Walk all (prompt, graph) pairs in body. */
        size_t pos = 0;
        LlmPair pair;
        while (extract_pair(body, &pos, &pair)) {
            /* Verify via libpipeline_ir. */
            int verify_ok = 1;
            if (cc->verify_via_pipeline_ir) {
                Pipeline *p = pipeline_parse_text_tolerant(pair.graph);
                if (!p) {
                    st->parse_failures++;
                    verify_ok = 0;
                } else {
                    PipelineRepairReport rep;
                    pipeline_repair(p, &rep);
                    if (pipeline_verify(p) != 0) {
                        st->verify_failures++;
                        verify_ok = 0;
                    }
                    pipeline_free(p);
                }
            }
            /* Audit. */
            int audit_ok = 1;
            if (verify_ok && n_heldout > 0) {
                for (int i = 0; i < n_heldout; i++) {
                    double j = llm_jaccard_bow(pair.prompt, heldout[i]);
                    if (j >= threshold) {
                        st->audit_failures++;
                        audit_ok = 0;
                        fprintf(log, "[audit] REJECT (j=%.3f against held-out): \"%s\"\n",
                                j, pair.prompt);
                        break;
                    }
                }
            }
            if (verify_ok && audit_ok) {
                fprintf(out, "// %s\n%s\n---\n", pair.prompt, pair.graph);
                st->survivors++;
                if (st->survivors % 100 == 0) {
                    time_t elapsed = time(NULL) - t0;
                    fprintf(log, "[progress] %d survivors / %d attempts (%.1f s elapsed)\n",
                            st->survivors, attempts, (double)elapsed);
                }
            }
            free(pair.prompt);
            free(pair.graph);
            if (st->survivors >= target) break;
        }
        free(body);
    }
    fclose(out);

    time_t t1 = time(NULL);
    double seconds = (double)(t1 - t0);
    fprintf(log, "\n[E12 stats]\n");
    fprintf(log, "  emissions:           %d\n", st->emissions);
    fprintf(log, "  parse_failures:      %d\n", st->parse_failures);
    fprintf(log, "  verify_failures:     %d\n", st->verify_failures);
    fprintf(log, "  audit_failures (T3): %d\n", st->audit_failures);
    fprintf(log, "  survivors:           %d (target %d)\n",
            st->survivors, target);
    fprintf(log, "  cache_hits (T6):     %d / %d emissions = %.1f%%\n",
            st->cache_hits, st->emissions,
            st->emissions > 0 ? 100.0 * st->cache_hits / st->emissions : 0.0);
    fprintf(log, "  wall_clock (T7):     %.1f s = %.2f hours\n",
            seconds, seconds / 3600.0);
    /* T2: verify pass rate = survivors / pairs-considered.  For the
     * smoke set we report just the verify+audit overall yield. */
    int total_pairs = st->survivors + st->parse_failures
                    + st->verify_failures + st->audit_failures;
    if (total_pairs > 0) {
        fprintf(log, "  yield (T2):          %d / %d pairs = %.1f%%\n",
                st->survivors, total_pairs,
                100.0 * st->survivors / total_pairs);
    }

    for (int i = 0; i < n_heldout; i++) free(heldout[i]);
    free(heldout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <script.oql>\n"
                "       %s --health-only      (just check endpoint)\n",
                argv[0], argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--health-only") == 0) {
        LlmSource s;
        memset(&s, 0, sizeof(s));
        s.model_id = (argc > 2) ? argv[2] : "qwen/qwen3.6-35b-a3b";
        s.endpoint_url = (argc > 3) ? argv[3] : "http://127.0.0.1:1234";
        s.max_retries = 3;
        int rc = llm_health_check(&s, stderr);
        return rc == 0 ? 0 : 1;
    }
    size_t src_len = 0;
    char *src = slurp_file(argv[1], &src_len);
    if (!src) {
        fprintf(stderr, "ERROR: cannot read %s\n", argv[1]);
        return 1;
    }
    OqlScript *script = oql_parse(src);
    free(src);
    if (!script || script->error) {
        fprintf(stderr, "PARSE ERROR: %s\n",
                script && script->error ? script->error : "(null)");
        if (script) oql_script_free(script);
        return 1;
    }
    int rc = 0;
    for (OqlStmt *s = script->head; s; s = s->next) {
        if (s->verb != OQL_VERB_CREATE_CORPUS_LLM) continue;
        FilterStats st;
        int r = run_one_llm_corpus(&s->u.create_corpus_llm, stderr, &st);
        if (r != 0) { rc = 1; break; }
    }
    oql_script_free(script);
    return rc;
}
