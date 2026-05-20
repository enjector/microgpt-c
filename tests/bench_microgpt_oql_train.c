/*
 * tests/bench_microgpt_oql_train.c — Experiment E10 long-horizon
 * loss-curve fidelity benchmark.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Runs the pre-registered E10 §1.4 T3 measurement at the exact sample
 * steps the spec calls for (100, 500, 1000, 2500, 5000) so the writeup
 * can quote per-step delta numbers.  Kept out of the CI smoke binary
 * (test_microgpt_oql_train) because 5000 steps takes ~3-10 seconds,
 * which is too long for the per-commit hot path.
 *
 * Run from build/ as:  ./bench_microgpt_oql_train
 * No asserts; prints a per-step delta table to stdout.
 */

#include "../src/microgpt.h"
#include "../src/microgpt_oql.h"
#include "../src/oql_runtime_train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_SEED       1337u
#define BENCH_NUM_STEPS  5000
#define BENCH_BATCH      4
#define BENCH_LR         0.01

static const char *bench_corpus_path(void) {
    static const char *candidates[] = {
        "c_names.txt",
        "../demos/character-level/names/c_names.txt",
        NULL
    };
    for (int i = 0; candidates[i]; ++i) {
        FILE *f = fopen(candidates[i], "r");
        if (f) { fclose(f); return candidates[i]; }
    }
    return NULL;
}

/* Run the names_demo training loop directly. */
static int run_c_loop(const char *path, int steps, double *loss_out) {
    seed_rng(BENCH_SEED); srand(BENCH_SEED);
    MicrogptConfig cfg = microgpt_default_config();
    cfg.batch_size = BENCH_BATCH; cfg.num_steps = steps;
    Docs docs = {0};
    if (load_docs(path, &docs, cfg.max_docs) != 0) return -1;
    shuffle_docs(&docs);
    Vocab vocab = {0}; build_vocab(&docs, &vocab);
    Model *model = model_create(vocab.vocab_size, &cfg);
    size_t np = model_num_params(model);
    scalar_t *g = (scalar_t *)calloc(np, sizeof(scalar_t));
    scalar_t *m_ = (scalar_t *)calloc(np, sizeof(scalar_t));
    scalar_t *v_ = (scalar_t *)calloc(np, sizeof(scalar_t));
    int nl = cfg.n_layer, bs_ = cfg.block_size;
    scalar_t **ks = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t*));
    scalar_t **vs = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t*));
    size_t *cl = (size_t *)calloc((size_t)nl, sizeof(size_t));
    for (int L = 0; L < nl; L++) {
        ks[L] = kv_cache_alloc(&cfg); vs[L] = kv_cache_alloc(&cfg);
    }
    size_t *tok = (size_t *)malloc(((size_t)bs_ + 2) * sizeof(size_t));
    size_t di = 0;
    for (int step = 0; step < steps; step++) {
        memset(g, 0, np * sizeof(scalar_t));
        scalar_t batch_loss = 0; size_t bp = 0;
        for (int b = 0; b < cfg.batch_size; b++) {
            for (int L = 0; L < nl; L++) cl[L] = 0;
            const char *d = docs.lines[di % docs.num_docs];
            size_t dl = docs.doc_lens[di % docs.num_docs];
            di++;
            size_t nt = tokenize(d, dl, &vocab, tok, (size_t)bs_ + 2);
            size_t n = nt - 1; if (n > (size_t)bs_) n = (size_t)bs_;
            bp += n;
            for (size_t p = 0; p < n; p++) {
                batch_loss += forward_backward_one(model, tok[p], p,
                    tok[p+1], ks, vs, cl, g);
            }
        }
        scalar_t mean = (bp > 0) ? batch_loss / (scalar_t)bp : (scalar_t)0;
        if (bp > 0) for (size_t i = 0; i < np; i++) g[i] /= (scalar_t)bp;
        adam_step(model, g, m_, v_, step);
        loss_out[step] = (double)mean;
    }
    for (int L = 0; L < nl; L++) { kv_cache_free(ks[L]); kv_cache_free(vs[L]); }
    free(ks); free(vs); free(cl); free(tok); free(g); free(m_); free(v_);
    model_free(model); free(vocab.chars); free_docs(&docs);
    return 0;
}

/* Run the same training loop through OQL. */
static int run_oql_loop(const char *path, int steps, double *loss_out) {
    char src[1024];
    snprintf(src, sizeof(src),
        "CREATE CORPUS bench FROM FILE '%s';\n"
        "CREATE ORGANELLE poet;\n"
        "TRAIN poet ON bench WITH STEPS = %d, LR = %g, "
        "BATCH_SIZE = %d, SEED = %u;\n",
        path, steps, BENCH_LR, BENCH_BATCH, BENCH_SEED);
    OqlScript *s = oql_parse(src);
    if (!s || s->error) { oql_script_free(s); return -1; }
    OqlRuntime rt; oql_runtime_init(&rt);
    oql_runtime_attach_loss_log(&rt, loss_out, steps);
    int failed_idx = 0;
    oql_status st = oql_execute_with_runtime(s, &rt, NULL, &failed_idx);
    oql_runtime_dispose(&rt); oql_script_free(s);
    return (st == OQL_OK) ? 0 : -1;
}

int main(void) {
    const char *corpus = bench_corpus_path();
    if (!corpus) { fprintf(stderr, "c_names.txt not found\n"); return 1; }

    int steps = BENCH_NUM_STEPS;
    double *c_loss = (double *)calloc(steps, sizeof(double));
    double *oql_loss = (double *)calloc(steps, sizeof(double));
    printf("E10 bench: %d steps, batch %d, seed %u, lr %g\n",
           steps, BENCH_BATCH, BENCH_SEED, BENCH_LR);

    clock_t t1 = clock();
    if (run_c_loop(corpus, steps, c_loss)) {
        fprintf(stderr, "C baseline failed\n"); return 2;
    }
    double c_sec = (double)(clock() - t1) / CLOCKS_PER_SEC;
    clock_t t2 = clock();
    if (run_oql_loop(corpus, steps, oql_loss)) {
        fprintf(stderr, "OQL run failed\n"); return 3;
    }
    double o_sec = (double)(clock() - t2) / CLOCKS_PER_SEC;

    /* Pre-reg §1.4 T3 sample points. */
    const int samples[] = { 100, 500, 1000, 2500, 5000 };
    const int n_samples = (int)(sizeof(samples) / sizeof(samples[0]));
    printf("\nT3 — Loss-curve fidelity (pre-reg §1.4)\n");
    printf("step    c_loss      oql_loss    |delta|/c\n");
    double max_rel = 0.0;
    for (int i = 0; i < n_samples; i++) {
        int s = samples[i] - 1;
        if (s >= steps) continue;
        double c = c_loss[s], q = oql_loss[s];
        double rel = (c > 1e-9) ? fabs(q - c) / c : fabs(q - c);
        if (rel > max_rel) max_rel = rel;
        printf("%4d    %.4f      %.4f      %.4e\n", samples[i], c, q, rel);
    }
    printf("\nmax |delta|/c = %.4e  (floor 0.10; skip-trigger 0.25)\n", max_rel);
    printf("C    timing: %.2fs    OQL  timing: %.2fs\n", c_sec, o_sec);
    free(c_loss); free(oql_loss);
    return (max_rel <= 0.10) ? 0 : 4;
}
