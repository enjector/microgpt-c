/*
 * tests/test_microgpt_oql_train.c — Experiment E10 loss-curve fidelity smoke test
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Lives in its own test binary so it can be linked against the *default*
 * microgpt_lib variant (N_EMBD=16 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=16 MLP_DIM=64),
 * which matches names_demo's dims exactly.  This eliminates the E09 §3.4
 * compile-time-macro silent-failure mode while keeping the test fast
 * (~4K params, < 10 sec at 500 steps).
 *
 * Pre-registered targets covered (per experiments/E10-oql-train-wiring.md §1.4):
 *
 *   T1: TRAIN clause parses cleanly  →  test_e10_train_parse_all_clauses
 *   T2: TRAIN executes end-to-end    →  test_e10_train_executes
 *   T3: Loss-curve fidelity ±10%     →  test_e10_train_loss_curve_fidelity
 *   T4: Checkpoint round-trip ≤ 1e-5 →  test_e10_train_checkpoint_round_trip
 *
 * T1 is also covered by tests/test_microgpt_oql.c::test_e10_train_full_clause_list_parses;
 * a duplicate assertion lives here so the smoke binary stands alone.
 */

#include "../src/microgpt.h"
#include "../src/microgpt_oql.h"
#include "../src/oql_runtime_train.h"
#include "test.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ────────────────────────────────────────────────────────────────
 *  Configuration — keep small so CI completes in seconds.
 * ──────────────────────────────────────────────────────────────── */

#define E10_SEED        1337u
#define E10_NUM_STEPS   200      /* enough to see loss descend ~30-50% */
#define E10_BATCH_SIZE  4        /* keep cheap */
#define E10_LR          0.01     /* matches engine default */

/* The checkpoint path is relative to CTest's WORKING_DIRECTORY (build/). */
#define E10_CKPT_PATH   "checkpoints/e10_oql_names.ckpt"

/* Locate the names corpus.  CTest WORKING_DIRECTORY is build/ so we try
 * the build-side copy first (each demo POST_BUILD copies its data file),
 * then fall back to the source tree.  */
static const char *e10_corpus_path(void) {
    static const char *candidates[] = {
        "c_names.txt",
        "../demos/character-level/names/c_names.txt",
        "demos/character-level/names/c_names.txt",
        NULL
    };
    for (int i = 0; candidates[i]; ++i) {
        FILE *f = fopen(candidates[i], "r");
        if (f) { fclose(f); return candidates[i]; }
    }
    return NULL;
}

/* Build the equivalent OQL script for our smoke test.  Mirrors §1.3 of
 * the pre-reg: a single TRAIN with all locked sub-clauses.  The
 * corpus is registered via CREATE CORPUS (the new E10 object type). */
static char *e10_make_script(const char *corpus_path, int with_save) {
    /* "CREATE CORPUS shake FROM FILE '<path>';
     *  CREATE ORGANELLE poet;
     *  TRAIN poet ON shake WITH STEPS=200, LR=0.01, BATCH_SIZE=4,
     *      SEED=1337[, SAVE='checkpoints/...'];" */
    char *buf = (char *)malloc(1024);
    if (!buf) return NULL;
    if (with_save) {
        snprintf(buf, 1024,
            "CREATE CORPUS names_tiny FROM FILE '%s';\n"
            "CREATE ORGANELLE poet;\n"
            "TRAIN poet ON names_tiny "
            "WITH STEPS = %d, LR = %g, BATCH_SIZE = %d, SEED = %u, "
            "SAVE = '%s';\n",
            corpus_path, E10_NUM_STEPS, E10_LR, E10_BATCH_SIZE, E10_SEED,
            E10_CKPT_PATH);
    } else {
        snprintf(buf, 1024,
            "CREATE CORPUS names_tiny FROM FILE '%s';\n"
            "CREATE ORGANELLE poet;\n"
            "TRAIN poet ON names_tiny "
            "WITH STEPS = %d, LR = %g, BATCH_SIZE = %d, SEED = %u;\n",
            corpus_path, E10_NUM_STEPS, E10_LR, E10_BATCH_SIZE, E10_SEED);
    }
    return buf;
}

/* ────────────────────────────────────────────────────────────────
 *  T1 — locked clause list parses
 * ──────────────────────────────────────────────────────────────── */
enx_test(test_e10_train_parse_all_clauses) {
    OqlScript *s = oql_parse(
        "TRAIN poet ON names_tiny WITH "
        "ROLE = planner, STEPS = 2000, LR = 0.001, "
        "BATCH_SIZE = 4, SAVE = 'checkpoints/poet.ckpt', SEED = 1337;");
    enx_assert_ptr_not_null(s);
    enx_assert_true(s->error == NULL);
    enx_assert_equal_int(s->head->verb, OQL_VERB_TRAIN);
    enx_assert_equal_string(s->head->u.train.target, "poet");
    enx_assert_equal_string(s->head->u.train.on_src.value, "names_tiny");
    OqlKV *kv = s->head->u.train.with_kv;
    enx_assert_equal_string(oql_kv_get(kv, "ROLE"),       "planner");
    enx_assert_equal_string(oql_kv_get(kv, "STEPS"),      "2000");
    enx_assert_equal_string(oql_kv_get(kv, "LR"),         "0.001");
    enx_assert_equal_string(oql_kv_get(kv, "BATCH_SIZE"), "4");
    enx_assert_equal_string(oql_kv_get(kv, "SAVE"),       "checkpoints/poet.ckpt");
    enx_assert_equal_string(oql_kv_get(kv, "SEED"),       "1337");
    oql_script_free(s);
}

/* ────────────────────────────────────────────────────────────────
 *  T2 — TRAIN executes a tiny names-corpus run end-to-end
 * ──────────────────────────────────────────────────────────────── */
enx_test(test_e10_train_executes) {
    const char *corpus = e10_corpus_path();
    if (!corpus) {
        printf("test_e10_train_executes: c_names.txt not found; skipping\n");
        return;
    }
    char *src = e10_make_script(corpus, /*with_save*/ 0);
    enx_assert_ptr_not_null(src);
    OqlScript *script = oql_parse(src);
    free(src);
    enx_assert_ptr_not_null(script);
    enx_assert_true(script->error == NULL);

    OqlRuntime rt;
    oql_runtime_init(&rt);
    int failed_idx = 0;
    oql_status st = oql_execute_with_runtime(script, &rt, NULL, &failed_idx);
    enx_assert_equal_int(OQL_OK, st);
    /* TRAIN populates rt->last_train_*. */
    enx_assert_equal_int(E10_NUM_STEPS, rt.last_train_steps);
    enx_assert_true(rt.last_train_final_loss > 0.0);
    enx_assert_true(rt.last_train_total_seconds >= 0.0);
    /* The organelle slot is now "loaded" with the in-memory trained model. */
    OqlOrganelle *o = oql_runtime_find_organelle(&rt, "poet");
    enx_assert_ptr_not_null(o);
    enx_assert_equal_int(1, o->loaded);
    enx_assert_ptr_not_null(o->model);
    oql_runtime_dispose(&rt);
    oql_script_free(script);
}

/* ────────────────────────────────────────────────────────────────
 *  Loss-curve baseline — runs the same training loop the OQL adapter
 *  runs, but driven directly through the engine API.  No OQL parsing,
 *  no script execution.  Returns the per-step mean loss into c_loss[].
 *
 *  This is the *equivalent* of the names_demo training loop with the
 *  same seed and hyperparameters.  The OQL adapter's loop in
 *  src/oql_runtime_train.c was written to be byte-for-byte equivalent
 *  to this body (down to the doc_idx wrap, batch accumulation, and
 *  cache-len reset order).  T3 asserts the loss curves agree within
 *  the pre-registered ±10% relative budget.
 * ──────────────────────────────────────────────────────────────── */
static int e10_c_loss_curve(const char *corpus_path,
                            int num_steps, int batch_size,
                            unsigned int seed, double *c_loss) {
    seed_rng(seed);
    srand(seed);
    MicrogptConfig cfg = microgpt_default_config();
    cfg.batch_size = batch_size;
    cfg.num_steps  = num_steps;

    Docs docs = {0};
    if (load_docs(corpus_path, &docs, cfg.max_docs) != 0) return -1;
    shuffle_docs(&docs);
    Vocab vocab = {0};
    build_vocab(&docs, &vocab);
    Model *model = model_create(vocab.vocab_size, &cfg);
    if (!model) {
        free(vocab.chars); free_docs(&docs);
        return -1;
    }
    size_t nparams = model_num_params(model);
    scalar_t *grad = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    scalar_t *m_b  = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    scalar_t *v_b  = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    const int nl = cfg.n_layer, bs = cfg.block_size;
    scalar_t **keys = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    scalar_t **vals = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    size_t *cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    for (int L = 0; L < nl; L++) {
        keys[L] = kv_cache_alloc(&cfg);
        vals[L] = kv_cache_alloc(&cfg);
    }
    size_t *tok = (size_t *)malloc(((size_t)bs + 2) * sizeof(size_t));
    size_t doc_idx = 0;
    for (int step = 0; step < num_steps; step++) {
        memset(grad, 0, nparams * sizeof(scalar_t));
        scalar_t batch_loss = 0;
        size_t batch_positions = 0;
        for (int b = 0; b < cfg.batch_size; b++) {
            for (int L = 0; L < nl; L++) cache_len[L] = 0;
            const char *doc = docs.lines[doc_idx % docs.num_docs];
            size_t doc_len = docs.doc_lens[doc_idx % docs.num_docs];
            doc_idx++;
            size_t n_tok = tokenize(doc, doc_len, &vocab, tok,
                                    (size_t)bs + 2);
            size_t n = n_tok - 1;
            if (n > (size_t)bs) n = (size_t)bs;
            batch_positions += n;
            for (size_t pos = 0; pos < n; pos++) {
                scalar_t loss = forward_backward_one(model,
                    tok[pos], pos, tok[pos + 1], keys, vals, cache_len, grad);
                batch_loss += loss;
            }
        }
        scalar_t mean = (batch_positions > 0)
            ? (batch_loss / (scalar_t)batch_positions) : (scalar_t)0;
        if (batch_positions > 0) {
            for (size_t i = 0; i < nparams; i++)
                grad[i] /= (scalar_t)batch_positions;
        }
        adam_step(model, grad, m_b, v_b, step);
        c_loss[step] = (double)mean;
    }
    for (int L = 0; L < nl; L++) {
        kv_cache_free(keys[L]); kv_cache_free(vals[L]);
    }
    free(keys); free(vals); free(cache_len);
    free(tok); free(grad); free(m_b); free(v_b);
    model_free(model);
    free(vocab.chars);
    free_docs(&docs);
    return 0;
}

/* T3 — Loss-curve fidelity vs the equivalent C training loop.
 *
 * Pre-reg §1.4 T3 floor: |oql_loss - c_loss| / c_loss ≤ 0.10 at steps
 * 100, 500, 1000, 2500, 5000.  The smoke test only runs to 200 steps
 * (CI budget); we sample at 50, 100, 150, 200 and assert the same
 * ratio.  The longer-step measurement is a follow-up benchmark (not
 * the per-CI smoke gate).  Skip rule (§1.5): if any step's delta > 0.25,
 * STOP — the binding is wrong (gradients dropped / optimiser steps
 * misordered). */
enx_test(test_e10_train_loss_curve_fidelity) {
    const char *corpus = e10_corpus_path();
    if (!corpus) {
        printf("test_e10_train_loss_curve_fidelity: c_names.txt not found; "
               "skipping\n");
        return;
    }
    /* Allocate per-step loss buffers. */
    double *c_loss = (double *)calloc(E10_NUM_STEPS, sizeof(double));
    double *oql_loss = (double *)calloc(E10_NUM_STEPS, sizeof(double));
    enx_assert_ptr_not_null(c_loss);
    enx_assert_ptr_not_null(oql_loss);

    /* Baseline run — C training loop with same hyperparameters / seed. */
    int rc = e10_c_loss_curve(corpus, E10_NUM_STEPS, E10_BATCH_SIZE,
                              E10_SEED, c_loss);
    enx_assert_equal_int(0, rc);

    /* OQL run with loss log attached. */
    char *src = e10_make_script(corpus, /*with_save*/ 0);
    enx_assert_ptr_not_null(src);
    OqlScript *script = oql_parse(src);
    free(src);
    enx_assert_true(script && script->error == NULL);
    OqlRuntime rt;
    oql_runtime_init(&rt);
    oql_runtime_attach_loss_log(&rt, oql_loss, E10_NUM_STEPS);
    int failed_idx = 0;
    oql_status st = oql_execute_with_runtime(script, &rt, NULL, &failed_idx);
    enx_assert_equal_int(OQL_OK, st);

    /* Per-step delta at sample positions. */
    const int sample_steps[] = { 50, 100, 150, 200 };
    const int n_sample = (int)(sizeof(sample_steps) / sizeof(sample_steps[0]));
    double max_rel_delta = 0.0;
    for (int i = 0; i < n_sample; i++) {
        int s = sample_steps[i] - 1;          /* zero-indexed buffer */
        if (s < 0 || s >= E10_NUM_STEPS) continue;
        double c = c_loss[s];
        double q = oql_loss[s];
        double rel = (c > 1e-9) ? fabs(q - c) / c : fabs(q - c);
        if (rel > max_rel_delta) max_rel_delta = rel;
        printf("[E10 T3] step %4d:  c=%.4f  oql=%.4f  |delta|/c=%.4f\n",
               sample_steps[i], c, q, rel);
        /* Hard-locked skip rule: > 0.25 means the binding is wrong. */
        enx_assert_true(rel <= 0.25);
    }
    printf("[E10 T3] max |delta|/c across sampled steps = %.4f "
           "(pre-reg floor 0.10; skip-trigger 0.25)\n", max_rel_delta);
    /* The pre-reg target is ≤ 0.10.  Bit-identical loss is the
     * realistic outcome here because the adapter mirrors the engine
     * loop byte-for-byte (same RNG seed, same shuffle, same
     * forward_backward_one call order, same adam_step call). */
    enx_assert_true(max_rel_delta <= 0.10);

    oql_runtime_dispose(&rt);
    oql_script_free(script);
    free(c_loss); free(oql_loss);
}

/* ────────────────────────────────────────────────────────────────
 *  T4 — Checkpoint round-trip.  After TRAIN ... SAVE writes a file,
 *  CREATE ORGANELLE FROM CHECKPOINT loads it and inference produces
 *  per-logit outputs within 1e-5 of the just-trained organelle.
 * ──────────────────────────────────────────────────────────────── */

/* Helper: forward_inference over a fixed prompt and return the
 * post-final-step logits as a malloc'd vector.  Caller frees. */
static scalar_t *e10_logits_for(Model *model, const MicrogptConfig *cfg,
                                size_t bos_id, size_t vocab_size) {
    const int nl = cfg->n_layer, bs = cfg->block_size;
    (void)bs;
    scalar_t **keys = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    scalar_t **vals = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    size_t *cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    for (int L = 0; L < nl; L++) {
        keys[L] = kv_cache_alloc(cfg);
        vals[L] = kv_cache_alloc(cfg);
    }
    scalar_t *logits = (scalar_t *)calloc(vocab_size, sizeof(scalar_t));
    /* One forward step with the BOS prompt.  This is enough to detect
     * any divergence in the saved/loaded weights — every weight matrix
     * participates. */
    forward_inference(model, bos_id, 0, keys, vals, cache_len, logits);
    for (int L = 0; L < nl; L++) {
        kv_cache_free(keys[L]); kv_cache_free(vals[L]);
    }
    free(keys); free(vals); free(cache_len);
    return logits;
}

enx_test(test_e10_train_checkpoint_round_trip) {
    const char *corpus = e10_corpus_path();
    if (!corpus) {
        printf("test_e10_train_checkpoint_round_trip: corpus not found; "
               "skipping\n");
        return;
    }

    /* Ensure the checkpoints dir exists (CTest WORKING_DIRECTORY=build). */
    int mkdir_rc = system("mkdir -p checkpoints 2>/dev/null");
    (void)mkdir_rc;

    /* 1. TRAIN with SAVE. */
    char *src = e10_make_script(corpus, /*with_save*/ 1);
    enx_assert_ptr_not_null(src);
    OqlScript *script = oql_parse(src);
    free(src);
    enx_assert_true(script && script->error == NULL);
    OqlRuntime rt;
    oql_runtime_init(&rt);
    int failed_idx = 0;
    oql_status st = oql_execute_with_runtime(script, &rt, NULL, &failed_idx);
    enx_assert_equal_int(OQL_OK, st);
    OqlOrganelle *trained = oql_runtime_find_organelle(&rt, "poet");
    enx_assert_ptr_not_null(trained);
    enx_assert_equal_int(1, trained->loaded);
    enx_assert_ptr_not_null(trained->model);

    /* Snapshot the trained logits for the BOS prompt. */
    const MicrogptConfig *cfg = model_config(trained->model);
    /* Vocab size is encoded in the model's lm_head dimension; we re-derive
     * by peeking at the saved checkpoint header (matches the load path). */
    size_t vocab_size = 0;
    {
        FILE *peek = fopen(E10_CKPT_PATH, "rb");
        enx_assert_ptr_not_null(peek);
        int header_step = 0;
        size_t header_vocab = 0;
        size_t r1 = fread(&header_step, sizeof(int), 1, peek);
        size_t r2 = fread(&header_vocab, sizeof(size_t), 1, peek);
        fclose(peek);
        enx_assert_equal_size((size_t)1, r1);
        enx_assert_equal_size((size_t)1, r2);
        vocab_size = header_vocab;
        enx_assert_true(vocab_size > 1);
        enx_assert_true(vocab_size <= (size_t)cfg->max_vocab);
    }
    /* Use the last vocab slot as BOS (matches build_vocab convention). */
    size_t bos_id = vocab_size - 1;
    scalar_t *trained_logits = e10_logits_for(trained->model, cfg,
                                              bos_id, vocab_size);
    enx_assert_ptr_not_null(trained_logits);

    /* 2. Load the just-saved checkpoint via CREATE ORGANELLE FROM CHECKPOINT
     *    and compare per-logit. */
    OqlRuntime rt2;
    oql_runtime_init(&rt2);
    char load_src[512];
    snprintf(load_src, sizeof(load_src),
        "CREATE ORGANELLE poet_reload FROM CHECKPOINT '%s';\n",
        E10_CKPT_PATH);
    OqlScript *load_script = oql_parse(load_src);
    enx_assert_true(load_script && load_script->error == NULL);
    failed_idx = 0;
    st = oql_execute_with_runtime(load_script, &rt2, NULL, &failed_idx);
    enx_assert_equal_int(OQL_OK, st);
    OqlOrganelle *reloaded_slot = oql_runtime_find_organelle(&rt2, "poet_reload");
    enx_assert_ptr_not_null(reloaded_slot);
    /* Trigger lazy load. */
    Model *reloaded = oql_runtime_load_organelle(&rt2, reloaded_slot, stdout);
    enx_assert_ptr_not_null(reloaded);

    const MicrogptConfig *cfg2 = model_config(reloaded);
    scalar_t *reload_logits = e10_logits_for(reloaded, cfg2,
                                             bos_id, vocab_size);
    enx_assert_ptr_not_null(reload_logits);

    /* Per-logit max delta. */
    double max_abs_delta = 0.0;
    for (size_t i = 0; i < vocab_size; i++) {
        double d = fabs((double)trained_logits[i] - (double)reload_logits[i]);
        if (d > max_abs_delta) max_abs_delta = d;
    }
    printf("[E10 T4] max |trained_logit - reload_logit| = %.6e "
           "(pre-reg floor 1e-5)\n", max_abs_delta);
    /* Pre-reg §1.4 T4 floor: outputs diverge beyond 1e-5 per-logit. */
    enx_assert_true(max_abs_delta <= 1e-5);

    free(trained_logits); free(reload_logits);
    oql_runtime_dispose(&rt2);
    oql_script_free(load_script);
    oql_runtime_dispose(&rt);
    oql_script_free(script);
}

/* ── Suite table ────────────────────────────────────────────── */

enx_test_case_t oql_train_tests[] = {
    enx_test_case(test_e10_train_parse_all_clauses),
    enx_test_case(test_e10_train_executes),
    enx_test_case(test_e10_train_loss_curve_fidelity),
    enx_test_case(test_e10_train_checkpoint_round_trip),
    enx_test_case_end()
};

int main(void) {
    test_suite suites[] = {
        {"oql_train_loss_curve_fidelity", oql_train_tests},
        {NULL, NULL},
    };
    return test_suite_run(suites) ? 0 : 1;
}
