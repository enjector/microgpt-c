/*
 * oql_runtime_train.c  —  OQL TRAIN adapter (Experiment E10)
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Binds OqlTrainSpec → engine training loop.  Mirrors the inline training
 * loop in demos/character-level/names/main.c exactly:
 *
 *   load_docs → shuffle_docs → build_vocab → model_create
 *     → loop { zero grads; for batch: forward_backward_one; adam_step }
 *     → optional checkpoint_save
 *     → leave Model* on the organelle for downstream RUN
 *
 * No new engine surface.  No new VM opcodes.  No modification to
 * src/microgpt.{h,c}.  Adapter is fully in OQL territory per the
 * experiments/E10-oql-train-wiring.md §1.3 Phase 3 constraint.
 *
 * Compile-time-macro caveat: the function uses `(scalar_t)spec->learning_rate`
 * for log printing only; the engine reads LEARNING_RATE / NUM_STEPS / etc.
 * from compile-time macros (cfg fields are display-only, except for vocab
 * sizing and the lifecycle quantities forward_backward_one needs).  To
 * faithfully match a particular C demo's loss curve, callers must run a
 * binary variant compiled with that demo's defines — see the `oql_names`
 * target in CMakeLists.txt.
 */

#include "oql_runtime_train.h"
#include "microgpt.h"
#include "microgpt_oql.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================
 *  Helpers
 * ============================================================ */

/* Slurp a file into a heap-allocated buffer, nul-terminated.  Used as a
 * fallback when the corpus's `contents` field is empty.  Returns NULL on
 * failure (file missing / OOM). */
static char *oql_train_slurp(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[r] = '\0';
    if (out_len) *out_len = r;
    return buf;
}

/* Default config used by TRAIN.  Built from the compile-time macros at
 * OQL-lib variant build time; the spec's per-statement overrides (steps /
 * LR / batch_size) take precedence in the training loop. */
static MicrogptConfig oql_train_default_cfg(void) {
    MicrogptConfig cfg = microgpt_default_config();
    cfg.n_embd     = N_EMBD;
    cfg.n_head     = N_HEAD;
    cfg.n_layer    = N_LAYER;
    cfg.block_size = BLOCK_SIZE;
    cfg.mlp_dim    = MLP_DIM;
    cfg.batch_size = BATCH_SIZE;
    cfg.num_steps  = NUM_STEPS;
    cfg.learning_rate = LEARNING_RATE;
    cfg.max_vocab    = MAX_VOCAB;
    cfg.max_docs     = MAX_DOCS;
    cfg.max_doc_len  = MAX_DOC_LEN;
    return cfg;
}

/* Resolve the corpus to a Docs structure.  Two paths supported:
 *  - spec->corpus_name is a registered OqlCorpus name → reuse path/slurp
 *  - spec->corpus_name is a bare file path → slurp directly
 *
 * In either case we then materialise the contents into a Docs view by
 * writing the slurped buffer to a temp file (cheap) and calling
 * load_docs — this keeps the engine-side API surface untouched.
 *
 * A more efficient path would expose an in-memory loader from
 * src/microgpt.c, but doing so would modify the frozen engine surface
 * (T5 hard-lock) — so we accept one temp-file round trip per TRAIN. */
static int oql_train_open_docs(OqlRuntime *rt, const OqlTrainSpec *spec,
                               Docs *out_docs, const MicrogptConfig *cfg,
                               FILE *out) {
    if (!rt || !spec || !out_docs || !cfg) return -1;
    const char *resolved_path = NULL;

    OqlCorpus *c = oql_runtime_find_corpus(rt, spec->corpus_name);
    if (c) {
        if (!c->file_path[0]) {
            if (out) fprintf(out,
                "TRAIN: corpus '%s' registered but has no file path\n",
                spec->corpus_name);
            return -1;
        }
        resolved_path = c->file_path;
        /* Lazy slurp (so future TRAINs against same corpus can reuse). */
        if (!c->contents) {
            c->contents = oql_train_slurp(c->file_path, &c->contents_len);
            if (!c->contents && out) {
                fprintf(out,
                    "TRAIN: cannot read corpus file '%s' (corpus '%s')\n",
                    c->file_path, c->name);
            }
        }
    } else {
        /* Bare path fallback — allows scripts to skip CREATE CORPUS. */
        resolved_path = spec->corpus_name;
    }

    if (!resolved_path) return -1;
    if (load_docs(resolved_path, out_docs, cfg->max_docs) != 0) {
        if (out) fprintf(out,
            "TRAIN: load_docs('%s') failed\n", resolved_path);
        return -1;
    }
    return 0;
}

/* ============================================================
 *  Entry point
 * ============================================================ */

oql_status oql_run_train(OqlRuntime *rt,
                         const OqlTrainSpec *spec,
                         FILE *out) {
    if (!rt || !spec) return OQL_ERR_RUNTIME;

    /* INT8 quant builds reject training (checkpoint_save returns -1).
     * Surface this as NOT_IMPLEMENTED so callers can branch cleanly. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    if (out) fprintf(out, "TRAIN: training disabled under INT8 quantisation\n");
    return OQL_ERR_NOT_IMPLEMENTED;
#endif

    /* 1. Locate the organelle slot the TRAIN populates. */
    OqlOrganelle *org = oql_runtime_find_organelle(rt, spec->organelle_name);
    if (!org) {
        if (out) fprintf(out,
            "TRAIN: unknown organelle '%s' (must be declared via CREATE "
            "ORGANELLE first)\n",
            spec->organelle_name ? spec->organelle_name : "(null)");
        return OQL_ERR_RUNTIME;
    }
    if (org->loaded) {
        if (out) fprintf(out,
            "TRAIN: organelle '%s' already loaded — TRAIN expects a blank "
            "slot.  Drop and re-create the organelle to retrain.\n",
            org->name);
        return OQL_ERR_RUNTIME;
    }

    /* 2. Build the runtime config (compile-time-macro variant). */
    MicrogptConfig local_cfg = oql_train_default_cfg();
    /* Per-spec overrides — these affect the training loop's bookkeeping
     * but the engine's matmul shapes are still macro-bound, so changing
     * batch_size here is fine (loop counter), but n_embd / n_layer
     * cannot be tuned at runtime. */
    if (spec->batch_size > 0) local_cfg.batch_size = spec->batch_size;
    if (spec->steps > 0)      local_cfg.num_steps  = spec->steps;
    if (spec->learning_rate > 0.0)
        local_cfg.learning_rate = spec->learning_rate;

    /* 3. Seed RNGs.  srand affects shuffle_docs; seed_rng affects the
     * engine's internal LCG (weight init, sampling). */
    seed_rng(spec->seed);
    srand(spec->seed);

    /* 4. Load corpus → Docs. */
    Docs docs = {0};
    if (oql_train_open_docs(rt, spec, &docs, &local_cfg, out) != 0) {
        return OQL_ERR_RUNTIME;
    }
    shuffle_docs(&docs);

    /* 5. Build vocab. */
    Vocab vocab = {0};
    build_vocab(&docs, &vocab);

    /* 6. Allocate model. */
    Model *model = model_create(vocab.vocab_size, &local_cfg);
    if (!model) {
        if (out) fprintf(out,
            "TRAIN: model_create(vocab=%zu) failed\n", vocab.vocab_size);
        free(vocab.chars);
        free_docs(&docs);
        return OQL_ERR_OOM;
    }
    size_t nparams = model_num_params(model);

    /* 7. Allocate optimiser state. */
    scalar_t *grad_buffer = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    scalar_t *m_buf       = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    scalar_t *v_buf       = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    if (!grad_buffer || !m_buf || !v_buf) {
        free(grad_buffer); free(m_buf); free(v_buf);
        model_free(model);
        free(vocab.chars);
        free_docs(&docs);
        return OQL_ERR_OOM;
    }

    /* 8. Per-layer KV cache. */
    const int nl = local_cfg.n_layer;
    const int bs = local_cfg.block_size;
    scalar_t **keys = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    scalar_t **values = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    size_t *cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    if (!keys || !values || !cache_len) {
        free(keys); free(values); free(cache_len);
        free(grad_buffer); free(m_buf); free(v_buf);
        model_free(model);
        free(vocab.chars);
        free_docs(&docs);
        return OQL_ERR_OOM;
    }
    for (int L = 0; L < nl; L++) {
        keys[L]   = kv_cache_alloc(&local_cfg);
        values[L] = kv_cache_alloc(&local_cfg);
    }

    /* 9. Training loop — mirrors names_demo lines 180-240 exactly so the
     * loss-curve fidelity smoke test in tests/test_microgpt_oql_train.c
     * can compare per-step. */
    size_t *token_buf = (size_t *)malloc(((size_t)bs + 2) * sizeof(size_t));
    size_t doc_idx = 0;
    int num_steps = (spec->steps > 0) ? spec->steps : local_cfg.num_steps;
    clock_t t_start = clock();
    double final_mean_loss = 0.0;

    for (int step = 0; step < num_steps; step++) {
        memset(grad_buffer, 0, nparams * sizeof(scalar_t));
        scalar_t batch_loss = 0;
        size_t batch_positions = 0;
        for (int b = 0; b < local_cfg.batch_size; b++) {
            for (int L = 0; L < nl; L++) cache_len[L] = 0;
            const char *doc = docs.lines[doc_idx % docs.num_docs];
            size_t doc_len = docs.doc_lens[doc_idx % docs.num_docs];
            doc_idx++;
            size_t n_tok = tokenize(doc, doc_len, &vocab, token_buf,
                                    (size_t)bs + 2);
            size_t n = n_tok - 1;
            if (n > (size_t)bs) n = (size_t)bs;
            batch_positions += n;
            for (size_t pos = 0; pos < n; pos++) {
                scalar_t loss = forward_backward_one(model, token_buf[pos],
                    pos, token_buf[pos + 1], keys, values, cache_len,
                    grad_buffer);
                batch_loss += loss;
            }
        }
        scalar_t mean_loss = (batch_positions > 0)
            ? (batch_loss / (scalar_t)batch_positions)
            : (scalar_t)0;
        if (batch_positions > 0) {
            for (size_t i = 0; i < nparams; i++)
                grad_buffer[i] /= (scalar_t)batch_positions;
        }
        adam_step(model, grad_buffer, m_buf, v_buf, step);

        /* Record on the per-step loss log if attached. */
        if (rt->loss_log && step < rt->loss_log_cap) {
            rt->loss_log[step] = (double)mean_loss;
        }
        final_mean_loss = (double)mean_loss;

        /* Periodic progress (matches names_demo's "every 100 steps" cadence). */
        if (out && ((step + 1) % 500 == 0 || step == 0)) {
            fprintf(out, "TRAIN %s: step %d/%d | loss %.4f\n",
                    spec->organelle_name, step + 1, num_steps,
                    (double)mean_loss);
        }
    }

    clock_t t_end = clock();
    double train_sec = (double)(t_end - t_start) / (double)CLOCKS_PER_SEC;

    /* 10. Optional checkpoint save. */
    int save_step = num_steps; /* one-past-last, matching names_demo */
    if (spec->save_path) {
        int rc = checkpoint_save(model, m_buf, v_buf, save_step,
                                 spec->save_path);
        if (rc != 0) {
            if (out) fprintf(out,
                "TRAIN: checkpoint_save('%s') returned %d\n",
                spec->save_path, rc);
            /* not fatal — still leave model on organelle */
        } else if (out) {
            fprintf(out, "TRAIN: saved checkpoint to '%s' (vocab=%zu step=%d)\n",
                    spec->save_path, vocab.vocab_size, save_step);
        }
    }

    /* 11. Hand model off to the organelle slot.  Note: vocab.chars and
     * docs survive for the model's lifetime indirectly only via the
     * checkpoint; we free them here because the engine's Model carries
     * its own internal copy of weights.  This matches names_demo's
     * teardown order. */
    org->model = model;
    org->loaded = 1;
    if (spec->save_path) {
        strncpy(org->checkpoint_path, spec->save_path,
                sizeof(org->checkpoint_path) - 1);
    }

    /* Record summary metrics on the runtime. */
    rt->last_train_steps         = num_steps;
    rt->last_train_final_loss    = final_mean_loss;
    rt->last_train_total_seconds = train_sec;

    if (out) {
        fprintf(out,
            "TRAIN %s: complete (%d steps, final loss %.4f, %.2fs, vocab=%zu, "
            "params=%zu)\n",
            spec->organelle_name, num_steps, final_mean_loss, train_sec,
            vocab.vocab_size, nparams);
    }

    /* Cleanup non-model resources. */
    for (int L = 0; L < nl; L++) {
        kv_cache_free(keys[L]);
        kv_cache_free(values[L]);
    }
    free(keys); free(values); free(cache_len);
    free(token_buf);
    free(grad_buffer); free(m_buf); free(v_buf);
    free(vocab.chars);
    free_docs(&docs);
    return OQL_OK;
}
