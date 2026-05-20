/*
 * tools/e15_train.c — Experiment E15 training driver.
 *
 * Trains a single transformer on an E15 oracle corpus (TSV of
 * "<state>\t<solution>\n" lines) and writes a checkpoint.
 *
 * Used for both Phase 4 (monolithic baseline) and Phase 5 (OPA
 * composition organelles).  The architecture is selected at *compile
 * time* via the engine macros (N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE,
 * MLP_DIM) that the linked microgpt_lib variant was compiled with —
 * see CMakeLists.txt's e15_mono_train + e15_opa_train add_executable
 * blocks.
 *
 * Compile-time-macro caveat (E09 §3.4): a binary linked against the
 * default microgpt_lib will use the default 16-dim engine.  The CMake
 * targets `e15_mono_train` and `e15_opa_train` link against variants
 * with the 900K and 300K parameter configs respectively, via
 * `_microgpt_lib_for_defines`.  Two binaries — two variants — by
 * construction.  T8 holds: engine surface unchanged.
 *
 * Training format:
 *   Each TSV line "<state>\t<solution>" is transformed into a single
 *   training document "<role_tag><state>|<solution>" where role_tag
 *   is one of: ""       (monolithic / planner role default)
 *               "P:"    (planner role  — when --role=planner)
 *               "M:"    (player role   — when --role=player)
 *               "J:"    (judge role    — when --role=judge)
 *
 * For OPA roles, the same corpus is used but each organelle sees its
 * own role-tagged copy of the docs.  This gives the organelles a
 * shared input distribution but a different learned mapping per role.
 *
 * Usage:
 *   e15_*_train --corpus build/klotski_optimal.tsv
 *               --save   checkpoints/klotski_mono.ckpt
 *               --steps  50000
 *               --batch  8
 *               --lr     0.001
 *              [--role   monolithic|planner|player|judge]
 *              [--seed   1337]
 *              [--vocab-save vocab.txt]   # persist char vocab for eval
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "microgpt.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Slurp a file into a heap buffer, nul-terminated. */
static char *slurp_file(const char *path, size_t *out_len) {
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
    if (out_len) *out_len = got;
    return buf;
}

/* Transform a TSV line "<state>\t<solution>" into a training doc
 *   "<role_tag><state>|<solution>".
 * Writes the result to *out (must be at least n_in + 8 bytes).
 * Returns the length of the produced doc. */
static size_t transform_line(const char *line, size_t llen,
                             const char *role_tag,
                             char *out, size_t out_cap) {
    size_t rlen = role_tag ? strlen(role_tag) : 0;
    size_t i = 0;
    /* role tag */
    if (rlen + llen + 1 >= out_cap) return 0;
    if (rlen) {
        memcpy(out + i, role_tag, rlen);
        i += rlen;
    }
    /* copy state up to '\t' */
    size_t j = 0;
    while (j < llen && line[j] != '\t' && line[j] != '\n') {
        if (i >= out_cap - 1) return 0;
        out[i++] = line[j++];
    }
    /* separator */
    if (i >= out_cap - 1) return 0;
    out[i++] = '|';
    /* skip '\t' */
    if (j < llen && line[j] == '\t') j++;
    /* copy solution up to '\n' */
    while (j < llen && line[j] != '\n') {
        if (i >= out_cap - 1) return 0;
        out[i++] = line[j++];
    }
    out[i] = '\0';
    return i;
}

/* Write the role-tagged docs to a temp file (one doc per line) so that
 * load_docs() can ingest them with no engine-surface changes. */
static int prepare_role_corpus(const char *src_tsv,
                                const char *role_tag,
                                const char *dst_path) {
    size_t blen = 0;
    char *buf = slurp_file(src_tsv, &blen);
    if (!buf) {
        fprintf(stderr, "e15_train: cannot read corpus '%s': %s\n",
                src_tsv, strerror(errno));
        return -1;
    }
    FILE *of = fopen(dst_path, "wb");
    if (!of) {
        fprintf(stderr, "e15_train: cannot write transformed '%s': %s\n",
                dst_path, strerror(errno));
        free(buf);
        return -1;
    }
    char doc[1024];
    char *p = buf, *end = buf + blen;
    size_t n_lines = 0;
    while (p < end) {
        char *nl = memchr(p, '\n', (size_t)(end - p));
        size_t llen = nl ? (size_t)(nl - p) : (size_t)(end - p);
        if (llen > 0) {
            size_t dlen = transform_line(p, llen, role_tag, doc, sizeof(doc));
            if (dlen > 0) {
                fwrite(doc, 1, dlen, of);
                fputc('\n', of);
                n_lines++;
            }
        }
        p = nl ? nl + 1 : end;
    }
    fclose(of);
    free(buf);
    fprintf(stdout, "[e15_train] prepared %zu role-tagged docs -> %s\n",
            n_lines, dst_path);
    return 0;
}

/* Save the vocab to a file so the eval driver can rebuild it identically. */
static int save_vocab(const Vocab *vocab, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    /* Format: first line = vocab_size, then each line = one char (or
     * the literal 'BOS' marker for the BOS slot).  Newline chars in
     * the vocab are emitted as '\\n'. */
    fprintf(f, "%zu\n", vocab->vocab_size);
    for (size_t i = 0; i < vocab->vocab_size; i++) {
        char c = vocab->chars[i];
        if (c == '\n') fputs("\\n", f);
        else if (c == '\\') fputs("\\\\", f);
        else fputc(c, f);
        fputc('\n', f);
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    const char *corpus_path = NULL;
    const char *save_path = NULL;
    const char *vocab_save = NULL;
    const char *role = "monolithic";
    int steps = 50000;
    int batch = 8;
    double lr = 1e-3;
    unsigned int seed = 1337;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--corpus") && i + 1 < argc) {
            corpus_path = argv[++i];
        } else if (!strcmp(argv[i], "--save") && i + 1 < argc) {
            save_path = argv[++i];
        } else if (!strcmp(argv[i], "--vocab-save") && i + 1 < argc) {
            vocab_save = argv[++i];
        } else if (!strcmp(argv[i], "--role") && i + 1 < argc) {
            role = argv[++i];
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--lr") && i + 1 < argc) {
            lr = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v")) {
            verbose = 1;
        } else {
            fprintf(stderr, "e15_train: unknown arg '%s'\n", argv[i]);
            return 2;
        }
    }
    if (!corpus_path || !save_path) {
        fprintf(stderr,
            "usage: e15_train --corpus <tsv> --save <ckpt> "
            "[--role monolithic|planner|player|judge] "
            "[--steps N] [--batch B] [--lr LR] [--seed S] [--vocab-save VOCAB]\n");
        return 2;
    }

    const char *role_tag = "";
    if (!strcmp(role, "planner")) role_tag = "P:";
    else if (!strcmp(role, "player")) role_tag = "M:";
    else if (!strcmp(role, "judge")) role_tag = "J:";
    else if (!strcmp(role, "monolithic")) role_tag = "";
    else {
        fprintf(stderr, "e15_train: unknown role '%s'\n", role);
        return 2;
    }

    /* Stage corpus into a transformed temp file (one doc per line). */
    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.docs.tmp", save_path);
    if (prepare_role_corpus(corpus_path, role_tag, tmp_path) != 0) {
        return 1;
    }

    /* Build runtime config from compile-time macros. */
    MicrogptConfig cfg = microgpt_default_config();
    cfg.n_embd     = N_EMBD;
    cfg.n_head     = N_HEAD;
    cfg.n_layer    = N_LAYER;
    cfg.block_size = BLOCK_SIZE;
    cfg.mlp_dim    = MLP_DIM;
    cfg.batch_size = batch;
    cfg.num_steps  = steps;
    cfg.learning_rate = lr;
    /* Bump corpus capacity if needed. */
    if (cfg.max_docs < 2200) cfg.max_docs = 2200;
    if (cfg.max_doc_len < 256) cfg.max_doc_len = 256;
    /* Vocab will be small (< 32); the default is plenty. */

    fprintf(stdout,
        "[e15_train] role=%s tag='%s' corpus=%s save=%s "
        "steps=%d batch=%d lr=%.4g seed=%u\n",
        role, role_tag, corpus_path, save_path, steps, batch, lr, seed);
    fprintf(stdout,
        "[e15_train] compile-time arch: N_EMBD=%d N_HEAD=%d N_LAYER=%d "
        "BLOCK_SIZE=%d MLP_DIM=%d\n",
        N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, MLP_DIM);

    seed_rng(seed);
    srand(seed);

    Docs docs = {0};
    if (load_docs(tmp_path, &docs, cfg.max_docs) != 0) {
        fprintf(stderr, "e15_train: load_docs('%s') failed\n", tmp_path);
        return 1;
    }
    shuffle_docs(&docs);

    Vocab vocab = {0};
    build_vocab(&docs, &vocab);
    fprintf(stdout, "[e15_train] docs=%zu vocab=%zu\n",
            docs.num_docs, vocab.vocab_size);

    if (vocab_save) {
        if (save_vocab(&vocab, vocab_save) == 0)
            fprintf(stdout, "[e15_train] vocab -> %s\n", vocab_save);
    }

    Model *model = model_create(vocab.vocab_size, &cfg);
    if (!model) {
        fprintf(stderr, "e15_train: model_create failed\n");
        free(vocab.chars);
        free_docs(&docs);
        return 1;
    }
    size_t nparams = model_num_params(model);
    fprintf(stdout, "[e15_train] params=%zu\n", nparams);

    scalar_t *grad_buffer = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    scalar_t *m_buf       = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    scalar_t *v_buf       = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    if (!grad_buffer || !m_buf || !v_buf) {
        fprintf(stderr, "e15_train: OOM allocating optimizer state\n");
        free(grad_buffer); free(m_buf); free(v_buf);
        model_free(model);
        free(vocab.chars);
        free_docs(&docs);
        return 1;
    }

    const int nl = cfg.n_layer;
    const int bs = cfg.block_size;
    scalar_t **keys = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    scalar_t **values = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    size_t *cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    for (int L = 0; L < nl; L++) {
        keys[L]   = kv_cache_alloc(&cfg);
        values[L] = kv_cache_alloc(&cfg);
    }

    size_t *token_buf = (size_t *)malloc(((size_t)bs + 2) * sizeof(size_t));
    size_t doc_idx = 0;
    clock_t t_start = clock();
    double final_mean_loss = 0.0;
    double best_loss = 1e9;

    for (int step = 0; step < steps; step++) {
        memset(grad_buffer, 0, nparams * sizeof(scalar_t));
        scalar_t batch_loss = 0;
        size_t batch_positions = 0;
        for (int b = 0; b < cfg.batch_size; b++) {
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
        final_mean_loss = (double)mean_loss;
        if (final_mean_loss < best_loss) best_loss = final_mean_loss;

        if ((step + 1) % 500 == 0 || step == 0 || step == steps - 1) {
            double el = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            fprintf(stdout,
                "[e15_train] step %d/%d | loss %.4f | best %.4f | %.1fs\n",
                step + 1, steps, final_mean_loss, best_loss, el);
            fflush(stdout);
        }
    }
    clock_t t_end = clock();
    double train_sec = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    fprintf(stdout,
        "[e15_train] DONE: %d steps, final_loss=%.4f, best_loss=%.4f, "
        "%.1fs total, params=%zu, vocab=%zu, role=%s\n",
        steps, final_mean_loss, best_loss, train_sec, nparams,
        vocab.vocab_size, role);

    int rc = checkpoint_save(model, m_buf, v_buf, steps, save_path);
    if (rc != 0) {
        fprintf(stderr, "e15_train: checkpoint_save('%s') failed\n", save_path);
    } else {
        fprintf(stdout, "[e15_train] checkpoint -> %s\n", save_path);
    }

    /* Cleanup */
    for (int L = 0; L < nl; L++) {
        kv_cache_free(keys[L]);
        kv_cache_free(values[L]);
    }
    free(keys); free(values); free(cache_len);
    free(token_buf);
    free(grad_buffer); free(m_buf); free(v_buf);
    model_free(model);
    free(vocab.chars);
    free_docs(&docs);
    (void)verbose;
    return rc == 0 ? 0 : 1;
}
