/*
 * MicroGPT-C — MSA Sliding-Window Recency A/B Benchmark
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 *
 * Trains an identical seeded char-level model on the names corpus,
 * concatenates held-out names into a single long sequence (much longer
 * than BLOCK_SIZE), then scores next-token cross-entropy under MSA.
 *
 * Two modes (selected by -DBENCH_MSA_SLIDING_WINDOW=1):
 *   OFF: existing MSA flow — chunk block_size/2 at overflow, re-inject
 *        single best chunk, lose half the active cache to memmove.
 *   ON:  MSA + uncompressed recency ring of MSA_WIN tokens. On overflow,
 *        all active tokens get pooled, then best-pool-chunk is re-injected
 *        at position 0 AND the recency window is laid down at positions
 *        1..MSA_WIN. The model thus retains MSA_WIN tokens at full
 *        fidelity across every chunking event.
 *
 * Output (machine-parseable):
 *   MSA_SLIDING_WINDOW: ON|OFF
 *   MSA_WIN: <int>
 *   FINAL_TRAIN_LOSS: <float>
 *   PRE_CHUNK_LOSS: <float>      // CE on positions 1..block_size-1
 *   POST_CHUNK_LOSS: <float>     // CE on positions past block_size
 *   POST_CHUNK_PERPLEXITY: <float>
 *   CHUNK_EVENTS: <int>          // number of MSA chunking events
 *   POST_CHUNK_TOKENS: <int>
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_msa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BENCH_NUM_STEPS
#define BENCH_NUM_STEPS 1500
#endif
#ifndef BENCH_BATCH_SIZE
#define BENCH_BATCH_SIZE 16
#endif
#ifndef BENCH_HOLDOUT_FRACTION
#define BENCH_HOLDOUT_FRACTION 0.10
#endif
#ifndef BENCH_LONG_SEQ_LEN
#define BENCH_LONG_SEQ_LEN 512  /* must be > BLOCK_SIZE */
#endif
#ifndef BENCH_MSA_SLIDING_WINDOW
#define BENCH_MSA_SLIDING_WINDOW 0
#endif
#ifndef MSA_WIN
#define MSA_WIN 16
#endif
/* Routing modes:
 *   0 = top-1 cosine (existing baseline)
 *   1 = top-k Lightning Indexer (multi-layer ReLU-summed scoring)
 */
#ifndef BENCH_MSA_ROUTING_MODE
#define BENCH_MSA_ROUTING_MODE 0
#endif
#ifndef BENCH_MSA_TOPK
#define BENCH_MSA_TOPK 1
#endif

/* ---------- Helper: build a long held-out token sequence ----------- */
static size_t build_long_sequence(const Docs *docs, size_t holdout_start,
                                  const Vocab *vocab, size_t *seq_out,
                                  size_t max_len) {
  size_t n = 0;
  size_t doc_idx = holdout_start;
  while (n < max_len) {
    if (doc_idx >= docs->num_docs) doc_idx = holdout_start; /* wrap */
    /* Tokenise this doc as plain chars (no BOS/EOS — we want a
     * continuous stream). */
    const char *doc = docs->lines[doc_idx];
    size_t doc_len = docs->doc_lens[doc_idx];
    for (size_t i = 0; i < doc_len && n < max_len; i++) {
      unsigned char ch = (unsigned char)doc[i];
      /* Linear-scan vocab lookup — vocab is small (<=27 chars). */
      size_t tok = vocab->bos_id; /* fallback */
      for (size_t v = 0; v < vocab->vocab_size - 1; v++) {
        if (vocab->chars[v] == ch) { tok = v; break; }
      }
      seq_out[n++] = tok;
    }
    /* Insert a BOS as separator between names. */
    if (n < max_len) seq_out[n++] = vocab->bos_id;
    doc_idx++;
  }
  return n;
}

/* ---------- MSA flow A: existing baseline (or top-k variant) ----------
 *
 * Memmove second half down, then inject either:
 *   ROUTING_MODE=0 : top-1 best chunk at pos 0 via cosine (existing)
 *   ROUTING_MODE=1 : top-K best chunks at pos 0..K-1 via Lightning Indexer
 */
static int msa_step_baseline(MsaPool *pool, scalar_t **inf_keys,
                             scalar_t **inf_values, size_t *inf_cache_len,
                             size_t *pos, const MicrogptConfig *cfg) {
  size_t chunk_size = (size_t)cfg->block_size / 2;
  msa_pool_chunk(pool, inf_keys, inf_values, chunk_size);
  for (int L = 0; L < cfg->n_layer; L++) {
    memmove(inf_keys[L],
            inf_keys[L] + (chunk_size * (size_t)cfg->n_embd),
            ((size_t)cfg->block_size - chunk_size) * (size_t)cfg->n_embd
                * sizeof(scalar_t));
    memmove(inf_values[L],
            inf_values[L] + (chunk_size * (size_t)cfg->n_embd),
            ((size_t)cfg->block_size - chunk_size) * (size_t)cfg->n_embd
                * sizeof(scalar_t));
    inf_cache_len[L] -= chunk_size;
  }
  *pos -= chunk_size;
  /* Build query from the most-recent K (across all layers). */
  scalar_t **q = (scalar_t **)malloc((size_t)cfg->n_layer * sizeof(scalar_t *));
  for (int L = 0; L < cfg->n_layer; L++) {
    q[L] = (scalar_t *)malloc((size_t)cfg->n_embd * sizeof(scalar_t));
    memcpy(q[L],
           inf_keys[L] + ((*pos > 0 ? *pos - 1 : 0) * (size_t)cfg->n_embd),
           (size_t)cfg->n_embd * sizeof(scalar_t));
  }
#if BENCH_MSA_ROUTING_MODE == 1
  int top[16];
  int n = msa_route_top_k(pool, q, BENCH_MSA_TOPK, top, NULL);
  for (int i = 0; i < n; i++) {
    if (top[i] >= 0)
      msa_expand_context(pool, top[i], inf_keys, inf_values, (size_t)i);
  }
#else
  int best = msa_route_top_1(pool, q);
  if (best >= 0) msa_expand_context(pool, best, inf_keys, inf_values, 0);
#endif
  for (int L = 0; L < cfg->n_layer; L++) free(q[L]);
  free(q);
  return 1;
}

/* ---------- MSA flow B: with sliding-window recency ---------- */
static int msa_step_sliding(MsaPool *pool, MsaRecency *rec,
                            scalar_t **inf_keys, scalar_t **inf_values,
                            size_t *inf_cache_len, size_t *pos,
                            const MicrogptConfig *cfg) {
  /* Pool ALL active tokens into MsaPool (not just half). */
  size_t chunk_size = (size_t)cfg->block_size;
  msa_pool_chunk(pool, inf_keys, inf_values, chunk_size);
  /* Wipe the active cache. */
  for (int L = 0; L < cfg->n_layer; L++) {
    memset(inf_keys[L], 0,
           (size_t)cfg->block_size * (size_t)cfg->n_embd * sizeof(scalar_t));
    memset(inf_values[L], 0,
           (size_t)cfg->block_size * (size_t)cfg->n_embd * sizeof(scalar_t));
    inf_cache_len[L] = 0;
  }
  /* 1) Inject best historical chunk at position 0. */
  scalar_t **q = (scalar_t **)malloc((size_t)cfg->n_layer * sizeof(scalar_t *));
  for (int L = 0; L < cfg->n_layer; L++) {
    q[L] = (scalar_t *)malloc((size_t)cfg->n_embd * sizeof(scalar_t));
    /* Use the most-recent recency-buffer K as query (best local signal). */
    if (rec && rec->length > 0) {
      const size_t stride_layer = (size_t)rec->n_embd;
      const size_t stride_token = (size_t)rec->n_layer * stride_layer;
      size_t newest_ring =
          (rec->length < rec->capacity) ? rec->length - 1
                                        : (rec->head + rec->capacity - 1) % rec->capacity;
      memcpy(q[L],
             rec->keys + newest_ring * stride_token + (size_t)L * stride_layer,
             stride_layer * sizeof(scalar_t));
    } else {
      memset(q[L], 0, (size_t)cfg->n_embd * sizeof(scalar_t));
    }
  }
  int best = msa_route_top_1(pool, q);
  size_t inject_pos = 0;
  if (best >= 0) {
    msa_expand_context(pool, best, inf_keys, inf_values, 0);
    inject_pos = 1;
  }
  for (int L = 0; L < cfg->n_layer; L++) free(q[L]);
  free(q);
  /* 2) Lay down the entire recency window starting at inject_pos. */
  size_t injected = msa_recency_inject(rec, inf_keys, inf_values, inject_pos);
  size_t total = inject_pos + injected;
  for (int L = 0; L < cfg->n_layer; L++) inf_cache_len[L] = total;
  *pos = total;
  return 1;
}

int main(void) {
  srand(42);
  seed_rng(42);

  MicrogptConfig cfg = microgpt_default_config();
  cfg.num_steps = BENCH_NUM_STEPS;
  cfg.batch_size = BENCH_BATCH_SIZE;
  microgpt_print_config("MicroGPT-C — MSA Sliding-Window A/B Benchmark", &cfg);

  const int nl = cfg.n_layer;
  const int bs = cfg.block_size;

  if (BENCH_LONG_SEQ_LEN <= bs) {
    fprintf(stderr, "ERROR: BENCH_LONG_SEQ_LEN (%d) must exceed BLOCK_SIZE (%d)\n",
            BENCH_LONG_SEQ_LEN, bs);
    return 1;
  }

  Docs docs = {0};
  if (load_docs("c_names.txt", &docs, cfg.max_docs) != 0) {
    fprintf(stderr, "ERROR: cannot open c_names.txt — run from build/ directory\n");
    return 1;
  }
  shuffle_docs(&docs);
  Vocab vocab = {0};
  build_vocab(&docs, &vocab);
  printf("corpus: %zu docs   vocab: %zu chars\n", docs.num_docs, vocab.vocab_size);

  size_t holdout_start = (size_t)((double)docs.num_docs * (1.0 - BENCH_HOLDOUT_FRACTION));
  if (holdout_start < 1) holdout_start = 1;
  if (holdout_start >= docs.num_docs) holdout_start = docs.num_docs - 1;
  size_t train_count = holdout_start;
  printf("train: %zu docs   holdout starts at: %zu\n", train_count, holdout_start);

  Model *model = model_create(vocab.vocab_size, &cfg);
  size_t nparams = model_num_params(model);
  printf("params: %zu\n", nparams);

  scalar_t *grad_buffer = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *m = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *v = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  scalar_t **keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    keys[L] = kv_cache_alloc(&cfg);
    values[L] = kv_cache_alloc(&cfg);
  }
  size_t *token_buf = (size_t *)malloc(((size_t)bs + 2) * sizeof(size_t));

  /* ---- Train on regular short docs (same as attention-sink benchmark) ---- */
  size_t doc_idx = 0;
  scalar_t final_train_loss = 0;
  for (int step = 0; step < cfg.num_steps; step++) {
    memset(grad_buffer, 0, nparams * sizeof(scalar_t));
    scalar_t batch_loss = 0;
    size_t batch_positions = 0;
    for (int b = 0; b < cfg.batch_size; b++) {
      for (int L = 0; L < nl; L++) cache_len[L] = 0;
      size_t idx = (doc_idx++) % train_count;
      size_t n_tok = tokenize(docs.lines[idx], docs.doc_lens[idx], &vocab,
                              token_buf, (size_t)bs + 2);
      if (n_tok < 2) continue;
      size_t n = n_tok - 1;
      if (n > (size_t)bs) n = (size_t)bs;
      batch_positions += n;
      for (size_t pos = 0; pos < n; pos++) {
        scalar_t loss = forward_backward_one(model, token_buf[pos], pos,
                                             token_buf[pos + 1], keys, values,
                                             cache_len, grad_buffer);
        batch_loss += loss;
      }
    }
    if (batch_positions == 0) continue;
    scalar_t mean_loss = batch_loss / (scalar_t)batch_positions;
    for (size_t i = 0; i < nparams; i++)
      grad_buffer[i] /= (scalar_t)batch_positions;
    adam_step(model, grad_buffer, m, v, step);
    final_train_loss = mean_loss;
    if ((step + 1) % 200 == 0 || step == 0)
      printf("  step %4d / %d   loss %.4f\n", step + 1, cfg.num_steps,
             (double)mean_loss);
  }

  /* ---- Build a long held-out token sequence ---- */
  size_t *long_seq = (size_t *)malloc((size_t)BENCH_LONG_SEQ_LEN * sizeof(size_t));
  size_t long_n = build_long_sequence(&docs, holdout_start, &vocab, long_seq,
                                      (size_t)BENCH_LONG_SEQ_LEN);
  printf("long held-out sequence: %zu tokens (BLOCK_SIZE=%d)\n", long_n, bs);

  /* ---- Long-context inference scoring ---- */
  for (int L = 0; L < nl; L++) cache_len[L] = 0;
  scalar_t *logits = (scalar_t *)malloc(vocab.vocab_size * sizeof(scalar_t));

  MsaPool *pool = msa_pool_create(1024, nl, cfg.n_embd);
  MsaRecency *rec = NULL;
#if BENCH_MSA_SLIDING_WINDOW
  rec = msa_recency_create((size_t)MSA_WIN, nl, cfg.n_embd);
#endif

  scalar_t pre_loss = 0;
  size_t pre_count = 0;
  scalar_t post_loss = 0;
  size_t post_count = 0;
  int chunk_events = 0;

  size_t pos = 0;
  for (size_t i = 0; i < long_n - 1; i++) {
    if (pos >= (size_t)bs) {
#if BENCH_MSA_SLIDING_WINDOW
      msa_step_sliding(pool, rec, keys, values, cache_len, &pos, &cfg);
#else
      msa_step_baseline(pool, keys, values, cache_len, &pos, &cfg);
#endif
      chunk_events++;
    }
    forward_inference(model, long_seq[i], pos, keys, values, cache_len, logits);
    /* Softmax */
    scalar_t maxl = logits[0];
    for (size_t j = 1; j < vocab.vocab_size; j++)
      if (logits[j] > maxl) maxl = logits[j];
    scalar_t Z = 0;
    for (size_t j = 0; j < vocab.vocab_size; j++)
      Z += (scalar_t)exp((double)(logits[j] - maxl));
    size_t tgt = long_seq[i + 1];
    scalar_t p = (scalar_t)exp((double)(logits[tgt] - maxl)) / Z;
    if (p < (scalar_t)1e-10) p = (scalar_t)1e-10;
    scalar_t loss_here = -(scalar_t)log((double)p);
    if (i < (size_t)bs) {
      pre_loss += loss_here;
      pre_count++;
    } else {
      post_loss += loss_here;
      post_count++;
    }

#if BENCH_MSA_SLIDING_WINDOW
    /* Push K and V of just-computed position into recency ring.
     * The cache slot used was at index pos (or cache_len-1). */
    if (rec) {
      scalar_t **tk = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
      scalar_t **tv = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
      size_t slot = pos; /* the position we just wrote into */
      for (int L = 0; L < nl; L++) {
        tk[L] = keys[L]   + slot * (size_t)cfg.n_embd;
        tv[L] = values[L] + slot * (size_t)cfg.n_embd;
      }
      msa_recency_push(rec, tk, tv);
      free(tk); free(tv);
    }
#endif
    pos++;
  }

  free(logits);

  scalar_t pre_avg  = pre_count  > 0 ? pre_loss  / (scalar_t)pre_count  : 0;
  scalar_t post_avg = post_count > 0 ? post_loss / (scalar_t)post_count : 0;
  scalar_t post_ppl = (scalar_t)exp((double)post_avg);

  printf("\n=== RESULTS ===\n");
#if BENCH_MSA_SLIDING_WINDOW
  printf("MSA_SLIDING_WINDOW: ON\n");
  printf("MSA_WIN: %d\n", MSA_WIN);
#else
  printf("MSA_SLIDING_WINDOW: OFF\n");
  printf("MSA_WIN: -\n");
#endif
  printf("FINAL_TRAIN_LOSS: %.6f\n", (double)final_train_loss);
  printf("PRE_CHUNK_LOSS: %.6f\n", (double)pre_avg);
  printf("POST_CHUNK_LOSS: %.6f\n", (double)post_avg);
  printf("POST_CHUNK_PERPLEXITY: %.6f\n", (double)post_ppl);
  printf("CHUNK_EVENTS: %d\n", chunk_events);
  printf("PRE_CHUNK_TOKENS: %zu\n", pre_count);
  printf("POST_CHUNK_TOKENS: %zu\n", post_count);
  printf("PARAMS: %zu\n", nparams);

  free(long_seq);
  msa_pool_free(pool);
  if (rec) msa_recency_free(rec);
  free(token_buf);
  for (int L = 0; L < nl; L++) {
    kv_cache_free(keys[L]);
    kv_cache_free(values[L]);
  }
  free(keys); free(values); free(cache_len);
  free(grad_buffer); free(m); free(v);
  model_free(model);
  free_docs(&docs);
  free(vocab.chars);
  return 0;
}
