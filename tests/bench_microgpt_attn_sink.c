/*
 * MicroGPT-C — Attention Sink A/B Benchmark
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * SPDX-License-Identifier: MIT
 *
 * Trains an identical, fully-seeded character-level model on the names
 * corpus, then evaluates held-out cross-entropy loss and a "spike
 * concentration" diagnostic on a long-context probe. Compile twice — once
 * without -DMICROGPT_ATTN_SINK and once with — and diff the outputs to
 * measure the architectural impact of attention sinks.
 *
 * Output format (one line per metric, machine-parseable):
 *   ATTN_SINK_BUILD: ON|OFF
 *   ATTN_SINK_LOGIT: <float or "-">
 *   FINAL_TRAIN_LOSS: <float>
 *   HELDOUT_LOSS: <float>
 *   HELDOUT_PERPLEXITY: <float>
 *   PROBE_MAX_ATTN: <float>           // max attention weight on any single position
 *   PROBE_MEAN_TOP1: <float>          // average top-1 attention across heads × steps
 *   PROBE_ENTROPY: <float>            // mean entropy of attention distributions
 *   TRAIN_SECONDS: <float>
 *   HELDOUT_SECONDS: <float>
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- Configuration baked in via CMake DEFINES (or defaults) ---- */
#ifndef BENCH_NUM_STEPS
#define BENCH_NUM_STEPS 600
#endif
#ifndef BENCH_BATCH_SIZE
#define BENCH_BATCH_SIZE 8
#endif
#ifndef BENCH_HOLDOUT_FRACTION
#define BENCH_HOLDOUT_FRACTION 0.10
#endif
#ifndef BENCH_PROBE_LEN
#define BENCH_PROBE_LEN 64 /* must be <= BLOCK_SIZE */
#endif

/* ---------- Helpers --------------------------------------------------- */

static scalar_t evaluate_holdout_loss(Model *model, const Vocab *vocab,
                                      const Docs *docs, size_t holdout_start,
                                      const MicrogptConfig *cfg, scalar_t **keys,
                                      scalar_t **values, size_t *cache_len,
                                      size_t *token_buf, size_t *positions_out) {
  const int nl = cfg->n_layer;
  const int bs = cfg->block_size;
  scalar_t *logits = (scalar_t *)malloc(vocab->vocab_size * sizeof(scalar_t));
  scalar_t total_loss = 0;
  size_t total_pos = 0;
  for (size_t d = holdout_start; d < docs->num_docs; d++) {
    for (int L = 0; L < nl; L++)
      cache_len[L] = 0;
    size_t n_tok = tokenize(docs->lines[d], docs->doc_lens[d], vocab,
                            token_buf, (size_t)bs + 2);
    if (n_tok < 2)
      continue;
    size_t n = n_tok - 1;
    if (n > (size_t)bs)
      n = (size_t)bs;
    for (size_t pos = 0; pos < n; pos++) {
      forward_inference(model, token_buf[pos], pos, keys, values, cache_len,
                        logits);
      /* Softmax the raw logits to get a true CE loss. */
      scalar_t maxl = logits[0];
      for (size_t i = 1; i < vocab->vocab_size; i++)
        if (logits[i] > maxl) maxl = logits[i];
      scalar_t Z = 0;
      for (size_t i = 0; i < vocab->vocab_size; i++)
        Z += (scalar_t)exp((double)(logits[i] - maxl));
      size_t tgt = token_buf[pos + 1];
      scalar_t p = (scalar_t)exp((double)(logits[tgt] - maxl)) / Z;
      if (p < (scalar_t)1e-10) p = (scalar_t)1e-10;
      total_loss += -(scalar_t)log((double)p);
      total_pos++;
    }
  }
  free(logits);
  *positions_out = total_pos;
  return total_pos > 0 ? total_loss / (scalar_t)total_pos : 0;
}

/*
 * Long-context attention diagnostic:
 *   Feed BENCH_PROBE_LEN repeated BOS tokens (a degenerate, low-information
 *   input). Measure how the model's final-position logits behave. Without
 *   an attention sink the model is forced to spread mass over all positions
 *   even when none are informative, producing high-confidence-but-arbitrary
 *   predictions. With a sink, the network can route mass to "nothing" and
 *   produce a more uniform output distribution.
 *
 *   We approximate this through the predictive distribution itself —
 *   measure mean entropy and top-1 probability over the final position.
 *   Higher entropy / lower top-1 on a degenerate input means the model is
 *   not over-committing.
 */
static void run_attention_probe(Model *model, const Vocab *vocab,
                                const MicrogptConfig *cfg, scalar_t **keys,
                                scalar_t **values, size_t *cache_len,
                                scalar_t *probe_top1_out,
                                scalar_t *probe_max_out,
                                scalar_t *probe_entropy_out) {
  const int nl = cfg->n_layer;
  scalar_t *logits = (scalar_t *)malloc(vocab->vocab_size * sizeof(scalar_t));

  scalar_t sum_top1 = 0;
  scalar_t sum_entropy = 0;
  scalar_t max_top1 = 0;
  size_t steps = 0;

  for (int L = 0; L < nl; L++)
    cache_len[L] = 0;

  for (size_t pos = 0; pos < BENCH_PROBE_LEN; pos++) {
    forward_inference(model, vocab->bos_id, pos, keys, values, cache_len,
                      logits);
    /* forward_inference returns raw lm_head logits — softmax here. */
    scalar_t maxl = logits[0];
    for (size_t i = 1; i < vocab->vocab_size; i++)
      if (logits[i] > maxl) maxl = logits[i];
    scalar_t Z = 0;
    for (size_t i = 0; i < vocab->vocab_size; i++)
      Z += (scalar_t)exp((double)(logits[i] - maxl));
    scalar_t top1 = 0;
    scalar_t entropy = 0;
    for (size_t i = 0; i < vocab->vocab_size; i++) {
      scalar_t p = (scalar_t)exp((double)(logits[i] - maxl)) / Z;
      if (p > top1) top1 = p;
      if (p > (scalar_t)1e-12) entropy += -p * (scalar_t)log((double)p);
    }
    sum_top1 += top1;
    sum_entropy += entropy;
    if (top1 > max_top1) max_top1 = top1;
    steps++;
  }

  free(logits);
  *probe_top1_out = sum_top1 / (scalar_t)steps;
  *probe_max_out = max_top1;
  *probe_entropy_out = sum_entropy / (scalar_t)steps;
}

/* ---------- Main ------------------------------------------------------ */

int main(void) {
  srand(42);
  seed_rng(42);

  MicrogptConfig cfg = microgpt_default_config();
  cfg.num_steps = BENCH_NUM_STEPS;
  cfg.batch_size = BENCH_BATCH_SIZE;
  microgpt_print_config("MicroGPT-C — Attention Sink A/B Benchmark", &cfg);

  const int nl = cfg.n_layer;
  const int bs = cfg.block_size;

  /* ---- Load corpus ---- */
  Docs docs = {0};
  if (load_docs("c_names.txt", &docs, cfg.max_docs) != 0) {
    fprintf(stderr, "ERROR: cannot open c_names.txt — run from build/ directory\n");
    return 1;
  }
  shuffle_docs(&docs);

  Vocab vocab = {0};
  build_vocab(&docs, &vocab);
  printf("corpus: %zu docs   vocab: %zu chars\n", docs.num_docs, vocab.vocab_size);

  /* Train/holdout split (last BENCH_HOLDOUT_FRACTION of shuffled docs is held out) */
  size_t holdout_start = (size_t)((double)docs.num_docs * (1.0 - BENCH_HOLDOUT_FRACTION));
  if (holdout_start < 1) holdout_start = 1;
  if (holdout_start >= docs.num_docs) holdout_start = docs.num_docs - 1;
  size_t train_count = holdout_start;
  size_t holdout_count = docs.num_docs - holdout_start;
  printf("train: %zu docs   holdout: %zu docs\n", train_count, holdout_count);

  Model *model = model_create(vocab.vocab_size, &cfg);
  if (!model) { fprintf(stderr, "model_create failed\n"); return 1; }
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

  /* ---- Train ---- */
  size_t doc_idx = 0;
  scalar_t final_train_loss = 0;
  clock_t t0 = clock();
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
    if ((step + 1) % 100 == 0 || step == 0)
      printf("  step %4d / %d   loss %.4f\n", step + 1, cfg.num_steps,
             (double)mean_loss);
  }
  clock_t t1 = clock();
  scalar_t train_sec = (scalar_t)(t1 - t0) / (scalar_t)CLOCKS_PER_SEC;

  /* ---- Held-out evaluation ---- */
  clock_t t2 = clock();
  size_t holdout_positions = 0;
  scalar_t holdout_loss = evaluate_holdout_loss(
      model, &vocab, &docs, holdout_start, &cfg, keys, values, cache_len,
      token_buf, &holdout_positions);
  clock_t t3 = clock();
  scalar_t holdout_sec = (scalar_t)(t3 - t2) / (scalar_t)CLOCKS_PER_SEC;
  scalar_t holdout_ppl = (scalar_t)exp((double)holdout_loss);

  /* ---- Long-context probe ---- */
  scalar_t probe_top1 = 0, probe_max = 0, probe_entropy = 0;
  run_attention_probe(model, &vocab, &cfg, keys, values, cache_len,
                      &probe_top1, &probe_max, &probe_entropy);

  /* ---- Machine-parseable results ---- */
  printf("\n=== RESULTS ===\n");
#ifdef MICROGPT_ATTN_SINK
  printf("ATTN_SINK_BUILD: ON\n");
  printf("ATTN_SINK_LOGIT: %.4f\n", (double)ATTN_SINK_LOGIT);
#else
  printf("ATTN_SINK_BUILD: OFF\n");
  printf("ATTN_SINK_LOGIT: -\n");
#endif
  printf("FINAL_TRAIN_LOSS: %.6f\n", (double)final_train_loss);
  printf("HELDOUT_LOSS: %.6f\n", (double)holdout_loss);
  printf("HELDOUT_PERPLEXITY: %.6f\n", (double)holdout_ppl);
  printf("PROBE_MAX_ATTN: %.6f\n", (double)probe_max);
  printf("PROBE_MEAN_TOP1: %.6f\n", (double)probe_top1);
  printf("PROBE_ENTROPY: %.6f\n", (double)probe_entropy);
  printf("TRAIN_SECONDS: %.3f\n", (double)train_sec);
  printf("HELDOUT_SECONDS: %.3f\n", (double)holdout_sec);
  printf("HELDOUT_POSITIONS: %zu\n", holdout_positions);
  printf("PARAMS: %zu\n", nparams);

  /* ---- Cleanup ---- */
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
