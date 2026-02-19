/*
 * MicroGPT-C — Shakespeare Character-Level Generation Example
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates character-level text generation using the MicroGPT library.
 * Trains a small GPT on Shakespeare's complete works and generates new
 * Shakespearean text character by character — no <unk> tokens, no missing
 * words.
 *
 * Each line of Shakespeare becomes a training document. The model learns
 * spelling, punctuation, verse meter, and dialogue structure at the
 * character level.
 *
 * Build:
 *   cmake --build build --target shakespeare_demo
 *   ./build/shakespeare_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_thread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SHAKES_SAMPLES 5
#define SHAKES_TEMP 0.7
#define GEN_LEN 300 /* characters to generate per sample */
#define CHECKPOINT_FILE "shakespeare.ckpt"
#define TRAINING_LOG "shakespeare.ckpt.log"

/* Max threads (actual count is auto-detected at runtime) */
#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

int main(void) {
  unsigned int train_seed = 42; /* deterministic training */
  unsigned int infer_seed = (unsigned int)time(NULL); /* varied inference */
  srand(train_seed);
  seed_rng(train_seed);

  /* Runtime configuration — Shakespeare-specific overrides */
  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = 128;
  cfg.n_head = 8;
  cfg.mlp_dim = 512;
  cfg.n_layer = 4;
  cfg.block_size = 256;
  cfg.batch_size = 16;
  cfg.num_steps = 30000;
  cfg.learning_rate = 0.001;
  cfg.max_vocab = 200;
  cfg.max_docs = 200000;
  cfg.max_doc_len = 512;
  microgpt_print_config("MicroGPT-C - Shakespeare Demo", &cfg);

  const int nl = cfg.n_layer;

  /* ---- Load Shakespeare as line-per-doc ---- */
  Docs docs = {0};
  if (load_docs("shakespeare.txt", &docs, cfg.max_docs) != 0) {
    fprintf(stderr, "Cannot open shakespeare.txt\n");
    return 1;
  }
  shuffle_docs(&docs);
  printf("loaded %zu lines of Shakespeare\n", docs.num_docs);

  /* Count total characters */
  size_t total_chars = 0;
  for (size_t i = 0; i < docs.num_docs; i++)
    total_chars += docs.doc_lens[i];
  printf("total characters: %zu (%.1f KB)\n", total_chars,
         (scalar_t)total_chars / 1024.0);

  /* Open training log (append mode) */
  FILE *logf = fopen(TRAINING_LOG, "a");
  if (logf) {
    time_t now = time(NULL);
    struct tm *lt = localtime(&now);
    fprintf(logf, "\n========================================\n");
    fprintf(logf, "Run: %04d-%02d-%02d %02d:%02d:%02d\n", lt->tm_year + 1900,
            lt->tm_mon + 1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
    fprintf(logf, "========================================\n");
    fprintf(logf, "Corpus: %zu lines | %zu chars (%.1f KB)\n", docs.num_docs,
            total_chars, (scalar_t)total_chars / 1024.0);
  }

  /* Build character-level vocabulary */
  Vocab vocab = {0};
  build_vocab(&docs, &vocab);
  printf("vocab: %zu characters (no <unk>!)\n", vocab.vocab_size);
  if (logf)
    fprintf(logf, "Vocab: %zu characters\n", vocab.vocab_size);

  if (vocab.vocab_size > (size_t)cfg.max_vocab) {
    fprintf(stderr, "vocab_size %zu exceeds max_vocab %d\n", vocab.vocab_size,
            cfg.max_vocab);
    free_docs(&docs);
    free(vocab.chars);
    return 1;
  }

  printf("N_EMBD=%d BLOCK_SIZE=%d N_LAYER=%d N_HEAD=%d\n\n", cfg.n_embd,
         cfg.block_size, cfg.n_layer, cfg.n_head);

  /* ---- Create or load model ---- */
  size_t nparams;
  scalar_t *grad_buffer, *m_buf, *v_buf;
  Model *model = NULL;
  int trained = 0;

  {
    Model *tmp = model_create(vocab.vocab_size, &cfg);
    if (!tmp) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
    nparams = model_num_params(tmp);
    model_free(tmp);
  }

  m_buf = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  v_buf = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  grad_buffer = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  if (!grad_buffer || !m_buf || !v_buf) {
    fprintf(stderr, "OOM (optimiser)\n");
    return 1;
  }

  /* Try loading checkpoint */
  int resume_step = 0;
  model = checkpoint_load(CHECKPOINT_FILE, vocab.vocab_size, &cfg, m_buf, v_buf,
                          &resume_step);
  if (model) {
    printf("loaded checkpoint '%s' (trained %d steps) — skipping training\n\n",
           CHECKPOINT_FILE, resume_step);
    trained = 1;
  } else {
    model = model_create(vocab.vocab_size, &cfg);
    if (!model) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
  }

  int nthreads = mgpt_default_threads(cfg.batch_size);
  printf("params: %zu | batch %d | steps %d | lr %.4f | threads %d\n\n",
         nparams, cfg.batch_size, cfg.num_steps, (scalar_t)cfg.learning_rate,
         nthreads);
  if (logf) {
    fprintf(logf,
            "Architecture: N_EMBD=%d N_LAYER=%d N_HEAD=%d BLOCK_SIZE=%d\n",
            cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.block_size);
    fprintf(logf, "Params: %zu\n", nparams);
    fprintf(logf, "Training: batch=%d steps=%d lr=%.4f threads=%d\n",
            cfg.batch_size, cfg.num_steps, (scalar_t)cfg.learning_rate,
            nthreads);
    fprintf(logf, "Seeds: train=%u infer=%u\n", train_seed, infer_seed);
    fprintf(logf, "Inference: temp=%.2f gen_len=%d samples=%d\n",
            (scalar_t)SHAKES_TEMP, GEN_LEN, SHAKES_SAMPLES);
    fflush(logf);
  }

  /* ---- Allocate per-thread resources ---- */
  TrainWorker *workers =
      (TrainWorker *)calloc((size_t)nthreads, sizeof(TrainWorker));
  mgpt_thread_t *threads =
      (mgpt_thread_t *)calloc((size_t)nthreads, sizeof(mgpt_thread_t));
  mgpt_thread_trampoline_t *tramps = (mgpt_thread_trampoline_t *)calloc(
      (size_t)nthreads, sizeof(mgpt_thread_trampoline_t));
  if (!workers || !threads || !tramps) {
    fprintf(stderr, "OOM\n");
    return 1;
  }

  for (int t = 0; t < nthreads; t++) {
    workers[t].model = model;
    workers[t].docs = &docs;
    workers[t].vocab = &vocab;
    workers[t].grads = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    workers[t].keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    workers[t].values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    workers[t].cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    workers[t].token_buf =
        (size_t *)malloc(((size_t)cfg.block_size + 2) * sizeof(size_t));
    if (!workers[t].grads || !workers[t].keys || !workers[t].values ||
        !workers[t].cache_len || !workers[t].token_buf) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
    for (int L = 0; L < nl; L++) {
      workers[t].keys[L] = kv_cache_alloc(&cfg);
      workers[t].values[L] = kv_cache_alloc(&cfg);
      if (!workers[t].keys[L] || !workers[t].values[L]) {
        fprintf(stderr, "OOM (thread KV cache)\n");
        return 1;
      }
    }
  }

  /* KV cache for inference */
  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    inf_keys[L] = kv_cache_alloc(&cfg);
    inf_values[L] = kv_cache_alloc(&cfg);
  }

  /* ---- Training ---- */
  if (!trained) {
    if (logf)
      fprintf(logf, "\n--- Training ---\n");
    size_t tokens_trained = 0;
    time_t t0 = time(NULL);

    for (int step = 0; step < cfg.num_steps; step++) {
      int batches_per_thread = cfg.batch_size / nthreads;
      int remainder = cfg.batch_size % nthreads;
      int cursor = 0;

      for (int t = 0; t < nthreads; t++) {
        workers[t].batch_start = cursor;
        int count = batches_per_thread + (t < remainder ? 1 : 0);
        workers[t].batch_end = cursor + count;
        cursor += count;
        workers[t].rng_seed = (unsigned int)(step * nthreads + t + 1);
      }

      for (int t = 0; t < nthreads; t++)
        mgpt_thread_create(&threads[t], &tramps[t], train_worker_run,
                           &workers[t]);
      for (int t = 0; t < nthreads; t++)
        mgpt_thread_join(threads[t]);

      scalar_t batch_loss = 0;
      size_t batch_positions = 0;
      memset(grad_buffer, 0, nparams * sizeof(scalar_t));

      for (int t = 0; t < nthreads; t++) {
        batch_loss += workers[t].loss;
        batch_positions += workers[t].positions;
        const scalar_t *tg = workers[t].grads;
        for (size_t i = 0; i < nparams; i++)
          grad_buffer[i] += tg[i];
      }
      tokens_trained += batch_positions;

      scalar_t mean_loss = batch_loss / (scalar_t)batch_positions;
      for (size_t i = 0; i < nparams; i++)
        grad_buffer[i] /= (scalar_t)batch_positions;
      adam_step(model, grad_buffer, m_buf, v_buf, step);

      if ((step + 1) % 500 == 0 || step == 0) {
        scalar_t elapsed = difftime(time(NULL), t0);
        if (elapsed < 1.0)
          elapsed = 1.0;
        scalar_t sps = (step + 1) / elapsed;
        scalar_t eta = (cfg.num_steps - step - 1) / sps;
        int el_m = (int)elapsed / 60, el_s = (int)elapsed % 60;
        int eta_m = (int)eta / 60, eta_s = (int)eta % 60;
        printf("step %5d / %d | loss %.4f | %dm%02ds elapsed, ETA %dm%02ds\n",
               step + 1, cfg.num_steps, mean_loss, el_m, el_s, eta_m, eta_s);

        if (logf) {
          fprintf(logf, "step %5d / %d | loss %.4f\n", step + 1, cfg.num_steps,
                  mean_loss);
          fflush(logf);
        }
      }
    }

    scalar_t train_sec = difftime(time(NULL), t0);
    if (train_sec < 1.0)
      train_sec = 1.0;
    printf("\nTraining: %.1fs | %.0f steps/s | %.1fk tok/s\n", train_sec,
           (scalar_t)cfg.num_steps / train_sec,
           (scalar_t)tokens_trained / train_sec / 1000.0);

    if (checkpoint_save(model, m_buf, v_buf, cfg.num_steps, CHECKPOINT_FILE) ==
        0)
      printf("checkpoint saved to '%s'\n", CHECKPOINT_FILE);
    else
      fprintf(stderr, "warning: failed to save checkpoint\n");

    if (logf) {
      fprintf(logf, "\nTraining complete: %.1fs | %.0f steps/s | %.1fk tok/s\n",
              train_sec, (scalar_t)cfg.num_steps / train_sec,
              (scalar_t)tokens_trained / train_sec / 1000.0);
      fprintf(logf, "Final checkpoint: %s\n", CHECKPOINT_FILE);
      fflush(logf);
    }
  } else {
    if (logf)
      fprintf(logf, "Loaded checkpoint: %s (trained %d steps)\n",
              CHECKPOINT_FILE, resume_step);
  }

  /* ---- Generate Shakespeare ---- */
  seed_rng(infer_seed); /* re-seed so each run generates different text */
  printf("\n--- generated Shakespeare (character-level) ---\n");
  scalar_t *logits_buf =
      (scalar_t *)malloc((size_t)cfg.max_vocab * sizeof(scalar_t));

  /* Inference timing accumulators */
  int total_inf_tokens = 0;
  double total_inf_time = 0.0;

  /* Seed prompts: just the first character */
  const char seeds[] = {'T', 'O', 'W', 'M', 'H'};
  const char *seed_names[] = {"T(he)", "O", "W(hat)", "M(y)", "H(ow)"};
  (void)seed_names;

  for (int s = 0; s < SHAKES_SAMPLES; s++) {
    for (int L = 0; L < nl; L++)
      inf_cache_len[L] = 0;

    /* Find the token ID for the seed character */
    size_t token = vocab.bos_id; /* fallback */
    for (size_t c = 0; c < vocab.vocab_size; c++) {
      if (vocab.chars[c] == (unsigned char)seeds[s]) {
        token = c;
        break;
      }
    }

    printf("\n[sample %d — seed: '%c']\n%c", s + 1, seeds[s], seeds[s]);

    /* Generate up to GEN_LEN characters but stay within block_size */
    int gen_count = GEN_LEN;
    if (gen_count > cfg.block_size - 1)
      gen_count = cfg.block_size - 1;
    int gen_len = 0;

    /* Start inference timer */
    struct timespec inf_start, inf_end;
    clock_gettime(CLOCK_MONOTONIC, &inf_start);

    for (int pos = 0; pos < gen_count; pos++) {
      forward_inference(model, token, (size_t)pos, inf_keys, inf_values,
                        inf_cache_len, logits_buf);
      token = sample_token(logits_buf, vocab.vocab_size, SHAKES_TEMP);
      if (token == vocab.bos_id)
        break;
      putchar((char)vocab.chars[token]);
      gen_len++;
    }

    /* Stop timer */
    clock_gettime(CLOCK_MONOTONIC, &inf_end);
    double inf_elapsed = (inf_end.tv_sec - inf_start.tv_sec) +
                         (inf_end.tv_nsec - inf_start.tv_nsec) / 1e9;
    int tokens_this_sample = gen_len + 1; /* +BOS/seed */
    double tok_per_sec = inf_elapsed > 0 ? tokens_this_sample / inf_elapsed : 0;
    total_inf_tokens += tokens_this_sample;
    total_inf_time += inf_elapsed;
    printf("\n  >> %d tok in %.3fs (%.0f tok/s)\n", tokens_this_sample,
           inf_elapsed, tok_per_sec);

    if (logf) {
      fprintf(logf, "  sample %d: seed='%c' | %d tok | %.0f tok/s\n", s + 1,
              seeds[s], tokens_this_sample, tok_per_sec);
    }
  }

  /* ---- Inference summary ---- */
  printf("\n--- inference summary ---\n");
  printf(
      "samples: %d | total tokens: %d | total time: %.3fs | avg: %.0f tok/s\n",
      SHAKES_SAMPLES, total_inf_tokens, total_inf_time,
      total_inf_time > 0 ? total_inf_tokens / total_inf_time : 0);

  if (logf) {
    fprintf(logf, "\n--- Inference Summary ---\n");
    fprintf(logf,
            "samples: %d | total_tokens: %d | time: %.3fs | avg: %.0f tok/s\n",
            SHAKES_SAMPLES, total_inf_tokens, total_inf_time,
            total_inf_time > 0 ? total_inf_tokens / total_inf_time : 0);
    fprintf(logf, "========================================\n\n");
    fclose(logf);
    logf = NULL;
    printf("training log appended to '%s'\n", TRAINING_LOG);
  }

  /* ---- Cleanup ---- */
  for (int t = 0; t < nthreads; t++) {
    for (int L = 0; L < nl; L++) {
      kv_cache_free(workers[t].keys[L]);
      kv_cache_free(workers[t].values[L]);
    }
    free(workers[t].keys);
    free(workers[t].values);
    free(workers[t].cache_len);
    free(workers[t].token_buf);
    free(workers[t].grads);
  }
  free(workers);
  free(threads);
  free(tramps);
  for (int L = 0; L < nl; L++) {
    kv_cache_free(inf_keys[L]);
    kv_cache_free(inf_values[L]);
  }
  free(inf_keys);
  free(inf_values);
  free(inf_cache_len);
  free(logits_buf);
  free(grad_buffer);
  free(m_buf);
  free(v_buf);
  model_free(model);
  free_docs(&docs);
  free(vocab.chars);
  return 0;
}
