/*
 * MicroGPT-C — Shakespeare Word-Level Generation Example
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates word-level text generation using the MicroGPT library's
 * WordVocab API.  Trains a small GPT on Shakespeare's complete works and
 * generates new Shakespearean text.
 *
 * Build:
 *   cmake --build build --target shakespeare_demo
 *   ./build/shakespeare_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SHAKES_SAMPLES 5
#define SHAKES_TEMP 0.7
#define KEEP_TOP_WORDS 10000
#define CHECKPOINT_FILE "shakespeare.ckpt"

/* Number of worker threads for batch parallelism */
#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

/* ---- Per-thread work descriptor ---- */
typedef struct {
  /* Shared (read-only during batch) */
  const Model *model;
  const size_t *all_tokens;
  size_t num_chunks;
  /* Per-thread owned */
  double *grads; /* thread-local gradient buffer */
  double *keys[N_LAYER];
  double *values[N_LAYER];
  size_t cache_len[N_LAYER];
  /* Work assignment */
  int batch_start; /* first batch index for this thread */
  int batch_end;   /* one past last batch index */
  /* Results */
  double loss;
  size_t positions;
  unsigned int rng_seed; /* thread-local RNG seed */
} WorkerArg;

static void *train_batch_worker(void *arg) {
  WorkerArg *w = (WorkerArg *)arg;
  size_t nparams = model_num_params(w->model);
  memset(w->grads, 0, nparams * sizeof(double));
  w->loss = 0;
  w->positions = 0;

  for (int b = w->batch_start; b < w->batch_end; b++) {
    for (int L = 0; L < N_LAYER; L++)
      w->cache_len[L] = 0;

    size_t offset = (size_t)rand_r(&w->rng_seed) % w->num_chunks;
    size_t n = BLOCK_SIZE;
    w->positions += n;

    for (size_t pos = 0; pos < n; pos++) {
      double loss =
          forward_backward_one(w->model, w->all_tokens[offset + pos], pos,
                               w->all_tokens[offset + pos + 1], w->keys,
                               w->values, w->cache_len, w->grads);
      w->loss += loss;
    }
  }
  return NULL;
}

int main(void) {
  srand(42);
  seed_rng(42);

  /* ---- Load Shakespeare ---- */
  size_t text_len;
  char *text = load_file("shakespeare.txt", &text_len);
  if (!text) {
    fprintf(stderr, "Cannot open shakespeare.txt\n");
    return 1;
  }
  printf("loaded %.1f KB of Shakespeare\n", (double)text_len / 1024.0);

  /* ---- Build word vocabulary using library API ---- */
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  if (build_word_vocab(text, text_len, KEEP_TOP_WORDS, &wv) != 0) {
    fprintf(stderr, "build_word_vocab failed\n");
    free(text);
    return 1;
  }

  if (wv.vocab_size > MAX_VOCAB) {
    fprintf(stderr,
            "vocab_size %zu exceeds MAX_VOCAB %d — increase MAX_VOCAB\n",
            wv.vocab_size, MAX_VOCAB);
    free_word_vocab(&wv);
    free(text);
    return 1;
  }

  printf("word vocab: %zu words kept | vocab_size %zu (incl. specials)\n",
         wv.num_words, wv.vocab_size);
  printf("N_EMBD=%d BLOCK_SIZE=%d N_LAYER=%d MAX_VOCAB=%d\n\n", N_EMBD,
         BLOCK_SIZE, N_LAYER, MAX_VOCAB);

  /* ---- Tokenize entire text into word token IDs ---- */
  size_t max_tokens = text_len;
  size_t *all_tokens = (size_t *)malloc(max_tokens * sizeof(size_t));
  if (!all_tokens) {
    fprintf(stderr, "OOM (tokenizing)\n");
    return 1;
  }
  size_t total_tokens =
      tokenize_words(text, text_len, &wv, all_tokens, max_tokens);
  printf("tokenized: %zu word tokens\n", total_tokens);
  free(text);

  /* ---- Create sliding-window chunks for training ---- */
  size_t num_chunks = 0;
  if (total_tokens > BLOCK_SIZE)
    num_chunks = total_tokens - BLOCK_SIZE;
  printf("training chunks: %zu (sliding window of %d tokens)\n\n", num_chunks,
         BLOCK_SIZE + 1);

  /* ---- Create or load model ---- */
  size_t nparams;
  double *grad_buffer, *m_buf, *v_buf;
  Model *model = NULL;
  int trained = 0;

  /* Try loading a saved checkpoint first */
  {
    Model *tmp = model_create(wv.vocab_size);
    if (!tmp) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
    nparams = model_num_params(tmp);
    model_free(tmp);
  }

  m_buf = (double *)calloc(nparams, sizeof(double));
  v_buf = (double *)calloc(nparams, sizeof(double));
  grad_buffer = (double *)calloc(nparams, sizeof(double));
  if (!grad_buffer || !m_buf || !v_buf) {
    fprintf(stderr, "OOM (optimiser)\n");
    return 1;
  }

  int resume_step = 0;
  model = checkpoint_load(CHECKPOINT_FILE, wv.vocab_size, m_buf, v_buf,
                          &resume_step);
  if (model) {
    printf("loaded checkpoint '%s' (trained %d steps) — skipping training\n\n",
           CHECKPOINT_FILE, resume_step);
    trained = 1;
  } else {
    model = model_create(wv.vocab_size);
    if (!model) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
  }

  printf("params: %zu | batch %d | steps %d | lr %.4f | threads %d\n\n",
         nparams, BATCH_SIZE, NUM_STEPS, (double)LEARNING_RATE, NUM_THREADS);

  /* ---- Allocate per-thread resources ---- */
  int nthreads = NUM_THREADS;
  if (nthreads > BATCH_SIZE)
    nthreads = BATCH_SIZE;

  WorkerArg workers[NUM_THREADS];
  pthread_t threads[NUM_THREADS];

  for (int t = 0; t < nthreads; t++) {
    workers[t].model = model;
    workers[t].all_tokens = all_tokens;
    workers[t].num_chunks = num_chunks;
    workers[t].grads = (double *)calloc(nparams, sizeof(double));
    if (!workers[t].grads) {
      fprintf(stderr, "OOM (thread grads)\n");
      return 1;
    }
    for (int L = 0; L < N_LAYER; L++) {
      workers[t].keys[L] =
          (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
      workers[t].values[L] =
          (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
      if (!workers[t].keys[L] || !workers[t].values[L]) {
        fprintf(stderr, "OOM (thread KV cache)\n");
        return 1;
      }
    }
  }

  /* KV cache for inference (single-threaded) */
  double *inf_keys[N_LAYER], *inf_values[N_LAYER];
  size_t inf_cache_len[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    inf_keys[L] =
        (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
    inf_values[L] =
        (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
  }

  /* ---- Training (skipped if checkpoint loaded) ---- */
  if (!trained) {
    size_t tokens_trained = 0;
    time_t t0 = time(NULL);

    for (int step = 0; step < NUM_STEPS; step++) {
      /* Distribute batches across threads */
      int batches_per_thread = BATCH_SIZE / nthreads;
      int remainder = BATCH_SIZE % nthreads;
      int cursor = 0;

      for (int t = 0; t < nthreads; t++) {
        workers[t].batch_start = cursor;
        int count = batches_per_thread + (t < remainder ? 1 : 0);
        workers[t].batch_end = cursor + count;
        cursor += count;
        workers[t].rng_seed = (unsigned int)(step * nthreads + t + 1);
      }

      /* Launch threads */
      for (int t = 0; t < nthreads; t++)
        pthread_create(&threads[t], NULL, train_batch_worker, &workers[t]);

      /* Wait for all threads */
      for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], NULL);

      /* Sum thread-local gradients and losses */
      double batch_loss = 0;
      size_t batch_positions = 0;
      memset(grad_buffer, 0, nparams * sizeof(double));

      for (int t = 0; t < nthreads; t++) {
        batch_loss += workers[t].loss;
        batch_positions += workers[t].positions;
        const double *tg = workers[t].grads;
        for (size_t i = 0; i < nparams; i++)
          grad_buffer[i] += tg[i];
      }
      tokens_trained += batch_positions;

      double mean_loss = batch_loss / (double)batch_positions;
      for (size_t i = 0; i < nparams; i++)
        grad_buffer[i] /= (double)batch_positions;
      adam_step(model, grad_buffer, m_buf, v_buf, step);

      if ((step + 1) % 500 == 0 || step == 0)
        printf("step %5d / %d | loss %.4f\n", step + 1, NUM_STEPS, mean_loss);
    }

    double train_sec = difftime(time(NULL), t0);
    if (train_sec < 1.0)
      train_sec = 1.0;
    printf("\nTraining: %.1fs | %.0f steps/s | %.1fk tok/s\n", train_sec,
           (double)NUM_STEPS / train_sec,
           (double)tokens_trained / train_sec / 1000.0);

    /* Save checkpoint for next run */
    if (checkpoint_save(model, m_buf, v_buf, NUM_STEPS, CHECKPOINT_FILE) == 0)
      printf("checkpoint saved to '%s'\n", CHECKPOINT_FILE);
    else
      fprintf(stderr, "warning: failed to save checkpoint\n");
  }

  /* ---- Generate Shakespeare-style text ---- */
  printf("\n--- generated Shakespeare (word-level) ---\n");
  double logits_buf[MAX_VOCAB];

  const char *seeds[] = {"The", "O", "What", "My", "How"};

  for (int s = 0; s < SHAKES_SAMPLES; s++) {
    for (int L = 0; L < N_LAYER; L++)
      inf_cache_len[L] = 0;

    size_t token = word_to_id(&wv, seeds[s]);
    printf("\n[sample %d]\n%s", s + 1, seeds[s]);

    for (int pos = 0; pos < BLOCK_SIZE - 1; pos++) {
      forward_inference(model, token, (size_t)pos, inf_keys, inf_values,
                        inf_cache_len, logits_buf);
      token = sample_token(logits_buf, wv.vocab_size, SHAKES_TEMP);
      if (token == wv.bos_id)
        break;

      const char *w = wv.words[token];
      if (token == wv.newline_id)
        printf("%s", w);
      else
        printf(" %s", w);
    }
    printf("\n");
  }

  /* Cleanup */
  for (int t = 0; t < nthreads; t++) {
    for (int L = 0; L < N_LAYER; L++) {
      free(workers[t].keys[L]);
      free(workers[t].values[L]);
    }
    free(workers[t].grads);
  }
  for (int L = 0; L < N_LAYER; L++) {
    free(inf_keys[L]);
    free(inf_values[L]);
  }
  free(grad_buffer);
  free(m_buf);
  free(v_buf);
  model_free(model);
  free(all_tokens);
  free_word_vocab(&wv);
  return 0;
}
