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

/* Max threads (actual count is auto-detected at runtime) */
#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

/* ---- Per-thread work descriptor ---- */
typedef struct {
  /* Shared (read-only during batch) */
  const Model *model;
  const Docs *docs;
  const Vocab *vocab;
  /* Per-thread owned */
  double *grads; /* thread-local gradient buffer */
  double *keys[N_LAYER];
  double *values[N_LAYER];
  size_t cache_len[N_LAYER];
  size_t token_buf[BLOCK_SIZE + 2];
  /* Work assignment */
  int batch_start;
  int batch_end;
  /* Results */
  double loss;
  size_t positions;
  unsigned int rng_seed;
  size_t doc_start; /* starting doc index for this thread's batch */
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

    /* Pick a random document (line of Shakespeare) */
    size_t di = (size_t)rand_r(&w->rng_seed) % w->docs->num_docs;
    const char *doc = w->docs->lines[di];
    size_t doc_len = w->docs->doc_lens[di];

    /* Tokenize: [BOS] chars... [BOS/EOS] */
    size_t n_tok =
        tokenize(doc, doc_len, w->vocab, w->token_buf, BLOCK_SIZE + 2);
    size_t n = n_tok - 1;
    if (n > BLOCK_SIZE)
      n = BLOCK_SIZE;
    if (n == 0)
      continue;
    w->positions += n;

    for (size_t pos = 0; pos < n; pos++) {
      double loss = forward_backward_one(w->model, w->token_buf[pos], pos,
                                         w->token_buf[pos + 1], w->keys,
                                         w->values, w->cache_len, w->grads);
      w->loss += loss;
    }
  }
  return NULL;
}

/* Utility: shuffle docs using Fisher-Yates */
static void shuffle_docs(Docs *docs) {
  for (size_t i = docs->num_docs; i > 1; i--) {
    size_t j = (size_t)rand() % i;
    char *tmp_line = docs->lines[j];
    size_t tmp_len = docs->doc_lens[j];
    docs->lines[j] = docs->lines[i - 1];
    docs->doc_lens[j] = docs->doc_lens[i - 1];
    docs->lines[i - 1] = tmp_line;
    docs->doc_lens[i - 1] = tmp_len;
  }
}

int main(void) {
  unsigned int train_seed = 42; /* deterministic training */
  unsigned int infer_seed = (unsigned int)time(NULL); /* varied inference */
  srand(train_seed);
  seed_rng(train_seed);

  /* ---- Load Shakespeare as line-per-doc ---- */
  Docs docs = {0};
  if (load_docs("shakespeare.txt", &docs) != 0) {
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
         (double)total_chars / 1024.0);

  /* Build character-level vocabulary */
  Vocab vocab = {0};
  build_vocab(&docs, &vocab);
  printf("vocab: %zu characters (no <unk>!)\n", vocab.vocab_size);

  if (vocab.vocab_size > MAX_VOCAB) {
    fprintf(stderr, "vocab_size %zu exceeds MAX_VOCAB %d\n", vocab.vocab_size,
            MAX_VOCAB);
    free_docs(&docs);
    free(vocab.chars);
    return 1;
  }

  printf("N_EMBD=%d BLOCK_SIZE=%d N_LAYER=%d N_HEAD=%d\n\n", N_EMBD, BLOCK_SIZE,
         N_LAYER, N_HEAD);

  /* ---- Create or load model ---- */
  size_t nparams;
  double *grad_buffer, *m_buf, *v_buf;
  Model *model = NULL;
  int trained = 0;

  {
    Model *tmp = model_create(vocab.vocab_size);
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

  /* Try loading checkpoint */
  int resume_step = 0;
  model = checkpoint_load(CHECKPOINT_FILE, vocab.vocab_size, m_buf, v_buf,
                          &resume_step);
  if (model) {
    printf("loaded checkpoint '%s' (trained %d steps) — skipping training\n\n",
           CHECKPOINT_FILE, resume_step);
    trained = 1;
  } else {
    model = model_create(vocab.vocab_size);
    if (!model) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
  }

  int nthreads = mgpt_default_threads(BATCH_SIZE);
  printf("params: %zu | batch %d | steps %d | lr %.4f | threads %d\n\n",
         nparams, BATCH_SIZE, NUM_STEPS, (double)LEARNING_RATE, nthreads);

  /* ---- Allocate per-thread resources ---- */
  WorkerArg *workers = (WorkerArg *)calloc((size_t)nthreads, sizeof(WorkerArg));
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
    workers[t].grads = (double *)calloc(nparams, sizeof(double));
    if (!workers[t].grads) {
      fprintf(stderr, "OOM\n");
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

  /* KV cache for inference */
  double *inf_keys[N_LAYER], *inf_values[N_LAYER];
  size_t inf_cache_len[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    inf_keys[L] =
        (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
    inf_values[L] =
        (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
  }

  /* ---- Training ---- */
  if (!trained) {
    size_t tokens_trained = 0;
    time_t t0 = time(NULL);

    for (int step = 0; step < NUM_STEPS; step++) {
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

      for (int t = 0; t < nthreads; t++)
        mgpt_thread_create(&threads[t], &tramps[t], train_batch_worker,
                           &workers[t]);
      for (int t = 0; t < nthreads; t++)
        mgpt_thread_join(threads[t]);

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

    if (checkpoint_save(model, m_buf, v_buf, NUM_STEPS, CHECKPOINT_FILE) == 0)
      printf("checkpoint saved to '%s'\n", CHECKPOINT_FILE);
    else
      fprintf(stderr, "warning: failed to save checkpoint\n");
  }

  /* ---- Generate Shakespeare ---- */
  seed_rng(infer_seed); /* re-seed so each run generates different text */
  printf("\n--- generated Shakespeare (character-level) ---\n");
  double logits_buf[MAX_VOCAB];

  /* Seed prompts: just the first character */
  const char seeds[] = {'T', 'O', 'W', 'M', 'H'};
  const char *seed_names[] = {"T(he)", "O", "W(hat)", "M(y)", "H(ow)"};

  for (int s = 0; s < SHAKES_SAMPLES; s++) {
    for (int L = 0; L < N_LAYER; L++)
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

    /* Generate up to GEN_LEN characters but stay within BLOCK_SIZE */
    int gen_count = GEN_LEN;
    if (gen_count > BLOCK_SIZE - 1)
      gen_count = BLOCK_SIZE - 1;

    for (int pos = 0; pos < gen_count; pos++) {
      forward_inference(model, token, (size_t)pos, inf_keys, inf_values,
                        inf_cache_len, logits_buf);
      token = sample_token(logits_buf, vocab.vocab_size, SHAKES_TEMP);
      if (token == vocab.bos_id)
        break;
      putchar((char)vocab.chars[token]);
    }
    printf("\n");
  }

  /* ---- Cleanup ---- */
  for (int t = 0; t < nthreads; t++) {
    for (int L = 0; L < N_LAYER; L++) {
      free(workers[t].keys[L]);
      free(workers[t].values[L]);
    }
    free(workers[t].grads);
  }
  free(workers);
  free(threads);
  free(tramps);
  for (int L = 0; L < N_LAYER; L++) {
    free(inf_keys[L]);
    free(inf_values[L]);
  }
  free(grad_buffer);
  free(m_buf);
  free(v_buf);
  model_free(model);
  free_docs(&docs);
  free(vocab.chars);
  return 0;
}
