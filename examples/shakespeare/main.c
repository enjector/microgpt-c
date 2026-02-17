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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SHAKES_SAMPLES 5
#define SHAKES_TEMP 0.7
#define KEEP_TOP_WORDS 6000

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

  /* ---- Create model ---- */
  Model *model = model_create(wv.vocab_size);
  if (!model) {
    fprintf(stderr, "OOM\n");
    return 1;
  }
  size_t nparams = model_num_params(model);
  printf("params: %zu | batch %d | steps %d | lr %.4f\n\n", nparams, BATCH_SIZE,
         NUM_STEPS, (double)LEARNING_RATE);

  /* Allocate optimiser state */
  double *grad_buffer = (double *)calloc(nparams, sizeof(double));
  double *m_buf = (double *)calloc(nparams, sizeof(double));
  double *v_buf = (double *)calloc(nparams, sizeof(double));
  if (!grad_buffer || !m_buf || !v_buf) {
    fprintf(stderr, "OOM (optimiser)\n");
    return 1;
  }

  /* KV cache */
  double *keys[N_LAYER], *values[N_LAYER];
  size_t cache_len[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    keys[L] = (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
    values[L] = (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
    if (!keys[L] || !values[L]) {
      fprintf(stderr, "OOM (KV cache)\n");
      return 1;
    }
  }

  /* ---- Training ---- */
  size_t tokens_trained = 0;
  clock_t t0 = clock();

  for (int step = 0; step < NUM_STEPS; step++) {
    memset(grad_buffer, 0, nparams * sizeof(double));
    double batch_loss = 0;
    size_t batch_positions = 0;

    for (int b = 0; b < BATCH_SIZE; b++) {
      for (int L = 0; L < N_LAYER; L++)
        cache_len[L] = 0;

      size_t offset = (size_t)rand() % num_chunks;
      size_t n = BLOCK_SIZE;
      batch_positions += n;
      tokens_trained += n;

      for (size_t pos = 0; pos < n; pos++) {
        double loss = forward_backward_one(model, all_tokens[offset + pos], pos,
                                           all_tokens[offset + pos + 1], keys,
                                           values, cache_len, grad_buffer);
        batch_loss += loss;
      }
    }

    double mean_loss = batch_loss / (double)batch_positions;
    for (size_t i = 0; i < nparams; i++)
      grad_buffer[i] /= (double)batch_positions;
    adam_step(model, grad_buffer, m_buf, v_buf, step);

    if ((step + 1) % 500 == 0 || step == 0)
      printf("step %5d / %d | loss %.4f\n", step + 1, NUM_STEPS, mean_loss);
  }

  double train_sec = (double)(clock() - t0) / (double)CLOCKS_PER_SEC;
  printf("\nTraining: %.1fs | %.0f steps/s | %.1fk tok/s\n", train_sec,
         (double)NUM_STEPS / train_sec,
         (double)tokens_trained / train_sec / 1000.0);

  /* ---- Generate Shakespeare-style text ---- */
  printf("\n--- generated Shakespeare (word-level) ---\n");
  double logits_buf[MAX_VOCAB];

  const char *seeds[] = {"The", "O", "What", "My", "How"};

  for (int s = 0; s < SHAKES_SAMPLES; s++) {
    for (int L = 0; L < N_LAYER; L++)
      cache_len[L] = 0;

    size_t token = word_to_id(&wv, seeds[s]);
    printf("\n[sample %d]\n%s", s + 1, seeds[s]);

    for (int pos = 0; pos < BLOCK_SIZE - 1; pos++) {
      forward_inference(model, token, (size_t)pos, keys, values, cache_len,
                        logits_buf);
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
  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(values[L]);
  }
  free(grad_buffer);
  free(m_buf);
  free(v_buf);
  model_free(model);
  free(all_tokens);
  free_word_vocab(&wv);
  return 0;
}
