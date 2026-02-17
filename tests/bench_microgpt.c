/*
 * MicroGPT-C Benchmarks
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Times key operations to measure throughput and latency.
 * Zero dependencies — uses only clock() for timing.
 *
 * Build:  cmake --build build --target bench_microgpt
 * Run:    ./build/bench_microgpt
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Timing helpers ---- */

static double elapsed_ms(clock_t start) {
  return (double)(clock() - start) / (double)CLOCKS_PER_SEC * 1000.0;
}

#define BENCH_HEADER(name)                                                     \
  printf("  %-40s ", name);                                                    \
  fflush(stdout)

#define BENCH_RESULT(ms, metric, unit)                                         \
  printf("%8.2f ms  |  %10.1f %s\n", (ms), (metric), (unit))

/* ==================================================================== */
/*                        BENCHMARKS                                     */
/* ==================================================================== */

static void bench_model_create(void) {
  BENCH_HEADER("model_create (vocab=100)");

  int iters = 50;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    seed_rng(42);
    Model *m = model_create(100);
    model_free(m);
  }
  double ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (double)iters / (ms / 1000.0), "creates/s");
}

static void bench_forward_inference(void) {
  BENCH_HEADER("forward_inference (1 token)");

  seed_rng(42);
  Model *m = model_create(50);
  double logits[MAX_VOCAB];
  double *keys[N_LAYER], *vals[N_LAYER];
  size_t cl[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    keys[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    vals[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
  }

  int iters = 10000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    for (int L = 0; L < N_LAYER; L++)
      cl[L] = 0;
    forward_inference(m, 0, 0, keys, vals, cl, logits);
  }
  double ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (double)iters / (ms / 1000.0), "infer/s");

  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  model_free(m);
}

static void bench_forward_backward(void) {
  BENCH_HEADER("forward_backward_one (1 pos)");

  seed_rng(42);
  Model *m = model_create(50);
  size_t np = model_num_params(m);
  double *grads = (double *)calloc(np, sizeof(double));
  double *keys[N_LAYER], *vals[N_LAYER];
  size_t cl[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    keys[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    vals[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
  }

  int iters = 5000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    for (int L = 0; L < N_LAYER; L++)
      cl[L] = 0;
    memset(grads, 0, np * sizeof(double));
    forward_backward_one(m, 0, 0, 1, keys, vals, cl, grads);
  }
  double ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (double)iters / (ms / 1000.0), "fwd+bwd/s");

  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(grads);
  model_free(m);
}

static void bench_adam_step(void) {
  BENCH_HEADER("adam_step");

  seed_rng(42);
  Model *m = model_create(50);
  size_t np = model_num_params(m);
  double *grads = (double *)calloc(np, sizeof(double));
  double *mom = (double *)calloc(np, sizeof(double));
  double *vel = (double *)calloc(np, sizeof(double));
  /* Fill grads with small values */
  for (size_t i = 0; i < np; i++)
    grads[i] = 0.001 * (double)(i % 100);

  int iters = 10000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    adam_step(m, grads, mom, vel, i);
  }
  double ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (double)iters / (ms / 1000.0), "steps/s");

  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}

static void bench_sample_token(void) {
  BENCH_HEADER("sample_token (vocab=50)");

  seed_rng(42);
  double logits[50];
  for (int i = 0; i < 50; i++)
    logits[i] = (double)(i % 7) - 3.0;

  int iters = 1000000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    sample_token(logits, 50, 0.8);
  }
  double ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (double)iters / (ms / 1000.0), "samples/s");
}

static void bench_tokenize_chars(void) {
  BENCH_HEADER("tokenize (char-level, 12 chars)");

  /* Build a small vocab */
  const char *doc = "hello world!";
  size_t doc_len = 12;
  unsigned char chars[256];
  int seen[256] = {0};
  size_t vc = 0;
  for (size_t i = 0; i < doc_len; i++) {
    unsigned char c = (unsigned char)doc[i];
    if (!seen[c]) {
      chars[vc++] = c;
      seen[c] = 1;
    }
  }
  Vocab vocab;
  vocab.chars = chars;
  vocab.vocab_size = vc + 1;
  vocab.bos_id = vc;

  size_t ids[64];
  int iters = 500000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    tokenize(doc, doc_len, &vocab, ids, 64);
  }
  double ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (double)iters / (ms / 1000.0), "tok/s");
}

static void bench_build_word_vocab(void) {
  BENCH_HEADER("build_word_vocab (1KB text)");

  /* Generate a ~1KB text */
  const char *words[] = {"the",  "quick", "brown", "fox", "jumps",
                         "over", "lazy",  "dog",   "and", "cat"};
  char text[2048];
  size_t pos = 0;
  for (int i = 0; i < 100 && pos < 1024; i++) {
    const char *w = words[i % 10];
    size_t wl = strlen(w);
    if (pos + wl + 1 >= 1024)
      break;
    if (pos > 0)
      text[pos++] = ' ';
    memcpy(text + pos, w, wl);
    pos += wl;
  }
  text[pos] = '\0';

  WordVocab wv;
  int iters = 5000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    memset(&wv, 0, sizeof(wv));
    build_word_vocab(text, pos, 100, &wv);
    free_word_vocab(&wv);
  }
  double ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (double)iters / (ms / 1000.0), "builds/s");
}

static void bench_tokenize_words(void) {
  BENCH_HEADER("tokenize_words (1KB text)");

  /* Generate a ~1KB text */
  const char *words[] = {"the",  "quick", "brown", "fox", "jumps",
                         "over", "lazy",  "dog",   "and", "cat"};
  char text[2048];
  size_t pos = 0;
  for (int i = 0; i < 100 && pos < 1024; i++) {
    const char *w = words[i % 10];
    size_t wl = strlen(w);
    if (pos + wl + 1 >= 1024)
      break;
    if (pos > 0)
      text[pos++] = ' ';
    memcpy(text + pos, w, wl);
    pos += wl;
  }
  text[pos] = '\0';

  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  build_word_vocab(text, pos, 100, &wv);

  size_t ids[512];
  int iters = 200000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    tokenize_words(text, pos, &wv, ids, 512);
  }
  double ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (double)iters / (ms / 1000.0), "tok/s");

  free_word_vocab(&wv);
}

static void bench_checkpoint_roundtrip(void) {
  BENCH_HEADER("checkpoint save+load (vocab=50)");

  seed_rng(42);
  Model *m = model_create(50);
  size_t np = model_num_params(m);
  double *mom = (double *)calloc(np, sizeof(double));
  double *vel = (double *)calloc(np, sizeof(double));
  for (size_t i = 0; i < np; i++) {
    mom[i] = 0.001 * (double)(i % 50);
    vel[i] = 0.0001 * (double)(i % 50);
  }

  int iters = 200;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    checkpoint_save(m, mom, vel, 42, "_bench_ckpt.bin");
    double *m2 = (double *)calloc(np, sizeof(double));
    double *v2 = (double *)calloc(np, sizeof(double));
    int step_out;
    Model *loaded = checkpoint_load("_bench_ckpt.bin", 50, m2, v2, &step_out);
    model_free(loaded);
    free(m2);
    free(v2);
  }
  double ms = elapsed_ms(t0);
  remove("_bench_ckpt.bin");
  BENCH_RESULT(ms / iters, (double)iters / (ms / 1000.0), "roundtrips/s");

  free(mom);
  free(vel);
  model_free(m);
}

static void bench_training_step(void) {
  BENCH_HEADER("full training step (batch=1, seq=8)");

  seed_rng(42);
  Model *m = model_create(20);
  size_t np = model_num_params(m);
  double *grads = (double *)calloc(np, sizeof(double));
  double *mom = (double *)calloc(np, sizeof(double));
  double *vel = (double *)calloc(np, sizeof(double));
  double *keys[N_LAYER], *vals[N_LAYER];
  size_t cl[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    keys[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    vals[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
  }

  size_t seq[] = {0, 1, 2, 3, 4, 5, 6, 7};
  size_t seq_len = 8;

  int iters = 2000;
  clock_t t0 = clock();
  for (int step = 0; step < iters; step++) {
    memset(grads, 0, np * sizeof(double));
    for (int L = 0; L < N_LAYER; L++)
      cl[L] = 0;

    for (size_t p = 0; p < seq_len - 1; p++)
      forward_backward_one(m, seq[p], p, seq[p + 1], keys, vals, cl, grads);

    for (size_t i = 0; i < np; i++)
      grads[i] /= (double)(seq_len - 1);

    adam_step(m, grads, mom, vel, step);
  }
  double ms = elapsed_ms(t0);
  double tok_per_step = (double)(seq_len - 1);
  BENCH_RESULT(ms / iters,
               tok_per_step * (double)iters / (ms / 1000.0) / 1000.0,
               "k tok/s");

  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}

/* ==================================================================== */
/*                            MAIN                                       */
/* ==================================================================== */

int main(void) {
  printf("\n=== MicroGPT-C Benchmarks ===\n");
  printf("N_EMBD=%d  N_LAYER=%d  BLOCK_SIZE=%d  N_HEAD=%d\n\n", N_EMBD, N_LAYER,
         BLOCK_SIZE, N_HEAD);

  printf("[Core Operations]\n");
  bench_model_create();
  bench_forward_inference();
  bench_forward_backward();
  bench_adam_step();
  bench_sample_token();

  printf("\n[Tokenisation]\n");
  bench_tokenize_chars();
  bench_build_word_vocab();
  bench_tokenize_words();

  printf("\n[Serialisation]\n");
  bench_checkpoint_roundtrip();

  printf("\n[End-to-End]\n");
  bench_training_step();

  printf("\n=== Done ===\n\n");
  return 0;
}
