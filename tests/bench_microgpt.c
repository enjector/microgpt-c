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

static MicrogptConfig g_cfg;

/* MSVC does not support Variable Length Arrays (VLAs).
 * We use fixed maximum capacities for benchmark buffers. */
#define MAX_LAYER_VAL 32
#define MAX_VOCAB_VAL 1024

/* ---- Timing helpers ---- */

static scalar_t elapsed_ms(clock_t start) {
  return (scalar_t)(clock() - start) / (scalar_t)CLOCKS_PER_SEC * 1000.0;
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
    Model *m = model_create(100, &g_cfg);
    model_free(m);
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (scalar_t)iters / (ms / 1000.0), "creates/s");
}

static void bench_forward_inference(void) {
  BENCH_HEADER("forward_inference (1 token)");

  seed_rng(42);
  Model *m = model_create(50, &g_cfg);
  scalar_t logits[MAX_VOCAB_VAL];
  scalar_t *keys[MAX_LAYER_VAL], *vals[MAX_LAYER_VAL];
  size_t cl[MAX_LAYER_VAL];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    keys[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
    vals[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
  }

  int iters = 10000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;
    forward_inference(m, 0, 0, keys, vals, cl, logits);
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (scalar_t)iters / (ms / 1000.0), "infer/s");

  for (int L = 0; L < g_cfg.n_layer; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  model_free(m);
}

static void bench_forward_backward(void) {
  BENCH_HEADER("forward_backward_one (1 pos)");

  seed_rng(42);
  Model *m = model_create(50, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[MAX_LAYER_VAL], *vals[MAX_LAYER_VAL];
  size_t cl[MAX_LAYER_VAL];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    keys[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
    vals[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
  }

  int iters = 5000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;
    memset(grads, 0, np * sizeof(scalar_t));
    forward_backward_one(m, 0, 0, 1, keys, vals, cl, grads);
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (scalar_t)iters / (ms / 1000.0), "fwd+bwd/s");

  for (int L = 0; L < g_cfg.n_layer; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(grads);
  model_free(m);
}

static void bench_adam_step(void) {
  BENCH_HEADER("adam_step");

  seed_rng(42);
  Model *m = model_create(50, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
  /* Fill grads with small values */
  for (size_t i = 0; i < np; i++)
    grads[i] = 0.001 * (scalar_t)(i % 100);

  int iters = 10000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    adam_step(m, grads, mom, vel, i);
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (scalar_t)iters / (ms / 1000.0), "steps/s");

  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}

static void bench_sample_token(void) {
  BENCH_HEADER("sample_token (vocab=50)");

  seed_rng(42);
  scalar_t logits[50];
  for (int i = 0; i < 50; i++)
    logits[i] = (scalar_t)(i % 7) - 3.0;

  int iters = 1000000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    sample_token(logits, 50, 0.8);
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (scalar_t)iters / (ms / 1000.0), "samples/s");
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
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (scalar_t)iters / (ms / 1000.0), "tok/s");
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
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, (scalar_t)iters / (ms / 1000.0), "builds/s");
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
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (scalar_t)iters / (ms / 1000.0), "tok/s");

  free_word_vocab(&wv);
}

static void bench_checkpoint_roundtrip(void) {
  BENCH_HEADER("checkpoint save+load (vocab=50)");

  seed_rng(42);
  Model *m = model_create(50, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
  for (size_t i = 0; i < np; i++) {
    mom[i] = 0.001 * (scalar_t)(i % 50);
    vel[i] = 0.0001 * (scalar_t)(i % 50);
  }

  int iters = 200;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    checkpoint_save(m, mom, vel, 42, "_bench_ckpt.bin");
    scalar_t *m2 = (scalar_t *)calloc(np, sizeof(scalar_t));
    scalar_t *v2 = (scalar_t *)calloc(np, sizeof(scalar_t));
    int step_out;
    Model *loaded =
        checkpoint_load("_bench_ckpt.bin", 50, &g_cfg, m2, v2, &step_out);
    model_free(loaded);
    free(m2);
    free(v2);
  }
  scalar_t ms = elapsed_ms(t0);
  remove("_bench_ckpt.bin");
  BENCH_RESULT(ms / iters, (scalar_t)iters / (ms / 1000.0), "roundtrips/s");

  free(mom);
  free(vel);
  model_free(m);
}

static void bench_training_step(void) {
  BENCH_HEADER("full training step (batch=1, seq=8)");

  seed_rng(42);
  Model *m = model_create(20, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[MAX_LAYER_VAL], *vals[MAX_LAYER_VAL];
  size_t cl[MAX_LAYER_VAL];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    keys[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
    vals[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
  }

  size_t seq[] = {0, 1, 2, 3, 4, 5, 6, 7};
  size_t seq_len = 8;

  int iters = 2000;
  clock_t t0 = clock();
  for (int step = 0; step < iters; step++) {
    memset(grads, 0, np * sizeof(scalar_t));
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;

    for (size_t p = 0; p < seq_len - 1; p++)
      forward_backward_one(m, seq[p], p, seq[p + 1], keys, vals, cl, grads);

    for (size_t i = 0; i < np; i++)
      grads[i] /= (scalar_t)(seq_len - 1);

    adam_step(m, grads, mom, vel, step);
  }
  scalar_t ms = elapsed_ms(t0);
  scalar_t tok_per_step = (scalar_t)(seq_len - 1);
  BENCH_RESULT(ms / iters,
               tok_per_step * (scalar_t)iters / (ms / 1000.0) / 1000.0,
               "k tok/s");

  for (int L = 0; L < g_cfg.n_layer; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}
/* ==================================================================== */
/*                    NEW BENCHMARKS                                      */
/* ==================================================================== */

static void bench_inference_sequence(void) {
  BENCH_HEADER("auto-regressive inference (seq=16)");

  seed_rng(42);
  Model *m = model_create(30, &g_cfg);
  scalar_t logits[MAX_VOCAB_VAL];
  scalar_t *keys[MAX_LAYER_VAL], *vals[MAX_LAYER_VAL];
  size_t cl[MAX_LAYER_VAL];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    keys[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
    vals[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
  }

  int iters = 500;
  int seq_len = 16;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;
    size_t tok = 0;
    for (int p = 0; p < seq_len; p++) {
      forward_inference(m, tok, (size_t)p, keys, vals, cl, logits);
      tok = sample_token(logits, 30, 0.8);
    }
  }
  scalar_t ms = elapsed_ms(t0);
  scalar_t total_tokens = (scalar_t)iters * (scalar_t)seq_len;
  BENCH_RESULT(ms / iters, total_tokens / (ms / 1000.0), "tok/s");

  for (int L = 0; L < g_cfg.n_layer; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  model_free(m);
}

static void bench_multi_position_fwd_bwd(void) {
  BENCH_HEADER("fwd+bwd multi-pos (seq=8)");

  seed_rng(42);
  Model *m = model_create(20, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[MAX_LAYER_VAL], *vals[MAX_LAYER_VAL];
  size_t cl[MAX_LAYER_VAL];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    keys[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
    vals[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
  }

  size_t seq[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int iters = 2000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    memset(grads, 0, np * sizeof(scalar_t));
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;
    for (int p = 0; p < 7; p++)
      forward_backward_one(m, seq[p], (size_t)p, seq[p + 1], keys, vals, cl,
                           grads);
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / iters, 7.0 * (scalar_t)iters / (ms / 1000.0), "pos/s");

  for (int L = 0; L < g_cfg.n_layer; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(grads);
  model_free(m);
}

static void bench_model_create_scaling(void) {
  BENCH_HEADER("model_create scaling");

  size_t vocab_sizes[] = {10, 50, 100, 200};
  int num_sizes = 4;
  printf("\n");

  for (int s = 0; s < num_sizes; s++) {
    size_t vs = vocab_sizes[s];
    char label[64];
    snprintf(label, sizeof(label), "    vocab=%zu", vs);

    int iters = 100;
    clock_t t0 = clock();
    for (int i = 0; i < iters; i++) {
      seed_rng(42);
      Model *m = model_create(vs, &g_cfg);
      model_free(m);
    }
    scalar_t ms = elapsed_ms(t0);

    seed_rng(42);
    Model *m = model_create(vs, &g_cfg);
    size_t np = model_num_params(m);
    model_free(m);

    printf("  %-40s %8.2f ms  |  %10zu params\n", label, ms / iters, np);
  }
}

static void bench_convergence_speed(void) {
  BENCH_HEADER("convergence 100 steps");

  seed_rng(42);
  Model *m = model_create(20, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[MAX_LAYER_VAL], *vals[MAX_LAYER_VAL];
  size_t cl[MAX_LAYER_VAL];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    keys[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
    vals[L] = (scalar_t *)calloc(
        (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t));
  }

  size_t seq[] = {0, 1, 2, 3, 4, 5, 6, 7};
  scalar_t first_loss = 0, last_loss = 0;

  clock_t t0 = clock();
  for (int step = 0; step < 100; step++) {
    memset(grads, 0, np * sizeof(scalar_t));
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;
    scalar_t step_loss = 0;
    for (int p = 0; p < 7; p++)
      step_loss += forward_backward_one(m, seq[p], (size_t)p, seq[p + 1], keys,
                                        vals, cl, grads);
    step_loss /= 7.0;
    for (size_t i = 0; i < np; i++)
      grads[i] /= 7.0;
    adam_step(m, grads, mom, vel, step);
    if (step == 0)
      first_loss = step_loss;
    if (step == 99)
      last_loss = step_loss;
  }
  scalar_t ms = elapsed_ms(t0);
  scalar_t reduction = (1.0 - last_loss / first_loss) * 100.0;
  printf("\n");
  printf("    %-40s %8.2f ms\n", "total training time", ms);
  printf("    %-40s %8.2f ms\n", "per step", ms / 100.0);
  printf("    %-40s %8.4f\n", "initial loss", first_loss);
  printf("    %-40s %8.4f\n", "final loss", last_loss);
  printf("    %-40s %7.1f%%\n", "loss reduction", reduction);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}

static void bench_memory_footprint(void) {
  BENCH_HEADER("memory footprint");
  printf("\n");

  size_t vocab_sizes[] = {10, 50, 100, 500};
  int num_sizes = 4;
  for (int s = 0; s < num_sizes; s++) {
    size_t vs = vocab_sizes[s];
    seed_rng(42);
    Model *m = model_create(vs, &g_cfg);
    size_t np = model_num_params(m);
    size_t weight_bytes = np * sizeof(scalar_t);
    size_t optimizer_bytes = np * 3 * sizeof(scalar_t); /* grads+m+v */
    size_t kv_bytes = (size_t)g_cfg.n_layer * 2 * (size_t)g_cfg.block_size *
                      (size_t)g_cfg.n_embd * sizeof(scalar_t);
    size_t total = weight_bytes + optimizer_bytes + kv_bytes;

    printf("    vocab=%-5zu  weights=%.1fKB  optim=%.1fKB  kv=%.1fKB  "
           "total=%.1fKB\n",
           vs, (scalar_t)weight_bytes / 1024.0,
           (scalar_t)optimizer_bytes / 1024.0, (scalar_t)kv_bytes / 1024.0,
           (scalar_t)total / 1024.0);
    model_free(m);
  }
}

/* ==================================================================== */
/*                   TILED LINEAR ALGEBRA                                */
/* ==================================================================== */

#define BENCH_TILE_R 32
#define BENCH_TILE_C 64

static scalar_t bench_randf(void) {
  return 2.0 * ((scalar_t)rand() / (scalar_t)RAND_MAX) - 1.0;
}

/* Naive: y = W @ x */
static void bench_naive_lin_fwd(const scalar_t *x, const scalar_t *W,
                                size_t nin, size_t nout, scalar_t *y) {
  for (size_t j = 0; j < nout; j++) {
    scalar_t s = 0;
    for (size_t i = 0; i < nin; i++)
      s += W[j * nin + i] * x[i];
    y[j] = s;
  }
}

/* Tiled: y = W @ x */
static void bench_tiled_lin_fwd(const scalar_t *x, const scalar_t *W,
                                size_t nin, size_t nout, scalar_t *y) {
  memset(y, 0, nout * sizeof(scalar_t));
  for (size_t j0 = 0; j0 < nout; j0 += BENCH_TILE_R) {
    size_t j1 = (j0 + BENCH_TILE_R < nout) ? j0 + BENCH_TILE_R : nout;
    for (size_t i0 = 0; i0 < nin; i0 += BENCH_TILE_C) {
      size_t i1 = (i0 + BENCH_TILE_C < nin) ? i0 + BENCH_TILE_C : nin;
      for (size_t j = j0; j < j1; j++) {
        scalar_t s = 0;
        const scalar_t *Wrow = W + j * nin + i0;
        for (size_t i = 0; i < i1 - i0; i++)
          s += x[i0 + i] * Wrow[i];
        y[j] += s;
      }
    }
  }
}

static void bench_tiled_matmul(void) {
  BENCH_HEADER("tiled lin_fwd (naive vs tiled)");
  printf("\n");

  struct {
    size_t nout, nin;
    const char *label;
  } sizes[] = {
      {32, 32, "32x32 (sub-tile)"},
      {128, 128, "128x128 (N_EMBD)"},
      {512, 512, "512x512 (future)"},
      {73, 97, "73x97 (non-aligned)"},
  };
  int nsizes = 4;

  for (int s = 0; s < nsizes; s++) {
    size_t nout = sizes[s].nout, nin = sizes[s].nin;
    scalar_t *W = (scalar_t *)malloc(nout * nin * sizeof(scalar_t));
    scalar_t *x = (scalar_t *)malloc(nin * sizeof(scalar_t));
    scalar_t *y = (scalar_t *)calloc(nout, sizeof(scalar_t));

    srand(42);
    for (size_t i = 0; i < nout * nin; i++)
      W[i] = bench_randf();
    for (size_t i = 0; i < nin; i++)
      x[i] = bench_randf();

    /* Warm-up */
    bench_naive_lin_fwd(x, W, nin, nout, y);
    bench_tiled_lin_fwd(x, W, nin, nout, y);

    int iters = (nout <= 128) ? 100000 : 10000;

    clock_t t0 = clock();
    for (int i = 0; i < iters; i++)
      bench_naive_lin_fwd(x, W, nin, nout, y);
    scalar_t ms_naive = elapsed_ms(t0);

    t0 = clock();
    for (int i = 0; i < iters; i++)
      bench_tiled_lin_fwd(x, W, nin, nout, y);
    scalar_t ms_tiled = elapsed_ms(t0);

    scalar_t speedup = ms_naive / ms_tiled;
    printf("    %-30s naive=%6.2f ms  tiled=%6.2f ms  speedup=%.2fx\n",
           sizes[s].label, ms_naive / iters * 1000.0, ms_tiled / iters * 1000.0,
           speedup);

    free(W);
    free(x);
    free(y);
  }
}

/* ==================================================================== */
/*                            MAIN                                       */
/* ==================================================================== */

int main(void) {
  g_cfg = microgpt_default_config();

  printf("\n=== MicroGPT-C Benchmarks ===\n");
  printf("n_embd=%d  n_layer=%d  block_size=%d  n_head=%d\n\n", g_cfg.n_embd,
         g_cfg.n_layer, g_cfg.block_size, g_cfg.n_head);

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

  printf("\n[Sequence Operations]\n");
  bench_inference_sequence();
  bench_multi_position_fwd_bwd();

  printf("\n[Scaling Analysis]\n");
  bench_model_create_scaling();
  bench_memory_footprint();

  printf("\n[Convergence]\n");
  bench_convergence_speed();

  printf("\n[Tiled Linear Algebra]\n");
  bench_tiled_matmul();

  printf("\n=== Done ===\n\n");
  return 0;
}
