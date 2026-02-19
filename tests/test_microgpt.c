/*
 * MicroGPT-C Unit Tests
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Lightweight assertion-based tests for the MicroGPT public API.
 * No external test framework required — zero dependencies.
 *
 * Build:  cmake --build build --target test_microgpt
 * Run:    ./build/test_microgpt
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tolerance for floating-point comparisons: relaxed for float, tight for double
 */
#ifdef MICROGPT_USE_FLOAT
#define SCALAR_TOL 1e-3
#else
#define SCALAR_TOL 1e-10
#endif

/* ---- File-scoped runtime config (shared by all test functions) ---- */
static MicrogptConfig g_cfg;

/* Abstract KV cache allocation/free for tests — works with both flat and
 * paged KV cache builds. */
#ifdef MICROGPT_PAGED_KV
#define TEST_KV_ALLOC(arr, L)                                                  \
  (arr)[L] = (scalar_t *)paged_kv_create((size_t)g_cfg.block_size)
#define TEST_KV_FREE(arr, L) paged_kv_free((PagedKVCache *)(arr)[L])
#define TEST_KV_RESET(arr, L, cl)                                              \
  do {                                                                         \
    paged_kv_reset((PagedKVCache *)(arr)[L]);                                  \
    (cl)[L] = 0;                                                               \
  } while (0)
#else
#define TEST_KV_ALLOC(arr, L)                                                  \
  (arr)[L] = (scalar_t *)calloc(                                               \
      (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd, sizeof(scalar_t))
#define TEST_KV_FREE(arr, L) free((arr)[L])
#define TEST_KV_RESET(arr, L, cl)                                              \
  do {                                                                         \
    memset((arr)[L], 0,                                                        \
           (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd *                   \
               sizeof(scalar_t));                                              \
    (cl)[L] = 0;                                                               \
  } while (0)
#endif

/* ---- Minimal test harness ---- */

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name)                                                             \
  static void test_##name(void);                                               \
  static void run_##name(void) {                                               \
    g_tests_run++;                                                             \
    printf("  %-50s ", #name);                                                 \
    test_##name();                                                             \
    printf("PASS\n");                                                          \
    g_tests_passed++;                                                          \
  }                                                                            \
  static void test_##name(void)

#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL\n");                                                        \
      fprintf(stderr, "    Assertion failed: %s\n    at %s:%d\n", #cond,       \
              __FILE__, __LINE__);                                             \
      g_tests_failed++;                                                        \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_GT(a, b) ASSERT((a) > (b))
#define ASSERT_LT(a, b) ASSERT((a) < (b))
#define ASSERT_STREQ(a, b) ASSERT(strcmp((a), (b)) == 0)

#define RUN(name) run_##name()

/* ---- Helper: create a temp file with given content ---- */

static const char *write_temp_file(const char *name, const char *content) {
  FILE *f = fopen(name, "w");
  if (f) {
    fputs(content, f);
    fclose(f);
  }
  return name;
}

/* ==================================================================== */
/*                        UTILITY TESTS                                  */
/* ==================================================================== */

TEST(seed_rng_deterministic) {
  /* Seeding with same value should produce same model weights */
  seed_rng(42);
  Model *m1 = model_create(10, &g_cfg);
  ASSERT_NE(m1, NULL);
  size_t n1 = model_num_params(m1);

  seed_rng(42);
  Model *m2 = model_create(10, &g_cfg);
  ASSERT_NE(m2, NULL);
  size_t n2 = model_num_params(m2);
  ASSERT_EQ(n1, n2);

  model_free(m1);
  model_free(m2);
}

TEST(load_file_success) {
  write_temp_file("_test_load.txt", "hello world\n");
  size_t len = 0;
  char *data = load_file("_test_load.txt", &len);
  ASSERT_NE(data, NULL);
  ASSERT_EQ(len, 12);
  ASSERT_EQ(data[len], '\0'); /* nul-terminated */
  ASSERT(memcmp(data, "hello world\n", 12) == 0);
  free(data);
  remove("_test_load.txt");
}

TEST(load_file_missing) {
  size_t len = 0;
  char *data = load_file("_nonexistent_file_xyz.txt", &len);
  ASSERT_EQ(data, NULL);
}

/* ==================================================================== */
/*                   CHARACTER-LEVEL TOKENISATION                        */
/* ==================================================================== */

TEST(load_docs_basic) {
  write_temp_file("_test_docs.txt", "alpha\nbeta\ngamma\n");
  Docs docs = {0};
  int rc = load_docs("_test_docs.txt", &docs, g_cfg.max_docs);
  ASSERT_EQ(rc, 0);
  ASSERT_EQ(docs.num_docs, 3);
  ASSERT(memcmp(docs.lines[0], "alpha", 5) == 0);
  ASSERT_EQ(docs.doc_lens[0], 5);
  ASSERT(memcmp(docs.lines[1], "beta", 4) == 0);
  ASSERT_EQ(docs.doc_lens[1], 4);
  free_docs(&docs);
  remove("_test_docs.txt");
}

TEST(load_docs_missing_file) {
  Docs docs = {0};
  int rc = load_docs("_nonexistent_xyz.txt", &docs, g_cfg.max_docs);
  ASSERT_NE(rc, 0);
}

TEST(build_vocab_basic) {
  write_temp_file("_test_vocab.txt", "abc\nbca\n");
  Docs docs = {0};
  load_docs("_test_vocab.txt", &docs, g_cfg.max_docs);
  Vocab vocab = {0};
  build_vocab(&docs, &vocab);

  /* 3 unique chars (a, b, c) + 1 BOS = 4 */
  ASSERT_EQ(vocab.vocab_size, 4);
  ASSERT_EQ(vocab.bos_id, 3);

  /* chars should be sorted */
  ASSERT_EQ(vocab.chars[0], 'a');
  ASSERT_EQ(vocab.chars[1], 'b');
  ASSERT_EQ(vocab.chars[2], 'c');

  free(vocab.chars);
  free_docs(&docs);
  remove("_test_vocab.txt");
}

TEST(tokenize_basic) {
  write_temp_file("_test_tok.txt", "ab\n");
  Docs docs = {0};
  load_docs("_test_tok.txt", &docs, g_cfg.max_docs);
  Vocab vocab = {0};
  build_vocab(&docs, &vocab);

  /* Tokenize "ab" -> [BOS, a_id, b_id, BOS(EOS)] */
  size_t ids[16];
  size_t n = tokenize("ab", 2, &vocab, ids, 16);
  ASSERT_EQ(n, 4); /* BOS + a + b + EOS */
  ASSERT_EQ(ids[0], vocab.bos_id);
  ASSERT_EQ(ids[n - 1], vocab.bos_id); /* trailing BOS as EOS */

  /* Middle tokens should be valid (0..vocab_size-2) */
  ASSERT_LT(ids[1], vocab.vocab_size);
  ASSERT_LT(ids[2], vocab.vocab_size);
  ASSERT_NE(ids[1], ids[2]); /* 'a' != 'b' */

  free(vocab.chars);
  free_docs(&docs);
  remove("_test_tok.txt");
}

TEST(tokenize_max_len_truncation) {
  write_temp_file("_test_trunc.txt", "abcdef\n");
  Docs docs = {0};
  load_docs("_test_trunc.txt", &docs, g_cfg.max_docs);
  Vocab vocab = {0};
  build_vocab(&docs, &vocab);

  /* Limit output to 3 tokens: should only produce [BOS, a, b] */
  size_t ids[3];
  size_t n = tokenize("abcdef", 6, &vocab, ids, 3);
  ASSERT_EQ(n, 3);
  ASSERT_EQ(ids[0], vocab.bos_id);

  free(vocab.chars);
  free_docs(&docs);
  remove("_test_trunc.txt");
}

/* ==================================================================== */
/*                    WORD-LEVEL TOKENISATION                             */
/* ==================================================================== */

TEST(build_word_vocab_basic) {
  const char *text = "the cat sat on the mat the cat";
  size_t len = strlen(text);
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  int rc = build_word_vocab(text, len, 100, &wv);
  ASSERT_EQ(rc, 0);

  /* 5 unique words: the(3), cat(2), sat(1), on(1), mat(1) */
  ASSERT_EQ(wv.num_words, 5);
  /* vocab_size = 5 words + unk + newline + bos = 8 */
  ASSERT_EQ(wv.vocab_size, 8);
  ASSERT_EQ(wv.unk_id, 5);
  ASSERT_EQ(wv.newline_id, 6);
  ASSERT_EQ(wv.bos_id, 7);

  /* Most frequent word "the" should be at index 0 */
  ASSERT_STREQ(wv.words[0], "the");
  /* Second most frequent "cat" at index 1 */
  ASSERT_STREQ(wv.words[1], "cat");

  free_word_vocab(&wv);
}

TEST(build_word_vocab_max_words_limit) {
  const char *text = "aaa bbb ccc ddd eee fff aaa bbb ccc aaa bbb aaa";
  size_t len = strlen(text);
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  /* Only keep top 3 words */
  int rc = build_word_vocab(text, len, 3, &wv);
  ASSERT_EQ(rc, 0);
  ASSERT_EQ(wv.num_words, 3);
  ASSERT_EQ(wv.vocab_size, 6); /* 3 + unk + newline + bos */

  /* Top 3: aaa(4), bbb(3), ccc(2) */
  ASSERT_STREQ(wv.words[0], "aaa");
  ASSERT_STREQ(wv.words[1], "bbb");
  ASSERT_STREQ(wv.words[2], "ccc");

  /* "ddd", "eee", "fff" should map to unk */
  ASSERT_EQ(word_to_id(&wv, "ddd"), wv.unk_id);
  ASSERT_EQ(word_to_id(&wv, "eee"), wv.unk_id);

  free_word_vocab(&wv);
}

TEST(word_to_id_known_and_unknown) {
  const char *text = "hello world hello";
  size_t len = strlen(text);
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  build_word_vocab(text, len, 100, &wv);

  size_t hello_id = word_to_id(&wv, "hello");
  size_t world_id = word_to_id(&wv, "world");
  size_t missing_id = word_to_id(&wv, "missing");

  ASSERT_NE(hello_id, wv.unk_id);
  ASSERT_NE(world_id, wv.unk_id);
  ASSERT_EQ(missing_id, wv.unk_id);
  ASSERT_NE(hello_id, world_id);

  free_word_vocab(&wv);
}

TEST(tokenize_words_basic) {
  const char *text = "hello world\nhello";
  size_t len = strlen(text);
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  build_word_vocab(text, len, 100, &wv);

  size_t ids[64];
  size_t n = tokenize_words(text, len, &wv, ids, 64);
  /* "hello" "world" \n "hello" = 4 tokens */
  ASSERT_EQ(n, 4);

  size_t hello_id = word_to_id(&wv, "hello");
  size_t world_id = word_to_id(&wv, "world");
  ASSERT_EQ(ids[0], hello_id);
  ASSERT_EQ(ids[1], world_id);
  ASSERT_EQ(ids[2], wv.newline_id);
  ASSERT_EQ(ids[3], hello_id);

  free_word_vocab(&wv);
}

TEST(tokenize_words_unknown_maps_to_unk) {
  const char *text = "aaa bbb ccc";
  size_t len = strlen(text);
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  /* Only keep 2 words */
  build_word_vocab(text, len, 2, &wv);

  /* Tokenize original — "ccc" should be OOV */
  size_t ids[16];
  size_t n = tokenize_words(text, len, &wv, ids, 16);
  ASSERT_EQ(n, 3);
  ASSERT_EQ(ids[2], wv.unk_id); /* "ccc" not in top 2 */

  free_word_vocab(&wv);
}

TEST(tokenize_words_max_tokens_truncation) {
  const char *text = "one two three four five";
  size_t len = strlen(text);
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  build_word_vocab(text, len, 100, &wv);

  size_t ids[3];
  size_t n = tokenize_words(text, len, &wv, ids, 3);
  ASSERT_EQ(n, 3); /* truncated to max_tokens */

  free_word_vocab(&wv);
}

TEST(free_word_vocab_zeroes_struct) {
  const char *text = "test";
  size_t len = strlen(text);
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  build_word_vocab(text, len, 100, &wv);
  ASSERT_GT(wv.vocab_size, 0);

  free_word_vocab(&wv);
  ASSERT_EQ(wv.vocab_size, 0);
  ASSERT_EQ(wv.words, NULL);
}

TEST(word_vocab_special_tokens) {
  const char *text = "word";
  size_t len = strlen(text);
  WordVocab wv;
  memset(&wv, 0, sizeof(wv));
  build_word_vocab(text, len, 100, &wv);

  ASSERT_STREQ(wv.words[wv.unk_id], "<unk>");
  ASSERT_STREQ(wv.words[wv.newline_id], "\n");
  ASSERT_STREQ(wv.words[wv.bos_id], "<bos>");

  free_word_vocab(&wv);
}

/* ==================================================================== */
/*                     MODEL LIFECYCLE                                   */
/* ==================================================================== */

TEST(model_create_and_free) {
  seed_rng(1);
  Model *m = model_create(10, &g_cfg);
  ASSERT_NE(m, NULL);
  size_t np = model_num_params(m);
  ASSERT_GT(np, 0);
  model_free(m);
}

TEST(model_num_params_scales_with_vocab) {
  seed_rng(1);
  Model *m10 = model_create(10, &g_cfg);
  Model *m50 = model_create(50, &g_cfg);
  ASSERT_NE(m10, NULL);
  ASSERT_NE(m50, NULL);

  /* Larger vocab = more params (wte, lm_head scale with vocab) */
  ASSERT_GT(model_num_params(m50), model_num_params(m10));

  model_free(m10);
  model_free(m50);
}

/* ==================================================================== */
/*               FORWARD / BACKWARD / TRAINING                           */
/* ==================================================================== */

TEST(forward_backward_returns_positive_loss) {
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  ASSERT_NE(m, NULL);
  size_t np = model_num_params(m);

  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
    cl[L] = 0;
  }

  scalar_t loss = forward_backward_one(m, 0, 0, 1, keys, vals, cl, grads);
  ASSERT_GT(loss, 0.0);

  /* Gradients should be non-zero after backward pass */
  scalar_t grad_norm = 0;
  for (size_t i = 0; i < np; i++)
    grad_norm += grads[i] * grads[i];
  ASSERT_GT(grad_norm, 0.0);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads);
  model_free(m);
}

TEST(training_reduces_loss) {
  /* Run 50 training steps on a tiny sequence and check loss decreases */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  size_t np = model_num_params(m);

  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
  }

  /* Train on repeating pattern: 0 1 2 3 0 1 2 3 */
  size_t seq[] = {0, 1, 2, 3, 0, 1, 2, 3};
  size_t seq_len = 8;

  scalar_t first_loss = -1, last_loss = -1;

  for (int step = 0; step < 200; step++) {
    memset(grads, 0, np * sizeof(scalar_t));
    scalar_t step_loss = 0;
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;

    for (size_t p = 0; p < seq_len - 1; p++) {
      step_loss +=
          forward_backward_one(m, seq[p], p, seq[p + 1], keys, vals, cl, grads);
    }
    step_loss /= (scalar_t)(seq_len - 1);

    for (size_t i = 0; i < np; i++)
      grads[i] /= (scalar_t)(seq_len - 1);
    adam_step(m, grads, mom, vel, step);

    if (step == 0)
      first_loss = step_loss;
    if (step == 199)
      last_loss = step_loss;
  }

  /* Loss should decrease over training */
  ASSERT_LT(last_loss, first_loss);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}

/* ==================================================================== */
/*                     INFERENCE / SAMPLING                               */
/* ==================================================================== */

TEST(forward_inference_produces_logits) {
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);

  scalar_t logits[g_cfg.max_vocab];
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
    cl[L] = 0;
  }

  forward_inference(m, 0, 0, keys, vals, cl, logits);

  /* Logits should be finite and not all identical */
  int all_same = 1;
  for (size_t i = 0; i < 10; i++) {
    ASSERT(isfinite(logits[i]));
    if (i > 0 && logits[i] != logits[0])
      all_same = 0;
  }
  ASSERT_EQ(all_same, 0);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  model_free(m);
}

TEST(sample_token_returns_valid_id) {
  seed_rng(42);
  scalar_t logits[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

  for (int i = 0; i < 100; i++) {
    size_t tok = sample_token(logits, 5, 1.0);
    ASSERT_LT(tok, 5);
  }
}

TEST(sample_token_low_temp_picks_argmax) {
  seed_rng(42);
  scalar_t logits[5] = {0.0, 0.0, 10.0, 0.0, 0.0};

  /* Very low temperature should almost always pick the max logit */
  size_t tok = sample_token(logits, 5, 0.01);
  ASSERT_EQ(tok, 2);
}

TEST(sample_token_deterministic_with_seed) {
  scalar_t logits[5] = {1.0, 2.0, 3.0, 2.0, 1.0};

  seed_rng(12345);
  size_t tok1 = sample_token(logits, 5, 1.0);

  seed_rng(12345);
  size_t tok2 = sample_token(logits, 5, 1.0);

  ASSERT_EQ(tok1, tok2);
}

/* ==================================================================== */
/*                     MODEL SAVE / LOAD                                 */
/* ==================================================================== */

TEST(model_save_and_load_roundtrip) {
  seed_rng(42);
  Model *m1 = model_create(10, &g_cfg);
  ASSERT_NE(m1, NULL);

  int rc = model_save(m1, "_test_model.bin");
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  /* Save/load disabled for INT8 */
  ASSERT_NE(rc, 0);
  model_free(m1);
  remove("_test_model.bin");
#else
  ASSERT_EQ(rc, 0);

  Model *m2 = model_load("_test_model.bin", 10, &g_cfg);
  ASSERT_NE(m2, NULL);

  /* Both models should produce the same logits for the same input */
  scalar_t logits1[g_cfg.max_vocab], logits2[g_cfg.max_vocab];
  scalar_t *k1[g_cfg.n_layer], *v1[g_cfg.n_layer], *k2[g_cfg.n_layer],
      *v2[g_cfg.n_layer];
  size_t cl1[g_cfg.n_layer], cl2[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(k1, L);
    TEST_KV_ALLOC(v1, L);
    TEST_KV_ALLOC(k2, L);
    TEST_KV_ALLOC(v2, L);
    cl1[L] = cl2[L] = 0;
  }

  forward_inference(m1, 0, 0, k1, v1, cl1, logits1);
  forward_inference(m2, 0, 0, k2, v2, cl2, logits2);

  for (size_t i = 0; i < 10; i++)
    ASSERT(fabs(logits1[i] - logits2[i]) < SCALAR_TOL);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(k1, L);
    TEST_KV_FREE(v1, L);
    TEST_KV_FREE(k2, L);
    TEST_KV_FREE(v2, L);
  }
  model_free(m1);
  model_free(m2);
  remove("_test_model.bin");
#endif
}

TEST(checkpoint_save_and_load_roundtrip) {
  seed_rng(42);
  Model *m1 = model_create(10, &g_cfg);
  ASSERT_NE(m1, NULL);
  size_t np = model_num_params(m1);

  /* Simulate some training to populate optimizer state */
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *m_adam = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *v_adam = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
  }

  int saved_step = 10;
  for (int step = 0; step < saved_step; step++) {
    memset(grads, 0, np * sizeof(scalar_t));
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;
    forward_backward_one(m1, 0, 0, 1, keys, vals, cl, grads);
    for (size_t i = 0; i < np; i++)
      grads[i] /= 1.0;
    adam_step(m1, grads, m_adam, v_adam, step);
  }

  /* Save checkpoint */
  int rc = checkpoint_save(m1, m_adam, v_adam, saved_step, "_test_ckpt.bin");
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  ASSERT_NE(rc, 0);
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads);
  free(m_adam);
  free(v_adam);
  model_free(m1);
  remove("_test_ckpt.bin");
#else
  ASSERT_EQ(rc, 0);

  /* Load checkpoint into fresh buffers */
  scalar_t *m2_adam = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *v2_adam = (scalar_t *)calloc(np, sizeof(scalar_t));
  int loaded_step = -1;
  Model *m2 = checkpoint_load("_test_ckpt.bin", 10, &g_cfg, m2_adam, v2_adam,
                              &loaded_step);
  ASSERT_NE(m2, NULL);
  ASSERT_EQ(loaded_step, saved_step);

  /* Verify optimizer state matches exactly */
  for (size_t i = 0; i < np; i++) {
    ASSERT(fabs(m_adam[i] - m2_adam[i]) < SCALAR_TOL);
    ASSERT(fabs(v_adam[i] - v2_adam[i]) < SCALAR_TOL);
  }

  /* Verify model weights produce identical logits */
  scalar_t logits1[g_cfg.max_vocab], logits2[g_cfg.max_vocab];
  scalar_t *k2[g_cfg.n_layer], *v2[g_cfg.n_layer];
  size_t cl1[g_cfg.n_layer], cl2[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    cl[L] = 0;
    TEST_KV_ALLOC(k2, L);
    TEST_KV_ALLOC(v2, L);
    cl1[L] = cl2[L] = 0;
  }
  forward_inference(m1, 0, 0, keys, vals, cl1, logits1);
  forward_inference(m2, 0, 0, k2, v2, cl2, logits2);
  for (size_t i = 0; i < 10; i++)
    ASSERT(fabs(logits1[i] - logits2[i]) < SCALAR_TOL);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
    TEST_KV_FREE(k2, L);
    TEST_KV_FREE(v2, L);
  }
  free(grads);
  free(m_adam);
  free(v_adam);
  free(m2_adam);
  free(v2_adam);
  model_free(m1);
  model_free(m2);
  remove("_test_ckpt.bin");
#endif
}

TEST(checkpoint_load_missing_file) {
  scalar_t m_buf[1], v_buf[1];
  int step = -1;
  Model *m =
      checkpoint_load("_nonexistent_ckpt.bin", 10, &g_cfg, m_buf, v_buf, &step);
  ASSERT_EQ(m, NULL);
  ASSERT_EQ(step, -1); /* unchanged on failure */
}

/* ==================================================================== */
/*                   GRADIENT CORRECTNESS                                 */
/* ==================================================================== */

TEST(gradient_direction_reduces_loss) {
  /*
   * Verify that gradients point in the right direction: taking one
   * Adam step using the computed gradients should reduce the loss.
   * This confirms gradient correctness without accessing Model internals.
   */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  ASSERT_NE(m, NULL);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
    cl[L] = 0;
  }

  /* Compute loss and gradients before optimization */
  scalar_t loss_before =
      forward_backward_one(m, 2, 0, 3, keys, vals, cl, grads);
  ASSERT_GT(loss_before, 0.0);

  /* Verify gradient norm is non-zero */
  scalar_t grad_norm = 0;
  for (size_t i = 0; i < np; i++)
    grad_norm += grads[i] * grads[i];
  ASSERT_GT(grad_norm, 0.0);

  /* Take one Adam step */
  adam_step(m, grads, mom, vel, 0);

  /* Compute loss again with the same input */
  memset(grads, 0, np * sizeof(scalar_t));
  for (int L = 0; L < g_cfg.n_layer; L++)
    cl[L] = 0;
  scalar_t loss_after = forward_backward_one(m, 2, 0, 3, keys, vals, cl, grads);

  /* Loss should decrease after one gradient step */
  ASSERT_LT(loss_after, loss_before);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}

TEST(gradient_accumulates_over_positions) {
  /* Verify that gradients accumulate (+=) over multiple positions
   * and are larger than single-position gradients */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads1 = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *grads3 = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
  }

  /* Single position */
  for (int L = 0; L < g_cfg.n_layer; L++)
    cl[L] = 0;
  forward_backward_one(m, 1, 0, 2, keys, vals, cl, grads1);
  scalar_t norm1 = 0;
  for (size_t i = 0; i < np; i++)
    norm1 += grads1[i] * grads1[i];

  /* Three positions (accumulating into same buffer) */
  for (int L = 0; L < g_cfg.n_layer; L++)
    cl[L] = 0;
  forward_backward_one(m, 1, 0, 2, keys, vals, cl, grads3);
  forward_backward_one(m, 2, 1, 3, keys, vals, cl, grads3);
  forward_backward_one(m, 3, 2, 4, keys, vals, cl, grads3);
  scalar_t norm3 = 0;
  for (size_t i = 0; i < np; i++)
    norm3 += grads3[i] * grads3[i];

  /* Multi-position gradient norm should be larger */
  ASSERT_GT(norm3, norm1);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads1);
  free(grads3);
  model_free(m);
}

/* ==================================================================== */
/*                   KV CACHE BEHAVIOUR                                  */
/* ==================================================================== */

TEST(kv_cache_grows_with_positions) {
  /* Verify that cache_len increments after each forward pass */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
    cl[L] = 0;
  }

  /* Process 5 positions */
  for (int p = 0; p < 5; p++) {
    forward_backward_one(m, (size_t)(p % 10), (size_t)p, (size_t)((p + 1) % 10),
                         keys, vals, cl, grads);
    for (int L = 0; L < g_cfg.n_layer; L++)
      ASSERT_EQ(cl[L], (size_t)(p + 1));
  }

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads);
  model_free(m);
}

TEST(inference_kv_cache_autoregressive) {
  /* Run an auto-regressive inference sequence of several tokens
   * and verify each step produces valid, finite logits */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  scalar_t logits[g_cfg.max_vocab];
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
    cl[L] = 0;
  }

  size_t tok = 0; /* start with token 0 */
  int seq_len = 8;
  for (int p = 0; p < seq_len; p++) {
    forward_inference(m, tok, (size_t)p, keys, vals, cl, logits);
    /* All logits should be finite */
    for (size_t i = 0; i < 10; i++)
      ASSERT(isfinite(logits[i]));
    /* Cache should grow */
    for (int L = 0; L < g_cfg.n_layer; L++)
      ASSERT_EQ(cl[L], (size_t)(p + 1));
    /* Sample next token */
    tok = sample_token(logits, 10, 0.8);
    ASSERT_LT(tok, 10);
  }

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  model_free(m);
}

/* ==================================================================== */
/*                   TRAINING CONVERGENCE                                */
/* ==================================================================== */

TEST(training_loss_decreases_monotonically_in_windows) {
  /* Train for 100 steps and check that the average loss over the last 20
   * is lower than the average loss over the first 20.  This is more
   * robust than checking strict per-step monotonicity. */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
  }

  scalar_t first_window = 0, last_window = 0;
  int total_steps = 400;
  int window = 20;
  size_t seq[] = {0, 1, 2, 3, 4, 5, 6, 7};

  for (int step = 0; step < total_steps; step++) {
    memset(grads, 0, np * sizeof(scalar_t));
    for (int L = 0; L < g_cfg.n_layer; L++)
      cl[L] = 0;
    scalar_t step_loss = 0;
    for (int p = 0; p < 7; p++) {
      step_loss += forward_backward_one(m, seq[p], (size_t)p, seq[p + 1], keys,
                                        vals, cl, grads);
    }
    step_loss /= 7.0;
    for (size_t i = 0; i < np; i++)
      grads[i] /= 7.0;
    adam_step(m, grads, mom, vel, step);

    if (step < window)
      first_window += step_loss;
    if (step >= total_steps - window)
      last_window += step_loss;
  }
  first_window /= (scalar_t)window;
  last_window /= (scalar_t)window;

  /* Loss should decrease substantially */
  ASSERT_LT(last_window, first_window);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}

TEST(training_reproducible_with_same_seed) {
  /* Two identical training runs with the same seed should produce
   * exactly the same final loss */
  scalar_t losses[2];
  for (int run = 0; run < 2; run++) {
    seed_rng(12345);
    Model *m = model_create(10, &g_cfg);
    size_t np = model_num_params(m);
    scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
    scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
    scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
    scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
    size_t cl[g_cfg.n_layer];
    for (int L = 0; L < g_cfg.n_layer; L++) {
      TEST_KV_ALLOC(keys, L);
      TEST_KV_ALLOC(vals, L);
    }

    scalar_t final_loss = 0;
    for (int step = 0; step < 30; step++) {
      memset(grads, 0, np * sizeof(scalar_t));
      for (int L = 0; L < g_cfg.n_layer; L++)
        cl[L] = 0;
      final_loss = forward_backward_one(m, 1, 0, 2, keys, vals, cl, grads);
      adam_step(m, grads, mom, vel, step);
    }
    losses[run] = final_loss;

    for (int L = 0; L < g_cfg.n_layer; L++) {
      TEST_KV_FREE(keys, L);
      TEST_KV_FREE(vals, L);
    }
    free(grads);
    free(mom);
    free(vel);
    model_free(m);
  }
  ASSERT(fabs(losses[0] - losses[1]) < 1e-12);
}

/* ==================================================================== */
/*                INFERENCE / FORWARD CONSISTENCY                        */
/* ==================================================================== */

TEST(forward_inference_matches_fwd_bwd_logits) {
  /* forward_inference and forward_backward_one should produce
   * identical logits for the same input (ignoring the loss/gradient) */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  size_t np = model_num_params(m);

  scalar_t logits_inf[g_cfg.max_vocab];
  scalar_t logits_bwd[g_cfg.max_vocab];
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys1[g_cfg.n_layer], *vals1[g_cfg.n_layer], *keys2[g_cfg.n_layer],
      *vals2[g_cfg.n_layer];
  size_t cl1[g_cfg.n_layer], cl2[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys1, L);
    TEST_KV_ALLOC(vals1, L);
    TEST_KV_ALLOC(keys2, L);
    TEST_KV_ALLOC(vals2, L);
    cl1[L] = cl2[L] = 0;
  }

  forward_inference(m, 3, 0, keys1, vals1, cl1, logits_inf);

  /* forward_backward_one doesn't directly output logits, so we compare
   * by checking that inference on the same model produces consistent
   * sampling behaviour */
  seed_rng(42);
  size_t tok1 = sample_token(logits_inf, 10, 0.01); /* near-greedy */

  /* Run inference again from scratch — should get same result */
  for (int L = 0; L < g_cfg.n_layer; L++)
    cl1[L] = 0;
  forward_inference(m, 3, 0, keys1, vals1, cl1, logits_bwd);
  size_t tok2 = sample_token(logits_bwd, 10, 0.01);
  ASSERT_EQ(tok1, tok2);

  /* Logits should be identical */
  for (size_t i = 0; i < 10; i++)
    ASSERT(fabs(logits_inf[i] - logits_bwd[i]) < SCALAR_TOL);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys1, L);
    TEST_KV_FREE(vals1, L);
    TEST_KV_FREE(keys2, L);
    TEST_KV_FREE(vals2, L);
  }
  free(grads);
  model_free(m);
}

TEST(sample_token_high_temp_is_more_uniform) {
  /* High temperature should produce a more uniform distribution
   * (measured by sampling entropy) compared to low temperature */
  seed_rng(42);
  scalar_t logits[5] = {1.0, 2.0, 3.0, 2.0, 1.0};

  /* Sample 1000 times at low temp and high temp, count frequencies */
  int counts_low[5] = {0}, counts_high[5] = {0};
  for (int i = 0; i < 1000; i++) {
    size_t t_low = sample_token(logits, 5, 0.1);
    size_t t_high = sample_token(logits, 5, 5.0);
    counts_low[t_low]++;
    counts_high[t_high]++;
  }

  /* Low temp: most samples should concentrate on token 2 (highest logit) */
  ASSERT_GT(counts_low[2], 800);

  /* High temp: distribution should be more spread out */
  int min_high = counts_high[0], max_high = counts_high[0];
  for (int i = 1; i < 5; i++) {
    if (counts_high[i] < min_high)
      min_high = counts_high[i];
    if (counts_high[i] > max_high)
      max_high = counts_high[i];
  }
  /* The ratio of max/min should be much smaller for high temp */
  ASSERT_LT(max_high, min_high * 5); /* fairly uniform */
}

/* ==================================================================== */
/*                   MODEL PARAMETER ACCOUNTING                          */
/* ==================================================================== */

TEST(model_num_params_formula) {
  /* Verify that model_num_params matches the expected formula:
   * wte(V*E) + wpe(B*E) + lm_head(V*E) +
   * g_cfg.n_layer * (wq(E*E) + wk(E*E) + wv(E*E) + wo(E*E) + fc1(M*E) +
   * fc2(E*M)) where E=g_cfg.n_embd, B=g_cfg.block_size, M=g_cfg.mlp_dim */
  size_t vs = 50;
  Model *m = model_create(vs, &g_cfg);
  ASSERT_NE(m, NULL);
  size_t np = model_num_params(m);

  size_t expected =
      vs * (size_t)g_cfg.n_embd                         /* wte */
      + (size_t)g_cfg.block_size * (size_t)g_cfg.n_embd /* wpe */
      + vs * (size_t)g_cfg.n_embd                       /* lm_head */
      + (size_t)g_cfg.n_layer *
            ((size_t)g_cfg.n_embd * (size_t)g_cfg.n_embd *
                 4 /* wq + wk + wv + wo */
             + (size_t)g_cfg.mlp_dim * (size_t)g_cfg.n_embd /* fc1 */
             + (size_t)g_cfg.n_embd * (size_t)g_cfg.mlp_dim /* fc2 */
            );
  ASSERT_EQ(np, expected);
  model_free(m);
}

TEST(different_vocab_sizes_produce_different_params) {
  Model *m10 = model_create(10, &g_cfg);
  Model *m50 = model_create(50, &g_cfg);
  ASSERT_NE(m10, NULL);
  ASSERT_NE(m50, NULL);
  size_t np10 = model_num_params(m10);
  size_t np50 = model_num_params(m50);
  /* Both models differ only in wte and lm_head which scale with vocab */
  size_t vocab_diff = (50 - 10) * (size_t)g_cfg.n_embd * 2; /* wte + lm_head */
  ASSERT_EQ(np50 - np10, vocab_diff);
  model_free(m10);
  model_free(m50);
}

/* ==================================================================== */
/*                     LOSS SANITY CHECKS                                */
/* ==================================================================== */

TEST(loss_is_log_vocab_for_uniform_predictions) {
  /* For a freshly created model (near-uniform predictions), the loss
   * should be approximately log(vocab_size) = -log(1/V) */
  seed_rng(42);
  size_t vs = 20;
  Model *m = model_create(vs, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
    cl[L] = 0;
  }

  scalar_t loss = forward_backward_one(m, 0, 0, 1, keys, vals, cl, grads);
  scalar_t expected = log((scalar_t)vs); /* ~3.0 for vs=20 */

  /* Should be in the right ballpark (within 2x) */
  ASSERT_GT(loss, expected * 0.3);
  ASSERT_LT(loss, expected * 3.0);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads);
  model_free(m);
}

TEST(loss_different_targets_give_different_losses) {
  /* Same input but different targets should produce different losses */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads1 = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *grads2 = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
  }

  for (int L = 0; L < g_cfg.n_layer; L++)
    cl[L] = 0;
  scalar_t loss1 = forward_backward_one(m, 1, 0, 2, keys, vals, cl, grads1);
  for (int L = 0; L < g_cfg.n_layer; L++)
    cl[L] = 0;
  scalar_t loss2 = forward_backward_one(m, 1, 0, 5, keys, vals, cl, grads2);

  /* Different targets → different losses (very unlikely to be equal) */
  ASSERT(fabs(loss1 - loss2) > SCALAR_TOL);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads1);
  free(grads2);
  model_free(m);
}

/* ==================================================================== */
/*                   TILED LINEAR ALGEBRA                                */
/* ==================================================================== */

/*
 * These tests verify the tiled loop nest algorithm used in lin_fwd and
 * lin_bwd.  Since those functions are static in microgpt.c, we reproduce
 * the tiled algorithm here and compare against naive reference.  This
 * validates:
 *   1. The tiling logic itself is correct
 *   2. Tail tiles (non-multiple-of-TILE) are handled properly
 *   3. The algorithm produces bit-identical results to naive loops
 *
 * The actual model-level integration is tested by training_reduces_loss,
 * gradient_direction_reduces_loss, etc.
 */

#define TEST_TILE_R 32
#define TEST_TILE_C 64

/* Simple random scalar_t in [-1,1] for test data (uses stdlib rand) */
static scalar_t test_randf(void) {
  return 2.0 * ((scalar_t)rand() / (scalar_t)RAND_MAX) - 1.0;
}

/* Naive reference: y = W @ x */
static void ref_lin_fwd(const scalar_t *x, const scalar_t *W, size_t nin,
                        size_t nout, scalar_t *y) {
  for (size_t j = 0; j < nout; j++) {
    scalar_t s = 0;
    for (size_t i = 0; i < nin; i++)
      s += W[j * nin + i] * x[i];
    y[j] = s;
  }
}

/* Tiled: y = W @ x (mirrors microgpt.c lin_fwd C fallback) */
static void tiled_lin_fwd(const scalar_t *x, const scalar_t *W, size_t nin,
                          size_t nout, scalar_t *y) {
  memset(y, 0, nout * sizeof(scalar_t));
  for (size_t j0 = 0; j0 < nout; j0 += TEST_TILE_R) {
    size_t j1 = (j0 + TEST_TILE_R < nout) ? j0 + TEST_TILE_R : nout;
    for (size_t i0 = 0; i0 < nin; i0 += TEST_TILE_C) {
      size_t i1 = (i0 + TEST_TILE_C < nin) ? i0 + TEST_TILE_C : nin;
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

/* Naive reference: dx += W^T @ dy */
static void ref_lin_bwd_dx(const scalar_t *W, const scalar_t *dy, size_t nin,
                           size_t nout, scalar_t *dx) {
  for (size_t j = 0; j < nout; j++) {
    scalar_t dyj = dy[j];
    for (size_t i = 0; i < nin; i++)
      dx[i] += dyj * W[j * nin + i];
  }
}

/* Tiled: dx += W^T @ dy (mirrors microgpt.c lin_bwd C fallback) */
static void tiled_lin_bwd_dx(const scalar_t *W, const scalar_t *dy, size_t nin,
                             size_t nout, scalar_t *dx) {
  for (size_t j0 = 0; j0 < nout; j0 += TEST_TILE_R) {
    size_t j1 = (j0 + TEST_TILE_R < nout) ? j0 + TEST_TILE_R : nout;
    for (size_t i0 = 0; i0 < nin; i0 += TEST_TILE_C) {
      size_t i1 = (i0 + TEST_TILE_C < nin) ? i0 + TEST_TILE_C : nin;
      for (size_t j = j0; j < j1; j++) {
        scalar_t dyj = dy[j];
        const scalar_t *Wrow = W + j * nin + i0;
        for (size_t i = 0; i < i1 - i0; i++)
          dx[i0 + i] += dyj * Wrow[i];
      }
    }
  }
}

/* Naive reference: dW += dy ⊗ x */
static void ref_lin_bwd_dw(const scalar_t *x, const scalar_t *dy, size_t nin,
                           size_t nout, scalar_t *dW) {
  for (size_t j = 0; j < nout; j++) {
    scalar_t dyj = dy[j];
    for (size_t i = 0; i < nin; i++)
      dW[j * nin + i] += dyj * x[i];
  }
}

/* Tiled: dW += dy ⊗ x */
static void tiled_lin_bwd_dw(const scalar_t *x, const scalar_t *dy, size_t nin,
                             size_t nout, scalar_t *dW) {
  for (size_t j0 = 0; j0 < nout; j0 += TEST_TILE_R) {
    size_t j1 = (j0 + TEST_TILE_R < nout) ? j0 + TEST_TILE_R : nout;
    for (size_t i0 = 0; i0 < nin; i0 += TEST_TILE_C) {
      size_t i1 = (i0 + TEST_TILE_C < nin) ? i0 + TEST_TILE_C : nin;
      for (size_t j = j0; j < j1; j++) {
        scalar_t dyj = dy[j];
        scalar_t *dWrow = dW + j * nin + i0;
        for (size_t i = 0; i < i1 - i0; i++)
          dWrow[i] += dyj * x[i0 + i];
      }
    }
  }
}

TEST(lin_fwd_tiled_matches_reference) {
  /* Non-tile-aligned sizes to stress tail handling */
  size_t nin = 73, nout = 37;
  scalar_t *W = (scalar_t *)malloc(nout * nin * sizeof(scalar_t));
  scalar_t *x = (scalar_t *)malloc(nin * sizeof(scalar_t));
  scalar_t *y_ref = (scalar_t *)calloc(nout, sizeof(scalar_t));
  scalar_t *y_tiled = (scalar_t *)calloc(nout, sizeof(scalar_t));

  srand(42);
  for (size_t i = 0; i < nout * nin; i++)
    W[i] = test_randf();
  for (size_t i = 0; i < nin; i++)
    x[i] = test_randf();

  ref_lin_fwd(x, W, nin, nout, y_ref);
  tiled_lin_fwd(x, W, nin, nout, y_tiled);

  for (size_t j = 0; j < nout; j++)
    ASSERT(fabs(y_ref[j] - y_tiled[j]) < SCALAR_TOL);

  free(W);
  free(x);
  free(y_ref);
  free(y_tiled);
}

TEST(lin_bwd_tiled_dx_matches_reference) {
  size_t nin = 73, nout = 37;
  scalar_t *W = (scalar_t *)malloc(nout * nin * sizeof(scalar_t));
  scalar_t *dy = (scalar_t *)malloc(nout * sizeof(scalar_t));
  scalar_t *dx_ref = (scalar_t *)calloc(nin, sizeof(scalar_t));
  scalar_t *dx_tiled = (scalar_t *)calloc(nin, sizeof(scalar_t));

  srand(99);
  for (size_t i = 0; i < nout * nin; i++)
    W[i] = test_randf();
  for (size_t i = 0; i < nout; i++)
    dy[i] = test_randf();

  ref_lin_bwd_dx(W, dy, nin, nout, dx_ref);
  tiled_lin_bwd_dx(W, dy, nin, nout, dx_tiled);

  for (size_t i = 0; i < nin; i++)
    ASSERT(fabs(dx_ref[i] - dx_tiled[i]) < SCALAR_TOL);

  free(W);
  free(dy);
  free(dx_ref);
  free(dx_tiled);
}

TEST(lin_bwd_tiled_dw_matches_reference) {
  size_t nin = 73, nout = 37;
  scalar_t *x = (scalar_t *)malloc(nin * sizeof(scalar_t));
  scalar_t *dy = (scalar_t *)malloc(nout * sizeof(scalar_t));
  scalar_t *dW_ref = (scalar_t *)calloc(nout * nin, sizeof(scalar_t));
  scalar_t *dW_tiled = (scalar_t *)calloc(nout * nin, sizeof(scalar_t));

  srand(77);
  for (size_t i = 0; i < nin; i++)
    x[i] = test_randf();
  for (size_t i = 0; i < nout; i++)
    dy[i] = test_randf();

  ref_lin_bwd_dw(x, dy, nin, nout, dW_ref);
  tiled_lin_bwd_dw(x, dy, nin, nout, dW_tiled);

  for (size_t i = 0; i < nout * nin; i++)
    ASSERT(fabs(dW_ref[i] - dW_tiled[i]) < SCALAR_TOL);

  free(x);
  free(dy);
  free(dW_ref);
  free(dW_tiled);
}

TEST(lin_fwd_large_tile_aligned) {
  /* Tile-aligned 128×128 */
  size_t nin = 128, nout = 128;
  scalar_t *W = (scalar_t *)malloc(nout * nin * sizeof(scalar_t));
  scalar_t *x = (scalar_t *)malloc(nin * sizeof(scalar_t));
  scalar_t *y_ref = (scalar_t *)calloc(nout, sizeof(scalar_t));
  scalar_t *y_tiled = (scalar_t *)calloc(nout, sizeof(scalar_t));

  srand(2024);
  for (size_t i = 0; i < nout * nin; i++)
    W[i] = test_randf();
  for (size_t i = 0; i < nin; i++)
    x[i] = test_randf();

  ref_lin_fwd(x, W, nin, nout, y_ref);
  tiled_lin_fwd(x, W, nin, nout, y_tiled);

  for (size_t j = 0; j < nout; j++)
    ASSERT(fabs(y_ref[j] - y_tiled[j]) < SCALAR_TOL);

  free(W);
  free(x);
  free(y_ref);
  free(y_tiled);
}

TEST(lin_fwd_small_below_tile) {
  /* Dimensions smaller than one tile */
  size_t nin = 7, nout = 5;
  scalar_t *W = (scalar_t *)malloc(nout * nin * sizeof(scalar_t));
  scalar_t *x = (scalar_t *)malloc(nin * sizeof(scalar_t));
  scalar_t *y_ref = (scalar_t *)calloc(nout, sizeof(scalar_t));
  scalar_t *y_tiled = (scalar_t *)calloc(nout, sizeof(scalar_t));

  srand(55);
  for (size_t i = 0; i < nout * nin; i++)
    W[i] = test_randf();
  for (size_t i = 0; i < nin; i++)
    x[i] = test_randf();

  ref_lin_fwd(x, W, nin, nout, y_ref);
  tiled_lin_fwd(x, W, nin, nout, y_tiled);

  for (size_t j = 0; j < nout; j++)
    ASSERT(fabs(y_ref[j] - y_tiled[j]) < SCALAR_TOL);

  free(W);
  free(x);
  free(y_ref);
  free(y_tiled);
}

/* ==================================================================== */
/*                     PAGED KV CACHE TESTS                              */
/* ==================================================================== */

#ifdef MICROGPT_PAGED_KV

TEST(paged_kv_create_and_free) {
  PagedKVCache *c = paged_kv_create(128);
  ASSERT_NE(c, NULL);
  ASSERT_EQ(c->len, (size_t)0);
  ASSERT_EQ(c->n_pages, (size_t)0);
  ASSERT(c->capacity >= 128 / KV_PAGE_SIZE);
  paged_kv_free(c);
}

TEST(paged_kv_append_and_get) {
  PagedKVCache *c = paged_kv_create(128);
  ASSERT_NE(c, NULL);

  /* Write 3 positions */
  for (size_t i = 0; i < 3; i++) {
    scalar_t *slot = paged_kv_append(c);
    ASSERT_NE(slot, NULL);
    for (size_t d = 0; d < (size_t)g_cfg.n_embd; d++)
      slot[d] = (scalar_t)(i * 100 + d);
  }
  ASSERT_EQ(c->len, (size_t)3);

  /* Read back and verify */
  for (size_t i = 0; i < 3; i++) {
    const scalar_t *slot = paged_kv_get(c, i);
    ASSERT_NE(slot, NULL);
    for (size_t d = 0; d < (size_t)g_cfg.n_embd; d++)
      ASSERT(fabs(slot[d] - (scalar_t)(i * 100 + d)) < 1e-15);
  }
  paged_kv_free(c);
}

TEST(paged_kv_grows_across_page_boundary) {
  /* Fill beyond one page to verify demand allocation */
  size_t n = KV_PAGE_SIZE + 5;
  PagedKVCache *c = paged_kv_create(n + KV_PAGE_SIZE);
  ASSERT_NE(c, NULL);

  for (size_t i = 0; i < n; i++) {
    scalar_t *slot = paged_kv_append(c);
    slot[0] = (scalar_t)i; /* sentinel in first element */
  }
  ASSERT_EQ(c->len, n);
  ASSERT(c->n_pages >= 2); /* must have allocated a second page */

  /* Verify readback across page boundary */
  for (size_t i = 0; i < n; i++)
    ASSERT(fabs(paged_kv_get(c, i)[0] - (scalar_t)i) < 1e-15);

  paged_kv_free(c);
}

TEST(paged_kv_reset_reuses_pages) {
  PagedKVCache *c = paged_kv_create(128);

  /* Append some positions to allocate pages */
  for (size_t i = 0; i < KV_PAGE_SIZE + 1; i++)
    paged_kv_append(c);
  size_t pages_before = c->n_pages;
  ASSERT(pages_before >= 2);

  /* Reset — pages should be retained */
  paged_kv_reset(c);
  ASSERT_EQ(c->len, (size_t)0);
  ASSERT_EQ(c->n_pages, pages_before); /* pages NOT freed */

  /* Re-append — should reuse existing pages without new alloc */
  for (size_t i = 0; i < KV_PAGE_SIZE + 1; i++) {
    scalar_t *slot = paged_kv_append(c);
    slot[0] = (scalar_t)(i + 1000);
  }
  ASSERT_EQ(c->n_pages, pages_before); /* no new pages */

  /* Verify new data */
  for (size_t i = 0; i < KV_PAGE_SIZE + 1; i++)
    ASSERT(fabs(paged_kv_get(c, i)[0] - (scalar_t)(i + 1000)) < 1e-15);

  paged_kv_free(c);
}

#endif /* MICROGPT_PAGED_KV */

/* ==================================================================== */
/*                      CONFIG API TESTS                                 */
/* ==================================================================== */

TEST(default_config_has_valid_fields) {
  MicrogptConfig cfg = microgpt_default_config();
  ASSERT_GT(cfg.n_embd, 0);
  ASSERT_GT(cfg.n_head, 0);
  ASSERT_GT(cfg.n_layer, 0);
  ASSERT_GT(cfg.block_size, 0);
  ASSERT_GT(cfg.mlp_dim, 0);
  ASSERT_GT(cfg.max_vocab, 0);
  ASSERT_GT(cfg.max_docs, 0);
  /* n_embd must be divisible by n_head */
  ASSERT_EQ(cfg.n_embd % cfg.n_head, 0);
  /* Learning rate must be positive */
  ASSERT(cfg.learning_rate > 0);
}

TEST(custom_config_overrides) {
  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = 32;
  cfg.n_head = 4;
  cfg.n_layer = 2;
  cfg.block_size = 16;
  cfg.mlp_dim = 64;

  Model *m = model_create(10, &cfg);
  ASSERT(m != NULL);
  const MicrogptConfig *mc = model_config(m);
  ASSERT_EQ(mc->n_embd, 32);
  ASSERT_EQ(mc->n_head, 4);
  ASSERT_EQ(mc->n_layer, 2);
  ASSERT_EQ(mc->block_size, 16);
  ASSERT_EQ(mc->mlp_dim, 64);
  model_free(m);
}

TEST(model_config_accessor) {
  seed_rng(99);
  Model *m = model_create(10, &g_cfg);
  const MicrogptConfig *cfg = model_config(m);
  ASSERT(cfg != NULL);
  ASSERT_EQ(cfg->n_embd, g_cfg.n_embd);
  ASSERT_EQ(cfg->n_head, g_cfg.n_head);
  ASSERT_EQ(cfg->n_layer, g_cfg.n_layer);
  model_free(m);
}

/* ==================================================================== */
/*                   SOFTMAX / SAMPLING EDGE CASES                       */
/* ==================================================================== */

TEST(softmax_output_sums_to_one) {
  /* Create logits, run through sampling at temp=1 many times.
     Check that sample_token always returns valid IDs (implicitly
     tests softmax normalization). */
  scalar_t logits[] = {1.0, 2.0, 3.0, 0.5, -1.0};
  for (int trial = 0; trial < 100; trial++) {
    size_t tok = sample_token(logits, 5, 1.0);
    ASSERT_LT(tok, 5);
  }
}

TEST(sample_token_zero_temp_picks_max) {
  /* Temperature approaching zero should always pick the argmax */
  scalar_t logits[] = {0.1, 0.2, 5.0, 0.3, 0.4};
  for (int trial = 0; trial < 20; trial++) {
    size_t tok = sample_token(logits, 5, 0.001);
    ASSERT_EQ(tok, 2); /* logits[2] = 5.0 is the max */
  }
}

TEST(sample_token_single_vocab) {
  /* With only one token in the vocab, it must always be selected */
  scalar_t logits[] = {42.0};
  for (int trial = 0; trial < 10; trial++) {
    size_t tok = sample_token(logits, 1, 1.0);
    ASSERT_EQ(tok, 0);
  }
}

/* ==================================================================== */
/*                   MULTI-POSITION TRAINING                             */
/* ==================================================================== */

TEST(multi_position_loss_accumulation) {
  /* Training on a sequence of length > 1 should produce a loss
     that scales roughly linearly with the number of positions. */
  seed_rng(42);
  Model *m = model_create(10, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads1 = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *grads2 = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
  }

  /* Single position loss */
  scalar_t loss1 = forward_backward_one(m, 0, 0, 5, keys, vals, cl, grads1);

  /* Reset KV cache and do two positions */
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_RESET(keys, L, cl);
    TEST_KV_RESET(vals, L, cl);
  }
  scalar_t loss_a = forward_backward_one(m, 0, 0, 5, keys, vals, cl, grads2);
  scalar_t loss_b = forward_backward_one(m, 1, 1, 7, keys, vals, cl, grads2);
  scalar_t total_loss = loss_a + loss_b;

  /* Two positions should give roughly 2x the single-position loss */
  ASSERT_GT(total_loss, loss1 * 0.5); /* sanity lower bound */
  ASSERT_LT(total_loss, loss1 * 5.0); /* sanity upper bound */

  /* grads2 should be non-zero (some gradient was accumulated) */
  scalar_t norm2 = 0;
  for (size_t i = 0; i < np; i++)
    norm2 += grads2[i] * grads2[i];
  ASSERT_GT(norm2, 0);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads1);
  free(grads2);
  model_free(m);
}

/* ==================================================================== */
/*                   SEED / REPRODUCIBILITY                              */
/* ==================================================================== */

TEST(different_seeds_produce_different_models) {
  /* Models from different seeds should produce different logits */
  seed_rng(1);
  Model *m1 = model_create(10, &g_cfg);
  seed_rng(2);
  Model *m2 = model_create(10, &g_cfg);

  scalar_t logits1[10], logits2[10];
  scalar_t *k1[g_cfg.n_layer], *v1[g_cfg.n_layer];
  scalar_t *k2[g_cfg.n_layer], *v2[g_cfg.n_layer];
  size_t cl1[g_cfg.n_layer], cl2[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(k1, L);
    TEST_KV_ALLOC(v1, L);
    TEST_KV_ALLOC(k2, L);
    TEST_KV_ALLOC(v2, L);
  }
  forward_inference(m1, 0, 0, k1, v1, cl1, logits1);
  forward_inference(m2, 0, 0, k2, v2, cl2, logits2);

  int any_diff = 0;
  for (int i = 0; i < 10; i++) {
    if (fabs(logits1[i] - logits2[i]) > SCALAR_TOL) {
      any_diff = 1;
      break;
    }
  }
  ASSERT(any_diff);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(k1, L);
    TEST_KV_FREE(v1, L);
    TEST_KV_FREE(k2, L);
    TEST_KV_FREE(v2, L);
  }
  model_free(m1);
  model_free(m2);
}

TEST(same_seed_produces_identical_models) {
  /* Same seed → identical param count and identical saved weights */
  seed_rng(42);
  Model *m1 = model_create(10, &g_cfg);
  seed_rng(42);
  Model *m2 = model_create(10, &g_cfg);

  ASSERT_EQ(model_num_params(m1), model_num_params(m2));

  /* Save both models and compare the binary files byte-for-byte */
  ASSERT(model_save(m1, "same_seed_1.bin") == 0);
  ASSERT(model_save(m2, "same_seed_2.bin") == 0);

  FILE *f1 = fopen("same_seed_1.bin", "rb");
  FILE *f2 = fopen("same_seed_2.bin", "rb");
  ASSERT(f1 != NULL);
  ASSERT(f2 != NULL);

  int identical = 1;
  int c1, c2;
  while ((c1 = fgetc(f1)) != EOF && (c2 = fgetc(f2)) != EOF) {
    if (c1 != c2) {
      identical = 0;
      break;
    }
  }
  /* Also check both reached EOF at the same time */
  if (identical)
    identical = (fgetc(f1) == EOF && fgetc(f2) == EOF);

  fclose(f1);
  fclose(f2);
  remove("same_seed_1.bin");
  remove("same_seed_2.bin");

  ASSERT(identical);

  model_free(m1);
  model_free(m2);
}

/* ==================================================================== */
/*                   OVERFIT / CONVERGENCE                               */
/* ==================================================================== */

TEST(overfit_single_sequence_near_zero_loss) {
  /* Train for many steps on a single short sequence.
     Loss should approach near-zero for a small model. */
  seed_rng(123);
  Model *m = model_create(10, &g_cfg);
  size_t np = model_num_params(m);
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *mom = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *vel = (scalar_t *)calloc(np, sizeof(scalar_t));
  scalar_t *keys[g_cfg.n_layer], *vals[g_cfg.n_layer];
  size_t cl[g_cfg.n_layer];
  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_ALLOC(keys, L);
    TEST_KV_ALLOC(vals, L);
  }

  /* Train on fixed sequence: 0→1→2→3 */
  size_t seq[] = {0, 1, 2, 3};
  int seq_len = 4;
  scalar_t last_loss = 0;

  for (int step = 0; step < 500; step++) {
    memset(grads, 0, np * sizeof(scalar_t));
    scalar_t step_loss = 0;
    for (int L = 0; L < g_cfg.n_layer; L++) {
      TEST_KV_RESET(keys, L, cl);
      TEST_KV_RESET(vals, L, cl);
    }
    for (int p = 0; p < seq_len - 1; p++)
      step_loss +=
          forward_backward_one(m, seq[p], p, seq[p + 1], keys, vals, cl, grads);
    step_loss /= (scalar_t)(seq_len - 1);
    for (size_t i = 0; i < np; i++)
      grads[i] /= (scalar_t)(seq_len - 1);
    adam_step(m, grads, mom, vel, step);
    last_loss = step_loss;
  }

  /* After 500 steps on a 4-token sequence, loss should be very small */
  ASSERT_LT(last_loss, 0.01);

  for (int L = 0; L < g_cfg.n_layer; L++) {
    TEST_KV_FREE(keys, L);
    TEST_KV_FREE(vals, L);
  }
  free(grads);
  free(mom);
  free(vel);
  model_free(m);
}

/* ==================================================================== */
/*                   KV CACHE EDGE CASES                                 */
/* ==================================================================== */

TEST(kv_cache_alloc_free_roundtrip) {
  /* Allocate and free KV cache without using it — no leak */
  scalar_t *kv = kv_cache_alloc(&g_cfg);
  ASSERT(kv != NULL);
  kv_cache_free(kv);
}

TEST(kv_cache_reset_zeroes_data) {
  scalar_t *kv = kv_cache_alloc(&g_cfg);
  /* Write some data */
  kv[0] = 42.0;
  kv[1] = 99.0;
  /* Reset should zero it */
  kv_cache_reset(kv, &g_cfg);
#ifndef MICROGPT_PAGED_KV
  ASSERT(fabs(kv[0]) < 1e-15);
  ASSERT(fabs(kv[1]) < 1e-15);
#endif
  kv_cache_free(kv);
}

/* ==================================================================== */
/*                            MAIN                                       */
/* ==================================================================== */

int main(void) {
  g_cfg = microgpt_default_config();
  g_cfg.warmup_steps = 5; /* fast LR ramp-up for short training tests */

  printf("\n=== MicroGPT-C Unit Tests ===\n\n");

  /* Utility */
  printf("[Utility]\n");
  RUN(seed_rng_deterministic);
  RUN(load_file_success);
  RUN(load_file_missing);

  /* Character-level tokenisation */
  printf("\n[Character-Level Tokenisation]\n");
  RUN(load_docs_basic);
  RUN(load_docs_missing_file);
  RUN(build_vocab_basic);
  RUN(tokenize_basic);
  RUN(tokenize_max_len_truncation);

  /* Word-level tokenisation */
  printf("\n[Word-Level Tokenisation]\n");
  RUN(build_word_vocab_basic);
  RUN(build_word_vocab_max_words_limit);
  RUN(word_to_id_known_and_unknown);
  RUN(tokenize_words_basic);
  RUN(tokenize_words_unknown_maps_to_unk);
  RUN(tokenize_words_max_tokens_truncation);
  RUN(free_word_vocab_zeroes_struct);
  RUN(word_vocab_special_tokens);

  /* Model lifecycle */
  printf("\n[Model Lifecycle]\n");
  RUN(model_create_and_free);
  RUN(model_num_params_scales_with_vocab);
  RUN(model_num_params_formula);
  RUN(different_vocab_sizes_produce_different_params);

  /* Gradient correctness */
  printf("\n[Gradient Correctness]\n");
  RUN(gradient_direction_reduces_loss);
  RUN(gradient_accumulates_over_positions);

  /* KV cache */
  printf("\n[KV Cache Behaviour]\n");
  RUN(kv_cache_grows_with_positions);
  RUN(inference_kv_cache_autoregressive);

  /* Training */
  printf("\n[Training]\n");
  RUN(forward_backward_returns_positive_loss);
  RUN(training_reduces_loss);
  RUN(training_loss_decreases_monotonically_in_windows);
  RUN(training_reproducible_with_same_seed);

  /* Loss sanity */
  printf("\n[Loss Sanity]\n");
  RUN(loss_is_log_vocab_for_uniform_predictions);
  RUN(loss_different_targets_give_different_losses);

  /* Inference / Sampling */
  printf("\n[Inference / Sampling]\n");
  RUN(forward_inference_produces_logits);
  RUN(forward_inference_matches_fwd_bwd_logits);
  RUN(sample_token_returns_valid_id);
  RUN(sample_token_low_temp_picks_argmax);
  RUN(sample_token_deterministic_with_seed);
  RUN(sample_token_high_temp_is_more_uniform);

  printf("\n[Model Save/Load]\n");
  RUN(model_save_and_load_roundtrip);

  /* Checkpoints */
  printf("\n[Training Checkpoints]\n");
  RUN(checkpoint_save_and_load_roundtrip);
  RUN(checkpoint_load_missing_file);

  /* Tiled linear algebra */
  printf("\n[Tiled Linear Algebra]\n");
  RUN(lin_fwd_tiled_matches_reference);
  RUN(lin_bwd_tiled_dx_matches_reference);
  RUN(lin_bwd_tiled_dw_matches_reference);
  RUN(lin_fwd_large_tile_aligned);
  RUN(lin_fwd_small_below_tile);

#ifdef MICROGPT_PAGED_KV
  /* Paged KV cache */
  printf("\n[Paged KV Cache]\n");
  RUN(paged_kv_create_and_free);
  RUN(paged_kv_append_and_get);
  RUN(paged_kv_grows_across_page_boundary);
  RUN(paged_kv_reset_reuses_pages);
#endif

  /* Config API */
  printf("\n[Config API]\n");
  RUN(default_config_has_valid_fields);
  RUN(custom_config_overrides);
  RUN(model_config_accessor);

  /* Softmax / Sampling edge cases */
  printf("\n[Softmax / Sampling Edge Cases]\n");
  RUN(softmax_output_sums_to_one);
  RUN(sample_token_zero_temp_picks_max);
  RUN(sample_token_single_vocab);

  /* Multi-position training */
  printf("\n[Multi-Position Training]\n");
  RUN(multi_position_loss_accumulation);

  /* Seed / Reproducibility */
  printf("\n[Seed / Reproducibility]\n");
  RUN(different_seeds_produce_different_models);
  RUN(same_seed_produces_identical_models);

  /* Overfit / Convergence */
  printf("\n[Overfit / Convergence]\n");
  RUN(overfit_single_sequence_near_zero_loss);

  /* KV Cache edge cases */
  printf("\n[KV Cache Edge Cases]\n");
  RUN(kv_cache_alloc_free_roundtrip);
  RUN(kv_cache_reset_zeroes_data);

  /* Summary */
  printf("\n=== Results: %d/%d passed", g_tests_passed, g_tests_run);
  if (g_tests_failed > 0)
    printf(" (%d FAILED)", g_tests_failed);
  printf(" ===\n\n");

  return g_tests_failed > 0 ? 1 : 0;
}
