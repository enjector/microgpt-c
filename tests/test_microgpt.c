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
  Model *m1 = model_create(10);
  ASSERT_NE(m1, NULL);
  size_t n1 = model_num_params(m1);

  seed_rng(42);
  Model *m2 = model_create(10);
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
  int rc = load_docs("_test_docs.txt", &docs);
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
  int rc = load_docs("_nonexistent_xyz.txt", &docs);
  ASSERT_NE(rc, 0);
}

TEST(build_vocab_basic) {
  write_temp_file("_test_vocab.txt", "abc\nbca\n");
  Docs docs = {0};
  load_docs("_test_vocab.txt", &docs);
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
  load_docs("_test_tok.txt", &docs);
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
  load_docs("_test_trunc.txt", &docs);
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
  Model *m = model_create(10);
  ASSERT_NE(m, NULL);
  size_t np = model_num_params(m);
  ASSERT_GT(np, 0);
  model_free(m);
}

TEST(model_num_params_scales_with_vocab) {
  seed_rng(1);
  Model *m10 = model_create(10);
  Model *m50 = model_create(50);
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
  Model *m = model_create(10);
  ASSERT_NE(m, NULL);
  size_t np = model_num_params(m);

  double *grads = (double *)calloc(np, sizeof(double));
  double *keys[N_LAYER], *vals[N_LAYER];
  size_t cl[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    keys[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    vals[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    cl[L] = 0;
  }

  double loss = forward_backward_one(m, 0, 0, 1, keys, vals, cl, grads);
  ASSERT_GT(loss, 0.0);

  /* Gradients should be non-zero after backward pass */
  double grad_norm = 0;
  for (size_t i = 0; i < np; i++)
    grad_norm += grads[i] * grads[i];
  ASSERT_GT(grad_norm, 0.0);

  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(grads);
  model_free(m);
}

TEST(training_reduces_loss) {
  /* Run 50 training steps on a tiny sequence and check loss decreases */
  seed_rng(42);
  Model *m = model_create(10);
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

  /* Train on repeating pattern: 0 1 2 3 0 1 2 3 */
  size_t seq[] = {0, 1, 2, 3, 0, 1, 2, 3};
  size_t seq_len = 8;

  double first_loss = -1, last_loss = -1;

  for (int step = 0; step < 50; step++) {
    memset(grads, 0, np * sizeof(double));
    double step_loss = 0;
    for (int L = 0; L < N_LAYER; L++)
      cl[L] = 0;

    for (size_t p = 0; p < seq_len - 1; p++) {
      step_loss +=
          forward_backward_one(m, seq[p], p, seq[p + 1], keys, vals, cl, grads);
    }
    step_loss /= (double)(seq_len - 1);

    for (size_t i = 0; i < np; i++)
      grads[i] /= (double)(seq_len - 1);
    adam_step(m, grads, mom, vel, step);

    if (step == 0)
      first_loss = step_loss;
    if (step == 49)
      last_loss = step_loss;
  }

  /* Loss should decrease over training */
  ASSERT_LT(last_loss, first_loss);

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
/*                     INFERENCE / SAMPLING                               */
/* ==================================================================== */

TEST(forward_inference_produces_logits) {
  seed_rng(42);
  Model *m = model_create(10);

  double logits[MAX_VOCAB];
  double *keys[N_LAYER], *vals[N_LAYER];
  size_t cl[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    keys[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    vals[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
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

  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  model_free(m);
}

TEST(sample_token_returns_valid_id) {
  seed_rng(42);
  double logits[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

  for (int i = 0; i < 100; i++) {
    size_t tok = sample_token(logits, 5, 1.0);
    ASSERT_LT(tok, 5);
  }
}

TEST(sample_token_low_temp_picks_argmax) {
  seed_rng(42);
  double logits[5] = {0.0, 0.0, 10.0, 0.0, 0.0};

  /* Very low temperature should almost always pick the max logit */
  size_t tok = sample_token(logits, 5, 0.01);
  ASSERT_EQ(tok, 2);
}

TEST(sample_token_deterministic_with_seed) {
  double logits[5] = {1.0, 2.0, 3.0, 2.0, 1.0};

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
  Model *m1 = model_create(10);
  ASSERT_NE(m1, NULL);

  int rc = model_save(m1, "_test_model.bin");
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  /* Save/load disabled for INT8 */
  ASSERT_NE(rc, 0);
  model_free(m1);
  remove("_test_model.bin");
#else
  ASSERT_EQ(rc, 0);

  Model *m2 = model_load("_test_model.bin", 10);
  ASSERT_NE(m2, NULL);

  /* Both models should produce the same logits for the same input */
  double logits1[MAX_VOCAB], logits2[MAX_VOCAB];
  double *k1[N_LAYER], *v1[N_LAYER], *k2[N_LAYER], *v2[N_LAYER];
  size_t cl1[N_LAYER], cl2[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    k1[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    v1[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    k2[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    v2[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    cl1[L] = cl2[L] = 0;
  }

  forward_inference(m1, 0, 0, k1, v1, cl1, logits1);
  forward_inference(m2, 0, 0, k2, v2, cl2, logits2);

  for (size_t i = 0; i < 10; i++)
    ASSERT(fabs(logits1[i] - logits2[i]) < 1e-10);

  for (int L = 0; L < N_LAYER; L++) {
    free(k1[L]);
    free(v1[L]);
    free(k2[L]);
    free(v2[L]);
  }
  model_free(m1);
  model_free(m2);
  remove("_test_model.bin");
#endif
}

TEST(checkpoint_save_and_load_roundtrip) {
  seed_rng(42);
  Model *m1 = model_create(10);
  ASSERT_NE(m1, NULL);
  size_t np = model_num_params(m1);

  /* Simulate some training to populate optimizer state */
  double *grads = (double *)calloc(np, sizeof(double));
  double *m_adam = (double *)calloc(np, sizeof(double));
  double *v_adam = (double *)calloc(np, sizeof(double));
  double *keys[N_LAYER], *vals[N_LAYER];
  size_t cl[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    keys[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    vals[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
  }

  int saved_step = 10;
  for (int step = 0; step < saved_step; step++) {
    memset(grads, 0, np * sizeof(double));
    for (int L = 0; L < N_LAYER; L++)
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
  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(grads);
  free(m_adam);
  free(v_adam);
  model_free(m1);
  remove("_test_ckpt.bin");
#else
  ASSERT_EQ(rc, 0);

  /* Load checkpoint into fresh buffers */
  double *m2_adam = (double *)calloc(np, sizeof(double));
  double *v2_adam = (double *)calloc(np, sizeof(double));
  int loaded_step = -1;
  Model *m2 =
      checkpoint_load("_test_ckpt.bin", 10, m2_adam, v2_adam, &loaded_step);
  ASSERT_NE(m2, NULL);
  ASSERT_EQ(loaded_step, saved_step);

  /* Verify optimizer state matches exactly */
  for (size_t i = 0; i < np; i++) {
    ASSERT(fabs(m_adam[i] - m2_adam[i]) < 1e-15);
    ASSERT(fabs(v_adam[i] - v2_adam[i]) < 1e-15);
  }

  /* Verify model weights produce identical logits */
  double logits1[MAX_VOCAB], logits2[MAX_VOCAB];
  double *k2[N_LAYER], *v2[N_LAYER];
  size_t cl1[N_LAYER], cl2[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    cl[L] = 0;
    k2[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    v2[L] = (double *)calloc((size_t)BLOCK_SIZE * N_EMBD, sizeof(double));
    cl1[L] = cl2[L] = 0;
  }
  forward_inference(m1, 0, 0, keys, vals, cl1, logits1);
  forward_inference(m2, 0, 0, k2, v2, cl2, logits2);
  for (size_t i = 0; i < 10; i++)
    ASSERT(fabs(logits1[i] - logits2[i]) < 1e-10);

  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(vals[L]);
    free(k2[L]);
    free(v2[L]);
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
  double m_buf[1], v_buf[1];
  int step = -1;
  Model *m = checkpoint_load("_nonexistent_ckpt.bin", 10, m_buf, v_buf, &step);
  ASSERT_EQ(m, NULL);
  ASSERT_EQ(step, -1); /* unchanged on failure */
}

/* ==================================================================== */
/*                            MAIN                                       */
/* ==================================================================== */

int main(void) {
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

  /* Training */
  printf("\n[Training]\n");
  RUN(forward_backward_returns_positive_loss);
  RUN(training_reduces_loss);

  /* Inference / Sampling */
  printf("\n[Inference / Sampling]\n");
  RUN(forward_inference_produces_logits);
  RUN(sample_token_returns_valid_id);
  RUN(sample_token_low_temp_picks_argmax);
  RUN(sample_token_deterministic_with_seed);

  printf("\n[Model Save/Load]\n");
  RUN(model_save_and_load_roundtrip);

  /* Checkpoints */
  printf("\n[Training Checkpoints]\n");
  RUN(checkpoint_save_and_load_roundtrip);
  RUN(checkpoint_load_missing_file);

  /* Summary */
  printf("\n=== Results: %d/%d passed", g_tests_passed, g_tests_run);
  if (g_tests_failed > 0)
    printf(" (%d FAILED)", g_tests_failed);
  printf(" ===\n\n");

  return g_tests_failed > 0 ? 1 : 0;
}
