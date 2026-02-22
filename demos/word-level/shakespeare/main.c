/*
 * MicroGPT-C — Shakespeare Word-Level Generation Example
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates word-level text generation using the MicroGPT library.
 * Trains a small GPT on Shakespeare's complete works and generates new
 * Shakespearean text word by word.
 *
 * This is the word-level counterpart to the character-level Shakespeare demo.
 * Word-level tokenisation trades vocabulary size for semantic density — each
 * token carries a whole word's meaning instead of a single letter.
 *
 * Build:
 *   cmake --build build --target shakespeare_word_demo
 *   ./build/shakespeare_word_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SHAKES_SAMPLES 5
#define SHAKES_TEMP 0.7
#define GEN_LEN 100 /* words to generate per sample */
#define CHECKPOINT_FILE "shakespeare_word.ckpt"
#define TRAINING_LOG "shakespeare_word.ckpt.log"

/* Max threads (actual count is auto-detected at runtime) */
#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

/* ---------- Text normalisation ----------
 * Reduces vocabulary fragmentation by:
 * 1. Lowercasing all characters ("The" → "the")
 * 2. Splitting trailing punctuation into separate tokens
 *    ("art," → "art ,")  ("John." → "john .")
 *
 * This means "the", "the,", "the.", "The" all share the vocab
 * slot for "the", and punctuation gets its own small set of tokens.
 *
 * The output may be up to 2× the input length (every char could
 * gain a space before its punctuation). Caller must allocate enough.
 */
static int is_split_punct(char c) {
  return c == ',' || c == '.' || c == ';' || c == ':' || c == '!' || c == '?' ||
         c == ')' || c == '(' || c == '[' || c == ']' || c == '{' || c == '}';
}

static size_t normalize_text(const char *src, size_t src_len, char *dst,
                             size_t dst_cap) {
  size_t j = 0;
  for (size_t i = 0; i < src_len && j + 2 < dst_cap; i++) {
    char c = src[i];

    /* Lowercase */
    if (c >= 'A' && c <= 'Z')
      c = (char)(c + 32);

    /* Split punctuation: insert space before it */
    if (is_split_punct(c)) {
      /* Don't double-space if already preceded by space */
      if (j > 0 && dst[j - 1] != ' ' && dst[j - 1] != '\n')
        dst[j++] = ' ';
      dst[j++] = c;
      /* Add trailing space too so next word is separate */
      if (i + 1 < src_len && src[i + 1] != ' ' && src[i + 1] != '\n')
        dst[j++] = ' ';
    } else {
      dst[j++] = c;
    }
  }
  dst[j] = '\0';
  return j;
}

int main(void) {
  unsigned int train_seed = 42; /* deterministic training */
  unsigned int infer_seed = (unsigned int)time(NULL); /* varied inference */
  srand(train_seed);
  seed_rng(train_seed);

  /* Runtime configuration — word-level Shakespeare overrides.
   * Word-level sequences are ~8 tokens/line (vs ~40 chars), so we use
   * a smaller/shallower model than char-level. The vocab embedding table
   * dominates param count (3000×48 = 144K), so keeping vocab moderate
   * is the main lever for model size. */
  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = 48;
  cfg.n_head = 4;
  cfg.mlp_dim = 192;
  cfg.n_layer = 1;
  cfg.block_size = 64;
  cfg.batch_size = 16;
  cfg.num_steps = 10000;
  cfg.learning_rate = 0.001;
  cfg.max_vocab = 5000;
  cfg.max_docs = 200000;
  cfg.max_doc_len = 512;
  microgpt_print_config("MicroGPT-C - Shakespeare Word-Level Demo", &cfg);

  const int nl = cfg.n_layer;

#ifdef MICROGPT_METAL
  if (metal_init() != 0)
    fprintf(stderr, "WARNING: Metal GPU init failed, falling back to CPU\n");
#endif

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

  /* ---- Build word-level vocabulary ---- */
  /* Concatenate all docs, then normalise (lowercase + punct split).
   * Note: build_word_vocab uses O(n²) dedup, so we limit vocab discovery
   * to the first ~500KB of normalised text. */
  size_t raw_size = total_chars + docs.num_docs + 1;
  char *raw_text = (char *)malloc(raw_size);
  if (!raw_text) {
    fprintf(stderr, "OOM\n");
    return 1;
  }
  size_t text_pos = 0;
  for (size_t d = 0; d < docs.num_docs; d++) {
    memcpy(raw_text + text_pos, docs.lines[d], docs.doc_lens[d]);
    text_pos += docs.doc_lens[d];
    raw_text[text_pos++] = '\n';
  }
  raw_text[text_pos] = '\0';

  /* Normalise: lowercase + split punctuation (output can be ~2× input) */
  size_t norm_cap = text_pos * 2 + 1;
  char *all_text = (char *)malloc(norm_cap);
  if (!all_text) {
    fprintf(stderr, "OOM\n");
    return 1;
  }
  size_t norm_len = normalize_text(raw_text, text_pos, all_text, norm_cap);
  free(raw_text);
  printf("normalised text: %zu chars (%.1f KB)\n", norm_len,
         (double)norm_len / 1024.0);

  /* Use first 500KB for vocab discovery (fast), full text for tokenization */
  size_t vocab_sample_len = norm_len;
  if (vocab_sample_len > 500000)
    vocab_sample_len = 500000;

  WordVocab wv = {0};
  size_t max_words = (size_t)cfg.max_vocab - 3; /* reserve unk, newline, bos */
  printf("building word vocab from first %.0f KB...\n",
         (double)vocab_sample_len / 1024.0);
  build_word_vocab(all_text, vocab_sample_len, max_words, &wv);
  printf("word vocab: %zu tokens (%zu words + 3 special)\n", wv.vocab_size,
         wv.num_words);

  /* Pre-tokenize all documents (normalise each line first) */
  size_t **doc_tokens = (size_t **)calloc(docs.num_docs, sizeof(size_t *));
  size_t *doc_tok_lens = (size_t *)calloc(docs.num_docs, sizeof(size_t));
  size_t total_word_tokens = 0;
  size_t unk_count = 0;

  /* Reusable normalisation buffer for per-doc tokenisation */
  size_t norm_buf_cap = (cfg.max_doc_len + 1) * 2;
  char *norm_buf = (char *)malloc(norm_buf_cap);
  if (!norm_buf) {
    fprintf(stderr, "OOM\n");
    return 1;
  }

  for (size_t d = 0; d < docs.num_docs; d++) {
    /* Normalise this document line */
    size_t nlen =
        normalize_text(docs.lines[d], docs.doc_lens[d], norm_buf, norm_buf_cap);

    /* Allocate generously: max one word token per character */
    size_t max_tok = nlen + 1;
    if (max_tok < 16)
      max_tok = 16;
    doc_tokens[d] = (size_t *)malloc(max_tok * sizeof(size_t));
    if (!doc_tokens[d]) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
    doc_tok_lens[d] =
        tokenize_words(norm_buf, nlen, &wv, doc_tokens[d], max_tok);
    total_word_tokens += doc_tok_lens[d];
    for (size_t t = 0; t < doc_tok_lens[d]; t++)
      if (doc_tokens[d][t] == wv.unk_id)
        unk_count++;
  }
  free(norm_buf);
  printf("total word tokens: %zu (avg %.1f per line)\n", total_word_tokens,
         (double)total_word_tokens / docs.num_docs);
  printf("unknown tokens: %zu (%.2f%%)\n", unk_count,
         total_word_tokens > 0 ? 100.0 * unk_count / total_word_tokens : 0.0);

  /* Print example tokenisations */
  printf("\n--- example tokenisations ---\n");
  for (size_t d = 0; d < 3 && d < docs.num_docs; d++) {
    printf("line[%zu] (%zu tokens): ", d, doc_tok_lens[d]);
    for (size_t t = 0; t < doc_tok_lens[d] && t < 15; t++) {
      size_t tid = doc_tokens[d][t];
      if (tid == wv.newline_id)
        printf("[\\n] ");
      else if (tid == wv.bos_id)
        printf("[BOS] ");
      else if (tid == wv.unk_id)
        printf("[UNK] ");
      else if (tid < wv.vocab_size && wv.words[tid])
        printf("[%s] ", wv.words[tid]);
      else
        printf("[?%zu] ", tid);
    }
    if (doc_tok_lens[d] > 15)
      printf("...");
    printf("\n");
  }
  printf("\n");

  if (wv.vocab_size > (size_t)cfg.max_vocab) {
    fprintf(stderr, "vocab_size %zu exceeds max_vocab %d\n", wv.vocab_size,
            cfg.max_vocab);
    return 1;
  }

  /* Open training log (append mode) */
  FILE *logf = fopen(TRAINING_LOG, "a");
  if (logf) {
    time_t now = time(NULL);
    struct tm *lt = localtime(&now);
    fprintf(logf, "\n========================================\n");
    fprintf(logf, "Run: %04d-%02d-%02d %02d:%02d:%02d\n", lt->tm_year + 1900,
            lt->tm_mon + 1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
    fprintf(logf, "========================================\n");
    fprintf(logf, "Tokenisation: WORD-LEVEL\n");
    fprintf(logf, "Corpus: %zu lines | %zu word tokens\n", docs.num_docs,
            total_word_tokens);
    fprintf(logf, "Vocab: %zu tokens (%zu words + 3 special)\n", wv.vocab_size,
            wv.num_words);
    fprintf(logf, "UNK rate: %.2f%%\n",
            total_word_tokens > 0 ? 100.0 * unk_count / total_word_tokens
                                  : 0.0);
    fflush(logf);
  }

  printf("N_EMBD=%d BLOCK_SIZE=%d N_LAYER=%d N_HEAD=%d\n\n", cfg.n_embd,
         cfg.block_size, cfg.n_layer, cfg.n_head);

  /* ---- Create or load model ---- */
  size_t nparams;
  scalar_t *grad_buffer, *m_buf, *v_buf;
  Model *model = NULL;
  int trained = 0;

  {
    Model *tmp = model_create(wv.vocab_size, &cfg);
    if (!tmp) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
    nparams = model_num_params(tmp);
    model_free(tmp);
  }
  printf("params: %zu\n", nparams);

  m_buf = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  v_buf = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  grad_buffer = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  if (!grad_buffer || !m_buf || !v_buf) {
    fprintf(stderr, "OOM (optimiser)\n");
    return 1;
  }

  /* Try loading checkpoint */
  int resume_step = 0;
  model = checkpoint_load(CHECKPOINT_FILE, wv.vocab_size, &cfg, m_buf, v_buf,
                          &resume_step);
  if (model) {
    printf("loaded checkpoint '%s' (trained %d steps)\n\n", CHECKPOINT_FILE,
           resume_step);
    trained = 1;
  } else {
    model = model_create(wv.vocab_size, &cfg);
    if (!model) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
  }

  if (logf) {
    fprintf(logf,
            "Architecture: N_EMBD=%d N_LAYER=%d N_HEAD=%d BLOCK_SIZE=%d\n",
            cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.block_size);
    fprintf(logf, "Params: %zu\n", nparams);
    fprintf(logf, "Training: batch=%d steps=%d lr=%.4f\n", cfg.batch_size,
            cfg.num_steps, (scalar_t)cfg.learning_rate);
    fflush(logf);
  }

  /* ---- KV caches ---- */
  scalar_t **train_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **train_values =
      (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *train_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    train_keys[L] = kv_cache_alloc(&cfg);
    train_values[L] = kv_cache_alloc(&cfg);
  }

  /* KV cache for inference */
  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    inf_keys[L] = kv_cache_alloc(&cfg);
    inf_values[L] = kv_cache_alloc(&cfg);
  }

  /* ---- Training (single-threaded for simplicity with word-level) ---- */
  if (!trained) {
    if (logf)
      fprintf(logf, "\n--- Training ---\n");
    size_t tokens_trained = 0;
    time_t t0 = time(NULL);

    for (int step = 0; step < cfg.num_steps; step++) {
      memset(grad_buffer, 0, nparams * sizeof(scalar_t));
      scalar_t step_loss = 0;
      size_t step_positions = 0;

      for (int b = 0; b < cfg.batch_size; b++) {
        /* Pick random document */
        size_t di = (size_t)(rand() % (int)docs.num_docs);
        size_t tok_len = doc_tok_lens[di];
        if (tok_len < 2)
          continue;

        /* Reset KV cache */
        for (int L = 0; L < nl; L++)
          train_cache_len[L] = 0;

        /* Limit to block_size */
        size_t seq_len = tok_len;
        if (seq_len > (size_t)cfg.block_size)
          seq_len = (size_t)cfg.block_size;

        /* BOS -> first token */
        step_loss += forward_backward_one(
            model, wv.bos_id, 0, doc_tokens[di][0], train_keys, train_values,
            train_cache_len, grad_buffer);
        step_positions++;

        /* Positions 1..seq_len-1 */
        for (size_t t = 1; t < seq_len; t++) {
          step_loss += forward_backward_one(
              model, doc_tokens[di][t - 1], t, doc_tokens[di][t], train_keys,
              train_values, train_cache_len, grad_buffer);
          step_positions++;
        }
      }

      tokens_trained += step_positions;
      scalar_t mean_loss = step_loss / (scalar_t)step_positions;
      for (size_t i = 0; i < nparams; i++)
        grad_buffer[i] /= (scalar_t)step_positions;
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

  /* ---- Generate Shakespeare (word-level) ---- */
  seed_rng(infer_seed); /* re-seed for varied inference */
  printf("\n--- generated Shakespeare (word-level) ---\n");
  scalar_t *logits_buf = (scalar_t *)malloc(wv.vocab_size * sizeof(scalar_t));

  /* Inference timing */
  int total_inf_tokens = 0;
  double total_inf_time = 0.0;

  /* Seed words — common Shakespeare openers */
  const char *seeds[] = {"the", "o", "what", "my", "how"};

  for (int s = 0; s < SHAKES_SAMPLES; s++) {
    for (int L = 0; L < nl; L++)
      inf_cache_len[L] = 0;

    /* Look up seed word ID */
    size_t token = word_to_id(&wv, seeds[s]);
    if (token == wv.unk_id) {
      /* Try lowercase */
      char lower[64];
      snprintf(lower, sizeof(lower), "%s", seeds[s]);
      for (int i = 0; lower[i]; i++)
        if (lower[i] >= 'A' && lower[i] <= 'Z')
          lower[i] += 32;
      token = word_to_id(&wv, lower);
    }

    printf("\n[sample %d — seed: \"%s\"]\n%s", s + 1, seeds[s], seeds[s]);

    /* Start inference timer */
    struct timespec inf_start, inf_end;
    clock_gettime(CLOCK_MONOTONIC, &inf_start);

    /* Generate words */
    int gen_len = GEN_LEN;
    if (gen_len > cfg.block_size - 1)
      gen_len = cfg.block_size - 1;
    int gen_count = 0;

    /* Feed BOS then seed word */
    forward_inference(model, wv.bos_id, 0, inf_keys, inf_values, inf_cache_len,
                      logits_buf);
    forward_inference(model, token, 1, inf_keys, inf_values, inf_cache_len,
                      logits_buf);
    size_t pos = 2;

    for (int g = 0; g < gen_len && pos < (size_t)cfg.block_size; g++) {
      token = sample_token(logits_buf, wv.vocab_size, SHAKES_TEMP);
      if (token == wv.bos_id)
        break;

      /* Print the word with appropriate spacing */
      if (token == wv.newline_id) {
        printf("\n");
      } else if (token == wv.unk_id) {
        printf("...");
      } else if (token < wv.vocab_size && wv.words[token]) {
        const char *w = wv.words[token];
        /* No space before punctuation */
        if (w[0] == ',' || w[0] == '.' || w[0] == ';' || w[0] == ':' ||
            w[0] == '!' || w[0] == '?' || w[0] == '\'' || w[0] == ')')
          printf("%s", w);
        else
          printf(" %s", w);
      } else {
        printf("...");
      }
      gen_count++;

      forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                        logits_buf);
      pos++;
    }

    /* Stop timer */
    clock_gettime(CLOCK_MONOTONIC, &inf_end);
    double inf_elapsed = (inf_end.tv_sec - inf_start.tv_sec) +
                         (inf_end.tv_nsec - inf_start.tv_nsec) / 1e9;
    int tokens_this_sample = gen_count + 2; /* +BOS+seed */
    double tok_per_sec = inf_elapsed > 0 ? tokens_this_sample / inf_elapsed : 0;
    total_inf_tokens += tokens_this_sample;
    total_inf_time += inf_elapsed;
    printf("\n  >> %d words in %.3fs (%.0f tok/s)\n", gen_count, inf_elapsed,
           tok_per_sec);

    if (logf) {
      fprintf(logf, "  sample %d: seed=\"%s\" | %d words | %.0f tok/s\n", s + 1,
              seeds[s], gen_count, tok_per_sec);
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
  for (int L = 0; L < nl; L++) {
    kv_cache_free(train_keys[L]);
    kv_cache_free(train_values[L]);
    kv_cache_free(inf_keys[L]);
    kv_cache_free(inf_values[L]);
  }
  free(train_keys);
  free(train_values);
  free(train_cache_len);
  free(inf_keys);
  free(inf_values);
  free(inf_cache_len);
  free(logits_buf);
  free(grad_buffer);
  free(m_buf);
  free(v_buf);
  model_free(model);
  for (size_t d = 0; d < docs.num_docs; d++)
    free(doc_tokens[d]);
  free(doc_tokens);
  free(doc_tok_lens);
  free(all_text);
  free_docs(&docs);
  free_word_vocab(&wv);
  return 0;
}
