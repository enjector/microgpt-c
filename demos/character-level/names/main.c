/*
 * MicroGPT-C - C99 implementation matching Andrej Karpathy's microgpt.py
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 * This is the entry point for the MicroGPT character-level language model.
 * It performs three phases:
 *
 *   1. DATA LOADING   - Read a text file where each line is one "document"
 *                        (e.g. a name).  Build a character-level vocabulary
 *                        and shuffle the documents for training variety.
 *
 *   2. TRAINING       - For num_steps optimiser steps:
 *                          a. Pick a document (round-robin over shuffled docs).
 *                          b. Tokenise it: [BOS] chars... [BOS].
 *                          c. Forward+backward every position to accumulate
 *                             gradients, computing cross-entropy loss.
 *                          d. Average gradients and apply one Adam step.
 *
 *   3. INFERENCE      - Generate NUM_SAMPLES new sequences autoregressively:
 *                          a. Start with BOS.
 *                          b. Repeatedly call forward_inference -> sample_token
 *                             until BOS (end) or block_size is reached.
 *
 * Run from the build directory; the build system copies data/names.txt
 * next to the executable automatically.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAINING_LOG "names.ckpt.log"

/* ---------- Main ------------------------------------------------------- */

int main(void) {
  /* Seed both stdlib rand (for shuffle) and internal PRNG (for
   * weights/sampling) */
  srand(42);
  seed_rng(42);

  /* Runtime configuration — start from defaults, override as needed */
  MicrogptConfig cfg = microgpt_default_config();
  microgpt_print_config("MicroGPT-C - Name Generation Demo", &cfg);

  const int nl = cfg.n_layer;
  const int bs = cfg.block_size;

  /* ---- Phase 1: Load and prepare data ---- */
  Docs docs = {0};
  const char *path = "names.txt";
  if (load_docs(path, &docs, cfg.max_docs) != 0) {
    fprintf(stderr, "Cannot open names.txt\n");
    return 1;
  }
  shuffle_docs(&docs);
  printf("num docs: %zu\n", docs.num_docs);

  /* Build character-level vocabulary from all documents */
  Vocab vocab = {0};
  build_vocab(&docs, &vocab);
  printf("vocab size: %zu\n", vocab.vocab_size);

  /* Allocate and randomly initialise the Transformer model */
  Model *model = model_create(vocab.vocab_size, &cfg);
  if (!model) {
    fprintf(stderr, "model_create failed\n");
    free_docs(&docs);
    free(vocab.chars);
    return 1;
  }
  size_t nparams = model_num_params(model);
  printf("num params: %zu\n", nparams);

  /* Open training log (append mode) */
  FILE *logf = fopen(TRAINING_LOG, "a");
  if (logf) {
    time_t now = time(NULL);
    struct tm *lt = localtime(&now);
    fprintf(logf, "\n========================================\n");
    fprintf(logf, "Run: %04d-%02d-%02d %02d:%02d:%02d\n", lt->tm_year + 1900,
            lt->tm_mon + 1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
    fprintf(logf, "========================================\n");
    fprintf(logf, "Corpus: %zu docs\n", docs.num_docs);
    fprintf(logf, "Vocab: %zu characters\n", vocab.vocab_size);
    fprintf(logf,
            "Architecture: N_EMBD=%d N_LAYER=%d N_HEAD=%d BLOCK_SIZE=%d\n",
            cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.block_size);
    fprintf(logf, "Params: %zu\n", nparams);
    fprintf(logf, "Training: batch=%d steps=%d lr=%.4f\n", cfg.batch_size,
            cfg.num_steps, (scalar_t)cfg.learning_rate);
    fprintf(logf, "Inference: temp=%.2f samples=%d\n",
            (scalar_t)cfg.temperature, NUM_SAMPLES);
    fflush(logf);
  }
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  printf("weights: INT8 (per-matrix scale)\n");
#endif

  /*
   * Allocate optimiser state:
   *   grad_buffer - accumulated gradients (zeroed each step)
   *   m           - Adam 1st moment estimates (running mean of gradients)
   *   v           - Adam 2nd moment estimates (running mean of squared
   * gradients)
   */
  scalar_t *grad_buffer = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *m = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *v = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  if (!grad_buffer || !m || !v) {
    fprintf(stderr, "alloc failed\n");
    model_free(model);
    free_docs(&docs);
    free(vocab.chars);
    return 1;
  }

  /*
   * Key/Value cache for causal self-attention.
   * Each layer maintains its own cache of projected K and V vectors
   * for all positions processed so far (up to block_size).
   */
  scalar_t **keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    keys[L] = kv_cache_alloc(&cfg);
    values[L] = kv_cache_alloc(&cfg);
    if (!keys[L] || !values[L]) {
      for (int i = 0; i <= L; i++) {
        kv_cache_free(keys[i]);
        kv_cache_free(values[i]);
      }
      free(keys);
      free(values);
      free(cache_len);
      free(grad_buffer);
      free(m);
      free(v);
      model_free(model);
      free_docs(&docs);
      free(vocab.chars);
      return 1;
    }
  }

  /* ---- Phase 2: Training loop (mini-batch gradient accumulation) ---- */

  if (logf)
    fprintf(logf, "\n--- Training ---\n");
  size_t *token_buf = (size_t *)malloc(((size_t)bs + 2) * sizeof(size_t));
  size_t total_train_tokens = 0;
  size_t doc_idx = 0; /* current position in the shuffled dataset */
  clock_t t_train_start = clock();

  for (int step = 0; step < cfg.num_steps; step++) {
    /* Zero gradient accumulator for this step */
    memset(grad_buffer, 0, nparams * sizeof(scalar_t));

    scalar_t batch_loss = 0;
    size_t batch_positions = 0;

    /* Accumulate gradients over batch_size documents */
    for (int b = 0; b < cfg.batch_size; b++) {
      /* Reset KV cache for each document */
      for (int L = 0; L < nl; L++)
        cache_len[L] = 0;

      /* Select next document (wrap around the shuffled dataset) */
      const char *doc = docs.lines[doc_idx % docs.num_docs];
      size_t doc_len = docs.doc_lens[doc_idx % docs.num_docs];
      doc_idx++;

      /* Tokenise: [BOS] [char_0] ... [char_n] [BOS/EOS] */
      size_t n_tok = tokenize(doc, doc_len, &vocab, token_buf, (size_t)bs + 2);
      size_t n = n_tok - 1; /* number of (input, target) pairs */
      if (n > (size_t)bs)
        n = (size_t)bs;
      total_train_tokens += n;
      batch_positions += n;

      /* Forward + backward for every position in the document */
      for (size_t pos = 0; pos < n; pos++) {
        scalar_t loss =
            forward_backward_one(model, token_buf[pos], pos, token_buf[pos + 1],
                                 keys, values, cache_len, grad_buffer);
        batch_loss += loss;
      }
    }

    scalar_t mean_loss = batch_loss / (scalar_t)batch_positions;

    /* Average gradients over all positions across the batch */
    for (size_t i = 0; i < nparams; i++)
      grad_buffer[i] /= (scalar_t)batch_positions;

    /* Apply one Adam optimiser step (updates model weights in-place) */
    adam_step(model, grad_buffer, m, v, step);

    if ((step + 1) % 100 == 0 || step == 0) {
      scalar_t elapsed =
          (scalar_t)(clock() - t_train_start) / (scalar_t)CLOCKS_PER_SEC;
      if (elapsed < 0.001)
        elapsed = 0.001;
      scalar_t sps = (step + 1) / elapsed;
      scalar_t eta = (cfg.num_steps - step - 1) / sps;
      int el_m = (int)elapsed / 60, el_s = (int)elapsed % 60;
      int eta_m = (int)eta / 60, eta_s = (int)eta % 60;
      printf("step %4d / %d | loss %.4f | %dm%02ds elapsed, ETA %dm%02ds\n",
             step + 1, cfg.num_steps, mean_loss, el_m, el_s, eta_m, eta_s);

      if (logf) {
        fprintf(logf, "step %4d / %d | loss %.4f\n", step + 1, cfg.num_steps,
                mean_loss);
        fflush(logf);
      }
    }
  }

  clock_t t_train_end = clock();
  scalar_t train_sec =
      (scalar_t)(t_train_end - t_train_start) / (scalar_t)CLOCKS_PER_SEC;
  printf("\nTraining: %.2f s | %.1f steps/s | %.1f k tokens/s\n", train_sec,
         (scalar_t)cfg.num_steps / train_sec,
         (scalar_t)total_train_tokens / train_sec / 1000.0);

  if (logf) {
    fprintf(logf, "\nTraining complete: %.2fs | %.1f steps/s | %.1fk tok/s\n",
            train_sec, (scalar_t)cfg.num_steps / train_sec,
            (scalar_t)total_train_tokens / train_sec / 1000.0);
    fflush(logf);
  }

  /* ---- Phase 3: Autoregressive inference ---- */

  printf("\n--- inference (new, hallucinated names) ---\n");
  scalar_t *logits_buf =
      (scalar_t *)malloc((size_t)cfg.max_vocab * sizeof(scalar_t));
  size_t total_infer_tokens = 0;
  clock_t t_infer_start = clock();

  for (int sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) {
    /* Reset KV cache for each new sample */
    for (int L = 0; L < nl; L++)
      cache_len[L] = 0;

    size_t token_id = vocab.bos_id; /* start with BOS token */
    char *name = (char *)malloc((size_t)bs + 1);
    size_t len = 0;

    /*
     * Autoregressive generation loop:
     *   1. Run forward_inference to get next-token logits.
     *   2. Sample from the softmax distribution (temperature-controlled).
     *   3. Stop if BOS (used as EOS) is sampled or max length reached.
     */
    for (size_t pos = 0; pos < (size_t)bs && len < (size_t)bs; pos++) {
      forward_inference(model, token_id, pos, keys, values, cache_len,
                        logits_buf);
      token_id =
          sample_token(logits_buf, vocab.vocab_size, (scalar_t)cfg.temperature);
      if (token_id == vocab.bos_id)
        break; /* EOS: end of generated sequence */
      name[len++] = (char)vocab.chars[token_id];
    }
    total_infer_tokens += len;
    name[len] = '\0';
    printf("sample %2d: %s\n", sample_idx + 1, name);
    free(name);
  }

  clock_t t_infer_end = clock();
  scalar_t infer_sec =
      (scalar_t)(t_infer_end - t_infer_start) / (scalar_t)CLOCKS_PER_SEC;
  if (infer_sec < 0.001)
    infer_sec = 0.001; /* avoid div-by-zero for very fast inference */
  printf("\nInference: %.2f s | %.1f samples/s | %.1f tokens/s\n", infer_sec,
         (scalar_t)NUM_SAMPLES / infer_sec,
         (scalar_t)total_infer_tokens / infer_sec);

  if (logf) {
    fprintf(logf, "\n--- Inference Summary ---\n");
    fprintf(logf, "samples: %d | tokens: %zu | time: %.2fs | %.1f tok/s\n",
            NUM_SAMPLES, total_infer_tokens, infer_sec,
            (scalar_t)total_infer_tokens / infer_sec);
    fprintf(logf, "========================================\n\n");
    fclose(logf);
    logf = NULL;
    printf("training log appended to '%s'\n", TRAINING_LOG);
  }

  /* ---- Cleanup: release all heap memory ---- */
  for (int L = 0; L < nl; L++) {
    kv_cache_free(keys[L]);
    kv_cache_free(values[L]);
  }
  free(keys);
  free(values);
  free(cache_len);
  free(token_buf);
  free(logits_buf);
  free(grad_buffer);
  free(m);
  free(v);
  model_free(model);
  free_docs(&docs);
  free(vocab.chars);
  return 0;
}
