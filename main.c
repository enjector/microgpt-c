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
 *   2. TRAINING       - For NUM_STEPS optimiser steps:
 *                          a. Pick a document (round-robin over shuffled docs).
 *                          b. Tokenise it: [BOS] chars... [BOS].
 *                          c. Forward+backward every position to accumulate
 *                             gradients, computing cross-entropy loss.
 *                          d. Average gradients and apply one Adam step.
 *
 *   3. INFERENCE      - Generate NUM_SAMPLES new sequences autoregressively:
 *                          a. Start with BOS.
 *                          b. Repeatedly call forward_inference -> sample_token
 *                             until BOS (end) or BLOCK_SIZE is reached.
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

/* ---------- Utility ---------------------------------------------------- */

/*
 * shuffle_docs - Fisher-Yates in-place shuffle of the document list.
 *   Randomises the order in which documents are presented during training,
 *   preventing the model from memorising sequential patterns in the dataset.
 */
static void shuffle_docs(Docs *docs) {
  for (size_t i = docs->num_docs; i > 1; i--) {
    size_t j = (size_t)rand() % i; /* uniform random index in [0, i) */
    /* Swap lines[j] <-> lines[i-1] and their lengths */
    char *tmp_line = docs->lines[j];
    size_t tmp_len = docs->doc_lens[j];
    docs->lines[j] = docs->lines[i - 1];
    docs->doc_lens[j] = docs->doc_lens[i - 1];
    docs->lines[i - 1] = tmp_line;
    docs->doc_lens[i - 1] = tmp_len;
  }
}

/* ---------- Main ------------------------------------------------------- */

int main(void) {
  /* Seed both stdlib rand (for shuffle) and internal PRNG (for
   * weights/sampling) */
  srand(42);
  seed_rng(42);

  /* ---- Phase 1: Load and prepare data ---- */
  Docs docs = {0};
  const char *path = "names.txt";
  if (load_docs(path, &docs) != 0) {
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
  Model *model = model_create(vocab.vocab_size);
  if (!model) {
    fprintf(stderr, "model_create failed\n");
    free_docs(&docs);
    free(vocab.chars);
    return 1;
  }
  size_t nparams = model_num_params(model);
  printf("num params: %zu\n", nparams);
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
  double *grad_buffer = (double *)calloc(nparams, sizeof(double));
  double *m = (double *)calloc(nparams, sizeof(double));
  double *v = (double *)calloc(nparams, sizeof(double));
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
   * for all positions processed so far (up to BLOCK_SIZE).
   */
  double *keys[N_LAYER];
  double *values[N_LAYER];
  size_t cache_len[N_LAYER];
  for (int L = 0; L < N_LAYER; L++) {
    keys[L] = (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
    values[L] = (double *)malloc((size_t)BLOCK_SIZE * N_EMBD * sizeof(double));
    if (!keys[L] || !values[L]) {
      for (int i = 0; i <= L; i++) {
        free(keys[i]);
        free(values[i]);
      }
      free(grad_buffer);
      free(m);
      free(v);
      model_free(model);
      free_docs(&docs);
      free(vocab.chars);
      return 1;
    }
  }

  /* ---- Phase 2: Training loop ---- */

  size_t token_buf[BLOCK_SIZE + 2]; /* +2 for BOS prefix and BOS/EOS suffix */
  size_t total_train_tokens = 0;
  clock_t t_train_start = clock();

  for (int step = 0; step < NUM_STEPS; step++) {
    /* Zero gradient accumulator for this step */
    memset(grad_buffer, 0, nparams * sizeof(double));
    /* Reset KV cache (each step processes one full document independently) */
    for (int L = 0; L < N_LAYER; L++)
      cache_len[L] = 0;

    /* Select document (round-robin over shuffled dataset) */
    const char *doc = docs.lines[step % docs.num_docs];
    size_t doc_len = docs.doc_lens[step % docs.num_docs];

    /* Tokenise: [BOS] [char_0] ... [char_n] [BOS/EOS] */
    size_t n_tok = tokenize(doc, doc_len, &vocab, token_buf, BLOCK_SIZE + 2);
    size_t n = n_tok - 1; /* number of (input, target) pairs */
    if (n > BLOCK_SIZE)
      n = BLOCK_SIZE;
    total_train_tokens += n;

    /*
     * Forward + backward for every position in the document.
     * Each call processes token_buf[pos] as input and token_buf[pos+1]
     * as the target, accumulating gradients into grad_buffer.
     */
    double total_loss = 0;
    for (size_t pos = 0; pos < n; pos++) {
      double loss =
          forward_backward_one(model, token_buf[pos], pos, token_buf[pos + 1],
                               keys, values, cache_len, grad_buffer);
      total_loss += loss;
    }
    double mean_loss = total_loss / (double)n;

    /* Average gradients over positions (mean reduction) */
    for (size_t i = 0; i < nparams; i++)
      grad_buffer[i] /= (double)n;

    /* Apply one Adam optimiser step (updates model weights in-place) */
    adam_step(model, grad_buffer, m, v, step);

    /* Progress logging every 100 steps (and at step 0) */
    if ((step + 1) % 100 == 0 || step == 0)
      printf("step %4d / %d | loss %.4f\n", step + 1, NUM_STEPS, mean_loss);
  }

  clock_t t_train_end = clock();
  double train_sec =
      (double)(t_train_end - t_train_start) / (double)CLOCKS_PER_SEC;
  printf("\nTraining: %.2f s | %.1f steps/s | %.1f k tokens/s\n", train_sec,
         (double)NUM_STEPS / train_sec,
         (double)total_train_tokens / train_sec / 1000.0);

  /* ---- Phase 3: Autoregressive inference ---- */

  printf("\n--- inference (new, hallucinated names) ---\n");
  double logits_buf[MAX_VOCAB];
  size_t total_infer_tokens = 0;
  clock_t t_infer_start = clock();

  for (int sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) {
    /* Reset KV cache for each new sample */
    for (int L = 0; L < N_LAYER; L++)
      cache_len[L] = 0;

    size_t token_id = vocab.bos_id; /* start with BOS token */
    char name[BLOCK_SIZE + 1];      /* buffer for the generated string */
    size_t len = 0;

    /*
     * Autoregressive generation loop:
     *   1. Run forward_inference to get next-token logits.
     *   2. Sample from the softmax distribution (temperature-controlled).
     *   3. Stop if BOS (used as EOS) is sampled or max length reached.
     */
    for (size_t pos = 0; pos < BLOCK_SIZE && len < BLOCK_SIZE; pos++) {
      forward_inference(model, token_id, pos, keys, values, cache_len,
                        logits_buf);
      token_id = sample_token(logits_buf, vocab.vocab_size, TEMPERATURE);
      if (token_id == vocab.bos_id)
        break; /* EOS: end of generated sequence */
      name[len++] = (char)vocab.chars[token_id];
    }
    total_infer_tokens += len;
    name[len] = '\0';
    printf("sample %2d: %s\n", sample_idx + 1, name);
  }

  clock_t t_infer_end = clock();
  double infer_sec =
      (double)(t_infer_end - t_infer_start) / (double)CLOCKS_PER_SEC;
  if (infer_sec < 0.001)
    infer_sec = 0.001; /* avoid div-by-zero for very fast inference */
  printf("\nInference: %.2f s | %.1f samples/s | %.1f tokens/s\n", infer_sec,
         (double)NUM_SAMPLES / infer_sec,
         (double)total_infer_tokens / infer_sec);

  /* ---- Cleanup: release all heap memory ---- */
  for (int L = 0; L < N_LAYER; L++) {
    free(keys[L]);
    free(values[L]);
  }
  free(grad_buffer);
  free(m);
  free(v);
  model_free(model);
  free_docs(&docs);
  free(vocab.chars);
  return 0;
}
