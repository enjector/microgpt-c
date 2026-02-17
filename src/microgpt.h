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
 * Algorithm matches ref/microgpt.py (Karpathy).
 *
 * ============================================================================
 *                    MicroGPT-C — Architecture Overview
 * ============================================================================
 *
 * This file declares the public API for a minimal GPT-style language model
 * implemented entirely in C99.  It supports both character-level and word-level
 * tokenisation.
 *
 *
 *  TRANSFORMER ARCHITECTURE (Decoder-Only, GPT-2 Style)
 *  =====================================================
 *
 *  The model predicts the next token given a sequence of past tokens.
 *  It processes one token at a time, caching Key/Value vectors for
 *  efficient autoregressive generation.
 *
 *     Input token_id ─────────────────────────────────────────────┐
 *     Input pos_id   ─────────────────────────────────────────┐   │
 *                                                             │   │
 *                                                             v   v
 *                                                    ┌────────────────────┐
 *                                                    │   x = wte[tok]     │
 *                                                    │     + wpe[pos]     │
 *                                                    │                    │
 *                                                    │  (Embed: lookup    │
 *                                                    │   token & position │
 *                                                    │   vectors, add     │
 *                                                    │   them together)   │
 *                                                    └────────┬───────────┘
 *                                                             │
 *                                                             v
 *                                                    ┌────────────────────┐
 *                                                    │     RMSNorm(x)     │
 *                                                    │                    │
 *                                                    │  x_i / sqrt(       │
 *                                                    │   mean(x^2) + eps) │
 *                                                    └────────┬───────────┘
 *                                                             │
 *                          ┌──────────────────────────────────-┤
 *                          │          x N_LAYER times          │
 *                          │   ┌───────────────────────────────┤
 *                          │   │                               v
 *                          │   │                      ┌────────────────────┐
 *                          │   │                      │  Multi-Head Causal │
 *                          │   │                      │    Attention       │
 *                          │   │                      │                    │
 *                          │   │                      │  Q = Wq @ x_norm  │
 *                          │   │                      │  K = Wk @ x_norm  │
 *                          │   │                      │  V = Wv @ x_norm  │
 *                          │   │                      │                    │
 *                          │   │                      │  Split into heads  │
 *                          │   │                      │  Score = Q·K^T /   │
 *                          │   │                      │         sqrt(d_k)  │
 *                          │   │                      │  Mask future pos   │
 *                          │   │                      │  Attn = softmax    │
 *                          │   │                      │  Out = Attn · V    │
 *                          │   │                      │  Concat heads      │
 *                          │   │                      │  y = Wo @ concat   │
 *                          │   │                      └────────┬───────────┘
 *                          │   │                               │
 *                          │   │       ┌─────────────────┐     │
 *                          │   │       │  + (residual)   │◄────┘
 *                          │   │       └────────┬────────┘
 *                          │   │                │
 *                          │   │                v
 *                          │   │       ┌────────────────────┐
 *                          │   │       │     RMSNorm        │
 *                          │   │       └────────┬───────────┘
 *                          │   │                │
 *                          │   │                v
 *                          │   │       ┌────────────────────┐
 *                          │   │       │   MLP (2-layer)    │
 *                          │   │       │                    │
 *                          │   │       │ h = ReLU(fc1 @ x)  │
 *                          │   │       │ y = fc2 @ h        │
 *                          │   │       │                    │
 *                          │   │       │ (fc1 expands to    │
 *                          │   │       │  4× width, fc2     │
 *                          │   │       │  projects back)    │
 *                          │   │       └────────┬───────────┘
 *                          │   │                │
 *                          │   │       ┌─────────────────┐
 *                          │   └──────►│  + (residual)   │
 *                          │           └────────┬────────┘
 *                          │                    │
 *                          └────────────────────┘  (loop back for next layer)
 *                                               │
 *                                               v
 *                                      ┌────────────────────┐
 *                                      │   lm_head          │
 *                                      │                    │
 *                                      │  logits = W @ x    │
 *                                      │  (project N_EMBD   │
 *                                      │   → vocab_size)    │
 *                                      └────────┬───────────┘
 *                                               │
 *                                               v
 *                                      ┌────────────────────┐
 *                                      │   Softmax          │
 *                                      │                    │
 *                                      │  p_i = exp(z_i)    │
 *                                      │        / Σ exp(z)  │
 *                                      └────────┬───────────┘
 *                                               │
 *                                               v
 *                                      next-token probabilities
 *
 *
 *  TRAINING PIPELINE
 *  =================
 *
 *  The training loop processes one token position at a time:
 *
 *    ┌─────────┐    ┌──────────────┐    ┌────────────┐    ┌───────────┐
 *    │ Tokenize │───►│   Forward     │───►│  CE Loss    │───►│ Backward  │
 *    │ text     │    │   Pass        │    │ -log(p[y])  │    │ (accum    │
 *    │          │    │   (above)     │    │             │    │  grads)   │
 *    └─────────┘    └──────────────┘    └────────────┘    └─────┬─────┘
 *                                                               │
 *                                                               v
 *                                                        ┌───────────┐
 *                                                        │   Adam    │
 *                                                        │ Optimiser │
 *                                                        │           │
 *                                                        │ w -= lr * │
 *                                                        │  m / √v   │
 *                                                        └───────────┘
 *
 *  LR SCHEDULE (Cosine with Warmup):
 *
 *    lr │  /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
 *       │ /                     \
 *       │/                       \
 *       │                         \___
 *       └─────────────────────────────── step
 *         ↑ warmup       cosine decay →
 *
 *  Optional: define QUANTIZATION_INT8 (or QUANTISATION_INT8) to store weights
 *  as 8-bit integers with per-matrix scales (smaller memory, same
 *  training/inference).
 */

#ifndef MICROGPT_H
#define MICROGPT_H

#include <stddef.h>
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
#include <stdint.h> /* int8_t, int32_t for quantised weight storage */
#endif

/* ========================== Model Architecture ========================== */

/*
 * MODEL DIMENSIONS — How everything fits together:
 *
 *   ┌──────────────────────────────────────────────────────────────────────┐
 *   │                         N_EMBD = 32                                 │
 *   │  Every token is represented as a vector of N_EMBD floating-point    │
 *   │  numbers.  This vector flows through every layer of the network.    │
 *   └──────────────────────────────────────────────────────────────────────┘
 *
 *   Multi-Head Attention splits the embedding into N_HEAD independent heads:
 *
 *     N_EMBD = 32
 *     ├───────┼───────┼───────┼───────┤
 *     │ Head0 │ Head1 │ Head2 │ Head3 │    N_HEAD = 4
 *     │ (8d)  │ (8d)  │ (8d)  │ (8d)  │    HEAD_DIM = N_EMBD / N_HEAD = 8
 *     └───────┴───────┴───────┴───────┘
 *
 *     Each head can learn to attend to different types of relationships
 *     (e.g. one head might track the previous word, another tracks the
 *     subject of the sentence).
 *
 *   MLP expands the embedding to a wider hidden layer, then projects back:
 *
 *     N_EMBD = 32 ──fc1──► MLP_DIM = 128 ──fc2──► N_EMBD = 32
 *                    (expand 4×)            (shrink back)
 *
 *     MLP_RATIO controls this expansion factor (default 4×).
 *     The expansion lets the network learn richer non-linear transformations.
 *
 *   Context window:
 *
 *     Token positions:  [0] [1] [2] ... [BLOCK_SIZE-1]
 *                        ◄───── BLOCK_SIZE = 32 ─────►
 *
 *     The model can "see" this many tokens into the past.  Larger values
 *     let the model capture longer-range dependencies but use more memory
 *     and compute (the KV cache grows linearly with BLOCK_SIZE).
 *
 *
 * All defines are wrapped in #ifndef so they can be overridden via compiler
 * flags (-DN_EMBD=64) or before #include, allowing different demo binaries
 * to use different model configurations from the same engine code.
 */
#ifndef N_EMBD
#define N_EMBD 16
#endif
#ifndef N_HEAD
#define N_HEAD 4
#endif
#ifndef N_LAYER
#define N_LAYER 1
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif
#define HEAD_DIM (N_EMBD / N_HEAD)
#ifndef MLP_RATIO
#define MLP_RATIO 4
#endif
#ifndef MLP_DIM
#define MLP_DIM (N_EMBD * MLP_RATIO)
#endif

/* ======================== Training Hyperparameters ======================== */

/*
 * NUM_STEPS     - Total number of optimiser steps (one doc per step).
 * LEARNING_RATE - Peak learning rate for Adam; cosine-annealed with warmup.
 * WARMUP_STEPS  - Number of linear-warmup steps before cosine decay begins.
 * BATCH_SIZE    - Number of documents to accumulate gradients over per step.
 * BETA1, BETA2  - Exponential decay rates for Adam's first and second moment
 *                 estimates.
 * EPS_ADAM      - Small constant added to the denominator in Adam for
 *                 numerical stability.
 * NUM_SAMPLES   - Number of names to generate during the inference demo.
 * TEMPERATURE   - Softmax temperature used during sampling; lower values
 *                 produce more deterministic output.
 */
#ifndef NUM_STEPS
#define NUM_STEPS 1000
#endif
#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01
#endif
#ifndef WARMUP_STEPS
#define WARMUP_STEPS (NUM_STEPS / 10)
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif
#ifndef BETA1
#define BETA1 0.85
#endif
#ifndef BETA2
#define BETA2 0.99
#endif
#ifndef EPS_ADAM
#define EPS_ADAM 1e-8
#endif
#ifndef NUM_SAMPLES
#define NUM_SAMPLES 20
#endif
#ifndef TEMPERATURE
#define TEMPERATURE 0.5
#endif
#ifndef INIT_STD
#define INIT_STD 0.08
#endif

/* ============================ Data Limits ================================ */

/*
 * MAX_VOCAB   - Maximum supported vocabulary size (256 byte values + 1 BOS).
 * MAX_DOCS    - Upper bound on the number of lines (documents) loaded from
 *               the input file.
 * MAX_DOC_LEN - Maximum character length of a single document (line).
 */
#ifndef MAX_VOCAB
#define MAX_VOCAB 257
#endif
#ifndef MAX_DOCS
#define MAX_DOCS 50000
#endif
#ifndef MAX_DOC_LEN
#define MAX_DOC_LEN 64
#endif

/* ============================== Data Types =============================== */

/*
 * Docs - Holds the raw file data and a line-indexed view into it.
 *   data     - Heap-allocated buffer containing the entire input file.
 *   lines    - Array of pointers, each pointing to the start of a line
 *              inside 'data' (not nul-terminated; use doc_lens for length).
 *   num_docs - Number of lines (documents) successfully parsed.
 *   doc_lens - Parallel array giving the byte length of each line.
 */
typedef struct {
  char *data;
  char **lines;
  size_t num_docs;
  size_t *doc_lens;
} Docs;

/*
 * Vocab - Character-level vocabulary built from the training corpus.
 *   chars      - Sorted array of unique byte values found in the corpus.
 *   vocab_size - Number of unique characters + 1 (the extra slot is BOS).
 *   bos_id     - Token ID used as the beginning-of-sequence marker; equals
 *                vocab_size - 1 (i.e. the last slot in the embedding table).
 */
typedef struct {
  unsigned char *chars;
  size_t vocab_size;
  size_t bos_id;
} Vocab;

/*
 * Model - Opaque handle to the Transformer model.
 *         Internal layout depends on whether INT8 quantisation is enabled.
 *         See microgpt.c for the full struct definition.
 */
typedef struct Model Model;

/* ======================== Data Loading & Tokenisation ==================== */

/*
 * load_docs  - Read a text file into 'docs'.  Each line becomes one document.
 *              Returns 0 on success, -1 on failure.
 *              The file must be <= 50 MiB.
 */
int load_docs(const char *path, Docs *docs);

/*
 * free_docs  - Release all heap memory owned by 'docs' and zero the struct.
 */
void free_docs(Docs *docs);

/*
 * build_vocab - Scan all documents and collect the unique characters into a
 *               sorted vocabulary.  Sets vocab->bos_id to the last index.
 */
void build_vocab(const Docs *docs, Vocab *vocab);

/*
 * tokenize   - Convert a raw character string into a sequence of token IDs.
 *              Prepends a BOS token, maps each character via the vocabulary,
 *              and appends a trailing BOS (as an EOS sentinel) if space allows.
 *              Returns the total number of token IDs written to 'ids'.
 *
 *   doc      - Pointer to the raw character data (not nul-terminated).
 *   doc_len  - Number of bytes in 'doc'.
 *   vocab    - Vocabulary used for character-to-ID lookup.
 *   ids      - Output buffer; must be at least 'max_len' elements.
 *   max_len  - Capacity of 'ids'.
 */
size_t tokenize(const char *doc, size_t doc_len, const Vocab *vocab,
                size_t *ids, size_t max_len);

/* ========================== Model Lifecycle =============================== */

/*
 * model_create    - Allocate and randomly initialise a Transformer model
 *                   with the given vocabulary size.  Weights are drawn from
 *                   N(0, 0.08²).  Returns NULL on allocation failure.
 */
Model *model_create(size_t vocab_size);

/*
 * model_free      - Free all weight buffers and the Model struct itself.
 */
void model_free(Model *model);

/*
 * model_num_params - Return the total number of scalar parameters in the model
 *                    (wte + wpe + lm_head + all per-layer attention & MLP
 * weights).
 */
size_t model_num_params(const Model *model);

/* ========================= Checkpointing (fp64) ========================== */

/*
 * model_save / model_load - Binary serialisation of all weights as fp64.
 *   model_save writes: [vocab_size (size_t)] [wte] [wpe] [lm_head] [per-layer
 * weights]. model_load reads the same format and returns a newly allocated
 * Model, or NULL on error.  Disabled (returns -1 / NULL) when INT8 quantisation
 *   is active.
 */
int model_save(const Model *model, const char *path);
Model *model_load(const char *path, size_t vocab_size);

/* ======================== Training Checkpoints =========================== */

/*
 * checkpoint_save - Save a full training checkpoint: model weights, Adam
 *                   optimizer state (m, v), and current step.  Enables
 *                   fully resumable training without loss of momentum.
 *
 *   model  - The trained model.
 *   m      - Adam first-moment buffer (length = model_num_params).
 *   v      - Adam second-moment buffer (length = model_num_params).
 *   step   - Current training step (0-indexed).
 *   path   - Output file path.
 *
 *   Returns 0 on success, -1 on failure.
 *   Disabled (returns -1) when INT8 quantisation is active.
 */
int checkpoint_save(const Model *model, const double *m, const double *v,
                    int step, const char *path);

/*
 * checkpoint_load - Load a training checkpoint.  Allocates and returns a new
 *                   Model, fills the provided m and v buffers, and writes
 *                   the saved step to *step_out.
 *
 *   path       - Input file path.
 *   vocab_size - Expected vocabulary size (must match the saved model).
 *   m          - Pre-allocated buffer for Adam first moments.
 *   v          - Pre-allocated buffer for Adam second moments.
 *   step_out   - Receives the training step stored in the checkpoint.
 *
 *   Returns a newly allocated Model on success, NULL on failure.
 *   Disabled (returns NULL) when INT8 quantisation is active.
 */
Model *checkpoint_load(const char *path, size_t vocab_size, double *m,
                       double *v, int *step_out);

/* ======================== Training (Forward + Backward) =================== */

/*
 * forward_backward_one - Run one position through the full Transformer,
 *   compute cross-entropy loss against 'target_id', and accumulate gradients
 *   into 'grad_buffer'.
 *
 *   token_id    - Input token index for this position.
 *   pos_id      - Positional index (0-based).
 *   target_id   - Ground-truth next-token ID for computing the loss.
 *   keys/values - Per-layer KV cache arrays; each keys[L] and values[L]
 *                 must have capacity for BLOCK_SIZE * N_EMBD doubles.
 *                 The function appends the current K and V vectors at
 *                 cache_len[L], then increments cache_len[L].
 *   cache_len   - Array of N_LAYER counters tracking how many positions
 *                 have been cached so far per layer.
 *   grad_buffer - Flat buffer of length model_num_params(); gradients are
 *                 *accumulated* (not overwritten) — caller must zero it
 *                 at the start of each training step.
 *
 *   Returns the cross-entropy loss for this single position.
 */
double forward_backward_one(const Model *model, size_t token_id, size_t pos_id,
                            size_t target_id, double **keys, double **values,
                            size_t *cache_len, double *grad_buffer);

/* ======================== Optimiser (Adam) ================================ */

/*
 * adam_step - Perform one Adam optimiser update on all model parameters.
 *
 *   grads - Gradient buffer (same layout as model parameters).
 *   m     - First-moment (mean) estimates; same size as 'grads'.
 *   v     - Second-moment (variance) estimates; same size as 'grads'.
 *   step  - Current training step (0-indexed); used for bias correction
 *           and learning-rate linear decay.
 *
 *   For INT8 models, Adam updates the fp64 master copy and then requantises
 *   all weight matrices back to int8 with fresh per-matrix scales.
 */
void adam_step(Model *model, const double *grads, double *m, double *v,
               int step);

/* ======================== Inference / Sampling ============================ */

/*
 * sample_token - Draw a token from a categorical distribution defined by
 *   raw logits, using softmax with the given temperature.
 *
 *   logits      - Array of raw (un-normalised) scores; length = vocab_size.
 *   vocab_size  - Number of classes.
 *   temperature - Sampling temperature (>0).  Lower = more greedy.
 *
 *   Returns the sampled token ID.
 */
size_t sample_token(const double *logits, size_t vocab_size,
                    double temperature);

/*
 * forward_inference - Inference-only forward pass (no loss computation, no
 *   gradient accumulation).  Appends the new KV entries to the cache and
 *   writes the raw next-token logits into 'logits_out'.
 *
 *   Parameters are the same as forward_backward_one, minus target_id and
 *   grad_buffer.  'logits_out' must have space for vocab_size doubles.
 */
void forward_inference(const Model *model, size_t token_id, size_t pos_id,
                       double **keys, double **values, size_t *cache_len,
                       double *logits_out);

/* =========================== Utility ===================================== */

/*
 * seed_rng - Seed the internal linear-congruential PRNG used by
 *            rand_gauss() and rand_u() for reproducible runs.
 */
void seed_rng(unsigned int seed);

/*
 * load_file - Read an entire file into a heap-allocated buffer.
 *             Sets *out_len to the file size in bytes.  The returned buffer
 *             is nul-terminated for convenience.
 *             Returns NULL on failure.  Caller must free() the result.
 */
char *load_file(const char *path, size_t *out_len);

/* ================== Word-Level Tokenisation ============================== */

/*
 * Word-level tokenisation splits text on whitespace, treating newlines as
 * separate tokens (preserving verse / paragraph structure).  A frequency-
 * based vocabulary keeps the top-N most common words; everything else maps
 * to an <unk> token.
 *
 * Typical usage:
 *   1. char *text = load_file("corpus.txt", &len);
 *   2. WordVocab wv;  build_word_vocab(text, len, max_words, &wv);
 *   3. size_t *ids = malloc(len * sizeof *ids);
 *      size_t n = tokenize_words(text, len, &wv, ids, len);
 *   4. Train / infer using ids[] and wv.vocab_size.
 *   5. free_word_vocab(&wv);
 */

#define MAX_WORD_LEN 48 /* max characters per word token */

/*
 * WordVocab - Word-level vocabulary built from frequency analysis.
 *   words      - Array of word strings (heap-allocated), indexed by token ID.
 *   vocab_size - Total tokens: kept_words + <unk> + \n + <bos>.
 *   unk_id     - Token ID for unknown (out-of-vocabulary) words.
 *   newline_id - Token ID for the newline character.
 *   bos_id     - Token ID for beginning-of-sequence marker.
 */
typedef struct {
  char **words;      /* words[id] -> string for that token */
  size_t vocab_size; /* total number of tokens */
  size_t num_words;  /* number of real (non-special) word tokens */
  size_t unk_id;
  size_t newline_id;
  size_t bos_id;
  /* Hash table for O(1) word_to_id lookup (populated by build_word_vocab) */
  char **ht_keys; /* hash table keys (word strings, not owned)          */
  size_t *ht_ids; /* hash table values (token IDs)                      */
  size_t ht_cap;  /* hash table capacity                                */
} WordVocab;

/*
 * build_word_vocab - Scan text, count word frequencies, keep the top
 *                    'max_words' most common words.  Assigns token IDs:
 *                      [0..N-1] = top words
 *                      N        = <unk>
 *                      N+1      = newline
 *                      N+2      = <bos>
 *                    Returns 0 on success, -1 on allocation failure.
 */
int build_word_vocab(const char *text, size_t text_len, size_t max_words,
                     WordVocab *wv);

/*
 * free_word_vocab - Release all heap memory owned by a WordVocab.
 */
void free_word_vocab(WordVocab *wv);

/*
 * word_to_id - Look up a word's token ID.  Returns wv->unk_id if the word
 *              is not in the vocabulary.
 */
size_t word_to_id(const WordVocab *wv, const char *word);

/*
 * tokenize_words - Tokenize text into word token IDs.
 *   Splits on whitespace; newlines become wv->newline_id tokens.
 *   Returns the number of token IDs written to 'ids'.
 *
 *   text       - Input text buffer.
 *   text_len   - Length in bytes.
 *   wv         - Word vocabulary.
 *   ids        - Output buffer (caller-allocated).
 *   max_tokens - Capacity of 'ids'.
 */
size_t tokenize_words(const char *text, size_t text_len, const WordVocab *wv,
                      size_t *ids, size_t max_tokens);

#endif
