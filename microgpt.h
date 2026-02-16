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
 * This file declares the public API for a minimal GPT-style character-level
 * language model implemented entirely in C99.  The architecture follows the
 * standard decoder-only Transformer design:
 *
 *   Token + Positional Embedding
 *       -> N_LAYER x (RMSNorm -> Multi-Head Self-Attention -> Residual
 *                      -> RMSNorm -> MLP [fc1 -> ReLU -> fc2] -> Residual)
 *       -> Linear (lm_head) -> Softmax -> next-token probabilities
 *
 * Training uses a per-position forward+backward pass with cross-entropy loss
 * and an Adam optimiser with linear learning-rate warm-down.
 *
 * Optional: define QUANTIZATION_INT8 (or QUANTISATION_INT8) to store weights
 * as 8-bit integers with per-matrix scales (smaller memory, same
 * training/inference).
 */

#ifndef MICROGPT_H
#define MICROGPT_H

#include <stddef.h>
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
#include <stdint.h> /* int8_t, int32_t for quantised weight storage */
#endif

/* ========================== Model Architecture ========================== */

/*
 * N_EMBD   - Embedding dimension (width of every hidden vector).
 * N_HEAD   - Number of attention heads.  Each head operates on HEAD_DIM dims.
 * N_LAYER  - Number of Transformer blocks stacked sequentially.
 * BLOCK_SIZE - Maximum context length (number of token positions the model
 *              can attend to).  Also doubles as the KV-cache capacity.
 * HEAD_DIM - Per-head dimension = N_EMBD / N_HEAD.
 * MLP_RATIO - Expansion ratio for the feed-forward (MLP) block.
 * MLP_DIM  - Hidden dimension inside the MLP = N_EMBD * MLP_RATIO.
 */
#define N_EMBD 16
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 16
#define HEAD_DIM (N_EMBD / N_HEAD)
#define MLP_RATIO 4
#define MLP_DIM (N_EMBD * MLP_RATIO)

/* ======================== Training Hyperparameters ======================== */

/*
 * NUM_STEPS     - Total number of optimiser steps (one doc per step).
 * LEARNING_RATE - Peak learning rate for Adam; linearly decays to 0 over
 *                 NUM_STEPS.
 * BETA1, BETA2  - Exponential decay rates for Adam's first and second moment
 *                 estimates.
 * EPS_ADAM      - Small constant added to the denominator in Adam for
 *                 numerical stability.
 * NUM_SAMPLES   - Number of names to generate during the inference demo.
 * TEMPERATURE   - Softmax temperature used during sampling; lower values
 *                 produce more deterministic output.
 */
#define NUM_STEPS 1000
#define LEARNING_RATE 0.01
#define BETA1 0.85
#define BETA2 0.99
#define EPS_ADAM 1e-8
#define NUM_SAMPLES 20
#define TEMPERATURE 0.5

/* ============================ Data Limits ================================ */

/*
 * MAX_VOCAB   - Maximum supported vocabulary size (256 byte values + 1 BOS).
 * MAX_DOCS    - Upper bound on the number of lines (documents) loaded from
 *               the input file.
 * MAX_DOC_LEN - Maximum character length of a single document (line).
 */
#define MAX_VOCAB 257
#define MAX_DOCS 50000
#define MAX_DOC_LEN 64

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

#endif
