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
#include <stdio.h> /* for printf in microgpt_print_config */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
#include <stdint.h> /* int8_t, int32_t for quantised weight storage */
#endif

/* ======================== Scalar Precision ================================ */
/*
 * scalar_t — compile-time toggle between float (32-bit) and double (64-bit).
 *
 * Define MICROGPT_USE_FLOAT (via CMake or -D flag) to use single precision.
 * Float roughly doubles SIMD throughput on ARM NEON (4 vs 2 elements per
 * 128-bit register) and halves memory bandwidth, at the cost of ~7 decimal
 * digits of precision vs ~15 for double.
 *
 * All weight matrices, activations, gradients, and KV cache entries use
 * scalar_t.  Hyperparameters (learning rate, betas, epsilon) remain double
 * for optimizer stability.
 */
#ifdef MICROGPT_USE_FLOAT
typedef float scalar_t;
#define SC_FMT "f"
#define SC_SCAN "f"
#define M_EXP expf
#define M_LOG logf
#define M_SQRT sqrtf
#define M_POW powf
#define M_FABS fabsf
#define M_SIN sinf
#define M_COS cosf
#ifdef MICROGPT_BLAS
#define CBLAS_GEMV cblas_sgemv
#define CBLAS_GER cblas_sger
#endif
#else
typedef double scalar_t;
#define SC_FMT "lf"
#define SC_SCAN "lf"
#define M_EXP exp
#define M_LOG log
#define M_SQRT sqrt
#define M_POW pow
#define M_FABS fabs
#define M_SIN sin
#define M_COS cos
#ifdef MICROGPT_BLAS
#define CBLAS_GEMV cblas_dgemv
#define CBLAS_GER cblas_dger
#endif
#endif

/* ======================== Runtime Configuration =========================== */
/*
 * MicrogptConfig — Holds all hyperparameters for display (config banner) and
 * API convenience.  The HOT inner loops read compile-time #define macros
 * (N_EMBD, N_LAYER, etc.) directly so the compiler can constant-fold and
 * unroll.  The struct's defaults are populated from those same macros by
 * microgpt_default_config().
 *
 * Use microgpt_default_config() to get the defaults, then override per-demo.
 */
typedef struct {
  /* Model Architecture */
  int n_embd;     /* embedding dimension                          */
  int n_head;     /* number of attention heads                    */
  int n_layer;    /* number of transformer layers                 */
  int block_size; /* maximum context window (KV cache length)     */
  int mlp_dim;    /* MLP hidden dimension (typically 4 × n_embd) */

  /* Training */
  int num_steps;        /* total training steps                    */
  double learning_rate; /* peak learning rate for Adam             */
  int batch_size;       /* documents per gradient accumulation     */
  int warmup_steps;     /* linear warmup before cosine decay       */
  double temperature;   /* sampling temperature for inference      */

  /* Data Limits */
  int max_vocab;   /* max vocabulary size                          */
  int max_docs;    /* max number of documents (lines) to load     */
  int max_doc_len; /* max characters per document                 */
} MicrogptConfig;

/*
 * Computed field: head_dim = n_embd / n_head.  Not stored in the struct;
 * use the macro below for convenience.
 */
#define MICROGPT_HEAD_DIM(cfg) ((cfg)->n_embd / (cfg)->n_head)

/* ======================== Compile-Time Constants ========================= */
/*
 * Architecture and training constants.  Demos and CMake can override any of
 * these with -DN_EMBD=128 etc.  The hot inner loops in microgpt.c read
 * these macros directly so the compiler can constant-fold and unroll.
 * microgpt_default_config() also reads from them for the config banner.
 */

/* ---- Architecture ---- */
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
#ifndef MLP_DIM
#define MLP_DIM 64
#endif

/* ---- Training ---- */
#ifndef NUM_STEPS
#define NUM_STEPS 1000
#endif
#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif
#ifndef WARMUP_STEPS
#define WARMUP_STEPS 100
#endif
#ifndef TEMPERATURE
#define TEMPERATURE 0.5
#endif

/* ---- Data limits ---- */
#ifndef MAX_VOCAB
#define MAX_VOCAB 257
#endif
#ifndef MAX_DOCS
#define MAX_DOCS 50000
#endif
#ifndef MAX_DOC_LEN
#define MAX_DOC_LEN 64
#endif

/* ---- Compile-time helpers (MSVC has no VLAs) ---- */
#define _MGPT_MAX(a, b) ((a) > (b) ? (a) : (b))
#define _MGPT_MAX_W _MGPT_MAX((MLP_DIM) * (N_EMBD), (N_EMBD) * (N_EMBD))

/* ---- Optimizer ---- */
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
#ifndef INIT_STD
#define INIT_STD 0.08
#endif

/*
 * microgpt_default_config - Return a config populated with sensible defaults.
 *   These match the compile-time constants above.
 */
static inline MicrogptConfig microgpt_default_config(void) {
  MicrogptConfig cfg;
  cfg.n_embd = N_EMBD;
  cfg.n_head = N_HEAD;
  cfg.n_layer = N_LAYER;
  cfg.block_size = BLOCK_SIZE;
  cfg.mlp_dim = MLP_DIM;

  cfg.num_steps = NUM_STEPS;
  cfg.learning_rate = LEARNING_RATE;
  cfg.batch_size = BATCH_SIZE;
  cfg.warmup_steps = WARMUP_STEPS;
  cfg.temperature = TEMPERATURE;

  cfg.max_vocab = MAX_VOCAB;
  cfg.max_docs = MAX_DOCS;
  cfg.max_doc_len = MAX_DOC_LEN;
  return cfg;
}

/* ========================= Configuration Banner ========================== */

/*
 * microgpt_print_config - Prints all configuration values from a config struct.
 *
 * Call at the top of main() in any demo to display the full runtime
 * configuration — model hyperparameters, training settings, data limits,
 * and feature switches.
 */
static inline void microgpt_print_config(const char *demo_name,
                                         const MicrogptConfig *cfg) {
  printf("================================================================\n");
  if (demo_name)
    printf("  %s\n", demo_name);
  printf("================================================================\n");
  printf("\n");

  printf("  [Model Architecture]\n");
  printf("    n_embd      = %d\n", cfg->n_embd);
  printf("    n_head       = %d\n", cfg->n_head);
  printf("    head_dim     = %d\n", MICROGPT_HEAD_DIM(cfg));
  printf("    mlp_dim      = %d\n", cfg->mlp_dim);
  printf("    n_layer      = %d\n", cfg->n_layer);
  printf("    block_size   = %d\n", cfg->block_size);
  printf("\n");

  printf("  [Training]\n");
  printf("    num_steps    = %d\n", cfg->num_steps);
  printf("    learning_rate= %.6f\n", cfg->learning_rate);
  printf("    batch_size   = %d\n", cfg->batch_size);
  printf("    warmup_steps = %d\n", cfg->warmup_steps);
  printf("    temperature  = %.4f\n", cfg->temperature);
  printf("\n");

  printf("  [Data Limits]\n");
  printf("    max_vocab    = %d\n", cfg->max_vocab);
  printf("    max_docs     = %d\n", cfg->max_docs);
  printf("    max_doc_len  = %d\n", cfg->max_doc_len);
  printf("\n");

  printf("  [Feature Switches]\n");
#ifdef MICROGPT_USE_FLOAT
  printf("    scalar_t     = float  (32-bit)\n");
#else
  printf("    scalar_t     = double (64-bit)\n");
#endif
#ifdef MICROGPT_SIMD
  printf("    SIMD         = ON\n");
#else
  printf("    SIMD         = OFF\n");
#endif
#ifdef QUANTIZATION_INT8
  printf("    INT8 quant   = ON\n");
#else
  printf("    INT8 quant   = OFF\n");
#endif
#ifdef MICROGPT_METAL
  printf("    Metal GPU    = ON\n");
#else
  printf("    Metal GPU    = OFF\n");
#endif
#ifdef MICROGPT_BLAS
  printf("    BLAS         = ON\n");
#else
  printf("    BLAS         = OFF\n");
#endif
#ifdef MICROGPT_PAGED_KV
  printf("    Paged KV     = ON\n");
#else
  printf("    Paged KV     = OFF\n");
#endif
#ifdef MICROGPT_HEAD_PARALLEL
  printf("    Head Parallel= ON\n");
#else
  printf("    Head Parallel= OFF\n");
#endif
  printf("\n");
  printf(
      "================================================================\n\n");
}

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
 *         Contains a copy of the MicrogptConfig used to create it.
 *         Internal layout depends on whether INT8 quantisation is enabled.
 *         See microgpt.c for the full struct definition.
 */
typedef struct Model Model;

/*
 * model_config - Return a pointer to the config stored inside a model.
 *   Allows callers to read the model's architecture without knowing
 *   the internal struct layout.
 */
const MicrogptConfig *model_config(const Model *model);

/* ======================== Data Loading & Tokenisation ==================== */

/*
 * load_docs  - Read a text file into 'docs'.  Each line becomes one document.
 *              Returns 0 on success, -1 on failure.
 *              The file must be <= 50 MiB.
 *              max_docs limits the number of lines loaded.
 */
int load_docs(const char *path, Docs *docs, int max_docs);

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
 *                   with the given vocabulary size and configuration.
 *                   Weights are drawn from N(0, 0.08²).
 *                   Returns NULL on allocation failure.
 */
Model *model_create(size_t vocab_size, const MicrogptConfig *cfg);

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
 *   model_save writes: [config] [vocab_size (size_t)] [wte] [wpe] [lm_head]
 *     [per-layer weights].
 *   model_load reads the same format and returns a newly allocated
 *     Model, or NULL on error.
 *   Disabled (returns -1 / NULL) when INT8 quantisation is active.
 */
int model_save(const Model *model, const char *path);
Model *model_load(const char *path, size_t vocab_size,
                  const MicrogptConfig *cfg);

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
int checkpoint_save(const Model *model, const scalar_t *m, const scalar_t *v,
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
Model *checkpoint_load(const char *path, size_t vocab_size,
                       const MicrogptConfig *cfg, scalar_t *m, scalar_t *v,
                       int *step_out);

/* ======================== Paged KV Cache ================================== */
#ifdef MICROGPT_PAGED_KV

/*
 * Demand-paged KV cache — allocates memory in fixed-size pages as the
 * sequence grows, rather than pre-allocating block_size × n_embd upfront.
 *
 * Each page holds KV_PAGE_SIZE positions × n_embd doubles.
 * Pages are allocated on first access and reused on reset.
 */
#ifndef KV_PAGE_SIZE
#define KV_PAGE_SIZE 64
#endif

typedef struct {
  scalar_t *data; /* KV_PAGE_SIZE × n_embd doubles */
} KVPage;

typedef struct {
  KVPage **pages;  /* page table: pages[page_idx] */
  size_t n_pages;  /* number of allocated pages */
  size_t capacity; /* max page table slots */
  size_t len;      /* number of positions stored */
  int n_embd;      /* embedding dimension for this cache */
} PagedKVCache;

/*
 * paged_kv_create - Allocate a page table for up to max_positions entries.
 *   No data pages are allocated until paged_kv_append is called.
 */
PagedKVCache *paged_kv_create(size_t max_positions, int n_embd);

/*
 * paged_kv_free - Free all pages and the page table.
 */
void paged_kv_free(PagedKVCache *c);

/*
 * paged_kv_reset - Reset the position counter to zero.
 *   Pages are retained (not freed) for reuse.
 */
void paged_kv_reset(PagedKVCache *c);

/*
 * paged_kv_append - Append one position.  Returns a pointer to an n_embd-
 *   sized slot where the caller should write the K or V vector.
 *   Allocates a new page if needed.
 */
scalar_t *paged_kv_append(PagedKVCache *c);

/*
 * paged_kv_get - Read access to position 'pos'.
 *   Returns pointer to the n_embd-sized slot.
 */
const scalar_t *paged_kv_get(const PagedKVCache *c, size_t pos);

#endif /* MICROGPT_PAGED_KV */

/* ======================== KV Cache Helpers =================================
 */

/*
 * Portable KV cache allocation — works with both flat and paged modes.
 * Callers should use these instead of raw malloc/calloc for KV arrays.
 *
 * When MICROGPT_PAGED_KV is active the returned pointer is actually a
 * PagedKVCache* cast to scalar_t*.  The engine's KV_WRITE/KV_READ macros
 * know how to interpret either representation.
 */
scalar_t *kv_cache_alloc(const MicrogptConfig *cfg);
void kv_cache_free(scalar_t *kv);
void kv_cache_reset(scalar_t *kv, const MicrogptConfig *cfg);

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
 *                 must have capacity for block_size * n_embd doubles.
 *                 The function appends the current K and V vectors at
 *                 cache_len[L], then increments cache_len[L].
 *   cache_len   - Array of n_layer counters tracking how many positions
 *                 have been cached so far per layer.
 *   grad_buffer - Flat buffer of length model_num_params(); gradients are
 *                 *accumulated* (not overwritten) — caller must zero it
 *                 at the start of each training step.
 *
 *   Returns the cross-entropy loss for this single position.
 */
scalar_t forward_backward_one(const Model *model, size_t token_id,
                              size_t pos_id, size_t target_id, scalar_t **keys,
                              scalar_t **values, size_t *cache_len,
                              scalar_t *grad_buffer);

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
void adam_step(Model *model, const scalar_t *grads, scalar_t *m, scalar_t *v,
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
size_t sample_token(const scalar_t *logits, size_t vocab_size,
                    scalar_t temperature);

/*
 * forward_inference - Inference-only forward pass (no loss computation, no
 *   gradient accumulation).  Appends the new KV entries to the cache and
 *   writes the raw next-token logits into 'logits_out'.
 *
 *   Parameters are the same as forward_backward_one, minus target_id and
 *   grad_buffer.  'logits_out' must have space for vocab_size doubles.
 */
void forward_inference(const Model *model, size_t token_id, size_t pos_id,
                       scalar_t **keys, scalar_t **values, size_t *cache_len,
                       scalar_t *logits_out);

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

/* ======================== Training Helpers ================================ */

/*
 * shuffle_docs - Fisher-Yates in-place shuffle of the document list.
 *   Randomises document order to prevent the model from memorising
 *   sequential patterns in the dataset.
 */
void shuffle_docs(Docs *docs);

/*
 * rand_u - Return a uniform random scalar_t in [0, 1).
 *   Uses the internal LCG seeded by seed_rng().
 */
scalar_t rand_u(void);

/*
 * TrainWorker - Per-thread work descriptor for batched training.
 *   Each thread processes a slice of the batch [batch_start, batch_end)
 *   and accumulates gradients + loss into its own buffers.
 *
 *   All arrays are dynamically allocated based on the model's config.
 */
typedef struct {
  const Model *model;
  const Docs *docs;
  const Vocab *vocab;
  scalar_t *grads;
  scalar_t **keys;   /* keys[n_layer], each a KV cache buffer */
  scalar_t **values; /* values[n_layer], each a KV cache buffer */
  size_t *cache_len; /* cache_len[n_layer] */
  size_t *token_buf; /* token_buf[block_size + 2] */
  int batch_start;
  int batch_end;
  scalar_t loss;
  size_t positions;
  unsigned int rng_seed;
} TrainWorker;

/*
 * train_worker_run - Thread entry point for batched training.
 *   Processes the documents assigned by [batch_start, batch_end),
 *   tokenises each, runs forward+backward, and accumulates gradients
 *   and loss into the TrainWorker's own buffers.
 *   Cast arg to TrainWorker*.
 */
void *train_worker_run(void *arg);

/* ======================== Portable Threading ============================== */
/*
 * Cross-platform threading abstraction (previously microgpt_thread.h).
 *
 * Provides a minimal API over:
 *   - Thread creation / join  (pthread on POSIX, Win32 on Windows)
 *   - CPU core count detection (sysconf / GetSystemInfo)
 *   - Default thread count recommendation
 *
 * On Windows, the user-facing function signature void*(*fn)(void*) is
 * wrapped into the __stdcall unsigned(void*) convention that
 * _beginthreadex expects, using a trampoline struct.  On POSIX the
 * trampoline is a no-op but kept for API uniformity.
 */

#ifdef _WIN32
/* ---- Windows ---- */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>

typedef HANDLE mgpt_thread_t;

/* Win32 thread proc signature: unsigned __stdcall fn(void*) */
typedef unsigned(__stdcall *mgpt_win32_fn)(void *);

/*
 * Trampoline: wraps the portable void*(*fn)(void*) signature into the
 * __stdcall unsigned(void*) convention required by _beginthreadex.
 * Caller must keep the trampoline struct alive until thread completes.
 */
typedef struct {
  void *(*fn)(void *);
  void *arg;
} mgpt_thread_trampoline_t;

static unsigned __stdcall mgpt_thread_trampoline_(void *p) {
  mgpt_thread_trampoline_t *t = (mgpt_thread_trampoline_t *)p;
  t->fn(t->arg);
  return 0;
}

static int mgpt_thread_create(mgpt_thread_t *thread,
                              mgpt_thread_trampoline_t *tramp,
                              void *(*fn)(void *), void *arg) {
  tramp->fn = fn;
  tramp->arg = arg;
  *thread =
      (HANDLE)_beginthreadex(NULL, 0, mgpt_thread_trampoline_, tramp, 0, NULL);
  return (*thread == 0) ? -1 : 0;
}

static int mgpt_thread_join(mgpt_thread_t thread) {
  WaitForSingleObject(thread, INFINITE);
  CloseHandle(thread);
  return 0;
}

static int mgpt_cpu_count(void) {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (int)si.dwNumberOfProcessors;
}

/* rand_r is not available on Windows; provide a simple replacement */
#ifndef HAVE_RAND_R
static unsigned int mgpt_rand_r(unsigned int *seed) {
  *seed = *seed * 1103515245u + 12345u;
  return (*seed >> 16) & 0x7fff;
}
#define rand_r(s) mgpt_rand_r(s)
#endif

/* clock_gettime is not available on older MSVC; provide a monotonic version */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

struct timespec; /* forward decl */

static int mgpt_clock_gettime(int clk_id, struct timespec *tp);

#include <time.h> /* for struct timespec */

static int mgpt_clock_gettime(int clk_id, struct timespec *tp) {
  if (clk_id != CLOCK_MONOTONIC)
    return -1;
  LARGE_INTEGER freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  tp->tv_sec = count.QuadPart / freq.QuadPart;
  tp->tv_nsec =
      (long)(((count.QuadPart % freq.QuadPart) * 1000000000L) / freq.QuadPart);
  return 0;
}
#define clock_gettime(id, tp) mgpt_clock_gettime(id, tp)

#else
/* ---- POSIX (Linux, macOS, etc.) ---- */
#include <pthread.h>
#include <unistd.h>

typedef pthread_t mgpt_thread_t;

/* Dummy trampoline struct — not needed on POSIX, but keeps API uniform */
typedef struct {
  void *(*fn)(void *);
  void *arg;
} mgpt_thread_trampoline_t;

static int mgpt_thread_create(mgpt_thread_t *thread,
                              mgpt_thread_trampoline_t *tramp,
                              void *(*fn)(void *), void *arg) {
  (void)tramp; /* unused on POSIX */
  return pthread_create(thread, NULL, fn, arg);
}

static int mgpt_thread_join(mgpt_thread_t thread) {
  return pthread_join(thread, NULL);
}

static int mgpt_cpu_count(void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? (int)n : 1;
}

#endif /* _WIN32 */

/*
 * mgpt_default_threads — Recommend a thread count for training.
 *   Returns min(cpu_count, batch_size) so we never have idle threads.
 */
static int mgpt_default_threads(int batch_size) {
  int ncpu = mgpt_cpu_count();
  return (ncpu < batch_size) ? ncpu : batch_size;
}

#endif /* MICROGPT_H */
