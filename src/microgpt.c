/*
 * MicroGPT-C - C99 implementation matching Andrej Karpathy's microgpt.py

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
 * ============================================================================
 *                  MicroGPT-C  —  Implementation Guide
 * ============================================================================
 *
 * This file contains the full implementation of a GPT-style language model.
 * Reading top-to-bottom, the code is organised as follows:
 *
 *   SECTION 1: Data Loading & Tokenisation (line ~130)
 *   ─────────────────────────────────────────────────
 *   load_docs()     — Read text file, split into lines ("documents")
 *   build_vocab()   — Collect unique characters, assign numeric IDs
 *   tokenize()      — Convert text → [BOS, char_0, char_1, ..., BOS]
 *
 *   SECTION 2: Model Allocation (line ~270)
 *   ─────────────────────────────────────────
 *   count_params()  — Total scalar count across all weight matrices
 *   model_create()  — Heap-allocate all weights, Gaussian init
 *   model_free()    — Release all memory
 *
 *   SECTION 3: Serialisation (line ~540)
 *   ────────────────────────────────────
 *   model_save/load()      — Binary weight I/O
 *   checkpoint_save/load()  — Weights + Adam state + step counter
 *
 *   SECTION 4: Neural Network Primitives (line ~818)
 *   ────────────────────────────────────────────────
 *   lin_fwd()       — Dense layer: y = W @ x (matrix-vector multiply)
 *   lin_bwd()       — Backward: compute dx, dW from upstream dy
 *   rmsnorm_fwd()   — RMS normalisation: x / M_SQRT(mean(x²) + eps)
 *   rmsnorm_bwd()   — Backward pass for RMSNorm
 *
 *   SECTION 5: Forward + Backward Pass (line ~922)
 *   ──────────────────────────────────────────────
 *   forward_backward_one() — The heart of training.  Runs one token
 *       through the full Transformer, computes cross-entropy loss,
 *       then backpropagates gradients through every layer.
 *
 *   SECTION 6: Inference (line ~1295)
 *   ────────────────────────────────
 *   forward_inference() — Forward-only pass for text generation.
 *       Uses KV cache for O(1) per-token work.
 *
 *   SECTION 7: Adam Optimiser (line ~1456)
 *   ──────────────────────────────────────
 *   adam_step()     — Update weights using accumulated gradients.
 *       Cosine LR schedule with linear warmup.
 *
 *   SECTION 8: Sampling (line ~1675)
 *   ───────────────────────────────
 *   sample_token()  — Temperature-scaled softmax → random draw.
 *
 *   SECTION 9: Word-Level Tokenisation (line ~1744)
 *   ──────────────────────────────────────────────
 *   build_word_vocab() — Frequency-ranked word vocabulary with hash table
 *   word_to_id()       — O(1) hash lookup for word → token ID
 *   tokenize_words()   — Split text on whitespace → token ID sequence
 *
 *
 * MEMORY LAYOUT (flat gradient / Adam buffers)
 * ────────────────────────────────────────────
 * All trainable parameters are stored in per-matrix allocations inside the
 * Model struct.  But the gradient buffer, Adam m[] and v[] arrays, and the
 * INT8 master copy are *flat* arrays with this layout:
 *
 *   ┌──────────┬──────────┬───────────┬─────┬─────┬─────┬─────┬──────┬──────┐
 *   │   wte    │   wpe    │  lm_head  │ wq₀ │ wk₀ │ wv₀ │ wo₀ │ fc1₀ │ fc2₀ │
 *   ├──────────┼──────────┼───────────┼─────┼─────┼─────┼─────┼──────┼──────┤
 *   │ V×E      │ B×E      │   V×E     │ E×E │ E×E │ E×E │ E×E │ M×E  │ E×M  │
 *   └──────────┴──────────┴───────────┴─────┴─────┴─────┴─────┴──────┴──────┘
 *   │◄── global matrices ──────────►│  │◄── repeated × N_LAYER ──────────────►│
 *
 *   V = vocab_size, E = N_EMBD, B = BLOCK_SIZE, M = MLP_DIM
 *
 *
 * BUILD MODES
 * ───────────
 *   FP64 (default)  - all weights stored as scalar_t precision.
 *   INT8 (optional)  - weights stored as int8_t with per-matrix fp64 scales.
 *                       A fp64 master copy is maintained for Adam updates;
 *                       after each step the master is requantised to int8.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MICROGPT_METAL
#include "microgpt_metal.h"
#endif

#ifdef MICROGPT_HEAD_PARALLEL
#include "microgpt_thread.h"
#endif

#ifdef MICROGPT_BLAS
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ---- Cache tiling parameters for lin_fwd / lin_bwd ----
 * TILE_R rows × TILE_C columns = panel that fits comfortably in L1.
 * 32 × 64 × 8B = 16 KB  (M2 Max L1 = 128 KB, leaves room for x/y/dx). */
#ifndef LIN_TILE_R
#define LIN_TILE_R 32
#endif
#ifndef LIN_TILE_C
#define LIN_TILE_C 64
#endif

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
/*
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │      INT8 QUANTISATION MODE  (disabled by default)                 │
 * │                                                                    │
 * │  Enable with:  -DQUANTIZATION_INT8  or  -DQUANTISATION_INT8        │
 * │                                                                    │
 * │  Purpose: Reduce memory usage by ~8× and enable integer SIMD.      │
 * │  Trades tiny accuracy loss for major memory & potential speed wins. │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * SYMMETRIC PER-MATRIX QUANTISATION:
 *
 *   fp64 weight:  [-3.2, 0.5, 1.7, -2.0]   (range: -3.2 .. 1.7)
 *   scale = max(|W|) / 127 = 3.2/127 ≈ 0.0252
 *   int8:         [-127,  20,  67,  -79]    (W / scale, rounded, clamped)
 *
 *   To recover fp64:  W_approx = scale × int8_value
 *
 * TRAINING WORKFLOW (master copy pattern):
 *
 *   ┌──────────┐    requantise    ┌──────────┐    forward     ┌──────┐
 *   │  master  │  ───────────►    │  int8    │  ──────────►   │ loss │
 *   │  (fp64)  │                  │ weights  │  (cheap ints)  └──┬───┘
 *   └────▲─────┘                  └──────────┘                   │
 *        │                                                  backward
 *   Adam update                                                  │
 *        │        ◄──────── gradients (fp64) ◄────────────────────┘
 *
 *   Adam updates the fp64 master copy (full precision needed for
 *   small gradient steps).  After each step the master is requantised
 *   to int8 so the next forward pass uses cheap integer arithmetic.
 *
 * N_SCALES counts the total number of per-matrix scale factors:
 *   3 global (wte, wpe, lm_head) + 6 per layer (wq, wk, wv, wo, fc1, fc2)
 */
#define N_SCALES(nl) (3 + 6 * (nl))
struct Model {
  MicrogptConfig cfg; /* runtime configuration                       */
  size_t vocab_size;  /* number of character tokens + BOS            */
  int8_t *wte;        /* token embedding      [vocab_size × n_embd]  */
  int8_t *wpe;        /* position embedding   [block_size × n_embd]  */
  int8_t *lm_head;    /* output projection    [vocab_size × n_embd]  */
  int8_t **attn_wq;   /* query   weight       [n_embd × n_embd]      */
  int8_t **attn_wk;   /* key     weight       [n_embd × n_embd]      */
  int8_t **attn_wv;   /* value   weight       [n_embd × n_embd]      */
  int8_t **attn_wo;   /* output  weight       [n_embd × n_embd]      */
  int8_t **mlp_fc1;   /* MLP up-projection    [mlp_dim × n_embd]     */
  int8_t **mlp_fc2;   /* MLP down-projection  [n_embd × mlp_dim]     */
  scalar_t *scale;    /* per-matrix quantisation scales               */
  scalar_t *master;   /* fp64 master weights for Adam updates         */
};
#else
/*
 * FP64 Model Layout (default)
 * ---------------------------
 * Every weight matrix is a heap-allocated scalar_t array.
 * Weight layouts follow row-major [output_dim x input_dim] convention so
 * that y = W @ x is computed as: y[j] = sum_i W[j*nin + i] * x[i].
 *
 * Why row-major [output_dim × input_dim]?
 *
 *   Each row of W corresponds to one output neuron.  Computing one output
 *   value means dotting a row of W with the input vector x:
 *
 *     W  (nout × nin)            x  (nin × 1)         y (nout × 1)
 *   ┌────────────────┐        ┌────┐               ┌────┐
 *   │ w00 w01 w02 w03│        │ x0 │               │ y0 │  y0 = w00·x0 + w01·x1
 * + ... │ w10 w11 w12 w13│   @    │ x1 │       =       │ y1 │  y1 = w10·x0 +
 * w11·x1 + ... │ w20 w21 w22 w23│        │ x2 │               │ y2 │  y2 =
 * w20·x0 + w21·x1 + ... └────────────────┘        │ x3 │               └────┘
 *                             └────┘
 *
 *   The key insight: reading W row-by-row is sequential in memory,
 *   which is cache-friendly.  Each row gives us one output element.
 */
struct Model {
  MicrogptConfig cfg; /* runtime configuration                       */
  size_t vocab_size;  /* number of character tokens + BOS             */
  scalar_t *wte;      /* token embedding      [vocab_size × n_embd]  */
  scalar_t *wpe;      /* position embedding   [block_size × n_embd]  */
  scalar_t *lm_head;  /* output projection    [vocab_size × n_embd]  */
  scalar_t **attn_wq; /* query   weight       [n_embd × n_embd]      */
  scalar_t **attn_wk; /* key     weight       [n_embd × n_embd]      */
  scalar_t **attn_wv; /* value   weight       [n_embd × n_embd]      */
  scalar_t **attn_wo; /* output  weight       [n_embd × n_embd]      */
  scalar_t **mlp_fc1; /* MLP up-projection    [mlp_dim × n_embd]     */
  scalar_t **mlp_fc2; /* MLP down-projection  [n_embd × mlp_dim]     */
};
#endif

/* model_config - Return a pointer to the config stored inside a model. */
const MicrogptConfig *model_config(const Model *model) { return &model->cfg; }

/* qsort comparator for unsigned chars — used to sort the vocabulary. */
static int char_cmp(const void *a, const void *b) {
  return *(const unsigned char *)a - *(const unsigned char *)b;
}

/* ---------- Pseudo-Random Number Generator (LCG) ---------- */
/*
 * A simple linear-congruential generator (LCG) following the classic
 * constants from ANSI C's rand().  Deterministic given the same seed,
 * which is essential for reproducibility during weight initialisation
 * and sampling.
 */
static unsigned long rng_state = 42;
void seed_rng(unsigned int seed) { rng_state = (unsigned long)seed; }

/* Return a uniform random scalar_t in [0, 1). */
scalar_t rand_u(void) {
  rng_state = rng_state * 1103515245UL + 12345UL;
  return (scalar_t)((rng_state / 65536UL) % 32768UL) / 32768.0;
}

/*
 * Box-Muller transform: convert two uniform samples into one sample
 * from N(0, 1).  Used for Gaussian weight initialisation.
 */
static scalar_t rand_gauss(void) {
  scalar_t u1 = rand_u(), u2 = rand_u();
  if (u1 < 1e-10)
    u1 = 1e-10; /* clamp to avoid M_LOG(0) */
  return M_SQRT(-2.0 * M_LOG(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/* ========================== Data Loading ================================= */

/*
 * load_docs - Read an entire text file into memory and split it into lines.
 *
 * Each non-empty line becomes one "document" (typically a single name in the
 * Karpathy character-level example).  The raw file bytes are stored in
 * docs->data; docs->lines[] point into this buffer.  Lines are NOT nul-
 * terminated — use docs->doc_lens[] for lengths.
 *
 * Returns 0 on success, -1 on any error (file not found, too large, OOM).
 * Maximum file size is capped at 50 MiB for safety.
 */
int load_docs(const char *path, Docs *docs, int max_docs) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;

  /* Determine file size */
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size <= 0 || size > 50 * 1024 * 1024) {
    fclose(f);
    return -1;
  }

  /* Slurp entire file into a single heap buffer */
  docs->data = (char *)malloc((size_t)size + 1);
  if (!docs->data) {
    fclose(f);
    return -1;
  }
  size_t nread = fread(docs->data, 1, (size_t)size, f);
  docs->data[nread] = '\0'; /* nul-terminate for safe scanning */
  fclose(f);

  /* Allocate line index arrays (up to max_docs entries) */
  docs->lines = (char **)malloc((size_t)max_docs * sizeof(char *));
  docs->doc_lens = (size_t *)malloc((size_t)max_docs * sizeof(size_t));
  if (!docs->lines || !docs->doc_lens) {
    free(docs->data);
    return -1;
  }

  /* Scan buffer and record each non-empty line */
  size_t nd = 0;
  char *p = docs->data;
  while (nd < (size_t)max_docs && *p) {
    char *start = p;
    while (*p && *p != '\r' && *p != '\n')
      p++; /* advance to EOL */
    size_t len = (size_t)(p - start);
    if (len > 0) {
      docs->lines[nd] = start;
      docs->doc_lens[nd] = len;
      nd++;
    }
    if (*p == '\r')
      p++; /* skip CR (Windows line endings) */
    if (*p == '\n')
      p++; /* skip LF */
  }
  docs->num_docs = nd;
  return 0;
}

/* Free all heap memory owned by a Docs struct and zero its fields. */
void free_docs(Docs *docs) {
  free(docs->data);
  free(docs->lines);
  free(docs->doc_lens);
  docs->data = NULL;
  docs->lines = NULL;
  docs->doc_lens = NULL;
  docs->num_docs = 0;
}

/*
 * build_vocab - Collect every unique byte value across all documents,
 *   sort them, and assign contiguous IDs [0 .. n-1].  The BOS/EOS token
 *   is placed at index n (i.e. vocab_size = n + 1).
 */
void build_vocab(const Docs *docs, Vocab *vocab) {
  unsigned char seen[256]; /* bitmap: have we seen byte value i? */
  memset(seen, 0, sizeof(seen));
  for (size_t i = 0; i < docs->num_docs; i++)
    for (size_t j = 0; j < docs->doc_lens[i]; j++)
      seen[(unsigned char)docs->lines[i][j]] = 1;

  /* Count unique characters */
  size_t n = 0;
  for (int i = 0; i < 256; i++)
    if (seen[i])
      n++;

  vocab->chars = (unsigned char *)malloc(n);
  if (!vocab->chars)
    return;
  n = 0;
  for (int i = 0; i < 256; i++)
    if (seen[i])
      vocab->chars[n++] = (unsigned char)i;
  qsort(vocab->chars, n, 1, char_cmp); /* sort for deterministic ordering */
  vocab->vocab_size = n + 1;           /* +1 for the BOS token */
  vocab->bos_id = n;                   /* BOS is the last token ID */
}

/*
 * char_to_id - Linear scan to find the token ID for a given character.
 *   Returns vocab_size (out-of-vocab sentinel) if the character is unknown.
 */
static size_t char_to_id(const Vocab *v, char c) {
  for (size_t i = 0; i < v->vocab_size - 1; i++)
    if (v->chars[i] == (unsigned char)c)
      return i;
  return v->vocab_size;
}

/*
 * tokenize - Convert raw characters into a token-ID sequence.
 *   Format: [BOS] [char_0] [char_1] ... [char_n] [BOS]
 *   The trailing BOS acts as an EOS sentinel so the model learns when to stop.
 *   Returns the number of IDs written to 'ids'.
 */
size_t tokenize(const char *doc, size_t doc_len, const Vocab *vocab,
                size_t *ids, size_t max_len) {
  if (max_len < 2)
    return 0; /* need room for at least BOS + one token */
  size_t k = 0;
  if (k >= max_len)
    return k;
  ids[k++] = vocab->bos_id; /* begin-of-sequence */
  for (size_t i = 0; i < doc_len && k < max_len; i++)
    ids[k++] = char_to_id(vocab, doc[i]);
  if (k < max_len)
    ids[k++] = vocab->bos_id; /* end-of-sequence */
  return k;
}

/* ======================= Model Allocation ================================ */

/*
 * count_params - Return the total number of fp64 scalars across all weight
 *   matrices.  This determines buffer sizes for the gradient, Adam m/v
 *   accumulators, and (in INT8 mode) the master weight copy.
 *
 *   Layout order: wte | wpe | lm_head | per-layer {wq, wk, wv, wo, fc1, fc2}
 */
static size_t count_params(size_t vs, const MicrogptConfig *cfg) {
  const size_t ne = (size_t)N_EMBD;
  const size_t md = (size_t)MLP_DIM;
  /* wte: vs*ne, wpe: block_size*ne, lm_head: vs*ne */
  size_t n = vs * ne * 2 + (size_t)BLOCK_SIZE * ne;
  /* Per-layer: 4 attention matrices (ne²) + 2 MLP matrices */
  for (int L = 0; L < N_LAYER; L++)
    n += ne * ne * 4 + md * ne + ne * md;
  return n;
}

/* ── INT8 quantisation helpers (only compiled when INT8 mode is enabled) ── */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
/*
 * quantize_fp64_to_int8 — Convert an fp64 weight matrix to int8.
 *
 *   SYMMETRIC PER-MATRIX quantisation:
 *     1. Find the absolute-maximum value in the matrix: max_abs
 *     2. Compute scale = max_abs / 127  (maps max_abs ↔ ±127)
 *     3. For each element: int8_val = clamp(round(fp64_val / scale), -127, 127)
 *
 *   Why symmetric?  The int8 range [-127, +127] is symmetric around zero,
 *   so we don't need a separate zero-point bias.  This keeps dequantisation
 *   a single multiply:  fp64_approx = scale × int8_val.
 *
 *   Why per-matrix (not per-tensor or per-channel)?  Per-matrix is a good
 *   balance: per-tensor loses too much precision when different layers have
 *   very different weight ranges; per-channel adds complexity.
 */
static void quantize_fp64_to_int8(const scalar_t *src, size_t n, int8_t *dst,
                                  scalar_t *scale_out) {
  scalar_t mx = 0;
  for (size_t i = 0; i < n; i++) {
    scalar_t a = M_FABS(src[i]);
    if (a > mx)
      mx = a;
  }
  scalar_t s = (mx > 1e-10) ? (mx / 127.0) : 1.0;
  *scale_out = s;
  for (size_t i = 0; i < n; i++) {
    int v = (int)round(src[i] / s);
    if (v > 127)
      v = 127;
    else if (v < -127)
      v = -127;
    dst[i] = (int8_t)v;
  }
}

/*
 * dequantize_int8_to_fp64 — Recover fp64 approximation from int8.
 *   fp64_val ≈ scale × int8_val
 *   Used during backward pass (gradients must be computed in fp64).
 */
static void dequantize_int8_to_fp64(const int8_t *src, scalar_t scale, size_t n,
                                    scalar_t *dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = scale * (scalar_t)src[i];
}

/*
 * quantize_vec_to_int8 — Quantize an activation vector (not weights).
 *   Same algorithm as weight quantisation, but returns the scale factor
 *   (needed for the caller to rescale the int8 matmul result).
 */
static scalar_t quantize_vec_to_int8(const scalar_t *x, size_t n,
                                     int8_t *x_i8) {
  scalar_t mx = 0;
  for (size_t i = 0; i < n; i++) {
    scalar_t a = M_FABS(x[i]);
    if (a > mx)
      mx = a;
  }
  scalar_t scale_x = (mx > 1e-10) ? (mx / 127.0) : 1.0;
  for (size_t i = 0; i < n; i++) {
    int v = (int)round(x[i] / scale_x);
    if (v > 127)
      v = 127;
    else if (v < -127)
      v = -127;
    x_i8[i] = (int8_t)v;
  }
  return scale_x;
}

/*
 * lin_fwd_int8 — Integer-only matrix-vector multiply.
 *
 *   Computes acc[j] = Σ_i  x_i8[i] × W_i8[j×nin + i]
 *
 *   The result is accumulated into int64 to avoid overflow:
 *   worst case each product is 127×127=16129, summed over nin elements.
 *   For nin=256: max_acc = 256×16129 ≈ 4M, well within int64 range.
 *
 *   The caller must rescale the result:  y_fp64[j] = sx × sw × acc[j]
 *   where sx = activation scale, sw = weight scale.
 */
static void lin_fwd_int8(const int8_t *x_i8, const int8_t *W_i8, size_t nin,
                         size_t nout, int64_t *acc) {
  for (size_t j = 0; j < nout; j++) {
    int64_t s = 0;
    for (size_t i = 0; i < nin; i++)
      s += (int64_t)x_i8[i] * (int64_t)W_i8[j * nin + i];
    acc[j] = s;
  }
}

/*
 * get_W — Return fp64 weights for backward pass.
 *   In INT8 mode: dequantises from int8 into a temporary fp64 buffer.
 *   In FP64 mode: simply casts the pointer (no copy needed).
 */
static const scalar_t *get_W(const Model *m, const void *ptr, int scale_idx,
                             size_t n, scalar_t *tmp) {
  dequantize_int8_to_fp64((const int8_t *)ptr, m->scale[scale_idx], n, tmp);
  return tmp;
}
#else
/* FP64 mode: get_W is a no-op that returns the pointer unchanged. */
static const scalar_t *get_W(const Model *m, const void *ptr, int scale_idx,
                             size_t n, scalar_t *tmp) {
  (void)m;
  (void)scale_idx;
  (void)n;
  (void)tmp;
  return (const scalar_t *)ptr;
}
#endif

/* ==================================================================== */
/*                       PAGED KV CACHE                                  */
/* ==================================================================== */

#ifdef MICROGPT_PAGED_KV

PagedKVCache *paged_kv_create(size_t max_positions, int n_embd) {
  PagedKVCache *c = (PagedKVCache *)calloc(1, sizeof(PagedKVCache));
  if (!c)
    return NULL;
  c->capacity = (max_positions + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
  c->pages = (KVPage **)calloc(c->capacity, sizeof(KVPage *));
  if (!c->pages) {
    free(c);
    return NULL;
  }
  c->n_pages = 0;
  c->len = 0;
  c->n_embd = n_embd;
  return c;
}

void paged_kv_free(PagedKVCache *c) {
  if (!c)
    return;
  for (size_t i = 0; i < c->n_pages; i++) {
    if (c->pages[i]) {
      free(c->pages[i]->data);
      free(c->pages[i]);
    }
  }
  free(c->pages);
  free(c);
}

void paged_kv_reset(PagedKVCache *c) {
  if (c)
    c->len = 0; /* pages retained for reuse */
}

scalar_t *paged_kv_append(PagedKVCache *c) {
  size_t page_idx = c->len / KV_PAGE_SIZE;
  size_t slot = c->len % KV_PAGE_SIZE;
  int ne = c->n_embd;

  /* Allocate new page on demand */
  if (page_idx >= c->n_pages) {
    KVPage *p = (KVPage *)calloc(1, sizeof(KVPage));
    p->data =
        (scalar_t *)calloc((size_t)KV_PAGE_SIZE * (size_t)ne, sizeof(scalar_t));
    c->pages[page_idx] = p;
    c->n_pages = page_idx + 1;
  }

  c->len++;
  return c->pages[page_idx]->data + slot * (size_t)ne;
}

const scalar_t *paged_kv_get(const PagedKVCache *c, size_t pos) {
  size_t page_idx = pos / KV_PAGE_SIZE;
  size_t slot = pos % KV_PAGE_SIZE;
  int ne = c->n_embd;
  return c->pages[page_idx]->data + slot * (size_t)ne;
}

/* Accessor macros — transparently switch between flat and paged KV access.
 * KV_WRITE(keys, L, cl, ne)  → pointer to write slot at cache_len[L]
 * KV_READ(keys, L, t, ne)    → pointer to read slot at position t
 */

/* Helper: sync paged len with caller's cache_len, then append. */
static inline scalar_t *paged_kv_sync_append(PagedKVCache *c,
                                             size_t caller_len) {
  if (c->len != caller_len)
    c->len = caller_len; /* caller reset cache_len to 0 */
  return paged_kv_append(c);
}

#define KV_WRITE(arr, L, cl, ne)                                               \
  paged_kv_sync_append((PagedKVCache *)(arr)[L], (cl)[L])
#define KV_READ(arr, L, t, ne) paged_kv_get((const PagedKVCache *)(arr)[L], t)

#else /* !MICROGPT_PAGED_KV — flat arrays (original path) */

#define KV_WRITE(arr, L, cl, ne) ((arr)[L] + (cl)[L] * (ne))
#define KV_READ(arr, L, t, ne) ((arr)[L] + (t) * (ne))

#endif /* MICROGPT_PAGED_KV */

/* ---- Portable KV cache helpers (work with both flat and paged modes) ---- */

scalar_t *kv_cache_alloc(const MicrogptConfig *cfg) {
#ifdef MICROGPT_PAGED_KV
  return (scalar_t *)paged_kv_create((size_t)BLOCK_SIZE, N_EMBD);
#else
  return (scalar_t *)calloc((size_t)BLOCK_SIZE * (size_t)N_EMBD,
                            sizeof(scalar_t));
#endif
}

void kv_cache_free(scalar_t *kv) {
#ifdef MICROGPT_PAGED_KV
  paged_kv_free((PagedKVCache *)kv);
#else
  free(kv);
#endif
}

void kv_cache_reset(scalar_t *kv, const MicrogptConfig *cfg) {
#ifdef MICROGPT_PAGED_KV
  paged_kv_reset((PagedKVCache *)kv);
#else
  memset(kv, 0, (size_t)BLOCK_SIZE * (size_t)N_EMBD * sizeof(scalar_t));
#endif
}

/* Helper: allocate the per-layer pointer arrays for the Model struct. */
static int alloc_layer_ptrs(Model *m, int n_layer) {
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  m->attn_wq = (int8_t **)calloc((size_t)n_layer, sizeof(int8_t *));
  m->attn_wk = (int8_t **)calloc((size_t)n_layer, sizeof(int8_t *));
  m->attn_wv = (int8_t **)calloc((size_t)n_layer, sizeof(int8_t *));
  m->attn_wo = (int8_t **)calloc((size_t)n_layer, sizeof(int8_t *));
  m->mlp_fc1 = (int8_t **)calloc((size_t)n_layer, sizeof(int8_t *));
  m->mlp_fc2 = (int8_t **)calloc((size_t)n_layer, sizeof(int8_t *));
  m->scale = (scalar_t *)calloc((size_t)N_SCALES(n_layer), sizeof(scalar_t));
#else
  m->attn_wq = (scalar_t **)calloc((size_t)n_layer, sizeof(scalar_t *));
  m->attn_wk = (scalar_t **)calloc((size_t)n_layer, sizeof(scalar_t *));
  m->attn_wv = (scalar_t **)calloc((size_t)n_layer, sizeof(scalar_t *));
  m->attn_wo = (scalar_t **)calloc((size_t)n_layer, sizeof(scalar_t *));
  m->mlp_fc1 = (scalar_t **)calloc((size_t)n_layer, sizeof(scalar_t *));
  m->mlp_fc2 = (scalar_t **)calloc((size_t)n_layer, sizeof(scalar_t *));
#endif
  if (!m->attn_wq || !m->attn_wk || !m->attn_wv || !m->attn_wo || !m->mlp_fc1 ||
      !m->mlp_fc2)
    return -1;
  return 0;
}

Model *model_create(size_t vocab_size, const MicrogptConfig *cfg) {
  Model *m = (Model *)calloc(1, sizeof(Model));
  if (!m)
    return NULL;
  m->cfg = *cfg;
  m->vocab_size = vocab_size;
  const size_t ne = (size_t)N_EMBD;
  const size_t bs = (size_t)BLOCK_SIZE;
  const size_t md = (size_t)MLP_DIM;
  const int nl = N_LAYER;
  scalar_t std = INIT_STD;

  if (alloc_layer_ptrs(m, nl) != 0) {
    free(m);
    return NULL;
  }

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  /* Allocate master (fp64) and int8 weight buffers */
  size_t np = count_params(vocab_size, cfg);
  m->master = (scalar_t *)malloc(np * sizeof(scalar_t));
  if (!m->master) {
    free(m);
    return NULL;
  }
  scalar_t *pm = m->master;
  for (size_t i = 0; i < vocab_size * ne; i++)
    pm[i] = rand_gauss() * std;
  pm += vocab_size * ne;
  for (size_t i = 0; i < bs * ne; i++)
    pm[i] = rand_gauss() * std;
  pm += bs * ne;
  for (size_t i = 0; i < vocab_size * ne; i++)
    pm[i] = rand_gauss() * std;
  pm += vocab_size * ne;
  for (int L = 0; L < nl; L++) {
    for (size_t i = 0; i < ne * ne; i++)
      pm[i] = rand_gauss() * std;
    pm += ne * ne;
    for (size_t i = 0; i < ne * ne; i++)
      pm[i] = rand_gauss() * std;
    pm += ne * ne;
    for (size_t i = 0; i < ne * ne; i++)
      pm[i] = rand_gauss() * std;
    pm += ne * ne;
    for (size_t i = 0; i < ne * ne; i++)
      pm[i] = rand_gauss() * std;
    pm += ne * ne;
    for (size_t i = 0; i < md * ne; i++)
      pm[i] = rand_gauss() * std;
    pm += md * ne;
    for (size_t i = 0; i < ne * md; i++)
      pm[i] = rand_gauss() * std;
    pm += ne * md;
  }
  /* Quantize master -> int8 and set scales */
  m->wte = (int8_t *)malloc(vocab_size * ne * sizeof(int8_t));
  m->wpe = (int8_t *)malloc(bs * ne * sizeof(int8_t));
  m->lm_head = (int8_t *)malloc(vocab_size * ne * sizeof(int8_t));
  if (!m->wte || !m->wpe || !m->lm_head)
    goto err_i8;
  quantize_fp64_to_int8(m->master, vocab_size * ne, m->wte, &m->scale[0]);
  quantize_fp64_to_int8(m->master + vocab_size * ne, bs * ne, m->wpe,
                        &m->scale[1]);
  quantize_fp64_to_int8(m->master + vocab_size * ne + bs * ne, vocab_size * ne,
                        m->lm_head, &m->scale[2]);
  size_t off = vocab_size * ne * 2 + bs * ne;
  for (int L = 0; L < nl; L++) {
    m->attn_wq[L] = (int8_t *)malloc(ne * ne * sizeof(int8_t));
    m->attn_wk[L] = (int8_t *)malloc(ne * ne * sizeof(int8_t));
    m->attn_wv[L] = (int8_t *)malloc(ne * ne * sizeof(int8_t));
    m->attn_wo[L] = (int8_t *)malloc(ne * ne * sizeof(int8_t));
    m->mlp_fc1[L] = (int8_t *)malloc(md * ne * sizeof(int8_t));
    m->mlp_fc2[L] = (int8_t *)malloc(ne * md * sizeof(int8_t));
    if (!m->attn_wq[L] || !m->attn_wk[L] || !m->attn_wv[L] || !m->attn_wo[L] ||
        !m->mlp_fc1[L] || !m->mlp_fc2[L])
      goto err_i8;
    int sidx = 3 + L * 6;
    quantize_fp64_to_int8(m->master + off, ne * ne, m->attn_wq[L],
                          &m->scale[sidx]);
    off += ne * ne;
    quantize_fp64_to_int8(m->master + off, ne * ne, m->attn_wk[L],
                          &m->scale[sidx + 1]);
    off += ne * ne;
    quantize_fp64_to_int8(m->master + off, ne * ne, m->attn_wv[L],
                          &m->scale[sidx + 2]);
    off += ne * ne;
    quantize_fp64_to_int8(m->master + off, ne * ne, m->attn_wo[L],
                          &m->scale[sidx + 3]);
    off += ne * ne;
    quantize_fp64_to_int8(m->master + off, md * ne, m->mlp_fc1[L],
                          &m->scale[sidx + 4]);
    off += md * ne;
    quantize_fp64_to_int8(m->master + off, ne * md, m->mlp_fc2[L],
                          &m->scale[sidx + 5]);
    off += ne * md;
  }
  return m;
err_i8:
  free(m->wte);
  free(m->wpe);
  free(m->lm_head);
  for (int L = 0; L < nl; L++) {
    free(m->attn_wq[L]);
    free(m->attn_wk[L]);
    free(m->attn_wv[L]);
    free(m->attn_wo[L]);
    free(m->mlp_fc1[L]);
    free(m->mlp_fc2[L]);
  }
  free(m->attn_wq);
  free(m->attn_wk);
  free(m->attn_wv);
  free(m->attn_wo);
  free(m->mlp_fc1);
  free(m->mlp_fc2);
  free(m->scale);
  free(m->master);
  free(m);
  return NULL;
#else
  m->wte = (scalar_t *)malloc(vocab_size * ne * sizeof(scalar_t));
  m->wpe = (scalar_t *)malloc(bs * ne * sizeof(scalar_t));
  m->lm_head = (scalar_t *)malloc(vocab_size * ne * sizeof(scalar_t));
  if (!m->wte || !m->wpe || !m->lm_head)
    goto err;
  for (size_t i = 0; i < vocab_size * ne; i++)
    m->wte[i] = rand_gauss() * std;
  for (size_t i = 0; i < bs * ne; i++)
    m->wpe[i] = rand_gauss() * std;
  for (size_t i = 0; i < vocab_size * ne; i++)
    m->lm_head[i] = rand_gauss() * std;
  for (int L = 0; L < nl; L++) {
    m->attn_wq[L] = (scalar_t *)malloc(ne * ne * sizeof(scalar_t));
    m->attn_wk[L] = (scalar_t *)malloc(ne * ne * sizeof(scalar_t));
    m->attn_wv[L] = (scalar_t *)malloc(ne * ne * sizeof(scalar_t));
    m->attn_wo[L] = (scalar_t *)malloc(ne * ne * sizeof(scalar_t));
    m->mlp_fc1[L] = (scalar_t *)malloc(md * ne * sizeof(scalar_t));
    m->mlp_fc2[L] = (scalar_t *)malloc(ne * md * sizeof(scalar_t));
    if (!m->attn_wq[L] || !m->attn_wk[L] || !m->attn_wv[L] || !m->attn_wo[L] ||
        !m->mlp_fc1[L] || !m->mlp_fc2[L])
      goto err;
    for (size_t i = 0; i < ne * ne; i++) {
      m->attn_wq[L][i] = rand_gauss() * std;
      m->attn_wk[L][i] = rand_gauss() * std;
      m->attn_wv[L][i] = rand_gauss() * std;
      m->attn_wo[L][i] = rand_gauss() * std;
    }
    for (size_t i = 0; i < md * ne; i++)
      m->mlp_fc1[L][i] = rand_gauss() * std;
    for (size_t i = 0; i < ne * md; i++)
      m->mlp_fc2[L][i] = rand_gauss() * std;
  }
  return m;
err:
  model_free(m);
  return NULL;
#endif
}

/*
 *   model_free - Release all weight buffers and the Model struct.
 *   Safe to call with NULL (no-op).
 */
void model_free(Model *m) {
  if (!m)
    return;
  free(m->wte);
  free(m->wpe);
  free(m->lm_head);
  for (int L = 0; L < m->cfg.n_layer; L++) {
    if (m->attn_wq)
      free(m->attn_wq[L]);
    if (m->attn_wk)
      free(m->attn_wk[L]);
    if (m->attn_wv)
      free(m->attn_wv[L]);
    if (m->attn_wo)
      free(m->attn_wo[L]);
    if (m->mlp_fc1)
      free(m->mlp_fc1[L]);
    if (m->mlp_fc2)
      free(m->mlp_fc2[L]);
  }
  free(m->attn_wq);
  free(m->attn_wk);
  free(m->attn_wv);
  free(m->attn_wo);
  free(m->mlp_fc1);
  free(m->mlp_fc2);
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  free(m->scale);
  free(m->master);
#endif
  free(m);
}

/* Return the total scalar parameter count (for buffer allocation). */
size_t model_num_params(const Model *m) {
  return count_params(m->vocab_size, &m->cfg);
}

/* ======================== Serialisation (fp64 only) ======================= */

#if !defined(QUANTIZATION_INT8) && !defined(QUANTISATION_INT8)
/* Helper: write 'n' doubles to a binary file.  Returns 0 on success. */
static int write_doubles(FILE *f, const scalar_t *p, size_t n) {
  return fwrite(p, sizeof(scalar_t), n, f) == n ? 0 : -1;
}
/* Helper: read 'n' doubles from a binary file.  Returns 0 on success. */
static int read_doubles(FILE *f, scalar_t *p, size_t n) {
  return fread(p, sizeof(scalar_t), n, f) == n ? 0 : -1;
}
/*
 * model_save - Serialise all weights to a binary file.
 *   Format: [vocab_size as size_t] [wte] [wpe] [lm_head] [per-layer weights]
 */
int model_save(const Model *m, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  size_t vs = m->vocab_size;
  const size_t ne = (size_t)N_EMBD;
  const size_t bs = (size_t)BLOCK_SIZE;
  const size_t md = (size_t)MLP_DIM;
  const int nl = N_LAYER;
  if (fwrite(&vs, sizeof(size_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (write_doubles(f, m->wte, vs * ne) != 0) {
    fclose(f);
    return -1;
  }
  if (write_doubles(f, m->wpe, bs * ne) != 0) {
    fclose(f);
    return -1;
  }
  if (write_doubles(f, m->lm_head, vs * ne) != 0) {
    fclose(f);
    return -1;
  }
  for (int L = 0; L < nl; L++) {
    if (write_doubles(f, m->attn_wq[L], ne * ne) != 0 ||
        write_doubles(f, m->attn_wk[L], ne * ne) != 0 ||
        write_doubles(f, m->attn_wv[L], ne * ne) != 0 ||
        write_doubles(f, m->attn_wo[L], ne * ne) != 0 ||
        write_doubles(f, m->mlp_fc1[L], md * ne) != 0 ||
        write_doubles(f, m->mlp_fc2[L], ne * md) != 0) {
      fclose(f);
      return -1;
    }
  }
  fclose(f);
  return 0;
}
/*
 * model_load - Deserialise weights from a binary checkpoint.
 *   Validates that the stored vocab_size matches the expected value.
 *   Returns a newly heap-allocated Model, or NULL on error.
 */
Model *model_load(const char *path, size_t vocab_size,
                  const MicrogptConfig *cfg) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  size_t vs = 0;
  if (fread(&vs, sizeof(size_t), 1, f) != 1 || vs != vocab_size) {
    fclose(f);
    return NULL;
  }
  Model *m = model_create(vocab_size, cfg);
  if (!m) {
    fclose(f);
    return NULL;
  }
  const size_t ne = (size_t)N_EMBD;
  const size_t bs = (size_t)BLOCK_SIZE;
  const size_t md = (size_t)MLP_DIM;
  const int nl = N_LAYER;
  if (read_doubles(f, m->wte, vs * ne) != 0 ||
      read_doubles(f, m->wpe, bs * ne) != 0 ||
      read_doubles(f, m->lm_head, vs * ne) != 0) {
    model_free(m);
    fclose(f);
    return NULL;
  }
  for (int L = 0; L < nl; L++) {
    if (read_doubles(f, m->attn_wq[L], ne * ne) != 0 ||
        read_doubles(f, m->attn_wk[L], ne * ne) != 0 ||
        read_doubles(f, m->attn_wv[L], ne * ne) != 0 ||
        read_doubles(f, m->attn_wo[L], ne * ne) != 0 ||
        read_doubles(f, m->mlp_fc1[L], md * ne) != 0 ||
        read_doubles(f, m->mlp_fc2[L], ne * md) != 0) {
      model_free(m);
      fclose(f);
      return NULL;
    }
  }
  fclose(f);
  return m;
}

/*
 * checkpoint_save - Write a full training checkpoint in a single pass.
 *   Format: [step (int)] [vocab_size (size_t)]
 *           [wte] [wpe] [lm_head] [per-layer weights]
 *           [Adam m buffer] [Adam v buffer]
 */
int checkpoint_save(const Model *model, const scalar_t *m_buf,
                    const scalar_t *v_buf, int step, const char *path) {
  size_t vs = model->vocab_size;
  const size_t ne = (size_t)N_EMBD;
  const size_t bs = (size_t)BLOCK_SIZE;
  const size_t md = (size_t)MLP_DIM;
  const int nl = N_LAYER;
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;

  /* Write step and vocab_size header */
  if (fwrite(&step, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fwrite(&vs, sizeof(size_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }

  /* Write model weights (same order as model_save) */
  if (write_doubles(f, model->wte, vs * ne) != 0 ||
      write_doubles(f, model->wpe, bs * ne) != 0 ||
      write_doubles(f, model->lm_head, vs * ne) != 0) {
    fclose(f);
    return -1;
  }
  for (int L = 0; L < nl; L++) {
    if (write_doubles(f, model->attn_wq[L], ne * ne) != 0 ||
        write_doubles(f, model->attn_wk[L], ne * ne) != 0 ||
        write_doubles(f, model->attn_wv[L], ne * ne) != 0 ||
        write_doubles(f, model->attn_wo[L], ne * ne) != 0 ||
        write_doubles(f, model->mlp_fc1[L], md * ne) != 0 ||
        write_doubles(f, model->mlp_fc2[L], ne * md) != 0) {
      fclose(f);
      return -1;
    }
  }

  /* Write optimizer state */
  size_t np = model_num_params(model);
  if (write_doubles(f, m_buf, np) != 0 || write_doubles(f, v_buf, np) != 0) {
    fclose(f);
    return -1;
  }

  fclose(f);
  return 0;
}

/*
 * checkpoint_load - Load a full training checkpoint in a single pass.
 */
Model *checkpoint_load(const char *path, size_t vocab_size,
                       const MicrogptConfig *cfg, scalar_t *m_buf,
                       scalar_t *v_buf, int *step_out) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;

  /* Read header */
  int step;
  size_t vs;
  if (fread(&step, sizeof(int), 1, f) != 1 ||
      fread(&vs, sizeof(size_t), 1, f) != 1 || vs != vocab_size) {
    fclose(f);
    return NULL;
  }

  /* Create model and read weights */
  Model *model = model_create(vocab_size, cfg);
  if (!model) {
    fclose(f);
    return NULL;
  }
  const size_t ne = (size_t)N_EMBD;
  const size_t bs = (size_t)BLOCK_SIZE;
  const size_t md = (size_t)MLP_DIM;
  const int nl = N_LAYER;

  if (read_doubles(f, model->wte, vs * ne) != 0 ||
      read_doubles(f, model->wpe, bs * ne) != 0 ||
      read_doubles(f, model->lm_head, vs * ne) != 0) {
    model_free(model);
    fclose(f);
    return NULL;
  }
  for (int L = 0; L < nl; L++) {
    if (read_doubles(f, model->attn_wq[L], ne * ne) != 0 ||
        read_doubles(f, model->attn_wk[L], ne * ne) != 0 ||
        read_doubles(f, model->attn_wv[L], ne * ne) != 0 ||
        read_doubles(f, model->attn_wo[L], ne * ne) != 0 ||
        read_doubles(f, model->mlp_fc1[L], md * ne) != 0 ||
        read_doubles(f, model->mlp_fc2[L], ne * md) != 0) {
      model_free(model);
      fclose(f);
      return NULL;
    }
  }

  /* Read optimizer state */
  size_t np = model_num_params(model);
  if (read_doubles(f, m_buf, np) != 0 || read_doubles(f, v_buf, np) != 0) {
    model_free(model);
    fclose(f);
    return NULL;
  }

  fclose(f);
  *step_out = step;
  return model;
}

#else
int model_save(const Model *m, const char *path) {
  (void)m;
  (void)path;
  return -1;
}
Model *model_load(const char *path, size_t vocab_size,
                  const MicrogptConfig *cfg) {
  (void)path;
  (void)vocab_size;
  (void)cfg;
  return NULL;
}
int checkpoint_save(const Model *model, const scalar_t *m, const scalar_t *v,
                    int step, const char *path) {
  (void)model;
  (void)m;
  (void)v;
  (void)step;
  (void)path;
  return -1;
}
Model *checkpoint_load(const char *path, size_t vocab_size,
                       const MicrogptConfig *cfg, scalar_t *m, scalar_t *v,
                       int *step_out) {
  (void)path;
  (void)vocab_size;
  (void)cfg;
  (void)m;
  (void)v;
  (void)step_out;
  return NULL;
}
#endif

/* ===================== Neural Network Primitives ========================= */
/*
 * These are the two fundamental building blocks of neural networks:
 *
 *   1. Linear (Dense) Layer:  y = W @ x     (matrix × vector)
 *   2. RMSNorm:               y = x / RMS(x)  (normalisation)
 *
 * Every other operation in the Transformer (attention, MLP, lm_head)
 * is built from combinations of these two primitives plus element-wise
 * operations (add, multiply, ReLU, softmax, exp).
 */

/*
 * lin_fwd - Dense (fully-connected) linear layer FORWARD pass.
 *
 *   Computes y = W @ x   (matrix-vector product)
 *   More precisely: y[j] = Σ_i  W[j·nin + i] · x[i]
 *
 *   W is stored in row-major layout [nout × nin].
 *
 *   When MICROGPT_BLAS is defined, uses CBLAS_GEMV for hardware-accelerated
 *   matrix-vector multiply (Apple Accelerate, OpenBLAS, MKL, etc.).
 */
static void lin_fwd(const scalar_t *restrict x, const scalar_t *restrict W,
                    size_t nin, size_t nout, scalar_t *restrict y) {
#ifdef MICROGPT_METAL
  if (metal_available()) {
    metal_lin_fwd(x, W, nin, nout, y);
    return;
  }
#endif
#ifdef MICROGPT_BLAS
  /* y = W @ x  via BLAS:  y = 1.0 * W * x + 0.0 * y
   * W is row-major [nout × nin], x is [nin × 1], y is [nout × 1] */
  CBLAS_GEMV(CblasRowMajor, CblasNoTrans, (int)nout,
             (int)nin,         /* M=nout rows, N=nin cols */
             1.0, W, (int)nin, /* alpha=1.0, A=W, lda=nin */
             x, 1,             /* x vector, stride=1 */
             0.0, y, 1);       /* beta=0.0, y vector, stride=1 */
#else
  /* Tiled y = W @ x : process W in TILE_R×TILE_C panels.
   * Each panel (16 KB) stays in L1.  Partial sums accumulate in y[]. */
  memset(y, 0, nout * sizeof(scalar_t));
  for (size_t j0 = 0; j0 < nout; j0 += LIN_TILE_R) {
    size_t j1 = (j0 + LIN_TILE_R < nout) ? j0 + LIN_TILE_R : nout;
    for (size_t i0 = 0; i0 < nin; i0 += LIN_TILE_C) {
      size_t i1 = (i0 + LIN_TILE_C < nin) ? i0 + LIN_TILE_C : nin;
      for (size_t j = j0; j < j1; j++) {
        scalar_t s = 0;
        const scalar_t *restrict Wrow = W + j * nin + i0;
#if defined(__clang__)
        _Pragma("clang loop vectorize(enable) interleave(enable)")
#endif
            for (size_t i = 0; i < i1 - i0; i++) s += x[i0 + i] * Wrow[i];
        y[j] += s;
      }
    }
  }
#endif
}

/*
 * lin_bwd - BACKWARD pass for the linear layer.
 *
 *   dx = W^T @ dy   (gradient w.r.t. input, accumulated)
 *   dW += dy ⊗ x    (gradient w.r.t. weights, outer product, accumulated)
 *
 *   When MICROGPT_BLAS is defined, uses CBLAS_GEMV for dx and CBLAS_GER
 *   for dW (hardware-accelerated rank-1 update).
 */
static void lin_bwd(const scalar_t *restrict x, const scalar_t *restrict W,
                    const scalar_t *restrict dy, size_t nin, size_t nout,
                    scalar_t *restrict dx, scalar_t *restrict dW) {
#ifdef MICROGPT_METAL
  if (metal_available()) {
    metal_lin_bwd(x, W, dy, nin, nout, dx, dW);
    return;
  }
#endif
#ifdef MICROGPT_BLAS
  if (dx)
    /* dx += W^T @ dy  via BLAS:  dx = 1.0 * W^T * dy + 1.0 * dx */
    CBLAS_GEMV(CblasRowMajor, CblasTrans, (int)nout,
               (int)nin,         /* M=nout, N=nin (W dimensions) */
               1.0, W, (int)nin, /* alpha=1.0, A=W, lda=nin */
               dy, 1,            /* dy vector, stride=1 */
               1.0, dx, 1);      /* beta=1.0 (accumulate), dx, stride=1 */
  if (dW)
    /* dW += dy ⊗ x  (rank-1 outer product update) */
    CBLAS_GER(CblasRowMajor, (int)nout, (int)nin, /* M=nout, N=nin */
              1.0,                                /* alpha=1.0 */
              dy, 1,                              /* dy vector, stride=1 */
              x, 1,                               /* x vector, stride=1 */
              dW, (int)nin);                      /* dW matrix, lda=nin */
#else
  /* Tiled dx += W^T @ dy : process W by TILE_R×TILE_C panels.
   * Reading W tile-by-tile keeps each 16 KB panel in L1 while
   * accumulating into the dx[] region covered by that tile. */
  if (dx)
    for (size_t j0 = 0; j0 < nout; j0 += LIN_TILE_R) {
      size_t j1 = (j0 + LIN_TILE_R < nout) ? j0 + LIN_TILE_R : nout;
      for (size_t i0 = 0; i0 < nin; i0 += LIN_TILE_C) {
        size_t i1 = (i0 + LIN_TILE_C < nin) ? i0 + LIN_TILE_C : nin;
        for (size_t j = j0; j < j1; j++) {
          scalar_t dyj = dy[j];
          const scalar_t *restrict Wrow = W + j * nin + i0;
#if defined(__clang__)
          _Pragma("clang loop vectorize(enable) interleave(enable)")
#endif
              for (size_t i = 0; i < i1 - i0; i++) dx[i0 + i] += dyj * Wrow[i];
        }
      }
    }
  /* Tiled dW += dy ⊗ x : same tiling for cache-friendly write to dW. */
  if (dW)
    for (size_t j0 = 0; j0 < nout; j0 += LIN_TILE_R) {
      size_t j1 = (j0 + LIN_TILE_R < nout) ? j0 + LIN_TILE_R : nout;
      for (size_t i0 = 0; i0 < nin; i0 += LIN_TILE_C) {
        size_t i1 = (i0 + LIN_TILE_C < nin) ? i0 + LIN_TILE_C : nin;
        for (size_t j = j0; j < j1; j++) {
          scalar_t dyj = dy[j];
          scalar_t *restrict dWrow = dW + j * nin + i0;
#if defined(__clang__)
          _Pragma("clang loop vectorize(enable) interleave(enable)")
#endif
              for (size_t i = 0; i < i1 - i0; i++) dWrow[i] += dyj * x[i0 + i];
        }
      }
    }
#endif
}

/*
 * rmsnorm_fwd - Root Mean Square Layer Normalisation (forward).
 *
 *   RMSNorm stabilises training by keeping activations at a consistent
 *   scale, preventing them from growing or shrinking as they flow
 *   through many layers.
 *
 *   Formula:
 *     rms = M_SQRT( (1/d) · Σ x[i]² + ε )
 *     out[i] = x[i] / rms
 *
 *   Example with d=4:
 *     x = [3.0, 4.0, 0.0, 0.0]
 *     rms = M_SQRT((9+16+0+0)/4 + 1e-5) = M_SQRT(6.25) = 2.5
 *     out = [1.2, 1.6, 0.0, 0.0]    ← unit "energy" scale
 *
 *   vs. LayerNorm:
 *   - LayerNorm subtracts the mean first, then divides by std.
 *   - RMSNorm skips the mean subtraction — simpler, fewer operations,
 *     and empirically works just as well for Transformers.
 *   - No learnable gamma/beta parameters in this implementation.
 *
 *   eps (1e-5) prevents division by zero when all x values are near 0.
 */
static void rmsnorm_fwd(const scalar_t *restrict x, size_t d,
                        scalar_t *restrict out) {
  scalar_t sum = 0;
#if defined(__clang__)
  _Pragma("clang loop vectorize(enable)")
#endif
      for (size_t i = 0; i < d; i++) sum += x[i] * x[i];
  scalar_t scale = 1.0 / M_SQRT(sum / (scalar_t)d + 1e-5);
#if defined(__clang__)
  _Pragma("clang loop vectorize(enable)")
#endif
      for (size_t i = 0; i < d; i++) out[i] = x[i] * scale;
}

/*
 * rmsnorm_bwd - Backward pass for RMSNorm.
 *   Given dy (upstream gradient) and x (original input, pre-norm):
 *     rms = M_SQRT(mean(x^2) + eps)
 *     dx[i] = (1/rms) * (dy[i] - out[i] * dot(dy, out) / d)
 *   where out[i] = x[i] / rms.
 *   Gradients are accumulated (+=) into dx.
 */
static void rmsnorm_bwd(const scalar_t *restrict x, const scalar_t *restrict dy,
                        size_t d, scalar_t *restrict dx) {
  scalar_t sum_sq = 0;
#if defined(__clang__)
  _Pragma("clang loop vectorize(enable)")
#endif
      for (size_t i = 0; i < d; i++) sum_sq += x[i] * x[i];
  scalar_t rms = M_SQRT(sum_sq / (scalar_t)d + 1e-5);
  scalar_t inv_rms = 1.0 / rms;
  /* Compute dot(dy, x) / (d * rms^2) */
  scalar_t dot = 0;
#if defined(__clang__)
  _Pragma("clang loop vectorize(enable)")
#endif
      for (size_t i = 0; i < d; i++) dot += dy[i] * x[i];
  scalar_t coeff = dot / ((scalar_t)d * rms * rms);
#if defined(__clang__)
  _Pragma("clang loop vectorize(enable)")
#endif
      for (size_t i = 0; i < d; i++) dx[i] += inv_rms * dy[i] - coeff * x[i];
}

/* ================== Per-Head Attention Parallelism ====================== */

#ifdef MICROGPT_HEAD_PARALLEL
/*
 * When MICROGPT_HEAD_PARALLEL is defined, attention heads are processed in
 * parallel using one thread per head.  Each head reads/writes non-overlapping
 * memory slices (keyed by h*hd and h*bs), so no locks are needed.
 *
 * This is most beneficial when batch-level parallelism is not active
 * (nthreads==1), e.g. during inference.  When combined with batch-level
 * threading, over-subscription may occur.
 */

/* ---- Forward attention head worker ---- */
typedef struct {
  int h;                  /* head index */
  scalar_t scale;         /* 1/M_SQRT(hd) */
  size_t T;               /* number of positions to attend over */
  size_t cache_len_L;     /* cached positions for this layer */
  const scalar_t *q;      /* full Q vector [ne] */
  const scalar_t *k;      /* current K vector [ne] */
  const scalar_t *v;      /* current V vector [ne] */
  scalar_t **keys;        /* full KV cache keys array */
  scalar_t **values;      /* full KV cache values array */
  int L;                  /* layer index */
  scalar_t *attn_weights; /* output: attn_weights[h*bs..] */
  scalar_t *x_attn;       /* output: x_attn[h*hd..] */
  size_t ne;              /* n_embd */
  size_t hd;              /* head_dim */
  size_t bs;              /* block_size */
} AttnHeadFwdArg;

static void *attn_head_fwd_worker(void *arg) {
  AttnHeadFwdArg *a = (AttnHeadFwdArg *)arg;
  const size_t ne = a->ne;
  const size_t hd = a->hd;
  const size_t bs = a->bs;
  size_t hoff = (size_t)a->h * hd;
  size_t hw = (size_t)a->h * bs;
  /* Step 1: Q·K scores */
  for (size_t t = 0; t < a->T; t++) {
    const scalar_t *kt = (t < a->cache_len_L)
                             ? (KV_READ(a->keys, a->L, t, ne) + hoff)
                             : (a->k + hoff);
    scalar_t s = 0;
    for (size_t d = 0; d < hd; d++)
      s += a->q[hoff + d] * kt[d];
    a->attn_weights[hw + t] = s * a->scale;
  }
  /* Step 2: Softmax */
  scalar_t max_s = a->attn_weights[hw];
  for (size_t t = 1; t < a->T; t++)
    if (a->attn_weights[hw + t] > max_s)
      max_s = a->attn_weights[hw + t];
  scalar_t sum = 0;
  for (size_t t = 0; t < a->T; t++) {
    a->attn_weights[hw + t] = M_EXP(a->attn_weights[hw + t] - max_s);
    sum += a->attn_weights[hw + t];
  }
  for (size_t t = 0; t < a->T; t++)
    a->attn_weights[hw + t] /= sum;
  /* Step 3: Weighted sum of values */
  for (size_t d = 0; d < hd; d++) {
    scalar_t s = 0;
    for (size_t t = 0; t < a->T; t++) {
      const scalar_t *vt = (t < a->cache_len_L)
                               ? (KV_READ(a->values, a->L, t, ne) + hoff)
                               : (a->v + hoff);
      s += a->attn_weights[hw + t] * vt[d];
    }
    a->x_attn[hoff + d] = s;
  }
  return NULL;
}

/* ---- Backward attention head worker ---- */
typedef struct {
  int h;                       /* head index */
  scalar_t attn_scale;         /* 1/M_SQRT(hd) */
  size_t TL;                   /* T for this layer (from saved forward) */
  const scalar_t *d_x_attn;    /* upstream gradient through x_attn [ne] */
  const scalar_t *sv_attn_w_L; /* saved attn weights [nh*bs] */
  const scalar_t *sv_q_L;      /* saved Q [ne] */
  scalar_t **keys;             /* full KV cache keys array */
  scalar_t **values;           /* full KV cache values array */
  int L;                       /* layer index */
  scalar_t *d_q;               /* output: d_q[h*hd..] */
  scalar_t *d_k_cur;           /* output: d_k_cur[h*hd..] */
  scalar_t *d_v_cur;           /* output: d_v_cur[h*hd..] */
  size_t ne;                   /* n_embd */
  size_t hd;                   /* head_dim */
  size_t bs;                   /* block_size */
} AttnHeadBwdArg;

static void *attn_head_bwd_worker(void *arg) {
  AttnHeadBwdArg *a = (AttnHeadBwdArg *)arg;
  const size_t ne = a->ne;
  const size_t hd = a->hd;
  const size_t bs = a->bs;
  size_t hoff = (size_t)a->h * hd;
  size_t hw = (size_t)a->h * bs;
  /* d_attn_h[t] = dot(d_x_attn[hoff..], V[t][hoff..]) */
  scalar_t d_attn_h[BLOCK_SIZE];
  for (size_t t = 0; t < a->TL; t++) {
    const scalar_t *vt = KV_READ(a->values, a->L, t, ne);
    scalar_t s = 0;
    for (size_t d = 0; d < hd; d++)
      s += a->d_x_attn[hoff + d] * vt[hoff + d];
    d_attn_h[t] = s;
  }
  /* Softmax backward */
  scalar_t dot_ad = 0;
  for (size_t t = 0; t < a->TL; t++)
    dot_ad += a->sv_attn_w_L[hw + t] * d_attn_h[t];
  scalar_t d_score_h[BLOCK_SIZE];
  for (size_t t = 0; t < a->TL; t++)
    d_score_h[t] =
        a->sv_attn_w_L[hw + t] * (d_attn_h[t] - dot_ad) * a->attn_scale;
  /* Q·K backward */
  for (size_t t = 0; t < a->TL; t++) {
    const scalar_t *kt = KV_READ(a->keys, a->L, t, ne);
    for (size_t d = 0; d < hd; d++)
      a->d_q[hoff + d] += d_score_h[t] * kt[hoff + d];
  }
  /* K backward for current position */
  for (size_t d = 0; d < hd; d++)
    a->d_k_cur[hoff + d] += d_score_h[a->TL - 1] * a->sv_q_L[hoff + d];
  /* V backward for current position */
  for (size_t d = 0; d < hd; d++)
    a->d_v_cur[hoff + d] +=
        a->sv_attn_w_L[hw + a->TL - 1] * a->d_x_attn[hoff + d];
  return NULL;
}

/* ---- Dispatch helper: spawn nh threads, join all ---- */
static void attn_heads_dispatch(void *(*fn)(void *), void *args,
                                size_t arg_size, int nh) {
  mgpt_thread_t threads[nh];
  mgpt_thread_trampoline_t tramps[nh];
  for (int h = 0; h < nh; h++)
    mgpt_thread_create(&threads[h], &tramps[h], fn,
                       (char *)args + (size_t)h * arg_size);
  for (int h = 0; h < nh; h++)
    mgpt_thread_join(threads[h]);
}

#endif /* MICROGPT_HEAD_PARALLEL */

/* ================== Forward + Backward (Training) ======================== */

/*
 * forward_backward_one - Full Transformer forward + backward for ONE position.
 *
 * This is the HEART of training.  It does everything needed to process
 * a single token position: run the token forward through the network,
 * compare the prediction to the true next token, compute the loss,
 * and backpropagate gradients to every weight.
 *
 * ┌──────────────────────── FORWARD PASS ─────────────────────────────┐
 * │                                                                   │
 * │  Step 1: EMBED                                                    │
 * │    x = wte[token_id] + wpe[pos_id]                                │
 * │    (Look up the token's learned vector + its position vector)     │
 * │                                                                   │
 * │  Step 2: INITIAL RMSNORM                                          │
 * │    x = RMSNorm(x)                                                 │
 * │    (Normalise the embedding to unit scale)                        │
 * │                                                                   │
 * │  Step 3: TRANSFORMER BLOCKS (× nl)                           │
 * │    For each layer L:                                              │
 * │      3a. ATTENTION SUB-BLOCK:                                     │
 * │          x_norm = RMSNorm(x)                                      │
 * │          Q = Wq @ x_norm                                          │
 * │          K = Wk @ x_norm  → append to KV cache                   │
 * │          V = Wv @ x_norm  → append to KV cache                   │
 * │          attn = softmax(Q · K^T / √d_k) · V   (per head)         │
 * │          x = x + Wo @ attn              (residual connection)     │
 * │                                                                   │
 * │      3b. MLP SUB-BLOCK:                                           │
 * │          x_norm = RMSNorm(x)                                      │
 * │          x = x + fc2 @ ReLU(fc1 @ x_norm)   (residual)           │
 * │                                                                   │
 * │  Step 4: LM HEAD                                                  │
 * │    logits = lm_head @ x   (project to vocab-size scores)         │
 * │    probs = softmax(logits)                                        │
 * │    loss = -M_LOG(probs[target_id])  (cross-entropy: how surprised?) │
 * │                                                                   │
 * └───────────────────────────────────────────────────────────────────┘
 *
 * ┌──────────────────────── BACKWARD PASS ─────────────────────────────┐
 * │                                                                    │
 * │  The backward pass runs in REVERSE order through the same layers.  │
 * │  At each step, it uses the chain rule to compute:                  │
 * │    ∂Loss/∂weight = ∂Loss/∂output × ∂output/∂weight                │
 * │                                                                    │
 * │  Starting from dL/d_logits = probs - one_hot(target):              │
 * │    → backprop through lm_head (lin_bwd)                            │
 * │    → backprop through MLP (fc2, ReLU, fc1)                         │
 * │    → backprop through attention (Wo, Q·K^T·V attention, Wq/Wk/Wv) │
 * │    → backprop through RMSNorm                                      │
 * │    → backprop through embedding lookup (scatter into wte/wpe rows) │
 * │                                                                    │
 * │  All gradients ACCUMULATE (+=) into grad_buffer, which has the     │
 * │  same flat layout as the model parameters.                         │
 * │                                                                    │
 * └────────────────────────────────────────────────────────────────────┘
 *
 * KEY DESIGN CHOICES:
 *   - Processes ONE position at a time (not batched), which keeps memory
 *     usage minimal.  Batch parallelism is handled at the caller level.
 *   - Saves all intermediate activations (sv_*) during forward pass so
 *     they're available during backward — this is the classic memory vs
 *     compute tradeoff of neural network training.
 *   - The KV cache (keys/values arrays) enables efficient autoregressive
 *     training where each new position attends to all previous positions.
 */
scalar_t forward_backward_one(const Model *model, size_t token_id,
                              size_t pos_id, size_t target_id, scalar_t **keys,
                              scalar_t **values, size_t *cache_len,
                              scalar_t *grad_buffer) {
  const size_t vs = model->vocab_size;
  const size_t ne = (size_t)N_EMBD;
  const int nl = N_LAYER;
  const int nh = N_HEAD;
  const size_t bs = (size_t)BLOCK_SIZE;
  const size_t md = (size_t)MLP_DIM;
  const size_t hd = ne / (size_t)nh;
  const size_t max_vocab = (size_t)MAX_VOCAB;
  const size_t T = cache_len[0] + 1;
  scalar_t W_tmp[_MGPT_MAX_W];

  /* Per-layer saved activations for backward */
  scalar_t sv_x_pre[N_LAYER][N_EMBD];   /* input to each layer */
  scalar_t sv_x_norm1[N_LAYER][N_EMBD]; /* pre-attention norm output */
  scalar_t sv_attn_w[N_LAYER]
                    [N_HEAD * BLOCK_SIZE]; /* per-head attention weights */
  scalar_t sv_q[N_LAYER][N_EMBD]; /* saved projected queries for backward */
  size_t sv_T[N_LAYER];           /* T at each layer */
  scalar_t sv_x_attn[N_LAYER][N_EMBD];      /* attention output (pre-Wo) */
  scalar_t sv_x_post_attn[N_LAYER][N_EMBD]; /* after attention residual */
  scalar_t sv_x_norm2[N_LAYER][N_EMBD];     /* pre-MLP norm output */
  scalar_t sv_mlp_pre[N_LAYER][MLP_DIM];    /* fc1 output pre-ReLU */
  scalar_t sv_mlp_post[N_LAYER][MLP_DIM];   /* fc1 output post-ReLU */
  scalar_t sv_x_embed[N_EMBD];              /* embedding before initial norm */

  /* Single-position activation buffers */
  scalar_t x0[N_EMBD], x_norm1[N_EMBD], q[N_EMBD], k[N_EMBD], v[N_EMBD];
  scalar_t attn_weights[N_HEAD * BLOCK_SIZE], x_attn[N_EMBD], x1[N_EMBD],
      x_norm2[N_EMBD];
  scalar_t mlp1[MLP_DIM], x2[N_EMBD], logits[MAX_VOCAB];
  scalar_t d_x[N_EMBD], d_logits[MAX_VOCAB];
  memset(d_x, 0, sizeof(d_x));
  memset(d_logits, 0, sizeof(d_logits));

  /* INT8 mode buffers: x_i8 holds quantised activations, acc_buf holds
   * integer matmul results before rescaling back to fp64.
   * These are only allocated when INT8 quantisation is enabled. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  int8_t x_i8[md];
  int64_t acc_buf[max_vocab];
#endif

  /* ── Step 1: EMBEDDING ──────────────────────────────────────────────── */
  /* Look up the token's embedding vector and add its position encoding.
   * INT8: embeddings are stored as int8 → must dequantise (scale × int8)
   * FP64: direct scalar_t lookup, no conversion needed. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  /* INT8 embedding: dequantise by multiplying scale × int8 value */
  for (size_t i = 0; i < ne; i++)
    x0[i] = model->scale[0] * (scalar_t)model->wte[token_id * ne + i] +
            model->scale[1] * (scalar_t)model->wpe[pos_id * ne + i];
#else
  for (size_t i = 0; i < ne; i++)
    x0[i] = model->wte[token_id * ne + i] + model->wpe[pos_id * ne + i];
#endif
  memcpy(sv_x_embed, x0, ne * sizeof(scalar_t));
  rmsnorm_fwd(x0, ne, x_norm1);
  memcpy(x0, x_norm1, ne * sizeof(scalar_t));

  for (int L = 0; L < nl; L++) {
    memcpy(sv_x_pre[L], x0, ne * sizeof(scalar_t));
    sv_T[L] = T;
    /* ── Step 3a: ATTENTION SUB-BLOCK ───────────────────────────────────── */
    rmsnorm_fwd(x0, ne, x_norm1);
    memcpy(sv_x_norm1[L], x_norm1, ne * sizeof(scalar_t));
    /* Project normalised input into Q, K, V vectors.
     * INT8: quantise x_norm1 → int8, do integer matmul with int8 weights,
     *   then rescale: q[j] = scale_x × scale_w × int_accumulator[j]
     *   Scale indices: wq=3+L*6, wk=4+L*6, wv=5+L*6, wo=6+L*6,
     *                  fc1=7+L*6, fc2=8+L*6  (3 global + 6 per layer)
     * FP64: standard scalar_t-precision matrix multiply. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      scalar_t sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wq[L], ne, ne, acc_buf);
      scalar_t sw = model->scale[3 + L * 6];
      for (size_t j = 0; j < ne; j++)
        q[j] = sx * sw * (scalar_t)acc_buf[j];
    }
    {
      scalar_t sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wk[L], ne, ne, acc_buf);
      scalar_t sw = model->scale[4 + L * 6];
      for (size_t j = 0; j < ne; j++)
        k[j] = sx * sw * (scalar_t)acc_buf[j];
    }
    {
      scalar_t sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wv[L], ne, ne, acc_buf);
      scalar_t sw = model->scale[5 + L * 6];
      for (size_t j = 0; j < ne; j++)
        v[j] = sx * sw * (scalar_t)acc_buf[j];
    }
#else
    lin_fwd(x_norm1, model->attn_wq[L], ne, ne, q);
    lin_fwd(x_norm1, model->attn_wk[L], ne, ne, k);
    lin_fwd(x_norm1, model->attn_wv[L], ne, ne, v);
#endif
    memcpy(sv_q[L], q, ne * sizeof(scalar_t));
    /* Append to cache (caller owns keys[L], values[L]; we write current at
     * cache_len[L]) */
    memcpy(KV_WRITE(keys, L, cache_len, ne), k, ne * sizeof(scalar_t));
    memcpy(KV_WRITE(values, L, cache_len, ne), v, ne * sizeof(scalar_t));

    /*
     * MULTI-HEAD CAUSAL SELF-ATTENTION
     * ================================
     *
     * The attention mechanism lets the model decide how much each
     * previous token (and the current one) should influence the current
     * output.  Multi-head means we do this nh times in parallel,
     * each head operating on a different hd-sized slice.
     *
     *   Full Q vector (ne = 32):
     *   ┌────────┬────────┬────────┬────────┐
     *   │ Head 0 │ Head 1 │ Head 2 │ Head 3 │
     *   │ 8 dims │ 8 dims │ 8 dims │ 8 dims │
     *   └────────┴────────┴────────┴────────┘
     *
     * For each head h:
     *
     *   1. SCORE: dot(Q_h, K_h[t]) / M_SQRT(hd)  for each past position t
     *      ─ How relevant is position t to the current position?
     *      ─ Divide by M_SQRT(d_k) to prevent dot products from growing too
     *        large (which would make softmax saturate to one-hot).
     *
     *   2. SOFTMAX: normalise scores to probabilities that sum to 1
     *      ─ Subtract max for numerical stability (prevents exp overflow)
     *
     *   3. WEIGHTED SUM: output_h = Σ_t  attn_weight[t] × V_h[t]
     *      ─ Blend all value vectors according to the attention weights
     *
     *   Causal masking: the model only attends to positions ≤ current.
     *   This is achieved implicitly because the KV cache only contains
     *   positions 0..T-1 (no future positions).
     */
    scalar_t scale = 1.0 / M_SQRT((scalar_t)hd);
#ifdef MICROGPT_HEAD_PARALLEL
    { /* Parallel forward attention: one thread per head */
      AttnHeadFwdArg fwd_args[nh];
      for (int h = 0; h < nh; h++) {
        fwd_args[h].h = h;
        fwd_args[h].scale = scale;
        fwd_args[h].T = T;
        fwd_args[h].cache_len_L = cache_len[L];
        fwd_args[h].q = q;
        fwd_args[h].k = k;
        fwd_args[h].v = v;
        fwd_args[h].keys = keys;
        fwd_args[h].values = values;
        fwd_args[h].L = L;
        fwd_args[h].attn_weights = attn_weights;
        fwd_args[h].x_attn = x_attn;
        fwd_args[h].ne = ne;
        fwd_args[h].hd = hd;
        fwd_args[h].bs = bs;
      }
      attn_heads_dispatch(attn_head_fwd_worker, fwd_args,
                          sizeof(AttnHeadFwdArg), nh);
    }
#else
    for (int h = 0; h < nh; h++) {
      size_t hoff = (size_t)h * hd; /* offset into the full ne vector */
      size_t hw = (size_t)h * bs;   /* offset into attn_weights storage */
      /* Step 1: Compute attention scores (Q · K for each cached position) */
      for (size_t t = 0; t < T; t++) {
        const scalar_t *kt =
            (t < cache_len[L]) ? (KV_READ(keys, L, t, ne) + hoff) : (k + hoff);
        scalar_t s = 0;
        for (size_t d = 0; d < hd; d++)
          s += q[hoff + d] * kt[d];
        attn_weights[hw + t] = s * scale;
      }
      /* Step 2: Softmax with numerical stability trick (subtract max) */
      scalar_t max_s = attn_weights[hw];
      for (size_t t = 1; t < T; t++)
        if (attn_weights[hw + t] > max_s)
          max_s = attn_weights[hw + t];
      scalar_t sum = 0;
      for (size_t t = 0; t < T; t++) {
        attn_weights[hw + t] = M_EXP(attn_weights[hw + t] - max_s);
        sum += attn_weights[hw + t];
      }
      for (size_t t = 0; t < T; t++)
        attn_weights[hw + t] /= sum;
      /* Step 3: Weighted sum of value vectors → attention output */
      for (size_t d = 0; d < hd; d++) {
        scalar_t s = 0;
        for (size_t t = 0; t < T; t++) {
          const scalar_t *vt = (t < cache_len[L])
                                   ? (KV_READ(values, L, t, ne) + hoff)
                                   : (v + hoff);
          s += attn_weights[hw + t] * vt[d];
        }
        x_attn[hoff + d] = s;
      }
    }
#endif
    memcpy(sv_attn_w[L], attn_weights, (size_t)nh * bs * sizeof(scalar_t));
    memcpy(sv_x_attn[L], x_attn, ne * sizeof(scalar_t));
    /* Project attention output through Wo.
     * INT8: quantise x_attn → int8, do integer matmul, rescale result.
     *   y_fp64 = scale_x × scale_w × int_accumulator
     * FP64: standard scalar_t-precision matrix multiply. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      scalar_t sx = quantize_vec_to_int8(x_attn, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wo[L], ne, ne, acc_buf);
      scalar_t sw = model->scale[6 + L * 6];
      for (size_t j = 0; j < ne; j++)
        x1[j] = sx * sw * (scalar_t)acc_buf[j];
    }
#else
    lin_fwd(x_attn, model->attn_wo[L], ne, ne, x1);
#endif
    for (size_t i = 0; i < ne; i++)
      x1[i] += x0[i]; /* residual */
    memcpy(x0, x1, ne * sizeof(scalar_t));
    memcpy(sv_x_post_attn[L], x0, ne * sizeof(scalar_t));

    /* ── Step 3b: MLP SUB-BLOCK ──────────────────────────────────────── */
    rmsnorm_fwd(x0, ne, x_norm2);
    memcpy(sv_x_norm2[L], x_norm2, ne * sizeof(scalar_t));
    /* MLP: expand to md via fc1, apply ReLU, then project back via fc2.
     * INT8: each matmul is quantise → int8 multiply → rescale.
     *   fc1 uses scale index 7+L*6, fc2 uses 8+L*6.
     * FP64: standard scalar_t-precision matrix multiply. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      scalar_t sx = quantize_vec_to_int8(x_norm2, ne, x_i8);
      lin_fwd_int8(x_i8, model->mlp_fc1[L], ne, md, acc_buf);
      scalar_t sw = model->scale[7 + L * 6];
      for (size_t j = 0; j < md; j++)
        mlp1[j] = sx * sw * (scalar_t)acc_buf[j];
    }
    memcpy(sv_mlp_pre[L], mlp1, md * sizeof(scalar_t));
    for (size_t i = 0; i < md; i++)
      mlp1[i] = mlp1[i] > 0 ? mlp1[i] : 0; /* ReLU activation */
    memcpy(sv_mlp_post[L], mlp1, md * sizeof(scalar_t));
    {
      scalar_t sx = quantize_vec_to_int8(mlp1, md, x_i8);
      lin_fwd_int8(x_i8, model->mlp_fc2[L], md, ne, acc_buf);
      scalar_t sw = model->scale[8 + L * 6];
      for (size_t j = 0; j < ne; j++)
        x2[j] = sx * sw * (scalar_t)acc_buf[j];
    }
#else
    lin_fwd(x_norm2, model->mlp_fc1[L], ne, md, mlp1);
    memcpy(sv_mlp_pre[L], mlp1, md * sizeof(scalar_t));
    for (size_t i = 0; i < md; i++)
      mlp1[i] = mlp1[i] > 0 ? mlp1[i] : 0;
    memcpy(sv_mlp_post[L], mlp1, md * sizeof(scalar_t));
    lin_fwd(mlp1, model->mlp_fc2[L], md, ne, x2);
#endif
    for (size_t i = 0; i < ne; i++)
      x2[i] += x0[i];
    memcpy(x0, x2, ne * sizeof(scalar_t));
  }

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  {
    scalar_t sx = quantize_vec_to_int8(x0, ne, x_i8);
    lin_fwd_int8(x_i8, model->lm_head, ne, vs, acc_buf);
    scalar_t sw = model->scale[2];
    for (size_t j = 0; j < vs; j++)
      logits[j] = sx * sw * (scalar_t)acc_buf[j];
  }
#else
  lin_fwd(x0, model->lm_head, ne, vs, logits);
#endif
  scalar_t max_l = logits[0];
  for (size_t i = 1; i < vs; i++)
    if (logits[i] > max_l)
      max_l = logits[i];
  scalar_t sum = 0;
  for (size_t i = 0; i < vs; i++) {
    logits[i] = M_EXP(logits[i] - max_l);
    sum += logits[i];
  }
  for (size_t i = 0; i < vs; i++)
    logits[i] /= sum;
  scalar_t loss = -M_LOG(logits[target_id] > 1e-10 ? logits[target_id] : 1e-10);

  /* =================== BACKWARD PASS =================== */

  /* d_logits = probs - one_hot(target) */
  for (size_t i = 0; i < vs; i++)
    d_logits[i] = logits[i] - (i == target_id ? 1.0 : 0.0);

  /* Grad buffer layout offsets */
  size_t off_wte = 0;
  size_t off_wpe = vs * ne;
  size_t off_lm = vs * ne + (size_t)bs * ne;
  size_t layer_stride = (size_t)ne * ne * 4 + (size_t)md * ne + (size_t)ne * md;
  size_t off_layers = off_lm + vs * ne;

  /* Backward through lm_head: d_x = lm_head^T @ d_logits */
  lin_bwd(x0, get_W(model, model->lm_head, 2, vs * ne, W_tmp), d_logits, ne, vs,
          d_x, grad_buffer + off_lm);

  /* Backward through layers (reverse order) */
  for (int L = nl - 1; L >= 0; L--) {
    size_t off_L = off_layers + (size_t)L * layer_stride;
    size_t off_wq = off_L;
    size_t off_wk = off_L + ne * ne;
    size_t off_wv = off_L + ne * ne * 2;
    size_t off_wo = off_L + ne * ne * 3;
    size_t off_fc1 = off_L + ne * ne * 4;
    size_t off_fc2 = off_L + ne * ne * 4 + (size_t)md * ne;
    size_t TL = sv_T[L];

    /* --- MLP residual: d_x flows through addition --- */
    /* d_x is the gradient of output of this layer (post-MLP-residual) */

    /* Backward through fc2 */
    scalar_t d_mlp_post[MLP_DIM];
    memset(d_mlp_post, 0, sizeof(d_mlp_post));
    lin_bwd(sv_mlp_post[L],
            get_W(model, model->mlp_fc2[L], 8 + L * 6, (size_t)ne * md, W_tmp),
            d_x, md, ne, d_mlp_post, grad_buffer + off_fc2);

    /* Backward through ReLU */
    scalar_t d_mlp_pre[MLP_DIM];
    for (size_t i = 0; i < md; i++)
      d_mlp_pre[i] = sv_mlp_pre[L][i] > 0 ? d_mlp_post[i] : 0;

    /* Backward through fc1 */
    scalar_t d_x_norm2[N_EMBD];
    memset(d_x_norm2, 0, sizeof(d_x_norm2));
    lin_bwd(sv_x_norm2[L],
            get_W(model, model->mlp_fc1[L], 7 + L * 6, (size_t)md * ne, W_tmp),
            d_mlp_pre, ne, md, d_x_norm2, grad_buffer + off_fc1);

    /* Backward through pre-MLP RMSNorm */
    scalar_t d_x_post_attn[N_EMBD];
    memset(d_x_post_attn, 0, sizeof(d_x_post_attn));
    rmsnorm_bwd(sv_x_post_attn[L], d_x_norm2, ne, d_x_post_attn);

    /* MLP residual: d_x1 = d_x (from residual skip) + d_x_post_attn */
    scalar_t d_x1[N_EMBD];
    for (size_t i = 0; i < ne; i++)
      d_x1[i] = d_x[i] + d_x_post_attn[i];

    /* Backward through Wo */
    scalar_t d_x_attn[N_EMBD];
    memset(d_x_attn, 0, sizeof(d_x_attn));
    lin_bwd(sv_x_attn[L],
            get_W(model, model->attn_wo[L], 6 + L * 6, ne * ne, W_tmp), d_x1,
            ne, ne, d_x_attn, grad_buffer + off_wo);

    /* Backward through multi-head attention */
    scalar_t d_q[N_EMBD];
    memset(d_q, 0, sizeof(d_q));
    scalar_t d_k_cur[N_EMBD];
    memset(d_k_cur, 0, sizeof(d_k_cur));
    scalar_t d_v_cur[N_EMBD];
    memset(d_v_cur, 0, sizeof(d_v_cur));
    scalar_t attn_scale = 1.0 / M_SQRT((scalar_t)hd);

#ifdef MICROGPT_HEAD_PARALLEL
    { /* Parallel backward attention: one thread per head */
      AttnHeadBwdArg bwd_args[N_HEAD];
      for (int h = 0; h < nh; h++) {
        bwd_args[h].h = h;
        bwd_args[h].attn_scale = attn_scale;
        bwd_args[h].TL = TL;
        bwd_args[h].d_x_attn = d_x_attn;
        bwd_args[h].sv_attn_w_L = sv_attn_w[L];
        bwd_args[h].sv_q_L = sv_q[L];
        bwd_args[h].keys = keys;
        bwd_args[h].values = values;
        bwd_args[h].L = L;
        bwd_args[h].d_q = d_q;
        bwd_args[h].d_k_cur = d_k_cur;
        bwd_args[h].d_v_cur = d_v_cur;
        bwd_args[h].ne = ne;
        bwd_args[h].hd = hd;
        bwd_args[h].bs = bs;
      }
      attn_heads_dispatch(attn_head_bwd_worker, bwd_args,
                          sizeof(AttnHeadBwdArg), nh);
    }
#else
    for (int h = 0; h < nh; h++) {
      size_t hoff = (size_t)h * hd;
      size_t hw = (size_t)h * bs;
      /* d_a_h[t] = sum_d d_x_attn[hoff+d] * V[t][hoff+d] */
      scalar_t d_attn_h[BLOCK_SIZE];
      for (size_t t = 0; t < TL; t++) {
        const scalar_t *vt = KV_READ(values, L, t, ne);
        scalar_t s = 0;
        for (size_t d = 0; d < hd; d++)
          s += d_x_attn[hoff + d] * vt[hoff + d];
        d_attn_h[t] = s;
      }
      /* Softmax backward per head */
      scalar_t dot_ad = 0;
      for (size_t t = 0; t < TL; t++)
        dot_ad += sv_attn_w[L][hw + t] * d_attn_h[t];
      scalar_t d_score_h[BLOCK_SIZE];
      for (size_t t = 0; t < TL; t++)
        d_score_h[t] =
            sv_attn_w[L][hw + t] * (d_attn_h[t] - dot_ad) * attn_scale;
      /* Q·K backward per head */
      for (size_t t = 0; t < TL; t++) {
        const scalar_t *kt = KV_READ(keys, L, t, ne);
        for (size_t d = 0; d < hd; d++)
          d_q[hoff + d] += d_score_h[t] * kt[hoff + d];
      }
      /* K backward for current position: d_k += d_score[TL-1] * q */
      for (size_t d = 0; d < hd; d++)
        d_k_cur[hoff + d] += d_score_h[TL - 1] * sv_q[L][hoff + d];
      /* V backward for current position: d_v += a_h[TL-1] * d_x_attn */
      for (size_t d = 0; d < hd; d++)
        d_v_cur[hoff + d] += sv_attn_w[L][hw + TL - 1] * d_x_attn[hoff + d];
    }
#endif

    /* Backward through Q = Wq @ x_norm1 */
    scalar_t d_x_norm1[N_EMBD];
    memset(d_x_norm1, 0, sizeof(d_x_norm1));
    lin_bwd(sv_x_norm1[L],
            get_W(model, model->attn_wq[L], 3 + L * 6, ne * ne, W_tmp), d_q, ne,
            ne, d_x_norm1, grad_buffer + off_wq);
    /* K and V backward through current position's projections */
    lin_bwd(sv_x_norm1[L],
            get_W(model, model->attn_wk[L], 4 + L * 6, ne * ne, W_tmp), d_k_cur,
            ne, ne, d_x_norm1, grad_buffer + off_wk);
    lin_bwd(sv_x_norm1[L],
            get_W(model, model->attn_wv[L], 5 + L * 6, ne * ne, W_tmp), d_v_cur,
            ne, ne, d_x_norm1, grad_buffer + off_wv);

    /* Backward through pre-attention RMSNorm */
    scalar_t d_x_pre[N_EMBD];
    memset(d_x_pre, 0, sizeof(d_x_pre));
    rmsnorm_bwd(sv_x_pre[L], d_x_norm1, ne, d_x_pre);

    /* Attention residual: d_x_pre += d_x1 (residual skip) */
    for (size_t i = 0; i < ne; i++)
      d_x_pre[i] += d_x1[i];

    /* d_x for next layer down = d_x_pre */
    memcpy(d_x, d_x_pre, ne * sizeof(scalar_t));
  }

  /* Backward through initial RMSNorm */
  scalar_t d_embed[N_EMBD];
  memset(d_embed, 0, sizeof(d_embed));
  rmsnorm_bwd(sv_x_embed, d_x, ne, d_embed);

  /* Accumulate embedding gradients */
  for (size_t i = 0; i < ne; i++) {
    grad_buffer[off_wte + token_id * ne + i] += d_embed[i];
    grad_buffer[off_wpe + pos_id * ne + i] += d_embed[i];
  }

  cache_len[0]++;
  for (int L = 1; L < nl; L++)
    cache_len[L]++;
  return loss;
}

/* ==================== Forward Pass (Inference Only) ====================== */

/*
 * forward_inference - Inference-only forward pass through the full Transformer.
 *   Identical structure to the forward half of forward_backward_one but without
 *   loss computation or gradient accumulation.  Produces raw (un-normalised)
 *   logits in 'logits_out' for the next token prediction.
 *
 *   Each call processes one token position, appends K/V to the per-layer
 *   cache (keys[L], values[L]), and increments cache_len[L].
 */
void forward_inference(const Model *model, size_t token_id, size_t pos_id,
                       scalar_t **keys, scalar_t **values, size_t *cache_len,
                       scalar_t *logits_out) {
  const size_t vs = model->vocab_size;
  const size_t ne = (size_t)N_EMBD;
  const int nl = N_LAYER;
  const int nh = N_HEAD;
  const size_t bs = (size_t)BLOCK_SIZE;
  const size_t md = (size_t)MLP_DIM;
  const size_t hd = ne / (size_t)nh;
  size_t T = cache_len[0] + 1; /* total positions including current */
  scalar_t W_tmp[_MGPT_MAX_W];
  scalar_t x0[N_EMBD], x_norm1[N_EMBD], q[N_EMBD], k[N_EMBD], v[N_EMBD];
  scalar_t attn_weights[N_HEAD * BLOCK_SIZE], x_attn[N_EMBD], x1[N_EMBD],
      x_norm2[N_EMBD];
  scalar_t mlp1[MLP_DIM], x2[N_EMBD];

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  int8_t x_i8[md];
  int64_t acc_buf[max_vocab];
  for (size_t i = 0; i < ne; i++)
    x0[i] = model->scale[0] * (scalar_t)model->wte[token_id * ne + i] +
            model->scale[1] * (scalar_t)model->wpe[pos_id * ne + i];
#else
  for (size_t i = 0; i < ne; i++)
    x0[i] = model->wte[token_id * ne + i] + model->wpe[pos_id * ne + i];
#endif
  rmsnorm_fwd(x0, ne, x_norm1);
  memcpy(x0, x_norm1, ne * sizeof(scalar_t));

  for (int L = 0; L < nl; L++) {
    rmsnorm_fwd(x0, ne, x_norm1);
    /* Project normalised input \u2192 Q, K, V (same as training, no saved
     * activations). INT8: quantise \u2192 int8 matmul \u2192 rescale.  FP64:
     * direct scalar_t matmul. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      scalar_t sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wq[L], ne, ne, acc_buf);
      scalar_t sw = model->scale[3 + L * 6];
      for (size_t j = 0; j < ne; j++)
        q[j] = sx * sw * (scalar_t)acc_buf[j];
    }
    {
      scalar_t sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wk[L], ne, ne, acc_buf);
      scalar_t sw = model->scale[4 + L * 6];
      for (size_t j = 0; j < ne; j++)
        k[j] = sx * sw * (scalar_t)acc_buf[j];
    }
    {
      scalar_t sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wv[L], ne, ne, acc_buf);
      scalar_t sw = model->scale[5 + L * 6];
      for (size_t j = 0; j < ne; j++)
        v[j] = sx * sw * (scalar_t)acc_buf[j];
    }
#else
    lin_fwd(x_norm1, model->attn_wq[L], ne, ne, q);
    lin_fwd(x_norm1, model->attn_wk[L], ne, ne, k);
    lin_fwd(x_norm1, model->attn_wv[L], ne, ne, v);
#endif
    memcpy(KV_WRITE(keys, L, cache_len, ne), k, ne * sizeof(scalar_t));
    memcpy(KV_WRITE(values, L, cache_len, ne), v, ne * sizeof(scalar_t));
    /* Multi-head attention */
    scalar_t scale = 1.0 / M_SQRT((scalar_t)hd);
#ifdef MICROGPT_HEAD_PARALLEL
    { /* Parallel inference attention: one thread per head */
      AttnHeadFwdArg fwd_args[nh];
      for (int h = 0; h < nh; h++) {
        fwd_args[h].h = h;
        fwd_args[h].scale = scale;
        fwd_args[h].T = T;
        fwd_args[h].cache_len_L = cache_len[L];
        fwd_args[h].q = q;
        fwd_args[h].k = k;
        fwd_args[h].v = v;
        fwd_args[h].keys = keys;
        fwd_args[h].values = values;
        fwd_args[h].L = L;
        fwd_args[h].attn_weights = attn_weights;
        fwd_args[h].x_attn = x_attn;
        fwd_args[h].ne = ne;
        fwd_args[h].hd = hd;
        fwd_args[h].bs = bs;
      }
      attn_heads_dispatch(attn_head_fwd_worker, fwd_args,
                          sizeof(AttnHeadFwdArg), nh);
    }
#else
    for (int h = 0; h < nh; h++) {
      size_t hoff = (size_t)h * hd;
      size_t hw = (size_t)h * bs;
      for (size_t t = 0; t < T; t++) {
        const scalar_t *kt =
            (t < cache_len[L]) ? (KV_READ(keys, L, t, ne) + hoff) : (k + hoff);
        scalar_t s = 0;
        for (size_t d = 0; d < hd; d++)
          s += q[hoff + d] * kt[d];
        attn_weights[hw + t] = s * scale;
      }
      scalar_t max_s = attn_weights[hw];
      for (size_t t = 1; t < T; t++)
        if (attn_weights[hw + t] > max_s)
          max_s = attn_weights[hw + t];
      scalar_t sum = 0;
      for (size_t t = 0; t < T; t++) {
        attn_weights[hw + t] = M_EXP(attn_weights[hw + t] - max_s);
        sum += attn_weights[hw + t];
      }
      for (size_t t = 0; t < T; t++)
        attn_weights[hw + t] /= sum;
      for (size_t d = 0; d < hd; d++) {
        scalar_t s = 0;
        for (size_t t = 0; t < T; t++) {
          const scalar_t *vt = (t < cache_len[L])
                                   ? (KV_READ(values, L, t, ne) + hoff)
                                   : (v + hoff);
          s += attn_weights[hw + t] * vt[d];
        }
        x_attn[hoff + d] = s;
      }
    }
#endif
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      scalar_t sx = quantize_vec_to_int8(x_attn, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wo[L], ne, ne, acc_buf);
      scalar_t sw = model->scale[6 + L * 6];
      for (size_t j = 0; j < ne; j++)
        x1[j] = sx * sw * (scalar_t)acc_buf[j];
    }
#else
    lin_fwd(x_attn, model->attn_wo[L], ne, ne, x1);
#endif
    for (size_t i = 0; i < ne; i++)
      x1[i] += x0[i];
    memcpy(x0, x1, ne * sizeof(scalar_t));
    /* ── MLP forward (inference) ── */
    rmsnorm_fwd(x0, ne, x_norm2);
    /* INT8: quantise → int8 matmul → rescale for both fc1 and fc2. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      scalar_t sx = quantize_vec_to_int8(x_norm2, ne, x_i8);
      lin_fwd_int8(x_i8, model->mlp_fc1[L], ne, md, acc_buf);
      scalar_t sw = model->scale[7 + L * 6];
      for (size_t j = 0; j < md; j++)
        mlp1[j] = sx * sw * (scalar_t)acc_buf[j];
    }
    for (size_t i = 0; i < md; i++)
      mlp1[i] = mlp1[i] > 0 ? mlp1[i] : 0;
    {
      scalar_t sx = quantize_vec_to_int8(mlp1, md, x_i8);
      lin_fwd_int8(x_i8, model->mlp_fc2[L], md, ne, acc_buf);
      scalar_t sw = model->scale[8 + L * 6];
      for (size_t j = 0; j < ne; j++)
        x2[j] = sx * sw * (scalar_t)acc_buf[j];
    }
#else
    lin_fwd(x_norm2, model->mlp_fc1[L], ne, md, mlp1);
    for (size_t i = 0; i < md; i++)
      mlp1[i] = mlp1[i] > 0 ? mlp1[i] : 0;
    lin_fwd(mlp1, model->mlp_fc2[L], md, ne, x2);
#endif
    for (size_t i = 0; i < ne; i++)
      x2[i] += x0[i];
    memcpy(x0, x2, ne * sizeof(scalar_t));
    T = cache_len[L] + 1;
  }
  /* ── LM head: project final hidden state to vocabulary logits ── */
  /* INT8: final quantise → int8 matmul → rescale.  scale[2] = lm_head scale. */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  {
    scalar_t sx = quantize_vec_to_int8(x0, ne, x_i8);
    lin_fwd_int8(x_i8, model->lm_head, ne, vs, acc_buf);
    scalar_t sw = model->scale[2];
    for (size_t j = 0; j < vs; j++)
      logits_out[j] = sx * sw * (scalar_t)acc_buf[j];
  }
#else
  lin_fwd(x0, model->lm_head, ne, vs, logits_out);
#endif
  for (int L = 0; L < nl; L++)
    cache_len[L]++;
}

/* ========================= Adam Optimiser ================================ */

/*
 * adam_step - One step of the Adam optimiser with cosine LR schedule.
 *
 * ADAM UPDATE RULE (per parameter w_i):
 * ─────────────────────────────────────
 *   m_i = β₁·m_i + (1-β₁)·g_i          (1st moment: exponential moving average)
 *   v_i = β₂·v_i + (1-β₂)·g_i²         (2nd moment: tracks gradient variance)
 *   m̂ = m_i / (1 - β₁^t)                (bias correction for early steps)
 *   v̂ = v_i / (1 - β₂^t)
 *   w_i -= lr · m̂ / (√v̂ + ε)
 *
 *   Intuition: Adam adapts the learning rate PER PARAMETER.
 *   - Parameters with small, consistent gradients get larger effective LR
 *   - Parameters with large, noisy gradients get smaller effective LR
 *   - The √v̂ in the denominator is what provides this adaptation
 *
 * LEARNING RATE SCHEDULE:
 * ─────────────────────────
 *   LR │ peak
 *      │ ╱╲
 *      │╱  ╲  ╌ ╌ cosine decay
 *      │     ╲
 *      │      ╲
 *      │       ╲
 *      │╱       ╲───
 *      └──────────────── step
 *      │warmup│    decay phase
 *
 *   Phase 1 (warmup): LR ramps linearly from 0 to LEARNING_RATE
 *   Phase 2 (decay):  LR follows cos(progress·π) from LEARNING_RATE to ~0
 *
 *   Why warmup?  Early gradients are noisy (random weights),
 *   so large LR could push the model to bad regions. Ramping gives
 *   the moment estimates (m, v) time to stabilize.
 *
 *   Why cosine decay?  Smoother than linear, avoids the sudden drop at
 *   the end that can destabilize training.
 *
 * INT8 MODE:
 * ──────────
 *   Updates are applied to the fp64 master copy (not to int8 weights
 *   directly — that would lose precision).  After all parameters are
 *   updated, the master copy is requantised to int8 with fresh scales.
 */
void adam_step(Model *model, const scalar_t *grads, scalar_t *m, scalar_t *v,
               int step) {
  /* ── Cosine LR schedule with linear warmup ─────────────────────────── */
  const int warmup = WARMUP_STEPS;
  const int total = NUM_STEPS;
  const scalar_t peak_lr = (scalar_t)LEARNING_RATE;
  scalar_t lr;
  if (step < warmup)
    /* Warmup: linearly ramp from 0 → peak_lr */
    lr = peak_lr * ((scalar_t)(step + 1) / (scalar_t)warmup);
  else {
    /* Cosine decay: peak_lr → ~0 following a half-cosine curve */
    scalar_t progress = (scalar_t)(step - warmup) / (scalar_t)(total - warmup);
    lr = peak_lr * 0.5 * (1.0 + cos(progress * 3.14159265358979323846));
  }
  scalar_t b1 = BETA1, b2 = BETA2, eps = EPS_ADAM;
  /* Pre-compute bias-correction denominators once per step (not per weight!
   * — these only depend on step number, not on individual gradients). */
  scalar_t bc1 = 1.0 - M_POW(b1, (scalar_t)(step + 1));
  scalar_t bc2 = 1.0 - M_POW(b2, (scalar_t)(step + 1));
  size_t vs = model->vocab_size;
  const size_t ne = (size_t)N_EMBD;
  const int nl = N_LAYER;
  const size_t bs = (size_t)BLOCK_SIZE;
  const size_t md = (size_t)MLP_DIM;
  size_t idx = 0; /* flat index into the gradient / moment arrays */
  /* ── INT8 path: update the fp64 master copy, then requantise later ── */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  scalar_t *master = model->master;
  /* Update wte (token embeddings) in the master copy */
  for (size_t i = 0; i < vs * ne; i++, idx++) {
    scalar_t g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    scalar_t mh = m[idx] / bc1;
    scalar_t vh = v[idx] / bc2;
    master[idx] -= lr * mh / (M_SQRT(vh) + eps);
  }
  for (size_t i = 0; i < bs * ne; i++, idx++) {
    scalar_t g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    scalar_t mh = m[idx] / bc1;
    scalar_t vh = v[idx] / bc2;
    master[idx] -= lr * mh / (M_SQRT(vh) + eps);
  }
  for (size_t i = 0; i < vs * ne; i++, idx++) {
    scalar_t g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    scalar_t mh = m[idx] / bc1;
    scalar_t vh = v[idx] / bc2;
    master[idx] -= lr * mh / (M_SQRT(vh) + eps);
  }
  for (int L = 0; L < nl; L++) {
    for (size_t i = 0; i < ne * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      master[idx] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < ne * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      master[idx] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < ne * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      master[idx] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < ne * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      master[idx] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < md * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      master[idx] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < ne * md; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      master[idx] -= lr * mh / (M_SQRT(vh) + eps);
    }
  }
  /* Requantize master -> int8 */
  size_t off = 0;
  quantize_fp64_to_int8(master + off, vs * ne, model->wte, &model->scale[0]);
  off += vs * ne;
  quantize_fp64_to_int8(master + off, bs * ne, model->wpe, &model->scale[1]);
  off += bs * ne;
  quantize_fp64_to_int8(master + off, vs * ne, model->lm_head,
                        &model->scale[2]);
  off += vs * ne;
  for (int L = 0; L < nl; L++) {
    quantize_fp64_to_int8(master + off, ne * ne, model->attn_wq[L],
                          &model->scale[3 + L * 6]);
    off += ne * ne;
    quantize_fp64_to_int8(master + off, ne * ne, model->attn_wk[L],
                          &model->scale[4 + L * 6]);
    off += ne * ne;
    quantize_fp64_to_int8(master + off, ne * ne, model->attn_wv[L],
                          &model->scale[5 + L * 6]);
    off += ne * ne;
    quantize_fp64_to_int8(master + off, ne * ne, model->attn_wo[L],
                          &model->scale[6 + L * 6]);
    off += ne * ne;
    quantize_fp64_to_int8(master + off, md * ne, model->mlp_fc1[L],
                          &model->scale[7 + L * 6]);
    off += md * ne;
    quantize_fp64_to_int8(master + off, ne * md, model->mlp_fc2[L],
                          &model->scale[8 + L * 6]);
    off += ne * md;
  }
#else
  for (size_t i = 0; i < vs * ne; i++, idx++) {
    scalar_t g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    scalar_t mh = m[idx] / bc1;
    scalar_t vh = v[idx] / bc2;
    model->wte[i] -= lr * mh / (M_SQRT(vh) + eps);
  }
  for (size_t i = 0; i < bs * ne; i++, idx++) {
    scalar_t g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    scalar_t mh = m[idx] / bc1;
    scalar_t vh = v[idx] / bc2;
    model->wpe[i] -= lr * mh / (M_SQRT(vh) + eps);
  }
  for (size_t i = 0; i < vs * ne; i++, idx++) {
    scalar_t g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    scalar_t mh = m[idx] / bc1;
    scalar_t vh = v[idx] / bc2;
    model->lm_head[i] -= lr * mh / (M_SQRT(vh) + eps);
  }
  for (int L = 0; L < nl; L++) {
    for (size_t i = 0; i < ne * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      model->attn_wq[L][i] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < ne * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      model->attn_wk[L][i] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < ne * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      model->attn_wv[L][i] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < ne * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      model->attn_wo[L][i] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < md * ne; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      model->mlp_fc1[L][i] -= lr * mh / (M_SQRT(vh) + eps);
    }
    for (size_t i = 0; i < ne * md; i++, idx++) {
      scalar_t g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      scalar_t mh = m[idx] / bc1;
      scalar_t vh = v[idx] / bc2;
      model->mlp_fc2[L][i] -= lr * mh / (M_SQRT(vh) + eps);
    }
  }
#endif
}

/* ========================= Sampling ====================================== */

/*
 * sample_token - Draw a token from a temperature-scaled softmax distribution.
 *
 *   1. Subtract max(logits) for numerical stability.
 *   2. Compute softmax: p_i = M_EXP((logit_i - max) / temperature) / Z
 *   3. Sample from the resulting categorical distribution using the
 *      internal PRNG (rand_u).
 *
 *   Lower temperature -> more deterministic (peaky distribution).
 *   Higher temperature -> more uniform / creative.
 *
 *   Falls through to the last token if floating-point rounding prevents
 *   the cumulative sum from reaching zero.
 */
size_t sample_token(const scalar_t *logits, size_t vocab_size,
                    scalar_t temperature) {
  scalar_t buf[MAX_VOCAB];
  /* Clamp temperature to prevent underflow in M_EXP() */
  if (temperature < 1e-4)
    temperature = 1e-4;
  /* Find max logit for numerical stability in M_EXP() */
  scalar_t max_val = logits[0];
  for (size_t i = 1; i < vocab_size; i++)
    if (logits[i] > max_val)
      max_val = logits[i];
  /* Temperature-scaled softmax */
  scalar_t sum = 0;
  for (size_t i = 0; i < vocab_size; i++) {
    buf[i] = M_EXP((logits[i] - max_val) / temperature);
    sum += buf[i];
  }
  for (size_t i = 0; i < vocab_size; i++)
    buf[i] /= sum; /* normalise to probabilities */
  /* Inverse-CDF sampling: draw r ~ U[0,1), subtract probs until r <= 0 */
  scalar_t r = rand_u();
  for (size_t i = 0; i < vocab_size; i++) {
    r -= buf[i];
    if (r <= 0)
      return i;
  }
  return vocab_size - 1; /* fallback for rounding edge cases */
}

/* ====================== File Loading Utility ============================= */

char *load_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  long fz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (fz <= 0) {
    fclose(f);
    return NULL;
  }
  char *data = (char *)malloc((size_t)fz + 1);
  if (!data) {
    fclose(f);
    return NULL;
  }
  fread(data, 1, (size_t)fz, f);
  data[fz] = '\0';
  fclose(f);
  *out_len = (size_t)fz;
  return data;
}

/* =================== Word-Level Tokenisation ============================= */
/*
 * WORD-LEVEL TOKENISATION (alternative to character-level)
 * ────────────────────────────────────────────────────────
 * Instead of treating each character as a token, we treat each whitespace-
 * delimited word as a token.  This dramatically reduces sequence length
 * (and thus the number of forward/backward passes per training sample).
 *
 * The vocabulary is built by:
 *   1. Scan all documents, counting word frequencies (hash table)
 *   2. Sort by frequency (descending)
 *   3. Keep the top max_vocab words; out-of-vocabulary words map to <UNK>
 *
 * For word lookup we use a HASH TABLE with OPEN ADDRESSING:
 *
 *   Hash table (capacity = max_vocab * 2):
 *   ┌───────┬───────┬───────┬───────┬───────┬───────┬─── ─ ─
 *   │ empty │ "the" │ empty │ "and" │ empty │  "of" │ ...
 *   │       │ id=3  │       │ id=7  │       │ id=5  │
 *   └───────┴───────┴───────┴───────┴───────┴───────┴─── ─ ─
 *               ▲               ▲
 *               │               │
 *     hash("the") % cap   hash("and") % cap  → if slot taken, probe next
 *
 *   Lookup is O(1) average case (vs O(V) linear scan without the table).
 *   This matters for Shakespeare-scale vocabularies (6000+ words).
 */

/* Internal: hash table entry for word frequency counting */
typedef struct {
  char word[MAX_WORD_LEN];
  size_t count;
} WordFreqEntry;

/*
 * word_hash — DJB2 hash function (Daniel J. Bernstein).
 *   h = h×33 + c  for each character.  Simple, fast, good distribution.
 *   The magic number 5381 is a well-chosen starting value.
 */
static unsigned int word_hash(const char *s) {
  unsigned int h = 5381;
  while (*s)
    h = h * 33 + (unsigned char)*s++;
  return h;
}

/*
 * word_ht_find_or_insert — Open-addressing hash table probe.
 *   Finds an existing entry for 'word', or inserts a new one.
 *   Collision resolution: linear probing (check next slot if taken).
 */
static size_t word_ht_find_or_insert(WordFreqEntry *table, size_t cap,
                                     size_t *n, const char *word) {
  unsigned int h = word_hash(word) % (unsigned int)cap;
  while (table[h].count > 0 || table[h].word[0] != '\0') {
    if (strcmp(table[h].word, word) == 0)
      return h;
    h = (h + 1) % (unsigned int)cap; /* linear probe: try next slot */
  }
  strncpy(table[h].word, word, MAX_WORD_LEN - 1);
  table[h].word[MAX_WORD_LEN - 1] = '\0';
  table[h].count = 0;
  (*n)++;
  return h;
}

static int cmp_freq_desc(const void *a, const void *b) {
  size_t ca = ((const WordFreqEntry *)a)->count;
  size_t cb = ((const WordFreqEntry *)b)->count;
  return (cb > ca) ? 1 : (cb < ca) ? -1 : 0;
}

int build_word_vocab(const char *text, size_t text_len, size_t max_words,
                     WordVocab *wv) {
  size_t ht_cap = max_words * 4; /* load factor ~0.25 for performance */
  if (ht_cap < 1024)
    ht_cap = 1024;
  WordFreqEntry *ht = (WordFreqEntry *)calloc(ht_cap, sizeof(WordFreqEntry));
  if (!ht)
    return -1;
  size_t num_unique = 0;

  const char *p = text;
  const char *end = text + text_len;
  char word_buf[MAX_WORD_LEN];

  while (p < end) {
    while (p < end && *p == ' ')
      p++;
    if (p >= end)
      break;
    if (*p == '\n' || *p == '\r') {
      p++;
      continue;
    }
    const char *ws = p;
    while (p < end && *p != ' ' && *p != '\n' && *p != '\r')
      p++;
    size_t wlen = (size_t)(p - ws);
    if (wlen >= MAX_WORD_LEN)
      wlen = MAX_WORD_LEN - 1;
    memcpy(word_buf, ws, wlen);
    word_buf[wlen] = '\0';
    if (num_unique < ht_cap - 1) {
      size_t idx = word_ht_find_or_insert(ht, ht_cap, &num_unique, word_buf);
      ht[idx].count++;
    }
  }

  /* Collect non-empty entries and sort by frequency */
  WordFreqEntry *sorted =
      (WordFreqEntry *)malloc(num_unique * sizeof(WordFreqEntry));
  if (!sorted) {
    free(ht);
    return -1;
  }
  size_t n = 0;
  for (size_t i = 0; i < ht_cap; i++)
    if (ht[i].count > 0)
      sorted[n++] = ht[i];
  free(ht);
  qsort(sorted, n, sizeof(WordFreqEntry), cmp_freq_desc);

  size_t keep = n < max_words ? n : max_words;

  /* Assign IDs: [0..keep-1] = top words, keep = unk, keep+1 = \n, keep+2 = BOS
   */
  wv->num_words = keep;
  wv->unk_id = keep;
  wv->newline_id = keep + 1;
  wv->bos_id = keep + 2;
  wv->vocab_size = keep + 3;

  wv->words = (char **)calloc(wv->vocab_size, sizeof(char *));
  if (!wv->words) {
    free(sorted);
    return -1;
  }
  for (size_t i = 0; i < keep; i++) {
    wv->words[i] = (char *)malloc(strlen(sorted[i].word) + 1);
    if (!wv->words[i])
      goto err_words;
    strcpy(wv->words[i], sorted[i].word);
  }
  /* Special tokens — strdup-like for uniform free() */
  wv->words[wv->unk_id] = (char *)malloc(6);
  if (!wv->words[wv->unk_id])
    goto err_words;
  strcpy(wv->words[wv->unk_id], "<unk>");
  wv->words[wv->newline_id] = (char *)malloc(2);
  if (!wv->words[wv->newline_id])
    goto err_words;
  strcpy(wv->words[wv->newline_id], "\n");
  wv->words[wv->bos_id] = (char *)malloc(6);
  if (!wv->words[wv->bos_id])
    goto err_words;
  strcpy(wv->words[wv->bos_id], "<bos>");
  /* Build lookup hash table for O(1) word_to_id */
  wv->ht_cap = wv->vocab_size * 4;
  if (wv->ht_cap < 64)
    wv->ht_cap = 64;
  wv->ht_keys = (char **)calloc(wv->ht_cap, sizeof(char *));
  wv->ht_ids = (size_t *)calloc(wv->ht_cap, sizeof(size_t));
  if (!wv->ht_keys || !wv->ht_ids)
    goto err_words;
  for (size_t i = 0; i < wv->vocab_size; i++) {
    if (!wv->words[i])
      continue;
    unsigned int h = word_hash(wv->words[i]) % (unsigned int)wv->ht_cap;
    while (wv->ht_keys[h])
      h = (h + 1) % (unsigned int)wv->ht_cap;
    wv->ht_keys[h] = wv->words[i]; /* points into words[], not owned */
    wv->ht_ids[h] = i;
  }

  free(sorted);
  return 0;

err_words:
  for (size_t i = 0; i < wv->vocab_size; i++)
    free(wv->words[i]);
  free(wv->words);
  wv->words = NULL;
  free(sorted);
  return -1;
}

void free_word_vocab(WordVocab *wv) {
  if (wv->words) {
    for (size_t i = 0; i < wv->vocab_size; i++)
      free(wv->words[i]);
    free(wv->words);
  }
  free(wv->ht_keys);
  free(wv->ht_ids);
  memset(wv, 0, sizeof(*wv));
}

size_t word_to_id(const WordVocab *wv, const char *word) {
  /* O(1) amortised lookup via hash table (populated by build_word_vocab) */
  if (wv->ht_keys && wv->ht_cap > 0) {
    unsigned int h = word_hash(word) % (unsigned int)wv->ht_cap;
    while (wv->ht_keys[h]) {
      if (strcmp(wv->ht_keys[h], word) == 0)
        return wv->ht_ids[h];
      h = (h + 1) % (unsigned int)wv->ht_cap;
    }
    return wv->unk_id;
  }
  /* Fallback: linear scan (should not happen after build_word_vocab) */
  for (size_t i = 0; i < wv->num_words; i++)
    if (wv->words[i] && strcmp(wv->words[i], word) == 0)
      return i;
  return wv->unk_id;
}

size_t tokenize_words(const char *text, size_t text_len, const WordVocab *wv,
                      size_t *ids, size_t max_tokens) {
  size_t k = 0;
  const char *p = text;
  const char *end = text + text_len;
  char word_buf[MAX_WORD_LEN];

  while (p < end && k < max_tokens) {
    while (p < end && *p == ' ')
      p++;
    if (p >= end)
      break;
    if (*p == '\n' || *p == '\r') {
      ids[k++] = wv->newline_id;
      if (*p == '\r')
        p++;
      if (p < end && *p == '\n')
        p++;
      continue;
    }
    const char *ws = p;
    while (p < end && *p != ' ' && *p != '\n' && *p != '\r')
      p++;
    size_t wlen = (size_t)(p - ws);
    if (wlen >= MAX_WORD_LEN)
      wlen = MAX_WORD_LEN - 1;
    memcpy(word_buf, ws, wlen);
    word_buf[wlen] = '\0';
    ids[k++] = word_to_id(wv, word_buf);
  }
  return k;
}

/* ======================== Training Helpers ================================ */

/*
 * shuffle_docs - Fisher-Yates in-place shuffle of the document list.
 */
void shuffle_docs(Docs *docs) {
  for (size_t i = docs->num_docs; i > 1; i--) {
    size_t j = (size_t)rand() % i;
    char *tmp_line = docs->lines[j];
    size_t tmp_len = docs->doc_lens[j];
    docs->lines[j] = docs->lines[i - 1];
    docs->doc_lens[j] = docs->doc_lens[i - 1];
    docs->lines[i - 1] = tmp_line;
    docs->doc_lens[i - 1] = tmp_len;
  }
}

/*
 * train_worker_run - Thread entry point for batched training.
 *   Processes documents [batch_start, batch_end), tokenises each document,
 *   runs forward+backward, and accumulates gradients and loss.
 */
void *train_worker_run(void *arg) {
  TrainWorker *w = (TrainWorker *)arg;
  const int nl = N_LAYER;
  const size_t bs = (size_t)BLOCK_SIZE;
  size_t nparams = model_num_params(w->model);
  memset(w->grads, 0, nparams * sizeof(scalar_t));
  w->loss = 0;
  w->positions = 0;

  for (int b = w->batch_start; b < w->batch_end; b++) {
    for (int L = 0; L < nl; L++)
      w->cache_len[L] = 0;

    /* Pick a random document */
    size_t di = (size_t)rand_r(&w->rng_seed) % w->docs->num_docs;
    const char *doc = w->docs->lines[di];
    size_t doc_len = w->docs->doc_lens[di];

    /* Tokenize: [BOS] chars... [BOS/EOS] */
    size_t n_tok = tokenize(doc, doc_len, w->vocab, w->token_buf, bs + 2);
    size_t n = n_tok - 1;
    if (n > bs)
      n = bs;
    if (n == 0)
      continue;
    w->positions += n;

    for (size_t pos = 0; pos < n; pos++) {
      scalar_t loss = forward_backward_one(w->model, w->token_buf[pos], pos,
                                           w->token_buf[pos + 1], w->keys,
                                           w->values, w->cache_len, w->grads);
      w->loss += loss;
    }
  }
  return NULL;
}
