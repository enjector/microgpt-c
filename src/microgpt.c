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
 * This file contains the full implementation of:
 *   - Data loading and character-level tokenisation
 *   - Model allocation with Gaussian-random weight initialisation
 *   - Forward pass (embedding -> Transformer blocks -> lm_head -> softmax)
 *   - Backward pass (gradient accumulation through cross-entropy + lm_head)
 *   - Adam optimiser with linear learning-rate decay
 *   - Autoregressive inference with temperature-controlled sampling
 *
 * Two build modes are supported:
 *   FP64 (default)  - all weights stored as double precision.
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

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
/*
 * INT8 Model Layout
 * -----------------
 * Weights are stored as int8_t arrays with one fp64 scale factor per matrix
 * (symmetric quantisation: scale = max(|W|)/127, W_i8 = round(W/scale)).
 *
 * N_SCALES counts the total number of scale factors:
 *   3 global matrices (wte, wpe, lm_head) + 6 per layer (wq, wk, wv, wo, fc1,
 * fc2).
 *
 * 'master' is the full-precision fp64 copy of all parameters.  Adam updates
 * are applied to 'master'; afterwards the int8 tensors are requantised so
 * that forward passes use the cheap integer arithmetic path.
 */
#define N_SCALES (3 + 6 * N_LAYER)
struct Model {
  size_t vocab_size;        /* number of character tokens + BOS            */
  int8_t *wte;              /* token embedding      [vocab_size x N_EMBD] */
  int8_t *wpe;              /* position embedding   [BLOCK_SIZE x N_EMBD] */
  int8_t *lm_head;          /* output projection    [vocab_size x N_EMBD] */
  int8_t *attn_wq[N_LAYER]; /* query   weight       [N_EMBD x N_EMBD]     */
  int8_t *attn_wk[N_LAYER]; /* key     weight       [N_EMBD x N_EMBD]     */
  int8_t *attn_wv[N_LAYER]; /* value   weight       [N_EMBD x N_EMBD]     */
  int8_t *attn_wo[N_LAYER]; /* output  weight       [N_EMBD x N_EMBD]     */
  int8_t *mlp_fc1[N_LAYER]; /* MLP up-projection    [MLP_DIM x N_EMBD]    */
  int8_t *mlp_fc2[N_LAYER]; /* MLP down-projection  [N_EMBD x MLP_DIM]    */
  double scale[N_SCALES];   /* per-matrix quantisation scales              */
  double *master;           /* fp64 master weights for Adam updates        */
};
#else
/*
 * FP64 Model Layout (default)
 * ---------------------------
 * Every weight matrix is a heap-allocated double array.
 * Weight layouts follow row-major [output_dim x input_dim] convention so
 * that y = W @ x is computed as: y[j] = sum_i W[j*nin + i] * x[i].
 */
struct Model {
  size_t vocab_size;        /* number of character tokens + BOS            */
  double *wte;              /* token embedding      [vocab_size x N_EMBD] */
  double *wpe;              /* position embedding   [BLOCK_SIZE x N_EMBD] */
  double *lm_head;          /* output projection    [vocab_size x N_EMBD] */
  double *attn_wq[N_LAYER]; /* query   weight       [N_EMBD x N_EMBD]     */
  double *attn_wk[N_LAYER]; /* key     weight       [N_EMBD x N_EMBD]     */
  double *attn_wv[N_LAYER]; /* value   weight       [N_EMBD x N_EMBD]     */
  double *attn_wo[N_LAYER]; /* output  weight       [N_EMBD x N_EMBD]     */
  double *mlp_fc1[N_LAYER]; /* MLP up-projection    [MLP_DIM x N_EMBD]    */
  double *mlp_fc2[N_LAYER]; /* MLP down-projection  [N_EMBD x MLP_DIM]    */
};
#endif

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

/* Return a uniform random double in [0, 1). */
static double rand_u(void) {
  rng_state = rng_state * 1103515245UL + 12345UL;
  return (double)((rng_state / 65536UL) % 32768UL) / 32768.0;
}

/*
 * Box-Muller transform: convert two uniform samples into one sample
 * from N(0, 1).  Used for Gaussian weight initialisation.
 */
static double rand_gauss(void) {
  double u1 = rand_u(), u2 = rand_u();
  if (u1 < 1e-10)
    u1 = 1e-10; /* clamp to avoid log(0) */
  return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
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
int load_docs(const char *path, Docs *docs) {
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

  /* Allocate line index arrays (up to MAX_DOCS entries) */
  docs->lines = (char **)malloc(MAX_DOCS * sizeof(char *));
  docs->doc_lens = (size_t *)malloc(MAX_DOCS * sizeof(size_t));
  if (!docs->lines || !docs->doc_lens) {
    free(docs->data);
    return -1;
  }

  /* Scan buffer and record each non-empty line */
  size_t nd = 0;
  char *p = docs->data;
  while (nd < MAX_DOCS && *p) {
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
static size_t count_params(size_t vs) {
  /* wte: vs*N_EMBD, wpe: BLOCK_SIZE*N_EMBD, lm_head: vs*N_EMBD */
  size_t n = vs * N_EMBD * 2 + vs * N_EMBD + BLOCK_SIZE * N_EMBD;
  /* Per-layer: 4 attention matrices (N_EMBD²) + 2 MLP matrices */
  for (int L = 0; L < N_LAYER; L++)
    n += N_EMBD * N_EMBD * 4 + MLP_DIM * N_EMBD + N_EMBD * MLP_DIM;
  return n;
}

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
/*
 * Symmetric per-matrix INT8 quantisation:
 *   scale = max(|W|) / 127
 *   W_i8[i] = clamp(round(W[i] / scale), -127, 127)
 *
 * This preserves the dynamic range of each matrix individually while
 * mapping values to the int8 representable range [-127, 127].
 */
static void quantize_fp64_to_int8(const double *src, size_t n, int8_t *dst,
                                  double *scale_out) {
  double mx = 0;
  for (size_t i = 0; i < n; i++) {
    double a = fabs(src[i]);
    if (a > mx)
      mx = a;
  }
  double s = (mx > 1e-10) ? (mx / 127.0) : 1.0;
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
static void dequantize_int8_to_fp64(const int8_t *src, double scale, size_t n,
                                    double *dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = scale * (double)src[i];
}
/* Quantize double vector to int8; scale_x = max(|x|)/127, x_i8 =
 * round(x/scale_x) clamped to [-127,127] */
static double quantize_vec_to_int8(const double *x, size_t n, int8_t *x_i8) {
  double mx = 0;
  for (size_t i = 0; i < n; i++) {
    double a = fabs(x[i]);
    if (a > mx)
      mx = a;
  }
  double scale_x = (mx > 1e-10) ? (mx / 127.0) : 1.0;
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
/* int8 matmul: acc[j] = sum_i x_i8[i] * W_i8[j*nin+i]; W layout [nout, nin] */
static void lin_fwd_int8(const int8_t *x_i8, const int8_t *W_i8, size_t nin,
                         size_t nout, int32_t *acc) {
  for (size_t j = 0; j < nout; j++) {
    int32_t s = 0;
    for (size_t i = 0; i < nin; i++)
      s += (int32_t)x_i8[i] * (int32_t)W_i8[j * nin + i];
    acc[j] = s;
  }
}
/* Return double* for a weight tensor (dequantize into tmp when INT8); used for
 * backward only */
static const double *get_W(const Model *m, const void *ptr, int scale_idx,
                           size_t n, double *tmp) {
  dequantize_int8_to_fp64((const int8_t *)ptr, m->scale[scale_idx], n, tmp);
  return tmp;
}
#else
static const double *get_W(const Model *m, const void *ptr, int scale_idx,
                           size_t n, double *tmp) {
  (void)m;
  (void)scale_idx;
  (void)n;
  (void)tmp;
  return (const double *)ptr;
}
#endif

Model *model_create(size_t vocab_size) {
  Model *m = (Model *)calloc(1, sizeof(Model));
  if (!m)
    return NULL;
  m->vocab_size = vocab_size;
  double std = 0.08;
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  /* Allocate master (fp64) and int8 weight buffers */
  size_t np = count_params(vocab_size);
  m->master = (double *)malloc(np * sizeof(double));
  if (!m->master) {
    free(m);
    return NULL;
  }
  double *pm = m->master;
  for (size_t i = 0; i < vocab_size * N_EMBD; i++)
    pm[i] = rand_gauss() * std;
  pm += vocab_size * N_EMBD;
  for (size_t i = 0; i < BLOCK_SIZE * N_EMBD; i++)
    pm[i] = rand_gauss() * std;
  pm += BLOCK_SIZE * N_EMBD;
  for (size_t i = 0; i < vocab_size * N_EMBD; i++)
    pm[i] = rand_gauss() * std;
  pm += vocab_size * N_EMBD;
  for (int L = 0; L < N_LAYER; L++) {
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++)
      pm[i] = rand_gauss() * std;
    pm += N_EMBD * N_EMBD;
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++)
      pm[i] = rand_gauss() * std;
    pm += N_EMBD * N_EMBD;
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++)
      pm[i] = rand_gauss() * std;
    pm += N_EMBD * N_EMBD;
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++)
      pm[i] = rand_gauss() * std;
    pm += N_EMBD * N_EMBD;
    for (size_t i = 0; i < MLP_DIM * N_EMBD; i++)
      pm[i] = rand_gauss() * std;
    pm += MLP_DIM * N_EMBD;
    for (size_t i = 0; i < N_EMBD * MLP_DIM; i++)
      pm[i] = rand_gauss() * std;
    pm += N_EMBD * MLP_DIM;
  }
  /* Quantize master -> int8 and set scales */
  m->wte = (int8_t *)malloc(vocab_size * N_EMBD * sizeof(int8_t));
  m->wpe = (int8_t *)malloc(BLOCK_SIZE * N_EMBD * sizeof(int8_t));
  m->lm_head = (int8_t *)malloc(vocab_size * N_EMBD * sizeof(int8_t));
  if (!m->wte || !m->wpe || !m->lm_head)
    goto err_i8;
  quantize_fp64_to_int8(m->master, vocab_size * N_EMBD, m->wte, &m->scale[0]);
  quantize_fp64_to_int8(m->master + vocab_size * N_EMBD, BLOCK_SIZE * N_EMBD,
                        m->wpe, &m->scale[1]);
  quantize_fp64_to_int8(m->master + vocab_size * N_EMBD + BLOCK_SIZE * N_EMBD,
                        vocab_size * N_EMBD, m->lm_head, &m->scale[2]);
  size_t off = vocab_size * N_EMBD * 2 + BLOCK_SIZE * N_EMBD;
  for (int L = 0; L < N_LAYER; L++) {
    m->attn_wq[L] = (int8_t *)malloc(N_EMBD * N_EMBD * sizeof(int8_t));
    m->attn_wk[L] = (int8_t *)malloc(N_EMBD * N_EMBD * sizeof(int8_t));
    m->attn_wv[L] = (int8_t *)malloc(N_EMBD * N_EMBD * sizeof(int8_t));
    m->attn_wo[L] = (int8_t *)malloc(N_EMBD * N_EMBD * sizeof(int8_t));
    m->mlp_fc1[L] = (int8_t *)malloc(MLP_DIM * N_EMBD * sizeof(int8_t));
    m->mlp_fc2[L] = (int8_t *)malloc(N_EMBD * MLP_DIM * sizeof(int8_t));
    if (!m->attn_wq[L] || !m->attn_wk[L] || !m->attn_wv[L] || !m->attn_wo[L] ||
        !m->mlp_fc1[L] || !m->mlp_fc2[L])
      goto err_i8;
    int sidx = 3 + L * 6;
    quantize_fp64_to_int8(m->master + off, N_EMBD * N_EMBD, m->attn_wq[L],
                          &m->scale[sidx]);
    off += N_EMBD * N_EMBD;
    quantize_fp64_to_int8(m->master + off, N_EMBD * N_EMBD, m->attn_wk[L],
                          &m->scale[sidx + 1]);
    off += N_EMBD * N_EMBD;
    quantize_fp64_to_int8(m->master + off, N_EMBD * N_EMBD, m->attn_wv[L],
                          &m->scale[sidx + 2]);
    off += N_EMBD * N_EMBD;
    quantize_fp64_to_int8(m->master + off, N_EMBD * N_EMBD, m->attn_wo[L],
                          &m->scale[sidx + 3]);
    off += N_EMBD * N_EMBD;
    quantize_fp64_to_int8(m->master + off, MLP_DIM * N_EMBD, m->mlp_fc1[L],
                          &m->scale[sidx + 4]);
    off += MLP_DIM * N_EMBD;
    quantize_fp64_to_int8(m->master + off, N_EMBD * MLP_DIM, m->mlp_fc2[L],
                          &m->scale[sidx + 5]);
    off += N_EMBD * MLP_DIM;
  }
  return m;
err_i8:
  free(m->wte);
  free(m->wpe);
  free(m->lm_head);
  for (int L = 0; L < N_LAYER; L++) {
    free(m->attn_wq[L]);
    free(m->attn_wk[L]);
    free(m->attn_wv[L]);
    free(m->attn_wo[L]);
    free(m->mlp_fc1[L]);
    free(m->mlp_fc2[L]);
  }
  free(m->master);
  free(m);
  return NULL;
#else
  m->wte = (double *)malloc(vocab_size * N_EMBD * sizeof(double));
  m->wpe = (double *)malloc(BLOCK_SIZE * N_EMBD * sizeof(double));
  m->lm_head = (double *)malloc(vocab_size * N_EMBD * sizeof(double));
  if (!m->wte || !m->wpe || !m->lm_head)
    goto err;
  for (size_t i = 0; i < vocab_size * N_EMBD; i++)
    m->wte[i] = rand_gauss() * std;
  for (size_t i = 0; i < BLOCK_SIZE * N_EMBD; i++)
    m->wpe[i] = rand_gauss() * std;
  for (size_t i = 0; i < vocab_size * N_EMBD; i++)
    m->lm_head[i] = rand_gauss() * std;
  for (int L = 0; L < N_LAYER; L++) {
    m->attn_wq[L] = (double *)malloc(N_EMBD * N_EMBD * sizeof(double));
    m->attn_wk[L] = (double *)malloc(N_EMBD * N_EMBD * sizeof(double));
    m->attn_wv[L] = (double *)malloc(N_EMBD * N_EMBD * sizeof(double));
    m->attn_wo[L] = (double *)malloc(N_EMBD * N_EMBD * sizeof(double));
    m->mlp_fc1[L] = (double *)malloc(MLP_DIM * N_EMBD * sizeof(double));
    m->mlp_fc2[L] = (double *)malloc(N_EMBD * MLP_DIM * sizeof(double));
    if (!m->attn_wq[L] || !m->attn_wk[L] || !m->attn_wv[L] || !m->attn_wo[L] ||
        !m->mlp_fc1[L] || !m->mlp_fc2[L])
      goto err;
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++) {
      m->attn_wq[L][i] = rand_gauss() * std;
      m->attn_wk[L][i] = rand_gauss() * std;
      m->attn_wv[L][i] = rand_gauss() * std;
      m->attn_wo[L][i] = rand_gauss() * std;
    }
    for (size_t i = 0; i < MLP_DIM * N_EMBD; i++)
      m->mlp_fc1[L][i] = rand_gauss() * std;
    for (size_t i = 0; i < N_EMBD * MLP_DIM; i++)
      m->mlp_fc2[L][i] = rand_gauss() * std;
  }
  return m;
err:
  model_free(m);
  return NULL;
#endif
}

/*
 * model_free - Release all weight buffers and the Model struct.
 *   Safe to call with NULL (no-op).
 */
void model_free(Model *m) {
  if (!m)
    return;
  free(m->wte);
  free(m->wpe);
  free(m->lm_head);
  for (int L = 0; L < N_LAYER; L++) {
    free(m->attn_wq[L]);
    free(m->attn_wk[L]);
    free(m->attn_wv[L]);
    free(m->attn_wo[L]);
    free(m->mlp_fc1[L]);
    free(m->mlp_fc2[L]);
  }
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  free(m->master);
#endif
  free(m);
}

/* Return the total scalar parameter count (for buffer allocation). */
size_t model_num_params(const Model *m) { return count_params(m->vocab_size); }

/* ======================== Serialisation (fp64 only) ======================= */

#if !defined(QUANTIZATION_INT8) && !defined(QUANTISATION_INT8)
/* Helper: write 'n' doubles to a binary file.  Returns 0 on success. */
static int write_doubles(FILE *f, const double *p, size_t n) {
  return fwrite(p, sizeof(double), n, f) == n ? 0 : -1;
}
/* Helper: read 'n' doubles from a binary file.  Returns 0 on success. */
static int read_doubles(FILE *f, double *p, size_t n) {
  return fread(p, sizeof(double), n, f) == n ? 0 : -1;
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
  if (fwrite(&vs, sizeof(size_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (write_doubles(f, m->wte, vs * N_EMBD) != 0) {
    fclose(f);
    return -1;
  }
  if (write_doubles(f, m->wpe, (size_t)BLOCK_SIZE * N_EMBD) != 0) {
    fclose(f);
    return -1;
  }
  if (write_doubles(f, m->lm_head, vs * N_EMBD) != 0) {
    fclose(f);
    return -1;
  }
  for (int L = 0; L < N_LAYER; L++) {
    if (write_doubles(f, m->attn_wq[L], N_EMBD * N_EMBD) != 0) {
      fclose(f);
      return -1;
    }
    if (write_doubles(f, m->attn_wk[L], N_EMBD * N_EMBD) != 0) {
      fclose(f);
      return -1;
    }
    if (write_doubles(f, m->attn_wv[L], N_EMBD * N_EMBD) != 0) {
      fclose(f);
      return -1;
    }
    if (write_doubles(f, m->attn_wo[L], N_EMBD * N_EMBD) != 0) {
      fclose(f);
      return -1;
    }
    if (write_doubles(f, m->mlp_fc1[L], MLP_DIM * N_EMBD) != 0) {
      fclose(f);
      return -1;
    }
    if (write_doubles(f, m->mlp_fc2[L], N_EMBD * MLP_DIM) != 0) {
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
Model *model_load(const char *path, size_t vocab_size) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  size_t vs = 0;
  if (fread(&vs, sizeof(size_t), 1, f) != 1 || vs != vocab_size) {
    fclose(f);
    return NULL;
  }
  Model *m = model_create(vocab_size);
  if (!m) {
    fclose(f);
    return NULL;
  }
  if (read_doubles(f, m->wte, vs * N_EMBD) != 0) {
    model_free(m);
    fclose(f);
    return NULL;
  }
  if (read_doubles(f, m->wpe, (size_t)BLOCK_SIZE * N_EMBD) != 0) {
    model_free(m);
    fclose(f);
    return NULL;
  }
  if (read_doubles(f, m->lm_head, vs * N_EMBD) != 0) {
    model_free(m);
    fclose(f);
    return NULL;
  }
  for (int L = 0; L < N_LAYER; L++) {
    if (read_doubles(f, m->attn_wq[L], N_EMBD * N_EMBD) != 0) {
      model_free(m);
      fclose(f);
      return NULL;
    }
    if (read_doubles(f, m->attn_wk[L], N_EMBD * N_EMBD) != 0) {
      model_free(m);
      fclose(f);
      return NULL;
    }
    if (read_doubles(f, m->attn_wv[L], N_EMBD * N_EMBD) != 0) {
      model_free(m);
      fclose(f);
      return NULL;
    }
    if (read_doubles(f, m->attn_wo[L], N_EMBD * N_EMBD) != 0) {
      model_free(m);
      fclose(f);
      return NULL;
    }
    if (read_doubles(f, m->mlp_fc1[L], MLP_DIM * N_EMBD) != 0) {
      model_free(m);
      fclose(f);
      return NULL;
    }
    if (read_doubles(f, m->mlp_fc2[L], N_EMBD * MLP_DIM) != 0) {
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
int checkpoint_save(const Model *model, const double *m_buf,
                    const double *v_buf, int step, const char *path) {
  size_t vs = model->vocab_size;
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
  if (write_doubles(f, model->wte, vs * N_EMBD) != 0 ||
      write_doubles(f, model->wpe, (size_t)BLOCK_SIZE * N_EMBD) != 0 ||
      write_doubles(f, model->lm_head, vs * N_EMBD) != 0) {
    fclose(f);
    return -1;
  }
  for (int L = 0; L < N_LAYER; L++) {
    if (write_doubles(f, model->attn_wq[L], N_EMBD * N_EMBD) != 0 ||
        write_doubles(f, model->attn_wk[L], N_EMBD * N_EMBD) != 0 ||
        write_doubles(f, model->attn_wv[L], N_EMBD * N_EMBD) != 0 ||
        write_doubles(f, model->attn_wo[L], N_EMBD * N_EMBD) != 0 ||
        write_doubles(f, model->mlp_fc1[L], MLP_DIM * N_EMBD) != 0 ||
        write_doubles(f, model->mlp_fc2[L], N_EMBD * MLP_DIM) != 0) {
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
Model *checkpoint_load(const char *path, size_t vocab_size, double *m_buf,
                       double *v_buf, int *step_out) {
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
  Model *model = model_create(vocab_size);
  if (!model) {
    fclose(f);
    return NULL;
  }

  if (read_doubles(f, model->wte, vs * N_EMBD) != 0 ||
      read_doubles(f, model->wpe, (size_t)BLOCK_SIZE * N_EMBD) != 0 ||
      read_doubles(f, model->lm_head, vs * N_EMBD) != 0) {
    model_free(model);
    fclose(f);
    return NULL;
  }
  for (int L = 0; L < N_LAYER; L++) {
    if (read_doubles(f, model->attn_wq[L], N_EMBD * N_EMBD) != 0 ||
        read_doubles(f, model->attn_wk[L], N_EMBD * N_EMBD) != 0 ||
        read_doubles(f, model->attn_wv[L], N_EMBD * N_EMBD) != 0 ||
        read_doubles(f, model->attn_wo[L], N_EMBD * N_EMBD) != 0 ||
        read_doubles(f, model->mlp_fc1[L], MLP_DIM * N_EMBD) != 0 ||
        read_doubles(f, model->mlp_fc2[L], N_EMBD * MLP_DIM) != 0) {
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
Model *model_load(const char *path, size_t vocab_size) {
  (void)path;
  (void)vocab_size;
  return NULL;
}
int checkpoint_save(const Model *model, const double *m, const double *v,
                    int step, const char *path) {
  (void)model;
  (void)m;
  (void)v;
  (void)step;
  (void)path;
  return -1;
}
Model *checkpoint_load(const char *path, size_t vocab_size, double *m,
                       double *v, int *step_out) {
  (void)path;
  (void)vocab_size;
  (void)m;
  (void)v;
  (void)step_out;
  return NULL;
}
#endif

/* ===================== Neural Network Primitives ========================= */

/*
 * lin_fwd - Dense (fully-connected) linear layer forward pass.
 *   Computes y = x @ W^T  (i.e. y[j] = sum_i x[i] * W[j * nin + i]).
 *   W is stored in row-major [nout x nin] layout.
 */
static void lin_fwd(const double *x, const double *W, size_t nin, size_t nout,
                    double *y) {
  for (size_t j = 0; j < nout; j++) {
    double s = 0;
    for (size_t i = 0; i < nin; i++)
      s += x[i] * W[j * nin + i];
    y[j] = s;
  }
}

/*
 * lin_bwd - Backward pass for the linear layer.
 *   Given upstream gradient dy (size nout):
 *     dx[i] += sum_j dy[j] * W[j*nin+i]   (gradient w.r.t. input x)
 *     dW[j*nin+i] += dy[j] * x[i]         (gradient w.r.t. weight W)
 *   Either dx or dW may be NULL to skip that gradient.
 *   Gradients are *accumulated* (+=), not overwritten.
 */
static void lin_bwd(const double *x, const double *W, const double *dy,
                    size_t nin, size_t nout, double *dx, double *dW) {
  if (dx)
    for (size_t i = 0; i < nin; i++) {
      double s = 0;
      for (size_t j = 0; j < nout; j++)
        s += dy[j] * W[j * nin + i];
      dx[i] += s;
    }
  if (dW)
    for (size_t j = 0; j < nout; j++)
      for (size_t i = 0; i < nin; i++)
        dW[j * nin + i] += dy[j] * x[i];
}

/*
 * rmsnorm_fwd - Root Mean Square Layer Normalisation (forward).
 *   out[i] = x[i] / sqrt(mean(x^2) + eps)
 *   Unlike LayerNorm, RMSNorm does not subtract the mean and has no
 *   learnable gamma/beta parameters in this minimal implementation.
 *   eps = 1e-5 for numerical stability.
 */
static void rmsnorm_fwd(const double *x, size_t d, double *out) {
  double sum = 0;
  for (size_t i = 0; i < d; i++)
    sum += x[i] * x[i];
  double scale = 1.0 / sqrt(sum / (double)d + 1e-5);
  for (size_t i = 0; i < d; i++)
    out[i] = x[i] * scale;
}

/*
 * rmsnorm_bwd - Backward pass for RMSNorm.
 *   Propagates gradients d_out back through the normalisation to d_x.
 *   Uses the chain rule on:  out = x * (1 / sqrt(mean(x^2) + eps))
 */
static void rmsnorm_bwd(const double *x, size_t d, const double *d_out,
                        double *d_x) {
  double sum = 0;
  for (size_t i = 0; i < d; i++)
    sum += x[i] * x[i];
  double ms = sum / (double)d + 1e-5; /* mean-square + eps */
  double scale = 1.0 / sqrt(ms);      /* normalisation factor */
  double d_scale = 0;
  for (size_t i = 0; i < d; i++)
    d_scale += d_out[i] * x[i]; /* chain rule: d(scale) */
  d_scale *= scale * (-0.5 / ms / (double)d);
  for (size_t i = 0; i < d; i++)
    d_x[i] += d_out[i] * scale + d_scale * x[i] * 2.0 / (double)d;
}

/* ================== Forward + Backward (Training) ======================== */

/*
 * forward_backward_one - Full Transformer forward pass for a single position,
 *   followed by cross-entropy loss computation and backward gradient
 *   accumulation through the lm_head projection.
 *
 * Pipeline:
 *   1. Embed: x0 = wte[token_id] + wpe[pos_id]
 *   2. RMSNorm(x0)
 *   3. For each layer:
 *      a. RMSNorm -> Q, K, V projections -> causal self-attention -> Wo ->
 * residual b. RMSNorm -> MLP(fc1 -> ReLU -> fc2) -> residual
 *   4. lm_head projection -> softmax -> cross-entropy loss
 *   5. Backward: gradients for lm_head, wte, wpe accumulated into grad_buffer
 *
 * NOTE: The current backward only propagates through lm_head; attention/MLP
 * layer gradients rely on the per-position loss signal through the embedding
 * gradient (a simplification from the reference implementation).
 */
double forward_backward_one(const Model *model, size_t token_id, size_t pos_id,
                            size_t target_id, double **keys, double **values,
                            size_t *cache_len, double *grad_buffer) {
  const size_t vs = model->vocab_size;
  const size_t ne = N_EMBD;
  const size_t T = cache_len[0] + 1; /* total positions including current */
#define MAX_W_SIZE (MLP_DIM * N_EMBD)
  double W_tmp[MAX_W_SIZE]; /* scratch buffer for dequantising weights (INT8) */

  /* Stack-allocated activation buffers for a single position */
  double x0[N_EMBD], x_norm1[N_EMBD], q[N_EMBD], k[N_EMBD], v[N_EMBD];
  double attn_weights[BLOCK_SIZE], x_attn[N_EMBD], x1[N_EMBD], x_norm2[N_EMBD];
  double mlp1[MLP_DIM], x2[N_EMBD], logits[MAX_VOCAB];
  double d_x2[N_EMBD], d_logits[MAX_VOCAB]; /* gradient accumulators */
  memset(d_x2, 0, sizeof(d_x2));
  memset(d_logits, 0, sizeof(d_logits));

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  int8_t x_i8[MLP_DIM];
  int32_t acc_buf[MAX_VOCAB];
#endif

  /* Embed + rmsnorm */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  for (size_t i = 0; i < ne; i++)
    x0[i] = model->scale[0] * (double)model->wte[token_id * ne + i] +
            model->scale[1] * (double)model->wpe[pos_id * ne + i];
#else
  for (size_t i = 0; i < ne; i++)
    x0[i] = model->wte[token_id * ne + i] + model->wpe[pos_id * ne + i];
#endif
  rmsnorm_fwd(x0, ne, x_norm1);
  memcpy(x0, x_norm1, ne * sizeof(double));

  for (int L = 0; L < N_LAYER; L++) {
    /* Attention */
    rmsnorm_fwd(x0, ne, x_norm1);
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      double sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wq[L], ne, ne, acc_buf);
      double sw = model->scale[3 + L * 6];
      for (size_t j = 0; j < ne; j++)
        q[j] = sx * sw * (double)acc_buf[j];
    }
    {
      double sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wk[L], ne, ne, acc_buf);
      double sw = model->scale[4 + L * 6];
      for (size_t j = 0; j < ne; j++)
        k[j] = sx * sw * (double)acc_buf[j];
    }
    {
      double sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wv[L], ne, ne, acc_buf);
      double sw = model->scale[5 + L * 6];
      for (size_t j = 0; j < ne; j++)
        v[j] = sx * sw * (double)acc_buf[j];
    }
#else
    lin_fwd(x_norm1, model->attn_wq[L], ne, ne, q);
    lin_fwd(x_norm1, model->attn_wk[L], ne, ne, k);
    lin_fwd(x_norm1, model->attn_wv[L], ne, ne, v);
#endif
    /* Append to cache (caller owns keys[L], values[L]; we write current at
     * cache_len[L]) */
    memcpy(keys[L] + cache_len[L] * ne, k, ne * sizeof(double));
    memcpy(values[L] + cache_len[L] * ne, v, ne * sizeof(double));

    double scale = 1.0 / sqrt((double)HEAD_DIM);
    for (size_t t = 0; t < T; t++) {
      const double *kt = (t < cache_len[L]) ? (keys[L] + t * ne) : k;
      double s = 0;
      for (size_t j = 0; j < ne; j++)
        s += x_norm1[j] * kt[j]; /* q·k for full dim; Python does per-head */
      attn_weights[t] = s * scale;
    }
    /* Causal: for single head we have one score per position; softmax over T */
    double max_s = attn_weights[0];
    for (size_t t = 1; t < T; t++)
      if (attn_weights[t] > max_s)
        max_s = attn_weights[t];
    double sum = 0;
    for (size_t t = 0; t < T; t++) {
      attn_weights[t] = exp(attn_weights[t] - max_s);
      sum += attn_weights[t];
    }
    for (size_t t = 0; t < T; t++)
      attn_weights[t] /= sum;

    for (size_t j = 0; j < ne; j++) {
      double s = 0;
      for (size_t t = 0; t < T; t++) {
        const double *vt = (t < cache_len[L]) ? (values[L] + t * ne) : v;
        s += attn_weights[t] * vt[j];
      }
      x_attn[j] = s;
    }
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      double sx = quantize_vec_to_int8(x_attn, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wo[L], ne, ne, acc_buf);
      double sw = model->scale[6 + L * 6];
      for (size_t j = 0; j < ne; j++)
        x1[j] = sx * sw * (double)acc_buf[j];
    }
#else
    lin_fwd(x_attn, model->attn_wo[L], ne, ne, x1);
#endif
    for (size_t i = 0; i < ne; i++)
      x1[i] += x0[i]; /* residual */
    memcpy(x0, x1, ne * sizeof(double));

    /* MLP */
    rmsnorm_fwd(x0, ne, x_norm2);
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      double sx = quantize_vec_to_int8(x_norm2, ne, x_i8);
      lin_fwd_int8(x_i8, model->mlp_fc1[L], ne, MLP_DIM, acc_buf);
      double sw = model->scale[7 + L * 6];
      for (size_t j = 0; j < MLP_DIM; j++)
        mlp1[j] = sx * sw * (double)acc_buf[j];
    }
    for (size_t i = 0; i < MLP_DIM; i++)
      mlp1[i] = mlp1[i] > 0 ? mlp1[i] : 0;
    {
      double sx = quantize_vec_to_int8(mlp1, MLP_DIM, x_i8);
      lin_fwd_int8(x_i8, model->mlp_fc2[L], MLP_DIM, ne, acc_buf);
      double sw = model->scale[8 + L * 6];
      for (size_t j = 0; j < ne; j++)
        x2[j] = sx * sw * (double)acc_buf[j];
    }
#else
    lin_fwd(x_norm2, model->mlp_fc1[L], ne, MLP_DIM, mlp1);
    for (size_t i = 0; i < MLP_DIM; i++)
      mlp1[i] = mlp1[i] > 0 ? mlp1[i] : 0;
    lin_fwd(mlp1, model->mlp_fc2[L], MLP_DIM, ne, x2);
#endif
    for (size_t i = 0; i < ne; i++)
      x2[i] += x0[i];
    memcpy(x0, x2, ne * sizeof(double));
  }

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  {
    double sx = quantize_vec_to_int8(x0, ne, x_i8);
    lin_fwd_int8(x_i8, model->lm_head, ne, vs, acc_buf);
    double sw = model->scale[2];
    for (size_t j = 0; j < vs; j++)
      logits[j] = sx * sw * (double)acc_buf[j];
  }
#else
  lin_fwd(x0, model->lm_head, ne, vs, logits);
#endif
  double max_l = logits[0];
  for (size_t i = 1; i < vs; i++)
    if (logits[i] > max_l)
      max_l = logits[i];
  double sum = 0;
  for (size_t i = 0; i < vs; i++) {
    logits[i] = exp(logits[i] - max_l);
    sum += logits[i];
  }
  for (size_t i = 0; i < vs; i++)
    logits[i] /= sum;
  double loss = -log(logits[target_id] > 1e-10 ? logits[target_id] : 1e-10);

  /* Backward: d_logits = probs - one_hot(target). Grad layout: wte, wpe,
   * lm_head, then per-layer wq,wk,wv,wo,fc1,fc2 */
  for (size_t i = 0; i < vs; i++)
    d_logits[i] = logits[i] - (i == target_id ? 1.0 : 0.0);

  /* Gradient through lm_head: d_x2 = lm_head^T @ d_logits, grad_lm_head +=
   * d_logits @ x0^T */
  {
    size_t off_wte = 0;
    size_t off_wpe = vs * ne;
    size_t off_lm = vs * ne + (size_t)BLOCK_SIZE * ne;
    lin_bwd(x0, get_W(model, model->lm_head, 2, vs * ne, W_tmp), d_logits, ne,
            vs, d_x2, grad_buffer + off_lm);
    for (size_t i = 0; i < ne; i++) {
      grad_buffer[off_wte + token_id * ne + i] += d_x2[i];
      grad_buffer[off_wpe + pos_id * ne + i] += d_x2[i];
    }
  }

  cache_len[0]++;
  for (int L = 1; L < N_LAYER; L++)
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
                       double **keys, double **values, size_t *cache_len,
                       double *logits_out) {
  const size_t vs = model->vocab_size;
  const size_t ne = N_EMBD;
  size_t T = cache_len[0] + 1; /* total positions including current */
#if !defined(QUANTIZATION_INT8) && !defined(QUANTISATION_INT8)
  double W_tmp[MLP_DIM * N_EMBD];
#endif
  double x0[N_EMBD], x_norm1[N_EMBD], q[N_EMBD], k[N_EMBD], v[N_EMBD];
  double attn_weights[BLOCK_SIZE], x_attn[N_EMBD], x1[N_EMBD], x_norm2[N_EMBD];
  double mlp1[MLP_DIM], x2[N_EMBD];

#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  int8_t x_i8[MLP_DIM];
  int32_t acc_buf[MAX_VOCAB];
  for (size_t i = 0; i < ne; i++)
    x0[i] = model->scale[0] * (double)model->wte[token_id * ne + i] +
            model->scale[1] * (double)model->wpe[pos_id * ne + i];
#else
  for (size_t i = 0; i < ne; i++)
    x0[i] = model->wte[token_id * ne + i] + model->wpe[pos_id * ne + i];
#endif
  rmsnorm_fwd(x0, ne, x_norm1);
  memcpy(x0, x_norm1, ne * sizeof(double));

  for (int L = 0; L < N_LAYER; L++) {
    rmsnorm_fwd(x0, ne, x_norm1);
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      double sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wq[L], ne, ne, acc_buf);
      double sw = model->scale[3 + L * 6];
      for (size_t j = 0; j < ne; j++)
        q[j] = sx * sw * (double)acc_buf[j];
    }
    {
      double sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wk[L], ne, ne, acc_buf);
      double sw = model->scale[4 + L * 6];
      for (size_t j = 0; j < ne; j++)
        k[j] = sx * sw * (double)acc_buf[j];
    }
    {
      double sx = quantize_vec_to_int8(x_norm1, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wv[L], ne, ne, acc_buf);
      double sw = model->scale[5 + L * 6];
      for (size_t j = 0; j < ne; j++)
        v[j] = sx * sw * (double)acc_buf[j];
    }
#else
    lin_fwd(x_norm1, model->attn_wq[L], ne, ne, q);
    lin_fwd(x_norm1, model->attn_wk[L], ne, ne, k);
    lin_fwd(x_norm1, model->attn_wv[L], ne, ne, v);
#endif
    memcpy(keys[L] + cache_len[L] * ne, k, ne * sizeof(double));
    memcpy(values[L] + cache_len[L] * ne, v, ne * sizeof(double));
    double scale = 1.0 / sqrt((double)HEAD_DIM);
    for (size_t t = 0; t < T; t++) {
      const double *kt = (t < cache_len[L]) ? (keys[L] + t * ne) : k;
      double s = 0;
      for (size_t j = 0; j < ne; j++)
        s += x_norm1[j] * kt[j];
      attn_weights[t] = s * scale;
    }
    double max_s = attn_weights[0];
    for (size_t t = 1; t < T; t++)
      if (attn_weights[t] > max_s)
        max_s = attn_weights[t];
    double sum = 0;
    for (size_t t = 0; t < T; t++) {
      attn_weights[t] = exp(attn_weights[t] - max_s);
      sum += attn_weights[t];
    }
    for (size_t t = 0; t < T; t++)
      attn_weights[t] /= sum;
    for (size_t j = 0; j < ne; j++) {
      double s = 0;
      for (size_t t = 0; t < T; t++) {
        const double *vt = (t < cache_len[L]) ? (values[L] + t * ne) : v;
        s += attn_weights[t] * vt[j];
      }
      x_attn[j] = s;
    }
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      double sx = quantize_vec_to_int8(x_attn, ne, x_i8);
      lin_fwd_int8(x_i8, model->attn_wo[L], ne, ne, acc_buf);
      double sw = model->scale[6 + L * 6];
      for (size_t j = 0; j < ne; j++)
        x1[j] = sx * sw * (double)acc_buf[j];
    }
#else
    lin_fwd(x_attn, model->attn_wo[L], ne, ne, x1);
#endif
    for (size_t i = 0; i < ne; i++)
      x1[i] += x0[i];
    memcpy(x0, x1, ne * sizeof(double));
    rmsnorm_fwd(x0, ne, x_norm2);
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
    {
      double sx = quantize_vec_to_int8(x_norm2, ne, x_i8);
      lin_fwd_int8(x_i8, model->mlp_fc1[L], ne, MLP_DIM, acc_buf);
      double sw = model->scale[7 + L * 6];
      for (size_t j = 0; j < MLP_DIM; j++)
        mlp1[j] = sx * sw * (double)acc_buf[j];
    }
    for (size_t i = 0; i < MLP_DIM; i++)
      mlp1[i] = mlp1[i] > 0 ? mlp1[i] : 0;
    {
      double sx = quantize_vec_to_int8(mlp1, MLP_DIM, x_i8);
      lin_fwd_int8(x_i8, model->mlp_fc2[L], MLP_DIM, ne, acc_buf);
      double sw = model->scale[8 + L * 6];
      for (size_t j = 0; j < ne; j++)
        x2[j] = sx * sw * (double)acc_buf[j];
    }
#else
    lin_fwd(x_norm2, model->mlp_fc1[L], ne, MLP_DIM, mlp1);
    for (size_t i = 0; i < MLP_DIM; i++)
      mlp1[i] = mlp1[i] > 0 ? mlp1[i] : 0;
    lin_fwd(mlp1, model->mlp_fc2[L], MLP_DIM, ne, x2);
#endif
    for (size_t i = 0; i < ne; i++)
      x2[i] += x0[i];
    memcpy(x0, x2, ne * sizeof(double));
    T = cache_len[L] + 1;
  }
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  {
    double sx = quantize_vec_to_int8(x0, ne, x_i8);
    lin_fwd_int8(x_i8, model->lm_head, ne, vs, acc_buf);
    double sw = model->scale[2];
    for (size_t j = 0; j < vs; j++)
      logits_out[j] = sx * sw * (double)acc_buf[j];
  }
#else
  lin_fwd(x0, model->lm_head, ne, vs, logits_out);
#endif
  for (int L = 0; L < N_LAYER; L++)
    cache_len[L]++;
}

/* ========================= Adam Optimiser ================================ */

/*
 * adam_step - One step of the Adam optimiser with linear LR warm-down.
 *
 *   For each parameter w_i:
 *     m_i = beta1 * m_i + (1 - beta1) * g_i          (1st moment / mean)
 *     v_i = beta2 * v_i + (1 - beta2) * g_i^2        (2nd moment / variance)
 *     m_hat = m_i / (1 - beta1^t)                     (bias correction)
 *     v_hat = v_i / (1 - beta2^t)
 *     w_i -= lr * m_hat / (sqrt(v_hat) + eps)
 *
 *   The learning rate decays linearly: lr = LR_PEAK * (1 - step / NUM_STEPS).
 *
 *   INT8 mode: updates are applied to the fp64 master copy; afterwards all
 *   weight matrices are requantised to int8 with fresh per-matrix scales.
 */
void adam_step(Model *model, const double *grads, double *m, double *v,
               int step) {
  double lr = LEARNING_RATE * (1.0 - (double)step / (double)NUM_STEPS);
  double b1 = BETA1, b2 = BETA2, eps = EPS_ADAM;
  size_t vs = model->vocab_size;
  size_t idx = 0; /* flat index into the gradient / moment arrays */
#if defined(QUANTIZATION_INT8) || defined(QUANTISATION_INT8)
  double *master = model->master;
  for (size_t i = 0; i < vs * N_EMBD; i++, idx++) {
    double g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
    double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
    master[idx] -= lr * mh / (sqrt(vh) + eps);
  }
  for (size_t i = 0; i < BLOCK_SIZE * N_EMBD; i++, idx++) {
    double g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
    double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
    master[idx] -= lr * mh / (sqrt(vh) + eps);
  }
  for (size_t i = 0; i < vs * N_EMBD; i++, idx++) {
    double g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
    double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
    master[idx] -= lr * mh / (sqrt(vh) + eps);
  }
  for (int L = 0; L < N_LAYER; L++) {
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      master[idx] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      master[idx] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      master[idx] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      master[idx] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < MLP_DIM * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      master[idx] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < N_EMBD * MLP_DIM; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      master[idx] -= lr * mh / (sqrt(vh) + eps);
    }
  }
  /* Requantize master -> int8 */
  size_t off = 0;
  quantize_fp64_to_int8(master + off, vs * N_EMBD, model->wte,
                        &model->scale[0]);
  off += vs * N_EMBD;
  quantize_fp64_to_int8(master + off, BLOCK_SIZE * N_EMBD, model->wpe,
                        &model->scale[1]);
  off += BLOCK_SIZE * N_EMBD;
  quantize_fp64_to_int8(master + off, vs * N_EMBD, model->lm_head,
                        &model->scale[2]);
  off += vs * N_EMBD;
  for (int L = 0; L < N_LAYER; L++) {
    quantize_fp64_to_int8(master + off, N_EMBD * N_EMBD, model->attn_wq[L],
                          &model->scale[3 + L * 6]);
    off += N_EMBD * N_EMBD;
    quantize_fp64_to_int8(master + off, N_EMBD * N_EMBD, model->attn_wk[L],
                          &model->scale[4 + L * 6]);
    off += N_EMBD * N_EMBD;
    quantize_fp64_to_int8(master + off, N_EMBD * N_EMBD, model->attn_wv[L],
                          &model->scale[5 + L * 6]);
    off += N_EMBD * N_EMBD;
    quantize_fp64_to_int8(master + off, N_EMBD * N_EMBD, model->attn_wo[L],
                          &model->scale[6 + L * 6]);
    off += N_EMBD * N_EMBD;
    quantize_fp64_to_int8(master + off, MLP_DIM * N_EMBD, model->mlp_fc1[L],
                          &model->scale[7 + L * 6]);
    off += MLP_DIM * N_EMBD;
    quantize_fp64_to_int8(master + off, N_EMBD * MLP_DIM, model->mlp_fc2[L],
                          &model->scale[8 + L * 6]);
    off += N_EMBD * MLP_DIM;
  }
#else
  for (size_t i = 0; i < vs * N_EMBD; i++, idx++) {
    double g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
    double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
    model->wte[i] -= lr * mh / (sqrt(vh) + eps);
  }
  for (size_t i = 0; i < BLOCK_SIZE * N_EMBD; i++, idx++) {
    double g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
    double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
    model->wpe[i] -= lr * mh / (sqrt(vh) + eps);
  }
  for (size_t i = 0; i < vs * N_EMBD; i++, idx++) {
    double g = grads[idx];
    m[idx] = b1 * m[idx] + (1 - b1) * g;
    v[idx] = b2 * v[idx] + (1 - b2) * g * g;
    double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
    double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
    model->lm_head[i] -= lr * mh / (sqrt(vh) + eps);
  }
  for (int L = 0; L < N_LAYER; L++) {
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      model->attn_wq[L][i] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      model->attn_wk[L][i] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      model->attn_wv[L][i] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < N_EMBD * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      model->attn_wo[L][i] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < MLP_DIM * N_EMBD; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      model->mlp_fc1[L][i] -= lr * mh / (sqrt(vh) + eps);
    }
    for (size_t i = 0; i < N_EMBD * MLP_DIM; i++, idx++) {
      double g = grads[idx];
      m[idx] = b1 * m[idx] + (1 - b1) * g;
      v[idx] = b2 * v[idx] + (1 - b2) * g * g;
      double mh = m[idx] / (1 - pow(b1, (double)(step + 1)));
      double vh = v[idx] / (1 - pow(b2, (double)(step + 1)));
      model->mlp_fc2[L][i] -= lr * mh / (sqrt(vh) + eps);
    }
  }
#endif
}

/* ========================= Sampling ====================================== */

/*
 * sample_token - Draw a token from a temperature-scaled softmax distribution.
 *
 *   1. Subtract max(logits) for numerical stability.
 *   2. Compute softmax: p_i = exp((logit_i - max) / temperature) / Z
 *   3. Sample from the resulting categorical distribution using the
 *      internal PRNG (rand_u).
 *
 *   Lower temperature -> more deterministic (peaky distribution).
 *   Higher temperature -> more uniform / creative.
 *
 *   Falls through to the last token if floating-point rounding prevents
 *   the cumulative sum from reaching zero.
 */
size_t sample_token(const double *logits, size_t vocab_size,
                    double temperature) {
  double buf[MAX_VOCAB];
  /* Find max logit for numerical stability in exp() */
  double max_val = logits[0];
  for (size_t i = 1; i < vocab_size; i++)
    if (logits[i] > max_val)
      max_val = logits[i];
  /* Temperature-scaled softmax */
  double sum = 0;
  for (size_t i = 0; i < vocab_size; i++) {
    buf[i] = exp((logits[i] - max_val) / temperature);
    sum += buf[i];
  }
  for (size_t i = 0; i < vocab_size; i++)
    buf[i] /= sum; /* normalise to probabilities */
  /* Inverse-CDF sampling: draw r ~ U[0,1), subtract probs until r <= 0 */
  double r = rand_u();
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

/* Internal: hash table entry for word frequency counting */
typedef struct {
  char word[MAX_WORD_LEN];
  size_t count;
} WordFreqEntry;

static unsigned int word_hash(const char *s) {
  unsigned int h = 5381;
  while (*s)
    h = h * 33 + (unsigned char)*s++;
  return h;
}

static size_t word_ht_find_or_insert(WordFreqEntry *table, size_t cap,
                                     size_t *n, const char *word) {
  unsigned int h = word_hash(word) % (unsigned int)cap;
  while (table[h].count > 0 || table[h].word[0] != '\0') {
    if (strcmp(table[h].word, word) == 0)
      return h;
    h = (h + 1) % (unsigned int)cap;
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
    if (wv->words[i])
      strcpy(wv->words[i], sorted[i].word);
  }
  /* Special tokens — strdup-like for uniform free() */
  wv->words[wv->unk_id] = (char *)malloc(6);
  if (wv->words[wv->unk_id])
    strcpy(wv->words[wv->unk_id], "<unk>");
  wv->words[wv->newline_id] = (char *)malloc(2);
  if (wv->words[wv->newline_id])
    strcpy(wv->words[wv->newline_id], "\n");
  wv->words[wv->bos_id] = (char *)malloc(6);
  if (wv->words[wv->bos_id])
    strcpy(wv->words[wv->bos_id], "<bos>");

  free(sorted);
  return 0;
}

void free_word_vocab(WordVocab *wv) {
  if (wv->words) {
    for (size_t i = 0; i < wv->vocab_size; i++)
      free(wv->words[i]);
    free(wv->words);
  }
  memset(wv, 0, sizeof(*wv));
}

size_t word_to_id(const WordVocab *wv, const char *word) {
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
