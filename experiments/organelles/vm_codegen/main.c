/*
 * MicroGPT-C — VM Code Generation with Word-Level Tokenisation
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Trains a word-level GPT on VM DSL functions.  Each token is a whole keyword,
 * identifier, number, or operator — not a single character.  This eliminates
 * spelling errors (can't produce "fun tion") and dramatically reduces sequence
 * lengths (~40 tokens per function vs ~175 characters).
 *
 * Uses vm_module_compile() as a deterministic syntax validation gate.
 *
 * Data flow:
 *   vm_functions.txt → VM-aware word tokeniser → token IDs
 *   → model training (forward_backward_one per position)
 *   → generation (sample_token → word string)
 *   → detokenize → vm_module_compile() validation gate
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_vm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MICROGPT_METAL
#include "microgpt_metal.h"
#endif

#define CODEGEN_TEMP 0.3
#define GEN_LEN 80 /* word tokens to generate per sample */
#define CHECKPOINT_FILE "vm_codegen.ckpt"
#define TRAINING_LOG "vm_codegen.ckpt.log"

#define MAX_TOKENS_PER_DOC 512
#define MAX_TOK_LEN 64

/* ====================== VM-Aware Word Scanner ========================== */
/*
 * Scans VM DSL source text and extracts one token at a time.
 * Handles: keywords, identifiers, numbers, multi-char operators,
 * single-char punctuation, newlines, and 4-space indents.
 *
 * Returns number of source characters consumed, writes token to out_tok.
 */
static int vm_scan_token(const char *p, char *out_tok, size_t max_len) {
  if (!*p)
    return 0;

  /* Newline */
  if (*p == '\n') {
    strcpy(out_tok, "\n");
    return 1;
  }
  if (*p == '\r') {
    strcpy(out_tok, "\n");
    return (*p == '\r' && *(p + 1) == '\n') ? 2 : 1;
  }

  /* 4-space indent */
  if (p[0] == ' ' && p[1] == ' ' && p[2] == ' ' && p[3] == ' ') {
    strcpy(out_tok, "    ");
    return 4;
  }

  /* Single space — skip (implicit between tokens) */
  if (*p == ' ') {
    out_tok[0] = '\0';
    return 1;
  }

  /* Comment marker */
  if (p[0] == '/' && p[1] == '/') {
    strcpy(out_tok, "//");
    return 2;
  }

  /* Multi-char operators (must check before single-char) */
  if (p[1]) {
    const char *mc[] = {"==", "!=", "<=", ">=", "&&", "||",
                        "++", "--", "+=", "-=", NULL};
    for (int i = 0; mc[i]; i++) {
      if (p[0] == mc[i][0] && p[1] == mc[i][1]) {
        strcpy(out_tok, mc[i]);
        return 2;
      }
    }
  }

  /* Single-char punctuation / operators */
  if (strchr("(){}[];:,.*+-/=<>!", *p)) {
    out_tok[0] = *p;
    out_tok[1] = '\0';
    return 1;
  }

  /* Identifier or keyword: [a-zA-Z_][a-zA-Z0-9_]* */
  if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || *p == '_') {
    int n = 0;
    while (p[n] && (size_t)n < max_len - 1 &&
           ((p[n] >= 'a' && p[n] <= 'z') || (p[n] >= 'A' && p[n] <= 'Z') ||
            (p[n] >= '0' && p[n] <= '9') || p[n] == '_')) {
      out_tok[n] = p[n];
      n++;
    }
    out_tok[n] = '\0';
    return n;
  }

  /* Number: [0-9]+(.[0-9]+)? */
  if (*p >= '0' && *p <= '9') {
    int n = 0;
    while (p[n] >= '0' && p[n] <= '9' && (size_t)n < max_len - 1) {
      out_tok[n] = p[n];
      n++;
    }
    if (p[n] == '.') {
      out_tok[n] = '.';
      n++;
      while (p[n] >= '0' && p[n] <= '9' && (size_t)n < max_len - 1) {
        out_tok[n] = p[n];
        n++;
      }
    }
    out_tok[n] = '\0';
    return n;
  }

  /* Unknown character — emit as single token */
  out_tok[0] = *p;
  out_tok[1] = '\0';
  return 1;
}

/* ====================== Corpus Loading ================================= */

static int load_docs_multiline(const char *path, Docs *docs, int max_docs) {
  FILE *f = fopen(path, "r");
  if (!f)
    return -1;
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (fsize <= 0 || fsize > 50 * 1024 * 1024) {
    fclose(f);
    return -1;
  }
  docs->data = (char *)malloc((size_t)fsize + 1);
  if (!docs->data) {
    fclose(f);
    return -1;
  }
  size_t rd = fread(docs->data, 1, (size_t)fsize, f);
  fclose(f);
  docs->data[rd] = '\0';
  docs->lines = (char **)malloc((size_t)max_docs * sizeof(char *));
  docs->doc_lens = (size_t *)malloc((size_t)max_docs * sizeof(size_t));
  if (!docs->lines || !docs->doc_lens) {
    free(docs->data);
    return -1;
  }
  docs->num_docs = 0;
  char *p = docs->data;
  while (*p && docs->num_docs < (size_t)max_docs) {
    while (*p == '\n' || *p == '\r')
      p++;
    if (!*p)
      break;
    char *start = p;
    while (*p) {
      if (*p == '\n') {
        char *next = p + 1;
        if (*next == '\r')
          next++;
        if (*next == '\n' || *next == '\0')
          break;
      }
      p++;
    }
    size_t len = (size_t)(p - start);
    while (len > 0 && (start[len - 1] == '\n' || start[len - 1] == '\r'))
      len--;
    if (len > 0) {
      docs->lines[docs->num_docs] = start;
      docs->doc_lens[docs->num_docs] = len;
      docs->num_docs++;
    }
    if (*p)
      p++;
  }
  return 0;
}

/* ====================== Word Vocabulary Builder ======================== */

/*
 * Build a word-level vocabulary by scanning ALL docs with vm_scan_token.
 * Unlike the generic build_word_vocab (whitespace-split), this properly
 * separates VM DSL tokens: keywords, identifiers, operators, numbers.
 */
static int build_vm_word_vocab(const Docs *docs, WordVocab *wv,
                               size_t max_words) {
  /* First pass: collect all unique tokens and count frequency */
  size_t cap = 4096;
  char **words = (char **)calloc(cap, sizeof(char *));
  size_t *freqs = (size_t *)calloc(cap, sizeof(size_t));
  size_t n_unique = 0;

  if (!words || !freqs) {
    free(words);
    free(freqs);
    return -1;
  }

  for (size_t d = 0; d < docs->num_docs; d++) {
    const char *p = docs->lines[d];
    const char *end = p + docs->doc_lens[d];
    char tok[MAX_TOK_LEN];

    while (p < end) {
      int consumed = vm_scan_token(p, tok, MAX_TOK_LEN);
      if (consumed <= 0)
        break;
      p += consumed;
      if (tok[0] == '\0')
        continue; /* skipped space */

      /* Find or add */
      size_t found = (size_t)-1;
      for (size_t i = 0; i < n_unique; i++) {
        if (strcmp(words[i], tok) == 0) {
          found = i;
          break;
        }
      }
      if (found != (size_t)-1) {
        freqs[found]++;
      } else {
        if (n_unique >= cap) {
          cap *= 2;
          words = (char **)realloc(words, cap * sizeof(char *));
          freqs = (size_t *)realloc(freqs, cap * sizeof(size_t));
        }
        words[n_unique] = strdup(tok);
        freqs[n_unique] = 1;
        n_unique++;
      }
    }
  }

  /* Sort by frequency (descending) to keep most common */
  for (size_t i = 0; i < n_unique; i++) {
    for (size_t j = i + 1; j < n_unique; j++) {
      if (freqs[j] > freqs[i]) {
        char *tw = words[i];
        words[i] = words[j];
        words[j] = tw;
        size_t tf = freqs[i];
        freqs[i] = freqs[j];
        freqs[j] = tf;
      }
    }
  }

  /* Keep top max_words, build the WordVocab */
  size_t keep = n_unique < max_words ? n_unique : max_words;

  wv->num_words = keep;
  wv->vocab_size = keep + 3; /* +unk, +newline, +bos */
  wv->unk_id = keep;
  wv->newline_id = keep + 1;
  wv->bos_id = keep + 2;

  wv->words = (char **)calloc(wv->vocab_size, sizeof(char *));
  if (!wv->words)
    return -1;

  for (size_t i = 0; i < keep; i++) {
    /* Check if this is the newline token — if so, skip and use the special
     * slot */
    if (strcmp(words[i], "\n") == 0) {
      /* Swap with next non-\n word to keep the count right */
      wv->words[i] = strdup(words[i]);
      continue;
    }
    wv->words[i] = strdup(words[i]);
  }
  wv->words[wv->unk_id] = strdup("<unk>");
  wv->words[wv->newline_id] = strdup("\n");
  wv->words[wv->bos_id] = strdup("<bos>");

  /* Build hash table for O(1) lookup */
  wv->ht_cap = wv->vocab_size * 4; /* load factor ~25% */
  wv->ht_keys = (char **)calloc(wv->ht_cap, sizeof(char *));
  wv->ht_ids = (size_t *)calloc(wv->ht_cap, sizeof(size_t));

  for (size_t i = 0; i < wv->vocab_size; i++) {
    if (!wv->words[i])
      continue;
    /* DJB2 hash — must match word_hash() in microgpt.c for word_to_id() */
    unsigned int h = 5381;
    for (const char *s = wv->words[i]; *s; s++)
      h = h * 33 + (unsigned char)*s;
    size_t slot = h % wv->ht_cap;
    while (wv->ht_keys[slot])
      slot = (slot + 1) % wv->ht_cap;
    wv->ht_keys[slot] = wv->words[i];
    wv->ht_ids[slot] = i;
  }

  /* Cleanup frequency data */
  for (size_t i = 0; i < n_unique; i++)
    free(words[i]);
  free(words);
  free(freqs);

  return 0;
}

/*
 * VM-aware tokenise: scan text with vm_scan_token, look up each in WordVocab.
 */
static size_t vm_tokenize(const char *text, size_t text_len,
                          const WordVocab *wv, size_t *ids, size_t max_tokens) {
  size_t k = 0;
  const char *p = text;
  const char *end = text + text_len;
  char tok[MAX_TOK_LEN];

  while (p < end && k < max_tokens) {
    int consumed = vm_scan_token(p, tok, MAX_TOK_LEN);
    if (consumed <= 0)
      break;
    p += consumed;
    if (tok[0] == '\0')
      continue; /* skipped space */

    /* Newline uses special ID */
    if (strcmp(tok, "\n") == 0) {
      ids[k++] = wv->newline_id;
    } else {
      ids[k++] = word_to_id(wv, tok);
    }
  }
  return k;
}

/*
 * Detokenize: convert token IDs back to text by concatenating word strings.
 * Inserts a space between non-whitespace tokens for readability.
 */
static void vm_detokenize(const size_t *tokens, size_t count,
                          const WordVocab *wv, char *buf, size_t bufsize) {
  size_t pos = 0;
  int prev_was_newline_or_indent = 1; /* no leading space at start */

  for (size_t i = 0; i < count && pos < bufsize - MAX_TOK_LEN - 2; i++) {
    if (tokens[i] >= wv->vocab_size)
      continue;
    if (tokens[i] == wv->bos_id)
      continue;

    const char *w = wv->words[tokens[i]];
    if (!w)
      continue;
    size_t wlen = strlen(w);

    int is_newline = (tokens[i] == wv->newline_id);
    int is_indent = (strcmp(w, "    ") == 0);
    int is_open = (strcmp(w, "(") == 0 || strcmp(w, "{") == 0);
    int is_close =
        (strcmp(w, ")") == 0 || strcmp(w, "}") == 0 || strcmp(w, ";") == 0 ||
         strcmp(w, ":") == 0 || strcmp(w, ",") == 0);

    /* Add space before this token? */
    if (!prev_was_newline_or_indent && !is_newline && !is_indent && !is_close &&
        pos > 0) {
      buf[pos++] = ' ';
    }

    memcpy(buf + pos, w, wlen);
    pos += wlen;

    prev_was_newline_or_indent = is_newline || is_indent || is_open;
  }
  buf[pos] = '\0';
}

/* ====================== VM Syntax Validation =========================== */

static int validate_vm_code(const char *code) {
  vm_module *module = NULL;
  result r = vm_module_compile(NULL, code, &module);
  int valid = (r == RESULT_OK && module && sequence_count(module->errors) == 0);
  if (module)
    vm_module_dispose(module);
  return valid;
}

/* ====================== Main =========================================== */

int main(void) {
  unsigned int train_seed = 42;
  unsigned int infer_seed = (unsigned int)time(NULL);
  srand(train_seed);
  seed_rng(train_seed);

  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = 96;
  cfg.n_head = 4;
  cfg.mlp_dim = 384;
  cfg.n_layer = 2;
  cfg.block_size = 256;
  cfg.batch_size = 16;
  cfg.num_steps = 15000;
  cfg.learning_rate = 0.0005;
  cfg.max_vocab = 1200; /* word tokens — expanded vocab for domain coverage */
  cfg.max_docs = 5000;
  cfg.max_doc_len = 512;
  microgpt_print_config("MicroGPT-C - VM Code Generation (Word-Level)", &cfg);

  const int nl = cfg.n_layer;

#ifdef MICROGPT_METAL
  if (metal_init() != 0)
    fprintf(stderr, "WARNING: Metal GPU init failed, falling back to CPU\n");
#endif

  /* ---- Load VM functions corpus ---- */
  Docs docs = {0};
  if (load_docs_multiline("vm_functions_combined.txt", &docs, cfg.max_docs) !=
      0) {
    fprintf(stderr, "Cannot open vm_functions_combined.txt\n");
    return 1;
  }
  shuffle_docs(&docs);
  printf("loaded %zu VM functions\n", docs.num_docs);

  size_t total_chars = 0;
  for (size_t i = 0; i < docs.num_docs; i++)
    total_chars += docs.doc_lens[i];
  printf("total characters: %zu (%.1f KB)\n", total_chars,
         (scalar_t)total_chars / 1024.0);

  /* ---- Build word-level vocabulary ---- */
  WordVocab wv = {0};
  if (build_vm_word_vocab(&docs, &wv, (size_t)cfg.max_vocab - 3) != 0) {
    fprintf(stderr, "Failed to build word vocabulary\n");
    return 1;
  }
  printf("word vocab: %zu tokens (%zu words + 3 special)\n", wv.vocab_size,
         wv.num_words);

  /* ---- Pre-tokenize all documents ---- */
  size_t **doc_tokens = (size_t **)calloc(docs.num_docs, sizeof(size_t *));
  size_t *doc_tok_lens = (size_t *)calloc(docs.num_docs, sizeof(size_t));
  size_t total_word_tokens = 0;
  size_t unk_count = 0;

  for (size_t d = 0; d < docs.num_docs; d++) {
    doc_tokens[d] = (size_t *)malloc(MAX_TOKENS_PER_DOC * sizeof(size_t));
    doc_tok_lens[d] = vm_tokenize(docs.lines[d], docs.doc_lens[d], &wv,
                                  doc_tokens[d], MAX_TOKENS_PER_DOC);
    total_word_tokens += doc_tok_lens[d];
    for (size_t t = 0; t < doc_tok_lens[d]; t++)
      if (doc_tokens[d][t] == wv.unk_id)
        unk_count++;
  }
  printf("total word tokens: %zu (avg %.1f per function)\n", total_word_tokens,
         (double)total_word_tokens / docs.num_docs);
  printf("unknown tokens: %zu (%.2f%%)\n", unk_count,
         total_word_tokens > 0 ? 100.0 * unk_count / total_word_tokens : 0.0);

  /* Print some example tokenisations */
  printf("\n--- example tokenisations ---\n");
  for (size_t d = 0; d < 3 && d < docs.num_docs; d++) {
    printf("doc[%zu] (%zu tokens): ", d, doc_tok_lens[d]);
    for (size_t t = 0; t < doc_tok_lens[d] && t < 20; t++) {
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
    if (doc_tok_lens[d] > 20)
      printf("...");
    printf("\n");
  }
  printf("\n");

  /* ---- Training log ---- */
  FILE *logf = fopen(TRAINING_LOG, "a");
  if (logf) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    fprintf(logf, "\n========================================\n");
    fprintf(logf, "Run: %04d-%02d-%02d %02d:%02d:%02d\n", t->tm_year + 1900,
            t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    fprintf(logf, "========================================\n");
    fprintf(logf, "Tokenisation: WORD-LEVEL\n");
    fprintf(logf, "Corpus: %zu functions | %zu word tokens\n", docs.num_docs,
            total_word_tokens);
    fprintf(logf, "Vocab: %zu tokens\n", wv.vocab_size);
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
    fprintf(logf, "Params: %zu\n", nparams);
    fprintf(logf, "Training: batch=%d steps=%d lr=%.4f\n", cfg.batch_size,
            cfg.num_steps, (scalar_t)cfg.learning_rate);
    fflush(logf);
  }

  /* ---- Allocate KV caches for training ---- */
  scalar_t **train_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **train_values =
      (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *train_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    train_keys[L] = kv_cache_alloc(&cfg);
    train_values[L] = kv_cache_alloc(&cfg);
  }

  /* ---- Training (single-threaded for simplicity with word-level) ---- */
  if (!trained) {
    if (logf)
      fprintf(logf, "\n--- Training ---\n");
    size_t tokens_trained = 0;
    time_t t0 = time(NULL);
    scalar_t best_loss = 1e9;
    char best_ckpt[256];
    best_ckpt[0] = '\0';

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

        /* Feed BOS then each token, computing loss on positions 1..seq_len-1
         */
        /* Position 0: feed BOS, target = first doc token */
        step_loss += forward_backward_one(
            model, wv.bos_id, 0, doc_tokens[di][0], train_keys, train_values,
            train_cache_len, grad_buffer);
        step_positions++;

        /* Positions 1..seq_len-1: feed token[i-1], target = token[i] */
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
        scalar_t eta = (cfg.num_steps - step - 1) / ((step + 1) / elapsed);
        int el_m = (int)elapsed / 60, el_s = (int)elapsed % 60;
        int eta_m = (int)eta / 60, eta_s = (int)eta % 60;
        printf("step %5d / %d | loss %.4f | %dm%02ds elapsed, ETA %dm%02ds",
               step + 1, cfg.num_steps, mean_loss, el_m, el_s, eta_m, eta_s);
        fflush(stdout);

        if (mean_loss < best_loss && step > 0) {
          best_loss = mean_loss;
          snprintf(best_ckpt, sizeof(best_ckpt),
                   "vm_codegen_best_step%d_loss%.4f.ckpt", step + 1, mean_loss);
          checkpoint_save(model, m_buf, v_buf, step + 1, best_ckpt);
          checkpoint_save(model, m_buf, v_buf, step + 1, CHECKPOINT_FILE);
          printf(" [BEST -> %s]", best_ckpt);
        }
        printf("\n");
        fflush(stdout);

        if (logf) {
          fprintf(logf, "step %5d | loss %.4f", step + 1, mean_loss);
          if (mean_loss < best_loss || mean_loss == best_loss)
            fprintf(logf, " [BEST]");
          fprintf(logf, "\n");
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
    checkpoint_save(model, m_buf, v_buf, cfg.num_steps, CHECKPOINT_FILE);
    printf("final checkpoint saved to '%s'\n", CHECKPOINT_FILE);

    if (logf) {
      fprintf(logf, "\nTraining complete: %.1fs\n", train_sec);
      fprintf(logf, "Best loss: %.4f\n", best_loss);
      fflush(logf);
    }
  }

  /* ---- Inference ---- */
  seed_rng(infer_seed);
  printf("\n--- generated VM functions (word-level, with syntax validation) "
         "---\n");
  fflush(stdout);

  scalar_t *logits_buf = (scalar_t *)malloc(wv.vocab_size * sizeof(scalar_t));

  /* KV cache for inference */
  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    inf_keys[L] = kv_cache_alloc(&cfg);
    inf_values[L] = kv_cache_alloc(&cfg);
  }

  int valid_count = 0, invalid_count = 0, skipped_count = 0;

  const char *prompts[] = {
      /* Novel compositions */
      "// compute cube root\n",
      "// calculate total with discount\n",
      "// find larger of three numbers\n",
      "// compute average of three numbers\n",
      "// compute sum of cubes from 1 to n\n",
      /* Controls — exact corpus matches */
      "// compute factorial of n\n",
      "// compute absolute value\n",
      "// compute simple interest\n",
      "// compute sum of numbers from 1 to n\n",
      "// compute circle area from radius\n",
  };
  int num_prompts = (int)(sizeof(prompts) / sizeof(prompts[0]));

  for (int s = 0; s < num_prompts; s++) {
    for (int L = 0; L < nl; L++)
      inf_cache_len[L] = 0;

    const char *prompt = prompts[s];
    size_t prompt_len = strlen(prompt);

    /* Tokenize the prompt with VM scanner */
    size_t prompt_tokens[MAX_TOKENS_PER_DOC];
    size_t n_prompt =
        vm_tokenize(prompt, prompt_len, &wv, prompt_tokens, MAX_TOKENS_PER_DOC);

    /* Feed prompt through model */
    size_t pos = 0;
    scalar_t prompt_conf = 0;

    /* BOS */
    forward_inference(model, wv.bos_id, pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;

    /* Feed each prompt token, tracking confidence */
    for (size_t pi = 0; pi < n_prompt && pos < (size_t)cfg.block_size - GEN_LEN;
         pi++) {
      /* Confidence: softmax prob of the actual next token */
      scalar_t max_v = logits_buf[0];
      for (size_t c = 1; c < wv.vocab_size; c++)
        if (logits_buf[c] > max_v)
          max_v = logits_buf[c];
      scalar_t sum = 0;
      for (size_t c = 0; c < wv.vocab_size; c++)
        sum += exp(logits_buf[c] - max_v);
      scalar_t prob = exp(logits_buf[prompt_tokens[pi]] - max_v) / sum;
      prompt_conf += prob;

      forward_inference(model, prompt_tokens[pi], pos, inf_keys, inf_values,
                        inf_cache_len, logits_buf);
      pos++;
    }
    prompt_conf /= (n_prompt > 0 ? (scalar_t)n_prompt : 1.0);

    /* Display prompt info */
    char pdisplay[256];
    snprintf(pdisplay, sizeof(pdisplay), "%s", prompt);
    size_t pdl = strlen(pdisplay);
    while (pdl > 0 && (pdisplay[pdl - 1] == '\n' || pdisplay[pdl - 1] == '\r'))
      pdisplay[--pdl] = '\0';

    const char *conf_icon = prompt_conf >= 0.60   ? "HIGH"
                            : prompt_conf >= 0.30 ? "MEDIUM"
                            : prompt_conf >= 0.15 ? "LOW"
                                                  : "NONE";
    printf("\n[%.0f%% %s] %s\n", prompt_conf * 100.0, conf_icon, pdisplay);
    fflush(stdout);

    if (prompt_conf < 0.10) {
      printf("  >> prompt not recognized — skipping\n");
      skipped_count++;
      continue;
    }

    /* Generate word tokens with brace-balanced stopping */
    size_t gen_tokens[GEN_LEN + 1];
    int gen_count = 0;
    int brace_depth = 0;
    int saw_open_brace = 0;

    /* Look up { and } token IDs for brace tracking */
    size_t open_brace_id = word_to_id(&wv, "{");
    size_t close_brace_id = word_to_id(&wv, "}");

    for (int g = 0; g < GEN_LEN && pos < (size_t)cfg.block_size; g++) {
      size_t tok = sample_token(logits_buf, wv.vocab_size, CODEGEN_TEMP);
      if (tok == wv.bos_id)
        break;
      gen_tokens[gen_count++] = tok;

      /* Track brace depth */
      if (tok == open_brace_id) {
        brace_depth++;
        saw_open_brace = 1;
      } else if (tok == close_brace_id) {
        brace_depth--;
        /* Stop when braces balance (function body complete) */
        if (saw_open_brace && brace_depth <= 0)
          break;
      }

      forward_inference(model, tok, pos, inf_keys, inf_values, inf_cache_len,
                        logits_buf);
      pos++;
    }

    /* Detokenize generated code */
    char gen_text[4096];
    vm_detokenize(gen_tokens, (size_t)gen_count, &wv, gen_text,
                  sizeof(gen_text));

    printf("%s\n", gen_text);
    fflush(stdout);

    /* Build full code: prompt + generated */
    char full_code[8192];
    snprintf(full_code, sizeof(full_code), "%s%s", prompt, gen_text);

    if (validate_vm_code(full_code)) {
      printf("  >> VM SYNTAX: ✅ VALID (%d word tokens)\n", gen_count);
      valid_count++;
    } else {
      printf("  >> VM SYNTAX: ❌ INVALID (%d word tokens)\n", gen_count);
      invalid_count++;
    }
    fflush(stdout);
  }

  /* ---- Summary ---- */
  int total_val = valid_count + invalid_count;
  printf("\n--- inference summary ---\n");
  printf("syntax validation: %d/%d valid (%.0f%%) | %d skipped\n", valid_count,
         total_val, total_val > 0 ? 100.0 * valid_count / total_val : 0.0,
         skipped_count);
  fflush(stdout);

  if (logf) {
    fprintf(logf, "\nSyntax validation: %d/%d valid (%.0f%%)\n", valid_count,
            total_val, total_val > 0 ? 100.0 * valid_count / total_val : 0.0);
    fprintf(logf, "========================================\n\n");
    fclose(logf);
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
  free_docs(&docs);
  free_word_vocab(&wv);
  return 0;
}
