/*
 * MicroGPT-C — VM Compose Pipeline (Phase 3)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Generate → Validate loop for VM DSL code generation.
 * Loads a trained vm_codegen checkpoint, generates VM DSL from natural
 * language intent comments, validates via vm_module_compile(), and
 * retries on failure (up to MAX_RETRIES attempts per intent).
 *
 * This is the minimal Phase 3 pipeline — proving that word-level
 * generation + deterministic VM validation works end-to-end.
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

#define GEN_LEN 80
#define MAX_CANDIDATES 10
#define MAX_RETRIES MAX_CANDIDATES /* backward compat for display strings */
#define CODEGEN_TEMP 0.3f          /* fallback for constrained decoding path */
#define MAX_TOKENS_PER_DOC 512
#define DEBUG_GEN 0 /* Set to 1 for raw token/code diagnostics */

/* Temperature schedule for best-of-N: explore diverse retrieval paths */
static const float CANDIDATE_TEMPS[MAX_CANDIDATES] = {
    0.3f, 0.4f, 0.5f, 0.5f, 0.6f, 0.6f, 0.7f, 0.7f, 0.8f, 0.5f};
#define MAX_TOK_LEN 64

/* ====================== VM-Aware Word Scanner ========================== */

static int vm_scan_token(const char *p, char *out_tok, size_t max_len) {
  if (!*p)
    return 0;

  if (*p == '\n') {
    strcpy(out_tok, "\n");
    return 1;
  }
  if (*p == '\r') {
    strcpy(out_tok, "\n");
    return (*p == '\r' && *(p + 1) == '\n') ? 2 : 1;
  }

  if (p[0] == ' ' && p[1] == ' ' && p[2] == ' ' && p[3] == ' ') {
    strcpy(out_tok, "    ");
    return 4;
  }

  if (*p == ' ') {
    out_tok[0] = '\0';
    return 1;
  }

  if (p[0] == '/' && p[1] == '/') {
    strcpy(out_tok, "//");
    return 2;
  }

  /* Multi-char operators */
  if ((p[0] == '<' && p[1] == '=') || (p[0] == '>' && p[1] == '=') ||
      (p[0] == '!' && p[1] == '=') || (p[0] == '=' && p[1] == '=') ||
      (p[0] == '+' && p[1] == '+') || (p[0] == '-' && p[1] == '-') ||
      (p[0] == '&' && p[1] == '&') || (p[0] == '|' && p[1] == '|')) {
    out_tok[0] = p[0];
    out_tok[1] = p[1];
    out_tok[2] = '\0';
    return 2;
  }

  /* Single punctuation / operator */
  if (*p == '(' || *p == ')' || *p == '{' || *p == '}' || *p == ';' ||
      *p == ':' || *p == ',' || *p == '+' || *p == '-' || *p == '*' ||
      *p == '/' || *p == '%' || *p == '=' || *p == '<' || *p == '>' ||
      *p == '!' || *p == '[' || *p == ']') {
    out_tok[0] = *p;
    out_tok[1] = '\0';
    return 1;
  }

  /* Number */
  if (*p >= '0' && *p <= '9') {
    int n = 0;
    while (p[n] && ((p[n] >= '0' && p[n] <= '9') || p[n] == '.') &&
           n < (int)max_len - 1) {
      out_tok[n] = p[n];
      n++;
    }
    out_tok[n] = '\0';
    return n;
  }

  /* Identifier or keyword */
  if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || *p == '_') {
    int n = 0;
    while (p[n] &&
           (((p[n] >= 'a') && (p[n] <= 'z')) ||
            ((p[n] >= 'A') && (p[n] <= 'Z')) ||
            ((p[n] >= '0') && (p[n] <= '9')) || p[n] == '_') &&
           n < (int)max_len - 1) {
      out_tok[n] = p[n];
      n++;
    }
    out_tok[n] = '\0';
    return n;
  }

  /* Unknown character — skip */
  out_tok[0] = '\0';
  return 1;
}

/* ====================== Corpus Loading ================================= */

static int load_docs_multiline(const char *path, Docs *docs, int max_docs) {
  FILE *f = fopen(path, "r");
  if (!f)
    return -1;

  /* Pre-allocate arrays */
  docs->lines = (char **)calloc((size_t)max_docs, sizeof(char *));
  docs->doc_lens = (size_t *)calloc((size_t)max_docs, sizeof(size_t));
  if (!docs->lines || !docs->doc_lens) {
    fclose(f);
    return -1;
  }

  char line[2048];
  char *current = NULL;
  size_t current_len = 0;
  size_t current_cap = 0;

  while (fgets(line, sizeof(line), f)) {
    size_t ll = strlen(line);
    while (ll > 0 && (line[ll - 1] == '\n' || line[ll - 1] == '\r'))
      ll--;

    /* Blank line = doc separator */
    if (ll == 0) {
      if (current && current_len > 0 && (int)docs->num_docs < max_docs) {
        docs->lines[docs->num_docs] = current;
        docs->doc_lens[docs->num_docs] = current_len;
        docs->num_docs++;
        current = NULL;
        current_len = 0;
        current_cap = 0;
      }
      continue;
    }

    /* Grow buffer */
    size_t need = current_len + ll + 2;
    if (need > current_cap) {
      current_cap = need * 2;
      current = (char *)realloc(current, current_cap);
    }
    if (current_len > 0)
      current[current_len++] = '\n';
    memcpy(current + current_len, line, ll);
    current_len += ll;
    current[current_len] = '\0';
  }

  /* Final doc */
  if (current && current_len > 0 && (int)docs->num_docs < max_docs) {
    docs->lines[docs->num_docs] = current;
    docs->doc_lens[docs->num_docs] = current_len;
    docs->num_docs++;
  } else {
    free(current);
  }

  fclose(f);
  return 0;
}

/* ====================== Word Vocabulary Builder ======================== */

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
    if (strcmp(words[i], "\n") == 0) {
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

/* ====================== Tokenisation ================================== */

static size_t vm_tokenize(const char *text, size_t text_len,
                          const WordVocab *wv, size_t *ids, size_t max_tokens) {
  const char *p = text;
  const char *end = text + text_len;
  size_t count = 0;
  char tok[MAX_TOK_LEN];

  while (p < end && count < max_tokens) {
    int consumed = vm_scan_token(p, tok, sizeof(tok));
    if (consumed <= 0)
      break;
    p += consumed;
    if (tok[0] == '\0')
      continue;

    ids[count++] = word_to_id(wv, tok);
  }
  return count;
}

/* ====================== Detokenisation ================================ */

static void vm_detokenize(const size_t *tokens, size_t count,
                          const WordVocab *wv, char *buf, size_t bufsize) {
  size_t pos = 0;
  for (size_t i = 0; i < count && pos + 2 < bufsize; i++) {
    size_t tid = tokens[i];
    if (tid == wv->bos_id)
      continue;

    const char *w = NULL;
    if (tid < wv->vocab_size && wv->words[tid])
      w = wv->words[tid];
    else
      w = "<unk>";

    size_t wlen = strlen(w);

    /* Spacing heuristics */
    int is_newline = (tid == wv->newline_id || (wlen == 1 && w[0] == '\n'));
    int is_indent = (strcmp(w, "    ") == 0);
    int is_open_paren = (wlen == 1 && w[0] == '(');
    int is_close = (wlen == 1 &&
                    (w[0] == ')' || w[0] == ';' || w[0] == ',' || w[0] == ':'));

    if (is_newline) {
      if (pos + 1 < bufsize)
        buf[pos++] = '\n';
      continue;
    }

    if (pos > 0 && !is_indent && !is_open_paren && !is_close &&
        buf[pos - 1] != '\n' && buf[pos - 1] != '(') {
      buf[pos++] = ' ';
    }

    for (size_t c = 0; c < wlen && pos + 1 < bufsize; c++)
      buf[pos++] = w[c];
  }
  buf[pos] = '\0';
}

/* ====================== VM Syntax Validation =========================== */

static int validate_vm_code(const char *code) {
  vm_module *module = NULL;
  vm_result r = vm_module_compile(NULL, code, &module);
  int valid = (r == VM_OK && module && vm_list_count(module->errors) == 0);
  if (module)
    vm_module_dispose(module);
  return valid;
}

/* ====================== Intent Loading ================================= */

static int load_intents(const char *path, char intents[][256], int max) {
  FILE *f = fopen(path, "r");
  if (!f)
    return 0;

  int count = 0;
  char line[256];
  while (fgets(line, sizeof(line), f) && count < max) {
    size_t ll = strlen(line);
    while (ll > 0 && (line[ll - 1] == '\n' || line[ll - 1] == '\r'))
      ll--;
    if (ll == 0)
      continue;
    memcpy(intents[count], line, ll);
    intents[count][ll] = '\0';
    count++;
  }
  fclose(f);
  return count;
}

/* ====================== Generation + Validation ======================== */

typedef struct {
  int valid;
  int attempts;
  double confidence;
  char code[4096];
} ComposeResult;

static ComposeResult generate_and_validate(
    Model *model, const WordVocab *wv, const MicrogptConfig *cfg,
    const char *intent, scalar_t *logits_buf, scalar_t **inf_keys,
    scalar_t **inf_values, size_t *inf_cache_len, unsigned int *rng_state) {

  ComposeResult res = {0};
  int nl = cfg->n_layer;

  /* Add newline to intent for prompting */
  char prompt[512];
  snprintf(prompt, sizeof(prompt), "%s\n", intent);
  size_t prompt_len = strlen(prompt);

  ComposeResult best = {0};
  best.confidence = -1.0;

  for (int attempt = 0; attempt < MAX_CANDIDATES; attempt++) {
    res.attempts = attempt + 1;
    float temp = CANDIDATE_TEMPS[attempt];

    /* Reset KV cache */
    for (int L = 0; L < nl; L++)
      inf_cache_len[L] = 0;

    /* Seed RNG differently per attempt */
    seed_rng(*rng_state + (unsigned int)attempt * 7919);

    /* Tokenize prompt */
    size_t prompt_tokens[MAX_TOKENS_PER_DOC];
    size_t n_prompt =
        vm_tokenize(prompt, prompt_len, wv, prompt_tokens, MAX_TOKENS_PER_DOC);

    /* Feed BOS */
    size_t pos = 0;
    forward_inference(model, wv->bos_id, pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;

    /* Feed prompt tokens, track confidence */
    scalar_t prompt_conf = 0;
    for (size_t pi = 0;
         pi < n_prompt && pos < (size_t)cfg->block_size - GEN_LEN; pi++) {
      scalar_t max_v = logits_buf[0];
      for (size_t c = 1; c < wv->vocab_size; c++)
        if (logits_buf[c] > max_v)
          max_v = logits_buf[c];
      scalar_t sum = 0;
      for (size_t c = 0; c < wv->vocab_size; c++)
        sum += (scalar_t)exp(logits_buf[c] - max_v);
      scalar_t prob =
          (scalar_t)exp(logits_buf[prompt_tokens[pi]] - max_v) / sum;
      prompt_conf += prob;

      forward_inference(model, prompt_tokens[pi], pos, inf_keys, inf_values,
                        inf_cache_len, logits_buf);
      pos++;
    }
    res.confidence =
        (double)(prompt_conf / (n_prompt > 0 ? (scalar_t)n_prompt : 1.0f));

    /* Generate with brace-balanced stopping */
    size_t gen_tokens[GEN_LEN + 1];
    int gen_count = 0;
    int brace_depth = 0;
    int saw_open = 0;
    size_t open_id = word_to_id(wv, "{");
    size_t close_id = word_to_id(wv, "}");
    size_t comment_id = word_to_id(wv, "//");

    for (int g = 0; g < GEN_LEN && pos < (size_t)cfg->block_size; g++) {
      size_t tok = sample_token(logits_buf, wv->vocab_size, temp);
      if (tok == wv->bos_id)
        break;

      /* Stop if we see '//' after closing brace — spilling into next function
       */
      if (tok == comment_id && saw_open && brace_depth <= 0)
        break;

      gen_tokens[gen_count++] = tok;

      if (tok == open_id) {
        brace_depth++;
        saw_open = 1;
      } else if (tok == close_id) {
        brace_depth--;
        if (saw_open && brace_depth <= 0)
          break;
      }

      forward_inference(model, tok, pos, inf_keys, inf_values, inf_cache_len,
                        logits_buf);
      pos++;
    }

    /* Detokenize */
    char gen_text[4096];
    vm_detokenize(gen_tokens, (size_t)gen_count, wv, gen_text,
                  sizeof(gen_text));

    /* Build full code */
    snprintf(res.code, sizeof(res.code), "%s%s", prompt, gen_text);

#if DEBUG_GEN
    fprintf(stderr, "\n[DEBUG] intent: %s\n", intent);
    fprintf(stderr,
            "[DEBUG] attempt %d (temp=%.1f), %d tokens, brace_depth=%d:\n",
            attempt + 1, temp, gen_count, brace_depth);
    fprintf(stderr, "[DEBUG] raw tokens: ");
    for (int t = 0; t < gen_count && t < 30; t++) {
      if (gen_tokens[t] < wv->num_words && wv->words[gen_tokens[t]])
        fprintf(stderr, "[%s]", wv->words[gen_tokens[t]]);
      else
        fprintf(stderr, "[?%zu]", gen_tokens[t]);
    }
    fprintf(stderr, "\n[DEBUG] code:\n%s\n[/DEBUG]\n", res.code);
#endif

    /* Validate — keep best valid candidate */
    if (validate_vm_code(res.code)) {
      res.valid = 1;
      if (res.confidence > best.confidence) {
        best = res;
      }
    }

    /* Retry with different seed */
    *rng_state += 13;
  }

  /* Return best valid candidate, or last failed attempt */
  if (best.valid)
    return best;
  return res;
}

/* ====================== Option C: Constrained Decoding ================= */
/*
 * Grammar state machine for VM DSL.  Tracks expected token categories
 * and masks logits of grammatically impossible tokens before sampling.
 *
 * States: COMMENT, FUNC_KW, FUNC_NAME, PARAMS, BODY, EXPR, DONE
 */

typedef enum {
  GS_COMMENT,    /* inside // ... \n */
  GS_FUNC_KW,    /* expecting 'function' keyword */
  GS_FUNC_NAME,  /* expecting function name (identifier) */
  GS_PARAMS,     /* inside parameter list (...) */
  GS_COLON_RET,  /* expecting ':' before return type */
  GS_RET_TYPE,   /* expecting return type (number/string) */
  GS_OPEN_BRACE, /* expecting '{' */
  GS_BODY,       /* inside function body */
  GS_DONE        /* generation complete */
} GrammarState;

static int is_identifier_tok(const char *tok) {
  if (!tok || tok[0] == '\0')
    return 0;
  /* Identifiers start with letter or underscore */
  if ((tok[0] >= 'a' && tok[0] <= 'z') || (tok[0] >= 'A' && tok[0] <= 'Z') ||
      tok[0] == '_')
    return 1;
  return 0;
}

static int is_keyword_tok(const char *tok) {
  return tok && (strcmp(tok, "function") == 0 || strcmp(tok, "var") == 0 ||
                 strcmp(tok, "if") == 0 || strcmp(tok, "for") == 0 ||
                 strcmp(tok, "return") == 0 || strcmp(tok, "number") == 0 ||
                 strcmp(tok, "string") == 0);
}

static int is_operator_tok(const char *tok) {
  if (!tok)
    return 0;
  return (strcmp(tok, "+") == 0 || strcmp(tok, "-") == 0 ||
          strcmp(tok, "*") == 0 || strcmp(tok, "/") == 0 ||
          strcmp(tok, "=") == 0 || strcmp(tok, "==") == 0 ||
          strcmp(tok, "!=") == 0 || strcmp(tok, "<") == 0 ||
          strcmp(tok, ">") == 0 || strcmp(tok, "<=") == 0 ||
          strcmp(tok, ">=") == 0 || strcmp(tok, "++") == 0);
}

/*
 * Apply grammar constraints: set logits to -1e9 for impossible tokens.
 * Returns updated grammar state based on most recent token.
 */
static GrammarState apply_grammar_mask(scalar_t *logits, const WordVocab *wv,
                                       GrammarState state, size_t last_tok,
                                       int brace_depth) {

  (void)last_tok; /* used for context in future refinements */

  /* Strategy: rather than enumerate all rules, use a light-touch approach:
   * - After '{' or statement start: allow identifiers, keywords, '}'
   * - After identifier: allow operators, '(', ')', ',', ':', ';', '\n'
   * - After operator: allow identifiers, numbers, '('
   * - After 'function': allow identifiers only
   * - After 'return': allow identifiers, numbers, '('
   * This prevents nonsense like ")) ++ function { number" */

  if (state == GS_DONE)
    return GS_DONE;

  const char *last_word = NULL;
  if (last_tok < wv->num_words && wv->words[last_tok])
    last_word = wv->words[last_tok];

  /* After 'function' keyword → only identifiers allowed */
  if (last_word && strcmp(last_word, "function") == 0) {
    for (size_t i = 0; i < wv->vocab_size; i++) {
      if (i >= wv->num_words) {
        logits[i] = -1e9f;
        continue;
      }
      if (!wv->words[i]) {
        logits[i] = -1e9f;
        continue;
      }
      if (!is_identifier_tok(wv->words[i]) || is_keyword_tok(wv->words[i]))
        logits[i] = -1e9f;
    }
    return GS_FUNC_NAME;
  }

  /* After return/var → allow identifiers, numbers, '(' */
  if (last_word &&
      (strcmp(last_word, "return") == 0 || strcmp(last_word, "var") == 0)) {
    for (size_t i = 0; i < wv->vocab_size; i++) {
      if (i >= wv->num_words) {
        logits[i] = -1e9f;
        continue;
      }
      if (!wv->words[i]) {
        logits[i] = -1e9f;
        continue;
      }
      const char *w = wv->words[i];
      if (is_identifier_tok(w) || (w[0] >= '0' && w[0] <= '9') ||
          strcmp(w, "(") == 0 || strcmp(w, "-") == 0)
        continue; /* allowed */
      logits[i] = -1e9f;
    }
    return GS_BODY;
  }

  /* After operator → allow identifiers, numbers, '(' */
  if (last_word && is_operator_tok(last_word)) {
    for (size_t i = 0; i < wv->vocab_size; i++) {
      if (i >= wv->num_words) {
        logits[i] = -1e9f;
        continue;
      }
      if (!wv->words[i]) {
        logits[i] = -1e9f;
        continue;
      }
      const char *w = wv->words[i];
      if (is_identifier_tok(w) || (w[0] >= '0' && w[0] <= '9') ||
          strcmp(w, "(") == 0)
        continue;
      logits[i] = -1e9f;
    }
    return state;
  }

  /* After '}' at depth 0 → done */
  if (last_word && strcmp(last_word, "}") == 0 && brace_depth <= 0) {
    return GS_DONE;
  }

  /* Default: no masking (let the model decide) */
  return state;
}

/* ====================== Option B: Template Infilling =================== */
/*
 * Infill mode: provide a function signature + '{' as prompt, generate
 * body tokens until brace-balanced '}', then validate.
 */

static int load_signatures(const char *path, char sigs[][512], int max) {
  FILE *f = fopen(path, "r");
  if (!f)
    return 0;
  int count = 0;
  while (count < max && fgets(sigs[count], 512, f)) {
    /* Strip trailing newline */
    size_t len = strlen(sigs[count]);
    while (len > 0 &&
           (sigs[count][len - 1] == '\n' || sigs[count][len - 1] == '\r'))
      sigs[count][--len] = '\0';
    if (len > 0)
      count++;
  }
  fclose(f);
  return count;
}

static ComposeResult generate_infill(Model *model, const WordVocab *wv,
                                     const MicrogptConfig *cfg,
                                     const char *signature, int use_constraints,
                                     scalar_t *logits_buf, scalar_t **inf_keys,
                                     scalar_t **inf_values,
                                     size_t *inf_cache_len,
                                     unsigned int *rng_state) {

  ComposeResult res = {0};
  int nl = cfg->n_layer;

  /* Build prompt: "// intent\nsignature\n" — the signature already ends with {
   */
  char prompt[1024];
  snprintf(prompt, sizeof(prompt), "%s\n", signature);
  size_t prompt_len = strlen(prompt);

  for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
    res.attempts = attempt + 1;

    /* Reset KV cache */
    for (int L = 0; L < nl; L++)
      inf_cache_len[L] = 0;

    seed_rng(*rng_state + (unsigned int)attempt * 7919);

    /* Tokenize prompt */
    size_t prompt_tokens[MAX_TOKENS_PER_DOC];
    size_t n_prompt =
        vm_tokenize(prompt, prompt_len, wv, prompt_tokens, MAX_TOKENS_PER_DOC);

    /* Feed BOS */
    size_t pos = 0;
    forward_inference(model, wv->bos_id, pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;

    /* Feed prompt tokens */
    scalar_t prompt_conf = 0;
    for (size_t pi = 0;
         pi < n_prompt && pos < (size_t)cfg->block_size - GEN_LEN; pi++) {
      scalar_t max_v = logits_buf[0];
      for (size_t c = 1; c < wv->vocab_size; c++)
        if (logits_buf[c] > max_v)
          max_v = logits_buf[c];
      scalar_t sum = 0;
      for (size_t c = 0; c < wv->vocab_size; c++)
        sum += (scalar_t)exp(logits_buf[c] - max_v);
      scalar_t prob =
          (scalar_t)exp(logits_buf[prompt_tokens[pi]] - max_v) / sum;
      prompt_conf += prob;

      forward_inference(model, prompt_tokens[pi], pos, inf_keys, inf_values,
                        inf_cache_len, logits_buf);
      pos++;
    }
    res.confidence =
        (double)(prompt_conf / (n_prompt > 0 ? (scalar_t)n_prompt : 1.0f));

    /* Generate body tokens */
    size_t gen_tokens[GEN_LEN + 1];
    int gen_count = 0;
    int brace_depth = 1; /* signature already has opening '{' */

    size_t open_id = word_to_id(wv, "{");
    size_t close_id = word_to_id(wv, "}");
    GrammarState gstate = GS_BODY;
    size_t prev_tok = open_id;

    for (int g = 0; g < GEN_LEN && pos < (size_t)cfg->block_size; g++) {
      /* Option C: apply grammar constraints if enabled */
      if (use_constraints) {
        gstate =
            apply_grammar_mask(logits_buf, wv, gstate, prev_tok, brace_depth);
        if (gstate == GS_DONE)
          break;
      }

      size_t tok = sample_token(logits_buf, wv->vocab_size, CODEGEN_TEMP);
      if (tok == wv->bos_id)
        break;
      gen_tokens[gen_count++] = tok;
      prev_tok = tok;

      if (tok == open_id)
        brace_depth++;
      else if (tok == close_id) {
        brace_depth--;
        if (brace_depth <= 0)
          break;
      }

      forward_inference(model, tok, pos, inf_keys, inf_values, inf_cache_len,
                        logits_buf);
      pos++;
    }

    /* Detokenize */
    char gen_text[4096];
    vm_detokenize(gen_tokens, (size_t)gen_count, wv, gen_text,
                  sizeof(gen_text));

    /* Build full code: signature + generated body */
    snprintf(res.code, sizeof(res.code), "%s%s", prompt, gen_text);

    /* Validate */
    if (validate_vm_code(res.code)) {
      res.valid = 1;
      return res;
    }

    *rng_state += 13;
  }

  return res;
}

/* ====================== Main =========================================== */

int main(void) {
  unsigned int rng_state = (unsigned int)time(NULL);
  seed_rng(42);

  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = 96;
  cfg.n_head = 4;
  cfg.mlp_dim = 384;
  cfg.n_layer = 2;
  cfg.block_size = 256;
  cfg.batch_size = 16;
  cfg.num_steps = 15000;
  cfg.learning_rate = 0.0005;
  cfg.max_vocab = 1200;
  cfg.max_docs = 5000;
  cfg.max_doc_len = 512;

  printf("================================================================\n");
  printf("  MicroGPT-C — VM Compose Pipeline (Phase 3)\n");
  printf("================================================================\n");
  printf("\n");
  printf("  Generate → Validate → Retry loop\n");
  printf("  Model: vm_codegen (129K params, word-level)\n");
  printf("  Validation: vm_module_compile() (Flex/Bison)\n");
  printf("  Max retries: %d per intent\n", MAX_RETRIES);
  printf("\n");
  printf(
      "================================================================\n\n");

  int nl = cfg.n_layer;

#ifdef MICROGPT_METAL
  if (metal_init() != 0)
    fprintf(stderr, "WARNING: Metal GPU init failed, falling back to CPU\n");
#endif

  /* ---- Load corpus for vocabulary ---- */
  Docs docs = {0};
  if (load_docs_multiline("c_vm_functions_combined.txt", &docs, cfg.max_docs) !=
      0) {
    fprintf(stderr, "Cannot open vm_functions_combined.txt\n");
    fprintf(stderr,
            "  Copy from experiments/organelles/vm_codegen/ to build/\n");
    return 1;
  }
  printf("loaded %zu VM functions for vocabulary\n", docs.num_docs);

  /* ---- Build vocabulary ---- */
  WordVocab wv = {0};
  if (build_vm_word_vocab(&docs, &wv, (size_t)cfg.max_vocab - 3) != 0) {
    fprintf(stderr, "Failed to build word vocabulary\n");
    return 1;
  }
  printf("word vocab: %zu tokens (%zu words + 3 special)\n", wv.vocab_size,
         wv.num_words);
  printf("  unk_id=%zu newline_id=%zu bos_id=%zu\n", wv.unk_id, wv.newline_id,
         wv.bos_id);
  printf("  vocab[0..9]: ");
  for (size_t i = 0; i < 10 && i < wv.vocab_size; i++) {
    if (wv.words[i] && strcmp(wv.words[i], "\n") == 0)
      printf("[\\n] ");
    else if (wv.words[i])
      printf("[%s] ", wv.words[i]);
    else
      printf("[NULL] ");
  }
  printf("\n\n");
  /* ---- Load trained checkpoint ---- */
  size_t nparams;
  {
    Model *tmp = model_create(wv.vocab_size, &cfg);
    nparams = model_num_params(tmp);
    model_free(tmp);
  }
  printf("model params: %zu\n", nparams);

  scalar_t *m_buf = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *v_buf = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  if (!m_buf || !v_buf) {
    fprintf(stderr, "OOM\n");
    return 1;
  }

  int resume_step = 0;
  Model *model = checkpoint_load("c_vm_codegen.ckpt", wv.vocab_size, &cfg, m_buf,
                                 v_buf, &resume_step);
  if (!model) {
    fprintf(stderr, "Cannot load vm_codegen.ckpt\n");
    fprintf(stderr,
            "  Copy from experiments/organelles/vm_codegen/ build output\n");
    return 1;
  }
  printf("loaded checkpoint (trained %d steps)\n\n", resume_step);

  /* ---- Allocate inference buffers ---- */
  scalar_t *logits_buf = (scalar_t *)malloc(wv.vocab_size * sizeof(scalar_t));
  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    inf_keys[L] = kv_cache_alloc(&cfg);
    inf_values[L] = kv_cache_alloc(&cfg);
  }

  /* ---- Load test intents ---- */
  char intents[100][256];
  int n_intents = load_intents("test_intents.txt", intents, 100);
  if (n_intents == 0) {
    fprintf(stderr, "Cannot load test_intents.txt\n");
    return 1;
  }
  printf("loaded %d test intents\n\n", n_intents);

  /* ---- Run generate→validate→retry loop ---- */
  printf("--- VM Compose Results ---\n\n");

  int total_valid = 0;
  int total_skipped = 0;
  int total_retries = 0;

  for (int i = 0; i < n_intents; i++) {
    ComposeResult res =
        generate_and_validate(model, &wv, &cfg, intents[i], logits_buf,
                              inf_keys, inf_values, inf_cache_len, &rng_state);

    if (res.confidence < 0.10) {
      printf("[%2d] ⚪ SKIP  %s\n", i + 1, intents[i]);
      printf("          (prompt not recognised, conf=%.0f%%)\n",
             res.confidence * 100);
      total_skipped++;
    } else if (res.valid) {
      printf("[%2d] ✅ PASS  %s\n", i + 1, intents[i]);
      printf("          (attempt %d/%d, conf=%.0f%%)\n", res.attempts,
             MAX_RETRIES, res.confidence * 100);
      printf("          %s\n",
             res.code + strlen(intents[i]) + 1); /* skip prompt */
      total_valid++;
    } else {
      printf("[%2d] ❌ FAIL  %s\n", i + 1, intents[i]);
      printf("          (all %d attempts failed, conf=%.0f%%)\n", MAX_RETRIES,
             res.confidence * 100);
    }
    total_retries += res.attempts;
    fflush(stdout);
  }

  /* ---- Summary ---- */
  int total_tested = n_intents - total_skipped;
  printf("\n--- compose summary ---\n");
  printf("intents: %d | tested: %d | skipped: %d\n", n_intents, total_tested,
         total_skipped);
  printf("passed: %d/%d (%.0f%%)\n", total_valid, total_tested,
         total_tested > 0 ? 100.0 * total_valid / total_tested : 0.0);
  printf("total attempts: %d (avg %.1f per intent)\n", total_retries,
         total_tested > 0 ? (double)total_retries / total_tested : 0.0);
  printf("retry benefit: %s\n",
         total_retries > total_tested ? "YES — retries helped" : "minimal");

  /* =================== Option B: Template Infilling ==================== */
  char sigs[100][512];
  int n_sigs = load_signatures("test_signatures.txt", sigs, 100);
  if (n_sigs > 0) {
    printf("\n\n==============================================================="
           "=\n");
    printf("  Option B: Template Infilling (%d signatures)\n", n_sigs);
    printf(
        "================================================================\n\n");

    int infill_valid = 0;
    int infill_total = 0;

    for (int i = 0; i < n_sigs; i++) {
      ComposeResult res = generate_infill(
          model, &wv, &cfg, sigs[i], 0 /* no constraints */, logits_buf,
          inf_keys, inf_values, inf_cache_len, &rng_state);
      infill_total++;

      if (res.valid) {
        printf("[%2d] ✅ PASS  %s\n", i + 1, sigs[i]);
        printf("          %s\n", res.code);
        infill_valid++;
      } else {
        printf("[%2d] ❌ FAIL  %s\n", i + 1, sigs[i]);
        printf("          (all %d attempts failed, conf=%.0f%%)\n", MAX_RETRIES,
               res.confidence * 100);
      }
      fflush(stdout);
    }

    printf("\n--- infill summary ---\n");
    printf("signatures: %d | passed: %d/%d (%.0f%%)\n", n_sigs, infill_valid,
           infill_total,
           infill_total > 0 ? 100.0 * infill_valid / infill_total : 0.0);
  } else {
    printf("\n(no test_signatures.txt found — skipping Option B infill)\n");
  }

  /* =================== Option C: Constrained Decoding ================== */
  printf(
      "\n\n================================================================\n");
  printf("  Option C: Constrained Decoding (%d intents)\n", n_intents);
  printf(
      "================================================================\n\n");

  int constrained_valid = 0;
  int constrained_skipped = 0;

  for (int i = 0; i < n_intents; i++) {
    /* Use constrained infill for intent-based generation */
    char intent_as_sig[512];
    snprintf(intent_as_sig, sizeof(intent_as_sig), "%s", intents[i]);

    ComposeResult cres = generate_infill(
        model, &wv, &cfg, intent_as_sig, 1 /* constraints ON */, logits_buf,
        inf_keys, inf_values, inf_cache_len, &rng_state);

    if (cres.confidence < 0.10) {
      printf("[%2d] ⚪ SKIP  %s\n", i + 1, intents[i]);
      constrained_skipped++;
    } else if (cres.valid) {
      printf("[%2d] ✅ PASS  %s\n", i + 1, intents[i]);
      printf("          (constrained, conf=%.0f%%)\n", cres.confidence * 100);
      printf("          %s\n", cres.code);
      constrained_valid++;
    } else {
      printf("[%2d] ❌ FAIL  %s\n", i + 1, intents[i]);
      printf("          (constrained, all %d attempts failed)\n", MAX_RETRIES);
    }
    fflush(stdout);
  }

  int constrained_tested = n_intents - constrained_skipped;
  printf("\n--- constrained decoding summary ---\n");
  printf("intents: %d | tested: %d | skipped: %d\n", n_intents,
         constrained_tested, constrained_skipped);
  printf("passed: %d/%d (%.0f%%)\n", constrained_valid, constrained_tested,
         constrained_tested > 0 ? 100.0 * constrained_valid / constrained_tested
                                : 0.0);
  printf("unconstrained: %d/%d vs constrained: %d/%d\n", total_valid,
         total_tested, constrained_valid, constrained_tested);

  /* =================== Option B+C: Infill + Constraints ================ */
  if (n_sigs > 0) {
    printf("\n\n==============================================================="
           "=\n");
    printf("  Option B+C: Infill + Constrained Decoding (%d signatures)\n",
           n_sigs);
    printf(
        "================================================================\n\n");

    int bc_valid = 0;

    for (int i = 0; i < n_sigs; i++) {
      ComposeResult res = generate_infill(
          model, &wv, &cfg, sigs[i], 1 /* constraints ON */, logits_buf,
          inf_keys, inf_values, inf_cache_len, &rng_state);

      if (res.valid) {
        printf("[%2d] ✅ PASS  %s\n", i + 1, sigs[i]);
        printf("          %s\n", res.code);
        bc_valid++;
      } else {
        printf("[%2d] ❌ FAIL  %s\n", i + 1, sigs[i]);
      }
      fflush(stdout);
    }

    printf("\n--- infill+constrained summary ---\n");
    printf("passed: %d/%d (%.0f%%)\n", bc_valid, n_sigs,
           n_sigs > 0 ? 100.0 * bc_valid / n_sigs : 0.0);
  }

  /* ---- Cleanup ---- */
  for (int L = 0; L < nl; L++) {
    kv_cache_free(inf_keys[L]);
    kv_cache_free(inf_values[L]);
  }
  free(inf_keys);
  free(inf_values);
  free(inf_cache_len);
  free(logits_buf);
  free(m_buf);
  free(v_buf);
  model_free(model);
  free_docs(&docs);
  free_word_vocab(&wv);
  return 0;
}
