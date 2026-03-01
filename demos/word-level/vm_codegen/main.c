/*
 * MicroGPT-C — VM Code Generation V2 (Organelle Word API)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Refactored vm_codegen to use the organelle word-level API:
 *   - organelle_train_words()  for training  (replaces custom training loop)
 *   - Custom inference with brace-balanced stopping + ensemble voting
 *   - vm_module_compile() validation gate (unchanged)
 *
 * The corpus must be pre-tokenized (spaces between all tokens) so that
 * the generic build_word_vocab() / tokenize_words() work correctly.
 * Run pretokenize_corpus.py to generate vm_functions_pretok.txt.
 *
 * Build:
 *   cmake --build build --target vm_codegen_v2
 *   ./build/vm_codegen_v2
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"
#include "microgpt_vm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MICROGPT_METAL
#include "microgpt_metal.h"
#endif

/* ----- Configuration ----- */
#ifndef GEN_LEN
#define GEN_LEN 80
#endif
#ifndef NUM_CANDIDATES
#define NUM_CANDIDATES 10
#endif
#ifndef WORD_VOCAB_SIZE
#define WORD_VOCAB_SIZE 600
#endif

/* Temperature schedule for best-of-N */
static const float CANDIDATE_TEMPS[NUM_CANDIDATES] = {
    0.3f, 0.4f, 0.5f, 0.5f, 0.6f, 0.6f, 0.7f, 0.7f, 0.8f, 0.5f};

/* ====================== Detokenisation ================================ */
/*
 * Convert word tokens back to VM DSL source code.
 * Spacing heuristics: no space before ), ;, ,, : and no space after (.
 */
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

    /* Spacing heuristics for code formatting */
    int is_newline = (tid == wv->newline_id || (wlen == 1 && w[0] == '\n'));
    int is_open_paren = (wlen == 1 && w[0] == '(');
    int is_close = (wlen == 1 &&
                    (w[0] == ')' || w[0] == ';' || w[0] == ',' || w[0] == ':'));

    if (is_newline) {
      if (pos + 1 < bufsize)
        buf[pos++] = '\n';
      continue;
    }

    if (pos > 0 && !is_open_paren && !is_close && buf[pos - 1] != '\n' &&
        buf[pos - 1] != '(') {
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

/* ====================== Custom Generation with Brace Stopping =========== */
/*
 * We can't use organelle_generate_words() because it stops on newline,
 * but VM code generation needs brace-balanced stopping.
 */

typedef struct {
  int valid;
  int attempts;
  double confidence;
  char code[4096];
} GenResult;

static GenResult generate_vm_code(Organelle *org, const MicrogptConfig *cfg,
                                  const char *prompt, float temperature) {
  GenResult res = {0};
  int nl = cfg->n_layer;

  /* Allocate inference buffers */
  scalar_t *logits_buf =
      (scalar_t *)calloc(org->word_vocab.vocab_size, sizeof(scalar_t));
  scalar_t **inf_keys = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));

  for (int L = 0; L < nl; L++) {
    inf_keys[L] = kv_cache_alloc(cfg);
    inf_values[L] = kv_cache_alloc(cfg);
    inf_cache_len[L] = 0;
  }

  /* Feed BOS */
  size_t pos = 0;
  forward_inference(org->model, org->word_vocab.bos_id, pos, inf_keys,
                    inf_values, inf_cache_len, logits_buf);
  pos++;

  /* Tokenize and feed prompt */
  size_t prompt_tokens[512];
  size_t n_prompt = tokenize_words(prompt, strlen(prompt), &org->word_vocab,
                                   prompt_tokens, 512);

  scalar_t prompt_conf = 0;
  for (size_t pi = 0; pi < n_prompt && pos < (size_t)cfg->block_size - GEN_LEN;
       pi++) {
    /* Track confidence */
    scalar_t max_v = logits_buf[0];
    for (size_t c = 1; c < org->word_vocab.vocab_size; c++)
      if (logits_buf[c] > max_v)
        max_v = logits_buf[c];
    scalar_t sum = 0;
    for (size_t c = 0; c < org->word_vocab.vocab_size; c++)
      sum += (scalar_t)exp(logits_buf[c] - max_v);
    scalar_t prob = (scalar_t)exp(logits_buf[prompt_tokens[pi]] - max_v) / sum;
    prompt_conf += prob;

    forward_inference(org->model, prompt_tokens[pi], pos, inf_keys, inf_values,
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

  size_t open_id = word_to_id(&org->word_vocab, "{");
  size_t close_id = word_to_id(&org->word_vocab, "}");
  size_t comment_id = word_to_id(&org->word_vocab, "//");

  for (int g = 0; g < GEN_LEN && pos < (size_t)cfg->block_size; g++) {
    size_t tok =
        sample_token(logits_buf, org->word_vocab.vocab_size, temperature);
    if (tok == org->word_vocab.bos_id)
      break;

    /* Stop if '//' after closing brace — spilling into next function */
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

    forward_inference(org->model, tok, pos, inf_keys, inf_values, inf_cache_len,
                      logits_buf);
    pos++;
  }

  /* Detokenize */
  char gen_text[4096];
  vm_detokenize(gen_tokens, (size_t)gen_count, &org->word_vocab, gen_text,
                sizeof(gen_text));

  /* Build full code: prompt + generated */
  char prompt_with_nl[512];
  snprintf(prompt_with_nl, sizeof(prompt_with_nl), "%s\n", prompt);
  snprintf(res.code, sizeof(res.code), "%s%s", prompt_with_nl, gen_text);

  /* Validate */
  res.valid = validate_vm_code(res.code);
  res.attempts = 1;

  /* Cleanup */
  for (int L = 0; L < nl; L++) {
    kv_cache_free(inf_keys[L]);
    kv_cache_free(inf_values[L]);
  }
  free(inf_keys);
  free(inf_values);
  free(inf_cache_len);
  free(logits_buf);

  return res;
}

/* ====================== Ensemble Generation ============================= */
/*
 * Generate N candidates at different temperatures, validate each,
 * return the highest-confidence valid candidate.
 */
static GenResult generate_ensemble(Organelle *org, const MicrogptConfig *cfg,
                                   const char *prompt) {
  GenResult best = {0};
  best.confidence = -1.0;

  unsigned int rng = (unsigned int)time(NULL);

  for (int attempt = 0; attempt < NUM_CANDIDATES; attempt++) {
    seed_rng(rng + (unsigned int)attempt * 7919);

    GenResult res =
        generate_vm_code(org, cfg, prompt, CANDIDATE_TEMPS[attempt]);
    res.attempts = attempt + 1;

    if (res.valid && res.confidence > best.confidence) {
      best = res;
      best.attempts = attempt + 1;
    }
  }

  /* Return best valid candidate, or last attempt */
  if (best.valid)
    return best;

  /* No valid candidate — return last attempt info */
  GenResult last = {0};
  last.valid = 0;
  last.attempts = NUM_CANDIDATES;
  return last;
}

/* ====================== Test Prompts ==================================== */

/* 5 control prompts (in training corpus) + 5 novel prompts */
static const char *CONTROL_PROMPTS[] = {
    "// compute factorial of n",
    "// compute fibonacci number at position n",
    "// compute absolute value",
    "// compute maximum of two numbers",
    "// compute the sum of numbers from 1 to n",
};

static const char *NOVEL_PROMPTS[] = {
    "// determine if a year is a leap year",
    "// compute the nth triangular number",
    "// convert degrees to radians",
    "// calculate the surface area of a cube",
    "// find the digital root of a number",
};

#define N_CONTROLS (sizeof(CONTROL_PROMPTS) / sizeof(CONTROL_PROMPTS[0]))
#define N_NOVELS (sizeof(NOVEL_PROMPTS) / sizeof(NOVEL_PROMPTS[0]))

/* ====================== Main =========================================== */

int main(void) {
  setbuf(stdout, NULL);
  printf("\n");
  printf("================================================================\n");
  printf("  MicroGPT-C — VM Code Generation V2 (Organelle Word API)\n");
  printf(
      "================================================================\n\n");

  /* ---- Configuration ---- */
  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = N_EMBD;
  cfg.n_head = N_HEAD;
  cfg.n_layer = N_LAYER;
  cfg.block_size = BLOCK_SIZE;
  cfg.mlp_dim = MLP_DIM;
  cfg.learning_rate = LEARNING_RATE;

  /* ---- Train using organelle word API ---- */
  Organelle *org = organelle_train_words(
      "vm_codegen_v2",           /* name */
      "w_vm_functions_pretok.txt", /* corpus (pre-tokenized) */
      "w_vm_codegen_v2.ckpt",      /* checkpoint */
      &cfg,                      /* config */
      NUM_STEPS,                 /* training steps */
      WORD_VOCAB_SIZE            /* max words */
  );

  if (!org || !org->model) {
    fprintf(stderr, "ERROR: Training failed\n");
    return 1;
  }

  printf("\n--- Model ready: vocab=%zu params=%zu ---\n\n",
         org->word_vocab.vocab_size, model_num_params(org->model));

  /* ---- Evaluate: Control prompts ---- */
  printf("================================================================\n");
  printf("  CONTROL PROMPTS (in training corpus)\n");
  printf(
      "================================================================\n\n");

  int ctrl_valid = 0;
  for (size_t i = 0; i < N_CONTROLS; i++) {
    GenResult res = generate_ensemble(org, &cfg, CONTROL_PROMPTS[i]);

    if (res.valid) {
      printf("[%zu] ✅ PASS (attempt %d, conf=%.3f)\n", i + 1, res.attempts,
             res.confidence);
      printf("    %s\n\n", res.code);
      ctrl_valid++;
    } else {
      printf("[%zu] ❌ FAIL (%d attempts)\n", i + 1, res.attempts);
      printf("    %s\n\n", CONTROL_PROMPTS[i]);
    }
  }

  /* ---- Evaluate: Novel prompts ---- */
  printf("================================================================\n");
  printf("  NOVEL PROMPTS (not in training corpus)\n");
  printf(
      "================================================================\n\n");

  int novel_valid = 0;
  for (size_t i = 0; i < N_NOVELS; i++) {
    GenResult res = generate_ensemble(org, &cfg, NOVEL_PROMPTS[i]);

    if (res.valid) {
      printf("[%zu] ✅ PASS (attempt %d, conf=%.3f)\n", i + 1, res.attempts,
             res.confidence);
      printf("    %s\n\n", res.code);
      novel_valid++;
    } else {
      printf("[%zu] ❌ FAIL (%d attempts)\n", i + 1, res.attempts);
      printf("    %s\n\n", NOVEL_PROMPTS[i]);
    }
  }

  /* ---- Summary ---- */
  int total = (int)(N_CONTROLS + N_NOVELS);
  int total_valid = ctrl_valid + novel_valid;

  printf(
      "\n================================================================\n");
  printf("              VM CODEGEN V2 RESULTS\n");
  printf("================================================================\n");
  printf("Controls:   %d/%zu (%.0f%%)\n", ctrl_valid, N_CONTROLS,
         100.0 * ctrl_valid / (double)N_CONTROLS);
  printf("Novel:      %d/%zu (%.0f%%)\n", novel_valid, N_NOVELS,
         100.0 * novel_valid / (double)N_NOVELS);
  printf("Total:      %d/%d (%.0f%%)\n", total_valid, total,
         100.0 * total_valid / (double)total);
  printf("================================================================\n");

  /* ---- Cleanup ---- */
  organelle_free(org);
  return 0;
}
