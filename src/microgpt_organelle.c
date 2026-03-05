/*
 * MicroGPT-C — Organelle Pipeline Architecture (OPA) Library
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 *
 * ============================================================================
 *              Organelle Pipeline Architecture — Implementation Guide
 * ============================================================================
 *
 * This file implements the generic infrastructure for the Organelle Pipeline
 * Architecture (OPA).  An "organelle" is a small, specialised neural network
 * — typically 30K–460K parameters — trained on a narrow corpus to perform
 * one well-defined task (e.g. "plan the next move", "validate a board state").
 *
 * Multiple organelles compose into a pipeline via pipe-delimited string
 * messages (the "flat-string protocol"), allowing complex multi-step
 * reasoning without shared weights or backpropagation across modules.
 *
 * Reading top-to-bottom, the code is organised as follows:
 *
 *   SECTION 1: Corpus Loader (line ~60)
 *   ────────────────────────────────────
 *   opa_load_docs_multiline()  — Read a corpus file where documents are
 *       separated by blank lines (double newlines).  Unlike microgpt.c's
 *       load_docs() which splits on single newlines, this handles multi-
 *       line documents like board states "STATE|...\nACTION|move".
 *
 *   SECTION 2: Organelle Inference (line ~100)
 *   ──────────────────────────────────────────
 *   organelle_generate()           — Feed a prompt through one organelle
 *       and generate a response up to a newline or max_len.  Manages
 *       KV cache allocation, prompt feeding, and auto-regressive decoding.
 *   organelle_generate_ensemble()  — Run N inference passes with jittered
 *       temperatures and return the majority-vote winner.  Provides a
 *       confidence score (vote fraction) for downstream gating.
 *
 *   SECTION 3: Organelle Training (line ~190)
 *   ─────────────────────────────────────────
 *   organelle_train()  — Full training pipeline for one organelle:
 *       1. Load corpus via opa_load_docs_multiline()
 *       2. Build character vocabulary
 *       3. Create or resume from checkpoint
 *       4. Multi-threaded training loop (same worker model as microgpt.c)
 *       5. Save checkpoint + training log (.ckpt.log)
 *
 *   SECTION 4: Organelle Cleanup (line ~440)
 *   ────────────────────────────────────────
 *   organelle_free()  — Release all memory owned by an organelle
 *       (model weights, documents, vocabulary).
 *
 *   SECTION 5: Kanban State Tracker (line ~460)
 *   ──────────────────────────────────────────
 *   opa_kanban_init()           — Initialise a Kanban board
 *   opa_kanban_add_blocked()    — Mark an action as blocked
 *   opa_kanban_is_blocked()     — Test if an action is on the blocked list
 *   opa_kanban_clear_blocked()  — Clear all blocked actions
 *   opa_kanban_add_last()       — Append to a fixed-size action history
 *
 *       The Kanban board tracks which actions have been tried and failed,
 *       preventing the pipeline from repeating dead-end moves.  It replaces
 *       mutable global state with an explicit, per-game state object.
 *
 *   SECTION 6: Cycle Detector (line ~530)
 *   ─────────────────────────────────────
 *   opa_cycle_init()      — Reset the detector
 *   opa_cycle_detected()  — Check for A,B,A,B repetition patterns
 *   opa_cycle_other()     — Return the "other" action in a detected cycle
 *   opa_cycle_record()    — Record an action into the sliding window
 *
 *       Detects when the pipeline oscillates between two actions (e.g.
 *       moving a piece left then right repeatedly).  Uses a circular
 *       buffer of OPA_CYCLE_WINDOW recent actions.
 *
 *   SECTION 7: Pipe-String Helpers (line ~565)
 *   ──────────────────────────────────────────
 *   opa_extract_pipe_value()  — Parse "KEY=value|KEY2=value2" strings
 *   opa_pipe_starts_with()    — Prefix match on pipe strings
 *
 *       The flat-string protocol uses pipe-delimited key=value pairs as
 *       the inter-organelle communication format.  This avoids JSON/XML
 *       parsing overhead and keeps the protocol C-friendly.
 *
 *   SECTION 8: Valid-Move Filter (line ~595)
 *   ───────────────────────────────────────
 *   opa_valid_filter()    — Check if an action is in a CSV valid list
 *   opa_valid_fallback()  — Find the first valid, non-blocked action
 *
 *       These functions constrain organelle outputs to legal moves,
 *       preventing the model from generating syntactically correct but
 *       game-illegal actions.
 *
 *
 * PIPELINE ARCHITECTURE
 * ─────────────────────
 * A typical two-organelle game pipeline looks like this:
 *
 *   ┌──────────┐    pipe-string     ┌──────────┐    action
 *   │ PLANNER  │  ──────────────►   │  PLAYER  │  ──────────►  Game
 *   │ (30-460K)│  "STATE|...|PLAN"  │ (30-460K)│  "move=R2"    Engine
 *   └──────────┘                    └──────────┘
 *        ▲                               ▲
 *        │                               │
 *   Kanban board                    Valid-move filter
 *   (blocked moves)                 (legal move CSV)
 *
 * Each organelle is a self-contained GPT model with its own vocabulary,
 * checkpoint, and training corpus.  The only coupling is the pipe-string
 * format they agree on.
 */

#include "microgpt_organelle.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ======================== Corpus Loader =================================== */

/*
 * opa_load_docs_multiline — Read a corpus file where documents are separated
 *   by blank lines (sequences of two or more consecutive newlines).
 *
 *   Unlike microgpt.c's load_docs() which treats each line as a separate
 *   document, this function groups consecutive non-empty lines into a
 *   single document.  This is essential for OPA corpora where each training
 *   example spans multiple lines, e.g.:
 *
 *     STATE|board=123456780
 *     ACTION|move=R
 *                              ← blank line = document boundary
 *     STATE|board=123450786
 *     ACTION|move=D
 *
 *   The raw file is slurped into docs->data; docs->lines[] and
 *   docs->doc_lens[] point into this buffer.
 *
 *   @param path      Path to the corpus text file
 *   @param docs      Output Docs struct (caller must free via free_docs())
 *   @param max_docs  Maximum number of documents to load
 *   @return          0 on success, -1 on error (file not found, OOM)
 */
int opa_load_docs_multiline(const char *path, Docs *docs, int max_docs) {
  FILE *f = fopen(path, "r");
  if (!f)
    return -1;

  /* Determine file size */
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  /* Slurp entire file into a single heap buffer */
  docs->data = (char *)malloc((size_t)fsize + 1);
  if (!docs->data) {
    fclose(f);
    return -1;
  }
  fread(docs->data, 1, (size_t)fsize, f);
  docs->data[fsize] = '\0';
  fclose(f);

  /* Allocate document index arrays */
  docs->lines = (char **)malloc(sizeof(char *) * (size_t)max_docs);
  docs->doc_lens = (size_t *)malloc(sizeof(size_t) * (size_t)max_docs);
  docs->num_docs = 0;

  /*
   * Scan buffer: skip blank lines (document boundaries), then collect
   * consecutive non-empty lines until the next blank line or EOF.
   *
   * Document boundary detection: a newline followed by another newline
   * (or EOF) marks the end of a document.
   */
  char *p = docs->data;
  while (*p && docs->num_docs < (size_t)max_docs) {
    /* Skip leading blank lines */
    while (*p == '\n')
      p++;
    if (!*p)
      break;

    /* Start of a new document */
    char *doc_start = p;

    /* Advance until we hit a blank-line boundary:
     * a newline followed by another newline or end-of-file */
    while (*p && !(*p == '\n' && (*(p + 1) == '\n' || *(p + 1) == '\0')))
      p++;
    if (*p == '\n')
      p++; /* include the terminating newline in the document */

    size_t doc_len = (size_t)(p - doc_start);
    if (doc_len > 0) {
      docs->lines[docs->num_docs] = doc_start;
      docs->doc_lens[docs->num_docs] = doc_len;
      docs->num_docs++;
    }
  }

  return 0;
}

/* ======================== Organelle Inference ==============================
 */

/*
 * organelle_generate — Feed a prompt into an organelle and generate a response.
 *
 *   The inference protocol follows a strict format:
 *     1. Feed BOS token (begin-of-vm_list)
 *     2. Feed each character of the prompt as a token
 *     3. Feed a newline separator ('\n') — this signals "your turn to respond"
 *     4. Auto-regressively sample tokens until:
 *        - A newline is generated (response terminator), or
 *        - BOS/EOS is generated, or
 *        - max_len characters have been produced, or
 *        - block_size is reached (context window full)
 *
 *   This matches the training format where each corpus document is:
 *     "PROMPT\nRESPONSE"
 *
 *   The function allocates a fresh KV cache per call, so it is stateless —
 *   each call starts inference from scratch.  This is fine for small models
 *   where the prompt is short (typically 20-80 tokens).
 *
 *   @param org          The organelle to use (model + vocabulary)
 *   @param cfg          Runtime configuration (architecture, block_size, etc.)
 *   @param prompt       Input string (e.g. "STATE|board=123456780")
 *   @param output       Buffer to receive generated response (caller-owned)
 *   @param max_len      Maximum characters to generate
 *   @param temperature  Sampling temperature (lower = more deterministic)
 */
void organelle_generate(const Organelle *org, const MicrogptConfig *cfg,
                        const char *prompt, char *output, int max_len,
                        scalar_t temperature) {
  const int nl = cfg->n_layer;

  /* Allocate per-layer KV cache for inference */
  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int l = 0; l < nl; l++) {
    inf_keys[l] = kv_cache_alloc(cfg);
    inf_values[l] = kv_cache_alloc(cfg);
  }

  scalar_t *logits_buf =
      (scalar_t *)malloc((size_t)cfg->max_vocab * sizeof(scalar_t));
  int pos = 0;
  int out_pos = 0;

  const Vocab *vocab = &org->vocab;

  /* Step 1: Feed BOS token to prime the model */
  size_t token = vocab->bos_id;
  forward_inference(org->model, token, pos, inf_keys, inf_values, inf_cache_len,
                    logits_buf);
  pos++;

  /* Step 2: Feed each prompt character as a token.
   * Linear scan through the vocabulary to find each character's ID.
   * This is O(prompt_len × vocab_size) but both are small (<128). */
  for (int i = 0; prompt[i] && pos < cfg->block_size - 1; i++) {
    token = 0;
    for (size_t v = 0; v < vocab->vocab_size; v++) {
      if (vocab->chars[v] == (unsigned char)prompt[i]) {
        token = v;
        break;
      }
    }
    forward_inference(org->model, token, pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;
  }

  /* Step 3: Feed newline separator — signals "your turn to respond" */
  token = 0;
  for (size_t v = 0; v < vocab->vocab_size; v++) {
    if (vocab->chars[v] == '\n') {
      token = v;
      break;
    }
  }
  forward_inference(org->model, token, pos, inf_keys, inf_values, inf_cache_len,
                    logits_buf);
  pos++;

  /* Step 4: Auto-regressive decoding — sample until stop condition */
  for (int g = 0; g < max_len && pos < cfg->block_size; g++) {
    token = sample_token(logits_buf, vocab->vocab_size, temperature);

    /* Stop on BOS/EOS token */
    if (token == vocab->bos_id)
      break;

    char ch = (char)vocab->chars[token];

    /* Stop on newline (response terminator in our training format) */
    if (ch == '\n')
      break;

    output[out_pos++] = ch;

    forward_inference(org->model, token, pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;
  }

  output[out_pos] = '\0';

  /* Cleanup: free KV caches and logits buffer */
  for (int l = 0; l < nl; l++) {
    kv_cache_free(inf_keys[l]);
    kv_cache_free(inf_values[l]);
  }
  free(inf_keys);
  free(inf_values);
  free(inf_cache_len);
  free(logits_buf);
}

/* ── organelle_generate_multiline ───────────────────────────────────────────
 */

/*
 * Like organelle_generate but stops on a blank line (double newline "\n\n")
 * instead of the first newline.  Required for multi-line corpus targets
 * such as C function bodies:  "void fn(...) {\n  ...\n}\n\n"
 *
 * The output includes newlines within the body but stops before the blank
 * line that separates documents.  Null-terminated.
 */
void organelle_generate_multiline(const Organelle *org,
                                  const MicrogptConfig *cfg, const char *prompt,
                                  char *output, int max_len,
                                  scalar_t temperature) {
  const int nl = cfg->n_layer;

  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int l = 0; l < nl; l++) {
    inf_keys[l] = kv_cache_alloc(cfg);
    inf_values[l] = kv_cache_alloc(cfg);
  }

  scalar_t *logits_buf =
      (scalar_t *)malloc((size_t)cfg->max_vocab * sizeof(scalar_t));
  int pos = 0;
  int out_pos = 0;

  const Vocab *vocab = &org->vocab;

  /* Feed BOS token */
  size_t token = vocab->bos_id;
  forward_inference(org->model, token, pos, inf_keys, inf_values, inf_cache_len,
                    logits_buf);
  pos++;

  /* Feed prompt characters */
  for (int i = 0; prompt[i] && pos < cfg->block_size - 1; i++) {
    token = 0;
    for (size_t v = 0; v < vocab->vocab_size; v++) {
      if (vocab->chars[v] == (unsigned char)prompt[i]) {
        token = v;
        break;
      }
    }
    forward_inference(org->model, token, pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;
  }

  /* Auto-regressive decoding — stop on double-newline or BOS or max_len */
  int prev_was_newline = 0;
  for (int g = 0; g < max_len && pos < cfg->block_size; g++) {
    token = sample_token(logits_buf, vocab->vocab_size, temperature);

    if (token == vocab->bos_id)
      break;

    char ch = (char)vocab->chars[token];

    /* Stop on double-newline: previous char was \n and this one is too */
    if (ch == '\n' && prev_was_newline)
      break;

    output[out_pos++] = ch;
    prev_was_newline = (ch == '\n');

    forward_inference(org->model, token, pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;
  }

  output[out_pos] = '\0';

  for (int l = 0; l < nl; l++) {
    kv_cache_free(inf_keys[l]);
    kv_cache_free(inf_values[l]);
  }
  free(inf_keys);
  free(inf_values);
  free(inf_cache_len);
  free(logits_buf);
}

/* ── organelle_generate_from_cache ─────────────────────────────────────────
 *
 * Decode-only path: starts generation from a pre-filled KV cache.
 * Used by ensemble voting after processing the prompt once.
 *
 * The caller passes:
 *   - keys/values/cache_len: already populated with prompt KV state
 *   - pos: the position index to start decoding at
 *   - prompt_logits: the logits from the last prompt token (for first sample)
 *
 * IMPORTANT: This function does NOT free the KV caches — the caller owns them.
 */
void organelle_generate_from_cache(const Organelle *org,
                                   const MicrogptConfig *cfg, scalar_t **keys,
                                   scalar_t **values, size_t *cache_len,
                                   int pos, const scalar_t *prompt_logits,
                                   char *output, int max_len,
                                   scalar_t temperature) {
  const Vocab *vocab = &org->vocab;
  scalar_t *logits_buf =
      (scalar_t *)malloc((size_t)cfg->max_vocab * sizeof(scalar_t));

  /* Copy the prompt logits so we can sample the first token */
  memcpy(logits_buf, prompt_logits, vocab->vocab_size * sizeof(scalar_t));

  int out_pos = 0;
  for (int g = 0; g < max_len && pos < cfg->block_size; g++) {
    size_t token = sample_token(logits_buf, vocab->vocab_size, temperature);

    if (token == vocab->bos_id)
      break;

    char ch = (char)vocab->chars[token];
    if (ch == '\n')
      break;

    output[out_pos++] = ch;

    forward_inference(org->model, token, pos, keys, values, cache_len,
                      logits_buf);
    pos++;
  }

  output[out_pos] = '\0';
  free(logits_buf);
}

/* ── organelle_generate_speculative ────────────────────────────────────────
 *
 * SSD-inspired speculative decoding with a draft+target organelle pair.
 *
 * Algorithm:
 *   1. Process prompt through BOTH draft and target models
 *   2. Repeat until output complete:
 *      a. Draft model generates spec_k candidate tokens autoregressively
 *      b. Target model processes all spec_k candidates to get target logits
 *      c. Compare: accept tokens where argmax(draft) == argmax(target)
 *      d. On first mismatch: use target's sample as recovery token
 *      e. Roll back draft/target KV caches to rejection point
 *   3. Report acceptance statistics
 *
 * Both organelles must share the same vocabulary.
 */
void organelle_generate_speculative(
    const Organelle *draft, const Organelle *target, const MicrogptConfig *cfg,
    const char *prompt, char *output, int max_len, scalar_t temperature,
    int spec_k, int *accepted_out, int *drafted_out) {
  const int nl = cfg->n_layer;
  const Vocab *vocab = &draft->vocab;

  /* Allocate KV caches for both draft and target */
  scalar_t **dk = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **dv = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *dcl = (size_t *)calloc((size_t)nl, sizeof(size_t));
  scalar_t **tk = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **tv = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *tcl = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int l = 0; l < nl; l++) {
    dk[l] = kv_cache_alloc(cfg);
    dv[l] = kv_cache_alloc(cfg);
    tk[l] = kv_cache_alloc(cfg);
    tv[l] = kv_cache_alloc(cfg);
  }

  scalar_t *draft_logits =
      (scalar_t *)malloc((size_t)cfg->max_vocab * sizeof(scalar_t));
  scalar_t *target_logits =
      (scalar_t *)malloc((size_t)cfg->max_vocab * sizeof(scalar_t));

  /* Buffers for speculated tokens */
  size_t *spec_tokens = (size_t *)malloc((size_t)spec_k * sizeof(size_t));

  int pos = 0;
  int out_pos = 0;
  int total_accepted = 0, total_drafted = 0;

  /* ── Phase 1: Process prompt through both models ── */
  size_t token = vocab->bos_id;
  forward_inference(draft->model, token, pos, dk, dv, dcl, draft_logits);
  forward_inference(target->model, token, pos, tk, tv, tcl, target_logits);
  pos++;

  for (int i = 0; prompt[i] && pos < cfg->block_size - 1; i++) {
    token = 0;
    for (size_t v = 0; v < vocab->vocab_size; v++) {
      if (vocab->chars[v] == (unsigned char)prompt[i]) {
        token = v;
        break;
      }
    }
    forward_inference(draft->model, token, pos, dk, dv, dcl, draft_logits);
    forward_inference(target->model, token, pos, tk, tv, tcl, target_logits);
    pos++;
  }

  /* Feed newline separator to both */
  token = 0;
  for (size_t v = 0; v < vocab->vocab_size; v++) {
    if (vocab->chars[v] == '\n') {
      token = v;
      break;
    }
  }
  forward_inference(draft->model, token, pos, dk, dv, dcl, draft_logits);
  forward_inference(target->model, token, pos, tk, tv, tcl, target_logits);
  pos++;

  /* ── Phase 2: Speculative decode loop ── */
  while (out_pos < max_len && pos < cfg->block_size - spec_k - 1) {

    /* Step 2a: Draft generates spec_k candidates */
    int k_actual = spec_k;
    if (pos + spec_k >= cfg->block_size)
      k_actual = cfg->block_size - pos - 1;
    if (k_actual < 1)
      break;

    /* Save draft KV cache lengths for rollback */
    size_t *dcl_save = (size_t *)malloc((size_t)nl * sizeof(size_t));
    memcpy(dcl_save, dcl, (size_t)nl * sizeof(size_t));
    int draft_pos_save = pos;

    for (int k = 0; k < k_actual; k++) {
      size_t draft_token =
          sample_token(draft_logits, vocab->vocab_size, temperature);
      spec_tokens[k] = draft_token;
      if (k < k_actual - 1) {
        forward_inference(draft->model, draft_token, pos + k, dk, dv, dcl,
                          draft_logits);
      }
    }
    total_drafted += k_actual;

    /* Step 2b: Target verifies each candidate token */
    /* Save target KV cache lengths for potential rollback */
    size_t *tcl_save = (size_t *)malloc((size_t)nl * sizeof(size_t));
    memcpy(tcl_save, tcl, (size_t)nl * sizeof(size_t));

    int accepted = 0;
    for (int k = 0; k < k_actual; k++) {
      /* Get target's opinion on this position */
      size_t target_token =
          sample_token(target_logits, vocab->vocab_size, temperature);

      if (target_token == spec_tokens[k]) {
        /* Accept: draft and target agree */
        char ch = (char)vocab->chars[target_token];
        if (ch == '\n' || target_token == vocab->bos_id) {
          /* Stop token — accepted but we're done */
          goto done;
        }
        output[out_pos++] = ch;
        accepted++;
        total_accepted++;

        /* Advance target KV cache */
        forward_inference(target->model, target_token, pos, tk, tv, tcl,
                          target_logits);
        pos++;

        if (out_pos >= max_len)
          goto done;
      } else {
        /* Reject: use target's token as recovery */
        char ch = (char)vocab->chars[target_token];
        if (ch == '\n' || target_token == vocab->bos_id)
          goto done;

        output[out_pos++] = ch;

        /* Advance target past the recovery token */
        forward_inference(target->model, target_token, pos, tk, tv, tcl,
                          target_logits);
        pos++;

        /* Roll back draft to match target state */
        /* Reset draft KV by re-syncing from the last accepted position */
        memcpy(dcl, tcl_save, (size_t)nl * sizeof(size_t));
        /* Re-process accepted + recovery tokens through draft */
        for (int j = 0; j < accepted; j++) {
          forward_inference(draft->model, spec_tokens[j], draft_pos_save + j,
                            dk, dv, dcl, draft_logits);
        }
        forward_inference(draft->model, target_token, draft_pos_save + accepted,
                          dk, dv, dcl, draft_logits);

        if (out_pos >= max_len) {
          free(dcl_save);
          free(tcl_save);
          goto done;
        }
        break; /* restart speculation from new position */
      }
    }

    /* If all k tokens accepted, advance draft KV cache to stay in sync */
    if (accepted == k_actual) {
      /* Draft already has tokens in KV; just reset to match pos */
      memcpy(dcl, dcl_save, (size_t)nl * sizeof(size_t));
      for (int k = 0; k < k_actual; k++) {
        forward_inference(draft->model, spec_tokens[k], draft_pos_save + k, dk,
                          dv, dcl, draft_logits);
      }
    }

    free(dcl_save);
    free(tcl_save);
  }

done:
  output[out_pos] = '\0';

  if (accepted_out)
    *accepted_out = total_accepted;
  if (drafted_out)
    *drafted_out = total_drafted;

  /* Cleanup */
  for (int l = 0; l < nl; l++) {
    kv_cache_free(dk[l]);
    kv_cache_free(dv[l]);
    kv_cache_free(tk[l]);
    kv_cache_free(tv[l]);
  }
  free(dk);
  free(dv);
  free(dcl);
  free(tk);
  free(tv);
  free(tcl);
  free(draft_logits);
  free(target_logits);
  free(spec_tokens);
}

/* ======================== Organelle Training ===============================
 */

/*
 * organelle_train — Full training pipeline for a single organelle.
 *
 *   This function encapsulates the entire lifecycle of training one organelle:
 *
 *   1. CORPUS LOADING: Reads the training data using opa_load_docs_multiline()
 *      which handles multi-line documents separated by blank lines.
 *
 *   2. VOCABULARY BUILDING: Constructs a character-level vocabulary from the
 *      corpus (typically 30-80 unique characters for game corpora).
 *
 *   3. CHECKPOINT RESUME: Attempts to load an existing checkpoint.  If found,
 *      training is skipped entirely — the organelle is ready for inference.
 *      This is the "instant inference" path used when pre-trained models are
 *      copied from models/organelles/ to the build directory.
 *
 *   4. MULTI-THREADED TRAINING: Runs the standard MicroGPT training loop
 *      using the same TrainWorker parallelism model as the foundation demos.
 *      Each thread processes a slice of the batch independently, gradients
 *      are accumulated, normalised by total positions, and applied via Adam.
 *
 *      Training loop invariants:
 *        - Gradients are accumulated across threads, then mean-normalised
 *        - doc_cursor wraps around the corpus (no epoch boundaries)
 *        - Loss is logged every 1000 steps and at step 0
 *        - Best loss is tracked for the training summary
 *
 *   5. CHECKPOINT + LOG SAVING: The model checkpoint (.ckpt) is saved using
 *      the standard MicroGPT binary format.  A training log (.ckpt.log) is
 *      appended with run metadata, architecture, loss curve, and timing.
 *
 *   Memory ownership:
 *     - The returned Organelle owns the model, docs, and vocab
 *     - Adam buffers (m_adam, v_adam) and gradient buffers are freed before
 * return
 *     - Thread worker resources are freed before return
 *     - Caller must free the Organelle via organelle_free()
 *
 *   @param name         Human-readable name (e.g. "connect4_planner")
 *   @param corpus_path  Path to the training corpus file
 *   @param ckpt_path    Path for saving/loading the checkpoint (.ckpt)
 *   @param cfg          Architecture and training hyperparameters
 *   @param num_steps    Total training steps (typically 25000)
 *   @return             Trained organelle ready for inference, or NULL on error
 */
static Organelle *organelle_train_impl_(const char *name,
                                        const char *corpus_path,
                                        const char *ckpt_path,
                                        MicrogptConfig *cfg, int num_steps,
                                        const Model *source_model) {
  const int nl = cfg->n_layer;

  /* Derive training log path by appending ".log" to the checkpoint path.
   * e.g. "connect4_planner.ckpt" → "connect4_planner.ckpt.log"
   * This ensures every checkpoint has a paired log file. */
  char log_path[512];
  snprintf(log_path, sizeof(log_path), "%s.log", ckpt_path);

  printf("\n========================================\n");
  printf("ORGANELLE: %s\n", name);
  printf("========================================\n");

  Organelle *org = (Organelle *)calloc(1, sizeof(Organelle));
  if (!org)
    return NULL;

  /* ---- Phase 1: Load corpus ---- */
  if (opa_load_docs_multiline(corpus_path, &org->docs, cfg->max_docs) != 0) {
    fprintf(stderr, "ERROR: cannot open %s\n", corpus_path);
    free(org);
    return NULL;
  }

  /* Report corpus statistics */
  size_t total_chars = 0;
  for (size_t i = 0; i < org->docs.num_docs; i++)
    total_chars += org->docs.doc_lens[i];
  printf("corpus: %zu docs | %zu chars (%.1f KB)\n", org->docs.num_docs,
         total_chars, (double)total_chars / 1024.0);

  /* ---- Phase 2: Build vocabulary ---- */
  build_vocab(&org->docs, &org->vocab);
  printf("vocab: %zu characters\n", org->vocab.vocab_size);

  /* ---- Phase 3: Create model and attempt checkpoint resume ---- */
  int resume_step = 0;
  Model *model = model_create(org->vocab.vocab_size, cfg);
  if (!model) {
    fprintf(stderr, "ERROR: model_create failed for %s\n", name);
    free_docs(&org->docs);
    free(org);
    return NULL;
  }

  /* Transfer internal weights from source model if provided (K-13) */
  if (source_model) {
    printf("  Transferring internal weights from source model...\n");
    model_transfer_weights(source_model, model, cfg);
  }
  size_t nparams = model_num_params(model);

  /* Allocate Adam optimiser state (first and second moment buffers) */
  scalar_t *m_adam = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *v_adam = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  /* Try loading an existing checkpoint — if found, skip training entirely */
  Model *loaded = checkpoint_load(ckpt_path, org->vocab.vocab_size, cfg, m_adam,
                                  v_adam, &resume_step);
  if (loaded) {
    printf("loaded checkpoint %s (step %d) — skipping training\n", ckpt_path,
           resume_step);

    /* Log the checkpoint load event for traceability */
    FILE *logf = fopen(log_path, "a");
    if (logf) {
      time_t now = time(NULL);
      struct tm *lt = localtime(&now);
      fprintf(logf, "\n========================================\n");
      fprintf(logf, "Run: %04d-%02d-%02d %02d:%02d:%02d\n", lt->tm_year + 1900,
              lt->tm_mon + 1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
      fprintf(logf, "========================================\n");
      fprintf(logf, "Organelle: %s\n", name);
      fprintf(logf, "Loaded checkpoint: %s (step %d)\n", ckpt_path,
              resume_step);
      fprintf(logf, "========================================\n\n");
      fclose(logf);
    }

    /* Discard the scratch model; use the loaded one */
    model_free(model);
    free(m_adam);
    free(v_adam);
    org->model = loaded;
    return org;
  }

  /* ---- Phase 4: Training ---- */
  printf("params: %zu | steps %d | lr %.4f\n\n", nparams, num_steps,
         (double)cfg->learning_rate);

  /* Open training log in append mode — previous runs are preserved */
  FILE *logf = fopen(log_path, "a");
  if (logf) {
    time_t now = time(NULL);
    struct tm *lt = localtime(&now);
    fprintf(logf, "\n========================================\n");
    fprintf(logf, "Run: %04d-%02d-%02d %02d:%02d:%02d\n", lt->tm_year + 1900,
            lt->tm_mon + 1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
    fprintf(logf, "========================================\n");
    fprintf(logf, "Organelle: %s\n", name);
    fprintf(logf, "Corpus: %zu docs | %zu chars (%.1f KB)\n",
            org->docs.num_docs, total_chars, (double)total_chars / 1024.0);
    fprintf(logf, "Vocab: %zu characters\n", org->vocab.vocab_size);
    fprintf(logf,
            "Architecture: N_EMBD=%d N_LAYER=%d N_HEAD=%d BLOCK_SIZE=%d "
            "MLP_DIM=%d\n",
            cfg->n_embd, cfg->n_layer, cfg->n_head, cfg->block_size,
            cfg->mlp_dim);
    fprintf(logf, "Params: %zu\n", nparams);
    fprintf(logf, "Training: batch=%d steps=%d lr=%.4f\n", cfg->batch_size,
            num_steps, (double)cfg->learning_rate);
    fprintf(logf, "\n--- Training ---\n");
    fflush(logf);
  }

  /* Shuffle documents for better gradient variance */
  shuffle_docs(&org->docs);

  /* Determine thread count — auto-detect but cap at corpus size
   * (no point having more threads than documents per batch) */
  int nthreads = mgpt_default_threads(cfg->batch_size);
  if (nthreads > (int)org->docs.num_docs)
    nthreads = (int)org->docs.num_docs;
  if (nthreads < 1)
    nthreads = 1;

  /* ---- Allocate per-thread training resources ---- */
  TrainWorker *workers =
      (TrainWorker *)calloc((size_t)nthreads, sizeof(TrainWorker));
  mgpt_thread_t *threads =
      (mgpt_thread_t *)calloc((size_t)nthreads, sizeof(mgpt_thread_t));
  mgpt_thread_trampoline_t *tramps = (mgpt_thread_trampoline_t *)calloc(
      (size_t)nthreads, sizeof(mgpt_thread_trampoline_t));

  for (int t = 0; t < nthreads; t++) {
    workers[t].model = model;
    workers[t].docs = &org->docs;
    workers[t].vocab = &org->vocab;
    workers[t].grads = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    workers[t].keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    workers[t].values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    workers[t].cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    workers[t].token_buf =
        (size_t *)malloc(((size_t)cfg->block_size + 2) * sizeof(size_t));
    for (int l = 0; l < nl; l++) {
      workers[t].keys[l] = kv_cache_alloc(cfg);
      workers[t].values[l] = kv_cache_alloc(cfg);
    }
    workers[t].rng_seed = (unsigned int)(42 + t);
  }

  /* ---- Training loop ---- */
  time_t t0 = time(NULL);
  scalar_t best_loss = 1e9;
  int doc_cursor = 0; /* wrapping cursor into the document array */
  scalar_t *grads = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  for (int step = 0; step < num_steps; step++) {
    /* Zero the accumulated gradient buffer */
    memset(grads, 0, nparams * sizeof(scalar_t));

    /* Distribute batch documents across threads.
     * Each thread gets docs_per_thread documents, with the remainder
     * distributed to the first few threads (round-robin). */
    int docs_per_step = cfg->batch_size;
    int docs_per_thread = docs_per_step / nthreads;
    int remainder = docs_per_step % nthreads;

    int cursor = doc_cursor;
    for (int t = 0; t < nthreads; t++) {
      int count = docs_per_thread + (t < remainder ? 1 : 0);
      workers[t].batch_start = cursor % (int)org->docs.num_docs;
      workers[t].batch_end = (cursor + count) % (int)org->docs.num_docs;
      if (workers[t].batch_end <= workers[t].batch_start && count > 0)
        workers[t].batch_end = workers[t].batch_start + count;
      workers[t].loss = 0;
      workers[t].positions = 0;
      memset(workers[t].grads, 0, nparams * sizeof(scalar_t));
      for (int l = 0; l < nl; l++)
        workers[t].cache_len[l] = 0;
      cursor += count;
    }
    doc_cursor = cursor;

    /* Launch all worker threads */
    for (int t = 0; t < nthreads; t++) {
      mgpt_thread_create(&threads[t], &tramps[t], train_worker_run,
                         &workers[t]);
    }

    /* Join threads and accumulate results:
     * - Sum per-thread losses → batch_loss
     * - Sum per-thread positions → batch_positions
     * - Sum per-thread gradient buffers → grads[] */
    scalar_t batch_loss = 0;
    size_t batch_positions = 0;
    for (int t = 0; t < nthreads; t++) {
      mgpt_thread_join(threads[t]);
      batch_loss += workers[t].loss;
      batch_positions += workers[t].positions;
      for (size_t p = 0; p < nparams; p++)
        grads[p] += workers[t].grads[p];
    }

    /* Normalise gradients by total positions (mean gradient) */
    if (batch_positions > 0) {
      scalar_t scale = 1.0 / (scalar_t)batch_positions;
      for (size_t p = 0; p < nparams; p++)
        grads[p] *= scale;
    }

    scalar_t mean_loss =
        batch_positions > 0 ? batch_loss / (scalar_t)batch_positions : 0;

    /* Apply gradient clipping (no-op if GRAD_CLIP <= 0) */
    clip_gradients(grads, nparams);

    /* Apply Adam optimiser step with cosine LR schedule */
    adam_step(model, grads, m_adam, v_adam, step);

    /* Progress logging: every 1000 steps + step 0 */
    if ((step + 1) % 1000 == 0 || step == 0) {
      double elapsed = difftime(time(NULL), t0);
      if (elapsed < 1.0)
        elapsed = 1.0;
      double eta = (num_steps - step - 1) / ((step + 1) / elapsed);
      printf("  [%s] step %5d/%d | loss %.4f | %.0fs elapsed, ETA %.0fs\n",
             name, step + 1, num_steps, mean_loss, elapsed, eta);
      fflush(stdout);

      /* Append loss to training log */
      if (logf) {
        fprintf(logf, "step %5d / %d | loss %.4f\n", step + 1, num_steps,
                mean_loss);
        fflush(logf);
      }
    }

    if (mean_loss < best_loss)
      best_loss = mean_loss;
  }

  /* ---- Phase 5: Save checkpoint and training log ---- */
  double train_time = difftime(time(NULL), t0);
  printf("  [%s] training complete: %.1fs | best loss: %.4f\n", name,
         train_time, best_loss);

  checkpoint_save(model, m_adam, v_adam, num_steps, ckpt_path);
  printf("  [%s] saved checkpoint: %s\n", name, ckpt_path);

  /* Finalise training log with summary */
  if (logf) {
    fprintf(logf, "\nTraining complete: %.1fs | best loss: %.4f\n", train_time,
            best_loss);
    fprintf(logf, "Checkpoint: %s\n", ckpt_path);
    fprintf(logf, "========================================\n\n");
    fclose(logf);
    printf("  [%s] training log: %s\n", name, log_path);
  }

  /* ---- Cleanup: free training-only resources ---- */
  for (int t = 0; t < nthreads; t++) {
    free(workers[t].grads);
    for (int l = 0; l < nl; l++) {
      kv_cache_free(workers[t].keys[l]);
      kv_cache_free(workers[t].values[l]);
    }
    free(workers[t].keys);
    free(workers[t].values);
    free(workers[t].cache_len);
    free(workers[t].token_buf);
  }
  free(workers);
  free(threads);
  free(tramps);
  free(grads);
  free(m_adam);
  free(v_adam);

  org->model = model;
  return org;
}

Organelle *organelle_train(const char *name, const char *corpus_path,
                           const char *ckpt_path, MicrogptConfig *cfg,
                           int num_steps) {
  return organelle_train_impl_(name, corpus_path, ckpt_path, cfg, num_steps,
                               NULL);
}

Organelle *organelle_train_transfer(const char *name, const char *corpus_path,
                                    const char *ckpt_path, MicrogptConfig *cfg,
                                    int num_steps, const Model *source_model) {
  return organelle_train_impl_(name, corpus_path, ckpt_path, cfg, num_steps,
                               source_model);
}

/* ── Model Soup: train N seeds, average weights ────────────────────────── */

/* Helper: evaluate model loss on a sample of the training corpus.
 * Returns average per-position cross-entropy loss over sample_docs documents.
 * This is used by greedy soup to decide whether adding a model improves
 * the average. */
static scalar_t soup_eval_loss_(const Model *model, const Docs *docs,
                                const Vocab *vocab, const MicrogptConfig *cfg,
                                int sample_docs) {
  const size_t ne = (size_t)cfg->n_embd;
  const int nl = cfg->n_layer;
  size_t np = model_num_params(model);

  /* Allocate KV cache and grad buffer (grads discarded) */
  scalar_t **keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **vals = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *clen = (size_t *)calloc((size_t)nl, sizeof(size_t));
  scalar_t *grads = (scalar_t *)calloc(np, sizeof(scalar_t));
  if (!keys || !vals || !clen || !grads) {
    free(keys);
    free(vals);
    free(clen);
    free(grads);
    return 999.0;
  }
  for (int L = 0; L < nl; L++) {
    keys[L] =
        (scalar_t *)calloc((size_t)cfg->block_size * ne, sizeof(scalar_t));
    vals[L] =
        (scalar_t *)calloc((size_t)cfg->block_size * ne, sizeof(scalar_t));
  }

  scalar_t total_loss = 0;
  int total_positions = 0;
  int n =
      (sample_docs < (int)docs->num_docs) ? sample_docs : (int)docs->num_docs;

  size_t *ids =
      (size_t *)malloc(((size_t)cfg->block_size + 2) * sizeof(size_t));
  if (!ids) { /* cleanup and bail */
    for (int L = 0; L < nl; L++) {
      free(keys[L]);
      free(vals[L]);
    }
    free(keys);
    free(vals);
    free(clen);
    free(grads);
    return 999.0;
  }

  for (int d = 0; d < n; d++) {
    size_t ntok = tokenize(docs->lines[d], docs->doc_lens[d], vocab, ids,
                           (size_t)cfg->block_size + 1);
    if (ntok < 3)
      continue;

    /* Reset KV cache */
    for (int L = 0; L < nl; L++)
      clen[L] = 0;
    memset(grads, 0, np * sizeof(scalar_t));

    /* Evaluate loss on first block_size tokens */
    int max_pos = (int)ntok - 1;
    if (max_pos > cfg->block_size)
      max_pos = cfg->block_size;
    for (int p = 0; p < max_pos; p++) {
      scalar_t loss = forward_backward_one(model, ids[p], (size_t)p, ids[p + 1],
                                           keys, vals, clen, grads);
      total_loss += loss;
      total_positions++;
    }
  }

  /* Cleanup */
  free(ids);
  for (int L = 0; L < nl; L++) {
    free(keys[L]);
    free(vals[L]);
  }
  free(keys);
  free(vals);
  free(clen);
  free(grads);

  return (total_positions > 0) ? total_loss / (scalar_t)total_positions : 999.0;
}

Organelle *organelle_train_soup(const char *name, const char *corpus_path,
                                MicrogptConfig *cfg, int num_steps,
                                int n_seeds) {
  if (n_seeds < 1)
    return NULL;
  if (n_seeds == 1) {
    /* Degenerate case: just train one model */
    return organelle_train(name, corpus_path, NULL, cfg, num_steps);
  }

#if defined(MODEL_SOUP_GREEDY) && MODEL_SOUP_GREEDY
  printf("\n  [%s] Greedy Soup: training %d seeds × %d steps\n", name, n_seeds,
         num_steps);
#else
  printf("\n  [%s] Model Soup: training %d seeds × %d steps\n", name, n_seeds,
         num_steps);
#endif

  /* Train N independent models with different seeds */
  Organelle **seeds =
      (Organelle **)calloc((size_t)n_seeds, sizeof(Organelle *));
  if (!seeds)
    return NULL;

  for (int s = 0; s < n_seeds; s++) {
    /* Each seed gets a unique RNG state */
    seed_rng((unsigned int)(42 + s * 7919)); /* 7919 is prime → well-spaced */

    char ckpt_name[256];
    snprintf(ckpt_name, sizeof(ckpt_name), "%s_soup_seed%d.ckpt", name, s);

    printf("  [%s] Seed %d/%d (rng=%d):\n", name, s + 1, n_seeds,
           42 + s * 7919);

    seeds[s] = organelle_train_impl_(name, corpus_path, ckpt_name, cfg,
                                     num_steps, NULL);
    if (!seeds[s]) {
      fprintf(stderr, "  [%s] Seed %d training failed\n", name, s + 1);
      /* Free previously trained seeds */
      for (int j = 0; j < s; j++)
        organelle_free(seeds[j]);
      free(seeds);
      return NULL;
    }
  }

#if defined(MODEL_SOUP_GREEDY) && MODEL_SOUP_GREEDY
  /* ── Greedy Soup: iteratively add models that improve validation loss ── */

  /* Evaluate each seed's loss on a sample of training data */
  scalar_t *losses = (scalar_t *)malloc((size_t)n_seeds * sizeof(scalar_t));
  for (int s = 0; s < n_seeds; s++) {
    losses[s] = soup_eval_loss_(seeds[s]->model, &seeds[s]->docs,
                                &seeds[s]->vocab, cfg, 50);
    printf("  [%s] Seed %d validation loss: %.4f\n", name, s + 1,
           (double)losses[s]);
  }

  /* Find the best single seed */
  int best_idx = 0;
  for (int s = 1; s < n_seeds; s++)
    if (losses[s] < losses[best_idx])
      best_idx = s;

  printf("  [%s] Best seed: %d (loss=%.4f)\n", name, best_idx + 1,
         (double)losses[best_idx]);

  /* Start with the best seed, try adding others */
  /* We'll keep the running average in a separate model to allow rollback */
  int n_in_soup = 1;
  int *in_soup = (int *)calloc((size_t)n_seeds, sizeof(int));
  in_soup[best_idx] = 1;

  for (int s = 0; s < n_seeds; s++) {
    if (s == best_idx)
      continue;

    /* Try averaging this seed into the current soup */
    Model **trial_models =
        (Model **)malloc(((size_t)n_in_soup + 1) * sizeof(Model *));
    int ti = 0;
    for (int j = 0; j < n_seeds; j++)
      if (in_soup[j])
        trial_models[ti++] = seeds[j]->model;
    trial_models[ti] = seeds[s]->model;

    /* Create a temporary model to test the average */
    Model *trial = model_create(seeds[s]->vocab.vocab_size, cfg);
    model_soup_average(trial, trial_models, n_in_soup + 1);

    /* Evaluate */
    scalar_t trial_loss = soup_eval_loss_(trial, &seeds[best_idx]->docs,
                                          &seeds[best_idx]->vocab, cfg, 50);

    scalar_t current_loss =
        soup_eval_loss_(seeds[best_idx]->model, &seeds[best_idx]->docs,
                        &seeds[best_idx]->vocab, cfg, 50);

    printf("  [%s] + seed %d: trial_loss=%.4f vs current=%.4f → ", name, s + 1,
           (double)trial_loss, (double)current_loss);

    if (trial_loss <= current_loss) {
      /* Accept: copy trial weights into best seed's model */
      model_soup_average(seeds[best_idx]->model, trial_models, n_in_soup + 1);
      in_soup[s] = 1;
      n_in_soup++;
      printf("ACCEPTED (%d seeds in soup)\n", n_in_soup);
    } else {
      printf("REJECTED\n");
    }

    model_free(trial);
    free(trial_models);
  }

  free(losses);
  free(in_soup);

  /* Keep best_idx (now contains greedy-averaged weights), free the rest */
  Organelle *result = seeds[best_idx];
  for (int s = 0; s < n_seeds; s++)
    if (s != best_idx)
      organelle_free(seeds[s]);
  free(seeds);

  printf("  [%s] Greedy soup complete: %d/%d seeds selected\n\n", name,
         n_in_soup, n_seeds);
  return result;

#else
  /* ── Uniform Soup: average all models' weights ── */

  Model **models = (Model **)malloc((size_t)n_seeds * sizeof(Model *));
  if (!models) {
    for (int s = 0; s < n_seeds; s++)
      organelle_free(seeds[s]);
    free(seeds);
    return NULL;
  }
  for (int s = 0; s < n_seeds; s++)
    models[s] = seeds[s]->model;

  model_soup_average(seeds[0]->model, models, n_seeds);
  free(models);

  /* Keep seeds[0] (now contains averaged weights), free the rest */
  Organelle *result = seeds[0];
  for (int s = 1; s < n_seeds; s++)
    organelle_free(seeds[s]);
  free(seeds);

  printf("  [%s] Soup complete: %d seeds averaged\n\n", name, n_seeds);
  return result;
#endif
}

/* ======================== Organelle Cleanup ================================
 */

/*
 * organelle_free — Release all memory owned by an organelle.
 *
 *   Frees the model (weights, layer pointers), document corpus
 *   (raw data buffer, line indices), and the Organelle struct itself.
 *   Safe to call with NULL (no-op).
 *
 *   @param org  Organelle to free (may be NULL)
 */
void organelle_free(Organelle *org) {
  if (!org)
    return;
  if (org->model)
    model_free(org->model);
  free_docs(&org->docs);
  if (org->word_level)
    free_word_vocab(&org->word_vocab);
  free(org);
}

/* ======================== Word-Level Organelle =============================
 */
/*
 * Word-level variants of the organelle train/generate pipeline.
 *
 * These mirror the character-level API but use WordVocab (from microgpt.h)
 * instead of Vocab.  Key differences:
 *   - Vocabulary is built via build_word_vocab() (frequency-based, top-N)
 *   - Tokenisation uses tokenize_words() (whitespace-delimited)
 *   - Training is single-threaded (inline loop, not TrainWorker)
 *   - Generation maps sampled tokens back to word strings
 *
 * Why single-threaded training: TrainWorker uses `const Vocab *vocab` and
 * train_worker_run() calls char-level tokenize().  Modifying TrainWorker
 * would affect all existing experiments.  Single-threaded is simpler and
 * sufficient for the initial word-level experiments (corpora are shorter
 * in word tokens, so training is already faster per step).
 */

/*
 * organelle_train_words — Full training pipeline for a word-level organelle.
 *
 *   1. Load corpus via opa_load_docs_multiline()
 *   2. Concatenate all docs into a single text buffer for vocabulary building
 *   3. Build word-level vocabulary (top max_words by frequency)
 *   4. Create model with vocab_size = word_vocab.vocab_size
 *   5. Single-threaded training loop:
 *      - For each step, pick a random doc
 *      - Tokenize with tokenize_words() (BOS prepended)
 *      - Forward+backward for each position pair
 *      - Accumulate gradients, apply Adam with cosine LR
 *   6. Save checkpoint + training log
 *   7. Set org->word_level = 1
 */
Organelle *organelle_train_words(const char *name, const char *corpus_path,
                                 const char *ckpt_path, MicrogptConfig *cfg,
                                 int num_steps, size_t max_words) {
  const int nl = cfg->n_layer;

  /* Derive training log path */
  char log_path[512];
  snprintf(log_path, sizeof(log_path), "%s.log", ckpt_path);

  printf("\n========================================\n");
  printf("ORGANELLE (word-level): %s\n", name);
  printf("========================================\n");

  Organelle *org = (Organelle *)calloc(1, sizeof(Organelle));
  if (!org)
    return NULL;
  org->word_level = 1;

  /* ---- Phase 1: Load corpus ---- */
  if (opa_load_docs_multiline(corpus_path, &org->docs, cfg->max_docs) != 0) {
    fprintf(stderr, "ERROR: cannot open %s\n", corpus_path);
    free(org);
    return NULL;
  }

  /* Concatenate all docs into a single text buffer for vocab building */
  size_t total_chars = 0;
  for (size_t i = 0; i < org->docs.num_docs; i++)
    total_chars += org->docs.doc_lens[i];

  char *all_text = (char *)malloc(total_chars + 1);
  if (!all_text) {
    free_docs(&org->docs);
    free(org);
    return NULL;
  }
  size_t offset = 0;
  for (size_t i = 0; i < org->docs.num_docs; i++) {
    memcpy(all_text + offset, org->docs.lines[i], org->docs.doc_lens[i]);
    offset += org->docs.doc_lens[i];
  }
  all_text[total_chars] = '\0';

  printf("corpus: %zu docs | %zu chars (%.1f KB)\n", org->docs.num_docs,
         total_chars, (double)total_chars / 1024.0);

  /* ---- Phase 2: Build word vocabulary ---- */
  if (build_word_vocab(all_text, total_chars, max_words, &org->word_vocab) !=
      0) {
    fprintf(stderr, "ERROR: build_word_vocab failed for %s\n", name);
    free(all_text);
    free_docs(&org->docs);
    free(org);
    return NULL;
  }
  free(all_text); /* no longer needed after vocab is built */

  printf("vocab: %zu word tokens (max_words=%zu, unk_id=%zu, nl_id=%zu, "
         "bos_id=%zu)\n",
         org->word_vocab.vocab_size, max_words, org->word_vocab.unk_id,
         org->word_vocab.newline_id, org->word_vocab.bos_id);

  /* ---- Phase 3: Create model and attempt checkpoint resume ---- */
  int resume_step = 0;
  Model *model = model_create(org->word_vocab.vocab_size, cfg);
  if (!model) {
    fprintf(stderr, "ERROR: model_create failed for %s\n", name);
    free_word_vocab(&org->word_vocab);
    free_docs(&org->docs);
    free(org);
    return NULL;
  }
  size_t nparams = model_num_params(model);

  /* Allocate Adam optimiser state */
  scalar_t *m_adam = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *v_adam = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  /* Try loading an existing checkpoint */
  if (ckpt_path) {
    Model *loaded = checkpoint_load(ckpt_path, org->word_vocab.vocab_size, cfg,
                                    m_adam, v_adam, &resume_step);
    if (loaded) {
      printf("loaded checkpoint %s (step %d) -- skipping training\n", ckpt_path,
             resume_step);
      model_free(model);
      free(m_adam);
      free(v_adam);
      org->model = loaded;
      return org;
    }
  }

  /* ---- Phase 4: Single-threaded word-level training ---- */
  printf("params: %zu | steps %d | lr %.4f\n\n", nparams, num_steps,
         (double)cfg->learning_rate);

  /* Open training log */
  FILE *logf = fopen(log_path, "a");
  if (logf) {
    time_t now = time(NULL);
    struct tm *lt = localtime(&now);
    fprintf(logf, "\n========================================\n");
    fprintf(logf, "Run: %04d-%02d-%02d %02d:%02d:%02d\n", lt->tm_year + 1900,
            lt->tm_mon + 1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
    fprintf(logf, "========================================\n");
    fprintf(logf, "Organelle (word-level): %s\n", name);
    fprintf(logf, "Corpus: %zu docs | %zu chars (%.1f KB)\n",
            org->docs.num_docs, total_chars, (double)total_chars / 1024.0);
    fprintf(logf, "Vocab: %zu word tokens (max_words=%zu)\n",
            org->word_vocab.vocab_size, max_words);
    fprintf(logf,
            "Architecture: N_EMBD=%d N_LAYER=%d N_HEAD=%d BLOCK_SIZE=%d "
            "MLP_DIM=%d\n",
            cfg->n_embd, cfg->n_layer, cfg->n_head, cfg->block_size,
            cfg->mlp_dim);
    fprintf(logf, "Params: %zu\n", nparams);
    fprintf(logf, "Training: steps=%d lr=%.4f\n", num_steps,
            (double)cfg->learning_rate);
    fprintf(logf, "\n--- Training ---\n");
    fflush(logf);
  }

  /* Allocate token buffer and gradient buffer */
  size_t *token_buf =
      (size_t *)malloc(((size_t)cfg->block_size + 2) * sizeof(size_t));
  scalar_t *grads = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  /* Allocate KV cache for training */
  scalar_t **keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int l = 0; l < nl; l++) {
    keys[l] = kv_cache_alloc(cfg);
    values[l] = kv_cache_alloc(cfg);
  }

  shuffle_docs(&org->docs);

  time_t t0 = time(NULL);
  scalar_t best_loss = 1e9;
  unsigned int rng_seed = 42;

  for (int step = 0; step < num_steps; step++) {
    memset(grads, 0, nparams * sizeof(scalar_t));
    scalar_t step_loss = 0;
    size_t step_positions = 0;

    /* Process batch_size documents per step */
    int docs_this_step = cfg->batch_size;
    if (docs_this_step < 1)
      docs_this_step = 1;

    for (int b = 0; b < docs_this_step; b++) {
      /* Reset KV cache for each document */
      for (int l = 0; l < nl; l++)
        cache_len[l] = 0;

      /* Pick a random document */
      size_t di = (size_t)rand_r(&rng_seed) % org->docs.num_docs;
      const char *doc = org->docs.lines[di];
      size_t doc_len = org->docs.doc_lens[di];

      /* Tokenize at word level: [BOS] word1 word2 ... */
      token_buf[0] = org->word_vocab.bos_id;
      size_t n_tok = tokenize_words(doc, doc_len, &org->word_vocab,
                                    token_buf + 1, (size_t)cfg->block_size) +
                     1;

      size_t n = n_tok - 1;
      if (n > (size_t)cfg->block_size)
        n = (size_t)cfg->block_size;
      if (n == 0)
        continue;

      step_positions += n;

      /* Forward+backward for each position */
      for (size_t pos = 0; pos < n; pos++) {
        scalar_t loss =
            forward_backward_one(model, token_buf[pos], pos, token_buf[pos + 1],
                                 keys, values, cache_len, grads);
        step_loss += loss;
      }
    }

    /* Normalise gradients by total positions */
    if (step_positions > 0) {
      scalar_t scale = 1.0 / (scalar_t)step_positions;
      for (size_t p = 0; p < nparams; p++)
        grads[p] *= scale;
    }

    scalar_t mean_loss =
        step_positions > 0 ? step_loss / (scalar_t)step_positions : 0;

    clip_gradients(grads, nparams);
    adam_step(model, grads, m_adam, v_adam, step);

    /* Progress logging */
    if ((step + 1) % 1000 == 0 || step == 0) {
      double elapsed = difftime(time(NULL), t0);
      if (elapsed < 1.0)
        elapsed = 1.0;
      double eta = (num_steps - step - 1) / ((step + 1) / elapsed);
      printf("  [%s] step %5d/%d | loss %.4f | %.0fs elapsed, ETA %.0fs\n",
             name, step + 1, num_steps, mean_loss, elapsed, eta);
      fflush(stdout);

      if (logf) {
        fprintf(logf, "step %5d / %d | loss %.4f\n", step + 1, num_steps,
                mean_loss);
        fflush(logf);
      }
    }

    if (mean_loss < best_loss)
      best_loss = mean_loss;
  }

  /* ---- Phase 5: Save checkpoint and training log ---- */
  double train_time = difftime(time(NULL), t0);
  printf("  [%s] training complete: %.1fs | best loss: %.4f\n", name,
         train_time, best_loss);

  if (ckpt_path) {
    checkpoint_save(model, m_adam, v_adam, num_steps, ckpt_path);
    printf("  [%s] saved checkpoint: %s\n", name, ckpt_path);
  }

  if (logf) {
    fprintf(logf, "\nTraining complete: %.1fs | best loss: %.4f\n", train_time,
            best_loss);
    if (ckpt_path)
      fprintf(logf, "Checkpoint: %s\n", ckpt_path);
    fprintf(logf, "========================================\n\n");
    fclose(logf);
    printf("  [%s] training log: %s\n", name, log_path);
  }

  /* ---- Cleanup training resources ---- */
  for (int l = 0; l < nl; l++) {
    kv_cache_free(keys[l]);
    kv_cache_free(values[l]);
  }
  free(keys);
  free(values);
  free(cache_len);
  free(token_buf);
  free(grads);
  free(m_adam);
  free(v_adam);

  org->model = model;
  return org;
}

/* ── organelle_generate_words ────────────────────────────────────────────────
 */
/*
 * Word-level inference.  Tokenises the prompt into word tokens, feeds them
 * through the model, then auto-regressively samples word tokens.
 *
 * Output format: words are separated by spaces.  The output stops at:
 *   - Newline token (word_vocab.newline_id)
 *   - BOS token
 *   - max_len characters written
 *   - block_size reached
 */
void organelle_generate_words(const Organelle *org, const MicrogptConfig *cfg,
                              const char *prompt, char *output, int max_len,
                              scalar_t temperature) {
  const int nl = cfg->n_layer;
  const WordVocab *wv = &org->word_vocab;

  /* Allocate KV cache */
  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int l = 0; l < nl; l++) {
    inf_keys[l] = kv_cache_alloc(cfg);
    inf_values[l] = kv_cache_alloc(cfg);
  }

  scalar_t *logits_buf =
      (scalar_t *)malloc((size_t)cfg->max_vocab * sizeof(scalar_t));

  int pos = 0;
  int out_pos = 0;

  /* Step 1: Feed BOS token */
  size_t token = wv->bos_id;
  forward_inference(org->model, token, pos, inf_keys, inf_values, inf_cache_len,
                    logits_buf);
  pos++;

  /* Step 2: Tokenize prompt at word level and feed each token */
  size_t prompt_len = strlen(prompt);
  size_t *prompt_ids =
      (size_t *)malloc(((size_t)cfg->block_size + 1) * sizeof(size_t));
  size_t n_prompt = tokenize_words(prompt, prompt_len, wv, prompt_ids,
                                   (size_t)cfg->block_size);

  for (size_t i = 0; i < n_prompt && pos < cfg->block_size - 1; i++) {
    forward_inference(org->model, prompt_ids[i], pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;
  }
  free(prompt_ids);

  /* Step 3: Feed newline separator (signals "your turn to respond") */
  forward_inference(org->model, wv->newline_id, pos, inf_keys, inf_values,
                    inf_cache_len, logits_buf);
  pos++;

  /* Step 4: Auto-regressive word decoding */
  for (int g = 0; g < max_len && pos < cfg->block_size; g++) {
    token = sample_token(logits_buf, wv->vocab_size, temperature);

    /* Stop on BOS token */
    if (token == wv->bos_id)
      break;

    /* Stop on newline token */
    if (token == wv->newline_id)
      break;

    /* Map token back to word string */
    const char *word = (token < wv->vocab_size && wv->words[token])
                           ? wv->words[token]
                           : "<unk>";

    /* Append word to output with space separator */
    size_t wlen = strlen(word);
    if (out_pos > 0 && out_pos < max_len) {
      output[out_pos++] = ' ';
    }
    for (size_t c = 0; c < wlen && out_pos < max_len; c++) {
      output[out_pos++] = word[c];
    }

    if (out_pos >= max_len)
      break;

    forward_inference(org->model, token, pos, inf_keys, inf_values,
                      inf_cache_len, logits_buf);
    pos++;
  }

  output[out_pos] = '\0';

  /* Cleanup */
  for (int l = 0; l < nl; l++) {
    kv_cache_free(inf_keys[l]);
    kv_cache_free(inf_values[l]);
  }
  free(inf_keys);
  free(inf_values);
  free(inf_cache_len);
  free(logits_buf);
}

/* ── organelle_generate_words_ensemble ───────────────────────────────────────
 */
/*
 * Majority-vote inference for word-level organelles.
 * Same pattern as organelle_generate_ensemble() but uses word-level inference.
 */
void organelle_generate_words_ensemble(const Organelle *org,
                                       const MicrogptConfig *cfg,
                                       const char *prompt, char *output,
                                       int max_len, int n_votes,
                                       scalar_t base_temp,
                                       scalar_t *confidence) {
  if (n_votes < 1)
    n_votes = 1;
  if (n_votes > OPA_MAX_VOTES)
    n_votes = OPA_MAX_VOTES;

  if (n_votes == 1) {
    organelle_generate_words(org, cfg, prompt, output, max_len, base_temp);
    if (confidence)
      *confidence = (scalar_t)1.0;
    return;
  }

  char candidates[OPA_MAX_VOTES][128];
  int vote_counts[OPA_MAX_VOTES];
  int unique = 0;

  for (int v = 0; v < n_votes; v++) {
    scalar_t jitter =
        OPA_TEMP_JITTER * ((scalar_t)v - (scalar_t)n_votes / (scalar_t)2.0);
    scalar_t temp = base_temp + jitter;
    if (temp < (scalar_t)0.01)
      temp = (scalar_t)0.01;

    char buf[128];
    int gen_len = max_len < 127 ? max_len : 127;
    organelle_generate_words(org, cfg, prompt, buf, gen_len, temp);

    int found = 0;
    for (int u = 0; u < unique; u++) {
      if (strcmp(candidates[u], buf) == 0) {
        vote_counts[u]++;
        found = 1;
        break;
      }
    }
    if (!found && unique < OPA_MAX_VOTES) {
      strncpy(candidates[unique], buf, 127);
      candidates[unique][127] = '\0';
      vote_counts[unique] = 1;
      unique++;
    }
  }

  int best_idx = 0;
  for (int u = 1; u < unique; u++) {
    if (vote_counts[u] > vote_counts[best_idx])
      best_idx = u;
  }

  strncpy(output, candidates[best_idx], (size_t)max_len);
  output[max_len] = '\0';

  if (confidence)
    *confidence = (scalar_t)vote_counts[best_idx] / (scalar_t)n_votes;
}

/*
 * The Kanban board is a lightweight state tracker used during inference
 * to prevent the pipeline from repeating failed actions.  It maintains:
 *
 *   blocked[] — CSV string of actions that have been tried and failed.
 *               Once an action is blocked, the pipeline will skip it
 *               and try alternatives via opa_valid_fallback().
 *
 *   last[]    — Fixed-length history of the N most recent actions taken.
 *               Used for cycle detection and debugging.
 *
 *   stalls    — Counter: how many consecutive turns produced no progress.
 *   replans   — Counter: how many times the planner was re-invoked.
 *
 * All strings use a comma-delimited format: "R,D,L,U"
 */

/*
 * opa_kanban_init — Initialise a Kanban board to its empty state.
 *
 *   @param kb            Kanban board to initialise
 *   @param max_history   Maximum number of actions to retain in last[]
 */
void opa_kanban_init(OpaKanban *kb, int max_history) {
  kb->blocked[0] = '\0';
  kb->last[0] = '\0';
  kb->stalls = 0;
  kb->replans = 0;
  kb->max_history = max_history;
}

/*
 * opa_kanban_add_blocked — Add an action to the blocked list.
 *
 *   If the action is already blocked, this is a no-op (idempotent).
 *   Actions are appended as comma-separated values: "R,D,L"
 *
 *   @param kb      Kanban board
 *   @param action  Action string to block (e.g. "R", "move_left")
 */
void opa_kanban_add_blocked(OpaKanban *kb, const char *action) {
  if (kb->blocked[0] != '\0') {
    /* Check for duplicates — don't block the same action twice */
    if (strstr(kb->blocked, action) != NULL)
      return;
    /* Append with comma separator if there's room */
    size_t len = strlen(kb->blocked);
    if (len + strlen(action) + 2 < sizeof(kb->blocked)) {
      kb->blocked[len] = ',';
      strcpy(kb->blocked + len + 1, action);
    }
  } else {
    /* First blocked action */
    strncpy(kb->blocked, action, sizeof(kb->blocked) - 1);
    kb->blocked[sizeof(kb->blocked) - 1] = '\0';
  }
}

/*
 * opa_kanban_is_blocked — Test if an action is on the blocked list.
 *
 *   Uses substring matching, so "R" will match in "R,D,L".
 *   This is safe because action names are short and distinct.
 *
 *   @param kb      Kanban board
 *   @param action  Action to check
 *   @return        1 if blocked, 0 if not
 */
int opa_kanban_is_blocked(const OpaKanban *kb, const char *action) {
  return strstr(kb->blocked, action) != NULL;
}

/* opa_kanban_clear_blocked — Clear all blocked actions (fresh start). */
void opa_kanban_clear_blocked(OpaKanban *kb) { kb->blocked[0] = '\0'; }

/*
 * opa_kanban_add_last — Append an action to the recent-history ring buffer.
 *
 *   Maintains at most max_history entries.  When full, the oldest entry
 *   is dropped (FIFO).  Entries are stored as comma-separated values.
 *
 *   @param kb      Kanban board
 *   @param action  Action to record (e.g. "R", "D")
 */
void opa_kanban_add_last(OpaKanban *kb, const char *action) {
  if (kb->max_history <= 0)
    return;

  /* Count existing entries by counting commas + 1 */
  int count = 0;
  if (kb->last[0] != '\0') {
    count = 1;
    for (const char *p = kb->last; *p; p++)
      if (*p == ',')
        count++;
  }

  /* If at capacity, drop the oldest entry (everything before first comma) */
  if (count >= kb->max_history) {
    char *comma = strchr(kb->last, ',');
    if (comma) {
      memmove(kb->last, comma + 1, strlen(comma + 1) + 1);
    } else {
      kb->last[0] = '\0';
    }
  }

  /* Append the new action */
  if (kb->last[0] == '\0') {
    strncpy(kb->last, action, sizeof(kb->last) - 1);
    kb->last[sizeof(kb->last) - 1] = '\0';
  } else {
    size_t len = strlen(kb->last);
    if (len + strlen(action) + 2 < sizeof(kb->last)) {
      kb->last[len] = ',';
      strcpy(kb->last + len + 1, action);
    }
  }
}

/* ======================== Cycle Detector ================================== */

/*
 * The cycle detector watches for repetitive action patterns that indicate
 * the pipeline is stuck.  It uses a circular buffer of OPA_CYCLE_WINDOW
 * recent actions and checks for A,B,A,B oscillation — the most common
 * failure mode in two-organelle pipelines.
 *
 * Example: Moving a puzzle piece right, then left, then right, then left
 * is a cycle.  The detector catches this after 3 recorded actions when
 * the 4th proposed action would complete the pattern.
 */

/*
 * opa_cycle_init — Reset the cycle detector to its empty state.
 *
 *   @param cd  Cycle detector to reset
 */
void opa_cycle_init(OpaCycleDetector *cd) {
  cd->len = 0;
  memset(cd->history, 0, sizeof(cd->history));
}

/*
 * opa_cycle_detected — Check if a proposed action would create an A,B,A,B
 * cycle.
 *
 *   Examines the 3 most recent actions in the history buffer:
 *     history[-3] = A, history[-2] = B, history[-1] = A
 *   If proposed_action == B, then we have A,B,A,B → cycle detected.
 *
 *   @param cd               Cycle detector
 *   @param proposed_action  Action ID being considered
 *   @return                 1 if cycle detected, 0 if safe to proceed
 */
int opa_cycle_detected(const OpaCycleDetector *cd, int proposed_action) {
  if (cd->len < 3 || proposed_action < 0)
    return 0;

  /* Extract the 3 most recent actions from the circular buffer */
  int h1 = cd->history[(cd->len - 1) % OPA_CYCLE_WINDOW]; /* most recent */
  int h2 = cd->history[(cd->len - 2) % OPA_CYCLE_WINDOW]; /* 2nd most recent */
  int h3 = cd->history[(cd->len - 3) % OPA_CYCLE_WINDOW]; /* 3rd most recent */

  /* Pattern: h3=A, h2=B, h1=A, proposed=B → A,B,A,B cycle */
  return (h2 == proposed_action && h1 == h3);
}

/*
 * opa_cycle_other — Return the "other" action in a detected A,B,A,B cycle.
 *
 *   When a cycle is detected, this returns B (the most recent action),
 *   which can be used to decide what to block.
 *
 *   @param cd               Cycle detector
 *   @param proposed_action  The proposed action (not used, but validates
 * context)
 *   @return                 The alternating action ID, or -1 if no history
 */
int opa_cycle_other(const OpaCycleDetector *cd, int proposed_action) {
  (void)proposed_action;
  if (cd->len < 1)
    return -1;
  return cd->history[(cd->len - 1) % OPA_CYCLE_WINDOW];
}

/*
 * opa_cycle_record — Record an action into the cycle detection window.
 *
 *   Writes to a circular buffer of OPA_CYCLE_WINDOW entries.
 *
 *   @param cd         Cycle detector
 *   @param action_id  Action ID to record
 */
void opa_cycle_record(OpaCycleDetector *cd, int action_id) {
  cd->history[cd->len % OPA_CYCLE_WINDOW] = action_id;
  cd->len++;
}

/* ======================== Pipe-String Helpers ==============================
 */

/*
 * The "flat-string protocol" is the inter-organelle communication format.
 * Instead of structured data (JSON, protobuf), organelles pass pipe-delimited
 * key=value strings that are both human-readable and trivially parseable in C:
 *
 *   "STATE|board=123456780|blank=8|moves=R,D,L|PLAN|dir=R"
 *
 * This design choice has several advantages for edge deployment:
 *   - No parser library needed (pure C string operations)
 *   - Constant-time field extraction via strstr()
 *   - Human-readable for debugging
 *   - Compatible with the character-level models (no special tokens)
 */

/*
 * opa_extract_pipe_value — Extract a value from a pipe-delimited string.
 *
 *   Searches for "key=" in the buffer and returns a pointer to the
 *   value that follows, NUL-terminating it at the next pipe '|' or
 *   newline '\n'.
 *
 *   WARNING: This function modifies the input buffer in-place by
 *   replacing delimiter characters with '\0'.  The returned pointer
 *   points into the modified buffer.
 *
 *   @param buf  Mutable buffer containing the pipe string
 *   @param key  Key to search for (e.g. "board", "dir")
 *   @return     Pointer to the value, or NULL if key not found
 */
const char *opa_extract_pipe_value(char *buf, const char *key) {
  char search[64];
  snprintf(search, sizeof(search), "%s=", key);

  char *p = strstr(buf, search);
  if (!p)
    return NULL;

  p += strlen(search); /* skip past "key=" to the value */

  /* NUL-terminate at the next delimiter (pipe or newline) */
  char *end = strchr(p, '|');
  if (end)
    *end = '\0';
  char *nl = strchr(p, '\n');
  if (nl)
    *nl = '\0';

  return p;
}

/*
 * opa_pipe_starts_with — Test if a pipe string begins with a given prefix.
 *
 *   @param buf     Buffer to test
 *   @param prefix  Prefix string (e.g. "STATE|", "PLAN|")
 *   @return        1 if buf starts with prefix, 0 otherwise
 */
int opa_pipe_starts_with(const char *buf, const char *prefix) {
  return strncmp(buf, prefix, strlen(prefix)) == 0;
}

/* ======================== Valid-Move Filter =============================== */

/*
 * The valid-move filter is a post-processing step that constrains organelle
 * outputs to legal game actions.  Without this, the model might generate
 * syntactically correct but game-illegal moves (e.g. moving a piece off
 * the board).
 *
 * The game engine provides a CSV string of valid moves for the current
 * state (e.g. "R,D,L").  The filter checks if the organelle's output
 * appears in this list, and if not, falls back to the first valid,
 * non-blocked alternative.
 */

/*
 * opa_valid_filter — Check if an action appears in a CSV valid-move list.
 *
 *   Performs exact token matching (not substring matching) by comparing
 *   each comma-delimited field against the action string.
 *
 *   @param action     Action string to validate (e.g. "R")
 *   @param valid_csv  CSV of valid moves (e.g. "R,D,L"), or NULL for no
 * constraint
 *   @return           1 if valid (or no constraint), 0 if not in the list
 */
int opa_valid_filter(const char *action, const char *valid_csv) {
  if (!valid_csv || valid_csv[0] == '\0')
    return 1; /* no constraint = all valid */
  if (!action || action[0] == '\0')
    return 0;

  /* Walk through the CSV, comparing each field */
  size_t action_len = strlen(action);
  const char *p = valid_csv;
  while (*p) {
    const char *comma = strchr(p, ',');
    size_t field_len = comma ? (size_t)(comma - p) : strlen(p);

    /* Exact match: same length and same bytes */
    if (field_len == action_len && strncmp(p, action, action_len) == 0)
      return 1;

    if (!comma)
      break;
    p = comma + 1; /* advance past comma to next field */
  }
  return 0;
}

/*
 * opa_valid_fallback — Find the first valid, non-blocked action.
 *
 *   Iterates through the valid-move CSV and returns the first action
 *   that is NOT on the Kanban blocked list.  This is the fallback path
 *   when the organelle's preferred action is invalid or blocked.
 *
 *   @param kb           Kanban board (for blocked-action check)
 *   @param valid_csv    CSV of valid moves
 *   @param fallback     Output buffer for the fallback action
 *   @param fallback_sz  Size of the output buffer
 *   @return             1 if a fallback was found, 0 if all valid moves are
 * blocked
 */
int opa_valid_fallback(const OpaKanban *kb, const char *valid_csv,
                       char *fallback, size_t fallback_sz) {
  if (!valid_csv || valid_csv[0] == '\0')
    return 0;

  const char *p = valid_csv;
  while (*p) {
    const char *comma = strchr(p, ',');
    size_t field_len = comma ? (size_t)(comma - p) : strlen(p);

    /* Extract this field into a temp buffer for comparison */
    char tok[16];
    size_t copy_len = field_len < sizeof(tok) - 1 ? field_len : sizeof(tok) - 1;
    memcpy(tok, p, copy_len);
    tok[copy_len] = '\0';

    /* Return the first valid action that is not blocked */
    if (!opa_kanban_is_blocked(kb, tok)) {
      size_t out_len = copy_len < fallback_sz - 1 ? copy_len : fallback_sz - 1;
      memcpy(fallback, tok, out_len);
      fallback[out_len] = '\0';
      return 1;
    }

    if (!comma)
      break;
    p = comma + 1;
  }

  return 0; /* all valid moves are blocked — deadlock */
}

/* ======================== Ensemble Voting ================================= */

/*
 * organelle_generate_ensemble — Majority-vote inference across N temperature-
 *   jittered samples.
 *
 *   When a single organelle inference is unreliable (e.g. the model sometimes
 *   generates invalid moves), ensemble voting improves robustness by running
 *   N independent inferences with slightly different temperatures and
 *   returning the most common output.
 *
 *   Temperature jitter: each of the N votes uses a temperature offset from
 *   the base temperature.  The offsets are evenly spaced around zero:
 *     temp[v] = base_temp + OPA_TEMP_JITTER × (v - N/2)
 *
 *   This produces diversity without straying far from the intended temperature.
 *   For n_votes=5 and base_temp=0.5 with jitter=0.05:
 *     temps = [0.40, 0.45, 0.50, 0.55, 0.60]
 *
 *   The confidence score is the fraction of votes for the winning candidate:
 *     confidence = vote_count_of_winner / n_votes
 *   A confidence of 1.0 means all votes agreed (high certainty).
 *   A confidence of 0.2 means only 1 out of 5 agreed (very uncertain).
 *
 *   @param org         Organelle to use
 *   @param cfg         Runtime configuration
 *   @param prompt      Input prompt string
 *   @param output      Buffer for the winning response (caller-owned)
 *   @param max_len     Maximum response length
 *   @param n_votes     Number of inference samples (1-OPA_MAX_VOTES)
 *   @param base_temp   Base sampling temperature
 *   @param confidence  Optional output: vote fraction of the winner [0.0, 1.0]
 */
void organelle_generate_ensemble(const Organelle *org,
                                 const MicrogptConfig *cfg, const char *prompt,
                                 char *output, int max_len, int n_votes,
                                 scalar_t base_temp, scalar_t *confidence) {
  /* Clamp n_votes to valid range */
  if (n_votes < 1)
    n_votes = 1;
  if (n_votes > OPA_MAX_VOTES)
    n_votes = OPA_MAX_VOTES;

  /* Short-circuit: 1 vote = no ensemble overhead */
  if (n_votes == 1) {
    organelle_generate(org, cfg, prompt, output, max_len, base_temp);
    if (confidence)
      *confidence = (scalar_t)1.0;
    return;
  }

  const int nl = cfg->n_layer;
  const Vocab *vocab = &org->vocab;

  /* ── Step 1: Process prompt ONCE to build shared KV cache ── */
  scalar_t **prefix_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **prefix_values =
      (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *prefix_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int l = 0; l < nl; l++) {
    prefix_keys[l] = kv_cache_alloc(cfg);
    prefix_values[l] = kv_cache_alloc(cfg);
  }

  scalar_t *prompt_logits =
      (scalar_t *)malloc((size_t)cfg->max_vocab * sizeof(scalar_t));
  int pos = 0;

  /* Feed BOS */
  size_t token = vocab->bos_id;
  forward_inference(org->model, token, pos, prefix_keys, prefix_values,
                    prefix_cache_len, prompt_logits);
  pos++;

  /* Feed prompt characters */
  for (int i = 0; prompt[i] && pos < cfg->block_size - 1; i++) {
    token = 0;
    for (size_t v = 0; v < vocab->vocab_size; v++) {
      if (vocab->chars[v] == (unsigned char)prompt[i]) {
        token = v;
        break;
      }
    }
    forward_inference(org->model, token, pos, prefix_keys, prefix_values,
                      prefix_cache_len, prompt_logits);
    pos++;
  }

  /* Feed newline separator */
  token = 0;
  for (size_t v = 0; v < vocab->vocab_size; v++) {
    if (vocab->chars[v] == '\n') {
      token = v;
      break;
    }
  }
  forward_inference(org->model, token, pos, prefix_keys, prefix_values,
                    prefix_cache_len, prompt_logits);
  pos++;

  /* ── Step 2: Run N decode-only passes with shared prefix cache ── */
  char candidates[OPA_MAX_VOTES][128];
  int vote_counts[OPA_MAX_VOTES];
  int unique = 0;

  for (int v = 0; v < n_votes; v++) {
    scalar_t jitter =
        OPA_TEMP_JITTER * ((scalar_t)v - (scalar_t)n_votes / (scalar_t)2.0);
    scalar_t temp = base_temp + jitter;
    if (temp < (scalar_t)0.01)
      temp = (scalar_t)0.01;

    /* Copy prefix KV cache for this vote */
    scalar_t **vote_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    scalar_t **vote_values =
        (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    size_t *vote_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    for (int l = 0; l < nl; l++) {
      vote_keys[l] = kv_cache_alloc(cfg);
      vote_values[l] = kv_cache_alloc(cfg);
      kv_cache_copy(prefix_keys[l], vote_keys[l], cfg, prefix_cache_len[l]);
      vote_cache_len[l] = prefix_cache_len[l];
    }

    /* Decode from cached prefix (skips prompt re-processing) */
    char buf[128];
    int gen_len = max_len < 127 ? max_len : 127;
    organelle_generate_from_cache(org, cfg, vote_keys, vote_values,
                                  vote_cache_len, pos, prompt_logits, buf,
                                  gen_len, temp);

    /* Free vote's KV caches */
    for (int l = 0; l < nl; l++) {
      kv_cache_free(vote_keys[l]);
      kv_cache_free(vote_values[l]);
    }
    free(vote_keys);
    free(vote_values);
    free(vote_cache_len);

    /* Tally: find matching candidate or register a new one */
    int found = 0;
    for (int u = 0; u < unique; u++) {
      if (strcmp(candidates[u], buf) == 0) {
        vote_counts[u]++;
        found = 1;
        break;
      }
    }
    if (!found && unique < OPA_MAX_VOTES) {
      strncpy(candidates[unique], buf, 127);
      candidates[unique][127] = '\0';
      vote_counts[unique] = 1;
      unique++;
    }
  }

  /* Free prefix KV caches */
  for (int l = 0; l < nl; l++) {
    kv_cache_free(prefix_keys[l]);
    kv_cache_free(prefix_values[l]);
  }
  free(prefix_keys);
  free(prefix_values);
  free(prefix_cache_len);
  free(prompt_logits);

  /* Find the mode: the candidate with the most votes */
  int best_idx = 0;
  for (int u = 1; u < unique; u++) {
    if (vote_counts[u] > vote_counts[best_idx])
      best_idx = u;
  }

  /* Copy the winning candidate to output */
  strncpy(output, candidates[best_idx], (size_t)max_len);
  output[max_len] = '\0';

  /* Report confidence: fraction of votes for the winner */
  if (confidence)
    *confidence = (scalar_t)vote_counts[best_idx] / (scalar_t)n_votes;
}

/* ======================== Reasoning Trace Recorder ======================== */

/*
 * The reasoning trace recorder captures the pipeline's decision-making process
 * for a single run (one puzzle solve or game play).  Each step records:
 *   - What action was proposed
 *   - What happened (accepted/rejected/stall/replan/cycle-break)
 *   - Progress metric before and after
 *   - Kanban blocked state at the time of decision
 *   - Whether the action came from the model or a fallback
 *
 * The serialised trace can be fed back as training data to teach organelles
 * the pipeline's coordination logic: "process retrieval" rather than
 * "answer retrieval."
 */

static const char *opa_outcome_str(OpaStepOutcome o) {
  switch (o) {
  case OPA_STEP_ACCEPTED:
    return "accepted";
  case OPA_STEP_REJECTED:
    return "rejected";
  case OPA_STEP_STALL:
    return "stall";
  case OPA_STEP_REPLAN:
    return "replan";
  case OPA_STEP_CYCLE_BREAK:
    return "cycle_break";
  default:
    return "unknown";
  }
}

/*
 * opa_trace_init — Initialise a trace recorder to its empty state.
 *
 *   @param trace           Trace to initialise
 *   @param initial_metric  Starting progress metric (e.g. manhattan distance)
 */
void opa_trace_init(OpaTrace *trace, int initial_metric) {
  memset(trace, 0, sizeof(OpaTrace));
  trace->initial_metric = initial_metric;
  trace->final_metric = initial_metric; /* default until finalised */
}

/*
 * opa_trace_record — Record one pipeline step in the trace.
 *
 *   Steps beyond OPA_TRACE_MAX_STEPS are silently dropped (no overflow).
 *
 *   @param trace          Trace recorder
 *   @param action         Proposed action string (e.g. "up", "ABCD")
 *   @param outcome        What happened to this action
 *   @param metric_before  Progress metric before this step
 *   @param metric_after   Progress metric after (-1 if rejected/N/A)
 *   @param blocked        Current kanban blocked[] string (may be NULL)
 *   @param from_model     1 = model-sourced, 0 = fallback
 */
void opa_trace_record(OpaTrace *trace, const char *action,
                      OpaStepOutcome outcome, int metric_before,
                      int metric_after, const char *blocked, int from_model) {
  if (trace->num_steps >= OPA_TRACE_MAX_STEPS)
    return; /* silently clamp */

  OpaTraceStep *s = &trace->steps[trace->num_steps];
  s->step = trace->num_steps + 1; /* 1-indexed */
  s->outcome = outcome;
  s->metric_before = metric_before;
  s->metric_after = metric_after;
  s->from_model = from_model;

  /* Safe copy of action string */
  if (action) {
    strncpy(s->action, action, sizeof(s->action) - 1);
    s->action[sizeof(s->action) - 1] = '\0';
  } else {
    s->action[0] = '\0';
  }

  /* Safe copy of blocked snapshot */
  if (blocked) {
    strncpy(s->blocked_snapshot, blocked, sizeof(s->blocked_snapshot) - 1);
    s->blocked_snapshot[sizeof(s->blocked_snapshot) - 1] = '\0';
  } else {
    s->blocked_snapshot[0] = '\0';
  }

  trace->num_steps++;
}

/*
 * opa_trace_finalise — Mark the trace as complete.
 *
 *   @param trace         Trace to finalise
 *   @param final_metric  Ending progress metric
 *   @param solved        1 if goal was reached, 0 if not
 */
void opa_trace_finalise(OpaTrace *trace, int final_metric, int solved) {
  trace->final_metric = final_metric;
  trace->solved = solved;
}

/*
 * opa_trace_to_corpus — Serialise a trace to corpus-format text.
 *
 *   Format (one line per step):
 *     "step|action|outcome|metric_before>metric_after|blocked|src\n"
 *
 *   A header line gives context:
 *     "TRACE|initial=N|final=N|solved=0|steps=N\n"
 *
 *   @param trace   Trace to serialise
 *   @param buf     Output buffer
 *   @param buf_sz  Buffer size
 *   @return        Bytes written (excluding NUL), or -1 if buffer too small
 */
int opa_trace_to_corpus(const OpaTrace *trace, char *buf, size_t buf_sz) {
  if (!buf || buf_sz == 0)
    return -1;

  size_t pos = 0;

  /* Header line */
  int n = snprintf(buf + pos, buf_sz - pos,
                   "TRACE|initial=%d|final=%d|solved=%d|steps=%d\n",
                   trace->initial_metric, trace->final_metric, trace->solved,
                   trace->num_steps);
  if (n < 0 || (size_t)n >= buf_sz - pos)
    return -1;
  pos += (size_t)n;

  /* One line per step */
  for (int i = 0; i < trace->num_steps; i++) {
    const OpaTraceStep *s = &trace->steps[i];
    n = snprintf(buf + pos, buf_sz - pos, "%d|%s|%s|%d>%d|%s|%s\n", s->step,
                 s->action, opa_outcome_str(s->outcome), s->metric_before,
                 s->metric_after,
                 s->blocked_snapshot[0] ? s->blocked_snapshot : "none",
                 s->from_model ? "model" : "fallback");
    if (n < 0 || (size_t)n >= buf_sz - pos)
      return -1;
    pos += (size_t)n;
  }

  return (int)pos;
}

/*
 * opa_trace_write — Append a serialised trace to a file.
 *
 *   Each trace is separated by a blank line so the corpus loader
 *   can treat each trace as a separate document.
 *
 *   @param trace  Trace to write
 *   @param path   File path (created if absent, appended if exists)
 *   @return       0 on success, -1 on error
 */
int opa_trace_write(const OpaTrace *trace, const char *path) {
  char buf[4096];
  int len = opa_trace_to_corpus(trace, buf, sizeof(buf));
  if (len < 0)
    return -1;

  FILE *f = fopen(path, "a");
  if (!f)
    return -1;

  fwrite(buf, 1, (size_t)len, f);
  fprintf(f, "\n"); /* blank line separator between traces */
  fclose(f);
  return 0;
}

/*
 * opa_trace_count — Count steps matching a specific outcome.
 *
 *   @param trace    Trace to examine
 *   @param outcome  Outcome type to count
 *   @return         Number of matching steps
 */
int opa_trace_count(const OpaTrace *trace, OpaStepOutcome outcome) {
  int count = 0;
  for (int i = 0; i < trace->num_steps; i++) {
    if (trace->steps[i].outcome == outcome)
      count++;
  }
  return count;
}

/*
 * opa_trace_has_recovery — Detect non-monotonic recovery in the trace.
 *
 *   A "recovery" occurs when the progress metric increases (regression)
 *   at some step, then later decreases (progress) at a subsequent step.
 *   This pattern indicates a successful detour: the system accepted a
 *   temporary penalty to reach a better overall position.
 *
 *   This is the hardest pattern for a retrieval system to learn, because
 *   it violates the greedy assumption that every step should improve the
 *   metric.
 *
 *   @param trace  Trace to examine
 *   @return       1 if a recovery pattern exists, 0 otherwise
 */
int opa_trace_has_recovery(const OpaTrace *trace) {
  int saw_regression = 0;
  for (int i = 0; i < trace->num_steps; i++) {
    const OpaTraceStep *s = &trace->steps[i];
    if (s->metric_after < 0)
      continue; /* skip rejected steps (no metric) */

    if (s->metric_after > s->metric_before) {
      /* Metric increased = regression */
      saw_regression = 1;
    } else if (s->metric_after < s->metric_before && saw_regression) {
      /* Metric decreased after a previous regression = recovery */
      return 1;
    }
  }
  return 0;
}
