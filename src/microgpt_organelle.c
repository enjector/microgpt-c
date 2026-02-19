/*
 * MicroGPT-C — Organelle Pipeline Architecture (OPA) Library
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 * Implementation of generic OPA infrastructure.
 * See microgpt_organelle.h for API documentation.
 */

#include "microgpt_organelle.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ======================== Corpus Loader =================================== */

int opa_load_docs_multiline(const char *path, Docs *docs, int max_docs) {
  FILE *f = fopen(path, "r");
  if (!f)
    return -1;

  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  docs->data = (char *)malloc((size_t)fsize + 1);
  if (!docs->data) {
    fclose(f);
    return -1;
  }
  fread(docs->data, 1, (size_t)fsize, f);
  docs->data[fsize] = '\0';
  fclose(f);

  docs->lines = (char **)malloc(sizeof(char *) * (size_t)max_docs);
  docs->doc_lens = (size_t *)malloc(sizeof(size_t) * (size_t)max_docs);
  docs->num_docs = 0;

  char *p = docs->data;
  while (*p && docs->num_docs < (size_t)max_docs) {
    while (*p == '\n')
      p++;
    if (!*p)
      break;

    char *doc_start = p;
    while (*p && !(*p == '\n' && (*(p + 1) == '\n' || *(p + 1) == '\0')))
      p++;
    if (*p == '\n')
      p++;

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

void organelle_generate(const Organelle *org, const MicrogptConfig *cfg,
                        const char *prompt, char *output, int max_len,
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

  /* Feed BOS */
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

  /* Feed newline separator */
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

  /* Generate response */
  for (int g = 0; g < max_len && pos < cfg->block_size; g++) {
    token = sample_token(logits_buf, vocab->vocab_size, temperature);
    if (token == vocab->bos_id)
      break;

    char ch = (char)vocab->chars[token];
    if (ch == '\n')
      break;

    output[out_pos++] = ch;

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

/* ======================== Organelle Training ===============================
 */

Organelle *organelle_train(const char *name, const char *corpus_path,
                           const char *ckpt_path, MicrogptConfig *cfg,
                           int num_steps) {
  const int nl = cfg->n_layer;

  printf("\n========================================\n");
  printf("ORGANELLE: %s\n", name);
  printf("========================================\n");

  Organelle *org = (Organelle *)calloc(1, sizeof(Organelle));
  if (!org)
    return NULL;

  if (opa_load_docs_multiline(corpus_path, &org->docs, cfg->max_docs) != 0) {
    fprintf(stderr, "ERROR: cannot open %s\n", corpus_path);
    free(org);
    return NULL;
  }

  size_t total_chars = 0;
  for (size_t i = 0; i < org->docs.num_docs; i++)
    total_chars += org->docs.doc_lens[i];
  printf("corpus: %zu docs | %zu chars (%.1f KB)\n", org->docs.num_docs,
         total_chars, (double)total_chars / 1024.0);

  build_vocab(&org->docs, &org->vocab);
  printf("vocab: %zu characters\n", org->vocab.vocab_size);

  int resume_step = 0;
  Model *model = model_create(org->vocab.vocab_size, cfg);
  if (!model) {
    fprintf(stderr, "ERROR: model_create failed for %s\n", name);
    free_docs(&org->docs);
    free(org);
    return NULL;
  }
  size_t nparams = model_num_params(model);

  scalar_t *m_adam = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *v_adam = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  Model *loaded = checkpoint_load(ckpt_path, org->vocab.vocab_size, cfg, m_adam,
                                  v_adam, &resume_step);
  if (loaded) {
    printf("loaded checkpoint %s (step %d) — skipping training\n", ckpt_path,
           resume_step);
    model_free(model);
    free(m_adam);
    free(v_adam);
    org->model = loaded;
    return org;
  }

  printf("params: %zu | steps %d | lr %.4f\n\n", nparams, num_steps,
         (double)cfg->learning_rate);

  shuffle_docs(&org->docs);

  int nthreads = mgpt_default_threads(cfg->batch_size);
  if (nthreads > (int)org->docs.num_docs)
    nthreads = (int)org->docs.num_docs;
  if (nthreads < 1)
    nthreads = 1;

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

  time_t t0 = time(NULL);
  scalar_t best_loss = 1e9;
  int doc_cursor = 0;
  scalar_t *grads = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  for (int step = 0; step < num_steps; step++) {
    memset(grads, 0, nparams * sizeof(scalar_t));

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

    for (int t = 0; t < nthreads; t++) {
      mgpt_thread_create(&threads[t], &tramps[t], train_worker_run,
                         &workers[t]);
    }

    scalar_t batch_loss = 0;
    size_t batch_positions = 0;
    for (int t = 0; t < nthreads; t++) {
      mgpt_thread_join(threads[t]);
      batch_loss += workers[t].loss;
      batch_positions += workers[t].positions;
      for (size_t p = 0; p < nparams; p++)
        grads[p] += workers[t].grads[p];
    }

    if (batch_positions > 0) {
      scalar_t scale = 1.0 / (scalar_t)batch_positions;
      for (size_t p = 0; p < nparams; p++)
        grads[p] *= scale;
    }

    scalar_t mean_loss =
        batch_positions > 0 ? batch_loss / (scalar_t)batch_positions : 0;

    adam_step(model, grads, m_adam, v_adam, step);

    if ((step + 1) % 1000 == 0 || step == 0) {
      double elapsed = difftime(time(NULL), t0);
      if (elapsed < 1.0)
        elapsed = 1.0;
      double eta = (num_steps - step - 1) / ((step + 1) / elapsed);
      printf("  [%s] step %5d/%d | loss %.4f | %.0fs elapsed, ETA %.0fs\n",
             name, step + 1, num_steps, mean_loss, elapsed, eta);
      fflush(stdout);
    }

    if (mean_loss < best_loss)
      best_loss = mean_loss;
  }

  double train_time = difftime(time(NULL), t0);
  printf("  [%s] training complete: %.1fs | best loss: %.4f\n", name,
         train_time, best_loss);

  checkpoint_save(model, m_adam, v_adam, num_steps, ckpt_path);
  printf("  [%s] saved checkpoint: %s\n", name, ckpt_path);

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

/* ======================== Organelle Free ================================== */

void organelle_free(Organelle *org) {
  if (!org)
    return;
  if (org->model)
    model_free(org->model);
  free_docs(&org->docs);
  free(org);
}

/* ======================== Kanban State ==================================== */

void opa_kanban_init(OpaKanban *kb, int max_history) {
  kb->blocked[0] = '\0';
  kb->last[0] = '\0';
  kb->stalls = 0;
  kb->replans = 0;
  kb->max_history = max_history;
}

void opa_kanban_add_blocked(OpaKanban *kb, const char *action) {
  if (kb->blocked[0] != '\0') {
    if (strstr(kb->blocked, action) != NULL)
      return;
    size_t len = strlen(kb->blocked);
    if (len + strlen(action) + 2 < sizeof(kb->blocked)) {
      kb->blocked[len] = ',';
      strcpy(kb->blocked + len + 1, action);
    }
  } else {
    strncpy(kb->blocked, action, sizeof(kb->blocked) - 1);
    kb->blocked[sizeof(kb->blocked) - 1] = '\0';
  }
}

int opa_kanban_is_blocked(const OpaKanban *kb, const char *action) {
  return strstr(kb->blocked, action) != NULL;
}

void opa_kanban_clear_blocked(OpaKanban *kb) { kb->blocked[0] = '\0'; }

void opa_kanban_add_last(OpaKanban *kb, const char *action) {
  if (kb->max_history <= 0)
    return;

  /* Count existing entries */
  int count = 0;
  if (kb->last[0] != '\0') {
    count = 1;
    for (const char *p = kb->last; *p; p++)
      if (*p == ',')
        count++;
  }

  if (count >= kb->max_history) {
    char *comma = strchr(kb->last, ',');
    if (comma) {
      memmove(kb->last, comma + 1, strlen(comma + 1) + 1);
    } else {
      kb->last[0] = '\0';
    }
  }

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

void opa_cycle_init(OpaCycleDetector *cd) {
  cd->len = 0;
  memset(cd->history, 0, sizeof(cd->history));
}

int opa_cycle_detected(const OpaCycleDetector *cd, int proposed_action) {
  /* Detect A,B,A,B pattern: we have [.., A, B, A] and proposed is B
   * or equivalently [.., B, A, B] and proposed is A  —
   * Check: hist[-2] == proposed AND hist[-1] == hist[-3] */
  if (cd->len < 3 || proposed_action < 0)
    return 0;
  int h1 = cd->history[(cd->len - 1) % OPA_CYCLE_WINDOW]; /* most recent */
  int h2 = cd->history[(cd->len - 2) % OPA_CYCLE_WINDOW]; /* 2nd most recent */
  int h3 = cd->history[(cd->len - 3) % OPA_CYCLE_WINDOW]; /* 3rd most recent */
  return (h2 == proposed_action && h1 == h3);
}

int opa_cycle_other(const OpaCycleDetector *cd, int proposed_action) {
  /* Returns the B when proposed is A in A,B,A,B */
  if (cd->len < 1)
    return -1;
  return cd->history[(cd->len - 1) % OPA_CYCLE_WINDOW];
}

void opa_cycle_record(OpaCycleDetector *cd, int action_id) {
  cd->history[cd->len % OPA_CYCLE_WINDOW] = action_id;
  cd->len++;
}

/* ======================== Pipe-String Helpers ==============================
 */

const char *opa_extract_pipe_value(char *buf, const char *key) {
  char search[64];
  snprintf(search, sizeof(search), "%s=", key);
  char *p = strstr(buf, search);
  if (!p)
    return NULL;
  p += strlen(search);
  char *end = strchr(p, '|');
  if (end)
    *end = '\0';
  char *nl = strchr(p, '\n');
  if (nl)
    *nl = '\0';
  return p;
}

int opa_pipe_starts_with(const char *buf, const char *prefix) {
  return strncmp(buf, prefix, strlen(prefix)) == 0;
}
