/*
 * MicroGPT-C — 8-Puzzle Multi-Organelle Demo (v3 — Generalisation)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Organelle Pipeline Architecture (OPA) with
 * displacement-based representation for genuine generalisation:
 *
 *   2 neural organelles (Strategist, Mover) + deterministic Judge
 *   collaborate via pipe-separated flat strings:
 *     - d=...:   per-tile displacement vector (structural encoding)
 *     - b=N:     blank position (0-8)
 *     - x=DIR:   blocked direction (prevents fixation)
 *
 * Pipeline flow:
 *   1. Strategist: d=...|md=N → "t=X" (most displaced tile index)
 *   2. Mover:      d=...|b=N[|x=DIR] → "up"/"down"/"left"/"right"
 *   3. Judge:      deterministic apply_move() boundary check
 *   4. If out-of-bounds → add to blocked, retry Mover
 *   5. Repeat until solved or max iterations reached
 *
 * The displacement encoding makes the model learn STRUCTURAL patterns
 * ("reduce displacement of the most-misplaced tile") rather than
 * memorising board→direction lookup tables.
 *
 * Build:
 *   cmake --build build --target puzzle8_demo
 *   ./build/puzzle8_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_thread.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Configuration ---- */
#define STRATEGIST_CORPUS "puzzle8_strategist.txt"
#define MOVER_CORPUS "puzzle8_mover.txt"
#define JUDGE_CORPUS "puzzle8_judge.txt"

#define STRATEGIST_CKPT "puzzle8_strategist_v3.ckpt"
#define MOVER_CKPT "puzzle8_mover_v3.ckpt"
#define JUDGE_CKPT "puzzle8_judge_v3.ckpt"

#define ORGANELLE_TEMP 0.2 /* low temperature for reliable retrieval */
#define INF_GEN_LEN 80     /* max chars per organelle generation */

#define MAX_PIPELINE_ITERS 40 /* max moves attempted per puzzle */
#define NUM_EASY_PUZZLES 10   /* md 1-4 puzzles */
#define NUM_MEDIUM_PUZZLES 10 /* md 5-8 puzzles */
#define NUM_HARD_PUZZLES 10   /* md 9+ puzzles */
#define NUM_TEST_PUZZLES                                                       \
  (NUM_EASY_PUZZLES + NUM_MEDIUM_PUZZLES + NUM_HARD_PUZZLES)
#define REPLAN_THRESHOLD 6 /* stalls before clearing blocked */

/* ---- File-scoped runtime config (shared by helper functions) ---- */
static MicrogptConfig g_cfg;

/* ---- Forward declarations ---- */
static int apply_move(int *board, const char *dir);
static int manhattan_distance(const int *board);

/* ---- Board Helpers ---- */

static int GOAL_BOARD[9] = {1, 2, 3, 4, 5, 6, 7, 8, 0};

static void board_to_str(const int *board, char *out) {
  for (int i = 0; i < 9; i++)
    out[i] = '0' + board[i];
  out[9] = '\0';
}

static void md_delta_str(const int *board, char *out, size_t out_sz) {
  /* Compute md after each possible move. 'x' means illegal.
   * Format: m=U,D,L,R  where values are the resulting md or 'x'. */
  const char *dir_names[] = {"up", "down", "left", "right"};
  int pos = 0;
  pos += snprintf(out + pos, out_sz - (size_t)pos, "m=");
  for (int d = 0; d < 4; d++) {
    if (d > 0)
      pos += snprintf(out + pos, out_sz - (size_t)pos, ",");
    int test[9];
    memcpy(test, board, 9 * sizeof(int));
    if (apply_move(test, dir_names[d])) {
      pos += snprintf(out + pos, out_sz - (size_t)pos, "%d",
                      manhattan_distance(test));
    } else {
      pos += snprintf(out + pos, out_sz - (size_t)pos, "x");
    }
  }
}

static int find_blank(const int *board) {
  for (int i = 0; i < 9; i++)
    if (board[i] == 0)
      return i;
  return -1;
}

static int is_goal(const int *board) {
  return memcmp(board, GOAL_BOARD, sizeof(GOAL_BOARD)) == 0;
}

static int manhattan_distance(const int *board) {
  int dist = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] == 0)
      continue;
    int goal_pos = board[i] - 1;
    dist += abs(i / 3 - goal_pos / 3) + abs(i % 3 - goal_pos % 3);
  }
  return dist;
}

static int apply_move(int *board, const char *dir) {
  int blank = find_blank(board);
  int r = blank / 3, c = blank % 3;
  int nr = r, nc = c;

  if (strcmp(dir, "up") == 0)
    nr = r - 1;
  else if (strcmp(dir, "down") == 0)
    nr = r + 1;
  else if (strcmp(dir, "left") == 0)
    nc = c - 1;
  else if (strcmp(dir, "right") == 0)
    nc = c + 1;
  else
    return 0;

  if (nr < 0 || nr >= 3 || nc < 0 || nc >= 3)
    return 0;

  int ni = nr * 3 + nc;
  board[blank] = board[ni];
  board[ni] = 0;
  return 1;
}

static void scramble_puzzle(int *board, int n_moves, unsigned int *seed) {
  memcpy(board, GOAL_BOARD, sizeof(GOAL_BOARD));
  const char *dirs[] = {"up", "down", "left", "right"};

  for (int i = 0; i < n_moves; i++) {
    for (int attempt = 0; attempt < 20; attempt++) {
      int d = rand_r(seed) % 4;
      int test[9];
      memcpy(test, board, sizeof(GOAL_BOARD));
      if (apply_move(test, dirs[d])) {
        memcpy(board, test, sizeof(GOAL_BOARD));
        break;
      }
    }
  }
}

/* ---- Blocked-Direction Tracker (simplified from Kanban) ---- */

static char g_blocked[64]; /* comma-separated blocked directions */

static void blocked_init(void) { g_blocked[0] = '\0'; }

static void blocked_add(const char *dir) {
  if (g_blocked[0] == '\0') {
    strncpy(g_blocked, dir, sizeof(g_blocked) - 1);
    g_blocked[sizeof(g_blocked) - 1] = '\0';
  } else {
    if (strstr(g_blocked, dir) != NULL)
      return;
    size_t len = strlen(g_blocked);
    if (len + strlen(dir) + 2 < sizeof(g_blocked)) {
      g_blocked[len] = ',';
      strcpy(g_blocked + len + 1, dir);
    }
  }
}

static void blocked_clear(void) { g_blocked[0] = '\0'; }

/* ---- Pipe-String Parser ---- */

static const char *extract_pipe_value(char *buf, const char *key) {
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

/* ---- Multi-line Document Loader ---- */
static int load_docs_multiline(const char *path, Docs *docs, int max_docs) {
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

/* ---- Organelle Inference ---- */

static void organelle_generate(const Model *model, const Vocab *vocab,
                               const char *prompt, char *output, int max_len) {
  const int nl = g_cfg.n_layer;
  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int l = 0; l < nl; l++) {
    inf_keys[l] = kv_cache_alloc(&g_cfg);
    inf_values[l] = kv_cache_alloc(&g_cfg);
  }

  scalar_t *logits_buf =
      (scalar_t *)malloc((size_t)g_cfg.max_vocab * sizeof(scalar_t));
  int pos = 0;
  int out_pos = 0;

  /* Feed BOS */
  size_t token = vocab->bos_id;
  forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                    logits_buf);
  pos++;

  /* Feed prompt characters */
  for (int i = 0; prompt[i] && pos < g_cfg.block_size - 1; i++) {
    token = 0;
    for (size_t v = 0; v < vocab->vocab_size; v++) {
      if (vocab->chars[v] == (unsigned char)prompt[i]) {
        token = v;
        break;
      }
    }
    forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                      logits_buf);
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
  forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                    logits_buf);
  pos++;

  /* Generate response */
  for (int g = 0; g < max_len && pos < g_cfg.block_size; g++) {
    token = sample_token(logits_buf, vocab->vocab_size, ORGANELLE_TEMP);
    if (token == vocab->bos_id)
      break;

    char ch = (char)vocab->chars[token];
    if (ch == '\n')
      break;

    output[out_pos++] = ch;

    forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                      logits_buf);
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

/* ---- Train one organelle ---- */

static Model *train_organelle(const char *name, const char *corpus_path,
                              const char *ckpt_path, Docs *docs, Vocab *vocab,
                              int num_steps) {
  const int nl = g_cfg.n_layer;

  printf("\n========================================\n");
  printf("ORGANELLE: %s\n", name);
  printf("========================================\n");

  if (load_docs_multiline(corpus_path, docs, g_cfg.max_docs) != 0) {
    fprintf(stderr, "ERROR: cannot open %s\n", corpus_path);
    return NULL;
  }

  size_t total_chars = 0;
  for (size_t i = 0; i < docs->num_docs; i++)
    total_chars += docs->doc_lens[i];
  printf("corpus: %zu docs | %zu chars (%.1f KB)\n", docs->num_docs,
         total_chars, (double)total_chars / 1024.0);

  build_vocab(docs, vocab);
  printf("vocab: %zu characters\n", vocab->vocab_size);

  int resume_step = 0;
  Model *model = model_create(vocab->vocab_size, &g_cfg);
  if (!model) {
    fprintf(stderr, "ERROR: model_create failed for %s\n", name);
    return NULL;
  }
  size_t nparams = model_num_params(model);

  scalar_t *m_adam = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  scalar_t *v_adam = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  Model *loaded = checkpoint_load(ckpt_path, vocab->vocab_size, &g_cfg, m_adam,
                                  v_adam, &resume_step);
  if (loaded) {
    printf("loaded checkpoint %s (step %d) — skipping training\n", ckpt_path,
           resume_step);
    model_free(model);
    free(m_adam);
    free(v_adam);
    return loaded;
  }

  printf("params: %zu | steps %d | lr %.4f\n\n", nparams, num_steps,
         (double)g_cfg.learning_rate);

  shuffle_docs(docs);

  int nthreads = mgpt_default_threads(g_cfg.batch_size);
  if (nthreads > (int)docs->num_docs)
    nthreads = (int)docs->num_docs;
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
    workers[t].docs = docs;
    workers[t].vocab = vocab;
    workers[t].grads = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    workers[t].keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    workers[t].values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    workers[t].cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    workers[t].token_buf =
        (size_t *)malloc(((size_t)g_cfg.block_size + 2) * sizeof(size_t));
    for (int l = 0; l < nl; l++) {
      workers[t].keys[l] = kv_cache_alloc(&g_cfg);
      workers[t].values[l] = kv_cache_alloc(&g_cfg);
    }
    workers[t].rng_seed = (unsigned int)(42 + t);
  }

  time_t t0 = time(NULL);
  scalar_t best_loss = 1e9;
  int doc_cursor = 0;
  scalar_t *grads = (scalar_t *)calloc(nparams, sizeof(scalar_t));

  for (int step = 0; step < num_steps; step++) {
    memset(grads, 0, nparams * sizeof(scalar_t));

    int docs_per_step = g_cfg.batch_size;
    int docs_per_thread = docs_per_step / nthreads;
    int remainder = docs_per_step % nthreads;

    int cursor = doc_cursor;
    for (int t = 0; t < nthreads; t++) {
      int count = docs_per_thread + (t < remainder ? 1 : 0);
      workers[t].batch_start = cursor % (int)docs->num_docs;
      workers[t].batch_end = (cursor + count) % (int)docs->num_docs;
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

  return model;
}

/* ---- Test Puzzle Generation (stratified by difficulty) ---- */

static void scramble_to_target_md(int *board, int target_md_min,
                                  int target_md_max, unsigned int *seed) {
  /* Keep scrambling until we land in the target manhattan distance band. */
  for (int attempts = 0; attempts < 500; attempts++) {
    int n_moves = 3 + (int)(rand_r(seed) % 25);
    scramble_puzzle(board, n_moves, seed);
    int md = manhattan_distance(board);
    if (md >= target_md_min && md <= target_md_max && !is_goal(board))
      return;
  }
  /* Fallback: just scramble hard enough */
  scramble_puzzle(board, target_md_min * 2, seed);
}

/* ---- Main ---- */

int main(void) {
  seed_rng(42);

  /* Runtime configuration — puzzle8-specific overrides */
  g_cfg = microgpt_default_config();
  g_cfg.n_embd = 48;
  g_cfg.n_head = 4;
  g_cfg.mlp_dim = 192;
  g_cfg.n_layer = 2;
  g_cfg.block_size = 128;
  g_cfg.batch_size = 8;
  g_cfg.num_steps = 25000;
  g_cfg.learning_rate = 0.001;
  g_cfg.max_vocab = 50;
  g_cfg.max_docs = 60000; /* larger corpus now */
  g_cfg.max_doc_len = 128;
  microgpt_print_config("MicroGPT-C - 8-Puzzle OPA Generalisation Demo (v3)",
                        &g_cfg);

  /* Suppress unused warnings for helper functions */
  (void)extract_pipe_value;

  /* ================================================================
   * PHASE 1: Train organelles (Strategist + Mover + Judge)
   * ================================================================ */

  printf("--- PHASE 1: TRAINING (%d steps each) ---\n", g_cfg.num_steps);

  Docs strat_docs = {0}, mover_docs = {0}, judge_docs = {0};
  Vocab strat_vocab = {0}, mover_vocab = {0}, judge_vocab = {0};

  Model *strategist =
      train_organelle("Strategist", STRATEGIST_CORPUS, STRATEGIST_CKPT,
                      &strat_docs, &strat_vocab, g_cfg.num_steps);
  if (!strategist) {
    fprintf(stderr, "FATAL: Strategist training failed\n");
    return 1;
  }

  Model *mover = train_organelle("Mover", MOVER_CORPUS, MOVER_CKPT, &mover_docs,
                                 &mover_vocab, g_cfg.num_steps);
  if (!mover) {
    fprintf(stderr, "FATAL: Mover training failed\n");
    return 1;
  }

  Model *judge_model =
      train_organelle("Judge", JUDGE_CORPUS, JUDGE_CKPT, &judge_docs,
                      &judge_vocab, g_cfg.num_steps);
  if (!judge_model) {
    fprintf(stderr, "FATAL: Judge training failed\n");
    return 1;
  }

  /* ================================================================
   * PHASE 2: Pipeline — Solve Stratified Test Puzzles
   * ================================================================ */

  printf("\n--- PHASE 2: OPA PIPELINE (displacement-based) ---\n");
  printf("Solving %d test puzzles (easy/medium/hard)...\n\n", NUM_TEST_PUZZLES);

  /* Per-band tracking */
  const char *band_names[] = {"EASY", "MEDIUM", "HARD"};
  int band_counts[] = {NUM_EASY_PUZZLES, NUM_MEDIUM_PUZZLES, NUM_HARD_PUZZLES};
  int band_md_min[] = {1, 5, 9};
  int band_md_max[] = {4, 8, 20};
  int band_solved[] = {0, 0, 0};
  int band_moves[] = {0, 0, 0};

  int total_solved = 0;
  int total_moves = 0;
  int total_accepts = 0;
  int total_rejects = 0;
  int total_parse_errors = 0;

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  /* Use a separate seed for test puzzles — no overlap with training */
  unsigned int puzzle_seed = 99999;
  int puzzle_global_idx = 0;

  for (int band = 0; band < 3; band++) {
    printf("=== %s band (md %d-%d) ===\n", band_names[band], band_md_min[band],
           band_md_max[band]);

    for (int pi = 0; pi < band_counts[band]; pi++) {
      puzzle_global_idx++;
      int board[9];
      scramble_to_target_md(board, band_md_min[band], band_md_max[band],
                            &puzzle_seed);

      char board_str[16];
      board_to_str(board, board_str);
      char dstr[64];
      md_delta_str(board, dstr, sizeof(dstr));
      int initial_md = manhattan_distance(board);

      printf("-- Puzzle %d/%d [%s] --\n", puzzle_global_idx, NUM_TEST_PUZZLES,
             band_names[band]);
      printf("   Board: %s  Disp: %s  (md=%d)\n", board_str, dstr, initial_md);

      /* Step 1: Ask Strategist for priority tile */
      char strat_prompt[128];
      snprintf(strat_prompt, sizeof(strat_prompt), "%s|md=%d", dstr,
               initial_md);

      char strat_output[INF_GEN_LEN + 1];
      organelle_generate(strategist, &strat_vocab, strat_prompt, strat_output,
                         INF_GEN_LEN);
      printf("   Strategist -> \"%s\"\n", strat_output);

      /* Step 2: Pipeline loop — Mover + Judge */
      blocked_init();
      int moves_made = 0;
      int solved = 0;
      int stalls = 0;
      int last_best_md = initial_md;

      for (int iter = 0; iter < MAX_PIPELINE_ITERS && !solved; iter++) {
        md_delta_str(board, dstr, sizeof(dstr));
        int blank = find_blank(board);
        int md_before = manhattan_distance(board);

        /* Clear blocked if stalled too long */
        if (stalls >= REPLAN_THRESHOLD) {
          blocked_clear();
          stalls = 0;
        }

        /* Build Mover prompt with displacement encoding */
        char mover_prompt[128];
        if (g_blocked[0] != '\0') {
          snprintf(mover_prompt, sizeof(mover_prompt), "%s|b=%d|x=%s", dstr,
                   blank, g_blocked);
        } else {
          snprintf(mover_prompt, sizeof(mover_prompt), "%s|b=%d", dstr, blank);
        }

        char move_output[INF_GEN_LEN + 1];
        organelle_generate(mover, &mover_vocab, mover_prompt, move_output,
                           INF_GEN_LEN);

        /* Parse direction */
        const char *dir = NULL;
        if (strncmp(move_output, "up", 2) == 0)
          dir = "up";
        else if (strncmp(move_output, "down", 4) == 0)
          dir = "down";
        else if (strncmp(move_output, "left", 4) == 0)
          dir = "left";
        else if (strncmp(move_output, "right", 5) == 0)
          dir = "right";

        if (!dir) {
          printf("   [%d] Mover -> \"%s\" (PARSE ERROR)\n", iter + 1,
                 move_output);
          total_parse_errors++;
          /* Fallback: try first legal move not blocked */
          const char *fallback_dirs[] = {"up", "down", "left", "right"};
          for (int d = 0; d < 4; d++) {
            if (strstr(g_blocked, fallback_dirs[d]) != NULL)
              continue;
            int test[9];
            memcpy(test, board, sizeof(board));
            if (apply_move(test, fallback_dirs[d])) {
              dir = fallback_dirs[d];
              break;
            }
          }
          if (!dir) {
            stalls++;
            continue;
          }
        }

        /* Deterministic Judge: try the move */
        int test_board[9];
        memcpy(test_board, board, sizeof(board));
        int is_valid = apply_move(test_board, dir);

        if (is_valid) {
          memcpy(board, test_board, sizeof(board));
          moves_made++;
          int md_after = manhattan_distance(board);
          char arrow =
              md_after < md_before ? '+' : (md_after > md_before ? '-' : '=');

          blocked_clear();
          if (md_after < last_best_md) {
            stalls = 0;
            last_best_md = md_after;
          } else {
            stalls++;
          }
          total_accepts++;

          printf("   [%d] move=%s %c (md: %d->%d)\n", iter + 1, dir, arrow,
                 md_before, md_after);
        } else {
          blocked_add(dir);
          stalls++;
          total_rejects++;
          printf("   [%d] move=%s -> OUT OF BOUNDS [blocked:%s]\n", iter + 1,
                 dir, g_blocked);
        }

        if (is_goal(board)) {
          solved = 1;
        }
      }

      board_to_str(board, board_str);
      int final_md = manhattan_distance(board);
      total_moves += moves_made;

      if (solved) {
        total_solved++;
        band_solved[band]++;
        band_moves[band] += moves_made;
        printf("   SOLVED in %d moves!\n\n", moves_made);
      } else {
        printf("   NOT SOLVED: board=%s md=%d (was %d) moves=%d\n\n", board_str,
               final_md, initial_md, moves_made);
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &pipeline_end);
  double pipeline_time =
      (double)(pipeline_end.tv_sec - pipeline_start.tv_sec) +
      (double)(pipeline_end.tv_nsec - pipeline_start.tv_nsec) / 1e9;

  /* ================================================================
   * PHASE 3: Results — Generalisation Report
   * ================================================================ */

  printf("================================================================\n");
  printf("--- GENERALISATION RESULTS ---\n");
  printf(
      "================================================================\n\n");

  printf("Overall: %d / %d solved (%.0f%%)\n\n", total_solved, NUM_TEST_PUZZLES,
         NUM_TEST_PUZZLES > 0 ? 100.0 * total_solved / NUM_TEST_PUZZLES : 0.0);

  printf("%-8s  %-10s  %-10s  %-10s\n", "Band", "Solved", "Rate", "Avg Moves");
  printf("%-8s  %-10s  %-10s  %-10s\n", "--------", "----------", "----------",
         "----------");
  for (int b = 0; b < 3; b++) {
    double rate =
        band_counts[b] > 0 ? 100.0 * band_solved[b] / band_counts[b] : 0.0;
    double avg_moves =
        band_solved[b] > 0 ? (double)band_moves[b] / band_solved[b] : 0.0;
    printf("%-8s  %d/%-8d  %5.0f%%       %.1f\n", band_names[b], band_solved[b],
           band_counts[b], rate, avg_moves);
  }

  printf("\nTotal moves:      %d\n", total_moves);
  printf("Valid moves:      %d\n", total_accepts);
  printf("OOB rejections:   %d\n", total_rejects);
  printf("Parse errors:     %d\n", total_parse_errors);
  printf("Pipeline time:    %.2fs\n", pipeline_time);
  printf("================================================================\n");

  /* Cleanup */
  model_free(strategist);
  model_free(mover);
  model_free(judge_model);
  free_docs(&strat_docs);
  free_docs(&mover_docs);
  free_docs(&judge_docs);

  return 0;
}
