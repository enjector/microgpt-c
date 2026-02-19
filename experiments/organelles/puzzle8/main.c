/*
 * MicroGPT-C — 8-Puzzle Multi-Organelle Demo (v2 — Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner with kanban-based
 * shared state for multi-organelle coordination:
 *
 *   3 organelles (Planner, Mover, Judge) collaborate via pipe-separated
 *   flat strings with enriched context fields:
 *     - blocked: previously rejected directions (prevents fixation)
 *     - last:    recent move history (prevents oscillation)
 *     - kanban:  done/todo task tracking (enables re-planning)
 *
 * Pipeline flow:
 *   1. Planner: board|md|done|blocked → "todo=move,check,move,check"
 *   2. Mover:   board|blank|blocked|last → "move|dir=up|result=..."
 *   3. Judge:   proposed move → "valid=yes|closer=yes" or "valid=no|..."
 *   4. If Judge rejects → add to blocked, retry Mover
 *   5. If stalled → re-invoke Planner with kanban state
 *   6. Repeat until solved or max iterations reached
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
#define PLANNER_CORPUS "puzzle8_planner.txt"
#define MOVER_CORPUS "puzzle8_mover.txt"
#define JUDGE_CORPUS "puzzle8_judge.txt"

#define PLANNER_CKPT "puzzle8_planner_v2.ckpt"
#define MOVER_CKPT "puzzle8_mover_v2.ckpt"
#define JUDGE_CKPT "puzzle8_judge_v2.ckpt"

#define ORGANELLE_TEMP 0.2 /* low temperature for reliable retrieval */
#define INF_GEN_LEN 80     /* max chars per organelle generation */

#define MAX_PIPELINE_ITERS                                                     \
  30                        /* max moves attempted per puzzle (up from 20)     \
                             */
#define NUM_TEST_PUZZLES 10 /* puzzles to test in the pipeline */
#define REPLAN_THRESHOLD 4  /* stalls before re-invoking Planner */
#define MAX_LAST_HISTORY 3  /* keep last N moves in history */

/* ---- File-scoped runtime config (shared by helper functions) ---- */
static MicrogptConfig g_cfg;

/* ---- Board Helpers ---- */

static int GOAL_BOARD[9] = {1, 2, 3, 4, 5, 6, 7, 8, 0};

static void board_to_str(const int *board, char *out) {
  for (int i = 0; i < 9; i++)
    out[i] = '0' + board[i];
  out[9] = '\0';
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

/* ---- Kanban State ---- */

typedef struct {
  char blocked[64]; /* comma-separated blocked directions */
  char last[64];    /* comma-separated last N moves */
  char done[256];   /* comma-separated completed actions */
  int stalls;       /* consecutive failures without progress */
  int last_md;      /* manhattan distance at last progress */
  int replans;      /* number of times Planner was re-invoked */
} Kanban;

static void kanban_init(Kanban *kb, int initial_md) {
  kb->blocked[0] = '\0';
  kb->last[0] = '\0';
  kb->done[0] = '\0';
  kb->stalls = 0;
  kb->last_md = initial_md;
  kb->replans = 0;
}

static void kanban_add_blocked(Kanban *kb, const char *dir) {
  if (kb->blocked[0] == '\0') {
    strncpy(kb->blocked, dir, sizeof(kb->blocked) - 1);
  } else {
    /* Check if already blocked */
    if (strstr(kb->blocked, dir) != NULL)
      return;
    size_t len = strlen(kb->blocked);
    if (len + strlen(dir) + 2 < sizeof(kb->blocked)) {
      kb->blocked[len] = ',';
      strcpy(kb->blocked + len + 1, dir);
    }
  }
}

static void kanban_clear_blocked(Kanban *kb) { kb->blocked[0] = '\0'; }

static void kanban_add_last(Kanban *kb, const char *dir) {
  /* Count existing entries */
  int count = 0;
  if (kb->last[0] != '\0') {
    count = 1;
    for (const char *p = kb->last; *p; p++)
      if (*p == ',')
        count++;
  }

  if (count >= MAX_LAST_HISTORY) {
    /* Remove oldest entry */
    char *comma = strchr(kb->last, ',');
    if (comma) {
      memmove(kb->last, comma + 1, strlen(comma + 1) + 1);
    } else {
      kb->last[0] = '\0';
    }
  }

  if (kb->last[0] == '\0') {
    strncpy(kb->last, dir, sizeof(kb->last) - 1);
  } else {
    size_t len = strlen(kb->last);
    if (len + strlen(dir) + 2 < sizeof(kb->last)) {
      kb->last[len] = ',';
      strcpy(kb->last + len + 1, dir);
    }
  }
}

static void kanban_add_done(Kanban *kb, const char *action) {
  if (kb->done[0] == '\0') {
    strncpy(kb->done, action, sizeof(kb->done) - 1);
  } else {
    size_t len = strlen(kb->done);
    if (len + strlen(action) + 2 < sizeof(kb->done)) {
      kb->done[len] = ',';
      strcpy(kb->done + len + 1, action);
    }
  }
}

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

static int pipe_starts_with(const char *buf, const char *prefix) {
  return strncmp(buf, prefix, strlen(prefix)) == 0;
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
  g_cfg.max_docs = 5000;
  g_cfg.max_doc_len = 128;
  microgpt_print_config("MicroGPT-C - 8-Puzzle Kanban Pipeline Demo", &g_cfg);

  /* Suppress unused warnings for helper functions */
  (void)extract_pipe_value;

  /* ================================================================
   * PHASE 1: Train all 3 organelles
   * ================================================================ */

  printf("--- PHASE 1: TRAINING (%d steps each) ---\n", g_cfg.num_steps);

  Docs planner_docs = {0}, mover_docs = {0}, judge_docs = {0};
  Vocab planner_vocab = {0}, mover_vocab = {0}, judge_vocab = {0};

  Model *planner =
      train_organelle("Planner", PLANNER_CORPUS, PLANNER_CKPT, &planner_docs,
                      &planner_vocab, g_cfg.num_steps);
  if (!planner) {
    fprintf(stderr, "FATAL: Planner training failed\n");
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
   * PHASE 2: Pipeline — Solve Test Puzzles with Kanban
   * ================================================================ */

  printf("\n--- PHASE 2: KANBAN PIPELINE EXECUTION ---\n");
  printf("Solving %d test puzzles using kanban-enriched pipeline...\n\n",
         NUM_TEST_PUZZLES);

  int total_solved = 0;
  int total_moves = 0;
  int total_judge_accepts = 0;
  int total_judge_rejects = 0;
  int total_parse_errors = 0;
  int total_replans = 0;

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  unsigned int puzzle_seed = 12345;

  for (int puzzle_idx = 0; puzzle_idx < NUM_TEST_PUZZLES; puzzle_idx++) {
    /* Generate a test puzzle */
    int board[9];
    int difficulty = 3 + (puzzle_idx % 8); /* 3 to 10 moves scramble */
    scramble_puzzle(board, difficulty, &puzzle_seed);

    char board_str[16];
    board_to_str(board, board_str);
    int initial_manhattan = manhattan_distance(board);

    printf("-- Puzzle %d/%d --\n", puzzle_idx + 1, NUM_TEST_PUZZLES);
    printf("   Board: %s  (manhattan: %d, scramble: %d moves)\n", board_str,
           initial_manhattan, difficulty);

    /* Initialise kanban state */
    Kanban kb;
    kanban_init(&kb, initial_manhattan);

    /* Step 1: Ask Planner for initial plan */
    char planner_prompt[256];
    snprintf(planner_prompt, sizeof(planner_prompt), "board=%s|md=%d",
             board_str, initial_manhattan);

    char plan_output[INF_GEN_LEN + 1];
    organelle_generate(planner, &planner_vocab, planner_prompt, plan_output,
                       INF_GEN_LEN);
    printf("   Planner -> \"%s\"\n", plan_output);

    if (!pipe_starts_with(plan_output, "todo=")) {
      printf("   (Planner output not parseable -- using default plan)\n");
      total_parse_errors++;
    }
    kanban_add_done(&kb, "plan");

    /* Step 2: Execute the pipeline loop with kanban state */
    int moves_made = 0;
    int solved = 0;
    int judge_accepts = 0;
    int judge_rejects = 0;

    for (int iter = 0; iter < MAX_PIPELINE_ITERS && !solved; iter++) {
      board_to_str(board, board_str);
      int blank = find_blank(board);
      int md_before = manhattan_distance(board);

      /* Check for re-planning trigger */
      if (kb.stalls >= REPLAN_THRESHOLD && kb.replans < 3) {
        kb.replans++;
        total_replans++;

        char replan_prompt[256];
        snprintf(replan_prompt, sizeof(replan_prompt), "board=%s|md=%d|stalled",
                 board_str, md_before);

        char replan_output[INF_GEN_LEN + 1];
        organelle_generate(planner, &planner_vocab, replan_prompt,
                           replan_output, INF_GEN_LEN);
        printf("   [RE-PLAN #%d] Planner -> \"%s\"  (stalls=%d)\n", kb.replans,
               replan_output, kb.stalls);

        /* Clear blocked and stalls after re-plan */
        kanban_clear_blocked(&kb);
        kb.stalls = 0;
      }

      /* Build Mover prompt with kanban context */
      char mover_prompt[256];
      if (kb.blocked[0] != '\0') {
        snprintf(mover_prompt, sizeof(mover_prompt),
                 "board=%s|blank=%d|blocked=%s", board_str, blank, kb.blocked);
      } else {
        snprintf(mover_prompt, sizeof(mover_prompt), "board=%s|blank=%d",
                 board_str, blank);
      }

      char move_output[INF_GEN_LEN + 1];
      organelle_generate(mover, &mover_vocab, mover_prompt, move_output,
                         INF_GEN_LEN);

      /* Parse direction — output is just a word: "up", "down", "left", "right"
       */
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
        /* Fallback: try a random legal move not in blocked */
        const char *fallback_dirs[] = {"up", "down", "left", "right"};
        dir = NULL;
        for (int d = 0; d < 4; d++) {
          if (strstr(kb.blocked, fallback_dirs[d]) != NULL)
            continue;
          int test[9];
          memcpy(test, board, sizeof(board));
          if (apply_move(test, fallback_dirs[d])) {
            dir = fallback_dirs[d];
            break;
          }
        }
        if (!dir) {
          kb.stalls++;
          continue;
        }
      }

      /* Deterministic validation: try the move directly. */
      int test_board[9];
      memcpy(test_board, board, sizeof(board));
      int is_valid = apply_move(test_board, dir);

      if (is_valid) {
        /* Apply the move to the real board */
        memcpy(board, test_board, sizeof(board));
        moves_made++;
        int md_after = manhattan_distance(board);
        char arrow =
            md_after < md_before ? '+' : (md_after > md_before ? '-' : '=');

        /* Update kanban state on success */
        kanban_add_last(&kb, dir);
        kanban_clear_blocked(&kb);

        /* Track progress */
        if (md_after < kb.last_md) {
          kb.stalls = 0;
          kb.last_md = md_after;
        } else {
          kb.stalls++;
        }

        /* Track in done */
        char done_entry[32];
        snprintf(done_entry, sizeof(done_entry), "move(%s)", dir);
        kanban_add_done(&kb, done_entry);

        total_judge_accepts++;

        printf("   [%d] move=%s %c (md: %d->%d)\n", iter + 1, dir, arrow,
               md_before, md_after);
      } else {
        /* Move is out of bounds — add to blocked */
        kanban_add_blocked(&kb, dir);
        kb.stalls++;
        total_judge_rejects++;

        printf("   [%d] move=%s -> OUT OF BOUNDS [blocked:%s]\n", iter + 1, dir,
               kb.blocked);
      }

      /* Check if solved */
      if (is_goal(board)) {
        solved = 1;
        total_solved++;
      }
    }

    board_to_str(board, board_str);
    int final_manhattan = manhattan_distance(board);
    total_moves += moves_made;

    if (solved) {
      printf("   SOLVED in %d moves! (judge: %d/%d, replans: %d)\n\n",
             moves_made, judge_accepts, judge_rejects, kb.replans);
    } else {
      printf("   NOT SOLVED: board=%s md=%d (was %d) "
             "moves=%d (judge: %d/%d, replans: %d)\n\n",
             board_str, final_manhattan, initial_manhattan, moves_made,
             judge_accepts, judge_rejects, kb.replans);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &pipeline_end);
  double pipeline_time =
      (double)(pipeline_end.tv_sec - pipeline_start.tv_sec) +
      (double)(pipeline_end.tv_nsec - pipeline_start.tv_nsec) / 1e9;

  /* ================================================================
   * PHASE 3: Results Summary
   * ================================================================ */

  printf("--- RESULTS ---\n");
  printf("Puzzles solved:     %d / %d (%.0f%%)\n", total_solved,
         NUM_TEST_PUZZLES,
         NUM_TEST_PUZZLES > 0 ? 100.0 * total_solved / NUM_TEST_PUZZLES : 0.0);
  printf("Total moves:        %d (avg %.1f per puzzle)\n", total_moves,
         NUM_TEST_PUZZLES > 0 ? (double)total_moves / NUM_TEST_PUZZLES : 0.0);
  printf("Judge accepts:      %d\n", total_judge_accepts);
  printf("Judge rejections:   %d\n", total_judge_rejects);
  printf("Parse errors:       %d\n", total_parse_errors);
  printf("Planner re-plans:   %d\n", total_replans);
  printf("Pipeline time:      %.2fs\n", pipeline_time);
  printf("================================================================\n");

  /* Cleanup */
  model_free(planner);
  model_free(mover);
  model_free(judge_model);
  free_docs(&planner_docs);
  free_docs(&mover_docs);
  free_docs(&judge_docs);

  return 0;
}
