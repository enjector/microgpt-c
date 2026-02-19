/*
 * MicroGPT-C — Connect-4 Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner on Connect-4:
 *   - 7 columns × 6 rows board
 *   - Two neural organelles (Planner + Player) coordinate via pipe-separated
 *     flat strings with kanban state, playing X against a random opponent O.
 *   - Judge is fully deterministic (column-not-full + win/draw check).
 *
 * Architecture: same as tic-tac-toe (n_embd=48, n_layer=2, ~64K params).
 *
 * Pipeline: Planner -> Player -> Judge(deterministic) -> Opponent(random)
 *
 * Build:
 *   cmake --build build --target connect4_demo
 *   ./build/connect4_demo
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
#define PLANNER_CORPUS "connect4_planner.txt"
#define PLAYER_CORPUS "connect4_player.txt"

#define PLANNER_CKPT "connect4_planner.ckpt"
#define PLAYER_CKPT "connect4_player.ckpt"

#define ORGANELLE_TEMP 0.2 /* low temperature for reliable retrieval */
#define INF_GEN_LEN 60     /* max chars per organelle generation */

#define NUM_TEST_GAMES 100 /* games to play against random */
#define REPLAN_THRESHOLD 3 /* stalls before re-invoking Planner */
#define MAX_LAST_HISTORY 3 /* keep last N moves in history */

/* ---- Board Constants ---- */
#define BOARD_ROWS 6
#define BOARD_COLS 7
#define BOARD_SIZE (BOARD_ROWS * BOARD_COLS) /* 42 */
#define EMPTY_CELL '.'
#define PLAYER_X 'X'
#define PLAYER_O 'O'

/* ---- File-scoped runtime config (shared by helper functions) ---- */
static MicrogptConfig g_cfg;

/* ---- Board Helpers ---- */

static int cell_idx(int r, int c) { return r * BOARD_COLS + c; }

static void board_to_str(const char *board, char *out) {
  memcpy(out, board, BOARD_SIZE);
  out[BOARD_SIZE] = '\0';
}

static int get_valid_columns(const char *board, int *columns) {
  /* A column is valid if the top row cell is empty */
  int count = 0;
  for (int c = 0; c < BOARD_COLS; c++) {
    if (board[cell_idx(0, c)] == EMPTY_CELL) {
      columns[count++] = c;
    }
  }
  return count;
}

static int drop_piece(char *board, int col, char player) {
  /* Drop piece into column. Returns the row it landed on, or -1 if full. */
  if (col < 0 || col >= BOARD_COLS)
    return -1;
  for (int r = BOARD_ROWS - 1; r >= 0; r--) {
    if (board[cell_idx(r, col)] == EMPTY_CELL) {
      board[cell_idx(r, col)] = player;
      return r;
    }
  }
  return -1; /* column full */
}

static int count_pieces(const char *board) {
  int count = 0;
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (board[i] != EMPTY_CELL)
      count++;
  }
  return count;
}

/* Win-checking directions: horizontal, vertical, diagonal-down, diagonal-up */
static const int WIN_DR[4] = {0, 1, 1, 1};
static const int WIN_DC[4] = {1, 0, 1, -1};

static char check_winner(const char *board) {
  /* Returns 'X', 'O', or '.' (no winner) */
  for (int r = 0; r < BOARD_ROWS; r++) {
    for (int c = 0; c < BOARD_COLS; c++) {
      if (board[cell_idx(r, c)] == EMPTY_CELL)
        continue;
      char player = board[cell_idx(r, c)];
      for (int d = 0; d < 4; d++) {
        int er = r + 3 * WIN_DR[d];
        int ec = c + 3 * WIN_DC[d];
        if (er < 0 || er >= BOARD_ROWS || ec < 0 || ec >= BOARD_COLS)
          continue;
        int match = 1;
        for (int i = 1; i < 4; i++) {
          if (board[cell_idx(r + i * WIN_DR[d], c + i * WIN_DC[d])] != player) {
            match = 0;
            break;
          }
        }
        if (match)
          return player;
      }
    }
  }
  return EMPTY_CELL;
}

static int is_draw(const char *board) {
  return check_winner(board) == EMPTY_CELL && count_pieces(board) == BOARD_SIZE;
}

static void print_board(const char *board) {
  printf("  0 1 2 3 4 5 6\n");
  for (int r = 0; r < BOARD_ROWS; r++) {
    printf("  ");
    for (int c = 0; c < BOARD_COLS; c++) {
      printf("%c ", board[cell_idx(r, c)]);
    }
    printf("\n");
  }
}

/* ---- Kanban State ---- */

typedef struct {
  char blocked[64]; /* comma-separated blocked columns */
  char last[64];    /* comma-separated last N columns */
  int stalls;       /* consecutive failures without progress */
  int replans;      /* number of times Planner was re-invoked */
} Kanban;

static void kanban_init(Kanban *kb) {
  kb->blocked[0] = '\0';
  kb->last[0] = '\0';
  kb->stalls = 0;
  kb->replans = 0;
}

static void kanban_add_blocked(Kanban *kb, int col) {
  char buf[8];
  snprintf(buf, sizeof(buf), "%d", col);
  if (kb->blocked[0] != '\0') {
    if (strstr(kb->blocked, buf) != NULL)
      return;
    size_t len = strlen(kb->blocked);
    if (len + strlen(buf) + 2 < sizeof(kb->blocked)) {
      kb->blocked[len] = ',';
      strcpy(kb->blocked + len + 1, buf);
    }
  } else {
    strncpy(kb->blocked, buf, sizeof(kb->blocked) - 1);
  }
}

static void kanban_clear_blocked(Kanban *kb) { kb->blocked[0] = '\0'; }

static void kanban_add_last(Kanban *kb, int col) {
  char buf[8];
  snprintf(buf, sizeof(buf), "%d", col);

  /* Count existing entries */
  int count = 0;
  if (kb->last[0] != '\0') {
    count = 1;
    for (const char *p = kb->last; *p; p++)
      if (*p == ',')
        count++;
  }

  if (count >= MAX_LAST_HISTORY) {
    char *comma = strchr(kb->last, ',');
    if (comma) {
      memmove(kb->last, comma + 1, strlen(comma + 1) + 1);
    } else {
      kb->last[0] = '\0';
    }
  }

  if (kb->last[0] == '\0') {
    strncpy(kb->last, buf, sizeof(kb->last) - 1);
  } else {
    size_t len = strlen(kb->last);
    if (len + strlen(buf) + 2 < sizeof(kb->last)) {
      kb->last[len] = ',';
      strcpy(kb->last + len + 1, buf);
    }
  }
}

/* ---- Pipe-String Helpers ---- */

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

  int nthreads = 1; /* single-threaded for stability */

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
    doc_cursor = cursor % (int)docs->num_docs;

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

  return model;
}

/* ---- Random Opponent ---- */

static int random_opponent_move(const char *board, unsigned int *seed) {
  int cols[BOARD_COLS];
  int count = get_valid_columns(board, cols);
  if (count == 0)
    return -1;
  return cols[rand_r(seed) % count];
}

/* ---- Main ---- */

int main(void) {
  setbuf(stdout, NULL); /* unbuffered for real-time progress */
  seed_rng(42);

  /* Runtime configuration — same as tictactoe/puzzle8 organelles */
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
  microgpt_print_config("MicroGPT-C - Connect-4 Kanban Pipeline Demo", &g_cfg);

  /* Suppress unused function warnings */
  (void)print_board;

  /* ================================================================
   * PHASE 1: Train organelles
   * ================================================================ */

  int train_steps = g_cfg.num_steps;
  printf("--- PHASE 1: TRAINING (%d steps each) ---\n", train_steps);

  Docs planner_docs = {0}, player_docs = {0};
  Vocab planner_vocab = {0}, player_vocab = {0};

  Model *planner = train_organelle("Planner", PLANNER_CORPUS, PLANNER_CKPT,
                                   &planner_docs, &planner_vocab, train_steps);
  if (!planner) {
    fprintf(stderr, "FATAL: Planner training failed\n");
    return 1;
  }

  Model *player = train_organelle("Player", PLAYER_CORPUS, PLAYER_CKPT,
                                  &player_docs, &player_vocab, train_steps);
  if (!player) {
    fprintf(stderr, "FATAL: Player training failed\n");
    return 1;
  }

  /* ================================================================
   * PHASE 2: Pipeline — Play Games vs Random Opponent
   * ================================================================ */

  printf("\n--- PHASE 2: KANBAN PIPELINE EXECUTION ---\n");
  printf("Playing %d games as X against random opponent O...\n\n",
         NUM_TEST_GAMES);

  int total_wins = 0;
  int total_draws = 0;
  int total_losses = 0;
  int total_moves = 0;
  int total_valid_moves = 0;
  int total_invalid_moves = 0;
  int total_parse_errors = 0;
  int total_replans = 0;

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  unsigned int game_seed = 12345;

  for (int game_idx = 0; game_idx < NUM_TEST_GAMES; game_idx++) {
    /* Initialize empty board */
    char board[BOARD_SIZE + 1];
    memset(board, EMPTY_CELL, BOARD_SIZE);
    board[BOARD_SIZE] = '\0';

    Kanban kb;
    kanban_init(&kb);

    char board_str[BOARD_SIZE + 2];
    board_to_str(board, board_str);

    int empties = BOARD_SIZE - count_pieces(board);

    /* Step 1: Ask Planner for initial plan */
    char planner_prompt[256];
    snprintf(planner_prompt, sizeof(planner_prompt), "board=%s|empties=%d",
             board_str, empties);

    char plan_output[INF_GEN_LEN + 1];
    organelle_generate(planner, &planner_vocab, planner_prompt, plan_output,
                       INF_GEN_LEN);

    if (!pipe_starts_with(plan_output, "todo=")) {
      total_parse_errors++;
    }

    /* Step 2: Play the game */
    int moves_made = 0;
    char result = EMPTY_CELL; /* 'X' win, 'O' loss, '.' ongoing */
    int game_draw = 0;
    int max_retries_per_turn = 7; /* at most 7 columns */

    if (game_idx < 15 || (game_idx + 1) % 10 == 0) {
      printf("-- Game %d/%d --\n", game_idx + 1, NUM_TEST_GAMES);
    }

    while (result == EMPTY_CELL && !game_draw) {
      board_to_str(board, board_str);
      empties = BOARD_SIZE - count_pieces(board);

      int valid_cols[BOARD_COLS];
      int num_valid = get_valid_columns(board, valid_cols);

      if (num_valid == 0) {
        game_draw = 1;
        break;
      }

      /* Re-plan if stalled */
      if (kb.stalls >= REPLAN_THRESHOLD && kb.replans < 3) {
        kb.replans++;
        total_replans++;

        char replan_prompt[256];
        snprintf(replan_prompt, sizeof(replan_prompt),
                 "board=%s|empties=%d|stalled", board_str, empties);

        char replan_output[INF_GEN_LEN + 1];
        organelle_generate(planner, &planner_vocab, replan_prompt,
                           replan_output, INF_GEN_LEN);

        kanban_clear_blocked(&kb);
        kb.stalls = 0;
      }

      /* Build Player prompt */
      char player_prompt[256];
      if (kb.blocked[0] != '\0') {
        snprintf(player_prompt, sizeof(player_prompt), "board=%s|blocked=%s",
                 board_str, kb.blocked);
      } else {
        snprintf(player_prompt, sizeof(player_prompt), "board=%s", board_str);
      }

      /* Generate move */
      char move_output[INF_GEN_LEN + 1];
      organelle_generate(player, &player_vocab, player_prompt, move_output,
                         INF_GEN_LEN);

      /* Parse column — output should be just "0"-"6" */
      int proposed_col = -1;
      if (move_output[0] >= '0' && move_output[0] <= '6') {
        proposed_col = move_output[0] - '0';
      }

      if (proposed_col < 0) {
        total_parse_errors++;
        /* Fallback: pick first valid column not in blocked */
        for (int i = 0; i < num_valid; i++) {
          char col_str[4];
          snprintf(col_str, sizeof(col_str), "%d", valid_cols[i]);
          if (strstr(kb.blocked, col_str) == NULL) {
            proposed_col = valid_cols[i];
            break;
          }
        }
        if (proposed_col < 0 && num_valid > 0)
          proposed_col = valid_cols[0]; /* desperate fallback */
        if (proposed_col < 0)
          break;
      }

      /* Deterministic Judge: is column valid (not full)? */
      int row = drop_piece(board, proposed_col, PLAYER_X);
      if (row >= 0) {
        moves_made++;
        total_valid_moves++;

        kanban_add_last(&kb, proposed_col);
        kanban_clear_blocked(&kb);
        kb.stalls = 0;

        /* Check win */
        result = check_winner(board);
        if (result == PLAYER_X) {
          if (game_idx < 15 || (game_idx + 1) % 10 == 0) {
            printf("   X wins in %d moves!\n", moves_made);
          }
          break;
        }

        /* Check draw */
        if (is_draw(board)) {
          game_draw = 1;
          if (game_idx < 15 || (game_idx + 1) % 10 == 0) {
            printf("   Draw after %d moves\n", moves_made);
          }
          break;
        }

        /* Opponent's turn (random) */
        int opp_col = random_opponent_move(board, &game_seed);
        if (opp_col >= 0) {
          drop_piece(board, opp_col, PLAYER_O);
          moves_made++;

          /* Check if opponent won */
          result = check_winner(board);
          if (result == PLAYER_O) {
            if (game_idx < 15 || (game_idx + 1) % 10 == 0) {
              printf("   O wins after %d moves (loss)\n", moves_made);
            }
            break;
          }

          if (is_draw(board)) {
            game_draw = 1;
            if (game_idx < 15 || (game_idx + 1) % 10 == 0) {
              printf("   Draw after %d moves\n", moves_made);
            }
            break;
          }
        }
      } else {
        /* Invalid move — column full, add to blocked */
        kanban_add_blocked(&kb, proposed_col);
        kb.stalls++;
        total_invalid_moves++;

        /* If too many retries on this turn, force a random valid column */
        if (kb.stalls >= max_retries_per_turn) {
          int fall_col = random_opponent_move(board, &game_seed);
          if (fall_col >= 0) {
            drop_piece(board, fall_col, PLAYER_X);
            moves_made++;
            total_valid_moves++;
            kanban_clear_blocked(&kb);
            kb.stalls = 0;

            result = check_winner(board);
            if (result == PLAYER_X)
              break;
            if (is_draw(board)) {
              game_draw = 1;
              break;
            }

            /* Opponent move */
            int opp_col = random_opponent_move(board, &game_seed);
            if (opp_col >= 0) {
              drop_piece(board, opp_col, PLAYER_O);
              moves_made++;
              result = check_winner(board);
              if (result == PLAYER_O)
                break;
              if (is_draw(board)) {
                game_draw = 1;
                break;
              }
            }
          } else {
            break; /* no valid moves left */
          }
        }
      }
    }

    total_moves += moves_made;

    if (result == PLAYER_X) {
      total_wins++;
    } else if (result == PLAYER_O) {
      total_losses++;
    } else {
      total_draws++;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &pipeline_end);
  double pipeline_time =
      (double)(pipeline_end.tv_sec - pipeline_start.tv_sec) +
      (double)(pipeline_end.tv_nsec - pipeline_start.tv_nsec) / 1e9;

  /* ================================================================
   * PHASE 3: Results Summary
   * ================================================================ */

  printf(
      "\n================================================================\n");
  printf("                    CONNECT-4 RESULTS\n");
  printf("================================================================\n");
  printf("Games won (X):      %d / %d (%.0f%%)\n", total_wins, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_wins / NUM_TEST_GAMES : 0.0);
  printf("Games drawn:        %d / %d (%.0f%%)\n", total_draws, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_draws / NUM_TEST_GAMES : 0.0);
  printf("Games lost (O won): %d / %d (%.0f%%)\n", total_losses, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_losses / NUM_TEST_GAMES : 0.0);
  printf("Win+Draw rate:      %.0f%%\n",
         NUM_TEST_GAMES > 0
             ? 100.0 * (total_wins + total_draws) / NUM_TEST_GAMES
             : 0.0);
  printf("Total moves:        %d (avg %.1f per game)\n", total_moves,
         NUM_TEST_GAMES > 0 ? (double)total_moves / NUM_TEST_GAMES : 0.0);
  printf("Valid moves:        %d\n", total_valid_moves);
  printf("Invalid moves:      %d\n", total_invalid_moves);
  printf("Parse errors:       %d\n", total_parse_errors);
  printf("Planner re-plans:   %d\n", total_replans);
  printf("Pipeline time:      %.2fs\n", pipeline_time);
  printf("================================================================\n");

  /* Cleanup */
  model_free(planner);
  model_free(player);
  free_docs(&planner_docs);
  free_docs(&player_docs);

  return 0;
}
