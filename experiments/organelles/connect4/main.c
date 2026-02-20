/*
 * MicroGPT-C — Connect-4 Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner on Connect-4:
 *   - 7 columns × 6 rows board
 *   - Two neural organelles (Planner + Player) coordinate via pipe-separated
 *     flat strings with kanban state, playing X against a random opponent O.
 *   - Judge is fully deterministic (column valid + win/draw check).
 *
 * Architecture: same as tictactoe/puzzle8 (n_embd=48, n_layer=2, ~64K params).
 *
 * Pipeline: Planner -> Player -> Judge(deterministic) -> Opponent -> repeat
 *
 * Build:
 *   cmake --build build --target connect4_demo
 *   ./build/connect4_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

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
#define ENSEMBLE_VOTES 3   /* worker votes per move (odd for tiebreak) */

/* Intelligence verification baseline mode:
 *   0 = Trained model (default)
 *   1 = Random baseline (random valid move, pipeline still runs)
 */
#ifndef RANDOM_BASELINE
#define RANDOM_BASELINE 0
#endif

/* ---- Board Constants ---- */
#define BOARD_ROWS 6
#define BOARD_COLS 7
#define BOARD_SIZE (BOARD_ROWS * BOARD_COLS) /* 42 */
#define EMPTY_CELL '.'
#define PLAYER_X 'X'
#define PLAYER_O 'O'

/* ---- File-scoped runtime config ---- */
static MicrogptConfig g_cfg;

/* ---- Board Helpers ---- */

static int cell_idx(int r, int c) { return r * BOARD_COLS + c; }

static void board_to_str(const char *board, char *out) {
  memcpy(out, board, BOARD_SIZE);
  out[BOARD_SIZE] = '\0';
}

static int get_valid_columns(const char *board, int *columns) {
  int count = 0;
  for (int c = 0; c < BOARD_COLS; c++) {
    if (board[cell_idx(0, c)] == EMPTY_CELL) {
      columns[count++] = c;
    }
  }
  return count;
}

static int drop_piece(char *board, int col, char player) {
  if (col < 0 || col >= BOARD_COLS)
    return -1;
  for (int r = BOARD_ROWS - 1; r >= 0; r--) {
    if (board[cell_idx(r, col)] == EMPTY_CELL) {
      board[cell_idx(r, col)] = player;
      return r;
    }
  }
  return -1;
}

static int count_pieces(const char *board) {
  int count = 0;
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (board[i] != EMPTY_CELL)
      count++;
  }
  return count;
}

static const int WIN_DR[4] = {0, 1, 1, 1};
static const int WIN_DC[4] = {1, 0, 1, -1};

static char check_winner(const char *board) {
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
  setbuf(stdout, NULL);
  seed_rng(42);

  /* Runtime configuration */
  g_cfg = microgpt_default_config();
  g_cfg.n_embd = N_EMBD;
  g_cfg.n_head = N_HEAD;
  g_cfg.mlp_dim = MLP_DIM;
  g_cfg.n_layer = N_LAYER;
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

  Organelle *planner = organelle_train("Planner", PLANNER_CORPUS, PLANNER_CKPT,
                                       &g_cfg, train_steps);
  if (!planner) {
    fprintf(stderr, "FATAL: Planner training failed\n");
    return 1;
  }

  Organelle *player = organelle_train("Player", PLAYER_CORPUS, PLAYER_CKPT,
                                      &g_cfg, train_steps);
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
  int total_model_sourced = 0;
  int total_fallback_sourced = 0;

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  unsigned int game_seed = 12345;

  for (int game_idx = 0; game_idx < NUM_TEST_GAMES; game_idx++) {
    /* Initialize empty board */
    char board[BOARD_SIZE + 1];
    memset(board, EMPTY_CELL, BOARD_SIZE);
    board[BOARD_SIZE] = '\0';

    OpaKanban kb;
    opa_kanban_init(&kb, MAX_LAST_HISTORY);

    char board_str[BOARD_SIZE + 2];
    board_to_str(board, board_str);

    int empties = BOARD_SIZE - count_pieces(board);

    /* Step 1: Ask Planner for initial plan */
    char planner_prompt[256];
    snprintf(planner_prompt, sizeof(planner_prompt), "board=%s|empties=%d",
             board_str, empties);

    char plan_output[INF_GEN_LEN + 1];
    organelle_generate(planner, &g_cfg, planner_prompt, plan_output,
                       INF_GEN_LEN, ORGANELLE_TEMP);

    if (!opa_pipe_starts_with(plan_output, "todo=")) {
      total_parse_errors++;
    }

    /* Step 2: Play the game */
    int moves_made = 0;
    char result = EMPTY_CELL;
    int game_draw = 0;
    int max_retries_per_turn = 7;

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
        organelle_generate(planner, &g_cfg, replan_prompt, replan_output,
                           INF_GEN_LEN, ORGANELLE_TEMP);

        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;
      }

      /* Build valid-move string from get_valid_columns */
      char valid_str[32] = "";
      size_t vs_pos = 0;
      for (int i = 0; i < num_valid; i++) {
        if (i > 0 && vs_pos < sizeof(valid_str)) {
          int n = snprintf(valid_str + vs_pos, sizeof(valid_str) - vs_pos, ",");
          if (n > 0)
            vs_pos += (size_t)n;
        }
        if (vs_pos < sizeof(valid_str)) {
          int n = snprintf(valid_str + vs_pos, sizeof(valid_str) - vs_pos, "%d",
                           valid_cols[i]);
          if (n > 0)
            vs_pos += (size_t)n;
        }
      }

      /* Build Player prompt with valid= field */
      char player_prompt[256];
      if (kb.blocked[0] != '\0') {
        snprintf(player_prompt, sizeof(player_prompt),
                 "board=%s|valid=%s|blocked=%s", board_str, valid_str,
                 kb.blocked);
      } else {
        snprintf(player_prompt, sizeof(player_prompt), "board=%s|valid=%s",
                 board_str, valid_str);
      }

      int proposed_col = -1;
      int from_model = 0;

#if RANDOM_BASELINE == 1
      /* RANDOM BASELINE: pick random valid column */
      proposed_col = valid_cols[rand_r(&game_seed) % num_valid];
      from_model = 0;
#else
      /* Generate move via ensemble voting */
      char move_output[INF_GEN_LEN + 1];
      scalar_t vote_conf = 0;
      organelle_generate_ensemble(player, &g_cfg, player_prompt, move_output,
                                  INF_GEN_LEN, ENSEMBLE_VOTES, ORGANELLE_TEMP,
                                  &vote_conf);

      /* Parse column */
      if (move_output[0] >= '0' && move_output[0] <= '6') {
        proposed_col = move_output[0] - '0';
      }

      /* Validate against valid list — if not valid, use fallback */
      if (proposed_col >= 0) {
        char col_str[4];
        snprintf(col_str, sizeof(col_str), "%d", proposed_col);
        if (!opa_valid_filter(col_str, valid_str)) {
          proposed_col = -1;
        }
      }

      if (proposed_col >= 0) {
        from_model = 1;
      }

      if (proposed_col < 0) {
        total_parse_errors++;
        from_model = 0;
        /* Use opa_valid_fallback to pick first valid non-blocked column */
        char fb[16];
        if (opa_valid_fallback(&kb, valid_str, fb, sizeof(fb))) {
          proposed_col = fb[0] - '0';
        } else if (num_valid > 0) {
          proposed_col = valid_cols[0];
        }
        if (proposed_col < 0)
          break;
      }
#endif

      if (from_model)
        total_model_sourced++;
      else
        total_fallback_sourced++;

      /* Deterministic Judge: is column valid? */
      int row = drop_piece(board, proposed_col, PLAYER_X);
      if (row >= 0) {
        moves_made++;
        total_valid_moves++;

        char col_str[4];
        snprintf(col_str, sizeof(col_str), "%d", proposed_col);
        opa_kanban_add_last(&kb, col_str);
        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;

        result = check_winner(board);
        if (result == PLAYER_X) {
          if (game_idx < 15 || (game_idx + 1) % 10 == 0) {
            printf("   X wins in %d moves!\n", moves_made);
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

        int opp_col = random_opponent_move(board, &game_seed);
        if (opp_col >= 0) {
          drop_piece(board, opp_col, PLAYER_O);
          moves_made++;

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
        /* Invalid move — column full */
        char col_str[4];
        snprintf(col_str, sizeof(col_str), "%d", proposed_col);
        opa_kanban_add_blocked(&kb, col_str);
        kb.stalls++;
        total_invalid_moves++;

        if (kb.stalls >= max_retries_per_turn) {
          int fall_col = random_opponent_move(board, &game_seed);
          if (fall_col >= 0) {
            drop_piece(board, fall_col, PLAYER_X);
            moves_made++;
            total_valid_moves++;
            opa_kanban_clear_blocked(&kb);
            kb.stalls = 0;

            result = check_winner(board);
            if (result == PLAYER_X)
              break;
            if (is_draw(board)) {
              game_draw = 1;
              break;
            }

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
            break;
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

  const char *mode_names[] = {"TRAINED MODEL", "RANDOM BASELINE",
                              "UNTRAINED MODEL"};
  printf(
      "\n================================================================\n");
  printf("          CONNECT-4 RESULTS [%s]\n", mode_names[RANDOM_BASELINE]);
  printf("================================================================\n");
  printf("Mode:               %s\n", mode_names[RANDOM_BASELINE]);
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
  printf("Model-sourced:      %d / %d (%.0f%%)\n", total_model_sourced,
         total_model_sourced + total_fallback_sourced,
         (total_model_sourced + total_fallback_sourced) > 0
             ? 100.0 * total_model_sourced /
                   (total_model_sourced + total_fallback_sourced)
             : 0.0);
  printf("Fallback-sourced:   %d / %d (%.0f%%)\n", total_fallback_sourced,
         total_model_sourced + total_fallback_sourced,
         (total_model_sourced + total_fallback_sourced) > 0
             ? 100.0 * total_fallback_sourced /
                   (total_model_sourced + total_fallback_sourced)
             : 0.0);
  printf("Parse errors:       %d\n", total_parse_errors);
  printf("Planner re-plans:   %d\n", total_replans);
  printf("Pipeline time:      %.2fs\n", pipeline_time);
  printf("================================================================\n");

  /* Cleanup */
  organelle_free(planner);
  organelle_free(player);

  return 0;
}
