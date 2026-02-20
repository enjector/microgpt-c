/*
 * MicroGPT-C — Tic-Tac-Toe Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner on an ADVERSARIAL game.
 * Two neural organelles (Planner + Player) coordinate via pipe-separated
 * flat strings with kanban state, playing X against a random opponent O.
 * Judge is fully deterministic (cell-empty + win/draw check).
 *
 * Architecture: same as 8-puzzle v2 (n_embd=48, n_layer=2, ~64K params).
 *
 * Pipeline: Planner -> Player -> Judge(deterministic) -> Opponent -> repeat
 *
 * Build:
 *   cmake --build build --target tictactoe_demo
 *   ./build/tictactoe_demo
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
#define PLANNER_CORPUS "tictactoe_planner.txt"
#define PLAYER_CORPUS "tictactoe_player.txt"

#define PLANNER_CKPT "tictactoe_planner.ckpt"
#define PLAYER_CKPT "tictactoe_player.ckpt"

#define ORGANELLE_TEMP 0.2 /* low temperature for reliable retrieval */
#define INF_GEN_LEN 60     /* max chars per organelle generation */

#define NUM_TEST_GAMES 100 /* games to play against random */
#define REPLAN_THRESHOLD 3 /* stalls before re-invoking Planner */
#define MAX_LAST_HISTORY 3 /* keep last N moves in history */
#define ENSEMBLE_VOTES 3   /* worker votes per move (odd for tiebreak) */

/* ---- Board Constants ---- */
#define EMPTY '_'
#define PLAYER_X 'X'
#define PLAYER_O 'O'

/* ---- File-scoped runtime config ---- */
static MicrogptConfig g_cfg;

/* ---- Board Helpers ---- */

static void board_to_str(const char *board, char *out) {
  memcpy(out, board, 9);
  out[9] = '\0';
}

static int get_empties(const char *board, int *positions) {
  int count = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] == EMPTY) {
      positions[count++] = i;
    }
  }
  return count;
}

static char check_winner(const char *board) {
  /* Returns 'X', 'O', or '_' (no winner) */
  static const int lines[8][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, /* rows */
      {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, /* cols */
      {0, 4, 8}, {2, 4, 6}             /* diags */
  };
  for (int i = 0; i < 8; i++) {
    int a = lines[i][0], b = lines[i][1], c = lines[i][2];
    if (board[a] != EMPTY && board[a] == board[b] && board[b] == board[c]) {
      return board[a];
    }
  }
  return EMPTY;
}

static int is_draw(const char *board) {
  for (int i = 0; i < 9; i++) {
    if (board[i] == EMPTY)
      return 0;
  }
  return check_winner(board) == EMPTY;
}

static int apply_move_ttt(char *board, int pos, char player) {
  if (pos < 0 || pos > 8)
    return 0;
  if (board[pos] != EMPTY)
    return 0;
  board[pos] = player;
  return 1;
}

/* ---- Random Opponent ---- */

static int random_opponent_move(const char *board, unsigned int *seed) {
  int empties[9];
  int count = get_empties(board, empties);
  if (count == 0)
    return -1;
  return empties[rand_r(seed) % count];
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
  microgpt_print_config("MicroGPT-C - Tic-Tac-Toe Kanban Pipeline Demo",
                        &g_cfg);

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

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  unsigned int game_seed = 12345;

  for (int game_idx = 0; game_idx < NUM_TEST_GAMES; game_idx++) {
    /* Initialize empty board */
    char board[10];
    memset(board, EMPTY, 9);
    board[9] = '\0';

    OpaKanban kb;
    opa_kanban_init(&kb, MAX_LAST_HISTORY);

    char board_str[16];
    board_to_str(board, board_str);

    int empties_arr[9];
    int num_empties = get_empties(board, empties_arr);

    /* Step 1: Ask Planner for initial plan */
    char planner_prompt[128];
    snprintf(planner_prompt, sizeof(planner_prompt), "board=%s|empties=%d",
             board_str, num_empties);

    char plan_output[INF_GEN_LEN + 1];
    organelle_generate(planner, &g_cfg, planner_prompt, plan_output,
                       INF_GEN_LEN, ORGANELLE_TEMP);

    if (!opa_pipe_starts_with(plan_output, "todo=")) {
      total_parse_errors++;
    }

    /* Step 2: Play the game */
    int moves_made = 0;
    char result = EMPTY;
    int game_draw = 0;

    printf("-- Game %d/%d --\n", game_idx + 1, NUM_TEST_GAMES);

    while (result == EMPTY && !game_draw) {
      board_to_str(board, board_str);
      num_empties = get_empties(board, empties_arr);

      if (num_empties == 0) {
        game_draw = 1;
        break;
      }

      /* Re-plan if stalled */
      if (kb.stalls >= REPLAN_THRESHOLD && kb.replans < 3) {
        kb.replans++;
        total_replans++;

        char replan_prompt[128];
        snprintf(replan_prompt, sizeof(replan_prompt),
                 "board=%s|empties=%d|stalled", board_str, num_empties);

        char replan_output[INF_GEN_LEN + 1];
        organelle_generate(planner, &g_cfg, replan_prompt, replan_output,
                           INF_GEN_LEN, ORGANELLE_TEMP);

        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;
      }

      /* Build valid-move string from empties */
      char valid_str[32] = "";
      int vs_pos = 0;
      for (int i = 0; i < num_empties; i++) {
        if (i > 0)
          vs_pos += snprintf(valid_str + vs_pos,
                             sizeof(valid_str) - (size_t)vs_pos, ",");
        vs_pos +=
            snprintf(valid_str + vs_pos, sizeof(valid_str) - (size_t)vs_pos,
                     "%d", empties_arr[i]);
      }

      /* Build Player prompt with valid= field */
      char player_prompt[128];
      if (kb.blocked[0] != '\0') {
        snprintf(player_prompt, sizeof(player_prompt),
                 "board=%s|valid=%s|blocked=%s", board_str, valid_str,
                 kb.blocked);
      } else {
        snprintf(player_prompt, sizeof(player_prompt), "board=%s|valid=%s",
                 board_str, valid_str);
      }

      /* Generate move via ensemble voting */
      char move_output[INF_GEN_LEN + 1];
      scalar_t vote_conf = 0;
      organelle_generate_ensemble(player, &g_cfg, player_prompt, move_output,
                                  INF_GEN_LEN, ENSEMBLE_VOTES, ORGANELLE_TEMP,
                                  &vote_conf);

      /* Parse position — output should be just "0"-"8" */
      int proposed_pos = -1;
      if (move_output[0] >= '0' && move_output[0] <= '8') {
        proposed_pos = move_output[0] - '0';
      }

      /* Validate against valid list */
      if (proposed_pos >= 0) {
        char pos_str[4];
        snprintf(pos_str, sizeof(pos_str), "%d", proposed_pos);
        if (!opa_valid_filter(pos_str, valid_str)) {
          proposed_pos = -1;
        }
      }

      if (proposed_pos < 0) {
        total_parse_errors++;
        /* Use opa_valid_fallback to pick first valid non-blocked position */
        char fb[16];
        if (opa_valid_fallback(&kb, valid_str, fb, sizeof(fb))) {
          proposed_pos = fb[0] - '0';
        } else if (num_empties > 0) {
          proposed_pos = empties_arr[0];
        }
        if (proposed_pos < 0)
          break;
      }

      /* Deterministic Judge: is cell empty? */
      if (apply_move_ttt(board, proposed_pos, PLAYER_X)) {
        moves_made++;
        total_valid_moves++;

        char pos_str[4];
        snprintf(pos_str, sizeof(pos_str), "%d", proposed_pos);
        opa_kanban_add_last(&kb, pos_str);
        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;

        /* Check win */
        result = check_winner(board);
        if (result == PLAYER_X) {
          printf("   X wins in %d moves!\n", moves_made);
          break;
        }

        /* Check draw */
        if (is_draw(board)) {
          game_draw = 1;
          printf("   Draw after %d moves\n", moves_made);
          break;
        }

        /* Opponent's turn (random) */
        int opp_pos = random_opponent_move(board, &game_seed);
        if (opp_pos >= 0) {
          apply_move_ttt(board, opp_pos, PLAYER_O);
          moves_made++;

          result = check_winner(board);
          if (result == PLAYER_O) {
            printf("   O wins after %d moves (loss)\n", moves_made);
            break;
          }

          if (is_draw(board)) {
            game_draw = 1;
            printf("   Draw after %d moves\n", moves_made);
            break;
          }
        }
      } else {
        /* Invalid move — blocked */
        char pos_str[4];
        snprintf(pos_str, sizeof(pos_str), "%d", proposed_pos);
        opa_kanban_add_blocked(&kb, pos_str);
        kb.stalls++;
        total_invalid_moves++;
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

  printf("\n--- RESULTS ---\n");
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
  organelle_free(planner);
  organelle_free(player);

  return 0;
}
