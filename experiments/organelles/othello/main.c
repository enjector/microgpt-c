/*
 * MicroGPT-C — Othello/Reversi 6x6 Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * 6x6 Othello with flipping mechanics. X (Black) vs random O (White).
 * Player output: "RrCc" e.g. "R2C3".
 * Judge is deterministic (validate flips, apply move, count pieces).
 *
 * Build:
 *   cmake --build build --target othello_demo
 *   ./build/othello_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PLANNER_CORPUS "othello_planner.txt"
#define PLAYER_CORPUS "othello_player.txt"
#define PLANNER_CKPT "othello_planner.ckpt"
#define PLAYER_CKPT "othello_player.ckpt"

#define ORGANELLE_TEMP 0.2
#define INF_GEN_LEN 60
#define NUM_TEST_GAMES 100
#define REPLAN_THRESHOLD 3
#define MAX_LAST_HISTORY 3
#define ENSEMBLE_VOTES 3

#define GRID 6
#define BOARD_SIZE (GRID * GRID)
#define EMPTY '.'
#define PLAYER_X 'X'
#define PLAYER_O 'O'

static MicrogptConfig g_cfg;

static int cell(int r, int c) { return r * GRID + c; }

static const int DIRS[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                               {0, 1},   {1, -1}, {1, 0},  {1, 1}};

static void board_to_str(const char *board, char *out) {
  memcpy(out, board, BOARD_SIZE);
  out[BOARD_SIZE] = '\0';
}

static void init_board(char *board) {
  memset(board, EMPTY, BOARD_SIZE);
  int mid = GRID / 2;
  board[cell(mid - 1, mid - 1)] = PLAYER_O;
  board[cell(mid - 1, mid)] = PLAYER_X;
  board[cell(mid, mid - 1)] = PLAYER_X;
  board[cell(mid, mid)] = PLAYER_O;
}

static int get_flips(const char *board, int r, int c, char player,
                     int flips[][2]) {
  char opp = (player == PLAYER_X) ? PLAYER_O : PLAYER_X;
  int total = 0;
  for (int d = 0; d < 8; d++) {
    int dr = DIRS[d][0], dc = DIRS[d][1];
    int tmp[BOARD_SIZE][2];
    int count = 0;
    int cr = r + dr, cc = c + dc;
    while (cr >= 0 && cr < GRID && cc >= 0 && cc < GRID &&
           board[cell(cr, cc)] == opp) {
      tmp[count][0] = cr;
      tmp[count][1] = cc;
      count++;
      cr += dr;
      cc += dc;
    }
    if (count > 0 && cr >= 0 && cr < GRID && cc >= 0 && cc < GRID &&
        board[cell(cr, cc)] == player) {
      for (int i = 0; i < count; i++) {
        flips[total][0] = tmp[i][0];
        flips[total][1] = tmp[i][1];
        total++;
      }
    }
  }
  return total;
}

static int get_valid_moves(const char *board, char player, int moves[][2]) {
  int count = 0;
  int flips[BOARD_SIZE][2];
  for (int r = 0; r < GRID; r++)
    for (int c = 0; c < GRID; c++)
      if (board[cell(r, c)] == EMPTY &&
          get_flips(board, r, c, player, flips) > 0) {
        moves[count][0] = r;
        moves[count][1] = c;
        count++;
      }
  return count;
}

static void apply_move(char *board, int r, int c, char player) {
  int flips[BOARD_SIZE][2];
  int nf = get_flips(board, r, c, player, flips);
  board[cell(r, c)] = player;
  for (int i = 0; i < nf; i++)
    board[cell(flips[i][0], flips[i][1])] = player;
}

static int count_pieces(const char *board, char player) {
  int c = 0;
  for (int i = 0; i < BOARD_SIZE; i++)
    if (board[i] == player)
      c++;
  return c;
}

int main(void) {
  setbuf(stdout, NULL);
  seed_rng(42);

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
  g_cfg.max_docs = 10000;
  g_cfg.max_doc_len = 128;
  microgpt_print_config("MicroGPT-C - Othello 6x6 Kanban Pipeline Demo",
                        &g_cfg);

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

  printf("\n--- PHASE 2: KANBAN PIPELINE EXECUTION ---\n");
  printf("Playing %d Othello games as X vs random O...\n\n", NUM_TEST_GAMES);

  int total_wins = 0, total_draws = 0, total_losses = 0;
  int total_moves = 0, total_valid = 0, total_invalid = 0;
  int total_parse_errors = 0, total_replans = 0;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  unsigned int game_seed = 12345;

  for (int gi = 0; gi < NUM_TEST_GAMES; gi++) {
    char board[BOARD_SIZE + 1];
    init_board(board);

    OpaKanban kb;
    opa_kanban_init(&kb, MAX_LAST_HISTORY);

    char turn = PLAYER_X;
    int consecutive_passes = 0;

    if (gi < 15 || (gi + 1) % 10 == 0)
      printf("-- Game %d/%d --\n", gi + 1, NUM_TEST_GAMES);

    while (consecutive_passes < 2) {
      int moves[BOARD_SIZE][2];
      int nmoves = get_valid_moves(board, turn, moves);

      if (nmoves == 0) {
        consecutive_passes++;
        turn = (turn == PLAYER_X) ? PLAYER_O : PLAYER_X;
        continue;
      }
      consecutive_passes = 0;

      if (turn == PLAYER_X) {
        /* Neural player */
        char board_str[BOARD_SIZE + 2];
        board_to_str(board, board_str);

        if (kb.stalls >= REPLAN_THRESHOLD && kb.replans < 3) {
          kb.replans++;
          total_replans++;
          opa_kanban_clear_blocked(&kb);
          kb.stalls = 0;
        }

        char valid_str[256] = "";
        size_t vs = 0;
        for (int i = 0; i < nmoves; i++) {
          if (i > 0 && vs < sizeof(valid_str)) {
            int n = snprintf(valid_str + vs, sizeof(valid_str) - vs, ",");
            if (n > 0)
              vs += (size_t)n;
          }
          if (vs < sizeof(valid_str)) {
            int n = snprintf(valid_str + vs, sizeof(valid_str) - vs, "R%dC%d",
                             moves[i][0], moves[i][1]);
            if (n > 0)
              vs += (size_t)n;
          }
        }

        char player_prompt[256];
        if (kb.blocked[0])
          snprintf(player_prompt, sizeof(player_prompt),
                   "board=%s|valid=%s|blocked=%s", board_str, valid_str,
                   kb.blocked);
        else
          snprintf(player_prompt, sizeof(player_prompt), "board=%s|valid=%s",
                   board_str, valid_str);

        char move_output[INF_GEN_LEN + 1];
        scalar_t conf = 0;
        organelle_generate_ensemble(player, &g_cfg, player_prompt, move_output,
                                    INF_GEN_LEN, ENSEMBLE_VOTES, ORGANELLE_TEMP,
                                    &conf);

        int pr = -1, pc = -1;
        if (strlen(move_output) >= 4 && move_output[0] == 'R' &&
            move_output[2] == 'C') {
          pr = move_output[1] - '0';
          pc = move_output[3] - '0';
        }

        /* Validate */
        int valid_move = 0;
        if (pr >= 0 && pr < GRID && pc >= 0 && pc < GRID) {
          int flips[BOARD_SIZE][2];
          if (board[cell(pr, pc)] == EMPTY &&
              get_flips(board, pr, pc, PLAYER_X, flips) > 0) {
            valid_move = 1;
          }
        }

        if (!valid_move) {
          total_parse_errors++;
          pr = moves[0][0];
          pc = moves[0][1];
          kb.stalls++;
        }

        apply_move(board, pr, pc, PLAYER_X);
        total_moves++;
        total_valid++;
        char ms[16];
        snprintf(ms, sizeof(ms), "R%dC%d", pr, pc);
        opa_kanban_add_last(&kb, ms);
        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;

      } else {
        /* Random opponent */
        int ri = rand_r(&game_seed) % nmoves;
        apply_move(board, moves[ri][0], moves[ri][1], PLAYER_O);
        total_moves++;
      }

      turn = (turn == PLAYER_X) ? PLAYER_O : PLAYER_X;
    }

    int x_count = count_pieces(board, PLAYER_X);
    int o_count = count_pieces(board, PLAYER_O);

    if (x_count > o_count) {
      total_wins++;
      if (gi < 15 || (gi + 1) % 10 == 0)
        printf("   X wins %d-%d\n", x_count, o_count);
    } else if (x_count < o_count) {
      total_losses++;
      if (gi < 15 || (gi + 1) % 10 == 0)
        printf("   O wins %d-%d\n", o_count, x_count);
    } else {
      total_draws++;
      if (gi < 15 || (gi + 1) % 10 == 0)
        printf("   Draw %d-%d\n", x_count, o_count);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double pt =
      (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

  printf(
      "\n================================================================\n");
  printf("                    OTHELLO 6x6 RESULTS\n");
  printf("================================================================\n");
  printf("Games won (X):      %d / %d (%.0f%%)\n", total_wins, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_wins / NUM_TEST_GAMES : 0.0);
  printf("Games drawn:        %d / %d (%.0f%%)\n", total_draws, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_draws / NUM_TEST_GAMES : 0.0);
  printf("Games lost:         %d / %d (%.0f%%)\n", total_losses, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_losses / NUM_TEST_GAMES : 0.0);
  printf("Win+Draw rate:      %.0f%%\n",
         NUM_TEST_GAMES > 0
             ? 100.0 * (total_wins + total_draws) / NUM_TEST_GAMES
             : 0.0);
  printf("Total moves:        %d (avg %.1f)\n", total_moves,
         NUM_TEST_GAMES > 0 ? (double)total_moves / NUM_TEST_GAMES : 0.0);
  printf("Parse errors:       %d\n", total_parse_errors);
  printf("Planner re-plans:   %d\n", total_replans);
  printf("Pipeline time:      %.2fs\n", pt);
  printf("================================================================\n");

  organelle_free(planner);
  organelle_free(player);
  return 0;
}
