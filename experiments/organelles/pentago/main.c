/*
 * MicroGPT-C — Pentago Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Pentago: 6x6, place + rotate a 3x3 quadrant, win with 5 in a row.
 * Player output: "PrrccQqDd" (place at r,c, rotate quadrant q, CW/CCW).
 *
 * Build:
 *   cmake --build build --target pentago_demo
 *   ./build/pentago_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PLANNER_CORPUS "pentago_planner.txt"
#define PLAYER_CORPUS "pentago_player.txt"
#define PLANNER_CKPT "pentago_planner.ckpt"
#define PLAYER_CKPT "pentago_player.ckpt"

#define ORGANELLE_TEMP 0.2
#define INF_GEN_LEN 60
#define NUM_TEST_GAMES 100
#define REPLAN_THRESHOLD 3
#define MAX_LAST_HISTORY 3
#define ENSEMBLE_VOTES 3

#define GRID 6
#define QSIZE 3
#define BOARD_SIZE (GRID * GRID)
#define EMPTY '.'
#define PLAYER_X 'X'
#define PLAYER_O 'O'

static MicrogptConfig g_cfg;

static int cell(int r, int c) { return r * GRID + c; }

static void board_to_str(const char *board, char *out) {
  memcpy(out, board, BOARD_SIZE);
  out[BOARD_SIZE] = '\0';
}

static void rotate_quad(char *board, int q, int clockwise) {
  int qr = (q / 2) * QSIZE, qc = (q % 2) * QSIZE;
  char vals[9];
  for (int r = 0; r < QSIZE; r++)
    for (int c = 0; c < QSIZE; c++)
      vals[r * QSIZE + c] = board[cell(qr + r, qc + c)];

  static const int CW[9] = {6, 3, 0, 7, 4, 1, 8, 5, 2};
  static const int CCW[9] = {2, 5, 8, 1, 4, 7, 0, 3, 6};
  const int *perm = clockwise ? CW : CCW;

  for (int r = 0; r < QSIZE; r++)
    for (int c = 0; c < QSIZE; c++)
      board[cell(qr + r, qc + c)] = vals[perm[r * QSIZE + c]];
}

static int check_five(const char *board, char player) {
  static const int DR[4] = {0, 1, 1, 1};
  static const int DC[4] = {1, 0, 1, -1};
  for (int r = 0; r < GRID; r++)
    for (int c = 0; c < GRID; c++)
      for (int d = 0; d < 4; d++) {
        int er = r + 4 * DR[d], ec = c + 4 * DC[d];
        if (er < 0 || er >= GRID || ec < 0 || ec >= GRID)
          continue;
        int ok = 1;
        for (int i = 0; i < 5 && ok; i++)
          if (board[cell(r + i * DR[d], c + i * DC[d])] != player)
            ok = 0;
        if (ok)
          return 1;
      }
  return 0;
}

static int get_empties(const char *board, int pos[][2]) {
  int n = 0;
  for (int r = 0; r < GRID; r++)
    for (int c = 0; c < GRID; c++)
      if (board[cell(r, c)] == EMPTY) {
        pos[n][0] = r;
        pos[n][1] = c;
        n++;
      }
  return n;
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
  microgpt_print_config("MicroGPT-C - Pentago Kanban Pipeline Demo", &g_cfg);

  int train_steps = g_cfg.num_steps;
  printf("--- PHASE 1: TRAINING (%d steps each) ---\n", train_steps);

  Organelle *planner = organelle_train("Planner", PLANNER_CORPUS, PLANNER_CKPT,
                                       &g_cfg, train_steps);
  if (!planner) {
    fprintf(stderr, "FATAL: Planner failed\n");
    return 1;
  }
  Organelle *player = organelle_train("Player", PLAYER_CORPUS, PLAYER_CKPT,
                                      &g_cfg, train_steps);
  if (!player) {
    fprintf(stderr, "FATAL: Player failed\n");
    return 1;
  }

  printf("\n--- PHASE 2: KANBAN PIPELINE EXECUTION ---\n");
  printf("Playing %d Pentago games as X vs random O...\n\n", NUM_TEST_GAMES);

  int total_wins = 0, total_draws = 0, total_losses = 0;
  int total_moves = 0, total_parse_errors = 0, total_replans = 0;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  unsigned int game_seed = 12345;

  for (int gi = 0; gi < NUM_TEST_GAMES; gi++) {
    char board[BOARD_SIZE + 1];
    memset(board, EMPTY, BOARD_SIZE);
    board[BOARD_SIZE] = '\0';

    OpaKanban kb;
    opa_kanban_init(&kb, MAX_LAST_HISTORY);
    char turn = PLAYER_X;
    char winner = EMPTY;

    if (gi < 15 || (gi + 1) % 10 == 0)
      printf("-- Game %d/%d --\n", gi + 1, NUM_TEST_GAMES);

    while (winner == EMPTY) {
      int empties[BOARD_SIZE][2];
      int ne = get_empties(board, empties);
      if (ne == 0)
        break;

      if (turn == PLAYER_X) {
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
        int limit = ne < 15 ? ne : 15;
        for (int i = 0; i < limit; i++) {
          if (i > 0 && vs < sizeof(valid_str)) {
            int n = snprintf(valid_str + vs, sizeof(valid_str) - vs, ",");
            if (n > 0)
              vs += (size_t)n;
          }
          if (vs < sizeof(valid_str)) {
            int n = snprintf(valid_str + vs, sizeof(valid_str) - vs, "R%dC%d",
                             empties[i][0], empties[i][1]);
            if (n > 0)
              vs += (size_t)n;
          }
        }

        char pp[256];
        if (kb.blocked[0])
          snprintf(pp, sizeof(pp), "board=%s|valid=%s|blocked=%s", board_str,
                   valid_str, kb.blocked);
        else
          snprintf(pp, sizeof(pp), "board=%s|valid=%s", board_str, valid_str);

        char mo[INF_GEN_LEN + 1];
        scalar_t conf = 0;
        organelle_generate_ensemble(player, &g_cfg, pp, mo, INF_GEN_LEN,
                                    ENSEMBLE_VOTES, ORGANELLE_TEMP, &conf);

        /* Parse "P23Q0DC" */
        int pr = -1, pc = -1, pq = 0, cw = 1;
        if (strlen(mo) >= 7 && mo[0] == 'P' && mo[3] == 'Q' && mo[5] == 'D') {
          pr = mo[1] - '0';
          pc = mo[2] - '0';
          pq = mo[4] - '0';
          cw = (mo[6] == 'C') ? 1 : 0;
        }

        if (pr < 0 || pr >= GRID || pc < 0 || pc >= GRID ||
            board[cell(pr, pc)] != EMPTY || pq < 0 || pq > 3) {
          total_parse_errors++;
          pr = empties[0][0];
          pc = empties[0][1];
          pq = 0;
          cw = 1;
          kb.stalls++;
        }

        board[cell(pr, pc)] = PLAYER_X;
        rotate_quad(board, pq, cw);
        total_moves++;

        char ms[16];
        snprintf(ms, sizeof(ms), "P%d%dQ%d", pr, pc, pq);
        opa_kanban_add_last(&kb, ms);
        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;

        if (check_five(board, PLAYER_X)) {
          winner = PLAYER_X;
          break;
        }
        if (check_five(board, PLAYER_O)) {
          winner = PLAYER_O;
          break;
        } /* rotation may help opponent */
      } else {
        int ri = rand_r(&game_seed) % ne;
        board[cell(empties[ri][0], empties[ri][1])] = PLAYER_O;
        rotate_quad(board, rand_r(&game_seed) % 4, rand_r(&game_seed) % 2);
        total_moves++;
        if (check_five(board, PLAYER_O)) {
          winner = PLAYER_O;
          break;
        }
        if (check_five(board, PLAYER_X)) {
          winner = PLAYER_X;
          break;
        }
      }

      turn = (turn == PLAYER_X) ? PLAYER_O : PLAYER_X;
    }

    if (winner == PLAYER_X) {
      total_wins++;
      if (gi < 15 || (gi + 1) % 10 == 0)
        printf("   X wins!\n");
    } else if (winner == PLAYER_O) {
      total_losses++;
      if (gi < 15 || (gi + 1) % 10 == 0)
        printf("   O wins\n");
    } else {
      total_draws++;
      if (gi < 15 || (gi + 1) % 10 == 0)
        printf("   Draw\n");
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double pt =
      (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

  printf(
      "\n================================================================\n");
  printf("                    PENTAGO RESULTS\n");
  printf("================================================================\n");
  printf("Games won (X):      %d / %d (%.0f%%)\n", total_wins, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_wins / NUM_TEST_GAMES : 0.0);
  printf("Games drawn:        %d / %d\n", total_draws, NUM_TEST_GAMES);
  printf("Games lost:         %d / %d (%.0f%%)\n", total_losses, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_losses / NUM_TEST_GAMES : 0.0);
  printf("Win+Draw rate:      %.0f%%\n",
         NUM_TEST_GAMES > 0
             ? 100.0 * (total_wins + total_draws) / NUM_TEST_GAMES
             : 0.0);
  printf("Parse errors:       %d\n", total_parse_errors);
  printf("Planner re-plans:   %d\n", total_replans);
  printf("Pipeline time:      %.2fs\n", pt);
  printf("================================================================\n");

  organelle_free(planner);
  organelle_free(player);
  return 0;
}
