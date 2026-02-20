/*
 * MicroGPT-C — Red Donkey (Huarong Dao) Multi-Organelle Demo
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Simplified Red Donkey: 4x3 grid, slide blocks to move 2x2 "donkey" down.
 * Player output: "BD" (block ID + direction U/D/L/R).
 * Judge is deterministic (collision check, goal check).
 *
 * Note: The classic 5x4 Red Donkey has ~10^10 states, making BFS corpus
 * generation intractable. This 4x3 variant tests the same OPA coordination
 * patterns (multi-cell block sliding, collision avoidance) in a smaller space.
 *
 * Build:
 *   cmake --build build --target reddonkey_demo
 *   ./build/reddonkey_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PLANNER_CORPUS "reddonkey_planner.txt"
#define PLAYER_CORPUS "reddonkey_player.txt"
#define PLANNER_CKPT "reddonkey_planner.ckpt"
#define PLAYER_CKPT "reddonkey_player.ckpt"

#define ORGANELLE_TEMP 0.2
#define INF_GEN_LEN 60
#define NUM_TEST_PUZZLES 100
#define REPLAN_THRESHOLD 3
#define MAX_LAST_HISTORY 3
#define ENSEMBLE_VOTES 3
#define MAX_SLIDES 60

#define ROWS 4
#define COLS 3
#define BOARD_SIZE (ROWS * COLS) /* 12 */
#define EMPTY '.'

static MicrogptConfig g_cfg;

static int cell(int r, int c) { return r * COLS + c; }

static void board_to_str(const char *board, char *out) {
  memcpy(out, board, BOARD_SIZE);
  out[BOARD_SIZE] = '\0';
}

static void dir_delta(char d, int *dr, int *dc) {
  *dr = 0;
  *dc = 0;
  if (d == 'U')
    *dr = -1;
  else if (d == 'D')
    *dr = 1;
  else if (d == 'L')
    *dc = -1;
  else if (d == 'R')
    *dc = 1;
}

static int can_move(const char *board, char block_id, char direction) {
  int dr, dc;
  dir_delta(direction, &dr, &dc);
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      if (board[cell(r, c)] == block_id) {
        int nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
          return 0;
        if (board[cell(nr, nc)] != EMPTY && board[cell(nr, nc)] != block_id)
          return 0;
      }
  return 1;
}

static void apply_move(char *board, char block_id, char direction) {
  int dr, dc;
  dir_delta(direction, &dr, &dc);
  int cells_r[BOARD_SIZE], cells_c[BOARD_SIZE];
  int ncells = 0;
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      if (board[cell(r, c)] == block_id) {
        cells_r[ncells] = r;
        cells_c[ncells] = c;
        ncells++;
      }
  for (int i = 0; i < ncells; i++)
    board[cell(cells_r[i], cells_c[i])] = EMPTY;
  for (int i = 0; i < ncells; i++)
    board[cell(cells_r[i] + dr, cells_c[i] + dc)] = block_id;
}

static int is_goal(const char *board) {
  /* A at bottom: rows 2-3, cols 0-1 */
  return board[cell(2, 0)] == 'A' && board[cell(2, 1)] == 'A' &&
         board[cell(3, 0)] == 'A' && board[cell(3, 1)] == 'A';
}

static void get_valid_moves(const char *board, char *valid_str, int max_len) {
  int pos = 0;
  char seen[256] = {0};
  for (int i = 0; i < BOARD_SIZE; i++) {
    char bid = board[i];
    if (bid == EMPTY || seen[(unsigned char)bid])
      continue;
    seen[(unsigned char)bid] = 1;
    const char dirs[] = "UDLR";
    for (int d = 0; d < 4; d++) {
      if (can_move(board, bid, dirs[d])) {
        if (pos > 0 && pos < max_len - 1)
          valid_str[pos++] = ',';
        if (pos < max_len - 2) {
          valid_str[pos++] = bid;
          valid_str[pos++] = dirs[d];
        }
      }
    }
  }
  valid_str[pos] = '\0';
}

static void make_start(char *board) {
  memset(board, EMPTY, BOARD_SIZE);
  /* A = 2x2 at (0,0) */
  board[cell(0, 0)] = 'A';
  board[cell(0, 1)] = 'A';
  board[cell(1, 0)] = 'A';
  board[cell(1, 1)] = 'A';
  /* B at (0,2), C at (1,2), D at (2,2), E at (3,2) */
  board[cell(0, 2)] = 'B';
  board[cell(1, 2)] = 'C';
  board[cell(2, 2)] = 'D';
  board[cell(3, 2)] = 'E';
  /* Empty: (2,0), (2,1), (3,0), (3,1) */
}

static void generate_puzzle(char *board, unsigned int *seed) {
  make_start(board);
  int scrambles = 3 + (int)(rand_r(seed) % 12);
  for (int s = 0; s < scrambles; s++) {
    char blocks[26];
    int nb = 0;
    char seen[256] = {0};
    for (int i = 0; i < BOARD_SIZE; i++) {
      if (board[i] != EMPTY && !seen[(unsigned char)board[i]]) {
        seen[(unsigned char)board[i]] = 1;
        blocks[nb++] = board[i];
      }
    }
    if (nb == 0)
      break;
    for (int t = 0; t < 20; t++) {
      char bid = blocks[rand_r(seed) % nb];
      const char dirs[] = "UDLR";
      char d = dirs[rand_r(seed) % 4];
      if (can_move(board, bid, d)) {
        apply_move(board, bid, d);
        break;
      }
    }
  }
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
  microgpt_print_config("MicroGPT-C - Red Donkey Kanban Pipeline Demo", &g_cfg);

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
  printf("Solving %d Red Donkey puzzles...\n\n", NUM_TEST_PUZZLES);

  int total_solved = 0, total_moves = 0;
  int total_parse_errors = 0, total_replans = 0;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  unsigned int seed = 12345;

  for (int pi = 0; pi < NUM_TEST_PUZZLES; pi++) {
    char board[BOARD_SIZE + 1];
    generate_puzzle(board, &seed);
    board[BOARD_SIZE] = '\0';

    if (is_goal(board)) {
      total_solved++;
      continue;
    }

    OpaKanban kb;
    opa_kanban_init(&kb, MAX_LAST_HISTORY);

    int moves = 0, solved = 0;

    if (pi < 15 || (pi + 1) % 10 == 0)
      printf("-- Puzzle %d/%d --\n", pi + 1, NUM_TEST_PUZZLES);

    while (!solved && moves < MAX_SLIDES) {
      char board_str[BOARD_SIZE + 2];
      board_to_str(board, board_str);

      if (kb.stalls >= REPLAN_THRESHOLD && kb.replans < 3) {
        kb.replans++;
        total_replans++;
        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;
      }

      char valid_str[256];
      get_valid_moves(board, valid_str, sizeof(valid_str));

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

      char bid = 0, dir = 0;
      if (strlen(mo) >= 2) {
        bid = mo[0];
        dir = mo[1];
      }

      if (bid && dir &&
          (dir == 'U' || dir == 'D' || dir == 'L' || dir == 'R') &&
          can_move(board, bid, dir)) {
        apply_move(board, bid, dir);
        moves++;
        char ms[8];
        ms[0] = bid;
        ms[1] = dir;
        ms[2] = '\0';
        opa_kanban_add_last(&kb, ms);
        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;
      } else {
        total_parse_errors++;
        if (valid_str[0] && valid_str[1]) {
          apply_move(board, valid_str[0], valid_str[1]);
          moves++;
        }
        kb.stalls++;
      }

      if (is_goal(board))
        solved = 1;
    }

    total_moves += moves;
    if (solved) {
      total_solved++;
      if (pi < 15 || (pi + 1) % 10 == 0)
        printf("   Solved in %d slides!\n", moves);
    } else {
      if (pi < 15 || (pi + 1) % 10 == 0)
        printf("   Not solved after %d slides\n", moves);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double pt =
      (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

  printf(
      "\n================================================================\n");
  printf("                    RED DONKEY RESULTS\n");
  printf("================================================================\n");
  printf("Puzzles solved:     %d / %d (%.0f%%)\n", total_solved,
         NUM_TEST_PUZZLES,
         NUM_TEST_PUZZLES > 0 ? 100.0 * total_solved / NUM_TEST_PUZZLES : 0.0);
  printf("Total slides:       %d (avg %.1f)\n", total_moves,
         NUM_TEST_PUZZLES > 0 ? (double)total_moves / NUM_TEST_PUZZLES : 0.0);
  printf("Parse errors:       %d\n", total_parse_errors);
  printf("Planner re-plans:   %d\n", total_replans);
  printf("Pipeline time:      %.2fs\n", pt);
  printf("================================================================\n");

  organelle_free(planner);
  organelle_free(player);
  return 0;
}
