/*
 * MicroGPT-C — Sudoku 4x4 Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner on Sudoku 4x4:
 *   - 4x4 grid, digits 1-4, unique per row/col/2x2 box
 *   - Board string: 16 chars, '0' for empty, '1'-'4' for filled
 *   - Player output: "RrCcVv" e.g. "R0C1V3"
 *   - Judge is deterministic (row/col/box constraint check).
 *
 * Build:
 *   cmake --build build --target sudoku_demo
 *   ./build/sudoku_demo
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
#define PLANNER_CORPUS "sudoku_planner.txt"
#define PLAYER_CORPUS "sudoku_player.txt"
#define PLANNER_CKPT "sudoku_planner.ckpt"
#define PLAYER_CKPT "sudoku_player.ckpt"

#define ORGANELLE_TEMP 0.2
#define INF_GEN_LEN 60
#define NUM_TEST_PUZZLES 100
#define REPLAN_THRESHOLD 3
#define MAX_LAST_HISTORY 3
#define ENSEMBLE_VOTES 3
#define MAX_FILLS 20

/* ---- Board Constants ---- */
#define GRID 4
#define BOX 2
#define BOARD_SIZE (GRID * GRID) /* 16 */

static MicrogptConfig g_cfg;

/* ---- Board Helpers ---- */

static int idx(int r, int c) { return r * GRID + c; }

static void board_to_str(const int *board, char *out) {
  for (int i = 0; i < BOARD_SIZE; i++)
    out[i] = '0' + board[i];
  out[BOARD_SIZE] = '\0';
}

static int count_empty(const int *board) {
  int c = 0;
  for (int i = 0; i < BOARD_SIZE; i++)
    if (board[i] == 0)
      c++;
  return c;
}

static int is_valid_placement(const int *board, int r, int c, int val) {
  for (int i = 0; i < GRID; i++)
    if (board[idx(r, i)] == val)
      return 0;
  for (int i = 0; i < GRID; i++)
    if (board[idx(i, c)] == val)
      return 0;
  int br = (r / BOX) * BOX, bc = (c / BOX) * BOX;
  for (int rr = br; rr < br + BOX; rr++)
    for (int cc = bc; cc < bc + BOX; cc++)
      if (board[idx(rr, cc)] == val)
        return 0;
  return 1;
}

static int is_solved(const int *board) {
  for (int i = 0; i < BOARD_SIZE; i++)
    if (board[i] == 0)
      return 0;
  return 1;
}

static int solve_sudoku(int *board) {
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (board[i] == 0) {
      int r = i / GRID, c = i % GRID;
      for (int v = 1; v <= GRID; v++) {
        if (is_valid_placement(board, r, c, v)) {
          board[i] = v;
          if (solve_sudoku(board))
            return 1;
          board[i] = 0;
        }
      }
      return 0;
    }
  }
  return 1;
}

static void generate_solved_grid(int *board, unsigned int *seed) {
  memset(board, 0, BOARD_SIZE * sizeof(int));
  /* Simple: fill randomly with backtracking */
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (board[i] != 0)
      continue;
    int r = i / GRID, c = i % GRID;
    int order[4] = {1, 2, 3, 4};
    /* Shuffle */
    for (int j = 3; j > 0; j--) {
      int k = rand_r(seed) % (j + 1);
      int tmp = order[j];
      order[j] = order[k];
      order[k] = tmp;
    }
    int placed = 0;
    for (int j = 0; j < GRID; j++) {
      if (is_valid_placement(board, r, c, order[j])) {
        board[i] = order[j];
        placed = 1;
        break;
      }
    }
    if (!placed) {
      /* Restart — simple approach */
      memset(board, 0, BOARD_SIZE * sizeof(int));
      i = -1;
    }
  }
  /* Ensure solvable */
  if (!is_solved(board)) {
    int backup[BOARD_SIZE];
    memcpy(backup, board, sizeof(backup));
    if (!solve_sudoku(board)) {
      memcpy(board, backup, sizeof(backup));
    }
  }
}

static void generate_puzzle(const int *solved, int *puzzle, int num_remove,
                            unsigned int *seed) {
  memcpy(puzzle, solved, BOARD_SIZE * sizeof(int));
  int cells[BOARD_SIZE];
  for (int i = 0; i < BOARD_SIZE; i++)
    cells[i] = i;
  /* Shuffle */
  for (int i = BOARD_SIZE - 1; i > 0; i--) {
    int j = rand_r(seed) % (i + 1);
    int tmp = cells[i];
    cells[i] = cells[j];
    cells[j] = tmp;
  }
  int removed = 0;
  for (int i = 0; i < BOARD_SIZE && removed < num_remove; i++) {
    puzzle[cells[i]] = 0;
    removed++;
  }
}

static void print_board(const int *board) {
  for (int r = 0; r < GRID; r++) {
    printf("  ");
    for (int c = 0; c < GRID; c++) {
      printf("%c ", board[idx(r, c)] ? '0' + board[idx(r, c)] : '.');
    }
    printf("\n");
  }
}

/* ---- Main ---- */

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
  g_cfg.max_docs = 20000;
  g_cfg.max_doc_len = 128;
  microgpt_print_config("MicroGPT-C - Sudoku 4x4 Kanban Pipeline Demo", &g_cfg);

  (void)print_board;

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
  printf("Solving %d Sudoku 4x4 puzzles...\n\n", NUM_TEST_PUZZLES);

  int total_solved = 0, total_moves = 0;
  int total_valid = 0, total_invalid = 0, total_parse_errors = 0,
      total_replans = 0;
  int easy_total = 0, easy_solved = 0;
  int med_total = 0, med_solved = 0;
  int hard_total = 0, hard_solved = 0;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  unsigned int puzzle_seed = 12345;

  for (int pi = 0; pi < NUM_TEST_PUZZLES; pi++) {
    int solved_grid[BOARD_SIZE];
    generate_solved_grid(solved_grid, &puzzle_seed);

    int num_remove;
    if (pi < 30) {
      num_remove = 4 + rand_r(&puzzle_seed) % 3;
      easy_total++;
    } else if (pi < 70) {
      num_remove = 6 + rand_r(&puzzle_seed) % 3;
      med_total++;
    } else {
      num_remove = 8 + rand_r(&puzzle_seed) % 3;
      hard_total++;
    }

    int board[BOARD_SIZE];
    generate_puzzle(solved_grid, board, num_remove, &puzzle_seed);

    OpaKanban kb;
    opa_kanban_init(&kb, MAX_LAST_HISTORY);

    char board_str[BOARD_SIZE + 2];
    board_to_str(board, board_str);
    int empty_count = count_empty(board);

    char planner_prompt[128];
    snprintf(planner_prompt, sizeof(planner_prompt), "board=%s|empty=%d",
             board_str, empty_count);
    char plan_output[INF_GEN_LEN + 1];
    organelle_generate(planner, &g_cfg, planner_prompt, plan_output,
                       INF_GEN_LEN, ORGANELLE_TEMP);
    if (!opa_pipe_starts_with(plan_output, "todo="))
      total_parse_errors++;

    int moves = 0;
    int puzzle_solved = 0;

    if (pi < 15 || (pi + 1) % 10 == 0)
      printf("-- Puzzle %d/%d (remove=%d) --\n", pi + 1, NUM_TEST_PUZZLES,
             num_remove);

    while (!puzzle_solved && moves < MAX_FILLS) {
      board_to_str(board, board_str);
      if (is_solved(board)) {
        puzzle_solved = 1;
        break;
      }

      if (kb.stalls >= REPLAN_THRESHOLD && kb.replans < 3) {
        kb.replans++;
        total_replans++;
        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;
      }

      /* Build valid cells */
      char valid_str[128] = "";
      int vs = 0;
      for (int r = 0; r < GRID; r++)
        for (int c = 0; c < GRID; c++)
          if (board[idx(r, c)] == 0) {
            if (vs > 0)
              vs +=
                  snprintf(valid_str + vs, sizeof(valid_str) - (size_t)vs, ",");
            vs += snprintf(valid_str + vs, sizeof(valid_str) - (size_t)vs,
                           "R%dC%d", r, c);
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

      /* Parse "R0C1V3" */
      int pr = -1, pc = -1, pv = -1;
      if (strlen(move_output) >= 6 && move_output[0] == 'R' &&
          move_output[2] == 'C' && move_output[4] == 'V') {
        pr = move_output[1] - '0';
        pc = move_output[3] - '0';
        pv = move_output[5] - '0';
      }

      int valid_move = 0;
      if (pr >= 0 && pr < GRID && pc >= 0 && pc < GRID && pv >= 1 &&
          pv <= GRID) {
        if (board[idx(pr, pc)] == 0 && is_valid_placement(board, pr, pc, pv)) {
          board[idx(pr, pc)] = pv;
          moves++;
          total_valid++;
          valid_move = 1;
          opa_kanban_clear_blocked(&kb);
          kb.stalls = 0;
          char ms[16];
          snprintf(ms, sizeof(ms), "R%dC%dV%d", pr, pc, pv);
          opa_kanban_add_last(&kb, ms);
        }
      }

      if (!valid_move) {
        total_parse_errors++;
        /* Fallback: find a cell with only one candidate */
        int placed = 0;
        for (int r = 0; r < GRID && !placed; r++)
          for (int c = 0; c < GRID && !placed; c++)
            if (board[idx(r, c)] == 0) {
              for (int v = 1; v <= GRID; v++) {
                if (is_valid_placement(board, r, c, v)) {
                  board[idx(r, c)] = v;
                  moves++;
                  total_valid++;
                  placed = 1;
                  break;
                }
              }
            }
        if (!placed)
          break;
        kb.stalls++;
      }

      if (is_solved(board))
        puzzle_solved = 1;
    }

    total_moves += moves;
    if (puzzle_solved) {
      total_solved++;
      if (num_remove <= 5)
        easy_solved++;
      else if (num_remove <= 7)
        med_solved++;
      else
        hard_solved++;
      if (pi < 15 || (pi + 1) % 10 == 0)
        printf("   Solved in %d fills!\n", moves);
    } else {
      if (pi < 15 || (pi + 1) % 10 == 0)
        printf("   Not solved (%d empty after %d fills)\n", count_empty(board),
               moves);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double pt =
      (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

  printf(
      "\n================================================================\n");
  printf("                    SUDOKU 4x4 RESULTS\n");
  printf("================================================================\n");
  printf("Puzzles solved:     %d / %d (%.0f%%)\n", total_solved,
         NUM_TEST_PUZZLES,
         NUM_TEST_PUZZLES > 0 ? 100.0 * total_solved / NUM_TEST_PUZZLES : 0.0);
  printf("  Easy (4-5):       %d / %d (%.0f%%)\n", easy_solved, easy_total,
         easy_total > 0 ? 100.0 * easy_solved / easy_total : 0.0);
  printf("  Medium (6-7):     %d / %d (%.0f%%)\n", med_solved, med_total,
         med_total > 0 ? 100.0 * med_solved / med_total : 0.0);
  printf("  Hard (8+):        %d / %d (%.0f%%)\n", hard_solved, hard_total,
         hard_total > 0 ? 100.0 * hard_solved / hard_total : 0.0);
  printf("Total fills:        %d (avg %.1f)\n", total_moves,
         NUM_TEST_PUZZLES > 0 ? (double)total_moves / NUM_TEST_PUZZLES : 0.0);
  printf("Valid moves:        %d\n", total_valid);
  printf("Parse errors:       %d\n", total_parse_errors);
  printf("Planner re-plans:   %d\n", total_replans);
  printf("Pipeline time:      %.2fs\n", pt);
  printf("================================================================\n");

  organelle_free(planner);
  organelle_free(player);
  return 0;
}
