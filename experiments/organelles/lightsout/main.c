/*
 * MicroGPT-C — Lights Out Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner on Lights Out (5x5):
 *   - 25-cell grid, each cell is on (1) or off (0)
 *   - Pressing a cell toggles it and its orthogonal neighbours
 *   - Goal: turn all lights off
 *   - Two neural organelles (Planner + Player) coordinate via pipe-separated
 *     flat strings with kanban state.
 *   - Judge is fully deterministic (cell valid + all-off check).
 *
 * Architecture: same as tictactoe/connect4 (n_embd=96, n_layer=4, ~460K
 * params).
 *
 * Pipeline: Planner -> Player -> Judge(deterministic) -> repeat
 *
 * Build:
 *   cmake --build build --target lightsout_demo
 *   ./build/lightsout_demo
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
#define PLANNER_CORPUS "lightsout_planner.txt"
#define PLAYER_CORPUS "lightsout_player.txt"

#define PLANNER_CKPT "lightsout_planner.ckpt"
#define PLAYER_CKPT "lightsout_player.ckpt"

#define ORGANELLE_TEMP 0.2 /* low temperature for reliable retrieval */
#define INF_GEN_LEN 60     /* max chars per organelle generation */

#define NUM_TEST_PUZZLES 100 /* puzzles to solve */
#define REPLAN_THRESHOLD 3   /* stalls before re-invoking Planner */
#define MAX_LAST_HISTORY 3   /* keep last N moves in history */
#define ENSEMBLE_VOTES 3     /* worker votes per move */
#define MAX_MOVES 30         /* max toggles per puzzle */

/* ---- Board Constants ---- */
#define GRID_SIZE 5
#define NUM_CELLS (GRID_SIZE * GRID_SIZE) /* 25 */

/* ---- File-scoped runtime config ---- */
static MicrogptConfig g_cfg;

/* ---- Board Helpers ---- */

static void board_to_str(const int *board, char *out) {
  for (int i = 0; i < NUM_CELLS; i++) {
    out[i] = '0' + board[i];
  }
  out[NUM_CELLS] = '\0';
}

static int count_lit(const int *board) {
  int count = 0;
  for (int i = 0; i < NUM_CELLS; i++) {
    if (board[i])
      count++;
  }
  return count;
}

static void toggle_cell(int *board, int cell) {
  if (cell < 0 || cell >= NUM_CELLS)
    return;
  int r = cell / GRID_SIZE;
  int c = cell % GRID_SIZE;

  board[cell] ^= 1;
  if (r > 0)
    board[(r - 1) * GRID_SIZE + c] ^= 1;
  if (r < GRID_SIZE - 1)
    board[(r + 1) * GRID_SIZE + c] ^= 1;
  if (c > 0)
    board[r * GRID_SIZE + c - 1] ^= 1;
  if (c < GRID_SIZE - 1)
    board[r * GRID_SIZE + c + 1] ^= 1;
}

static int is_solved(const int *board) { return count_lit(board) == 0; }

static void print_board(const int *board) {
  for (int r = 0; r < GRID_SIZE; r++) {
    printf("  ");
    for (int c = 0; c < GRID_SIZE; c++) {
      printf("%c ", board[r * GRID_SIZE + c] ? '*' : '.');
    }
    printf("\n");
  }
}

/* ---- Puzzle Generation ---- */

static void generate_puzzle(int *board, unsigned int *seed, int num_presses) {
  memset(board, 0, NUM_CELLS * sizeof(int));
  for (int i = 0; i < num_presses; i++) {
    int cell = rand_r(seed) % NUM_CELLS;
    toggle_cell(board, cell);
  }
  /* Ensure at least one light is on */
  if (is_solved(board)) {
    toggle_cell(board, rand_r(seed) % NUM_CELLS);
  }
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
  g_cfg.max_docs = 20000;
  g_cfg.max_doc_len = 128;
  microgpt_print_config("MicroGPT-C - Lights Out Kanban Pipeline Demo", &g_cfg);

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
   * PHASE 2: Pipeline — Solve Puzzles
   * ================================================================ */

  printf("\n--- PHASE 2: KANBAN PIPELINE EXECUTION ---\n");
  printf("Solving %d Lights Out puzzles...\n\n", NUM_TEST_PUZZLES);

  int total_solved = 0;
  int total_moves = 0;
  int total_valid_moves = 0;
  int total_invalid_moves = 0;
  int total_parse_errors = 0;
  int total_replans = 0;

  /* Difficulty buckets */
  int easy_total = 0, easy_solved = 0; /* 1-3 presses */
  int med_total = 0, med_solved = 0;   /* 4-7 presses */
  int hard_total = 0, hard_solved = 0; /* 8+ presses */

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  unsigned int puzzle_seed = 12345;

  for (int pi = 0; pi < NUM_TEST_PUZZLES; pi++) {
    /* Generate puzzle with varying difficulty */
    int num_presses;
    if (pi < 30) {
      num_presses = 1 + (rand_r(&puzzle_seed) % 3); /* easy: 1-3 */
    } else if (pi < 70) {
      num_presses = 4 + (rand_r(&puzzle_seed) % 4); /* medium: 4-7 */
    } else {
      num_presses = 8 + (rand_r(&puzzle_seed) % 8); /* hard: 8-15 */
    }

    int board[NUM_CELLS];
    generate_puzzle(board, &puzzle_seed, num_presses);

    int initial_lit = count_lit(board);

    if (num_presses <= 3)
      easy_total++;
    else if (num_presses <= 7)
      med_total++;
    else
      hard_total++;

    OpaKanban kb;
    opa_kanban_init(&kb, MAX_LAST_HISTORY);

    char board_str[NUM_CELLS + 2];
    board_to_str(board, board_str);

    /* Step 1: Ask Planner for initial plan */
    char planner_prompt[128];
    snprintf(planner_prompt, sizeof(planner_prompt), "board=%s|lit=%d",
             board_str, initial_lit);

    char plan_output[INF_GEN_LEN + 1];
    organelle_generate(planner, &g_cfg, planner_prompt, plan_output,
                       INF_GEN_LEN, ORGANELLE_TEMP);

    if (!opa_pipe_starts_with(plan_output, "todo=")) {
      total_parse_errors++;
    }

    /* Step 2: Solve the puzzle */
    int moves_made = 0;
    int solved = 0;

    if (pi < 15 || (pi + 1) % 10 == 0) {
      printf("-- Puzzle %d/%d (presses=%d, lit=%d) --\n", pi + 1,
             NUM_TEST_PUZZLES, num_presses, initial_lit);
    }

    while (!solved && moves_made < MAX_MOVES) {
      board_to_str(board, board_str);
      int lit = count_lit(board);

      if (lit == 0) {
        solved = 1;
        break;
      }

      /* Re-plan if stalled */
      if (kb.stalls >= REPLAN_THRESHOLD && kb.replans < 3) {
        kb.replans++;
        total_replans++;

        char replan_prompt[128];
        snprintf(replan_prompt, sizeof(replan_prompt),
                 "board=%s|lit=%d|stalled", board_str, lit);

        char replan_output[INF_GEN_LEN + 1];
        organelle_generate(planner, &g_cfg, replan_prompt, replan_output,
                           INF_GEN_LEN, ORGANELLE_TEMP);

        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;
      }

      /* Build valid-move string: all cells 0-24 */
      char valid_str[128] = "";
      size_t vs_pos = 0;
      for (int i = 0; i < NUM_CELLS; i++) {
        if (i > 0 && vs_pos < sizeof(valid_str)) {
          int n = snprintf(valid_str + vs_pos, sizeof(valid_str) - vs_pos, ",");
          if (n > 0)
            vs_pos += (size_t)n;
        }
        if (vs_pos < sizeof(valid_str)) {
          int n =
              snprintf(valid_str + vs_pos, sizeof(valid_str) - vs_pos, "%d", i);
          if (n > 0)
            vs_pos += (size_t)n;
        }
      }

      /* Build Player prompt */
      char player_prompt[256];
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

      /* Parse cell number (0-24) */
      int proposed_cell = -1;
      {
        char *endp;
        long val = strtol(move_output, &endp, 10);
        if (endp != move_output && val >= 0 && val < NUM_CELLS) {
          proposed_cell = (int)val;
        }
      }

      if (proposed_cell < 0) {
        total_parse_errors++;
        /* Fallback: pick a lit cell */
        for (int i = 0; i < NUM_CELLS; i++) {
          if (board[i] &&
              !opa_kanban_is_blocked(&kb, (char[4]){(char)('0' + i / 10),
                                                    (char)('0' + i % 10), '\0',
                                                    '\0'})) {
            proposed_cell = i;
            break;
          }
        }
        if (proposed_cell < 0) {
          /* Pick any lit cell */
          for (int i = 0; i < NUM_CELLS; i++) {
            if (board[i]) {
              proposed_cell = i;
              break;
            }
          }
        }
        if (proposed_cell < 0)
          break;
      }

      /* Apply toggle */
      toggle_cell(board, proposed_cell);
      moves_made++;
      total_valid_moves++;

      char cell_str[8];
      snprintf(cell_str, sizeof(cell_str), "%d", proposed_cell);
      opa_kanban_add_last(&kb, cell_str);
      opa_kanban_clear_blocked(&kb);
      kb.stalls = 0;

      if (is_solved(board)) {
        solved = 1;
      }
    }

    total_moves += moves_made;

    if (solved) {
      total_solved++;
      if (num_presses <= 3)
        easy_solved++;
      else if (num_presses <= 7)
        med_solved++;
      else
        hard_solved++;

      if (pi < 15 || (pi + 1) % 10 == 0) {
        printf("   Solved in %d toggles!\n", moves_made);
      }
    } else {
      if (pi < 15 || (pi + 1) % 10 == 0) {
        printf("   Not solved (lit=%d after %d toggles)\n", count_lit(board),
               moves_made);
      }
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
  printf("                    LIGHTS OUT RESULTS\n");
  printf("================================================================\n");
  printf("Puzzles solved:     %d / %d (%.0f%%)\n", total_solved,
         NUM_TEST_PUZZLES,
         NUM_TEST_PUZZLES > 0 ? 100.0 * total_solved / NUM_TEST_PUZZLES : 0.0);
  printf("  Easy (1-3):       %d / %d (%.0f%%)\n", easy_solved, easy_total,
         easy_total > 0 ? 100.0 * easy_solved / easy_total : 0.0);
  printf("  Medium (4-7):     %d / %d (%.0f%%)\n", med_solved, med_total,
         med_total > 0 ? 100.0 * med_solved / med_total : 0.0);
  printf("  Hard (8+):        %d / %d (%.0f%%)\n", hard_solved, hard_total,
         hard_total > 0 ? 100.0 * hard_solved / hard_total : 0.0);
  printf("Total toggles:      %d (avg %.1f per puzzle)\n", total_moves,
         NUM_TEST_PUZZLES > 0 ? (double)total_moves / NUM_TEST_PUZZLES : 0.0);
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
