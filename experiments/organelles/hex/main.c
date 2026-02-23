/*
 * MicroGPT-C — Hex 7x7 Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Hex: X connects top-bottom, O connects left-right.
 * Board: 49-char string. Player output: "RrCc".
 * Judge is deterministic (BFS path connectivity check).
 *
 * Build:
 *   cmake --build build --target hex_demo
 *   ./build/hex_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PLANNER_CORPUS "hex_planner.txt"
#define PLAYER_CORPUS "hex_player.txt"
#define PLANNER_CKPT "hex_planner.ckpt"
#define PLAYER_CKPT "hex_player.ckpt"

#define ORGANELLE_TEMP 0.2
#define INF_GEN_LEN 60
#define NUM_TEST_GAMES 100
#define REPLAN_THRESHOLD 3
#define MAX_LAST_HISTORY 3
#define ENSEMBLE_VOTES 3

#define GRID 7
#define BOARD_SIZE (GRID * GRID) /* 49 */
#define EMPTY '.'
#define PLAYER_X 'X'
#define PLAYER_O 'O'

static MicrogptConfig g_cfg;

static int cell(int r, int c) { return r * GRID + c; }

/* Hex neighbours: 6 directions */
static const int HEX_DR[6] = {-1, -1, 0, 0, 1, 1};
static const int HEX_DC[6] = {0, 1, -1, 1, -1, 0};

static void board_to_str(const char *board, char *out) {
  memcpy(out, board, BOARD_SIZE);
  out[BOARD_SIZE] = '\0';
}

static int get_empties(const char *board, int positions[][2]) {
  int count = 0;
  for (int r = 0; r < GRID; r++)
    for (int c = 0; c < GRID; c++)
      if (board[cell(r, c)] == EMPTY) {
        positions[count][0] = r;
        positions[count][1] = c;
        count++;
      }
  return count;
}

/* --- Topological feature functions --- */

static int count_groups(const char *board, char player) {
  /* Count connected components (Betti-0) of player's stones via BFS. */
  int visited[BOARD_SIZE];
  memset(visited, 0, sizeof(visited));
  int groups = 0;
  int queue[BOARD_SIZE][2];

  for (int r = 0; r < GRID; r++) {
    for (int c = 0; c < GRID; c++) {
      if (board[cell(r, c)] == player && !visited[cell(r, c)]) {
        groups++;
        int qh = 0, qt = 0;
        queue[qt][0] = r;
        queue[qt][1] = c;
        qt++;
        visited[cell(r, c)] = 1;
        while (qh < qt) {
          int cr = queue[qh][0], cc = queue[qh][1];
          qh++;
          for (int d = 0; d < 6; d++) {
            int nr = cr + HEX_DR[d], nc = cc + HEX_DC[d];
            if (nr >= 0 && nr < GRID && nc >= 0 && nc < GRID &&
                !visited[cell(nr, nc)] && board[cell(nr, nc)] == player) {
              visited[cell(nr, nc)] = 1;
              queue[qt][0] = nr;
              queue[qt][1] = nc;
              qt++;
            }
          }
        }
      }
    }
  }
  return groups;
}

static int shortest_edge_distance(const char *board, char player) {
  /* BFS distance from player's stones to target edge.
   * X targets bottom (r=GRID-1), O targets right (c=GRID-1).
   * Returns 0 if connected, 99 if no stones. */
  int dist[BOARD_SIZE];
  for (int i = 0; i < BOARD_SIZE; i++)
    dist[i] = 99;
  int queue[BOARD_SIZE * 2][3]; /* r, c, d */
  int qh = 0, qt = 0;

  /* Seed from target edge */
  if (player == PLAYER_X) {
    for (int c = 0; c < GRID; c++) {
      queue[qt][0] = GRID - 1;
      queue[qt][1] = c;
      queue[qt][2] = 0;
      qt++;
      dist[cell(GRID - 1, c)] = 0;
    }
  } else {
    for (int r = 0; r < GRID; r++) {
      queue[qt][0] = r;
      queue[qt][1] = GRID - 1;
      queue[qt][2] = 0;
      qt++;
      dist[cell(r, GRID - 1)] = 0;
    }
  }

  while (qh < qt) {
    int r = queue[qh][0], c = queue[qh][1], d = queue[qh][2];
    qh++;
    for (int dir = 0; dir < 6; dir++) {
      int nr = r + HEX_DR[dir], nc = c + HEX_DC[dir];
      if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID)
        continue;
      int nd = d;
      char ch = board[cell(nr, nc)];
      if (ch == player)
        nd = d; /* free to traverse own stone */
      else if (ch == EMPTY)
        nd = d + 1;
      else
        continue; /* can't traverse enemy */
      if (nd < dist[cell(nr, nc)]) {
        dist[cell(nr, nc)] = nd;
        queue[qt][0] = nr;
        queue[qt][1] = nc;
        queue[qt][2] = nd;
        qt++;
      }
    }
  }

  /* Min dist over all player stones */
  int min_d = 99;
  for (int r = 0; r < GRID; r++)
    for (int c = 0; c < GRID; c++)
      if (board[cell(r, c)] == player && dist[cell(r, c)] < min_d)
        min_d = dist[cell(r, c)];
  return min_d;
}

static int count_bridges(const char *board, char player) {
  /* Count empty cells adjacent to 2+ friendly stones. */
  int bridges = 0;
  for (int r = 0; r < GRID; r++)
    for (int c = 0; c < GRID; c++)
      if (board[cell(r, c)] == EMPTY) {
        int friendly = 0;
        for (int d = 0; d < 6; d++) {
          int nr = r + HEX_DR[d], nc = c + HEX_DC[d];
          if (nr >= 0 && nr < GRID && nc >= 0 && nc < GRID &&
              board[cell(nr, nc)] == player)
            friendly++;
        }
        if (friendly >= 2)
          bridges++;
      }
  return bridges;
}

static int has_friendly_neighbour(const char *board, int r, int c,
                                  char player) {
  for (int d = 0; d < 6; d++) {
    int nr = r + HEX_DR[d], nc = c + HEX_DC[d];
    if (nr >= 0 && nr < GRID && nc >= 0 && nc < GRID &&
        board[cell(nr, nc)] == player)
      return 1;
  }
  return 0;
}

static int check_connection(const char *board, char player) {
  /* BFS: X top→bottom, O left→right */
  int visited[BOARD_SIZE];
  memset(visited, 0, sizeof(visited));
  int queue[BOARD_SIZE][2];
  int qh = 0, qt = 0;

  if (player == PLAYER_X) {
    for (int c = 0; c < GRID; c++)
      if (board[cell(0, c)] == player) {
        queue[qt][0] = 0;
        queue[qt][1] = c;
        qt++;
        visited[cell(0, c)] = 1;
      }
  } else {
    for (int r = 0; r < GRID; r++)
      if (board[cell(r, 0)] == player) {
        queue[qt][0] = r;
        queue[qt][1] = 0;
        qt++;
        visited[cell(r, 0)] = 1;
      }
  }

  while (qh < qt) {
    int r = queue[qh][0], c = queue[qh][1];
    qh++;
    if (player == PLAYER_X && r == GRID - 1)
      return 1;
    if (player == PLAYER_O && c == GRID - 1)
      return 1;
    for (int d = 0; d < 6; d++) {
      int nr = r + HEX_DR[d], nc = c + HEX_DC[d];
      if (nr >= 0 && nr < GRID && nc >= 0 && nc < GRID &&
          !visited[cell(nr, nc)] && board[cell(nr, nc)] == player) {
        visited[cell(nr, nc)] = 1;
        queue[qt][0] = nr;
        queue[qt][1] = nc;
        qt++;
      }
    }
  }
  return 0;
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
  microgpt_print_config("MicroGPT-C - Hex 7x7 Kanban Pipeline Demo", &g_cfg);

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
  printf("Playing %d Hex games as X vs random O...\n\n", NUM_TEST_GAMES);

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
    int moves = 0;
    char winner = EMPTY;

    if (gi < 15 || (gi + 1) % 10 == 0)
      printf("-- Game %d/%d --\n", gi + 1, NUM_TEST_GAMES);

    while (winner == EMPTY) {
      int empties[BOARD_SIZE][2];
      int nempty = get_empties(board, empties);
      if (nempty == 0)
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

        /* Compute topological features */
        int xg = count_groups(board, PLAYER_X);
        int xd = shortest_edge_distance(board, PLAYER_X);
        int og = count_groups(board, PLAYER_O);
        int od = shortest_edge_distance(board, PLAYER_O);
        int xb = count_bridges(board, PLAYER_X);

        char player_prompt[256];
        snprintf(player_prompt, sizeof(player_prompt),
                 "board=%s|xg=%d|xd=%d|og=%d|od=%d|xb=%d", board_str, xg, xd,
                 og, od, xb);

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

        if (pr < 0 || pr >= GRID || pc < 0 || pc >= GRID ||
            board[cell(pr, pc)] != EMPTY) {
          total_parse_errors++;
          pr = empties[0][0];
          pc = empties[0][1];
          kb.stalls++;
        }

        /* Topological Judge: reject isolated placements */
        if (nempty > 5 && xg > 0 &&
            !has_friendly_neighbour(board, pr, pc, PLAYER_X)) {
          /* Move would create a disconnected stone — find a connected
           * alternative */
          for (int i = 0; i < nempty; i++) {
            if (has_friendly_neighbour(board, empties[i][0], empties[i][1],
                                       PLAYER_X)) {
              pr = empties[i][0];
              pc = empties[i][1];
              break;
            }
          }
          kb.stalls++;
        }

        board[cell(pr, pc)] = PLAYER_X;
        moves++;
        char ms[16];
        snprintf(ms, sizeof(ms), "R%dC%d", pr, pc);
        opa_kanban_add_last(&kb, ms);
        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;

        if (check_connection(board, PLAYER_X)) {
          winner = PLAYER_X;
          break;
        }
      } else {
        int ri = rand_r(&game_seed) % nempty;
        board[cell(empties[ri][0], empties[ri][1])] = PLAYER_O;
        moves++;
        if (check_connection(board, PLAYER_O)) {
          winner = PLAYER_O;
          break;
        }
      }

      turn = (turn == PLAYER_X) ? PLAYER_O : PLAYER_X;
    }

    total_moves += moves;

    if (winner == PLAYER_X) {
      total_wins++;
      if (gi < 15 || (gi + 1) % 10 == 0)
        printf("   X wins in %d moves!\n", moves);
    } else if (winner == PLAYER_O) {
      total_losses++;
      if (gi < 15 || (gi + 1) % 10 == 0)
        printf("   O wins in %d moves\n", moves);
    } else {
      total_draws++;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double pt =
      (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

  printf(
      "\n================================================================\n");
  printf("                    HEX 7x7 RESULTS\n");
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
