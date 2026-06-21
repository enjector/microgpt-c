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
#define PLANNER_CORPUS "c_connect4_planner.txt"
#define PLAYER_CORPUS "c_connect4_player.txt"

#define PLANNER_CKPT "c_connect4_planner.ckpt"
#define PLAYER_CKPT "c_connect4_player.ckpt"

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

/* ============================================================================
 * E19 oracle-first probe (VibeThinker CLR study — see experiments/E18).
 *
 * The Connect-4 demo's Judge is structural only ("column valid + win/draw
 * check"). E18 showed re-ranking can't help unless a *semantic* (quality)
 * verifier exists. This probe builds the cheapest such verifier — a 1-ply
 * move-quality oracle — and measures, on the decisions where quality
 * actually matters, whether a good move is even present in a best-of-N
 * candidate pool (Oracle@N) versus whether the demo's ensemble pick is good
 * (baseline). If Oracle@N >> baseline, a quality-verifier + CLR re-rank has
 * real headroom; if Oracle@N == baseline, it's a generation ceiling (no fix
 * possible from re-ranking) — the same dichotomy E18 settled for wiring.
 *
 * Pure measurement, gated on the C4_ORACLE env var; zero default-behaviour
 * change.
 *
 * c4_classify_move: simulate `me` dropping at `col`. Returns
 *   2 = immediate win, 0 = leaves `opp` an immediate winning reply (blunder),
 *   1 = safe (neither), -1 = illegal column.
 * ============================================================================ */
static int c4_classify_move(const char *board, int col, char me, char opp) {
  char tmp[BOARD_SIZE + 1];
  memcpy(tmp, board, BOARD_SIZE);
  tmp[BOARD_SIZE] = '\0';
  if (drop_piece(tmp, col, me) < 0)
    return -1;
  if (check_winner(tmp) == me)
    return 2;
  for (int c = 0; c < BOARD_COLS; c++) {
    char t2[BOARD_SIZE + 1];
    memcpy(t2, tmp, BOARD_SIZE);
    t2[BOARD_SIZE] = '\0';
    if (drop_piece(t2, c, opp) < 0)
      continue;
    if (check_winner(t2) == opp)
      return 0; /* blunder: opp wins on the immediate reply */
  }
  return 1;
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

/* E13 Pathway B — minimal CLI to override the player corpus + ckpt path
 * and optionally skip planner re-training / game playback.  Default
 * (no flags) reproduces the original C demo's behaviour byte-for-byte.
 *
 * Flags:
 *   --player-corpus=PATH  override the player corpus path used by
 *                         organelle_train (default: c_connect4_player.txt).
 *                         Uses opa_load_docs_multiline — same loader that
 *                         the OQL C4 inference runtime uses, so vocab is
 *                         compatible by construction.
 *   --player-ckpt=PATH    override the saved player checkpoint path
 *                         (default: c_connect4_player.ckpt).
 *   --skip-planner-train  reuse the existing planner checkpoint instead
 *                         of re-training it (saves ~5min on the E13 run).
 *   --skip-play           train only; do not run the 100-game playback.
 *
 * No new build deps; no engine-surface change.  E13 §1.5 / T9 / T5 hold. */
typedef struct {
  const char *player_corpus;
  const char *player_ckpt;
  int skip_planner_train;
  int skip_play;
  int max_docs;       /* 0 = use compile-time default (5000) */
} C4Args;

static void parse_c4_args(int argc, char **argv, C4Args *a) {
  a->player_corpus = NULL;
  a->player_ckpt = NULL;
  a->skip_planner_train = 0;
  a->skip_play = 0;
  a->max_docs = 0;
  for (int i = 1; i < argc; i++) {
    const char *s = argv[i];
    if (!strncmp(s, "--player-corpus=", 16))      a->player_corpus = s + 16;
    else if (!strncmp(s, "--player-ckpt=", 14))   a->player_ckpt = s + 14;
    else if (!strncmp(s, "--max-docs=", 11))      a->max_docs = atoi(s + 11);
    else if (!strcmp(s, "--skip-planner-train")) a->skip_planner_train = 1;
    else if (!strcmp(s, "--skip-play"))           a->skip_play = 1;
    else if (!strcmp(s, "--help") || !strcmp(s, "-h")) {
      fprintf(stderr,
        "usage: %s [--player-corpus=PATH] [--player-ckpt=PATH]\n"
        "          [--max-docs=N] [--skip-planner-train] [--skip-play]\n"
        "  Default: trains planner+player from corpora next to the binary\n"
        "           and plays 100 games vs random.  E13 Pathway B uses the\n"
        "           --player-corpus / --player-ckpt flags to train the\n"
        "           distillation student.  --max-docs lifts the 5000-doc\n"
        "           cap so an augmented corpus's LLM moves AND baseline\n"
        "           records both fit in the training window.\n", argv[0]);
      exit(0);
    } else {
      fprintf(stderr, "warn: unknown arg '%s' (use --help)\n", s);
    }
  }
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  seed_rng(42);

  C4Args cli;
  parse_c4_args(argc, argv, &cli);
  const char *player_corpus = cli.player_corpus ? cli.player_corpus : PLAYER_CORPUS;
  const char *player_ckpt   = cli.player_ckpt   ? cli.player_ckpt   : PLAYER_CKPT;

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
  g_cfg.max_docs = (cli.max_docs > 0) ? cli.max_docs : 5000;
  g_cfg.max_doc_len = 128;
  microgpt_print_config("MicroGPT-C - Connect-4 Kanban Pipeline Demo", &g_cfg);

  /* Suppress unused function warnings */
  (void)print_board;

  /* ================================================================
   * PHASE 1: Train organelles
   * ================================================================ */

  int train_steps = g_cfg.num_steps;
  printf("--- PHASE 1: TRAINING (%d steps each) ---\n", train_steps);
  printf("  player corpus    : %s\n", player_corpus);
  printf("  player ckpt      : %s\n", player_ckpt);
  printf("  skip planner train: %s\n", cli.skip_planner_train ? "yes" : "no");
  printf("  skip play         : %s\n", cli.skip_play ? "yes" : "no");

  Organelle *planner = NULL;
  if (!cli.skip_planner_train) {
    planner = organelle_train("Planner", PLANNER_CORPUS, PLANNER_CKPT,
                              &g_cfg, train_steps);
    if (!planner) {
      fprintf(stderr, "FATAL: Planner training failed\n");
      return 1;
    }
  } else {
    printf("(planner training skipped — assumes %s already on disk)\n",
           PLANNER_CKPT);
  }

  Organelle *player = organelle_train("Player", player_corpus, player_ckpt,
                                      &g_cfg, train_steps);
  if (!player) {
    fprintf(stderr, "FATAL: Player training failed\n");
    return 1;
  }

  /* E13: optionally exit after training (the 100-game playback is owned
   * by ./oql_c4 run experiments/connect4_distilled.oql). */
  if (cli.skip_play) {
    printf("\n--- PHASE 1 complete; --skip-play set, exiting before playback ---\n");
    if (planner) organelle_free(planner);
    organelle_free(player);
    return 0;
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

  /* E19 oracle-first probe (gated on C4_ORACLE env var). */
  const int c4_oracle_on = (getenv("C4_ORACLE") != NULL);
  const int ORC_N = 16;            /* candidate pool size, matches wiring N */
  int orc_critical = 0;            /* decisions where 1-ply quality matters */
  int orc_baseline_good = 0;       /* ensemble pick was a good (non-blunder) move */
  int orc_pool_good = 0;           /* >=1 of ORC_N sampled candidates was good */

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

      /* E19 oracle-first probe: on decisions where 1-ply quality matters,
       * compare the demo's ensemble pick (baseline) to the best achievable
       * over an ORC_N candidate pool (oracle). board/proposed_col are the
       * pre-move state and the played move. */
      if (c4_oracle_on) {
        int legal[BOARD_COLS];
        int nlegal = get_valid_columns(board, legal);
        int has_win = 0, has_loss = 0, has_nonloss = 0;
        for (int i = 0; i < nlegal; i++) {
          int l = c4_classify_move(board, legal[i], PLAYER_X, PLAYER_O);
          if (l == 2) has_win = 1;
          else if (l == 0) has_loss = 1;
          if (l >= 1) has_nonloss = 1;
        }
        /* "critical" = the choice changes the 1-ply outcome: a win is
         * available (and >1 legal move), or some move loses-in-1 while
         * another does not. Non-critical decisions are excluded — every
         * legal move is equally (non-)blundering there. */
        int critical = (has_win && nlegal > 1) || (has_loss && has_nonloss);
        if (critical) {
          orc_critical++;
          /* good(col): if a win exists it must be taken; else any non-losing. */
          #define C4_GOOD(L) ((L) >= 0 && (has_win ? ((L) == 2) : ((L) >= 1)))
          int played_l = c4_classify_move(board, proposed_col, PLAYER_X, PLAYER_O);
          if (C4_GOOD(played_l)) orc_baseline_good++;
          int pool_good = 0;
          for (int v = 0; v < ORC_N && !pool_good; v++) {
            char mo[INF_GEN_LEN + 1];
            scalar_t t = (scalar_t)(0.20 + 0.05 * v); /* temp jitter, matches wiring */
            organelle_generate(player, &g_cfg, player_prompt, mo, INF_GEN_LEN, t);
            int col = (mo[0] >= '0' && mo[0] <= '6') ? mo[0] - '0' : -1;
            if (col < 0) continue;
            int isleg = 0;
            for (int j = 0; j < nlegal; j++) if (legal[j] == col) isleg = 1;
            if (!isleg) continue;
            int l = c4_classify_move(board, col, PLAYER_X, PLAYER_O);
            if (C4_GOOD(l)) pool_good = 1;
          }
          if (pool_good) orc_pool_good++;
          #undef C4_GOOD
        }
      }

      /* E11 T4 trace — log first N X-moves of first M games for token-
       * divergence comparison vs the OQL run.  Activate with
       *   OQL_TRACE_FIRST_N_MOVES=5 OQL_TRACE_GAMES=10
       * (env-vars shared with src/oql_runtime_games.c). */
      {
        const char *tn = getenv("OQL_TRACE_FIRST_N_MOVES");
        const char *tg = getenv("OQL_TRACE_GAMES");
        int max_moves = tn ? atoi(tn) : 0;
        int max_games = tg ? atoi(tg) : 0;
        if (max_moves > 0 && max_games > 0 &&
            game_idx < max_games && moves_made < max_moves) {
          fprintf(stderr, "CDEMO_TRACE game=%d move=%d col=%d "
                          "from_model=%d\n",
                  game_idx, moves_made, proposed_col, from_model);
        }
      }
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

  if (c4_oracle_on) {
    int d = orc_critical > 0 ? orc_critical : 1;
    printf("\n  --- E19 oracle-first probe (1-ply quality verifier, N=%d) ---\n",
           ORC_N);
    printf("Critical decisions (win-to-take / threat-to-block): %d\n",
           orc_critical);
    printf("Baseline good (ensemble pick non-blunder):  %d/%d (%.0f%%)\n",
           orc_baseline_good, orc_critical, 100.0 * orc_baseline_good / d);
    printf("Oracle@%-2d good (>=1 candidate non-blunder): %d/%d (%.0f%%)\n",
           ORC_N, orc_pool_good, orc_critical, 100.0 * orc_pool_good / d);
    printf("=> verifier-rerank headroom on critical decisions: %+.0f pp\n",
           100.0 * (orc_pool_good - orc_baseline_good) / d);
    printf("   (Oracle>>baseline => quality-verifier+CLR has room; "
           "Oracle==baseline => generation ceiling, like wiring/E18)\n");
    printf("================================================================\n");
  }

  /* Cleanup */
  if (planner) organelle_free(planner);
  organelle_free(player);

  return 0;
}
