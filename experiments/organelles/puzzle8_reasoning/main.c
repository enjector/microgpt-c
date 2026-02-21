/*
 * MicroGPT-C — 8-Puzzle with Reasoning Trace Capture
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Reasoning trace variant of the 8-puzzle OPA demo. Identical pipeline
 * to puzzle8_demo, but captures every decision via OpaTrace for later
 * use as training data (process retrieval vs answer retrieval).
 *
 *   5 neural organelles (Strategist, Greedy-Mover, Detour-Detector,
 *   Detour-Mover, Judge) collaborate via pipe-separated flat strings.
 *
 * Pipeline flow:
 *   1. Strategist: m=U,D,L,R|md=N → direction (priority hint)
 *   2. Detour-Detector: m=U,D,L,R|b=N → "g" or "d" (annotation)
 *   3. Greedy-Mover: m=U,D,L,R|b=N[|x=DIR] → direction
 *   4. Judge: deterministic apply_move() boundary check
 *   5. Oscillation breaker: detect A↔B cycles, force unexplored dir
 *   6. OpaTrace records outcome at each step
 *   7. Repeat until solved or max iterations reached
 *
 * Build:
 *   cmake --build build --target puzzle8_reasoning_demo
 *   ./build/puzzle8_reasoning_demo
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
#define STRATEGIST_CORPUS "puzzle8_strategist.txt"
#define MOVER_CORPUS_STD "puzzle8_mover.txt"
#define MOVER_CORPUS_ENRICHED "puzzle8_mover_enriched.txt"
#define MOVER_CORPUS_COMBINED "puzzle8_mover_combined.txt"
#define DETECTOR_CORPUS "puzzle8_detour_detector.txt"
#define DETOUR_MOVER_CORPUS "puzzle8_detour_mover.txt"
#define JUDGE_CORPUS "puzzle8_judge.txt"

/* Set USE_ENRICHED_CORPUS: 0=standard, 1=enriched-only, 2=combined */
#ifndef USE_ENRICHED_CORPUS
#define USE_ENRICHED_CORPUS 0
#endif

/* Set DISABLE_PIPELINE_ASSISTS=1 to remove scaffolding (Phase 5 test) */
#ifndef DISABLE_PIPELINE_ASSISTS
#define DISABLE_PIPELINE_ASSISTS 0
#endif

#define STRATEGIST_CKPT "puzzle8_strategist_v3b.ckpt"
#define MOVER_CKPT "puzzle8_mover_v3b.ckpt"
#define DETECTOR_CKPT "puzzle8_detector_v3b.ckpt"
#define DETOUR_MOVER_CKPT "puzzle8_detour_mover_v3b.ckpt"
#define JUDGE_CKPT "puzzle8_judge_v3b.ckpt"

#define ORGANELLE_TEMP 0.2 /* low temperature for reliable retrieval */
#define INF_GEN_LEN 80     /* max chars per organelle generation */

#define MAX_PIPELINE_ITERS 40 /* max moves attempted per puzzle */
#define NUM_EASY_PUZZLES 10   /* md 1-4 puzzles */
#define NUM_MEDIUM_PUZZLES 10 /* md 5-8 puzzles */
#define NUM_HARD_PUZZLES 10   /* md 9+ puzzles */
#define NUM_TEST_PUZZLES                                                       \
  (NUM_EASY_PUZZLES + NUM_MEDIUM_PUZZLES + NUM_HARD_PUZZLES)
#define REPLAN_THRESHOLD 6 /* stalls before clearing blocked */
#define ENSEMBLE_VOTES 3   /* worker votes per move (odd for tiebreak) */

/* Reasoning trace and enriched corpus output */
#define TRACE_OUTPUT_FILE "puzzle8_reasoning_traces.txt"
#define ENRICHED_CORPUS_FILE "puzzle8_mover_enriched.txt"

/* ---- File-scoped runtime config ---- */
static MicrogptConfig g_cfg;

/* ---- Forward declarations ---- */
static int apply_move(int *board, const char *dir);
static int manhattan_distance(const int *board);

/* Build comma-separated list of valid directions for the current board. */
static void get_valid_dirs(const int *board, char *out, size_t out_sz) {
  const char *dir_names[] = {"up", "down", "left", "right"};
  size_t pos = 0;
  int first = 1;
  for (int d = 0; d < 4; d++) {
    int test[9];
    memcpy(test, board, 9 * sizeof(int));
    if (apply_move(test, dir_names[d])) {
      if (!first && pos < out_sz) {
        int n = snprintf(out + pos, out_sz - pos, ",");
        if (n > 0)
          pos += (size_t)n;
      }
      if (pos < out_sz) {
        int n = snprintf(out + pos, out_sz - pos, "%s", dir_names[d]);
        if (n > 0)
          pos += (size_t)n;
      }
      first = 0;
    }
  }
  if (first)
    out[0] = '\0';
}

/* ---- Board Helpers ---- */

static int GOAL_BOARD[9] = {1, 2, 3, 4, 5, 6, 7, 8, 0};

static void board_to_str(const int *board, char *out) {
  for (int i = 0; i < 9; i++)
    out[i] = '0' + board[i];
  out[9] = '\0';
}

static void md_delta_str(const int *board, char *out, size_t out_sz) {
  /* Compute md after each possible move. 'x' means illegal.
   * Format: m=U,D,L,R  where values are the resulting md or 'x'. */
  const char *dir_names[] = {"up", "down", "left", "right"};
  size_t pos = 0;
  int n = snprintf(out + pos, out_sz - pos, "m=");
  if (n < 0 || (size_t)n >= out_sz - pos) {
    if (out_sz > 0) {
      out[out_sz - 1] = '\0';
    }
    return;
  }
  pos += (size_t)n;
  for (int d = 0; d < 4; d++) {
    if (d > 0) {
      n = snprintf(out + pos, out_sz - pos, ",");
      if (n < 0 || (size_t)n >= out_sz - pos) {
        if (out_sz > 0) {
          out[out_sz - 1] = '\0';
        }
        return;
      }
      pos += (size_t)n;
    }
    int test[9];
    memcpy(test, board, 9 * sizeof(int));
    if (apply_move(test, dir_names[d])) {
      n = snprintf(out + pos, out_sz - pos, "%d", manhattan_distance(test));
    } else {
      n = snprintf(out + pos, out_sz - pos, "x");
    }
    if (n < 0 || (size_t)n >= out_sz - pos) {
      if (out_sz > 0) {
        out[out_sz - 1] = '\0';
      }
      return;
    }
    pos += (size_t)n;
  }
}

static int find_blank(const int *board) {
  for (int i = 0; i < 9; i++)
    if (board[i] == 0)
      return i;
  return -1;
}

static int is_goal(const int *board) {
  return memcmp(board, GOAL_BOARD, sizeof(GOAL_BOARD)) == 0;
}

static int manhattan_distance(const int *board) {
  int dist = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] == 0)
      continue;
    int goal_pos = board[i] - 1;
    dist += abs(i / 3 - goal_pos / 3) + abs(i % 3 - goal_pos % 3);
  }
  return dist;
}

static int apply_move(int *board, const char *dir) {
  int blank = find_blank(board);
  int r = blank / 3, c = blank % 3;
  int nr = r, nc = c;

  if (strcmp(dir, "up") == 0)
    nr = r - 1;
  else if (strcmp(dir, "down") == 0)
    nr = r + 1;
  else if (strcmp(dir, "left") == 0)
    nc = c - 1;
  else if (strcmp(dir, "right") == 0)
    nc = c + 1;
  else
    return 0;

  if (nr < 0 || nr >= 3 || nc < 0 || nc >= 3)
    return 0;

  int ni = nr * 3 + nc;
  board[blank] = board[ni];
  board[ni] = 0;
  return 1;
}

static void scramble_puzzle(int *board, int n_moves, unsigned int *seed) {
  memcpy(board, GOAL_BOARD, sizeof(GOAL_BOARD));
  const char *dirs[] = {"up", "down", "left", "right"};

  for (int i = 0; i < n_moves; i++) {
    for (int attempt = 0; attempt < 20; attempt++) {
      int d = rand_r(seed) % 4;
      int test[9];
      memcpy(test, board, sizeof(GOAL_BOARD));
      if (apply_move(test, dirs[d])) {
        memcpy(board, test, sizeof(GOAL_BOARD));
        break;
      }
    }
  }
}

/* ---- Test Puzzle Generation (stratified by difficulty) ---- */

static void scramble_to_target_md(int *board, int target_md_min,
                                  int target_md_max, unsigned int *seed) {
  for (int attempts = 0; attempts < 500; attempts++) {
    int n_moves = 3 + (int)(rand_r(seed) % 25);
    scramble_puzzle(board, n_moves, seed);
    int md = manhattan_distance(board);
    if (md >= target_md_min && md <= target_md_max && !is_goal(board))
      return;
  }
  scramble_puzzle(board, target_md_min * 2, seed);
}

/* ---- Direction ID encoding (for cycle detector) ---- */

static int dir_to_id(const char *dir) {
  if (strcmp(dir, "up") == 0)
    return 0;
  if (strcmp(dir, "down") == 0)
    return 1;
  if (strcmp(dir, "left") == 0)
    return 2;
  if (strcmp(dir, "right") == 0)
    return 3;
  return -1;
}

/* ---- Main ---- */

int main(void) {
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
  g_cfg.max_docs = 60000;
  g_cfg.max_doc_len = 128;
  const char *mover_corpus = (USE_ENRICHED_CORPUS == 2) ? MOVER_CORPUS_COMBINED
                             : (USE_ENRICHED_CORPUS == 1)
                                 ? MOVER_CORPUS_ENRICHED
                                 : MOVER_CORPUS_STD;
  const char *mode_name = (USE_ENRICHED_CORPUS == 2)   ? "COMBINED"
                          : (USE_ENRICHED_CORPUS == 1) ? "ENRICHED"
                                                       : "STANDARD";
  printf("Mover corpus: %s (%s mode)\n", mover_corpus, mode_name);
  printf("Pipeline assists: %s\n",
         DISABLE_PIPELINE_ASSISTS ? "DISABLED (bare model)" : "ENABLED");
  microgpt_print_config("MicroGPT-C - 8-Puzzle Reasoning Trace Demo", &g_cfg);

  /* ================================================================
   * PHASE 1: Train organelles
   * ================================================================ */

  printf("--- PHASE 1: TRAINING (%d steps each) ---\n", g_cfg.num_steps);

  Organelle *strategist =
      organelle_train("Strategist", STRATEGIST_CORPUS, STRATEGIST_CKPT, &g_cfg,
                      g_cfg.num_steps);
  if (!strategist) {
    fprintf(stderr, "FATAL: Strategist training failed\n");
    return 1;
  }

  const char *mover_ckpt =
      (USE_ENRICHED_CORPUS == 2)   ? "puzzle8_mover_combined_v3b.ckpt"
      : (USE_ENRICHED_CORPUS == 1) ? "puzzle8_mover_enriched_v3b.ckpt"
                                   : MOVER_CKPT;
  Organelle *mover = organelle_train("Greedy-Mover", mover_corpus, mover_ckpt,
                                     &g_cfg, g_cfg.num_steps);
  if (!mover) {
    fprintf(stderr, "FATAL: Greedy-Mover training failed\n");
    return 1;
  }

  Organelle *detector = organelle_train("Detour-Detector", DETECTOR_CORPUS,
                                        DETECTOR_CKPT, &g_cfg, g_cfg.num_steps);
  if (!detector) {
    fprintf(stderr, "FATAL: Detour-Detector training failed\n");
    return 1;
  }

  Organelle *detour_mover =
      organelle_train("Detour-Mover", DETOUR_MOVER_CORPUS, DETOUR_MOVER_CKPT,
                      &g_cfg, g_cfg.num_steps);
  if (!detour_mover) {
    fprintf(stderr, "FATAL: Detour-Mover training failed\n");
    return 1;
  }

  Organelle *judge = organelle_train("Judge", JUDGE_CORPUS, JUDGE_CKPT, &g_cfg,
                                     g_cfg.num_steps);
  if (!judge) {
    fprintf(stderr, "FATAL: Judge training failed\n");
    return 1;
  }

  /* ================================================================
   * PHASE 2: Pipeline — Solve Stratified Test Puzzles
   * ================================================================ */

  printf("\n--- PHASE 2: DUAL-MODEL PIPELINE (greedy + detour) ---\n");
  printf("Solving %d test puzzles (easy/medium/hard)...\n\n", NUM_TEST_PUZZLES);

  const char *band_names[] = {"EASY", "MEDIUM", "HARD"};
  int band_counts[] = {NUM_EASY_PUZZLES, NUM_MEDIUM_PUZZLES, NUM_HARD_PUZZLES};
  int band_md_min[] = {1, 5, 9};
  int band_md_max[] = {4, 8, 20};
  int band_solved[] = {0, 0, 0};
  int band_moves[] = {0, 0, 0};

  int total_solved = 0;
  int total_moves = 0;
  int total_accepts = 0;
  int total_rejects = 0;
  int total_parse_errors = 0;
  int total_detour_uses = 0;
  int total_greedy_uses = 0;
  int total_cycle_breaks = 0;
  int total_traces_written = 0;
  int total_recovery_traces = 0;
  int corpus_lines_written = 0;

  /* Open enriched corpus file for writing (Phase 2 generation) */
  FILE *corpus_fp = NULL;
  if (!USE_ENRICHED_CORPUS) {
    corpus_fp = fopen(ENRICHED_CORPUS_FILE, "w");
    if (!corpus_fp)
      fprintf(stderr, "WARNING: Could not open %s for writing\n",
              ENRICHED_CORPUS_FILE);
    else
      printf("Enriched corpus: writing to %s\n", ENRICHED_CORPUS_FILE);
  }

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  unsigned int puzzle_seed = 99999;
  int puzzle_global_idx = 0;

  for (int band = 0; band < 3; band++) {
    printf("=== %s band (md %d-%d) ===\n", band_names[band], band_md_min[band],
           band_md_max[band]);

    for (int pi = 0; pi < band_counts[band]; pi++) {
      puzzle_global_idx++;
      int board[9];
      scramble_to_target_md(board, band_md_min[band], band_md_max[band],
                            &puzzle_seed);

      char board_str[16];
      board_to_str(board, board_str);
      char dstr[64];
      md_delta_str(board, dstr, sizeof(dstr));
      int initial_md = manhattan_distance(board);

      printf("-- Puzzle %d/%d [%s] --\n", puzzle_global_idx, NUM_TEST_PUZZLES,
             band_names[band]);
      printf("   Board: %s  Disp: %s  (md=%d)\n", board_str, dstr, initial_md);

      /* Step 1: Ask Strategist for priority direction */
      char strat_prompt[128];
      snprintf(strat_prompt, sizeof(strat_prompt), "%s|md=%d", dstr,
               initial_md);

      char strat_output[INF_GEN_LEN + 1];
      organelle_generate(strategist, &g_cfg, strat_prompt, strat_output,
                         INF_GEN_LEN, ORGANELLE_TEMP);
      printf("   Strategist -> \"%s\"\n", strat_output);

      /* Step 2: Pipeline loop — Mover + Judge + Oscillation Breaker */
      OpaKanban kb;
      opa_kanban_init(&kb, 4); /* track last 4 moves for trace context */
      OpaCycleDetector cd;
      opa_cycle_init(&cd);

      /* Reasoning trace: capture every pipeline decision */
      OpaTrace trace;
      opa_trace_init(&trace, initial_md);

      int moves_made = 0;
      int solved = 0;
      int last_best_md = initial_md;
      int from_model = 1; /* tracks whether current move is model-sourced */

      for (int iter = 0; iter < MAX_PIPELINE_ITERS && !solved; iter++) {
        md_delta_str(board, dstr, sizeof(dstr));
        int blank = find_blank(board);
        int md_before = manhattan_distance(board);

        /* Clear blocked if stalled too long (replan) */
        if (!DISABLE_PIPELINE_ASSISTS && kb.stalls >= REPLAN_THRESHOLD) {
          int md_now = manhattan_distance(board);
          opa_trace_record(&trace, "replan", OPA_STEP_REPLAN, md_now, md_now,
                           kb.blocked, 0);
          opa_kanban_clear_blocked(&kb);
          kb.stalls = 0;
        }

        /* Ask Detour Detector: greedy or detour? (annotation only) */
        char det_prompt[128];
        snprintf(det_prompt, sizeof(det_prompt), "%s|b=%d", dstr, blank);
        char det_output[INF_GEN_LEN + 1];
        organelle_generate(detector, &g_cfg, det_prompt, det_output,
                           INF_GEN_LEN, ORGANELLE_TEMP);
        int is_detour = (det_output[0] == 'd');
        if (is_detour)
          total_detour_uses++;
        else
          total_greedy_uses++;

        /* Build Mover prompt with MD-delta encoding + valid directions */
        char valid_dirs[32];
        get_valid_dirs(board, valid_dirs, sizeof(valid_dirs));
        char mover_prompt[128];
        if (!DISABLE_PIPELINE_ASSISTS && kb.blocked[0] != '\0') {
          snprintf(mover_prompt, sizeof(mover_prompt), "%s|b=%d|valid=%s|x=%s",
                   dstr, blank, valid_dirs, kb.blocked);
        } else {
          snprintf(mover_prompt, sizeof(mover_prompt), "%s|b=%d|valid=%s", dstr,
                   blank, valid_dirs);
        }

        /* Generate move via ensemble voting */
        char move_output[INF_GEN_LEN + 1];
        scalar_t vote_conf = 0;
        organelle_generate_ensemble(mover, &g_cfg, mover_prompt, move_output,
                                    INF_GEN_LEN, ENSEMBLE_VOTES, ORGANELLE_TEMP,
                                    &vote_conf);

        /* Parse direction */
        const char *dir = NULL;
        if (strncmp(move_output, "up", 2) == 0)
          dir = "up";
        else if (strncmp(move_output, "down", 4) == 0)
          dir = "down";
        else if (strncmp(move_output, "left", 4) == 0)
          dir = "left";
        else if (strncmp(move_output, "right", 5) == 0)
          dir = "right";

        /* Validate against valid directions list */
        if (dir && !opa_valid_filter(dir, valid_dirs)) {
          dir = NULL;
        }

        if (!dir) {
          printf("   [%d] Mover -> \"%s\" (PARSE ERROR)\n", iter + 1,
                 move_output);
          total_parse_errors++;
          from_model = 0; /* switching to fallback */
          /* Use opa_valid_fallback for first valid non-blocked direction */
          char fb[16];
          if (opa_valid_fallback(&kb, valid_dirs, fb, sizeof(fb))) {
            if (strcmp(fb, "up") == 0)
              dir = "up";
            else if (strcmp(fb, "down") == 0)
              dir = "down";
            else if (strcmp(fb, "left") == 0)
              dir = "left";
            else if (strcmp(fb, "right") == 0)
              dir = "right";
          }
          if (!dir) {
            opa_trace_record(&trace, move_output, OPA_STEP_REJECTED, md_before,
                             -1, kb.blocked, 0);
            kb.stalls++;
            continue;
          }
        } else {
          from_model = 1; /* model produced a valid parse */
        }

        /* ---- Oscillation Breaker (disabled in bare mode) ---- */
        int dir_id = dir_to_id(dir);
        if (!DISABLE_PIPELINE_ASSISTS && opa_cycle_detected(&cd, dir_id)) {
          int cycle_a = dir_id;
          int cycle_b = opa_cycle_other(&cd, dir_id);
          const char *dir_names[] = {"up", "down", "left", "right"};

          const char *best_alt = NULL;
          int best_alt_md = 999;
          for (int d = 0; d < 4; d++) {
            if (d == cycle_a || d == cycle_b)
              continue;
            int test[9];
            memcpy(test, board, sizeof(board));
            if (apply_move(test, dir_names[d])) {
              int alt_md = manhattan_distance(test);
              if (alt_md < best_alt_md) {
                best_alt_md = alt_md;
                best_alt = dir_names[d];
              }
            }
          }
          if (best_alt) {
            printf("   [%d] CYCLE DETECTED (%s<->%s) -> forcing %s\n", iter + 1,
                   dir_names[cycle_a], dir_names[cycle_b], best_alt);
            opa_trace_record(&trace, dir, OPA_STEP_CYCLE_BREAK, md_before,
                             best_alt_md, kb.blocked, from_model);
            dir = best_alt;
            dir_id = dir_to_id(dir);
            total_cycle_breaks++;
            from_model = 0; /* cycle breaker overrode the model */
          }
        }

        /* Deterministic Judge: try the move */
        int test_board[9];
        memcpy(test_board, board, sizeof(board));
        int is_valid = apply_move(test_board, dir);

        if (is_valid) {
          memcpy(board, test_board, sizeof(board));
          moves_made++;
          int md_after = manhattan_distance(board);
          char arrow =
              md_after < md_before ? '+' : (md_after > md_before ? '-' : '=');

          opa_kanban_clear_blocked(&kb);
          if (md_after < last_best_md) {
            kb.stalls = 0;
            last_best_md = md_after;
          } else {
            kb.stalls++;
          }
          total_accepts++;

          /* Record in cycle detector */
          if (dir_id >= 0)
            opa_cycle_record(&cd, dir_id);

          /* Reasoning trace: ACCEPTED if progress, STALL if no progress */
          OpaStepOutcome step_outcome =
              (md_after < md_before) ? OPA_STEP_ACCEPTED : OPA_STEP_STALL;
          opa_trace_record(&trace, dir, step_outcome, md_before, md_after,
                           kb.blocked, from_model);

          /* Write enriched corpus entry (Phase 2 generation) */
          if (corpus_fp && from_model) {
            if (kb.last[0] != '\0') {
              fprintf(corpus_fp, "%s|stalls=%d|last=%s\n%s\n\n", mover_prompt,
                      kb.stalls, kb.last, dir);
            } else {
              fprintf(corpus_fp, "%s|stalls=%d\n%s\n\n", mover_prompt,
                      kb.stalls, dir);
            }
            corpus_lines_written++;
          }

          /* Record move in Kanban history (for next iteration's context) */
          opa_kanban_add_last(&kb, dir);

          printf("   [%d] %s move=%s %c (md: %d->%d)\n", iter + 1,
                 is_detour ? "[D]" : "[G]", dir, arrow, md_before, md_after);
        } else {
          /* Reasoning trace: REJECTED (out of bounds) */
          opa_trace_record(&trace, dir, OPA_STEP_REJECTED, md_before, -1,
                           kb.blocked, from_model);
          if (!DISABLE_PIPELINE_ASSISTS)
            opa_kanban_add_blocked(&kb, dir);
          kb.stalls++;
          total_rejects++;
          printf("   [%d] move=%s -> OUT OF BOUNDS [blocked:%s]\n", iter + 1,
                 dir, kb.blocked);
        }

        if (is_goal(board)) {
          solved = 1;
        }
      }

      board_to_str(board, board_str);
      int final_md = manhattan_distance(board);
      total_moves += moves_made;

      /* Finalise and write reasoning trace */
      opa_trace_finalise(&trace, final_md, solved);
      opa_trace_write(&trace, TRACE_OUTPUT_FILE);
      total_traces_written++;
      if (opa_trace_has_recovery(&trace))
        total_recovery_traces++;

      if (solved) {
        total_solved++;
        band_solved[band]++;
        band_moves[band] += moves_made;
        printf("   SOLVED in %d moves! (trace: %d steps, recovery=%s)\n\n",
               moves_made, trace.num_steps,
               opa_trace_has_recovery(&trace) ? "yes" : "no");
      } else {
        printf("   NOT SOLVED: board=%s md=%d (was %d) moves=%d "
               "(trace: %d steps)\n\n",
               board_str, final_md, initial_md, moves_made, trace.num_steps);
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &pipeline_end);
  double pipeline_time =
      (double)(pipeline_end.tv_sec - pipeline_start.tv_sec) +
      (double)(pipeline_end.tv_nsec - pipeline_start.tv_nsec) / 1e9;

  /* ================================================================
   * PHASE 3: Results — Generalisation Report
   * ================================================================ */

  printf("================================================================\n");
  printf("--- GENERALISATION RESULTS ---\n");
  printf(
      "================================================================\n\n");

  printf("Overall: %d / %d solved (%.0f%%)\n\n", total_solved, NUM_TEST_PUZZLES,
         NUM_TEST_PUZZLES > 0 ? 100.0 * total_solved / NUM_TEST_PUZZLES : 0.0);

  printf("%-8s  %-10s  %-10s  %-10s\n", "Band", "Solved", "Rate", "Avg Moves");
  printf("%-8s  %-10s  %-10s  %-10s\n", "--------", "----------", "----------",
         "----------");
  for (int b = 0; b < 3; b++) {
    double rate =
        band_counts[b] > 0 ? 100.0 * band_solved[b] / band_counts[b] : 0.0;
    double avg_moves =
        band_solved[b] > 0 ? (double)band_moves[b] / band_solved[b] : 0.0;
    printf("%-8s  %d/%-8d  %5.0f%%       %.1f\n", band_names[b], band_solved[b],
           band_counts[b], rate, avg_moves);
  }

  printf("\nTotal moves:      %d\n", total_moves);
  printf("Valid moves:      %d\n", total_accepts);
  printf("OOB rejections:   %d\n", total_rejects);
  printf("Parse errors:     %d\n", total_parse_errors);
  printf("Greedy dispatches:%d\n", total_greedy_uses);
  printf("Detour dispatches:%d\n", total_detour_uses);
  printf("Cycle breaks:     %d\n", total_cycle_breaks);
  printf("\nReasoning Traces:\n");
  printf("Traces written:   %d\n", total_traces_written);
  printf("Recovery traces:  %d (%.0f%%)\n", total_recovery_traces,
         total_traces_written > 0
             ? 100.0 * total_recovery_traces / total_traces_written
             : 0.0);
  printf("Trace file:       %s\n", TRACE_OUTPUT_FILE);
  if (corpus_fp) {
    fclose(corpus_fp);
    printf("Enriched corpus:  %d entries written to %s\n", corpus_lines_written,
           ENRICHED_CORPUS_FILE);
  }
  printf("Mode:             %s\n", mode_name);
  printf("Assists:          %s\n",
         DISABLE_PIPELINE_ASSISTS ? "DISABLED" : "ENABLED");
  printf("Pipeline time:    %.2fs\n", pipeline_time);
  printf("================================================================\n");

  /* Cleanup */
  organelle_free(strategist);
  organelle_free(mover);
  organelle_free(detector);
  organelle_free(detour_mover);
  organelle_free(judge);

  return 0;
}
