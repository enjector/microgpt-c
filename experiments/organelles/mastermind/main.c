/*
 * MicroGPT-C — Mastermind Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner on Mastermind:
 *   - 4 pegs, 6 colours (A-F)
 *   - Secret code: random 4-char string from {A,B,C,D,E,F}
 *   - Feedback: Black pegs (exact match), White pegs (colour match)
 *   - Two neural organelles (Planner + Player) coordinate via pipe-separated
 *     flat strings with kanban state.
 *   - Judge is fully deterministic (compute B/W score).
 *
 * Architecture: same as tictactoe/connect4 (n_embd=96, n_layer=4, ~460K
 * params).
 *
 * Pipeline: Planner -> Player -> Judge(deterministic) -> repeat
 *
 * Build:
 *   cmake --build build --target mastermind_demo
 *   ./build/mastermind_demo
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
#define PLANNER_CORPUS "mastermind_planner.txt"
#define PLAYER_CORPUS "mastermind_player.txt"

#define PLANNER_CKPT "mastermind_planner.ckpt"
#define PLAYER_CKPT "mastermind_player.ckpt"

#define ORGANELLE_TEMP 0.2
#define INF_GEN_LEN 60

#define NUM_TEST_GAMES 100
#define REPLAN_THRESHOLD 3
#define MAX_LAST_HISTORY 3
#define ENSEMBLE_VOTES 3
#define MAX_GUESSES 10

/* ---- Game Constants ---- */
#define NUM_PEGS 4
#define NUM_COLOURS 6
#define COLOUR_BASE 'A' /* A-F */

/* ---- File-scoped runtime config ---- */
static MicrogptConfig g_cfg;

/* ---- Game Helpers ---- */

static void compute_score(const char *guess, const char *secret, int *black,
                          int *white) {
  *black = 0;
  *white = 0;

  int guess_counts[NUM_COLOURS] = {0};
  int secret_counts[NUM_COLOURS] = {0};

  for (int i = 0; i < NUM_PEGS; i++) {
    if (guess[i] == secret[i]) {
      (*black)++;
    } else {
      guess_counts[guess[i] - COLOUR_BASE]++;
      secret_counts[secret[i] - COLOUR_BASE]++;
    }
  }

  for (int c = 0; c < NUM_COLOURS; c++) {
    *white += (guess_counts[c] < secret_counts[c]) ? guess_counts[c]
                                                   : secret_counts[c];
  }
}

static int is_valid_guess(const char *guess) {
  if (strlen(guess) < (size_t)NUM_PEGS)
    return 0;
  for (int i = 0; i < NUM_PEGS; i++) {
    if (guess[i] < COLOUR_BASE || guess[i] >= COLOUR_BASE + NUM_COLOURS)
      return 0;
  }
  return 1;
}

static void generate_secret(char *secret, unsigned int *seed) {
  for (int i = 0; i < NUM_PEGS; i++) {
    secret[i] = COLOUR_BASE + (char)(rand_r(seed) % NUM_COLOURS);
  }
  secret[NUM_PEGS] = '\0';
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
  g_cfg.max_docs = 10000;
  g_cfg.max_doc_len = 128;
  microgpt_print_config("MicroGPT-C - Mastermind Kanban Pipeline Demo", &g_cfg);

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
   * PHASE 2: Pipeline — Play Mastermind Games
   * ================================================================ */

  printf("\n--- PHASE 2: KANBAN PIPELINE EXECUTION ---\n");
  printf("Playing %d Mastermind games...\n\n", NUM_TEST_GAMES);

  int total_solved = 0;
  int total_guesses = 0;
  int total_valid_guesses = 0;
  int total_invalid_guesses = 0;
  int total_parse_errors = 0;
  int total_replans = 0;

  /* Buckets by solve count */
  int solved_in[MAX_GUESSES + 1];
  memset(solved_in, 0, sizeof(solved_in));

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  unsigned int game_seed = 12345;

  for (int gi = 0; gi < NUM_TEST_GAMES; gi++) {
    char secret[NUM_PEGS + 1];
    generate_secret(secret, &game_seed);

    OpaKanban kb;
    opa_kanban_init(&kb, MAX_LAST_HISTORY);

    /* Build feedback history string */
    char feedback_history[256] = "none";
    int guesses_made = 0;
    int solved = 0;

    if (gi < 15 || (gi + 1) % 10 == 0) {
      printf("-- Game %d/%d (secret=%s) --\n", gi + 1, NUM_TEST_GAMES, secret);
    }

    /* Step 1: Ask Planner for initial plan */
    char planner_prompt[128];
    snprintf(planner_prompt, sizeof(planner_prompt), "guesses=0|feedback=none");

    char plan_output[INF_GEN_LEN + 1];
    organelle_generate(planner, &g_cfg, planner_prompt, plan_output,
                       INF_GEN_LEN, ORGANELLE_TEMP);

    if (!opa_pipe_starts_with(plan_output, "todo=")) {
      total_parse_errors++;
    }

    /* Step 2: Play the game */
    while (!solved && guesses_made < MAX_GUESSES) {
      /* Re-plan if stalled */
      if (kb.stalls >= REPLAN_THRESHOLD && kb.replans < 3) {
        kb.replans++;
        total_replans++;

        char replan_prompt[256];
        snprintf(replan_prompt, sizeof(replan_prompt),
                 "guesses=%d|feedback=%s|stalled", guesses_made,
                 feedback_history);

        char replan_output[INF_GEN_LEN + 1];
        organelle_generate(planner, &g_cfg, replan_prompt, replan_output,
                           INF_GEN_LEN, ORGANELLE_TEMP);

        opa_kanban_clear_blocked(&kb);
        kb.stalls = 0;
      }

      /* Build Player prompt */
      char player_prompt[256];
      if (kb.blocked[0] != '\0') {
        snprintf(player_prompt, sizeof(player_prompt), "feedback=%s|blocked=%s",
                 feedback_history, kb.blocked);
      } else {
        snprintf(player_prompt, sizeof(player_prompt), "feedback=%s",
                 feedback_history);
      }

      /* Truncate to fit block_size */
      if (strlen(player_prompt) > 100)
        player_prompt[100] = '\0';

      /* Generate guess via ensemble voting */
      char move_output[INF_GEN_LEN + 1];
      scalar_t vote_conf = 0;
      organelle_generate_ensemble(player, &g_cfg, player_prompt, move_output,
                                  INF_GEN_LEN, ENSEMBLE_VOTES, ORGANELLE_TEMP,
                                  &vote_conf);

      /* Validate guess */
      char guess[NUM_PEGS + 1];
      memset(guess, 0, sizeof(guess));

      if (strlen(move_output) >= (size_t)NUM_PEGS) {
        memcpy(guess, move_output, NUM_PEGS);
        guess[NUM_PEGS] = '\0';
      }

      if (!is_valid_guess(guess)) {
        total_parse_errors++;
        /* Fallback: generate random guess */
        for (int i = 0; i < NUM_PEGS; i++) {
          guess[i] = COLOUR_BASE + (char)(rand_r(&game_seed) % NUM_COLOURS);
        }
        guess[NUM_PEGS] = '\0';
      }

      /* Compute score */
      int black, white;
      compute_score(guess, secret, &black, &white);
      guesses_made++;
      total_valid_guesses++;

      /* Update feedback history */
      if (strcmp(feedback_history, "none") == 0) {
        snprintf(feedback_history, sizeof(feedback_history), "%s:B%dW%d", guess,
                 black, white);
      } else {
        size_t len = strlen(feedback_history);
        snprintf(feedback_history + len, sizeof(feedback_history) - len,
                 ",%s:B%dW%d", guess, black, white);
      }

      opa_kanban_add_last(&kb, guess);

      if (black == NUM_PEGS) {
        solved = 1;
        if (gi < 15 || (gi + 1) % 10 == 0) {
          printf("   Solved in %d guesses! (%s)\n", guesses_made, secret);
        }
      } else {
        if (gi < 15 || (gi + 1) % 10 == 0) {
          printf("   Guess %d: %s → B%dW%d\n", guesses_made, guess, black,
                 white);
        }
      }
    }

    total_guesses += guesses_made;

    if (solved) {
      total_solved++;
      if (guesses_made <= MAX_GUESSES) {
        solved_in[guesses_made]++;
      }
    } else {
      if (gi < 15 || (gi + 1) % 10 == 0) {
        printf("   Not solved in %d guesses (secret=%s)\n", MAX_GUESSES,
               secret);
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
  printf("                    MASTERMIND RESULTS\n");
  printf("================================================================\n");
  printf("Games solved:       %d / %d (%.0f%%)\n", total_solved, NUM_TEST_GAMES,
         NUM_TEST_GAMES > 0 ? 100.0 * total_solved / NUM_TEST_GAMES : 0.0);
  printf("Avg guesses:        %.1f (solved games)\n",
         total_solved > 0 ? (double)total_guesses / total_solved : 0.0);
  printf("Distribution:\n");
  for (int i = 1; i <= MAX_GUESSES; i++) {
    if (solved_in[i] > 0) {
      printf("  %d guesses:       %d games\n", i, solved_in[i]);
    }
  }
  printf("Valid guesses:      %d\n", total_valid_guesses);
  printf("Invalid guesses:    %d\n", total_invalid_guesses);
  printf("Parse errors:       %d\n", total_parse_errors);
  printf("Planner re-plans:   %d\n", total_replans);
  printf("Pipeline time:      %.2fs\n", pipeline_time);
  printf("================================================================\n");

  /* Cleanup */
  organelle_free(planner);
  organelle_free(player);

  return 0;
}
