/*
 * MicroGPT-C — Training Optimisation Benchmark
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Quick A/B benchmark for comparing training optimisation techniques.
 * Trains on a text corpus and reports loss at regular intervals.
 * Uses organelle_train() for consistency with game experiments.
 *
 * Build variants are created via CMake with different -D flags:
 *   optim_bench_baseline    - no optimisations (WEIGHT_DECAY=0 GRAD_CLIP=0)
 *   optim_bench_gradclip    - GRAD_CLIP=1.0
 *   optim_bench_weightdecay - WEIGHT_DECAY=0.01
 *   optim_bench_adamw_tuned - BETA1=0.9 BETA2=0.999 LEARNING_RATE=0.0006
 *   optim_bench_labelsmooth - LABEL_SMOOTH=0.1
 *   optim_bench_combined    - all tier-1 optimisations
 *
 * Usage:
 *   cmake --build build --target optim_bench_baseline
 *   ./build/optim_bench_baseline
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Configuration ---- */
#ifndef BENCH_CORPUS
#define BENCH_CORPUS "c_tictactoe_player.txt"
#endif

#define PLAYER_CKPT "c_optim_bench.ckpt"
#define ORGANELLE_TEMP 0.2
#define INF_GEN_LEN 60

/* File-scoped runtime config */
static MicrogptConfig g_cfg;

int main(void) {
  setbuf(stdout, NULL);
  seed_rng(42);

  /* Runtime configuration — matches tictactoe game experiment */
  g_cfg = microgpt_default_config();
  g_cfg.n_embd = N_EMBD;
  g_cfg.n_head = N_HEAD;
  g_cfg.mlp_dim = MLP_DIM;
  g_cfg.n_layer = N_LAYER;
  g_cfg.block_size = BLOCK_SIZE;
  g_cfg.batch_size = BATCH_SIZE;
  g_cfg.num_steps = NUM_STEPS;
  g_cfg.learning_rate = LEARNING_RATE;
  g_cfg.max_vocab = MAX_VOCAB;
  g_cfg.max_docs = MAX_DOCS;
  g_cfg.max_doc_len = MAX_DOC_LEN;
  microgpt_print_config("MicroGPT-C - Optimisation Benchmark", &g_cfg);

  /* Print optimisation flags */
  printf("  [Optimisation Flags]\n");
  printf("    WEIGHT_DECAY = %.4f\n", (double)WEIGHT_DECAY);
  printf("    GRAD_CLIP    = %.1f\n", (double)GRAD_CLIP);
  printf("    LABEL_SMOOTH = %.2f\n", (double)LABEL_SMOOTH);
  printf("    BETA1        = %.3f\n", (double)BETA1);
  printf("    BETA2        = %.4f\n", (double)BETA2);
  printf("\n");

  /* Remove any old checkpoint so we always train fresh */
  remove(PLAYER_CKPT);

  /* Train organelle */
  int train_steps = g_cfg.num_steps;
  printf("--- TRAINING (%d steps) ---\n", train_steps);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  Organelle *player =
      organelle_train("Player", BENCH_CORPUS, PLAYER_CKPT, &g_cfg, train_steps);
  if (!player) {
    fprintf(stderr, "FATAL: training failed\n");
    return 1;
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("\nTraining complete in %.1fs\n", elapsed);

  /* Quick inference test — generate a few samples to verify quality */
  printf("\n--- INFERENCE SAMPLES ---\n");
  const char *prompts[] = {
      "board=_________",
      "board=X___O____",
      "board=XO_X_____",
  };
  int n_prompts = 3;

  for (int i = 0; i < n_prompts; i++) {
    char output[INF_GEN_LEN + 1];
    organelle_generate(player, &g_cfg, prompts[i], output, INF_GEN_LEN,
                       ORGANELLE_TEMP);
    printf("  prompt: %s\n  output: %s\n\n", prompts[i], output);
  }

  printf("================================================================\n");

  organelle_free(player);
  return 0;
}
