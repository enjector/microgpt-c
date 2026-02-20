/*
 * MicroGPT-C — EuroMillions Lottery Multi-Organelle Demo (Kanban Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner on lottery number suggestion:
 *   - 5 balls from 1-50, 2 lucky stars from 1-12
 *   - Two neural organelles (Analyser + Predictor) coordinate via
 *     flat strings; semicolon-separated output to avoid pipe confusion.
 *   - Validator is fully deterministic (range + uniqueness checks).
 *
 * Architecture: N_EMBD=48, N_HEAD=4, N_LAYER=3, ~92K params/organelle.
 *
 * Pipeline: Analyser → Predictor → Validator(deterministic) → output
 *
 * NOTE: Lottery draws are independent random events. This pipeline learns
 *       statistical patterns (frequency, recency) from historical data,
 *       producing statistically-weighted suggestions, not true predictions.
 *
 * Build:
 *   cmake --build build --target lottery_demo
 *   ./build/lottery_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Configuration ---- */
#define ANALYSER_CORPUS "lottery_analyser.txt"
#define PREDICTOR_CORPUS "lottery_predictor.txt"
#define CSV_FILE "euromillions-draw-history.csv"

#define ANALYSER_CKPT "lottery_analyser.ckpt"
#define PREDICTOR_CKPT "lottery_predictor.ckpt"

#define ORGANELLE_TEMP 0.8
#define INF_GEN_LEN 120

#define NUM_PREDICTIONS 7
#define MAX_RETRIES 15
#define ENSEMBLE_VOTES 5
#define BACKTEST_HOLDOUT 10

/* Fibonacci look-back windows for diverse prediction horizons */
static const int FIB_WINDOWS[] = {3, 5, 8, 13, 21, 34, 55};

/* ---- EuroMillions Constants ---- */
#define NUM_BALLS 5
#define MAX_BALL 50
#define NUM_STARS 2
#define MAX_STAR 12
#define MAX_DRAWS 2000

/* ---- File-scoped runtime config ---- */
static MicrogptConfig g_cfg;

/* ---- Draw storage ---- */
typedef struct {
  char date[32];
  int balls[NUM_BALLS];
  int stars[NUM_STARS];
} Draw;

/* ---- CSV Loader ---- */
static int load_draws_from_csv(const char *path, Draw *draws, int max_draws) {
  FILE *f = fopen(path, "r");
  if (!f)
    return 0;

  char line[1024];
  int count = 0;

  /* Skip header */
  if (!fgets(line, sizeof(line), f)) {
    fclose(f);
    return 0;
  }

  /* CSV is newest-first; load in file order */
  while (fgets(line, sizeof(line), f) && count < max_draws) {
    Draw *d = &draws[count];
    if (sscanf(line, "%31[^,],%d,%d,%d,%d,%d,%d,%d", d->date, &d->balls[0],
               &d->balls[1], &d->balls[2], &d->balls[3], &d->balls[4],
               &d->stars[0], &d->stars[1]) == 8) {
      count++;
    }
  }

  fclose(f);
  return count;
}

/* ---- Frequency Analysis ---- */

/**
 * Compute top-N most/least frequent numbers from a window of draws.
 * hot_out/cold_out must have space for top_n ints each.
 * Returns count written to hot_out.
 */
static int compute_hot_cold(const Draw *draws, int num_draws, int is_stars,
                            int *hot_out, int *cold_out, int top_n) {
  int max_val = is_stars ? MAX_STAR : MAX_BALL;
  int freq[MAX_BALL + 1];
  memset(freq, 0, sizeof(freq));

  for (int i = 0; i < num_draws; i++) {
    int count = is_stars ? NUM_STARS : NUM_BALLS;
    const int *nums = is_stars ? draws[i].stars : draws[i].balls;
    for (int j = 0; j < count; j++) {
      if (nums[j] >= 1 && nums[j] <= max_val)
        freq[nums[j]]++;
    }
  }

  /* Find top-N hot (highest frequency) */
  int used[MAX_BALL + 1];
  memset(used, 0, sizeof(used));
  int hot_count = 0;
  for (int k = 0; k < top_n && k < max_val; k++) {
    int best = -1, best_freq = -1;
    for (int n = 1; n <= max_val; n++) {
      if (!used[n] && freq[n] > best_freq) {
        best = n;
        best_freq = freq[n];
      }
    }
    if (best > 0) {
      hot_out[hot_count++] = best;
      used[best] = 1;
    }
  }

  /* Find top-N cold (lowest frequency) */
  memset(used, 0, sizeof(used));
  int cold_count = 0;
  for (int k = 0; k < top_n && k < max_val; k++) {
    int best = -1, best_freq = 999999;
    for (int n = 1; n <= max_val; n++) {
      if (!used[n] && freq[n] < best_freq) {
        best = n;
        best_freq = freq[n];
      }
    }
    if (best > 0) {
      cold_out[cold_count++] = best;
      used[best] = 1;
    }
  }

  return hot_count;
}

/* ---- Sort helper ---- */
static int cmp_int(const void *a, const void *b) {
  return *(const int *)a - *(const int *)b;
}

/**
 * Build a hot/cold analysis prompt string matching the corpus format:
 *   "hot=N,N,N,N,N|cold=N,N,N,N,N|stars_hot=N,N,N|stars_cold=N,N,N"
 */
static void build_analysis_prompt(const Draw *draws, int num_draws, char *buf,
                                  size_t buf_sz) {
  int hot_b[5], cold_b[5], hot_s[3], cold_s[3];
  int nh = compute_hot_cold(draws, num_draws, 0, hot_b, cold_b, 5);
  int ns = compute_hot_cold(draws, num_draws, 1, hot_s, cold_s, 3);

  /* Sort for consistency */
  qsort(hot_b, nh, sizeof(int), cmp_int);
  qsort(cold_b, 5, sizeof(int), cmp_int);
  qsort(hot_s, ns, sizeof(int), cmp_int);
  qsort(cold_s, 3, sizeof(int), cmp_int);

  int off = 0;
  off += snprintf(buf + off, buf_sz - (size_t)off, "hot=");
  for (int i = 0; i < nh && off < (int)buf_sz - 4; i++)
    off += snprintf(buf + off, buf_sz - (size_t)off, "%s%d", i ? "," : "",
                    hot_b[i]);

  off += snprintf(buf + off, buf_sz - (size_t)off, "|cold=");
  for (int i = 0; i < 5 && off < (int)buf_sz - 4; i++)
    off += snprintf(buf + off, buf_sz - (size_t)off, "%s%d", i ? "," : "",
                    cold_b[i]);

  off += snprintf(buf + off, buf_sz - (size_t)off, "|stars_hot=");
  for (int i = 0; i < ns && off < (int)buf_sz - 4; i++)
    off += snprintf(buf + off, buf_sz - (size_t)off, "%s%d", i ? "," : "",
                    hot_s[i]);

  off += snprintf(buf + off, buf_sz - (size_t)off, "|stars_cold=");
  for (int i = 0; i < 3 && off < (int)buf_sz - 4; i++)
    off += snprintf(buf + off, buf_sz - (size_t)off, "%s%d", i ? "," : "",
                    cold_s[i]);
}

/* Build a "window=N|recent=..." prompt matching the analyser corpus format */
static void build_analyser_prompt(const Draw *draws, int num_draws,
                                  int window_size, char *buf, size_t buf_sz) {
  int off = snprintf(buf, buf_sz, "window=%d|recent=", window_size);

  /* Format 3 most recent draws, most-recent first */
  int show = num_draws < 3 ? num_draws : 3;
  for (int i = 0; i < show && off < (int)buf_sz - 20; i++) {
    const Draw *d = &draws[num_draws - 1 - i]; /* newest first */
    if (i > 0 && off < (int)buf_sz - 1)
      buf[off++] = ';';
    for (int j = 0; j < NUM_BALLS && off < (int)buf_sz - 4; j++)
      off += snprintf(buf + off, buf_sz - (size_t)off, "%s%d", j ? "," : "",
                      d->balls[j]);
    if (off < (int)buf_sz - 1)
      buf[off++] = '|';
    for (int j = 0; j < NUM_STARS && off < (int)buf_sz - 4; j++)
      off += snprintf(buf + off, buf_sz - (size_t)off, "%s%d", j ? "," : "",
                      d->stars[j]);
  }
  if (off < (int)buf_sz)
    buf[off] = '\0';

  /* Truncate to 110 chars (matching corpus gen) */
  if (strlen(buf) > 110)
    buf[110] = '\0';
}

/* ---- Validation Helpers ---- */

static int is_valid_ball(int n) { return n >= 1 && n <= MAX_BALL; }

static int is_valid_star(int n) { return n >= 1 && n <= MAX_STAR; }

static int has_duplicates(const int *arr, int len) {
  for (int i = 0; i < len; i++) {
    for (int j = i + 1; j < len; j++) {
      if (arr[i] == arr[j])
        return 1;
    }
  }
  return 0;
}

static int validate_prediction(const int *balls, const int *stars) {
  for (int i = 0; i < NUM_BALLS; i++) {
    if (!is_valid_ball(balls[i]))
      return 0;
  }
  for (int i = 0; i < NUM_STARS; i++) {
    if (!is_valid_star(stars[i]))
      return 0;
  }
  if (has_duplicates(balls, NUM_BALLS))
    return 0;
  if (has_duplicates(stars, NUM_STARS))
    return 0;
  return 1;
}

/* ---- Parsing Helpers ---- */

/**
 * Parse a prediction string like "5,17,23,29,41;3,10"
 * into balls[5] and stars[2] arrays.
 * Uses semicolon separator (not pipe) to avoid confusion with
 * the pipe-separated analysis prompt.
 * Returns 1 on success, 0 on failure.
 */
static int parse_prediction(const char *str, int *balls, int *stars) {
  const char *sep = strchr(str, ';');
  if (!sep)
    return 0;

  /* Parse balls (before separator) */
  int ball_count = 0;
  const char *p = str;
  while (p < sep && ball_count < NUM_BALLS) {
    while (p < sep && (*p < '0' || *p > '9'))
      p++;
    if (p >= sep)
      break;
    int n = 0;
    int digits = 0;
    while (p < sep && *p >= '0' && *p <= '9' && digits < 3) {
      n = n * 10 + (*p - '0');
      p++;
      digits++;
    }
    if (n >= 1 && n <= MAX_BALL) {
      balls[ball_count++] = n;
    } else {
      return 0; /* out-of-range number */
    }
    if (p < sep && *p == ',')
      p++;
  }

  if (ball_count != NUM_BALLS)
    return 0;

  /* Parse stars (after separator) */
  int star_count = 0;
  p = sep + 1;
  const char *end = str + strlen(str);
  while (p < end && star_count < NUM_STARS) {
    while (p < end && (*p < '0' || *p > '9'))
      p++;
    if (p >= end)
      break;
    int n = 0;
    int digits = 0;
    while (p < end && *p >= '0' && *p <= '9' && digits < 3) {
      n = n * 10 + (*p - '0');
      p++;
      digits++;
    }
    if (n >= 1 && n <= MAX_STAR) {
      stars[star_count++] = n;
    } else {
      return 0;
    }
    if (p < end && *p == ',')
      p++;
  }

  if (star_count != NUM_STARS)
    return 0;

  return 1;
}

/* ---- Random fallback ---- */
static void random_prediction(int *balls, int *stars, unsigned int *seed) {
  int used[MAX_BALL + 1];
  memset(used, 0, sizeof(used));
  for (int i = 0; i < NUM_BALLS; i++) {
    int n;
    do {
      n = 1 + (int)(rand_r(seed) % MAX_BALL);
    } while (used[n]);
    used[n] = 1;
    balls[i] = n;
  }

  int sused[MAX_STAR + 1];
  memset(sused, 0, sizeof(sused));
  for (int i = 0; i < NUM_STARS; i++) {
    int n;
    do {
      n = 1 + (int)(rand_r(seed) % MAX_STAR);
    } while (sused[n]);
    sused[n] = 1;
    stars[i] = n;
  }

  qsort(balls, NUM_BALLS, sizeof(int), cmp_int);
  qsort(stars, NUM_STARS, sizeof(int), cmp_int);
}

/* Count how many numbers in `predicted` appear in `actual` */
static int count_hits(const int *predicted, int pred_n, const int *actual,
                      int actual_n) {
  int hits = 0;
  for (int i = 0; i < pred_n; i++) {
    for (int j = 0; j < actual_n; j++) {
      if (predicted[i] == actual[j])
        hits++;
    }
  }
  return hits;
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
  g_cfg.block_size = BLOCK_SIZE;
  g_cfg.batch_size = BATCH_SIZE;
  g_cfg.num_steps = NUM_STEPS;
  g_cfg.learning_rate = LEARNING_RATE;
  g_cfg.max_vocab = MAX_VOCAB;
  g_cfg.max_docs = MAX_DOCS;
  g_cfg.max_doc_len = MAX_DOC_LEN;
  microgpt_print_config(
      "MicroGPT-C - EuroMillions Lottery Kanban Pipeline Demo", &g_cfg);

  /* ---- Load draw history from CSV ---- */
  Draw all_draws[MAX_DRAWS];
  int total_draws = load_draws_from_csv(CSV_FILE, all_draws, MAX_DRAWS);

  if (total_draws < 10) {
    fprintf(stderr, "FATAL: Need at least 10 draws, found %d\n", total_draws);
    return 1;
  }

  printf("\nLoaded %d draws from %s\n", total_draws, CSV_FILE);
  printf("Most recent: %s [%d,%d,%d,%d,%d | %d,%d]\n", all_draws[0].date,
         all_draws[0].balls[0], all_draws[0].balls[1], all_draws[0].balls[2],
         all_draws[0].balls[3], all_draws[0].balls[4], all_draws[0].stars[0],
         all_draws[0].stars[1]);
  printf(
      "Oldest:      %s [%d,%d,%d,%d,%d | %d,%d]\n\n",
      all_draws[total_draws - 1].date, all_draws[total_draws - 1].balls[0],
      all_draws[total_draws - 1].balls[1], all_draws[total_draws - 1].balls[2],
      all_draws[total_draws - 1].balls[3], all_draws[total_draws - 1].balls[4],
      all_draws[total_draws - 1].stars[0], all_draws[total_draws - 1].stars[1]);

  /* ================================================================
   * PHASE 1: Train organelles
   * ================================================================ */

  int train_steps = g_cfg.num_steps;
  printf("--- PHASE 1: TRAINING (%d steps each) ---\n", train_steps);

  Organelle *analyser = organelle_train("Analyser", ANALYSER_CORPUS,
                                        ANALYSER_CKPT, &g_cfg, train_steps);
  if (!analyser) {
    fprintf(stderr, "FATAL: Analyser training failed\n");
    return 1;
  }

  Organelle *predictor = organelle_train("Predictor", PREDICTOR_CORPUS,
                                         PREDICTOR_CKPT, &g_cfg, train_steps);
  if (!predictor) {
    fprintf(stderr, "FATAL: Predictor training failed\n");
    return 1;
  }

  /* ================================================================
   * PHASE 2: Backtest — predict each of the most recent N draws
   *          using only the draws that came before it
   * ================================================================ */

  printf("\n--- PHASE 2: BACKTEST (%d most recent draws) ---\n\n",
         BACKTEST_HOLDOUT);

  /*
   * The CSV is newest-first: all_draws[0] is the most recent.
   * For backtest draw i (0-indexed), the "history" is draws i+1..end
   * (everything older than the target draw).
   * We use that history to build the analyser/predictor prompts,
   * then check how many numbers the model gets right.
   */

  int bt_total_ball_hits = 0, bt_total_star_hits = 0;
  int bt_rand_ball_hits = 0, bt_rand_star_hits = 0;
  int bt_model_valid = 0;
  unsigned int bt_seed = 99999;

  struct timespec pipeline_start, pipeline_end;
  clock_gettime(CLOCK_MONOTONIC, &pipeline_start);

  for (int bt = 0; bt < BACKTEST_HOLDOUT && bt < total_draws - 10; bt++) {
    const Draw *target = &all_draws[bt];
    /* History = everything older than this draw */
    const Draw *history = &all_draws[bt + 1];
    int hist_len = total_draws - bt - 1;

    printf("  Backtest %d: %s → target [%d,%d,%d,%d,%d | %d,%d]\n", bt + 1,
           target->date, target->balls[0], target->balls[1], target->balls[2],
           target->balls[3], target->balls[4], target->stars[0],
           target->stars[1]);

    /* Step 1: Build analyser prompt from history window (matching corpus) */
    char analyser_prompt[256];
    int window = hist_len < 10 ? hist_len : 10;
    build_analyser_prompt(history, window, window, analyser_prompt,
                          sizeof(analyser_prompt));

    /* Step 2: Run Analyser */
    char analysis_output[INF_GEN_LEN + 1];
    organelle_generate(analyser, &g_cfg, analyser_prompt, analysis_output,
                       INF_GEN_LEN, ORGANELLE_TEMP);

    printf("    Analyser: \"%s\"\n", analysis_output);

    /* Step 3: Build predictor prompt — use analyser output if valid,
     *         otherwise compute hot/cold directly from history */
    char predictor_prompt[256];
    if (strstr(analysis_output, "hot=") && strstr(analysis_output, "cold=") &&
        strstr(analysis_output, "stars_hot=")) {
      snprintf(predictor_prompt, sizeof(predictor_prompt), "%s",
               analysis_output);
    } else {
      /* Analyser garbled — build prompt directly from history */
      build_analysis_prompt(history, window, predictor_prompt,
                            sizeof(predictor_prompt));
      printf("    (analyser garbled, using direct analysis)\n");
    }

    /* Truncate to fit block_size */
    if (strlen(predictor_prompt) > 100)
      predictor_prompt[100] = '\0';

    /* Step 4: Generate prediction with retries */
    int pred_balls[NUM_BALLS], pred_stars[NUM_STARS];
    int from_model = 0;

    for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
      char pred_output[INF_GEN_LEN + 1];
      scalar_t vote_conf = 0;
      organelle_generate_ensemble(predictor, &g_cfg, predictor_prompt,
                                  pred_output, INF_GEN_LEN, ENSEMBLE_VOTES,
                                  ORGANELLE_TEMP, &vote_conf);

      if (parse_prediction(pred_output, pred_balls, pred_stars) &&
          validate_prediction(pred_balls, pred_stars)) {
        from_model = 1;
        qsort(pred_balls, NUM_BALLS, sizeof(int), cmp_int);
        qsort(pred_stars, NUM_STARS, sizeof(int), cmp_int);
        printf("    Predictor (attempt %d): [%d,%d,%d,%d,%d | %d,%d] "
               "(conf=%.2f)\n",
               attempt + 1, pred_balls[0], pred_balls[1], pred_balls[2],
               pred_balls[3], pred_balls[4], pred_stars[0], pred_stars[1],
               (double)vote_conf);
        break;
      }

      if (attempt < 3 || attempt == MAX_RETRIES - 1) {
        printf("    Predictor (attempt %d): raw=\"%s\" — invalid\n",
               attempt + 1, pred_output);
      }
    }

    if (!from_model) {
      random_prediction(pred_balls, pred_stars, &bt_seed);
      printf("    → Fallback random: [%d,%d,%d,%d,%d | %d,%d]\n", pred_balls[0],
             pred_balls[1], pred_balls[2], pred_balls[3], pred_balls[4],
             pred_stars[0], pred_stars[1]);
    } else {
      bt_model_valid++;
    }

    int bh = count_hits(pred_balls, NUM_BALLS, target->balls, NUM_BALLS);
    int sh = count_hits(pred_stars, NUM_STARS, target->stars, NUM_STARS);
    bt_total_ball_hits += bh;
    bt_total_star_hits += sh;

    /* Random baseline comparison */
    int rballs[NUM_BALLS], rstars[NUM_STARS];
    random_prediction(rballs, rstars, &bt_seed);
    int rbh = count_hits(rballs, NUM_BALLS, target->balls, NUM_BALLS);
    int rsh = count_hits(rstars, NUM_STARS, target->stars, NUM_STARS);
    bt_rand_ball_hits += rbh;
    bt_rand_star_hits += rsh;

    printf("    Result: %d ball hit(s), %d star hit(s) | random: %d ball, "
           "%d star\n\n",
           bh, sh, rbh, rsh);
  }

  /* ================================================================
   * PHASE 3: Generate Fresh Predictions (next 2 draws)
   * ================================================================ */

  printf("--- PHASE 3: GENERATING %d PREDICTIONS (Fibonacci windows) ---\n\n",
         NUM_PREDICTIONS);

  /* Analysis from the full history for fresh predictions */
  char full_analysis[256];
  build_analysis_prompt(all_draws, total_draws < 10 ? total_draws : 10,
                        full_analysis, sizeof(full_analysis));
  printf("Full-history analysis: %s\n\n", full_analysis);

  int total_model_sourced = 0, total_fallbacks = 0;
  unsigned int pred_seed = 54321;

  for (int pi = 0; pi < NUM_PREDICTIONS; pi++) {
    int fib_win = FIB_WINDOWS[pi % 7];
    if (fib_win > total_draws)
      fib_win = total_draws;
    printf("--- Prediction %d/%d (window=%d draws) ---\n", pi + 1,
           NUM_PREDICTIONS, fib_win);

    /* Use Fibonacci window for analyser prompt */
    char a_prompt[256];
    build_analyser_prompt(all_draws,
                          total_draws < fib_win ? total_draws : fib_win,
                          fib_win, a_prompt, sizeof(a_prompt));

    char a_output[INF_GEN_LEN + 1];
    organelle_generate(analyser, &g_cfg, a_prompt, a_output, INF_GEN_LEN,
                       ORGANELLE_TEMP);
    printf("  Analyser: \"%s\"\n", a_output);

    /* Pick predictor prompt */
    char p_prompt[256];
    if (strstr(a_output, "hot=") && strstr(a_output, "cold=") &&
        strstr(a_output, "stars_hot=")) {
      snprintf(p_prompt, sizeof(p_prompt), "%s", a_output);
    } else {
      snprintf(p_prompt, sizeof(p_prompt), "%s", full_analysis);
      printf("  (using direct analysis)\n");
    }
    if (strlen(p_prompt) > 100)
      p_prompt[100] = '\0';

    /* Generate prediction */
    int balls[NUM_BALLS], stars[NUM_STARS];
    int from_model = 0;

    for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
      char pred_output[INF_GEN_LEN + 1];
      scalar_t vote_conf = 0;

      /* Vary temperature slightly per prediction set */
      scalar_t temp = ORGANELLE_TEMP + (scalar_t)pi * 0.05;
      organelle_generate_ensemble(predictor, &g_cfg, p_prompt, pred_output,
                                  INF_GEN_LEN, ENSEMBLE_VOTES, temp,
                                  &vote_conf);

      printf("  Attempt %d: raw=\"%s\" (conf=%.2f)\n", attempt + 1, pred_output,
             (double)vote_conf);

      if (parse_prediction(pred_output, balls, stars) &&
          validate_prediction(balls, stars)) {
        from_model = 1;
        qsort(balls, NUM_BALLS, sizeof(int), cmp_int);
        qsort(stars, NUM_STARS, sizeof(int), cmp_int);
        break;
      }
    }

    if (!from_model) {
      random_prediction(balls, stars, &pred_seed);
      total_fallbacks++;
      printf("  → Fallback to random\n");
    } else {
      total_model_sourced++;
    }

    printf("\n  ★ Prediction %d: Balls=[%d, %d, %d, %d, %d]  Stars=[%d, %d]",
           pi + 1, balls[0], balls[1], balls[2], balls[3], balls[4], stars[0],
           stars[1]);
    printf("  (%s)\n\n", from_model ? "model" : "fallback");
  }

  clock_gettime(CLOCK_MONOTONIC, &pipeline_end);
  double pipeline_time =
      (double)(pipeline_end.tv_sec - pipeline_start.tv_sec) +
      (double)(pipeline_end.tv_nsec - pipeline_start.tv_nsec) / 1e9;

  /* ================================================================
   * PHASE 4: Results Summary
   * ================================================================ */

  printf(
      "\n================================================================\n");
  printf("          EUROMILLIONS LOTTERY RESULTS\n");
  printf("================================================================\n");

  printf("\n--- Backtest (%d draws) ---\n", BACKTEST_HOLDOUT);
  printf("Model-valid predictions: %d / %d\n", bt_model_valid,
         BACKTEST_HOLDOUT);
  printf("Model ball hits:    %d / %d (%.1f%%)\n", bt_total_ball_hits,
         BACKTEST_HOLDOUT * NUM_BALLS,
         100.0 * bt_total_ball_hits / (BACKTEST_HOLDOUT * NUM_BALLS));
  printf("Model star hits:    %d / %d (%.1f%%)\n", bt_total_star_hits,
         BACKTEST_HOLDOUT * NUM_STARS,
         100.0 * bt_total_star_hits / (BACKTEST_HOLDOUT * NUM_STARS));
  printf("Random ball hits:   %d / %d (%.1f%%)\n", bt_rand_ball_hits,
         BACKTEST_HOLDOUT * NUM_BALLS,
         100.0 * bt_rand_ball_hits / (BACKTEST_HOLDOUT * NUM_BALLS));
  printf("Random star hits:   %d / %d (%.1f%%)\n", bt_rand_star_hits,
         BACKTEST_HOLDOUT * NUM_STARS,
         100.0 * bt_rand_star_hits / (BACKTEST_HOLDOUT * NUM_STARS));

  double exp_ball = (double)NUM_BALLS * NUM_BALLS / MAX_BALL;
  double exp_star = (double)NUM_STARS * NUM_STARS / MAX_STAR;
  printf("Expected random:    %.2f balls/draw, %.2f stars/draw\n", exp_ball,
         exp_star);

  int model_total = bt_total_ball_hits + bt_total_star_hits;
  int random_total = bt_rand_ball_hits + bt_rand_star_hits;
  if (model_total > random_total) {
    printf("Backtest result:    Model BEATS random by %d total hits\n",
           model_total - random_total);
  } else if (model_total == random_total) {
    printf("Backtest result:    Model TIES random\n");
  } else {
    printf("Backtest result:    Random beats model by %d total hits\n",
           random_total - model_total);
  }

  printf("\n--- Predictions ---\n");
  printf("Model-sourced:      %d / %d\n", total_model_sourced, NUM_PREDICTIONS);
  printf("Fallback-sourced:   %d / %d\n", total_fallbacks, NUM_PREDICTIONS);
  printf("Pipeline time:      %.2fs\n", pipeline_time);

  printf("\n> NOTE: Lottery draws are independent random events. These are\n"
         ">       statistically-weighted suggestions, not true predictions.\n");
  printf("================================================================\n");

  /* Cleanup */
  organelle_free(analyser);
  organelle_free(predictor);

  return 0;
}
