/*
 * MicroGPT-C — Market Regime Detection Multi-Organelle Demo
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Demonstrates the Adaptive Organelle Planner on real-world financial data:
 *   - 3 neural organelles coordinate via flat strings
 *   - Cross-Asset Analyser: summarises multi-asset market state
 *   - Regime Classifier: identifies market regime (RISK_ON/OFF/etc.)
 *   - Sector Rotator: recommends sector over/underweights
 *
 * Unlike lottery (independent random events), markets have learnable
 * cross-asset correlations and persistent regime states.
 *
 * Architecture: N_EMBD=64, N_HEAD=4, N_LAYER=3, ~159K params/organelle.
 *
 * Build:
 *   cmake --build build --target markets_demo
 *   ./build/markets_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Configuration ---- */
#define ANALYSER_CORPUS "market_analyser.txt"
#define REGIME_CORPUS "market_regime.txt"
#define ROTATOR_CORPUS "market_rotator.txt"
#define DATA_FILE "market_data.csv"

#define ANALYSER_CKPT "market_analyser.ckpt"
#define REGIME_CKPT "market_regime.ckpt"
#define ROTATOR_CKPT "market_rotator.ckpt"

#define ORGANELLE_TEMP 0.5
#define INF_GEN_LEN 80

#define MAX_RETRIES 5
#define ENSEMBLE_VOTES 5
#define BACKTEST_DAYS 60

/* ---- Regime labels (compact single-char encoding) ---- */
static const char REGIME_CHARS[] = {'R', 'O', 'I', 'D', 'T'};
static const char *REGIME_NAMES[] = {"RISK_ON", "RISK_OFF", "INFLATIONARY",
                                     "DEFLATIONARY", "TRANSITIONAL"};
#define NUM_REGIMES 5

/* ---- Market data row ---- */
#define MAX_ROWS 4000
#define MAX_LINE 2048

typedef struct {
  char date[12];
  double spy_ret1d, qqq_ret1d, tlt_ret1d, gld_ret1d, uso_ret1d;
  double uup_ret1d;
  double vix_close;
  double spy_ret21d, tlt_ret21d, gld_ret21d, uup_ret21d;
} MarketRow;

/* ---- Global config ---- */
static MicrogptConfig g_cfg;

/* ---- CSV column finder ---- */
static int find_col(const char *header, const char *name) {
  char buf[MAX_LINE];
  strncpy(buf, header, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  int col = 0;
  char *tok = strtok(buf, ",");
  while (tok) {
    /* Strip whitespace/quotes */
    while (*tok == ' ' || *tok == '"')
      tok++;
    char *end = tok + strlen(tok) - 1;
    while (end > tok &&
           (*end == ' ' || *end == '"' || *end == '\n' || *end == '\r'))
      *end-- = '\0';

    if (strcmp(tok, name) == 0)
      return col;
    tok = strtok(NULL, ",");
    col++;
  }
  return -1;
}

/* ---- Load market data from CSV ---- */
static int load_market_data(const char *filename, MarketRow *rows,
                            int max_rows) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", filename);
    return 0;
  }

  char line[MAX_LINE];
  if (!fgets(line, sizeof(line), f)) {
    fclose(f);
    return 0;
  }

  /* Find column indices */
  int col_date = find_col(line, "date");
  int col_spy_ret1d = find_col(line, "SP500_ret1d");
  int col_qqq_ret1d = find_col(line, "NASDAQ_ret1d");
  int col_tlt_ret1d = find_col(line, "BOND_20Y_ret1d");
  int col_gld_ret1d = find_col(line, "GOLD_ret1d");
  int col_uso_ret1d = find_col(line, "OIL_ret1d");
  int col_uup_ret1d = find_col(line, "USD_INDEX_ret1d");
  int col_vix = find_col(line, "VIX_close");
  int col_spy_ret21d = find_col(line, "SP500_ret21d");
  int col_tlt_ret21d = find_col(line, "BOND_20Y_ret21d");
  int col_gld_ret21d = find_col(line, "GOLD_ret21d");
  int col_uup_ret21d = find_col(line, "USD_INDEX_ret21d");

  if (col_date < 0 || col_spy_ret1d < 0 || col_vix < 0) {
    fprintf(stderr, "Missing required columns in %s\n", filename);
    fclose(f);
    return 0;
  }

  int count = 0;
  while (fgets(line, sizeof(line), f) && count < max_rows) {
    /* Strip trailing newline/carriage return */
    size_t linelen = strlen(line);
    while (linelen > 0 &&
           (line[linelen - 1] == '\n' || line[linelen - 1] == '\r'))
      line[--linelen] = '\0';

    /* Skip empty lines */
    if (linelen == 0)
      continue;

    /* Parse CSV line */
    char *fields[150];
    int nfields = 0;
    char *p = line;
    while (*p && nfields < 150) {
      fields[nfields++] = p;
      char *comma = strchr(p, ',');
      if (comma) {
        *comma = '\0';
        p = comma + 1;
      } else {
        break;
      }
    }

    /* Validate date field: must be YYYY-MM-DD format */
    if (col_date >= nfields)
      continue;
    if (strlen(fields[col_date]) < 10 || fields[col_date][4] != '-')
      continue;

    MarketRow *r = &rows[count];
    memset(r, 0, sizeof(MarketRow));

    strncpy(r->date, fields[col_date], sizeof(r->date) - 1);

    if (col_spy_ret1d < nfields)
      r->spy_ret1d = atof(fields[col_spy_ret1d]);
    if (col_qqq_ret1d < nfields)
      r->qqq_ret1d = atof(fields[col_qqq_ret1d]);
    if (col_tlt_ret1d < nfields)
      r->tlt_ret1d = atof(fields[col_tlt_ret1d]);
    if (col_gld_ret1d < nfields)
      r->gld_ret1d = atof(fields[col_gld_ret1d]);
    if (col_uso_ret1d < nfields)
      r->uso_ret1d = atof(fields[col_uso_ret1d]);
    if (col_uup_ret1d < nfields)
      r->uup_ret1d = atof(fields[col_uup_ret1d]);
    if (col_vix < nfields)
      r->vix_close = atof(fields[col_vix]);
    if (col_spy_ret21d < nfields)
      r->spy_ret21d = atof(fields[col_spy_ret21d]);
    if (col_tlt_ret21d < nfields)
      r->tlt_ret21d = atof(fields[col_tlt_ret21d]);
    if (col_gld_ret21d < nfields)
      r->gld_ret21d = atof(fields[col_gld_ret21d]);
    if (col_uup_ret21d < nfields)
      r->uup_ret21d = atof(fields[col_uup_ret21d]);

    count++;
  }

  fclose(f);
  return count;
}

/* ---- Build analyser prompt from a market row ---- */
static void build_analyser_prompt(const MarketRow *row, char *buf,
                                  size_t bufsz) {
  snprintf(buf, bufsz,
           "SPY=%+.1f%%|QQQ=%+.1f%%|TLT=%+.1f%%|GLD=%+.1f%%|USO=%+.1f%%|"
           "VIX=%.1f|UUP=%+.1f%%",
           row->spy_ret1d, row->qqq_ret1d, row->tlt_ret1d, row->gld_ret1d,
           row->uso_ret1d, row->vix_close, row->uup_ret1d);
}

/* ---- Classify regime (ground truth — returns single char) ---- */
static char classify_regime_char(const MarketRow *row) {
  double vix = row->vix_close;
  double spy_ret = row->spy_ret21d;
  double gld_ret = row->gld_ret21d;
  double tlt_ret = row->tlt_ret21d;

  if (vix < 18.0 && spy_ret > 0.0)
    return 'R'; /* RISK_ON */
  if (vix > 25.0 && spy_ret < 0.0)
    return 'O'; /* RISK_OFF */
  if (gld_ret > 0.0 && tlt_ret < 0.0)
    return 'I'; /* INFLATIONARY */
  if (gld_ret < 0.0 && tlt_ret > 0.0)
    return 'D'; /* DEFLATIONARY */
  return 'T';   /* TRANSITIONAL */
}

/* ---- Get human-readable name from regime char ---- */
static const char *regime_name(char c) {
  for (int i = 0; i < NUM_REGIMES; i++) {
    if (REGIME_CHARS[i] == c)
      return REGIME_NAMES[i];
  }
  return "UNKNOWN";
}

/* ---- Check if output contains a valid regime char ---- */
static int match_regime(const char *output, char *matched_char) {
  for (const char *p = output; *p; p++) {
    for (int i = 0; i < NUM_REGIMES; i++) {
      if (*p == REGIME_CHARS[i]) {
        *matched_char = REGIME_CHARS[i];
        return 1;
      }
    }
  }
  return 0;
}

/* ---- Build compact 5-char analysis summary (fallback) ---- */
static void build_analysis_summary(const MarketRow *row, char *buf,
                                   size_t bufsz) {
  if (bufsz < 6)
    return;
  buf[0] = row->spy_ret21d > 0 ? 'U' : 'D'; /* equity */
  buf[1] = row->tlt_ret21d > 0 ? 'U' : 'D'; /* bond */
  buf[2] = row->gld_ret21d > 0 ? 'U' : 'D'; /* commodity */
  buf[3] = row->vix_close < 20 ? 'L' : (row->vix_close < 30 ? 'M' : 'H');
  buf[4] = row->uup_ret21d > 0.5 ? 'S' : (row->uup_ret21d < -0.5 ? 'W' : 'F');
  buf[5] = '\0';
}

/* ==== Main ==== */

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
  microgpt_print_config("MicroGPT-C - Market Regime Detection Pipeline",
                        &g_cfg);

  /* ---- Load market data from CSV ---- */
  MarketRow *rows = (MarketRow *)malloc(MAX_ROWS * sizeof(MarketRow));
  if (!rows) {
    fprintf(stderr, "FATAL: Cannot allocate market data\n");
    return 1;
  }

  int total_rows = load_market_data(DATA_FILE, rows, MAX_ROWS);
  if (total_rows < BACKTEST_DAYS + 10) {
    fprintf(stderr, "FATAL: Need at least %d rows, found %d\n",
            BACKTEST_DAYS + 10, total_rows);
    free(rows);
    return 1;
  }

  printf("\nLoaded %d trading days from %s\n", total_rows, DATA_FILE);
  printf("Date range: %s → %s\n", rows[0].date, rows[total_rows - 1].date);
  printf("Latest VIX: %.1f\n\n", rows[total_rows - 1].vix_close);

  /* ================================================================
   * PHASE 1: Train organelles
   * ================================================================ */

  int train_steps = g_cfg.num_steps;
  printf("--- PHASE 1: TRAINING (%d steps each) ---\n\n", train_steps);

  Organelle *analyser = organelle_train("Analyser", ANALYSER_CORPUS,
                                        ANALYSER_CKPT, &g_cfg, train_steps);
  if (!analyser) {
    fprintf(stderr, "FATAL: Analyser training failed\n");
    free(rows);
    return 1;
  }

  Organelle *classifier = organelle_train("Regime Classifier", REGIME_CORPUS,
                                          REGIME_CKPT, &g_cfg, train_steps);
  if (!classifier) {
    fprintf(stderr, "FATAL: Regime Classifier training failed\n");
    free(rows);
    return 1;
  }

  Organelle *rotator = organelle_train("Sector Rotator", ROTATOR_CORPUS,
                                       ROTATOR_CKPT, &g_cfg, train_steps);
  if (!rotator) {
    fprintf(stderr, "FATAL: Sector Rotator training failed\n");
    free(rows);
    return 1;
  }

  /* ================================================================
   * PHASE 2: Backtest on most recent BACKTEST_DAYS trading days
   * ================================================================ */

  printf("\n--- PHASE 2: BACKTEST (%d most recent trading days) ---\n\n",
         BACKTEST_DAYS);

  int correct_regime = 0;
  int valid_regime = 0;
  int regime_dist[NUM_REGIMES] = {0};
  int actual_dist[NUM_REGIMES] = {0};

  int start_idx = total_rows - BACKTEST_DAYS;
  if (start_idx < 0)
    start_idx = 0;

  clock_t bt_start = clock();

  for (int i = start_idx; i < total_rows; i++) {
    MarketRow *row = &rows[i];
    char actual_char = classify_regime_char(row);

    /* Count actual regime distribution */
    for (int r = 0; r < NUM_REGIMES; r++) {
      if (actual_char == REGIME_CHARS[r]) {
        actual_dist[r]++;
        break;
      }
    }

    /* Step 1: Generate analysis */
    char analyser_prompt[256];
    build_analyser_prompt(row, analyser_prompt, sizeof(analyser_prompt));

    char analysis_output[INF_GEN_LEN + 1];
    organelle_generate(analyser, &g_cfg, analyser_prompt, analysis_output,
                       INF_GEN_LEN, ORGANELLE_TEMP);

    /* Use analysis output if it looks like compact 5-char format (UDUMS) */
    char regime_prompt[256];
    int analysis_valid =
        (strlen(analysis_output) >= 5 &&
         (analysis_output[0] == 'U' || analysis_output[0] == 'D') &&
         (analysis_output[3] == 'L' || analysis_output[3] == 'M' ||
          analysis_output[3] == 'H'));
    if (analysis_valid) {
      snprintf(regime_prompt, sizeof(regime_prompt), "%s", analysis_output);
    } else {
      build_analysis_summary(row, regime_prompt, sizeof(regime_prompt));
    }

    /* Step 2: Classify regime */
    char regime_output[INF_GEN_LEN + 1];
    scalar_t vote_conf = 0;
    organelle_generate_ensemble(classifier, &g_cfg, regime_prompt,
                                regime_output, INF_GEN_LEN, ENSEMBLE_VOTES,
                                ORGANELLE_TEMP, &vote_conf);

    char predicted_char = 0;
    int regime_valid = match_regime(regime_output, &predicted_char);

    if (regime_valid) {
      valid_regime++;
      if (predicted_char == actual_char) {
        correct_regime++;
      }

      for (int r = 0; r < NUM_REGIMES; r++) {
        if (predicted_char == REGIME_CHARS[r]) {
          regime_dist[r]++;
          break;
        }
      }
    }

    /* Print every 10th day or last 5 */
    if ((i - start_idx) % 10 == 0 || i >= total_rows - 5) {
      printf("  %s: actual=%-14s predicted=%-14s %s (conf=%.2f)\n", row->date,
             regime_name(actual_char),
             regime_valid ? regime_name(predicted_char) : "(invalid)",
             (regime_valid && predicted_char == actual_char) ? "\xe2\x9c\x93"
                                                             : "\xe2\x9c\x97",
             (double)vote_conf);
    }
  }

  double bt_time = (double)(clock() - bt_start) / CLOCKS_PER_SEC;

  /* ================================================================
   * PHASE 3: Current Market Assessment
   * ================================================================ */

  printf("\n--- PHASE 3: CURRENT MARKET ASSESSMENT ---\n\n");

  MarketRow *latest = &rows[total_rows - 1];
  printf("Latest data: %s (VIX=%.1f, SPY 1d=%+.1f%%)\n\n", latest->date,
         latest->vix_close, latest->spy_ret1d);

  /* Step 1: Analyse */
  char cur_prompt[256];
  build_analyser_prompt(latest, cur_prompt, sizeof(cur_prompt));
  printf("  Analyser prompt: %s\n", cur_prompt);

  char cur_analysis[INF_GEN_LEN + 1];
  organelle_generate(analyser, &g_cfg, cur_prompt, cur_analysis, INF_GEN_LEN,
                     ORGANELLE_TEMP);
  printf("  Analyser output: \"%s\"\n", cur_analysis);

  /* Step 2: Classify regime */
  char cur_regime_prompt[256];
  int cur_analysis_valid =
      (strlen(cur_analysis) >= 5 &&
       (cur_analysis[0] == 'U' || cur_analysis[0] == 'D') &&
       (cur_analysis[3] == 'L' || cur_analysis[3] == 'M' ||
        cur_analysis[3] == 'H'));
  if (cur_analysis_valid) {
    snprintf(cur_regime_prompt, sizeof(cur_regime_prompt), "%s", cur_analysis);
  } else {
    build_analysis_summary(latest, cur_regime_prompt,
                           sizeof(cur_regime_prompt));
    printf("  (analyser garbled, using direct analysis)\n");
  }

  char cur_regime_out[INF_GEN_LEN + 1];
  scalar_t cur_conf = 0;
  organelle_generate_ensemble(classifier, &g_cfg, cur_regime_prompt,
                              cur_regime_out, INF_GEN_LEN, ENSEMBLE_VOTES,
                              ORGANELLE_TEMP, &cur_conf);

  char cur_regime_char = 'T'; /* default TRANSITIONAL */
  if (!match_regime(cur_regime_out, &cur_regime_char)) {
    /* Fall back to rule-based classification */
    cur_regime_char = classify_regime_char(latest);
    printf("  (classifier garbled, using rule-based fallback)\n");
  }
  printf("  Regime: %s (conf=%.2f)\n\n", regime_name(cur_regime_char),
         (double)cur_conf);

  /* Step 3: Sector rotation (compact prompt: R:UDUMS) */
  char rotator_prompt[256];
  snprintf(rotator_prompt, sizeof(rotator_prompt), "%c:%s", cur_regime_char,
           cur_regime_prompt);
  if (strlen(rotator_prompt) > 100)
    rotator_prompt[100] = '\0';

  char rotator_output[INF_GEN_LEN + 1];
  organelle_generate(rotator, &g_cfg, rotator_prompt, rotator_output,
                     INF_GEN_LEN, ORGANELLE_TEMP);
  printf("  Sector Rotator: \"%s\"\n", rotator_output);

  /* ================================================================
   * Results Summary
   * ================================================================ */

  printf(
      "\n================================================================\n");
  printf("          MARKET REGIME DETECTION RESULTS\n");
  printf("================================================================\n");

  printf("\n--- Backtest (%d trading days) ---\n", BACKTEST_DAYS);
  printf("Valid regime predictions: %d / %d (%.1f%%)\n", valid_regime,
         BACKTEST_DAYS, 100.0 * valid_regime / BACKTEST_DAYS);
  printf("Correct regime:          %d / %d (%.1f%%)\n", correct_regime,
         valid_regime > 0 ? valid_regime : 1,
         valid_regime > 0 ? 100.0 * correct_regime / valid_regime : 0);
  printf("Random baseline:         %.1f%% (1/%d classes)\n",
         100.0 / NUM_REGIMES, NUM_REGIMES);

  printf("\n  Actual regime distribution:\n");
  for (int r = 0; r < NUM_REGIMES; r++) {
    if (actual_dist[r] > 0)
      printf("    %-14s: %d days\n", REGIME_NAMES[r], actual_dist[r]);
  }

  printf("\n  Predicted regime distribution:\n");
  for (int r = 0; r < NUM_REGIMES; r++) {
    if (regime_dist[r] > 0)
      printf("    %-14s: %d days\n", REGIME_NAMES[r], regime_dist[r]);
  }

  printf("\n--- Current Assessment ---\n");
  printf("Date:     %s\n", latest->date);
  printf("VIX:      %.1f\n", latest->vix_close);
  printf("Regime:   %s\n", regime_name(cur_regime_char));
  printf("Sectors:  %s\n", rotator_output);
  printf("Pipeline: %.2fs\n", bt_time);

  printf("\n> NOTE: Market regimes are derived classifications, not\n");
  printf(">       guaranteed predictions. Use for research only.\n");
  printf("================================================================\n");

  /* Cleanup */
  free(rows);
  return 0;
}
