/*
 * MicroGPT-C -- OPA Code Composition Pipeline (c_compose)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Trains a c_planner organelle on flat-string compositions and a c_judge
 * organelle on PASS/FAIL validation. Then evaluates the pipeline on
 * held-out test intents:
 *   comment  ->  c_planner  ->  "seq|fn1|fn2"  ->  c_judge  ->  PASS/FAIL
 *
 * This is the first OPA experiment for code synthesis: composing
 * well-known C functions into novel pipelines via structured flat-string
 * communication between sub-1M parameter organelles.
 */

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MICROGPT_METAL
#include "microgpt_metal.h"
#endif

#define PLANNER_CKPT "c_planner.ckpt"
#define JUDGE_CKPT "c_judge.ckpt"

#define ENSEMBLE_K 3
#define GEN_LEN 80
#define PLANNER_TEMP 0.2
#define JUDGE_TEMP 0.1

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

/* ── Registry: known function names ─────────────────────────────── */

#define MAX_REGISTRY 600

static struct {
  char name[64];
  char ret_type[16];
} g_registry[MAX_REGISTRY];
static int g_registry_n = 0;

static void load_registry(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    printf("[WARN] Cannot open registry %s\n", path);
    return;
  }
  char line[128];
  while (fgets(line, sizeof(line), f) && g_registry_n < MAX_REGISTRY) {
    /* Format: name|type */
    char *sep = strchr(line, '|');
    if (!sep)
      continue;
    *sep = '\0';
    char *type = sep + 1;
    /* trim newline */
    char *nl = strchr(type, '\n');
    if (nl)
      *nl = '\0';
    strncpy(g_registry[g_registry_n].name, line, 63);
    g_registry[g_registry_n].name[63] = '\0';
    strncpy(g_registry[g_registry_n].ret_type, type, 15);
    g_registry[g_registry_n].ret_type[15] = '\0';
    g_registry_n++;
  }
  fclose(f);
  printf("[REGISTRY] Loaded %d function names\n", g_registry_n);
}

static int registry_lookup(const char *name) {
  for (int i = 0; i < g_registry_n; i++) {
    if (strcmp(g_registry[i].name, name) == 0)
      return i;
  }
  return -1;
}

/* ── Load test intents ──────────────────────────────────────────── */

#define MAX_TESTS 200

static struct {
  char comment[256];
  char expected_plan[256];
} g_tests[MAX_TESTS];
static int g_test_n = 0;

static void load_tests(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    printf("[WARN] Cannot open test file %s\n", path);
    return;
  }
  char line[512];
  while (g_test_n < MAX_TESTS) {
    /* Read comment line */
    if (!fgets(line, sizeof(line), f))
      break;
    char *nl = strchr(line, '\n');
    if (nl)
      *nl = '\0';
    if (line[0] == '\0')
      continue; /* skip blank lines */
    if (line[0] != '/')
      continue; /* skip non-comment lines */
    strncpy(g_tests[g_test_n].comment, line, 255);
    g_tests[g_test_n].comment[255] = '\0';

    /* Read expected plan line */
    if (!fgets(line, sizeof(line), f))
      break;
    nl = strchr(line, '\n');
    if (nl)
      *nl = '\0';
    strncpy(g_tests[g_test_n].expected_plan, line, 255);
    g_tests[g_test_n].expected_plan[255] = '\0';
    g_test_n++;
  }
  fclose(f);
  printf("[TEST] Loaded %d test intents\n", g_test_n);
}

/* ── Parse and validate a flat-string plan ──────────────────────── */

#define MAX_PLAN_FNS 10

static int parse_plan(const char *raw, char fns[MAX_PLAN_FNS][64]) {
  /* Expected format: seq|fn1|fn2|... */
  if (strncmp(raw, "seq|", 4) != 0)
    return 0;
  const char *p = raw + 4;
  int count = 0;
  while (*p && count < MAX_PLAN_FNS) {
    const char *sep = strchr(p, '|');
    size_t len = sep ? (size_t)(sep - p) : strlen(p);
    if (len == 0 || len > 63)
      break;
    memcpy(fns[count], p, len);
    fns[count][len] = '\0';
    count++;
    if (!sep)
      break;
    p = sep + 1;
  }
  return count;
}

static int validate_plan_fns(char fns[MAX_PLAN_FNS][64], int nfns) {
  for (int i = 0; i < nfns; i++) {
    if (registry_lookup(fns[i]) < 0)
      return 0;
  }
  return 1;
}

/* ── Constrained decoding: nearest-name filter ──────────────────── */

static int edit_distance(const char *a, const char *b) {
  int la = (int)strlen(a), lb = (int)strlen(b);
  if (la > 63 || lb > 63)
    return 99;
  /* Simple Levenshtein with static buffer */
  int dp[64][64];
  for (int i = 0; i <= la; i++)
    dp[i][0] = i;
  for (int j = 0; j <= lb; j++)
    dp[0][j] = j;
  for (int i = 1; i <= la; i++) {
    for (int j = 1; j <= lb; j++) {
      int cost = (a[i - 1] != b[j - 1]) ? 1 : 0;
      int d1 = dp[i - 1][j] + 1;
      int d2 = dp[i][j - 1] + 1;
      int d3 = dp[i - 1][j - 1] + cost;
      dp[i][j] = d1 < d2 ? (d1 < d3 ? d1 : d3) : (d2 < d3 ? d2 : d3);
    }
  }
  return dp[la][lb];
}

static void constrain_plan(char fns[MAX_PLAN_FNS][64], int nfns) {
  /* For each function name, if not in registry, find closest match */
  for (int i = 0; i < nfns; i++) {
    if (registry_lookup(fns[i]) >= 0)
      continue; /* already valid */
    int best_dist = 99;
    int best_idx = -1;
    for (int r = 0; r < g_registry_n; r++) {
      int d = edit_distance(fns[i], g_registry[r].name);
      if (d < best_dist) {
        best_dist = d;
        best_idx = r;
      }
    }
    /* Accept correction only if edit distance <= 3 */
    if (best_idx >= 0 && best_dist <= 3) {
      strncpy(fns[i], g_registry[best_idx].name, 63);
      fns[i][63] = '\0';
    }
  }
}

/* ── Ensemble planner: generate K plans and vote ────────────────── */

static int plan_matches(const char *a, const char *b) {
  return strcmp(a, b) == 0;
}

static void generate_plan(Organelle *planner, const MicrogptConfig *cfg,
                          const char *intent, char *out_plan, int out_len) {
  char best[256] = {0};
  int votes[ENSEMBLE_K];
  char plans[ENSEMBLE_K][256];

  for (int k = 0; k < ENSEMBLE_K; k++) {
    scalar_t temp = (scalar_t)PLANNER_TEMP + (scalar_t)(k - 1) * (scalar_t)0.05;
    if (temp < 0.05)
      temp = 0.05;

    char raw[512];
    organelle_generate(planner, cfg, intent, raw, sizeof(raw), temp);

    /* Extract just the first line after the intent (the plan) */
    char *plan_start = strstr(raw, "seq|");
    if (plan_start) {
      char *end = strchr(plan_start, '\n');
      size_t plen = end ? (size_t)(end - plan_start) : strlen(plan_start);
      if (plen > 255)
        plen = 255;
      memcpy(plans[k], plan_start, plen);
      plans[k][plen] = '\0';
    } else {
      plans[k][0] = '\0';
    }
  }

  /* Majority vote */
  for (int k = 0; k < ENSEMBLE_K; k++) {
    votes[k] = 0;
    for (int j = 0; j < ENSEMBLE_K; j++) {
      if (plan_matches(plans[k], plans[j]))
        votes[k]++;
    }
  }

  int best_idx = 0;
  for (int k = 1; k < ENSEMBLE_K; k++) {
    if (votes[k] > votes[best_idx])
      best_idx = k;
  }

  strncpy(out_plan, plans[best_idx], (size_t)(out_len - 1));
  out_plan[out_len - 1] = '\0';
}

/* ── Judge: validate a plan ─────────────────────────────────────── */

static int judge_plan(Organelle *judge, const MicrogptConfig *cfg,
                      const char *plan) {
  char raw[256];
  organelle_generate(judge, cfg, plan, raw, sizeof(raw), (scalar_t)JUDGE_TEMP);

  /* Look for PASS or FAIL in output */
  if (strstr(raw, "PASS"))
    return 1;
  if (strstr(raw, "FAIL"))
    return 0;
  return -1; /* parse error */
}

/* ── Main ───────────────────────────────────────────────────────── */

int main(void) {
  printf("=== OPA Code Composition Pipeline ===\n\n");

  /* ── Load registry ── */
  load_registry("c_registry.txt");
  load_tests("test_intents.txt");

  /* ── Configure model ── */
  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = N_EMBD;
  cfg.n_head = N_HEAD;
  cfg.mlp_dim = MLP_DIM;
  cfg.n_layer = N_LAYER;

  /* ── Train/load c_planner ── */
  printf("\n[1/2] Training c_planner organelle...\n");
  Organelle *planner = organelle_train("c_planner", "c_planner.txt",
                                       PLANNER_CKPT, &cfg, NUM_STEPS);
  if (!planner) {
    fprintf(stderr, "FATAL: c_planner training failed\n");
    return 1;
  }

  /* ── Train/load c_judge ── */
  printf("\n[2/2] Training c_judge organelle...\n");
  Organelle *judge =
      organelle_train("c_judge", "c_judge.txt", JUDGE_CKPT, &cfg, NUM_STEPS);
  if (!judge) {
    fprintf(stderr, "FATAL: c_judge training failed\n");
    return 1;
  }

  /* ── Evaluate pipeline on test intents ── */
  printf("\n=== Pipeline Evaluation (%d test intents) ===\n\n", g_test_n);

  int total = 0, parse_ok = 0, registry_ok = 0, judge_pass = 0;
  int exact_match = 0, partial_match = 0;
  int parse_errors = 0, judge_errors = 0;

  for (int t = 0; t < g_test_n; t++) {
    total++;

    /* Step 1: Generate plan from intent */
    char plan[256];
    generate_plan(planner, &cfg, g_tests[t].comment, plan, sizeof(plan));

    /* Step 2: Parse the plan */
    char fns[MAX_PLAN_FNS][64];
    int nfns = parse_plan(plan, fns);

    int plan_parseable = (nfns > 0);
    if (plan_parseable)
      parse_ok++;
    else {
      parse_errors++;
      goto report;
    }

    /* Step 2.5: Constrained decoding — fix up near-miss function names */
    constrain_plan(fns, nfns);

    /* Reconstruct plan string after constraint correction */
    {
      char corrected[256] = "seq";
      for (int i = 0; i < nfns; i++) {
        strncat(corrected, "|", sizeof(corrected) - strlen(corrected) - 1);
        strncat(corrected, fns[i], sizeof(corrected) - strlen(corrected) - 1);
      }
      strncpy(plan, corrected, sizeof(plan) - 1);
      plan[sizeof(plan) - 1] = '\0';
    }

    /* Step 3: Validate function names against registry */
    int all_valid = validate_plan_fns(fns, nfns);
    if (all_valid)
      registry_ok++;

    /* Step 4: Judge the plan */
    int verdict = judge_plan(judge, &cfg, plan);
    if (verdict == 1)
      judge_pass++;
    else if (verdict < 0)
      judge_errors++;

    /* Step 5: Compare with expected plan */
    if (strcmp(plan, g_tests[t].expected_plan) == 0) {
      exact_match++;
    } else {
      /* Check if any function names overlap */
      char exp_fns[MAX_PLAN_FNS][64];
      int exp_n = parse_plan(g_tests[t].expected_plan, exp_fns);
      int overlap = 0;
      for (int i = 0; i < nfns; i++) {
        for (int j = 0; j < exp_n; j++) {
          if (strcmp(fns[i], exp_fns[j]) == 0) {
            overlap++;
            break;
          }
        }
      }
      if (overlap > 0)
        partial_match++;
    }

  report:
    printf("[%3d] %s\n", t + 1, g_tests[t].comment);
    printf("      expected: %s\n", g_tests[t].expected_plan);
    printf("      got:      %s", plan);
    if (nfns == 0)
      printf("  [PARSE ERROR]");
    else if (!validate_plan_fns(fns, nfns))
      printf("  [UNKNOWN FN]");
    if (strcmp(plan, g_tests[t].expected_plan) == 0)
      printf("  ✓ EXACT");
    printf("\n\n");
  }

  /* ── Summary ── */
  printf("=== Results Summary ===\n\n");
  printf("Test intents:        %d\n", total);
  printf("Plans parseable:     %d / %d (%.0f%%)\n", parse_ok, total,
         100.0 * parse_ok / total);
  printf("All fns in registry: %d / %d (%.0f%%)\n", registry_ok, total,
         100.0 * registry_ok / total);
  printf("Judge PASS:          %d / %d (%.0f%%)\n", judge_pass, total,
         100.0 * judge_pass / total);
  printf("Exact match:         %d / %d (%.0f%%)\n", exact_match, total,
         100.0 * exact_match / total);
  printf("Partial match:       %d / %d (%.0f%%)\n", partial_match, total,
         100.0 * partial_match / total);
  printf("Parse errors:        %d\n", parse_errors);
  printf("Judge errors:        %d\n", judge_errors);
  printf("\n");

  /* ── Cleanup ── */
  organelle_free(planner);
  organelle_free(judge);

  return 0;
}
