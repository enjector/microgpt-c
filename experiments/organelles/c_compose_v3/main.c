/*
 * MicroGPT-C -- OPA Code Composition Pipeline v3 (c_compose_v3)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Extends c_compose with a third organelle (c_wiringgen) that converts the
 * planner's flat-string plan into an actual C function body, gated by a
 * deterministic C syntax judge (gcc -fsyntax-only).
 *
 * Pipeline:
 *   intent  →  c_planner  →  "seq|fn1|fn2"
 *           →  c_wiringgen →  C function body (candidate)
 *           →  C Syntax Judge (gcc -fsyntax-only) → PASS/FAIL
 *           →  c_judge (neural)  →  PASS/FAIL on semantics
 *           →  OpaKanban retry if FAIL (up to MAX_RETRIES)
 *
 * The C Syntax Judge is the key deterministic gate:
 *   - 0% false positives (gcc either rejects or it doesn't)
 *   - ~5ms per check (no object file produced)
 *   - Frees the neural organelles from needing perfect C syntax
 *
 * Build:
 *   cmake --build build --target c_compose_v3
 *   ./build/c_compose_v3
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef MICROGPT_METAL
#include "microgpt_metal.h"
#endif

/* ── Build-time defaults (overridden by CMake DEFINES) ────────────────── */
#ifndef PLANNER_CKPT
#define PLANNER_CKPT "c_planner.ckpt"
#endif
#ifndef WIRING_CKPT
#define WIRING_CKPT "c_wiringgen.ckpt"
#endif
#ifndef JUDGE_CKPT
#define JUDGE_CKPT "c_judge.ckpt"
#endif
#ifndef PLANNER_TEMP
#define PLANNER_TEMP 0.2f
#endif
#ifndef WIRING_TEMP
#define WIRING_TEMP 0.3f
#endif
#ifndef JUDGE_TEMP
#define JUDGE_TEMP 0.1f
#endif
#ifndef ENSEMBLE_K
#define ENSEMBLE_K 3
#endif
#ifndef MAX_RETRIES
#define MAX_RETRIES 3
#endif
#ifndef GEN_LEN
#define GEN_LEN 400
#endif

/* ── Architecture macros for wiringgen (separate from planner/judge) ───── */
#ifndef WIRING_N_EMBD
#define WIRING_N_EMBD 128
#endif
#ifndef WIRING_N_HEAD
#define WIRING_N_HEAD 4
#endif
#ifndef WIRING_N_LAYER
#define WIRING_N_LAYER 4
#endif
#ifndef WIRING_BLOCK_SIZE
#define WIRING_BLOCK_SIZE 128
#endif
#ifndef WIRING_MLP_DIM
#define WIRING_MLP_DIM 512
#endif
#ifndef WIRING_NUM_STEPS
#define WIRING_NUM_STEPS 20000
#endif

/* ── Registry ─────────────────────────────────────────────────────────── */

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
    char *sep = strchr(line, '|');
    if (!sep)
      continue;
    *sep = '\0';
    char *type = sep + 1;
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
  for (int i = 0; i < g_registry_n; i++)
    if (strcmp(g_registry[i].name, name) == 0)
      return i;
  return -1;
}

/* ── Flat-string plan parsing ─────────────────────────────────────────── */

#define MAX_PLAN_FNS 10

static int parse_plan(const char *raw, char fns[MAX_PLAN_FNS][64]) {
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
  for (int i = 0; i < nfns; i++)
    if (registry_lookup(fns[i]) < 0)
      return 0;
  return 1;
}

/* ── Levenshtein constrained decoding ─────────────────────────────────── */

static int edit_distance(const char *a, const char *b) {
  int la = (int)strlen(a), lb = (int)strlen(b);
  if (la > 63 || lb > 63)
    return 99;
  int dp[64][64];
  for (int i = 0; i <= la; i++)
    dp[i][0] = i;
  for (int j = 0; j <= lb; j++)
    dp[0][j] = j;
  for (int i = 1; i <= la; i++) {
    for (int j = 1; j <= lb; j++) {
      int cost = (a[i - 1] != b[j - 1]) ? 1 : 0;
      int d1 = dp[i - 1][j] + 1, d2 = dp[i][j - 1] + 1,
          d3 = dp[i - 1][j - 1] + cost;
      dp[i][j] = d1 < d2 ? (d1 < d3 ? d1 : d3) : (d2 < d3 ? d2 : d3);
    }
  }
  return dp[la][lb];
}

static void constrain_plan(char fns[MAX_PLAN_FNS][64], int nfns) {
  for (int i = 0; i < nfns; i++) {
    if (registry_lookup(fns[i]) >= 0)
      continue;
    int best_dist = 99, best_idx = -1;
    for (int r = 0; r < g_registry_n; r++) {
      int d = edit_distance(fns[i], g_registry[r].name);
      if (d < best_dist) {
        best_dist = d;
        best_idx = r;
      }
    }
    if (best_idx >= 0 && best_dist <= 3) {
      strncpy(fns[i], g_registry[best_idx].name, 63);
      fns[i][63] = '\0';
    }
  }
}

/* ── Ensemble planner ─────────────────────────────────────────────────── */

static void generate_plan(Organelle *planner, const MicrogptConfig *cfg,
                          const char *intent, char *out_plan, int out_len) {
  char plans[ENSEMBLE_K][256];
  int votes[ENSEMBLE_K];
  for (int k = 0; k < ENSEMBLE_K; k++) {
    float temp = (float)PLANNER_TEMP + (float)(k - 1) * 0.05f;
    if (temp < 0.05f)
      temp = 0.05f;
    char raw[512];
    organelle_generate(planner, cfg, intent, raw, sizeof(raw), (scalar_t)temp);
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
  for (int k = 0; k < ENSEMBLE_K; k++) {
    votes[k] = 0;
    for (int j = 0; j < ENSEMBLE_K; j++)
      if (strcmp(plans[k], plans[j]) == 0)
        votes[k]++;
  }
  int best_idx = 0;
  for (int k = 1; k < ENSEMBLE_K; k++)
    if (votes[k] > votes[best_idx])
      best_idx = k;
  strncpy(out_plan, plans[best_idx], (size_t)(out_len - 1));
  out_plan[out_len - 1] = '\0';
}

/* ── C Syntax Judge (deterministic) ──────────────────────────────────── */

/*
 * Build a minimal C translation unit from the generated body and check it
 * with gcc -fsyntax-only.  Returns 1 on valid C, 0 on syntax error.
 *
 * The header_decls string provides forward declarations for known primitives
 * so the wiringgen body can reference them without full definitions.
 */
static const char *K_FORWARD_DECLS =
    "#include <math.h>\n"
    "#include <stdlib.h>\n"
    "#include <string.h>\n"
    "/* forward declarations for registry primitives */\n"
    "double mean(const double *x, int n);\n"
    "double variance(const double *x, int n);\n"
    "double stddev(const double *x, int n);\n"
    "void normalize_z(double *out, const double *x, int n);\n"
    "void rolling_mean(double *out, const double *x, int n, int w);\n"
    "void rolling_std(double *out, const double *x, int n, int w);\n"
    "void moving_avg(double *y, const double *x, int n, int w);\n"
    "double ema(double prev, double cur, double alpha);\n"
    "void ema_series(double *out, const double *x, int n, double alpha);\n"
    "void diff_central(double *dy, const double *y, double h, int n);\n"
    "void first_diff(double *dy, const double *x, int n);\n"
    "void softmax(double *p, const double *x, int n);\n"
    "double sigmoid(double x);\n"
    "double relu(double x);\n"
    "void relu_array(double *out, const double *x, int n);\n"
    "double cosine_similarity(const double *a, const double *b, int n);\n"
    "double dot(const double *a, const double *b, int n);\n"
    "double vec_norm(const double *v, int n);\n"
    "void vec_normalize(double *v, int n);\n"
    "void fft_radix2(double *re, double *im, int n);\n"
    "void fft_magnitude(double *mag, const double *re, const double *im, int "
    "n);\n"
    "void fft_power(double *power, const double *re, const double *im, int "
    "n);\n"
    "void hann_window(double *w, int n);\n"
    "void apply_window(double *out, const double *signal, const double "
    "*window, int n);\n"
    "void fft_windowed_hann(double *re, double *im, const double *signal, int "
    "n);\n"
    "double correlation(const double *x, const double *y, int n);\n"
    "double covariance(const double *x, const double *y, int n);\n"
    "double max_val(const double *x, int n);\n"
    "double min_val(const double *x, int n);\n"
    "void bubble_sort(double *a, int n);\n"
    "double median(double *a, int n);\n"
    "void cumsum(double *y, const double *x, int n);\n"
    "void lowpass(double *y, const double *x, int n, double alpha);\n"
    "void highpass_iir(double *y, const double *x, int n, double alpha);\n"
    "void bandpass_filter(double *y, const double *x, int n, double lo, double "
    "hi);\n"
    "void downsample(double *y, const double *x, int n, int factor);\n"
    "void upsample(double *y, const double *x, int n, int factor);\n"
    "double signal_energy(const double *x, int n);\n"
    "double rms_value(const double *x, int n);\n"
    "void detrend(double *out, const double *x, int n);\n"
    "double autocorrelation(const double *x, int n, int lag);\n"
    "void trimmed_mean(double *out, const double *x, int n, double frac);\n"
    "double percentile(double *a, int n, double p);\n"
    "double kurtosis(const double *x, int n);\n"
    "double skewness(const double *x, int n);\n"
    "void zscore(double *out, const double *x, int n);\n"
    "void minmax_normalize(double *out, const double *x, int n);\n"
    "void center_scale(double *out, const double *x, int n);\n"
    "void smooth_diff(double *out, const double *x, int n, int w);\n"
    "void double_smooth(double *out, const double *x, int n, int w1, int w2);\n"
    "void triple_smooth(double *out, const double *x, int n, int w);\n"
    "double entropy(const double *p, int n);\n";

static int c_syntax_judge(const char *function_body) {
  char tmpfile[128];
  snprintf(tmpfile, sizeof(tmpfile), "/tmp/c_compose_v3_%d.c", (int)getpid());

  FILE *f = fopen(tmpfile, "w");
  if (!f)
    return 0;

  /* Write forward decls + generated body */
  fprintf(f, "%s\n", K_FORWARD_DECLS);
  fprintf(f, "%s\n", function_body);
  fclose(f);

  /* gcc -fsyntax-only: fast (~5ms), no object file emitted */
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "gcc -fsyntax-only -std=c99 -w \"%s\" 2>/dev/null",
           tmpfile);
  int ret = system(cmd);
  remove(tmpfile);
  return (ret == 0) ? 1 : 0;
}

/* ── Neural judge ─────────────────────────────────────────────────────── */

static int judge_plan(Organelle *judge, const MicrogptConfig *cfg,
                      const char *plan) {
  char raw[256];
  organelle_generate(judge, cfg, plan, raw, sizeof(raw), (scalar_t)JUDGE_TEMP);
  if (strstr(raw, "PASS"))
    return 1;
  if (strstr(raw, "FAIL"))
    return 0;
  return -1; /* parse error */
}

/* ── Build wiring prompt from plan ────────────────────────────────────── */

/*
 * Converts a flat-string plan like "seq|normalize_z|cosine_similarity"
 * into the comment-style prompt that c_wiringgen was trained on:
 *   "/* normalize_z then cosine_similarity *\/"
 * This mirrors the c_wiring.txt training format.
 */
static void plan_to_wiring_prompt(const char *plan, char fns[MAX_PLAN_FNS][64],
                                  int nfns, char *prompt_out, int prompt_sz) {
  (void)plan;
  if (nfns == 0) {
    strncpy(prompt_out, "/* compose */", (size_t)(prompt_sz - 1));
    prompt_out[prompt_sz - 1] = '\0';
    return;
  }
  /* Build prompt: fn1 then fn2 then fn3 */
  char buf[512];
  strncpy(buf, "/* ", sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  for (int i = 0; i < nfns; i++) {
    size_t rem = sizeof(buf) - strlen(buf) - 1;
    strncat(buf, fns[i], rem);
    if (i < nfns - 1) {
      rem = sizeof(buf) - strlen(buf) - 1;
      strncat(buf, " then ", rem);
    }
  }
  size_t rem = sizeof(buf) - strlen(buf) - 1;
  strncat(buf, " */", rem);
  strncpy(prompt_out, buf, (size_t)(prompt_sz - 1));
  prompt_out[prompt_sz - 1] = '\0';
}

/* ── Test intents ─────────────────────────────────────────────────────── */

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
    if (!fgets(line, sizeof(line), f))
      break;
    char *nl = strchr(line, '\n');
    if (nl)
      *nl = '\0';
    if (line[0] == '\0')
      continue;
    if (line[0] != '/')
      continue;
    strncpy(g_tests[g_test_n].comment, line, 255);
    g_tests[g_test_n].comment[255] = '\0';
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

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(void) {
  printf("=== OPA Code Composition Pipeline v3 (with Syntax Judge) ===\n\n");

  /* Verify gcc is available — required for syntax judge */
  if (system("gcc --version > /dev/null 2>&1") != 0) {
    fprintf(stderr, "[ERROR] gcc not found — C Syntax Judge requires gcc\n");
    fprintf(stderr, "        Install gcc or set CC to a compatible compiler\n");
    return 1;
  }
  printf("[JUDGE] gcc available — C Syntax Judge active\n\n");

  load_registry("c_registry.txt");
  load_tests("test_intents.txt");

  /* ── Configure planner/judge (same arch as c_compose v2) ── */
  MicrogptConfig cfg_pj = microgpt_default_config();
  cfg_pj.n_embd = N_EMBD;
  cfg_pj.n_head = N_HEAD;
  cfg_pj.mlp_dim = MLP_DIM;
  cfg_pj.n_layer = N_LAYER;
  cfg_pj.block_size = BLOCK_SIZE;
  cfg_pj.batch_size = BATCH_SIZE;
  cfg_pj.max_vocab = MAX_VOCAB;
  cfg_pj.max_docs = MAX_DOCS;
  cfg_pj.max_doc_len = MAX_DOC_LEN;

  /* ── Configure wiringgen (separate arch — larger block for C bodies) ── */
  MicrogptConfig cfg_wg = microgpt_default_config();
  cfg_wg.n_embd = WIRING_N_EMBD;
  cfg_wg.n_head = WIRING_N_HEAD;
  cfg_wg.mlp_dim = WIRING_MLP_DIM;
  cfg_wg.n_layer = WIRING_N_LAYER;
  cfg_wg.block_size = WIRING_BLOCK_SIZE;
  cfg_wg.batch_size = 16;
  cfg_wg.max_vocab = 200;
  cfg_wg.max_docs = 5000;
  cfg_wg.max_doc_len = 1024;

  /* ── Train / load organelles ── */
  printf("[1/3] Training c_planner organelle...\n");
  Organelle *planner = organelle_train("c_planner", "c_planner.txt",
                                       PLANNER_CKPT, &cfg_pj, NUM_STEPS);
  if (!planner) {
    fprintf(stderr, "FATAL: c_planner training failed\n");
    return 1;
  }

  printf("\n[2/3] Training c_judge organelle...\n");
  Organelle *judge =
      organelle_train("c_judge", "c_judge.txt", JUDGE_CKPT, &cfg_pj, NUM_STEPS);
  if (!judge) {
    fprintf(stderr, "FATAL: c_judge training failed\n");
    return 1;
  }

  printf("\n[3/3] Training c_wiringgen organelle...\n");
  Organelle *wiring = organelle_train("c_wiringgen", "c_wiring.txt",
                                      WIRING_CKPT, &cfg_wg, WIRING_NUM_STEPS);
  if (!wiring) {
    fprintf(stderr, "FATAL: c_wiringgen training failed\n");
    return 1;
  }

  /* ── Evaluate ── */
  printf("\n=== Pipeline Evaluation (%d test intents) ===\n\n", g_test_n);

  int total = 0, parse_ok = 0, registry_ok = 0, judge_pass = 0;
  int exact_match = 0, partial_match = 0;
  int syntax_ok = 0, syntax_fail = 0, syntax_retries = 0;
  int wiring_attempted = 0, wiring_ok = 0;
  int parse_errors = 0, neural_judge_errors = 0;

  for (int t = 0; t < g_test_n; t++) {
    total++;
    OpaKanban kb;
    opa_kanban_init(&kb, 4);

    /* ── Step 1: Planner ── */
    char plan[256];
    generate_plan(planner, &cfg_pj, g_tests[t].comment, plan, sizeof(plan));

    char fns[MAX_PLAN_FNS][64];
    int nfns = parse_plan(plan, fns);
    int plan_parseable = (nfns > 0);
    if (plan_parseable)
      parse_ok++;
    else {
      parse_errors++;
      goto report;
    }

    /* Constrained decoding — fix near-miss function names */
    constrain_plan(fns, nfns);
    {
      char corrected[256] = "seq";
      for (int i = 0; i < nfns; i++) {
        strncat(corrected, "|", sizeof(corrected) - strlen(corrected) - 1);
        strncat(corrected, fns[i], sizeof(corrected) - strlen(corrected) - 1);
      }
      strncpy(plan, corrected, sizeof(plan) - 1);
      plan[sizeof(plan) - 1] = '\0';
    }

    /* Re-parse after correction */
    nfns = parse_plan(plan, fns);

    if (validate_plan_fns(fns, nfns))
      registry_ok++;

    /* ── Step 2: Judge plan (neural) ── */
    int verdict = judge_plan(judge, &cfg_pj, plan);
    if (verdict == 1)
      judge_pass++;
    else if (verdict < 0)
      neural_judge_errors++;

    if (strcmp(plan, g_tests[t].expected_plan) == 0) {
      exact_match++;
    } else {
      char exp_fns[MAX_PLAN_FNS][64];
      int exp_n = parse_plan(g_tests[t].expected_plan, exp_fns);
      int overlap = 0;
      for (int i = 0; i < nfns; i++)
        for (int j = 0; j < exp_n; j++)
          if (strcmp(fns[i], exp_fns[j]) == 0) {
            overlap++;
            break;
          }
      if (overlap > 0)
        partial_match++;
    }

    /* ── Step 3: Wiringgen → C body → Syntax Judge (Kanban retry) ── */
    if (nfns > 0 && validate_plan_fns(fns, nfns)) {
      wiring_attempted++;
      char wiring_prompt[512];
      char c_body[2048];
      int syntax_passed = 0;

      for (int attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        /* Build prompt from plan functions, excluding blocked patterns */
        plan_to_wiring_prompt(plan, fns, nfns, wiring_prompt,
                              sizeof(wiring_prompt));

        /* Append blocked hint if Kanban has failures */
        if (kb.stalls > 0 && strlen(kb.blocked) > 0) {
          size_t wp_len = strlen(wiring_prompt);
          /* Strip closing marker to add the blocked note */
          if (wp_len >= 3 && strcmp(wiring_prompt + wp_len - 3, "*/") == 0) {
            wiring_prompt[wp_len - 3] = '\0';
            size_t rem = sizeof(wiring_prompt) - strlen(wiring_prompt) - 1;
            strncat(wiring_prompt, "(retry, avoid: ", rem);
            rem = sizeof(wiring_prompt) - strlen(wiring_prompt) - 1;
            strncat(wiring_prompt, kb.blocked, rem);
            rem = sizeof(wiring_prompt) - strlen(wiring_prompt) - 1;
            strncat(wiring_prompt, ") */", rem);
          }
        }

        /* Generate C body from wiring organelle */
        organelle_generate(wiring, &cfg_wg, wiring_prompt, c_body,
                           sizeof(c_body), (scalar_t)WIRING_TEMP);

        /* Deterministic syntax gate */
        if (c_syntax_judge(c_body)) {
          syntax_passed = 1;
          syntax_ok++;
          wiring_ok++;
          break;
        } else {
          syntax_fail++;
          if (attempt < MAX_RETRIES) {
            syntax_retries++;
            /* Record failure in Kanban and retry */
            /* Use the first generated word as the "blocked pattern" token */
            char blocked_tok[32] = "syntax_err";
            /* Extract first identifier from c_body if present */
            const char *p = c_body;
            while (*p && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '/'))
              p++;
            if (*p) {
              int ti = 0;
              while (p[ti] && (p[ti] != ' ' && p[ti] != '\n' && p[ti] != '(') &&
                     ti < 31) {
                blocked_tok[ti] = p[ti];
                ti++;
              }
              blocked_tok[ti] = '\0';
            }
            opa_kanban_add_blocked(&kb, blocked_tok);
            kb.stalls++;
          }
        }
      }

      if (!syntax_passed) {
        /* All retries exhausted */
        c_body[0] = '\0';
      }

      /* Print the wiring result */
      printf("  [WIRE] prompt:  %s\n", wiring_prompt);
      if (syntax_passed) {
        printf("  [WIRE] result:  SYNTAX OK (attempt %d/%d)\n", kb.stalls + 1,
               MAX_RETRIES + 1);
        /* Print first 200 chars of generated body */
        int show = (int)strlen(c_body);
        if (show > 200)
          show = 200;
        printf("  [WIRE] body:    %.*s%s\n", show, c_body,
               (int)strlen(c_body) > 200 ? "..." : "");
      } else {
        printf("  [WIRE] result:  SYNTAX FAIL after %d retries\n",
               MAX_RETRIES + 1);
      }
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
  printf("Pipeline stage          | Count  | Rate\n");
  printf("------------------------|--------|------\n");
  printf("Test intents            | %6d | 100%%\n", total);
  printf("Plans parseable         | %6d | %.0f%%\n", parse_ok,
         100.0 * parse_ok / total);
  printf("All fns in registry     | %6d | %.0f%%\n", registry_ok,
         100.0 * registry_ok / total);
  printf("Neural judge PASS       | %6d | %.0f%%\n", judge_pass,
         100.0 * judge_pass / total);
  printf("Exact plan match        | %6d | %.0f%%\n", exact_match,
         100.0 * exact_match / total);
  printf("Partial plan match      | %6d | %.0f%%\n", partial_match,
         100.0 * partial_match / total);
  printf("------------------------+--------+------\n");
  printf("Wiring attempted        | %6d | %.0f%%\n", wiring_attempted,
         100.0 * wiring_attempted / total);
  printf("Wiring syntax OK        | %6d | %.0f%%\n", wiring_ok,
         wiring_attempted > 0 ? 100.0 * wiring_ok / wiring_attempted : 0.0);
  printf("Syntax failures         | %6d |\n", syntax_fail);
  printf("Syntax retries          | %6d |\n", syntax_retries);
  printf("------------------------+--------+------\n");
  printf("Parse errors            | %6d |\n", parse_errors);
  printf("Neural judge errors     | %6d |\n", neural_judge_errors);
  printf("\n");
  printf("Key: Wiring = c_wiringgen body gen gated by gcc -fsyntax-only\n");
  printf("     Syntax OK = valid C generated within %d attempts\n\n",
         MAX_RETRIES + 1);

  /* ── Cleanup ── */
  organelle_free(planner);
  organelle_free(judge);
  organelle_free(wiring);
  return 0;
}
