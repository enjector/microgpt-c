/*
 * MicroGPT-C — EML Hybrid OPA Pipeline Demo
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd.  MIT License.
 *
 * Demonstrates neuro-symbolic composition in the OPA style:
 *
 *   - A tiny neural organelle (Direction Predictor) handles the fuzzy
 *     pattern-matching task:  given a few recent direction tokens
 *     {U, D, F} = up / down / flat, predict the next token.  Trained
 *     char-level on a small corpus of synthetic sequences with autocorr.
 *
 *   - Two EML organelles handle the deterministic numeric transforms:
 *       LogPrice    : y = log(p)        (depth-3 EML tree)
 *       Compound    : y = exp(rt)       (depth-1 EML tree)
 *     Both are exact, deterministic, no checkpoint to ship.  See
 *     docs/research/RESEARCH_EML_ORGANELLE.md for the underlying snap
 *     and capacity-confirmation story.
 *
 *   - An OpaKanban shared state tracks the rolling direction history.
 *     The cycle detector breaks degenerate U/D oscillations the neural
 *     organelle might fall into when extrapolating (the same role it
 *     plays in the connect4 / tictactoe demos).
 *
 * Per-step output:
 *   (price, rt, recent_history) ─▶ pipeline ─▶ (predicted_dir,
 *                                              log_price,
 *                                              discount_factor)
 *
 * The point of the demo is *complementarity*, not benchmarking against
 * a single-organelle baseline:
 *   - The neural organelle, alone, can only emit categorical tokens; it
 *     cannot produce log-prices or discount factors.
 *   - The EML organelles, alone, cannot predict direction from a
 *     sequence of categorical observations — that's not an elementary
 *     closed-form relation.
 *   - Together, with the deterministic Kanban routing between them, the
 *     pipeline produces a structured (direction, log_price, discount)
 *     output that no single organelle could.
 *
 * Build / run:
 *   cmake --build build --target eml_hybrid_pipeline_demo
 *   ./build/eml_hybrid_pipeline_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"
#include "microgpt_eml.h"
#include "c_eml_logprice.h"
#include "c_eml_compound.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Configuration ---- */
#define DIRECTION_CORPUS "c_directions.txt"
#define DIRECTION_CKPT   "c_directions.ckpt"

#define DIRECTION_TEMP   0.3   /* low-temp sampling; we want the dominant pattern */
#define INF_GEN_LEN      8     /* a few tokens after the prompt is plenty */
#define HISTORY_LEN      6     /* tokens fed to the predictor each step */
#define PIPELINE_STEPS   10    /* simulated time steps in the demo */

#define KANBAN_HISTORY   8

static MicrogptConfig g_cfg;

/* Token IDs for the cycle detector (which works on integer actions). */
#define DIR_U 1
#define DIR_D 2
#define DIR_F 3

static int dir_token_id(char c) {
  switch (c) {
    case 'U': return DIR_U;
    case 'D': return DIR_D;
    case 'F': return DIR_F;
    default:  return 0;
  }
}

/* ---- Synthetic streaming source -------------------------------------- */

/*
 * Tiny portable RNG (Park-Miller minimal-standard LCG) so the demo prints
 * the same table on every run regardless of platform.
 */
typedef struct { unsigned long s; } DemoRng;
static double rng_uniform(DemoRng *r, double lo, double hi) {
  r->s = (r->s * 48271UL) % 2147483647UL;
  return lo + (hi - lo) * ((double)r->s / 2147483647.0);
}

/*
 * Generate the next observation tuple:
 *   - true_dir: ground truth direction this step (drawn with autocorr)
 *   - price:    a positive scalar in [0.5, 5.0]
 *   - rt:       rate × time in [-0.5, 0.5]  (already-multiplied input;
 *               the multiplication itself is depth-8 in EML so it stays
 *               outside the EML organelle's responsibility — exactly
 *               the boundary-map principle from §Boundary Map of the
 *               RESEARCH_EML_ORGANELLE doc).
 */
typedef struct {
  char true_dir;
  double price;
  double rt;
} StreamObs;

static StreamObs next_observation(DemoRng *rng, char prev_dir) {
  StreamObs obs;
  /* 70 % autocorrelation: most often repeats the previous direction.
   * This is the structural pattern the neural organelle is trained on. */
  double u = rng_uniform(rng, 0.0, 1.0);
  if (prev_dir != '\0' && u < 0.70) {
    obs.true_dir = prev_dir;
  } else {
    /* Flip to one of the other two with equal weight. */
    double v = rng_uniform(rng, 0.0, 1.0);
    if (prev_dir == 'U') obs.true_dir = (v < 0.5) ? 'D' : 'F';
    else if (prev_dir == 'D') obs.true_dir = (v < 0.5) ? 'U' : 'F';
    else if (prev_dir == 'F') obs.true_dir = (v < 0.5) ? 'U' : 'D';
    else obs.true_dir = (v < 0.33) ? 'U' : (v < 0.66) ? 'D' : 'F';
  }
  obs.price = rng_uniform(rng, 0.5, 5.0);
  obs.rt = rng_uniform(rng, -0.5, 0.5);
  return obs;
}

/* ---- Output of a single pipeline step ------------------------------- */

typedef struct {
  char true_dir;
  char predicted_dir;
  double price;
  double log_price;       /* from EML organelle */
  double rt;
  double discount_factor; /* from EML organelle */
  int    cycle_broken;    /* 1 if the cycle detector overrode the predictor */
  char   history[HISTORY_LEN + 1];
} StepResult;

/* ====================================================================== *
 * Pipeline                                                                *
 * ====================================================================== */

/*
 * One step of the hybrid pipeline:
 *   1. Read the rolling direction history from kanban.
 *   2. Neural organelle predicts the next direction token.
 *   3. Cycle detector checks for U↔D oscillation and may override.
 *   4. Two EML organelles compute log(price) and exp(rt) deterministically.
 *   5. Update kanban with the chosen direction.
 *
 * The neural organelle's output is the only stochastic piece; every other
 * step is deterministic C scaffolding (Kanban + cycle detector + EML).
 */
static StepResult run_pipeline_step(const Organelle *predictor,
                                    OpaKanban *kb,
                                    OpaCycleDetector *cd,
                                    const StreamObs *obs,
                                    const char *history) {
  StepResult res;
  res.true_dir = obs->true_dir;
  res.price = obs->price;
  res.rt = obs->rt;
  memcpy(res.history, history, HISTORY_LEN);
  res.history[HISTORY_LEN] = '\0';

  /* ----- Neural organelle: predict next direction ----- */
  char prompt[HISTORY_LEN + 8];
  snprintf(prompt, sizeof(prompt), "%s", history);
  char gen_out[INF_GEN_LEN + 1] = {0};
  organelle_generate(predictor, &g_cfg, prompt, gen_out, INF_GEN_LEN,
                     (scalar_t)DIRECTION_TEMP);
  /* Take the first U/D/F token from the generation, ignoring whitespace. */
  char predicted = '\0';
  for (size_t i = 0; gen_out[i]; ++i) {
    if (gen_out[i] == 'U' || gen_out[i] == 'D' || gen_out[i] == 'F') {
      predicted = gen_out[i];
      break;
    }
  }
  if (predicted == '\0') predicted = history[HISTORY_LEN - 1];  /* fallback */

  /* ----- Cycle detector: break U↔D oscillations ----- */
  res.cycle_broken = 0;
  int proposed_id = dir_token_id(predicted);
  if (proposed_id != 0 && opa_cycle_detected(cd, proposed_id)) {
    int other = opa_cycle_other(cd, proposed_id);
    /* Prefer F (flat) over swapping inside the cycle, since flat is the
     * "do-nothing" option in this categorical scheme. */
    (void)other;
    predicted = 'F';
    res.cycle_broken = 1;
    proposed_id = DIR_F;
  }
  res.predicted_dir = predicted;
  opa_cycle_record(cd, proposed_id);

  /* ----- EML organelle 1: log-price transform ----- */
  res.log_price = (double)eml_eval(&eml_logprice, (scalar_t)obs->price,
                                   (scalar_t)1.0);

  /* ----- EML organelle 2: compounding factor ----- */
  res.discount_factor = (double)eml_eval(&eml_compound, (scalar_t)obs->rt,
                                         (scalar_t)1.0);

  /* ----- Kanban: store last decision ----- */
  char last_str[2] = { predicted, '\0' };
  opa_kanban_add_last(kb, last_str);
  return res;
}

/* ---- History helper ------------------------------------------------- */
static void roll_history(char *history, char next) {
  memmove(history, history + 1, HISTORY_LEN - 1);
  history[HISTORY_LEN - 1] = next;
}

/* ====================================================================== *
 * Main                                                                    *
 * ====================================================================== */

int main(void) {
  printf("MicroGPT-C  EML Hybrid OPA Pipeline Demo\n");
  printf("Neural direction predictor + EML deterministic transforms\n");
  printf("(see docs/research/RESEARCH_EML_ORGANELLE.md §Boundary Map)\n\n");

  g_cfg = microgpt_default_config();

  /* Train (or resume) the tiny direction predictor. */
  printf("Training direction-predictor organelle (resumable from %s)...\n",
         DIRECTION_CKPT);
  Organelle *predictor = organelle_train("DirPredictor", DIRECTION_CORPUS,
                                         DIRECTION_CKPT, &g_cfg, 600);
  if (!predictor) {
    fprintf(stderr, "Failed to train DirPredictor.\n");
    return 1;
  }
  printf("Predictor ready (vocab=%zu, %zu documents).\n\n",
         predictor->vocab.vocab_size, predictor->docs.num_docs);

  /* Initialise OPA scaffolding. */
  OpaKanban kb;        opa_kanban_init(&kb, KANBAN_HISTORY);
  OpaCycleDetector cd; opa_cycle_init(&cd);

  /* Seed history with a known starting pattern. */
  char history[HISTORY_LEN + 1] = "UUUDDF";

  /* Reproducible synthetic stream. */
  DemoRng rng = { .s = 14001 };

  /* Run PIPELINE_STEPS pipeline steps and print a unified table. */
  printf(" step  history        true  pred  cyc   price    log(p)        rt  exp(rt)\n");
  printf(" ----  -------------  ----  ----  ---  ------  --------   ------  -------\n");
  int correct = 0;
  for (int t = 0; t < PIPELINE_STEPS; ++t) {
    StreamObs obs = next_observation(&rng,
        history[HISTORY_LEN - 1]);
    StepResult r = run_pipeline_step(predictor, &kb, &cd, &obs, history);
    int hit = (r.predicted_dir == r.true_dir) ? 1 : 0;
    correct += hit;
    printf(" %4d  %-13s   %c     %c    %s   %5.2f   %7.4f  %+6.3f   %5.3f\n",
           t, r.history, r.true_dir, r.predicted_dir,
           r.cycle_broken ? "yes" : " no ",
           r.price, r.log_price, r.rt, r.discount_factor);
    roll_history(history, r.true_dir);
  }
  printf("\n");
  printf("Direction predictor accuracy: %d/%d  (random baseline ≈ 33%%)\n",
         correct, PIPELINE_STEPS);

  /* Sanity self-check: the EML organelles are deterministic, so we can
   * verify them against math.h on a fixed grid right here, with the same
   * tolerance discipline as test_microgpt_eml.c. */
  {
    double max_err_log = 0.0, max_err_exp = 0.0;
    for (int i = 0; i < 50; ++i) {
      double p = 0.5 + (double)i * 0.1;
      scalar_t got = eml_eval(&eml_logprice, (scalar_t)p, (scalar_t)1.0);
      double want = log(p);
      double err = fabs((double)got - want);
      if (err > max_err_log) max_err_log = err;

      double rt = -2.0 + (double)i * 0.08;
      scalar_t gotc = eml_eval(&eml_compound, (scalar_t)rt, (scalar_t)1.0);
      double wantc = exp(rt);
      double errc = fabs((double)gotc - wantc);
      if (errc > max_err_exp) max_err_exp = errc;
    }
    printf("EML self-checks (math.h reference):\n");
    printf("  log(p) tree max abs err over 50 points: %.3e\n", max_err_log);
    printf("  exp(x) tree max abs err over 50 points: %.3e\n", max_err_exp);
  }
  printf("\nKanban final state: history=\"%s\", stalls=%d\n", kb.last,
         kb.stalls);
  printf("\nPipeline composition note:\n");
  printf("  * neural organelle handled categorical pattern matching only\n");
  printf("  * EML organelles handled all numeric transforms deterministically\n");
  printf("  * the cycle detector overrode the neural prediction when it\n");
  printf("    would have continued a U-D oscillation (\"cyc=yes\" rows above)\n");
  printf("  This is OPA composition: the neural part learns *what gradient\n");
  printf("  descent is good at*; the deterministic C scaffolding (Kanban +\n");
  printf("  cycle detector) and the EML organelles handle *what gradient\n");
  printf("  descent is wasteful at*.  Same total compute budget, sharper\n");
  printf("  allocation per the OPA philosophy.\n");

  /* organelle_free uses the project's standard free; safe to call on the
   * predictor once we're done.  microgpt_organelle.c handles the inner
   * allocations. */
  return 0;
}
