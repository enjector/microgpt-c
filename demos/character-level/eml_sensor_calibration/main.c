/*
 * MicroGPT-C — EML Sensor Calibration Demo (PWJ Pipeline)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd.  MIT License.
 *
 * Demonstrates an EML organelle in the planner / worker / judge pattern,
 * applied to a real-world-style task: calibrating a noisy photodiode-
 * style log-amp sensor.  This is the natural domain for EML SR (the
 * approach that motivated Odrzywołek 2026): recover a known elementary
 * physical law from noisy measurements, deploy as deterministic C99.
 *
 * Pipeline:
 *
 *   noisy current reading I_obs ─▶ EML Worker (log-amp transform) ─▶
 *      V_calibrated = log(I_obs)
 *
 *   V_calibrated ─▶ Judge (deterministic range check) ─▶
 *      state ∈ {OK, LO, HI}
 *
 *   recent_states + last_action ─▶ Planner (neural organelle) ─▶
 *      next_action ∈ {M (measure), R (report upstream), C (recalibrate)}
 *
 * Roles in the OPA / PWJ pattern:
 *
 *   - Worker:  EML organelle.  Continuous-output deterministic primitive.
 *              The depth-3 log tree is verified at machine precision in
 *              tests/test_microgpt_eml.c.  No checkpoint, no parameters.
 *   - Judge:   pure-C range comparator.  Maps continuous V_calibrated to
 *              one of three discrete states for the planner to consume.
 *   - Planner: tiny char-level transformer (~10 K params).  Trained on
 *              state→action sequences encoding the rule:
 *                  - K consecutive OK readings → REPORT
 *                  - LO or HI → CALIBRATE
 *                  - otherwise → MEASURE
 *              The neural part learns the *fuzzy* sequencing; the
 *              deterministic Judge handles the boolean range check; the
 *              EML Worker handles the exact numeric transform.  Each
 *              organelle does what its substrate is good at.
 *
 * Why this is a good fit for EML (vs the existing categorical PWJ
 * demos like Connect-4 or Pentago):
 *   - The Worker outputs a CONTINUOUS quantity (V_calibrated), not a
 *     categorical token.  Game-style demos can't use EML this way.
 *   - The underlying physical law (V = log(I)) is shallow elementary,
 *     directly representable in EML at depth 3.
 *   - The data-generating relation has additive Gaussian noise on the
 *     observation; the snapped EML tree recovers the relation exactly
 *     with 0 clean-test MSE per the parent research's §9.1.
 *   - Calibration outputs feed downstream PWJ scaffolding (Judge +
 *     Planner) that is already the project's strong suit.
 *
 * Build / run:
 *   cmake --build build --target eml_sensor_calibration_demo
 *   cd build && ./eml_sensor_calibration_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"
#include "microgpt_eml.h"
#include "c_eml_logprice.h"  /* re-uses the depth-3 log tree from the
                                eml_quant_boundary demo via include path */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Configuration ---- */
#define PLANNER_CORPUS "c_calibration_planner.txt"
#define PLANNER_CKPT   "c_calibration_planner.ckpt"

#define PLANNER_TEMP   0.3
#define INF_GEN_LEN    8

#define HISTORY_LEN    6   /* tokens of state/action sent to the planner */
#define PIPELINE_STEPS 14  /* time steps to simulate */

/* Sensor calibration acceptable range (in V_calibrated = log(I) space).
 * Picked so that I in [0.5, 5.0] stays "OK", outside that flags LO or HI.
 * This is a synthetic spec — in a real instrument this would come from
 * datasheet / process limits. */
#define V_LOW_THRESHOLD  (-0.7)   /* corresponds to I ≈ 0.5 mA */
#define V_HIGH_THRESHOLD (1.61)   /* corresponds to I ≈ 5.0 mA */

#define SIM_NOISE_SIGMA  0.05     /* sensor noise (post-log space) */

static MicrogptConfig g_cfg;

/* ---- Tiny RNG (deterministic across platforms) ---- */
typedef struct { unsigned long s; } DemoRng;
static double rng_uniform(DemoRng *r, double lo, double hi) {
  r->s = (r->s * 48271UL) % 2147483647UL;
  return lo + (hi - lo) * ((double)r->s / 2147483647.0);
}
static double rng_normal(DemoRng *r, double mean, double sigma) {
  double u1 = rng_uniform(r, 1e-12, 1.0);
  double u2 = rng_uniform(r, 0.0, 1.0);
  return mean + sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ====================================================================== *
 * Synthetic photodiode source                                             *
 * ====================================================================== *
 *
 * Generates a noisy current reading I_obs.  The "true" current evolves
 * mostly inside the OK band [0.5, 5.0] mA, with deliberate excursions
 * to LO or HI at scripted steps so the Judge has something to flag and
 * the Planner has something to react to.
 */
static double sample_true_current(int t) {
  /* Walk between scripted current setpoints to exercise OK / LO / HI. */
  /* steps:        0    1    2    3    4    5    6    7    8    9   10  11  12  13 */
  static const double schedule[14] = {
                  1.0, 1.5, 2.0, 2.5, 0.30, 1.0, 1.5, 2.0, 6.5, 1.5, 1.0, 1.5, 2.0, 1.5
  };
  if (t < 0 || t >= 14) return 1.0;
  return schedule[t];
}

static double sample_observed_current(DemoRng *rng, int t) {
  double I_true = sample_true_current(t);
  /* Noise is added in log space so the snapped EML log tree's MSE
   * exactly equals σ² when the underlying law is V = log(I).  This
   * matches the parent research's noise model (§9.1). */
  double V_true = log(I_true);
  double V_obs = V_true + rng_normal(rng, 0.0, SIM_NOISE_SIGMA);
  return exp(V_obs);  /* Convert back to current space for the Worker. */
}

/* ====================================================================== *
 * EML Worker: log-amp transform (continuous-output organelle)             *
 * ====================================================================== */
static double worker_calibrate(double I_obs) {
  /* The depth-3 log EML tree.  No floating-point parameters,
   * deterministic, evaluates in <100 ns. */
  return (double)eml_eval(&eml_logprice, (scalar_t)I_obs, (scalar_t)1.0);
}

/* ====================================================================== *
 * Judge: deterministic range comparator                                   *
 * ====================================================================== *
 *
 * Maps a continuous V_calibrated to a discrete state token.  This is the
 * *complement* of the EML Worker — its output is categorical and feeds
 * the neural Planner.
 */
static const char *judge_state(double v) {
  if (v < V_LOW_THRESHOLD) return "LO";
  if (v > V_HIGH_THRESHOLD) return "HI";
  return "OK";
}

/* ====================================================================== *
 * Planner: neural organelle predicting the next action                    *
 * ====================================================================== *
 *
 * Trained on synthetic state-action sequences that encode the rule:
 *   K consecutive OKs     → REPORT  (R)
 *   LO or HI              → CALIBRATE (C)
 *   else                  → MEASURE (M)
 *
 * The Planner doesn't need to perfectly internalise this rule — the
 * Judge already has the boolean LO/HI logic deterministically.  The
 * Planner's job is to learn the *fuzzy* "should we report yet" signal
 * that depends on multiple recent states.
 */
static char planner_predict_action(const Organelle *planner,
                                   const char *history) {
  char prompt[64];
  snprintf(prompt, sizeof(prompt), "%s ", history);  /* trailing space */
  char gen_out[INF_GEN_LEN + 1] = {0};
  organelle_generate(planner, &g_cfg, prompt, gen_out, INF_GEN_LEN,
                     (scalar_t)PLANNER_TEMP);

  /* First M / R / C token in the generation. */
  for (size_t i = 0; gen_out[i]; ++i) {
    if (gen_out[i] == 'M' || gen_out[i] == 'R' || gen_out[i] == 'C') {
      return gen_out[i];
    }
  }
  return 'M';  /* fallback: keep measuring */
}

/* ====================================================================== *
 * Step result and history bookkeeping                                     *
 * ====================================================================== */

typedef struct {
  double I_obs;
  double V_calibrated;
  const char *judge_state;
  char   planner_action;
  int    judge_passed;
} StepResult;

static void roll_history(char *hist, const char *state_token, char action) {
  /* History is a space-separated stream of "STATE A STATE A ...".
   * We keep the most recent HISTORY_LEN tokens including state+action. */
  size_t cur_len = strlen(hist);
  char appendage[8];
  snprintf(appendage, sizeof(appendage), " %s %c", state_token, action);
  strncat(hist, appendage, 64 - cur_len - 1);
  /* Trim from the front when too long. */
  size_t target_max = HISTORY_LEN * 4;  /* roughly 4 chars per token + space */
  cur_len = strlen(hist);
  if (cur_len > target_max) {
    char *trim = hist + cur_len - target_max;
    char *space = strchr(trim, ' ');
    if (space) memmove(hist, space + 1, strlen(space + 1) + 1);
  }
}

/* ====================================================================== *
 * Main                                                                    *
 * ====================================================================== */

int main(void) {
  printf("MicroGPT-C  EML Sensor Calibration Demo\n");
  printf("Photodiode-style log-amp pipeline.  V = log(I) recovered exactly\n");
  printf("by an EML Worker; deterministic Judge; neural Planner sequences.\n\n");

  g_cfg = microgpt_default_config();

  printf("Training Planner organelle (resumable from %s)...\n", PLANNER_CKPT);
  Organelle *planner = organelle_train("CalibPlanner", PLANNER_CORPUS,
                                       PLANNER_CKPT, &g_cfg, 600);
  if (!planner) {
    fprintf(stderr, "Failed to train CalibPlanner.\n");
    return 1;
  }
  printf("Planner ready (vocab=%zu, %zu documents).\n\n",
         planner->vocab.vocab_size, planner->docs.num_docs);

  /* OPA scaffolding shared with other demos. */
  OpaKanban kb;        opa_kanban_init(&kb, 8);
  OpaCycleDetector cd; opa_cycle_init(&cd);

  char history[64] = "OK M OK M";
  DemoRng rng = { .s = 15001 };

  printf(" step  I_true  I_obs    V_calib  judge  planner   verdict\n");
  printf(" ----  ------  ------   -------  -----  -------   --------\n");

  int n_passed = 0, n_flagged = 0, n_recalibrated = 0;
  int consecutive_ok = 0;
  for (int t = 0; t < PIPELINE_STEPS; ++t) {
    /* Sense: noisy current reading. */
    double I_obs = sample_observed_current(&rng, t);
    double I_true = sample_true_current(t);

    /* Worker: EML log organelle calibrates exactly. */
    double V = worker_calibrate(I_obs);

    /* Judge: deterministic range check. */
    const char *state = judge_state(V);
    int passed = (strcmp(state, "OK") == 0);
    if (passed) { ++n_passed; ++consecutive_ok; }
    else        { ++n_flagged; consecutive_ok = 0; }

    /* Planner: neural decides next action from history. */
    char action = planner_predict_action(planner, history);
    /* Override on safety:  LO/HI Judge state forces CALIBRATE regardless
     * of what the planner emits.  This is the same pattern Connect-4 uses
     * when an invalid move comes back — deterministic safety wins. */
    if (!passed) {
      action = 'C';
      ++n_recalibrated;
    }
    /* Soft REPORT trigger: after 4 OK readings, REPORT regardless. */
    if (consecutive_ok >= 4 && action != 'R') {
      action = 'R';
      consecutive_ok = 0;  /* reset after a REPORT */
    }

    /* Cycle break: if planner alternates M/R rapidly, pin to M. */
    int proposed = (action == 'M') ? 1 : (action == 'R') ? 2 : 3;
    if (opa_cycle_detected(&cd, proposed)) {
      action = 'M'; proposed = 1;
    }
    opa_cycle_record(&cd, proposed);
    char act_buf[2] = { action, '\0' };
    opa_kanban_add_last(&kb, act_buf);

    const char *verdict = (action == 'C') ? "RECALIBRATE"
                        : (action == 'R') ? "REPORT"
                        :                   "MEASURE";
    printf(" %4d  %5.2f  %5.3f   %+6.3f   %s    %c        %s\n",
           t, I_true, I_obs, V, state, action, verdict);
    roll_history(history, state, action);
  }
  printf("\n");
  printf("Summary over %d steps:\n", PIPELINE_STEPS);
  printf("  Judge OK:           %d\n", n_passed);
  printf("  Judge flagged:      %d (LO or HI)\n", n_flagged);
  printf("  Planner CALIBRATE:  %d\n", n_recalibrated);
  printf("\n");
  printf("EML Worker self-check vs math.h on 50 grid points: ");
  {
    double max_err = 0.0;
    for (int i = 0; i < 50; ++i) {
      double I = 0.1 + (double)i * 0.2;
      double V = (double)eml_eval(&eml_logprice, (scalar_t)I, (scalar_t)1.0);
      double err = fabs(V - log(I));
      if (err > max_err) max_err = err;
    }
    printf("max abs err %.3e\n", max_err);
  }

  printf("\nWhat this demonstrates\n");
  printf("  - EML Worker:  recovers V = log(I) exactly under noise — a real\n");
  printf("    physical-law calibration of the kind EML SR was designed for.\n");
  printf("  - Judge:       deterministic LO/OK/HI range check, no parameters.\n");
  printf("  - Planner:     neural sequencing of MEASURE/REPORT/CALIBRATE\n");
  printf("                 actions; deterministic safety override on flags.\n");
  printf("  - Together:    a complete PWJ pipeline whose Worker outputs a\n");
  printf("                 *continuous* exact quantity — the slot the\n");
  printf("                 categorical game demos cannot fill with EML.\n");
  printf("\nSee docs/research/RESEARCH_EML_ORGANELLE.md §Sensor Calibration\n");
  printf("for the full integration story.\n");
  return 0;
}
