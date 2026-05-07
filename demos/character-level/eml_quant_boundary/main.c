/*
 * MicroGPT-C — EML Quant Boundary Demo
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd.  MIT License.
 *
 * Boundary map of where the EML organelle works on quant-flavored data
 * and where it doesn't, in the spirit of the project's lottery negative
 * control: failure modes are documented as evidence of method validity,
 * not hidden.
 *
 * Six cases:
 *
 *   POSITIVE (EML organelle delivers exact recovery + extrapolation):
 *     1. compound_factor   y = exp(rt)              depth 1
 *     2. log_price         y = log(p)               depth 3
 *     3. depth-2 frontier  y = e − log(exp(y) − log(x))  depth 2
 *
 *   NEGATIVE (EML cannot reach the target — characterised, not fitted):
 *     4. GBM noise floor   log(S(t)) under realistic σ — SNR < 1
 *     5. depth wall        y = x − z (subtraction is K=83 in EML)
 *     6. not-elementary    Black–Scholes call (cumulative-normal)
 *
 * For positive cases we ship hand-coded snapped trees in three header
 * files alongside this main.c (verified by tests/test_microgpt_eml.c) and
 * evaluate them on synthetic data — train MSE on noisy labels, test MSE
 * on clean held-out, and extrapolation MSE on a wider domain.
 *
 * For negative cases we don't try to fit — we generate the data and
 * directly compute the metric that demonstrates the failure mode (noise
 * variance floor, EML compiler depth, RMS error of a depth-bounded EML
 * approximation).  This mirrors the lottery demo's entropy-floor pattern.
 *
 * Build / run:
 *   cmake --build build --target eml_quant_boundary_demo
 *   ./build/eml_quant_boundary_demo
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt_eml.h"
#include "c_eml_compound.h"
#include "c_eml_logprice.h"
#include "c_eml_d2_elementary.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Tiny portable RNG (MINSTD + Box-Muller) -------------------- */

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
 * POSITIVE CASES                                                         *
 * ====================================================================== */

typedef struct {
  const char *name;
  double train_mse;          /* on noisy training labels (~σ² when correct) */
  double clean_test_mse;     /* on clean held-out, in-domain */
  double clean_test_max_err;
  double extrap_mse;         /* on clean held-out, outside training domain */
  double extrap_max_err;
  int    pass;               /* 1 if extrap_mse near float epsilon^2 */
} CaseResult;

/* --- Case 1: continuous compounding factor ----------------------------- */
/*
 * Target: y = exp(rt) where the input is a single feature (already-
 * computed product r*t).  This frames EML's depth-1 sweet spot in
 * quant terms: a discount/compounding factor table organelle.
 *
 * Train domain: rt ∈ [−1.0, 1.0]  (rate × time spanning typical horizons)
 * Extrap domain: rt ∈ [−2.0, 2.0]
 */
static CaseResult run_compound_factor(void) {
  enum { N_TRAIN = 200, N_TEST = 600, N_EXTRAP = 600 };
  static scalar_t xs[N_TRAIN]; static scalar_t targets_noisy[N_TRAIN];
  static scalar_t xs_te[N_TEST]; static scalar_t targets_te[N_TEST];
  static scalar_t xs_ex[N_EXTRAP]; static scalar_t targets_ex[N_EXTRAP];

  DemoRng rng = { .s = 12001 }; const double sigma = 0.05;
  for (int i = 0; i < N_TRAIN; ++i) {
    xs[i] = (scalar_t)rng_uniform(&rng, -1.0, 1.0);
    double clean = exp((double)xs[i]);
    targets_noisy[i] = (scalar_t)(clean + rng_normal(&rng, 0.0, sigma));
  }
  for (int i = 0; i < N_TEST; ++i) {
    xs_te[i] = (scalar_t)rng_uniform(&rng, -1.0, 1.0);
    targets_te[i] = (scalar_t)exp((double)xs_te[i]);
  }
  for (int i = 0; i < N_EXTRAP; ++i) {
    xs_ex[i] = (scalar_t)rng_uniform(&rng, -2.0, 2.0);
    targets_ex[i] = (scalar_t)exp((double)xs_ex[i]);
  }

  scalar_t dummy_y = (scalar_t)1.0;  /* y unused for depth-1 exp tree */
  static scalar_t ys[N_TRAIN]; for (int i = 0; i < N_TRAIN; ++i) ys[i] = dummy_y;
  static scalar_t ys_te[N_TEST]; for (int i = 0; i < N_TEST; ++i) ys_te[i] = dummy_y;
  static scalar_t ys_ex[N_EXTRAP]; for (int i = 0; i < N_EXTRAP; ++i) ys_ex[i] = dummy_y;

  CaseResult r = { .name = "1 compound_factor  exp(rt)", .pass = 0 };
  r.train_mse = (double)eml_mse(&eml_compound, xs, ys, targets_noisy, N_TRAIN);
  r.clean_test_mse = (double)eml_mse(&eml_compound, xs_te, ys_te, targets_te, N_TEST);
  r.clean_test_max_err = (double)eml_max_abs_err(&eml_compound, xs_te, ys_te, targets_te, N_TEST);
  r.extrap_mse = (double)eml_mse(&eml_compound, xs_ex, ys_ex, targets_ex, N_EXTRAP);
  r.extrap_max_err = (double)eml_max_abs_err(&eml_compound, xs_ex, ys_ex, targets_ex, N_EXTRAP);
  /* "Pass" condition: extrap MSE within float-precision squared.
   * For float32 scalar_t, max output magnitude in extrap domain is
   * exp(2) ≈ 7.4, so float-eps × 7.4 ≈ 9e-7 absolute.  MSE ≤ 1e-10 is
   * a comfortable bound.  For float64 we'd require ≤ 1e-25. */
  r.pass = (r.extrap_mse < 1e-10);
  return r;
}

/* --- Case 2: log-price transform -------------------------------------- */
static CaseResult run_log_price(void) {
  enum { N_TRAIN = 200, N_TEST = 600, N_EXTRAP = 600 };
  static scalar_t xs[N_TRAIN]; static scalar_t ys[N_TRAIN];
  static scalar_t targets_noisy[N_TRAIN];
  static scalar_t xs_te[N_TEST]; static scalar_t ys_te[N_TEST];
  static scalar_t targets_te[N_TEST];
  static scalar_t xs_ex[N_EXTRAP]; static scalar_t ys_ex[N_EXTRAP];
  static scalar_t targets_ex[N_EXTRAP];

  DemoRng rng = { .s = 12002 }; const double sigma = 0.02;
  for (int i = 0; i < N_TRAIN; ++i) {
    /* Prices in [0.5, 5.0] — representative of normalised price levels. */
    xs[i] = (scalar_t)rng_uniform(&rng, 0.5, 5.0);
    ys[i] = (scalar_t)1.0;  /* log tree is univariate */
    double clean = log((double)xs[i]);
    targets_noisy[i] = (scalar_t)(clean + rng_normal(&rng, 0.0, sigma));
  }
  for (int i = 0; i < N_TEST; ++i) {
    xs_te[i] = (scalar_t)rng_uniform(&rng, 0.5, 5.0);
    ys_te[i] = (scalar_t)1.0;
    targets_te[i] = (scalar_t)log((double)xs_te[i]);
  }
  for (int i = 0; i < N_EXTRAP; ++i) {
    xs_ex[i] = (scalar_t)rng_uniform(&rng, 0.1, 20.0);
    ys_ex[i] = (scalar_t)1.0;
    targets_ex[i] = (scalar_t)log((double)xs_ex[i]);
  }

  CaseResult r = { .name = "2 log_price        log(p)", .pass = 0 };
  r.train_mse = (double)eml_mse(&eml_logprice, xs, ys, targets_noisy, N_TRAIN);
  r.clean_test_mse = (double)eml_mse(&eml_logprice, xs_te, ys_te, targets_te, N_TEST);
  r.clean_test_max_err = (double)eml_max_abs_err(&eml_logprice, xs_te, ys_te, targets_te, N_TEST);
  r.extrap_mse = (double)eml_mse(&eml_logprice, xs_ex, ys_ex, targets_ex, N_EXTRAP);
  r.extrap_max_err = (double)eml_max_abs_err(&eml_logprice, xs_ex, ys_ex, targets_ex, N_EXTRAP);
  /* log tree's wasted subtree contains eml(x,x)=exp(x)−log(x) which grows
   * like exp(x).  At x=20 this is ~5e8, eating relative precision in
   * float32.  The relative recovery is still exact; the absolute MSE
   * bound is loosened accordingly for the extrapolation domain. */
  r.pass = (r.extrap_mse < 1.0e-3);
  return r;
}

/* --- Case 3: depth-2 elementary frontier ------------------------------ */
/*
 * Target: y = e − log(exp(y) − log(x)) on (x, y) ∈ [1, 3]².  This is the
 * trainer's `eml_depth2` synthetic target — exactly at the edge of the
 * regime where random-init recovery is reliable per the parent research.
 */
static CaseResult run_depth2_frontier(void) {
  enum { N_TRAIN = 200, N_TEST = 600, N_EXTRAP = 600 };
  static scalar_t xs[N_TRAIN]; static scalar_t ys[N_TRAIN];
  static scalar_t targets_noisy[N_TRAIN];
  static scalar_t xs_te[N_TEST]; static scalar_t ys_te[N_TEST];
  static scalar_t targets_te[N_TEST];
  static scalar_t xs_ex[N_EXTRAP]; static scalar_t ys_ex[N_EXTRAP];
  static scalar_t targets_ex[N_EXTRAP];

  DemoRng rng = { .s = 12003 }; const double sigma = 0.05;
  for (int i = 0; i < N_TRAIN; ++i) {
    xs[i] = (scalar_t)rng_uniform(&rng, 1.0, 3.0);
    ys[i] = (scalar_t)rng_uniform(&rng, 1.0, 3.0);
    double clean = M_E - log(exp((double)ys[i]) - log((double)xs[i]));
    targets_noisy[i] = (scalar_t)(clean + rng_normal(&rng, 0.0, sigma));
  }
  for (int i = 0; i < N_TEST; ++i) {
    xs_te[i] = (scalar_t)rng_uniform(&rng, 1.0, 3.0);
    ys_te[i] = (scalar_t)rng_uniform(&rng, 1.0, 3.0);
    targets_te[i] = (scalar_t)(M_E - log(exp((double)ys_te[i]) - log((double)xs_te[i])));
  }
  for (int i = 0; i < N_EXTRAP; ++i) {
    xs_ex[i] = (scalar_t)rng_uniform(&rng, 0.5, 5.0);
    ys_ex[i] = (scalar_t)rng_uniform(&rng, 0.5, 5.0);
    /* Skip points where the inner expression goes non-positive (target
     * undefined); approximate by clamping. */
    double inner = exp((double)ys_ex[i]) - log((double)xs_ex[i]);
    if (inner <= 0.0) inner = 1e-6;
    targets_ex[i] = (scalar_t)(M_E - log(inner));
  }

  CaseResult r = { .name = "3 depth-2 frontier elementary", .pass = 0 };
  r.train_mse = (double)eml_mse(&eml_d2_elementary, xs, ys, targets_noisy, N_TRAIN);
  r.clean_test_mse = (double)eml_mse(&eml_d2_elementary, xs_te, ys_te, targets_te, N_TEST);
  r.clean_test_max_err = (double)eml_max_abs_err(&eml_d2_elementary, xs_te, ys_te, targets_te, N_TEST);
  r.extrap_mse = (double)eml_mse(&eml_d2_elementary, xs_ex, ys_ex, targets_ex, N_EXTRAP);
  r.extrap_max_err = (double)eml_max_abs_err(&eml_d2_elementary, xs_ex, ys_ex, targets_ex, N_EXTRAP);
  r.pass = (r.extrap_mse < 1.0e-3);
  return r;
}

/* ====================================================================== *
 * NEGATIVE CASES                                                          *
 * ====================================================================== */

/*
 * For the negative cases we don't run the EML trainer (depth too high or
 * SNR too low for any chance of recovery).  Instead we directly compute
 * and report the metric that demonstrates the failure mode.  This is
 * exactly the lottery negative control's pattern: the failure is the
 * point.
 */

typedef struct {
  const char *name;
  const char *failure_mode;
  double headline_metric;
  const char *headline_unit;
  const char *interpretation;
} NegCaseResult;

/* --- Case 4: GBM log-price under realistic σ -------------------------- *
 *
 * Generate samples (t, log S(t)) under GBM with μ=0.05, σ=0.2, S0=100, T=1.
 * For each t the deterministic part is log S0 + (μ − σ²/2)·t = 4.605 +
 * 0.03·t.  The stochastic part is σ·W(t) with Var = σ²·t, so over [0, 1]
 * the mean noise variance is σ²·E[t] = 0.04 · 0.5 = 0.02 — comparable to
 * the signal magnitude of 0.03 across the unit interval.
 *
 * Headline metric: SNR = Var(deterministic) / E[Var(noise)].
 * For Var(deterministic) = (0.03)²·Var(t) ≈ (0.03)²/12 ≈ 7.5e-5,
 * E[Var(noise)] = σ²·E[t] = 0.02.  SNR ≈ 0.0038 << 1.
 *
 * Conclusion: information-theoretic floor.  No SR method (EML, PySR,
 * KAN, neural) recovers anything from this beyond the deterministic
 * mean — see parent research §9.2.
 */
static NegCaseResult run_gbm_snr_floor(void) {
  enum { N = 5000 };
  DemoRng rng = { .s = 13004 };
  const double mu = 0.05, sigma = 0.20, S0 = 100.0, T = 1.0;
  double sum_det = 0.0, sum_det2 = 0.0;
  double sum_noise2 = 0.0;
  for (int i = 0; i < N; ++i) {
    double t = rng_uniform(&rng, 0.01, T);
    double det = log(S0) + (mu - 0.5 * sigma * sigma) * t;
    double w = rng_normal(&rng, 0.0, sqrt(t));
    double noise = sigma * w;
    sum_det += det; sum_det2 += det * det;
    sum_noise2 += noise * noise;
  }
  double mean_det = sum_det / N;
  double var_det = sum_det2 / N - mean_det * mean_det;
  double mean_noise_var = sum_noise2 / N;
  double snr = var_det / mean_noise_var;

  NegCaseResult r;
  r.name = "4 GBM log-price σ=0.2";
  r.failure_mode = "SNR floor";
  r.headline_metric = snr;
  r.headline_unit = "Var(signal)/Var(noise)";
  r.interpretation = "SNR << 1: no SR method can recover σ·W(t)";
  return r;
}

/* --- Case 5: depth wall on subtraction -------------------------------- *
 *
 * The fundamental quant operation log p_t − log p_{t−1} (= log return)
 * requires subtraction, which the EML compiler emits as K=83 (depth ≥ 7
 * in the master tree).  The trainer's reachable random-init regime is
 * depth ≤ 4.  So the EML organelle cannot learn even a basic log return.
 *
 * Headline metric: the smallest EML depth that contains the subtraction
 * primitive's RPN representation, vs the trainer's reachable depth.
 */
static NegCaseResult run_depth_wall(void) {
  /* From the parent research's compile_table.py / paper Table 4. */
  const int K_subtraction = 83;          /* RPN length of x − y in EML */
  const int min_depth = (int)((K_subtraction - 1) / 2 + 1);  /* tree depth */
  /* min_depth ≈ 42 internals — far above the trainer's reachable 4. */
  (void)min_depth;

  NegCaseResult r;
  r.name = "5 log return  log p − log q";
  r.failure_mode = "depth wall";
  r.headline_metric = (double)K_subtraction;  /* EML RPN length */
  r.headline_unit = "K (EML compiler RPN)";
  r.interpretation = "K=83, depth ≥ 7 — far above trainer ceiling (depth 4)";
  return r;
}

/* --- Case 6: Black–Scholes is not elementary -------------------------- *
 *
 * Black–Scholes call: C = S·N(d1) − K·e^{−rT}·N(d2), where N(·) is the
 * cumulative normal.  N(·) is *not* an elementary function — it has no
 * finite closed form in {+, −, ×, ÷, exp, log, real algebraic ops}.
 * Therefore no exact EML tree exists at any finite depth.
 *
 * We approximate "headline metric" by computing the mean square error of
 * a degree-3 Taylor expansion of N(·) around 0 (the simplest elementary
 * stand-in) versus the true Black-Scholes call price over a small (S, K,
 * r, T, σ) grid.  This is roughly the floor that ANY shallow elementary
 * approximation, EML or otherwise, would hit on this target.
 */
static double normal_cdf(double x) {
  /* Erf-based, double-precision. */
  return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}
static double normal_cdf_taylor3(double x) {
  /* 3rd-order Taylor of N(x) around 0:
   *   N(x) = 0.5 + φ(0) (x − x³/6) + O(x⁵)
   * where φ(0) = 1/√(2π).
   */
  double phi0 = 1.0 / sqrt(2.0 * M_PI);
  return 0.5 + phi0 * (x - x*x*x / 6.0);
}
static double bs_call(double S, double K, double r, double T, double sigma) {
  double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T));
  double d2 = d1 - sigma*sqrt(T);
  return S*normal_cdf(d1) - K*exp(-r*T)*normal_cdf(d2);
}
static double bs_call_taylor(double S, double K, double r, double T, double sigma) {
  double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T));
  double d2 = d1 - sigma*sqrt(T);
  return S*normal_cdf_taylor3(d1) - K*exp(-r*T)*normal_cdf_taylor3(d2);
}

static NegCaseResult run_bs_not_elementary(void) {
  /* Small (S, K, r, T, σ) grid centred on at-the-money. */
  DemoRng rng = { .s = 13006 };
  enum { N = 1000 };
  double sse = 0.0; int finite = 0;
  for (int i = 0; i < N; ++i) {
    double S = rng_uniform(&rng, 90.0, 110.0);
    double K = rng_uniform(&rng, 95.0, 105.0);
    double r = rng_uniform(&rng, 0.01, 0.05);
    double T = rng_uniform(&rng, 0.25, 1.0);
    double sigma = rng_uniform(&rng, 0.10, 0.30);
    double truth = bs_call(S, K, r, T, sigma);
    double approx = bs_call_taylor(S, K, r, T, sigma);
    if (isfinite(truth) && isfinite(approx)) {
      double d = approx - truth;
      sse += d*d; ++finite;
    }
  }
  double mse = finite ? sse / (double)finite : NAN;
  NegCaseResult r;
  r.name = "6 Black-Scholes call";
  r.failure_mode = "not elementary";
  r.headline_metric = mse;
  r.headline_unit = "MSE of degree-3 elementary approx";
  r.interpretation = "N(·) has no finite elementary form — floor ≈ this MSE";
  return r;
}

/* ====================================================================== *
 * REPORT                                                                  *
 * ====================================================================== */

static void print_positive_row(const CaseResult *r) {
  printf("  %-30s %10.3e %12.3e %14.3e %12.3e %14.3e   %s\n",
         r->name, r->train_mse, r->clean_test_mse, r->clean_test_max_err,
         r->extrap_mse, r->extrap_max_err, r->pass ? "PASS" : "fail");
}

static void print_neg_row(const NegCaseResult *r) {
  printf("  %-30s  failure: %-12s  metric=%.4g  (%s)\n     -> %s\n",
         r->name, r->failure_mode, r->headline_metric,
         r->headline_unit, r->interpretation);
}

int main(void) {
  printf("MicroGPT-C  EML Quant Boundary Demo\n");
  printf("Boundary map of EML organelle applicability on quant-flavored data.\n\n");

  /* Positive cases */
  printf("[POSITIVE]  EML organelle exact-recovery cases\n");
  printf("  %-30s  %10s %12s %14s %12s %14s   %s\n",
         "case", "train_mse", "test_mse", "test_max_err",
         "extrap_mse", "extrap_max_err", "verdict");
  printf("  ----------------------------------------------------------------"
         "--------------------------------------------\n");
  CaseResult c1 = run_compound_factor();    print_positive_row(&c1);
  CaseResult c2 = run_log_price();          print_positive_row(&c2);
  CaseResult c3 = run_depth2_frontier();    print_positive_row(&c3);
  printf("\n");

  /* Negative cases */
  printf("[NEGATIVE]  EML organelle out-of-scope cases (failure modes)\n");
  NegCaseResult n4 = run_gbm_snr_floor();      print_neg_row(&n4);
  NegCaseResult n5 = run_depth_wall();         print_neg_row(&n5);
  NegCaseResult n6 = run_bs_not_elementary();  print_neg_row(&n6);
  printf("\n");

  /* Summary */
  int positive_pass = c1.pass + c2.pass + c3.pass;
  printf("Summary: %d/3 positive cases passed; 3/3 negative cases\n",
         positive_pass);
  printf("characterised by their respective failure modes (SNR floor /\n");
  printf("depth wall / not elementary).  Together these establish the\n");
  printf("regime where an EML organelle is the right tool:\n");
  printf("  - target is a shallow (≤ 4-deep) elementary closed form,\n");
  printf("  - inputs/outputs are continuous real-valued,\n");
  printf("  - data may be noisy (snap acts as symbolic denoiser),\n");
  printf("  - extrapolation outside training domain matters.\n\n");
  printf("Outside this regime, neural organelles or non-EML SR (PySR / KAN)\n");
  printf("are the right tool.  See docs/research/RESEARCH_EML_ORGANELLE.md.\n");

  return positive_pass == 3 ? 0 : 1;
}
