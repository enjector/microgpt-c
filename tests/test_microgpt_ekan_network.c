/*
 * test_ekan_network.c — Tests for microgpt_ekan_network.h
 *
 * Verifies correctness of:
 *   1. Fourier basis evaluation and derivatives
 *   2. Network initialization (Glorot, gate alpha)
 *   3. Forward pass (output dimensions, bounded output)
 *   4. Autoencoder training convergence
 *   5. Reconstruction error decreases with training
 *   6. Sensitivity analysis (perturbation-based)
 *   7. Adam optimizer moment updates
 *   8. Gradient clipping
 */

#include "microgpt_ekan_network.h"
#include "test.h"
#include <math.h>
#include <stdio.h>

/* ── Fourier Basis Tests ── */

enx_test(test_fourier_eval_dc) {
  /* a₀=1.0, all others zero → should return 1.0 */
  double coeffs[EKAN_NET_FOURIER_COEFFS] = {0};
  coeffs[0] = 1.0;
  double val = ekan_fourier_eval(coeffs, EKAN_NET_FOURIER_N, 0.5);
  enx_assert_true(fabs(val - 1.0) < 1e-10);
}

enx_test(test_fourier_eval_cos) {
  /* a₁=1.0 → cos(πx), at x=0: cos(0) = 1.0 */
  double coeffs[EKAN_NET_FOURIER_COEFFS] = {0};
  coeffs[1] = 1.0; /* a₁ */
  double val = ekan_fourier_eval(coeffs, EKAN_NET_FOURIER_N, 0.0);
  enx_assert_true(fabs(val - 1.0) < 1e-10);

  /* At x=1: cos(π) = -1.0 */
  val = ekan_fourier_eval(coeffs, EKAN_NET_FOURIER_N, 1.0);
  enx_assert_true(fabs(val - (-1.0)) < 1e-10);
}

enx_test(test_fourier_eval_sin) {
  /* b₁=1.0 → sin(πx), at x=0.5: sin(π/2) = 1.0 */
  double coeffs[EKAN_NET_FOURIER_COEFFS] = {0};
  coeffs[2] = 1.0; /* b₁ */
  double val = ekan_fourier_eval(coeffs, EKAN_NET_FOURIER_N, 0.5);
  enx_assert_true(fabs(val - 1.0) < 1e-10);
}

enx_test(test_fourier_deriv_consistency) {
  /* Numerical derivative should match analytical */
  double coeffs[EKAN_NET_FOURIER_COEFFS];
  uint64_t rng = 42;
  for (int i = 0; i < EKAN_NET_FOURIER_COEFFS; i++)
    coeffs[i] = ekan_rand_normal(&rng) * 0.1;

  double x = 0.3;
  double eps = 1e-6;
  double f_plus = ekan_fourier_eval(coeffs, EKAN_NET_FOURIER_N, x + eps);
  double f_minus = ekan_fourier_eval(coeffs, EKAN_NET_FOURIER_N, x - eps);
  double numerical_deriv = (f_plus - f_minus) / (2.0 * eps);
  double analytical_deriv = ekan_fourier_deriv(coeffs, EKAN_NET_FOURIER_N, x);

  enx_assert_true(fabs(numerical_deriv - analytical_deriv) < 1e-4);
}

enx_test(test_fourier_basis_partition_of_unity) {
  /* Basis values at any x: [1, cos(πx), sin(πx), ...] */
  double basis[EKAN_NET_FOURIER_COEFFS];
  ekan_fourier_basis(EKAN_NET_FOURIER_N, 0.7, basis);
  /* basis[0] should always be 1.0 */
  enx_assert_true(fabs(basis[0] - 1.0) < 1e-10);
  /* sin²+cos² = 1 for each frequency */
  for (int n = 1; n <= EKAN_NET_FOURIER_N; n++) {
    double c = basis[2*n - 1]; /* cos */
    double s = basis[2*n];     /* sin */
    enx_assert_true(fabs(c*c + s*s - 1.0) < 1e-10);
  }
}

/* ── SiLU Tests ── */

enx_test(test_silu_at_zero) {
  /* SiLU(0) = 0 × σ(0) = 0 */
  enx_assert_true(fabs(ekan_silu(0.0)) < 1e-10);
}

enx_test(test_silu_positive_monotone) {
  /* SiLU should be monotonically increasing for x > 0 */
  double prev = ekan_silu(0.0);
  for (double x = 0.1; x < 5.0; x += 0.1) {
    double cur = ekan_silu(x);
    enx_assert_true(cur > prev);
    prev = cur;
  }
}

enx_test(test_silu_deriv_numerical) {
  /* Check SiLU derivative against numerical */
  double x = 1.5;
  double eps = 1e-6;
  double numerical = (ekan_silu(x + eps) - ekan_silu(x - eps)) / (2.0 * eps);
  double analytical = ekan_silu_deriv(x);
  enx_assert_true(fabs(numerical - analytical) < 1e-4);
}

/* ── Network Init Tests ── */

enx_test(test_network_init_topology) {
  EkanNetwork net;
  int topo[] = {12, 8, 4, 8, 12};
  ekan_net_init(&net, topo, 5);

  enx_assert_equal_int(net.num_layers, 4);
  enx_assert_equal_int(net.topology[0], 12);
  enx_assert_equal_int(net.topology[2], 4);
  enx_assert_equal_int(net.topology[4], 12);
  enx_assert_equal_int(net.adam_step, 0);
}

enx_test(test_network_init_gate_alpha) {
  /* Gate alpha should start at 0 (σ(0) = 0.5 = equal KAN/skip) */
  EkanNetwork net;
  int topo[] = {4, 3, 4};
  ekan_net_init(&net, topo, 3);

  enx_assert_true(fabs(net.layers[0].gate_alpha) < 1e-10);
  enx_assert_true(fabs(net.layers[1].gate_alpha) < 1e-10);
}

enx_test(test_network_init_deterministic) {
  /* Same seed → same weights */
  EkanNetwork a, b;
  int topo[] = {4, 3, 4};
  ekan_net_init(&a, topo, 3);
  ekan_net_init(&b, topo, 3);

  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 4; i++)
      enx_assert_true(a.layers[0].mixing[j][i] == b.layers[0].mixing[j][i]);
}

/* ── Forward Pass Tests ── */

enx_test(test_forward_pass_identity_input) {
  EkanNetwork net;
  int topo[] = {4, 3, 4};
  ekan_net_init(&net, topo, 3);

  double input[] = {1.0, 0.0, -1.0, 0.5};
  double output[4];
  ekan_net_predict(&net, input, output);

  /* Output should be finite and bounded */
  for (int i = 0; i < 4; i++) {
    enx_assert_true(isfinite(output[i]));
    enx_assert_true(fabs(output[i]) < 1000.0);
  }
}

enx_test(test_forward_pass_zero_input) {
  EkanNetwork net;
  int topo[] = {4, 4};
  ekan_net_init(&net, topo, 2);

  double input[] = {0.0, 0.0, 0.0, 0.0};
  double output[4];
  ekan_net_predict(&net, input, output);

  /* With zero input, SiLU(0)=0, Fourier(0) = a₀ only.
   * Output should be finite. */
  for (int i = 0; i < 4; i++)
    enx_assert_true(isfinite(output[i]));
}

/* ── Autoencoder Training Tests ── */

enx_test(test_autoencoder_loss_decreases) {
  /* Train on a fixed vector — loss should decrease */
  EkanNetwork net;
  int topo[] = {4, 3, 4};
  ekan_net_init(&net, topo, 3);

  double data[] = {0.5, -0.3, 0.8, 0.1};
  double first_loss = ekan_net_update_online(&net, data, data, 0.01);

  double last_loss = first_loss;
  for (int i = 0; i < 200; i++)
    last_loss = ekan_net_update_online(&net, data, data, 0.01);

  printf("    Loss: %.6f → %.6f (%.1fx reduction)\n", first_loss, last_loss,
         first_loss / (last_loss + 1e-10));
  enx_assert_true(last_loss < first_loss * 0.5);
}

enx_test(test_recon_error_decreases_with_training) {
  EkanNetwork net;
  int topo[] = {4, 3, 4};
  ekan_net_init(&net, topo, 3);

  double data[] = {0.2, 0.4, -0.1, 0.6};
  double err_before = ekan_net_recon_error(&net, data);

  for (int i = 0; i < 500; i++)
    ekan_net_update_online(&net, data, data, 0.005);

  double err_after = ekan_net_recon_error(&net, data);
  printf("    Recon error: %.6f → %.6f\n", err_before, err_after);
  enx_assert_true(err_after < err_before * 0.1);
}

enx_test(test_adam_step_counter) {
  EkanNetwork net;
  int topo[] = {4, 3, 4};
  ekan_net_init(&net, topo, 3);
  enx_assert_equal_int(net.adam_step, 0);

  double data[] = {0.1, 0.2, 0.3, 0.4};
  ekan_net_update_online(&net, data, data, 0.01);
  enx_assert_equal_int(net.adam_step, 1);

  ekan_net_update_online(&net, data, data, 0.01);
  enx_assert_equal_int(net.adam_step, 2);
}

enx_test(test_multi_sample_convergence) {
  /* Train on multiple different vectors */
  EkanNetwork net;
  int topo[] = {4, 3, 4};
  ekan_net_init(&net, topo, 3);

  double samples[][4] = {
    { 0.5, -0.3,  0.8,  0.1},
    {-0.2,  0.6, -0.4,  0.9},
    { 0.1,  0.1,  0.1, -0.5},
  };

  for (int epoch = 0; epoch < 300; epoch++)
    for (int s = 0; s < 3; s++)
      ekan_net_update_online(&net, samples[s], samples[s], 0.005);

  /* Check all three reconstruct reasonably */
  double total_err = 0.0;
  for (int s = 0; s < 3; s++)
    total_err += ekan_net_recon_error(&net, samples[s]);
  printf("    Mean recon error (3 samples): %.6f\n", total_err / 3.0);
  enx_assert_true(total_err / 3.0 < 0.5);
}

/* ── Sensitivity Analysis Tests ── */

enx_test(test_sensitivity_nonnegative) {
  EkanNetwork net;
  int topo[] = {4, 3, 4};
  ekan_net_init(&net, topo, 3);

  double input[] = {0.5, -0.3, 0.8, 0.1};
  double sens[4];
  ekan_net_sensitivity(&net, input, sens, 0.01);

  for (int i = 0; i < 4; i++) {
    enx_assert_true(sens[i] >= 0.0);
    enx_assert_true(isfinite(sens[i]));
  }
}

/* ── Gradient Clipping Tests ── */

enx_test(test_gradient_clipping) {
  enx_assert_true(fabs(ekan_clip_grad(100.0, 1.0) - 1.0) < 1e-10);
  enx_assert_true(fabs(ekan_clip_grad(-100.0, 1.0) - (-1.0)) < 1e-10);
  enx_assert_true(fabs(ekan_clip_grad(0.5, 1.0) - 0.5) < 1e-10);
}

/* ── Full {12,8,4,8,12} Topology Test ── */

enx_test(test_full_fraud_topology) {
  EkanNetwork net;
  int topo[] = {12, 8, 4, 8, 12};
  ekan_net_init(&net, topo, 5);

  double input[12] = {0.1, -0.2, 0.3, 0.0, 0.5, -0.1,
                       0.4, 0.2, -0.3, 0.6, 0.0, -0.4};
  double output[12];
  ekan_net_predict(&net, input, output);

  /* All outputs should be finite */
  for (int i = 0; i < 12; i++)
    enx_assert_true(isfinite(output[i]));

  /* Train and verify loss decreases */
  double first_loss = ekan_net_update_online(&net, input, input, 0.001);
  double loss = first_loss;
  for (int i = 0; i < 500; i++)
    loss = ekan_net_update_online(&net, input, input, 0.001);
  printf("    Full topology loss: %.6f → %.6f\n", first_loss, loss);
  enx_assert_true(loss < first_loss);
}

/* ── Test Runner ── */

static enx_test_case_t test_cases[] = {
    /* Fourier basis */
    enx_test_case(test_fourier_eval_dc),
    enx_test_case(test_fourier_eval_cos),
    enx_test_case(test_fourier_eval_sin),
    enx_test_case(test_fourier_deriv_consistency),
    enx_test_case(test_fourier_basis_partition_of_unity),
    /* SiLU */
    enx_test_case(test_silu_at_zero),
    enx_test_case(test_silu_positive_monotone),
    enx_test_case(test_silu_deriv_numerical),
    /* Network init */
    enx_test_case(test_network_init_topology),
    enx_test_case(test_network_init_gate_alpha),
    enx_test_case(test_network_init_deterministic),
    /* Forward pass */
    enx_test_case(test_forward_pass_identity_input),
    enx_test_case(test_forward_pass_zero_input),
    /* Autoencoder training */
    enx_test_case(test_autoencoder_loss_decreases),
    enx_test_case(test_recon_error_decreases_with_training),
    enx_test_case(test_adam_step_counter),
    enx_test_case(test_multi_sample_convergence),
    /* Sensitivity */
    enx_test_case(test_sensitivity_nonnegative),
    /* Grad clipping */
    enx_test_case(test_gradient_clipping),
    /* Full topology */
    enx_test_case(test_full_fraud_topology),
    enx_test_case_end()
};

const char *suite_name = "EKAN Network (Fourier Autoencoder)";

int main() {
  test_suite suites[] = {{suite_name, test_cases}, {NULL, NULL}};
  if (!test_suite_run(suites)) {
    return 1;
  }
  return 0;
}
