#include "microgpt_ekan.h"
#include "test.h"

enx_test(test_find_knot_span_clamping) {
  int32_t knots[] = {0,       0,       0,       0,       1000000,
                     2000000, 3000000, 3000000, 3000000, 3000000};
  int num_points = 6; // Grid=2, Degree=3 -> 6 control points
  // Max knot is knots[num_points] = knots[6] = 3000000
  // Min valid knot is knots[3] = 0

  // Test lower boundary clamp
  enx_assert_equal_int(ekan_find_knot_span_fp(-500000, num_points, knots), 3);
  enx_assert_equal_int(ekan_find_knot_span_fp(0, num_points, knots), 3);

  // Test upper boundary clamp
  enx_assert_equal_int(ekan_find_knot_span_fp(3000000, num_points, knots), 5);
  enx_assert_equal_int(ekan_find_knot_span_fp(4000000, num_points, knots), 5);

  // Test middle
  enx_assert_equal_int(ekan_find_knot_span_fp(1500000, num_points, knots), 4);
}

enx_test(test_bspline_basis_evaluation) {
  int32_t knots[] = {0,       0,       0,       0,       1000000,
                     2000000, 3000000, 3000000, 3000000, 3000000};
  int32_t N[4];

  // Test directly on a knot
  ekan_bspline_basis_fp(1000000, 4, knots, N);

  // The sum of basis functions should always equal 1.0 (BONSAI_FP_SCALE)
  int32_t sum = N[0] + N[1] + N[2] + N[3];
  // Allow a small rounding error due to fixed point math, e.g. within 10 units
  enx_assert_true(sum > BONSAI_FP_SCALE - 10 && sum < BONSAI_FP_SCALE + 10);

  // Verify basis functions are positive
  enx_assert_true(N[0] >= 0 && N[1] >= 0 && N[2] >= 0 && N[3] >= 0);
}

enx_test(test_ekan_edge_pulse) {
  int32_t knots[] = {0,       0,       0,       0,       1000000,
                     2000000, 3000000, 3000000, 3000000, 3000000};
  int32_t control_points[] = {500000,  -200000, 800000,
                              1500000, -100000, 300000};
  int32_t base_weight = 2000000; // 2.0

  int32_t x = 1500000; // 1.5
  int span = ekan_find_knot_span_fp(x, 6, knots);

  int32_t pulse = ekan_edge_pulse(x, span, knots, control_points, base_weight);

  // Base activation should be 1.5 * 2.0 = 3.0 (3000000)
  // Spline activation will be some bounded value dependent on control points.
  // We mainly want to ensure it calculates without overflow/crashing.
  enx_assert_true(pulse != 0); // highly unlikely to be exactly 0
}

static enx_test_case_t test_cases[] = {
    enx_test_case(test_find_knot_span_clamping),
    enx_test_case(test_bspline_basis_evaluation),
    enx_test_case(test_ekan_edge_pulse), enx_test_case_end()};

const char *suite_name = "EKAN C99 Memory T-Cell";

int main() {
  test_suite suites[] = {{suite_name, test_cases}, {NULL, NULL}};
  if (!test_suite_run(suites)) {
    return 1;
  }
  return 0;
}
