#ifndef MICROGPT_EKAN_H
#define MICROGPT_EKAN_H

#include <stdbool.h>
#include <stdint.h>

/* Portable force-inline: MSVC uses __forceinline, GCC/Clang use __attribute__ */
#ifdef _MSC_VER
  #define EKAN_FORCE_INLINE static __forceinline
#else
  #define EKAN_FORCE_INLINE static inline __attribute__((always_inline))
#endif

#define BONSAI_FP_SCALE 1000000LL // Enforce 64-bit scaling
#define EKAN_DEGREE 3             // Standard Cubic B-Spline

// Define maximum geometry to ensure $O(1)$ memory arena and zero-allocation.
#define MAX_EKAN_EDGES 128
#define MAX_SPLINE_GRID_SIZE 64

// Safe Fixed-Point Multiplication
EKAN_FORCE_INLINE int32_t fp_mul(int32_t a,
                                                            int32_t b) {
  return (int32_t)(((int64_t)a * (int64_t)b) / BONSAI_FP_SCALE);
}

// Safe Fixed-Point Division (with zero-division protection)
EKAN_FORCE_INLINE int32_t fp_div(int32_t num,
                                                            int32_t denom) {
  if (denom == 0)
    return 0; // Spline boundary singularity protection
  return (int32_t)((((int64_t)num * BONSAI_FP_SCALE) / denom));
}

// C99 EKAN Memory Arena ("Memory T-Cell")
typedef struct {
  int32_t control_points[MAX_EKAN_EDGES][MAX_SPLINE_GRID_SIZE];
  int32_t base_weights[MAX_EKAN_EDGES];
  int32_t knots[MAX_SPLINE_GRID_SIZE + EKAN_DEGREE + 1];
  int num_edges;
  int grid_size;  // Number of grid intervals
  int num_points; // Number of control points (grid_size + EKAN_DEGREE)
} EKAN_Organelle;

/**
 * Rapidly locates the knot span 'i' such that knots[i] <= x < knots[i+1].
 * Uses Binary Search to guarantee sub-nanosecond resolution.
 *
 * @param x          The incoming fixed-point market feature
 * @param num_points The total number of control points in this EKAN edge
 * @param knots      The fixed-point knot vector array
 * @return           The integer index of the active knot span
 */
EKAN_FORCE_INLINE int
ekan_find_knot_span_fp(int32_t x, int num_points, const int32_t *knots) {
  int n = num_points - 1;
  if (x >= knots[n + 1])
    return n;
  if (x <= knots[EKAN_DEGREE])
    return EKAN_DEGREE;

  int low = EKAN_DEGREE;
  int high = n + 1;
  int mid = (low + high) / 2;

  while (x < knots[mid] || x >= knots[mid + 1]) {
    if (x < knots[mid])
      high = mid;
    else
      low = mid;
    mid = (low + high) / 2;
  }
  return mid;
}

/**
 * Evaluates the 4 non-zero basis functions for a Cubic B-Spline.
 * @param x      The incoming market feature (scaled by BONSAI_FP_SCALE)
 * @param span   The identified knot span index where knots[span] <= x <
 * knots[span+1]
 * @param knots  The fixed-point knot vector array
 * @param N      Output array of size [EKAN_DEGREE + 1] holding the basis
 * weights
 */
EKAN_FORCE_INLINE void
ekan_bspline_basis_fp(int32_t x, int span, const int32_t *knots, int32_t *N) {
  int32_t left[EKAN_DEGREE + 1];
  int32_t right[EKAN_DEGREE + 1];
  N[0] = (int32_t)BONSAI_FP_SCALE;

  for (int j = 1; j <= EKAN_DEGREE; j++) {
    left[j] = x - knots[span + 1 - j];
    right[j] = knots[span + j] - x;
    int32_t saved = 0;

    for (int r = 0; r < j; r++) {
      int32_t denom = right[r + 1] + left[j - r];
      int32_t temp = fp_div(N[r], denom);
      N[r] = saved + fp_mul(right[r + 1], temp);
      saved = fp_mul(left[j - r], temp);
    }
    N[j] = saved;
  }
}

/**
 * Evaluates a single EKAN 1D activation (Edge)
 * @param x               The incoming market feature
 * @param span            The active knot span
 * @param knots           The fixed-point knot vector array
 * @param control_points  The fixed-point control points array for this edge
 * @param base_weight     The linear bypass base weight for this edge
 * @return                The fixed-point activation output
 */
EKAN_FORCE_INLINE int32_t
ekan_edge_pulse(int32_t x, int span, const int32_t *knots,
                const int32_t *control_points, int32_t base_weight) {
  int32_t N[EKAN_DEGREE + 1];
  ekan_bspline_basis_fp(x, span, knots, N);

  int32_t base_activation = fp_mul(x, base_weight);
  int32_t spline_activation = 0;

  for (int i = 0; i <= EKAN_DEGREE; i++) {
    int cp_index = span - EKAN_DEGREE + i;
    spline_activation += fp_mul(control_points[cp_index], N[i]);
  }

  return base_activation + spline_activation;
}

#endif // MICROGPT_EKAN_H
