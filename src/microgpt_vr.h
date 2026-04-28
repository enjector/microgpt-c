/*
 * microgpt_vr.h — C99 Vietoris-Rips Persistent Cohomology Engine
 *
 * Ported from EnX-cpp vr_engine.hpp (C++17, template-driven).
 * Fixed at 12 dimensions, 64 max points for fraud detection.
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 *
 * Features:
 *   - L2 distance matrix
 *   - Flag complex filtration with bitmask clique expansion
 *   - F₂ persistent cohomology (apparent pairs + clearing)
 *   - Persistence diagram + Betti numbers (β₀, β₁, β₂)
 *
 * Usage:
 *   VREngine engine;
 *   vr_engine_init(&engine, 10.0f, 2);
 *   VRPoint points[8] = { ... };
 *   VRDiagram diagram = vr_compute(&engine, points, 8, 0.1f);
 *   int betti[3];
 *   vr_betti_at_radius(&diagram, 5.0f, betti);
 */

#ifndef MICROGPT_VR_H
#define MICROGPT_VR_H

#include <stddef.h>
#include <stdint.h>

#define VR_MAX_DIMS      12
#define VR_MAX_PTS       64
#define VR_MAX_EDGES     2048  /* C(64,2) = 2016 */
#define VR_MAX_TRIANGLES 4096
#define VR_MAX_SIMPLICES 8192
#define VR_MAX_INTERVALS 512

/* =========================================================================
 * Data Types
 * ========================================================================= */

typedef struct {
    float coords[VR_MAX_DIMS];
    int   id;
    int   n_dims;  /* Actual dimensionality (≤ VR_MAX_DIMS) */
} VRPoint;

typedef struct {
    int   dimension;   /* 0, 1, or 2 */
    float birth;
    float death;
} VRInterval;

typedef struct {
    VRInterval intervals[VR_MAX_INTERVALS];
    int count;
} VRDiagram;

typedef struct {
    float max_radius;
    float max_radius_sq;
    int   max_dim;       /* 0, 1, or 2 */
    int   n_dims;        /* Point dimensionality */
} VREngine;

/* =========================================================================
 * Public API
 * ========================================================================= */

void vr_engine_init(VREngine *engine, float max_radius, int max_dim, int n_dims);

VRDiagram vr_compute(VREngine *engine, const VRPoint *points, int n_points,
                     float min_persistence);

void vr_betti_numbers(VREngine *engine, const VRPoint *points, int n_points,
                      float at_radius, float min_persistence, int betti_out[3]);

/* Query: Betti number at dimension dim at given filtration radius */
int vr_betti_at(const VRDiagram *diagram, int dim, float filtration);

/* Helper: make a VR point from float array */
VRPoint vr_make_point(const float *coords, int n_dims, int id);

#endif /* MICROGPT_VR_H */
