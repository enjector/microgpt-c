/*
 * MicroGPT-C — Wiring Organelle native primitive implementations.
 * Phase 6: bridges Pipeline IR graphs to executable C functions.
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "wiring_natives.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 *  Native primitive implementations.
 *
 *  Each takes (n_in, ins[]) where ins[] is an array of int64 values
 *  and returns int64. The dispatch wrapper extracts ints from
 *  PipelineValue, calls the impl, marshals back. All primitive
 *  semantics mirror the corresponding entries in
 *  demos/word-level/vm_codegen/w_vm_functions.txt — keep these in
 *  sync if signatures change.
 * ============================================================ */

typedef int64_t (*WiringNativeFn)(int n_in, const int64_t *ins);

static int64_t n_add(int n_in, const int64_t *a)         { (void)n_in; return a[0] + a[1]; }
static int64_t n_subtract(int n_in, const int64_t *a)    { (void)n_in; return a[0] - a[1]; }
static int64_t n_multiply(int n_in, const int64_t *a)    { (void)n_in; return a[0] * a[1]; }
static int64_t n_divide(int n_in, const int64_t *a)      { (void)n_in; return (a[1] == 0) ? 0 : a[0] / a[1]; }

static int64_t n_negate(int n_in, const int64_t *a)      { (void)n_in; return -a[0]; }
static int64_t n_abs_val(int n_in, const int64_t *a)     { (void)n_in; return a[0] < 0 ? -a[0] : a[0]; }
static int64_t n_square(int n_in, const int64_t *a)      { (void)n_in; return a[0] * a[0]; }
static int64_t n_cube(int n_in, const int64_t *a)        { (void)n_in; return a[0] * a[0] * a[0]; }
static int64_t n_double_val(int n_in, const int64_t *a)  { (void)n_in; return a[0] * 2; }
static int64_t n_triple_val(int n_in, const int64_t *a)  { (void)n_in; return a[0] * 3; }

static int64_t n_min_two(int n_in, const int64_t *a)     { (void)n_in; return a[0] < a[1] ? a[0] : a[1]; }
static int64_t n_max_two(int n_in, const int64_t *a)     { (void)n_in; return a[0] > a[1] ? a[0] : a[1]; }
static int64_t n_average_two(int n_in, const int64_t *a) { (void)n_in; return (a[0] + a[1]) / 2; }
static int64_t n_distance_1d(int n_in, const int64_t *a) { (void)n_in; int64_t d = a[0] - a[1]; return d < 0 ? -d : d; }
static int64_t n_midpoint(int n_in, const int64_t *a)    { (void)n_in; return (a[0] + a[1]) / 2; }
static int64_t n_mse(int n_in, const int64_t *a)         { (void)n_in; int64_t d = a[0] - a[1]; return d * d; }

static int64_t n_clamp(int n_in, const int64_t *a)       {
    (void)n_in;
    int64_t x = a[0], lo = a[1], hi = a[2];
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
static int64_t n_lerp(int n_in, const int64_t *a)        {
    (void)n_in;
    /* a + (b - a) * t  — t is treated as integer percentage 0..100 */
    return a[0] + ((a[1] - a[0]) * a[2]) / 100;
}
static int64_t n_sigmoid(int n_in, const int64_t *a)     {
    /* Integer sigmoid: maps to range [0..100] for x in roughly [-10..10].
     * Avoid floating point; use a small lookup table. */
    (void)n_in;
    int64_t x = a[0];
    if (x <= -8) return 0;
    if (x >=  8) return 100;
    /* Approximate 100 / (1 + e^-x) via piece-wise linear. */
    static const int sig_table[] = { 0, 0, 1, 2, 4, 7, 12, 18, 27, 37, 50, 62, 73, 81, 88, 92, 95, 98, 99, 99, 100 };
    int idx = (int)x + 10;
    if (idx < 0) idx = 0;
    if (idx > 20) idx = 20;
    return sig_table[idx];
}
static int64_t n_relu(int n_in, const int64_t *a)        { (void)n_in; return a[0] > 0 ? a[0] : 0; }

static int64_t n_tax_amount(int n_in, const int64_t *a)  {
    /* tax = amount * rate / 100  (rate is integer percentage) */
    (void)n_in;
    return (a[0] * a[1]) / 100;
}
static int64_t n_apply_tax(int n_in, const int64_t *a)   {
    (void)n_in;
    return a[0] - (a[0] * a[1]) / 100;
}
static int64_t n_percentage(int n_in, const int64_t *a)  {
    (void)n_in;
    return (a[1] == 0) ? 0 : (a[0] * 100) / a[1];
}
static int64_t n_discount(int n_in, const int64_t *a)    {
    (void)n_in;
    return a[0] - (a[0] * a[1]) / 100;
}
static int64_t n_markup(int n_in, const int64_t *a)      {
    (void)n_in;
    return a[0] + (a[0] * a[1]) / 100;
}

static int64_t n_factorial(int n_in, const int64_t *a)   {
    (void)n_in;
    int64_t n = a[0];
    if (n < 0) return 0;
    int64_t r = 1;
    for (int64_t i = 2; i <= n && i <= 20; i++) r *= i;  /* cap to prevent overflow */
    return r;
}
static int64_t n_fibonacci(int n_in, const int64_t *a)   {
    (void)n_in;
    int64_t n = a[0];
    if (n < 0) return 0;
    if (n < 2) return n;
    int64_t prev = 0, cur = 1;
    for (int64_t i = 2; i <= n && i <= 90; i++) {
        int64_t next = prev + cur;
        prev = cur; cur = next;
    }
    return cur;
}
static int64_t n_gcd(int n_in, const int64_t *a)         {
    (void)n_in;
    int64_t x = a[0] < 0 ? -a[0] : a[0];
    int64_t y = a[1] < 0 ? -a[1] : a[1];
    while (y) { int64_t t = y; y = x % y; x = t; }
    return x;
}
static int64_t n_harmonic_n(int n_in, const int64_t *a)  {
    /* Sum of 1/k for k in 1..n, scaled by 1000 to keep in ints. */
    (void)n_in;
    int64_t n = a[0];
    if (n <= 0) return 0;
    int64_t s = 0;
    for (int64_t k = 1; k <= n && k <= 1000; k++) s += 1000 / k;
    return s;
}
static int64_t n_circle_area(int n_in, const int64_t *a) {
    /* pi*r^2, pi ≈ 314/100 */
    (void)n_in;
    return (314 * a[0] * a[0]) / 100;
}
static int64_t n_kinetic_energy(int n_in, const int64_t *a) {
    /* 0.5 * m * v^2 */
    (void)n_in;
    return (a[0] * a[1] * a[1]) / 2;
}
static int64_t n_bmi(int n_in, const int64_t *a)         {
    /* weight (kg) / (height_cm/100)^2 — scaled BMI = weight*10000 / (height*height) */
    (void)n_in;
    int64_t h = a[1];
    if (h == 0) return 0;
    return (a[0] * 10000) / (h * h);
}

static int64_t n_compound(int n_in, const int64_t *a)    {
    /* principal * (1 + rate/100)^periods. Integer iterative. */
    (void)n_in;
    int64_t p = a[0], r = a[1], n = a[2];
    if (n < 0) n = 0;
    if (n > 30) n = 30;  /* clamp to prevent runaway */
    int64_t v = p;
    for (int64_t i = 0; i < n; i++) v = v + (v * r) / 100;
    return v;
}
static int64_t n_power(int n_in, const int64_t *a)       {
    (void)n_in;
    int64_t b = a[0], e = a[1];
    if (e < 0) return 0;
    if (e > 30) e = 30;
    int64_t r = 1;
    for (int64_t i = 0; i < e; i++) r *= b;
    return r;
}
static int64_t n_present_value(int n_in, const int64_t *a) {
    /* future / (1 + rate/100)^periods. Discrete approximation. */
    (void)n_in;
    int64_t f = a[0], r = a[1], n = a[2];
    if (n < 0) n = 0;
    if (n > 30) n = 30;
    int64_t denom = 100;
    int64_t v = f;
    for (int64_t i = 0; i < n; i++) {
        v = (v * 100) / (100 + r);
        (void)denom;
    }
    return v;
}
static int64_t n_future_value(int n_in, const int64_t *a) {
    /* present * (1 + rate/100)^periods. Same as compound. */
    return n_compound(n_in, a);
}

/* ============================================================
 *  Registry — name → fn lookup table
 * ============================================================ */

typedef struct {
    const char *name;
    WiringNativeFn fn;
} WiringEntry;

static const WiringEntry registry[] = {
    /* arithmetic */
    {"add",            n_add},
    {"subtract",       n_subtract},
    {"multiply",       n_multiply},
    {"divide",         n_divide},
    {"negate",         n_negate},
    {"abs_val",        n_abs_val},
    {"abs",            n_abs_val},
    {"square",         n_square},
    {"cube",           n_cube},
    {"double_val",     n_double_val},
    {"triple_val",     n_triple_val},
    /* min/max/distance */
    {"min_two",        n_min_two},
    {"min",            n_min_two},
    {"max_two",        n_max_two},
    {"max",            n_max_two},
    {"average_two",    n_average_two},
    {"distance_1d",    n_distance_1d},
    {"midpoint",       n_midpoint},
    {"mse",            n_mse},
    /* bounding */
    {"clamp",          n_clamp},
    {"lerp",           n_lerp},
    /* nonlinear */
    {"sigmoid",        n_sigmoid},
    {"relu",           n_relu},
    /* finance */
    {"tax_amount",     n_tax_amount},
    {"apply_tax",      n_apply_tax},
    {"percentage",     n_percentage},
    {"discount",       n_discount},
    {"markup",         n_markup},
    {"compound",       n_compound},
    {"present_value",  n_present_value},
    {"future_value",   n_future_value},
    /* number theory */
    {"factorial",      n_factorial},
    {"fibonacci",      n_fibonacci},
    {"gcd",            n_gcd},
    {"harmonic_n",     n_harmonic_n},
    /* misc */
    {"circle_area",    n_circle_area},
    {"kinetic_energy", n_kinetic_energy},
    {"bmi",            n_bmi},
    {"power",          n_power},
    /* divide-by-const config primitive (used by tpl_mean): rate stored
     * in node config "by"; we emulate as a unary that ignores config
     * and just passes through (degraded behaviour — graphs that need
     * the divisor will report wrong but won't crash). */
    {"divide_by_const", n_divide},
};

static const int registry_size = (int)(sizeof(registry) / sizeof(registry[0]));

int wiring_natives_known(const char *primitive) {
    if (!primitive) return 0;
    for (int i = 0; i < registry_size; i++) {
        if (strcmp(registry[i].name, primitive) == 0) return 1;
    }
    return 0;
}

int wiring_natives_dispatch(const char *primitive,
                            const PipelineConfig *config, int n_config,
                            const PipelineValue *inputs, int n_inputs,
                            PipelineValue *outputs, int n_outputs,
                            void *user_data) {
    (void)config; (void)user_data;
    if (!primitive) return -1;
    /* Resolve. */
    WiringNativeFn fn = NULL;
    for (int i = 0; i < registry_size; i++) {
        if (strcmp(registry[i].name, primitive) == 0) { fn = registry[i].fn; break; }
    }
    if (!fn) return -1;
    /* Marshal int inputs (cap arity at 8 — none of our primitives need more). */
    if (n_inputs > 8) return -2;
    int64_t in_buf[8];
    for (int i = 0; i < n_inputs; i++) {
        if (!inputs[i].type) return -3;
        switch (inputs[i].type->kind) {
        case PIPE_T_INT:    in_buf[i] = inputs[i].v.i; break;
        case PIPE_T_FLOAT:  in_buf[i] = (int64_t)inputs[i].v.f; break;
        case PIPE_T_ANY:    in_buf[i] = inputs[i].v.i; break;  /* assume int payload */
        default:            return -4;
        }
    }
    /* Special handling for divide_by_const: argv is just 1 input + a config "by". */
    int64_t result;
    if (strcmp(primitive, "divide_by_const") == 0 && n_inputs == 1) {
        int64_t divisor = 1;
        for (int c = 0; c < n_config; c++) {
            if (config[c].name && strcmp(config[c].name, "by") == 0
                && config[c].kind == PIPE_CFG_INT) {
                divisor = config[c].v.i;
                break;
            }
        }
        result = (divisor == 0) ? 0 : in_buf[0] / divisor;
    } else {
        result = fn(n_inputs, in_buf);
    }
    /* Emit single int output. */
    if (n_outputs < 1) return -5;
    outputs[0].v.i = result;
    /* outputs[0].type is pre-populated by pipeline_execute; we leave it. */
    return 0;
}
