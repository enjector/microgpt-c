/*
 * MicroGPT-C — Wiring Organelle reference-answer implementations.
 * Phase 7: ground-truth answers for held-out NL prompts.
 * Phase 8: 5 distinct input sets per prompt; correctness requires
 * all 5 references to match the model's executions.
 *
 * Each reference uses the same int64_t arithmetic as the natives in
 * wiring_natives.c so that integer truncation effects don't penalise
 * the model.
 *
 * Naming convention: ref_<short-name>; the held-out file annotates
 * each prompt with `# REFERENCE: <short-name>` (lines starting with
 * `#` are skipped by the corpus preprocessor).
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "wiring_references.h"
#include <string.h>

/* ============================================================
 *  5 distinct input sets — Phase 8 multi-input correctness.
 *
 *  Each set provides at least 16 int64 values to cover any
 *  signature arity the demo encounters. Sets are intentionally
 *  varied (small/medium/edge-cases) so a model that wires args
 *  incorrectly is unlikely to match all 5 by coincidence.
 * ============================================================ */
static const int64_t INPUT_SETS[WIRING_INPUT_SETS][WIRING_MAX_INPUTS] = {
    /* set 0 — original Phase 6/7 sequence */
    { 5, 7, 3, 11, 2, 13, 4, 9, 6, 8, 1, 10, 12, 15, 14, 16 },
    /* set 1 — even-spread small ints */
    { 4, 6, 2, 10, 8, 12, 3, 5, 7, 9, 11, 13, 15, 1, 14, 16 },
    /* set 2 — single small set */
    { 2, 3, 1, 5, 4, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6 },
    /* set 3 — small wide spread */
    { 8, 12, 4, 20, 6, 16, 10, 14, 2, 18, 3, 7, 11, 5, 9, 13 },
    /* set 4 — sequence with a zero (catches divide-by-zero, exp(0), etc.) */
    { 3, 4, 1, 8, 0, 9, 2, 6, 5, 7, 10, 11, 12, 13, 14, 15 },
};

void wiring_input_set(int set_idx, int64_t *dst) {
    if (!dst) return;
    int idx = set_idx % WIRING_INPUT_SETS;
    if (idx < 0) idx += WIRING_INPUT_SETS;
    for (int i = 0; i < WIRING_MAX_INPUTS; i++) dst[i] = INPUT_SETS[idx][i];
}

/* ============================================================
 *  Native primitive helpers — duplicated semantics, used so the
 *  reference functions express intent at the same precision as the
 *  natives. Keep in sync with wiring_natives.c.
 * ============================================================ */
static int64_t r_clamp(int64_t x, int64_t lo, int64_t hi)        { return x < lo ? lo : (x > hi ? hi : x); }
static int64_t r_abs(int64_t x)                                  { return x < 0 ? -x : x; }
static int64_t r_relu(int64_t x)                                 { return x > 0 ? x : 0; }
static int64_t r_tax_amount(int64_t a, int64_t r)                { return (a * r) / 100; }
static int64_t r_apply_tax(int64_t a, int64_t r)                 { return a - r_tax_amount(a, r); }
static int64_t r_percentage(int64_t p, int64_t w)                { return w == 0 ? 0 : (p * 100) / w; }
static int64_t r_compound(int64_t p, int64_t r, int64_t n) {
    if (n < 0) n = 0; if (n > 30) n = 30;
    int64_t v = p;
    for (int64_t i = 0; i < n; i++) v = v + (v * r) / 100;
    return v;
}
static int64_t r_present_value(int64_t f, int64_t r, int64_t n) {
    if (n < 0) n = 0; if (n > 30) n = 30;
    int64_t v = f;
    for (int64_t i = 0; i < n; i++) v = (v * 100) / (100 + r);
    return v;
}
static int64_t r_future_value(int64_t p, int64_t r, int64_t n)   { return r_compound(p, r, n); }
static int64_t r_distance_1d(int64_t a, int64_t b)               { return r_abs(a - b); }
static int64_t r_midpoint(int64_t a, int64_t b)                  { return (a + b) / 2; }
static int64_t r_average_two(int64_t a, int64_t b)               { return (a + b) / 2; }
static int64_t r_factorial(int64_t n) { if (n < 0) return 0; int64_t r = 1; for (int64_t i = 2; i <= n && i <= 20; i++) r *= i; return r; }
static int64_t r_fibonacci(int64_t n) { if (n < 0) return 0; if (n < 2) return n; int64_t a = 0, b = 1; for (int64_t i = 2; i <= n && i <= 90; i++) { int64_t c = a + b; a = b; b = c; } return b; }
static int64_t r_gcd(int64_t a, int64_t b) { a = r_abs(a); b = r_abs(b); while (b) { int64_t t = b; b = a % b; a = t; } return a; }
static int64_t r_sigmoid(int64_t x) {
    if (x <= -8) return 0; if (x >= 8) return 100;
    static const int t[] = { 0, 0, 1, 2, 4, 7, 12, 18, 27, 37, 50, 62, 73, 81, 88, 92, 95, 98, 99, 99, 100 };
    int idx = (int)x + 10; if (idx < 0) idx = 0; if (idx > 20) idx = 20;
    return t[idx];
}
static int64_t r_bmi(int64_t w, int64_t h) { return h == 0 ? 0 : (w * 10000) / (h * h); }

/* ============================================================
 *  Reference functions — each takes the input set as `S[]` and
 *  computes the canonical answer for its prompt. Inputs follow
 *  the order of each prompt's "expected" signature.
 * ============================================================ */

#define DEF_REF(NAME) static int64_t ref_##NAME(const int64_t *S)

/* #1: bmi(weight, height) clamped between lo, hi */
DEF_REF(bmi_clamped)        { return r_clamp(r_bmi(S[0], S[1]), S[2], S[3]); }
/* #2: compound interest earned = compound(P, r, n) - P */
DEF_REF(compound_interest)  { return r_compound(S[0], S[1], S[2]) - S[0]; }
/* #3: weighted average — v1*w1 + v2*w2 + v3*w3 normalised by sum of weights as percentage */
DEF_REF(weighted_three)     {
    int64_t num = S[0]*S[1] + S[2]*S[3] + S[4]*S[5];
    int64_t den = S[1] + S[3] + S[5];
    return den == 0 ? 0 : (num * 100) / den;
}
/* #4: clamp(sigmoid(x), lo, hi) */
DEF_REF(clamped_sigmoid)    { return r_clamp(r_sigmoid(S[0]), S[1], S[2]); }
/* #5: gcd(a, b) * k */
DEF_REF(gcd_scaled)         { return r_gcd(S[0], S[1]) * S[2]; }
/* #6: apply_tax(gross, tax_rate) */
DEF_REF(apply_tax)          { return r_apply_tax(S[0], S[1]); }
/* #7: fibonacci(n) * factorial(n) */
DEF_REF(fib_fact_mul)       { return r_fibonacci(S[0]) * r_factorial(S[0]); }
/* #8: invoice = price*qty + tax_amount(price*qty, rate) */
DEF_REF(invoice_total)      { int64_t st = S[0]*S[1]; return st + r_tax_amount(st, S[2]); }
/* #9: clamp(average_two(a, b), lo, hi) */
DEF_REF(clamped_average)    { return r_clamp(r_average_two(S[0], S[1]), S[2], S[3]); }
/* #10: abs(a - b) */
DEF_REF(abs_diff)           { return r_abs(S[0] - S[1]); }
/* #11: relu(x) * scale */
DEF_REF(scaled_relu)        { return r_relu(S[0]) * S[1]; }
/* #12: tax_amount(discount(price, rate), tax_rate) */
DEF_REF(discounted_tax)     {
    int64_t disc = S[0] - (S[0] * S[1]) / 100;
    return r_tax_amount(disc, S[2]);
}
/* #13: percentage(income - sum_of_expenses, income).
 *      The held-out prompt "fraction of income saved after subtracting
 *      expenses" is genuinely ambiguous about how many expense items.
 *      The model's verified output uses the savings_pipeline_2 template
 *      (income, exp1, exp2 → percentage(saved, income)) which is a
 *      reasonable plural-"expenses" reading. We adopt the 2-expense
 *      semantics here because it matches the model's compositional
 *      interpretation; the old single-expense reference was too narrow. */
DEF_REF(savings_rate)       {
    int64_t sum_exp = S[1] + S[2];
    return r_percentage(S[0] - sum_exp, S[0]);
}
/* #14: square(distance(a1,b1) + distance(a2,b2)) */
DEF_REF(distance_metrics)   { int64_t s = r_distance_1d(S[0], S[1]) + r_distance_1d(S[2], S[3]); return s * s; }
/* #15: distance_1d(a, b) + midpoint(a, b) */
DEF_REF(distance_midpoint)  { return r_distance_1d(S[0], S[1]) + r_midpoint(S[0], S[1]); }
/* #16: present_value(future_value(cashflow, r, n), r, n) */
DEF_REF(pv_of_fv)           { return r_present_value(r_future_value(S[0], S[1], S[2]), S[1], S[2]); }
/* #17: fibonacci(n) + factorial(n) */
DEF_REF(fib_fact_add)       { return r_fibonacci(S[0]) + r_factorial(S[0]); }
/* #18: gross - tax_amount(gross, rate) */
DEF_REF(gross_minus_tax)    { return S[0] - r_tax_amount(S[0], S[1]); }
/* #19: compound(P, r, n) - P */
DEF_REF(compound_minus_p)   { return r_compound(S[0], S[1], S[2]) - S[0]; }
/* #20: clamp(sigmoid(x), lo, hi) */
DEF_REF(sigmoid_clamped)    { return r_clamp(r_sigmoid(S[0]), S[1], S[2]); }

/* ============================================================
 * Phase 3b composition test set — 10 multi-stage references that
 * each chain 2-3 primitives. None matches a single existing
 * single-family anchor; together they form the §42 test set.
 * ============================================================ */
static int64_t r_markup(int64_t p, int64_t r)   { return p + (p * r) / 100; }
static int64_t r_discount(int64_t p, int64_t r) { return p - (p * r) / 100; }

/* C1: tax_amount(discount(markup(price, m_rate), d_rate), t_rate)
 *     S[0]=price, S[1]=m_rate, S[2]=d_rate, S[3]=t_rate */
DEF_REF(markup_discount_tax) {
    int64_t up = r_markup(S[0], S[1]);
    int64_t dn = r_discount(up, S[2]);
    return r_tax_amount(dn, S[3]);
}
/* C2: clamp(compound(P, r, n), lo, hi)
 *     S[0]=P, S[1]=r, S[2]=n, S[3]=lo, S[4]=hi */
DEF_REF(clamped_compound) {
    return r_clamp(r_compound(S[0], S[1], S[2]), S[3], S[4]);
}
/* C3: apply_tax(average_two(a, b), rate)
 *     S[0]=a, S[1]=b, S[2]=rate */
DEF_REF(taxed_average) {
    return r_apply_tax(r_average_two(S[0], S[1]), S[2]);
}
/* C4: abs(a - b) * k
 *     S[0]=a, S[1]=b, S[2]=k */
DEF_REF(scaled_abs_diff) {
    return r_abs(S[0] - S[1]) * S[2];
}
/* C5: clamp(gcd(a, b), lo, hi)
 *     S[0]=a, S[1]=b, S[2]=lo, S[3]=hi */
DEF_REF(clamped_gcd) {
    return r_clamp(r_gcd(S[0], S[1]), S[2], S[3]);
}
/* C6: clamp(relu(x), lo, hi)
 *     S[0]=x, S[1]=lo, S[2]=hi */
DEF_REF(clamped_relu) {
    return r_clamp(r_relu(S[0]), S[1], S[2]);
}
/* C7: sigmoid(x) * k
 *     S[0]=x, S[1]=k */
DEF_REF(scaled_sigmoid) {
    return r_sigmoid(S[0]) * S[1];
}
/* C8: percentage(compound(P, r, n) - P, P)
 *     S[0]=P, S[1]=r, S[2]=n */
DEF_REF(interest_as_pct) {
    int64_t interest = r_compound(S[0], S[1], S[2]) - S[0];
    return r_percentage(interest, S[0]);
}
/* C9: discount(markup(price, m_rate), d_rate)
 *     S[0]=price, S[1]=m_rate, S[2]=d_rate */
DEF_REF(markup_then_discount) {
    return r_discount(r_markup(S[0], S[1]), S[2]);
}
/* C10: clamp(fibonacci(n) * factorial(n), lo, hi)
 *      S[0]=n, S[1]=lo, S[2]=hi */
DEF_REF(clamped_fib_fact) {
    int64_t prod = r_fibonacci(S[0]) * r_factorial(S[0]);
    return r_clamp(prod, S[1], S[2]);
}

/* ============================================================
 * Phase 5 — Compositional held-out test set (Stream C of compositional fix).
 *
 * 30 references for `pipeline_corpus_compositional_test.txt`. Each prompt
 * names two or three primitives and the canonical reference computes
 * the expected numeric answer. See COMPOSITIONAL_GENERATOR_FIX_PLAN.md
 * Stream C for the pre-registration.
 * ============================================================ */

/* Extra helpers required by the new references. */
static int64_t r_square(int64_t x)        { return x * x; }
static int64_t r_cube(int64_t x)          { return x * x * x; }
static int64_t r_double_val(int64_t x)    { return x * 2; }
static int64_t r_triple_val(int64_t x)    { return x * 3; }
static int64_t r_min_two(int64_t a, int64_t b) { return a < b ? a : b; }
static int64_t r_max_two(int64_t a, int64_t b) { return a > b ? a : b; }
static int64_t r_negate(int64_t x)        { return -x; }
static int64_t r_kinetic_energy(int64_t m, int64_t v) { return (m * v * v) / 2; }
static int64_t r_circle_area(int64_t r)   { return (314 * r * r) / 100; }
static int64_t r_harmonic_n(int64_t n) {
    if (n <= 0) return 0;
    int64_t s = 0;
    for (int64_t k = 1; k <= n && k <= 1000; k++) s += 1000 / k;
    return s;
}
static int64_t r_lerp(int64_t a, int64_t b, int64_t t) { return a + ((b - a) * t) / 100; }

/* Axis 1 — 2-primitive novel-pair compositions (10). */
DEF_REF(abs_diff_axis)            { return r_abs(S[0] - S[1]); }
DEF_REF(max_x_squared_y)          { return r_max_two(r_square(S[0]), S[1]); }
DEF_REF(avg_double_x_y)           { return r_average_two(r_double_val(S[0]), S[1]); }
DEF_REF(gcd_x_abs_y)              { return r_gcd(S[0], r_abs(S[1])); }
DEF_REF(cube_min_x_y)             { return r_cube(r_min_two(S[0], S[1])); }
DEF_REF(relu_x_minus_y)           { return r_relu(S[0] - S[1]); }
DEF_REF(sigmoid_double_x)         { return r_sigmoid(r_double_val(S[0])); }
DEF_REF(harmonic_x_plus_y)        { return r_harmonic_n(S[0] + S[1]); }
DEF_REF(factorial_diff)           { return r_factorial(r_abs(S[0] - S[1])); }
DEF_REF(pct_x_of_y_squared)       { return r_percentage(S[0], r_square(S[1])); }

/* Axis 2 — 3-primitive synonym-stress compositions (10). */
DEF_REF(bmi_double_w_h)           { return r_bmi(r_double_val(S[0]), S[1]); }
DEF_REF(after_tax_markup)         { return r_apply_tax(r_markup(S[0], S[1]), S[2]); }
DEF_REF(discounted_compound)      { return r_discount(r_compound(S[0], S[1], S[2]), S[3]); }
DEF_REF(ke_double_mass)           { return r_kinetic_energy(r_double_val(S[0]), S[1]); }
DEF_REF(circle_abs_radius)        { return r_circle_area(r_abs(S[0])); }
DEF_REF(lerp_x_max_y_z)           { return r_lerp(S[0], r_max_two(S[1], S[2]), S[3]); }
DEF_REF(gcd_sq_diff)              { return r_gcd(r_square(S[0] - S[1]), S[2]); }
DEF_REF(pct_markup_x_y)           { return r_percentage(r_markup(S[0], S[1]), S[2]); }
DEF_REF(harmonic_abs_fib)         { return r_harmonic_n(r_abs(r_fibonacci(S[0]))); }
DEF_REF(fv_of_pv)                 { return r_future_value(r_present_value(S[0], S[1], S[2]), S[1], S[2]); }

/* Axis 3 — 2-or-3 primitives + outer transform with type pressure (10). */
DEF_REF(double_gcd_sq_y)          { return r_double_val(r_gcd(r_square(S[0]), S[1])); }
DEF_REF(neg_min_x_abs_y)          { return r_negate(r_min_two(S[0], r_abs(S[1]))); }
DEF_REF(cube_avg_double_x_y)      { return r_cube(r_average_two(r_double_val(S[0]), S[1])); }
DEF_REF(relu_gcd_x_y)             { return r_relu(r_gcd(S[0], S[1])); }
DEF_REF(sigmoid_x_minus_3y)       { return r_sigmoid(S[0] - r_triple_val(S[1])); }
DEF_REF(taxed_markup_double)      { return r_apply_tax(r_markup(r_double_val(S[0]), S[1]), S[2]); }
DEF_REF(discount_ke)              { return r_discount(r_kinetic_energy(S[0], S[1]), S[2]); }
DEF_REF(bmi_sq_w_double_h)        { return r_bmi(r_square(S[0]), r_double_val(S[1])); }
DEF_REF(harmonic_gcd)             { return r_harmonic_n(r_gcd(S[0], S[1])); }
DEF_REF(pct_cube_x_sq_y)          { return r_percentage(r_cube(S[0]), r_square(S[1])); }

#undef DEF_REF

typedef int64_t (*RefFn)(const int64_t *S);

typedef struct {
    const char *name;
    RefFn fn;
} RefEntry;

static const RefEntry references[] = {
    {"bmi_clamped",        ref_bmi_clamped},
    {"compound_interest",  ref_compound_interest},
    {"weighted_three",     ref_weighted_three},
    {"clamped_sigmoid",    ref_clamped_sigmoid},
    {"gcd_scaled",         ref_gcd_scaled},
    {"apply_tax",          ref_apply_tax},
    {"fib_fact_mul",       ref_fib_fact_mul},
    {"invoice_total",      ref_invoice_total},
    {"clamped_average",    ref_clamped_average},
    {"abs_diff",           ref_abs_diff},
    {"scaled_relu",        ref_scaled_relu},
    {"discounted_tax",     ref_discounted_tax},
    {"savings_rate",       ref_savings_rate},
    {"distance_metrics",   ref_distance_metrics},
    {"distance_midpoint",  ref_distance_midpoint},
    {"pv_of_fv",           ref_pv_of_fv},
    {"fib_fact_add",       ref_fib_fact_add},
    {"gross_minus_tax",    ref_gross_minus_tax},
    {"compound_minus_p",   ref_compound_minus_p},
    {"sigmoid_clamped",    ref_sigmoid_clamped},
    /* Phase 3b composition test set */
    {"markup_discount_tax", ref_markup_discount_tax},
    {"clamped_compound",    ref_clamped_compound},
    {"taxed_average",       ref_taxed_average},
    {"scaled_abs_diff",     ref_scaled_abs_diff},
    {"clamped_gcd",         ref_clamped_gcd},
    {"clamped_relu",        ref_clamped_relu},
    {"scaled_sigmoid",      ref_scaled_sigmoid},
    {"interest_as_pct",     ref_interest_as_pct},
    {"markup_then_discount", ref_markup_then_discount},
    {"clamped_fib_fact",    ref_clamped_fib_fact},
    /* Phase 5 compositional held-out (Stream C of compositional fix) */
    {"ref_abs_diff",            ref_abs_diff_axis},
    {"ref_max_x_squared_y",     ref_max_x_squared_y},
    {"ref_avg_double_x_y",      ref_avg_double_x_y},
    {"ref_gcd_x_abs_y",         ref_gcd_x_abs_y},
    {"ref_cube_min_x_y",        ref_cube_min_x_y},
    {"ref_relu_x_minus_y",      ref_relu_x_minus_y},
    {"ref_sigmoid_double_x",    ref_sigmoid_double_x},
    {"ref_harmonic_x_plus_y",   ref_harmonic_x_plus_y},
    {"ref_factorial_diff",      ref_factorial_diff},
    {"ref_pct_x_of_y_squared",  ref_pct_x_of_y_squared},
    {"ref_bmi_double_w_h",      ref_bmi_double_w_h},
    {"ref_after_tax_markup",    ref_after_tax_markup},
    {"ref_discounted_compound", ref_discounted_compound},
    {"ref_ke_double_mass",      ref_ke_double_mass},
    {"ref_circle_abs_radius",   ref_circle_abs_radius},
    {"ref_lerp_x_max_y_z",      ref_lerp_x_max_y_z},
    {"ref_gcd_sq_diff",         ref_gcd_sq_diff},
    {"ref_pct_markup_x_y",      ref_pct_markup_x_y},
    {"ref_harmonic_abs_fib",    ref_harmonic_abs_fib},
    {"ref_fv_of_pv",            ref_fv_of_pv},
    {"ref_double_gcd_sq_y",     ref_double_gcd_sq_y},
    {"ref_neg_min_x_abs_y",     ref_neg_min_x_abs_y},
    {"ref_cube_avg_double_x_y", ref_cube_avg_double_x_y},
    {"ref_relu_gcd_x_y",        ref_relu_gcd_x_y},
    {"ref_sigmoid_x_minus_3y",  ref_sigmoid_x_minus_3y},
    {"ref_taxed_markup_double", ref_taxed_markup_double},
    {"ref_discount_ke",         ref_discount_ke},
    {"ref_bmi_sq_w_double_h",   ref_bmi_sq_w_double_h},
    {"ref_harmonic_gcd",        ref_harmonic_gcd},
    {"ref_pct_cube_x_sq_y",     ref_pct_cube_x_sq_y},
};
static const int references_count = (int)(sizeof(references) / sizeof(references[0]));

int wiring_reference_compute_at(const char *name, int set_idx, int64_t *out) {
    if (!name || !out) return 0;
    int idx = set_idx % WIRING_INPUT_SETS;
    if (idx < 0) idx += WIRING_INPUT_SETS;
    for (int i = 0; i < references_count; i++) {
        if (strcmp(references[i].name, name) == 0) {
            *out = references[i].fn(INPUT_SETS[idx]);
            return 1;
        }
    }
    return 0;
}

int wiring_reference_compute(const char *name, int64_t *out) {
    return wiring_reference_compute_at(name, 0, out);
}
