/*
 * MicroGPT-C — Wiring Organelle reference-answer implementations.
 * Phase 7: ground-truth answers for held-out NL prompts.
 *
 * Each reference uses the same int64_t arithmetic as the natives in
 * wiring_natives.c so that integer truncation effects don't penalise
 * the model. The test inputs are a fixed sequence (5, 7, 3, 11, 2, ...)
 * — adjusting them changes both the model's executions and the
 * reference answers in lock-step.
 *
 * Naming convention: ref_<short-name>; the held-out file annotates
 * each prompt with `# REFERENCE: <short-name>` (lines starting with
 * `#` are skipped by the corpus preprocessor — they are pure metadata).
 *
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "wiring_references.h"
#include <string.h>

/* Test input sequence. Must match demo's test_seq[] in main.c. */
static const int64_t S[] = {5, 7, 3, 11, 2, 13, 4, 9, 6, 8, 1, 10, 12, 15, 14, 16};

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
static int64_t r_min(int64_t a, int64_t b)                       { return a < b ? a : b; }
static int64_t r_max(int64_t a, int64_t b)                       { return a > b ? a : b; }
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
 *  Reference functions, one per held-out prompt.
 *  Inputs follow the order of each prompt's "expected" signature.
 * ============================================================ */

/* #1: bmi(weight, height) clamped between lo, hi */
static int64_t ref_bmi_clamped(void)        { return r_clamp(r_bmi(S[0], S[1]), S[2], S[3]); }
/* #2: compound interest earned = compound(P, r, n) - P */
static int64_t ref_compound_interest(void)  { return r_compound(S[0], S[1], S[2]) - S[0]; }
/* #3: weighted average — assume v1*w1 + v2*w2 + v3*w3, then divide by sum of weights via percentage(s,w) */
static int64_t ref_weighted_three(void)     {
    /* inputs: v1=5, w1=7, v2=3, w2=11, v3=2, w3=13 */
    int64_t num = S[0]*S[1] + S[2]*S[3] + S[4]*S[5];
    int64_t den = S[1] + S[3] + S[5];
    return den == 0 ? 0 : (num * 100) / den;  /* percentage form */
}
/* #4: clamp(sigmoid(x), lo, hi) */
static int64_t ref_clamped_sigmoid(void)    { return r_clamp(r_sigmoid(S[0]), S[1], S[2]); }
/* #5: gcd(a, b) * k */
static int64_t ref_gcd_scaled(void)         { return r_gcd(S[0], S[1]) * S[2]; }
/* #6: apply_tax(gross, tax_rate) */
static int64_t ref_apply_tax(void)          { return r_apply_tax(S[0], S[1]); }
/* #7: fibonacci(n) * factorial(n) */
static int64_t ref_fib_fact_mul(void)       { return r_fibonacci(S[0]) * r_factorial(S[0]); }
/* #8: invoice = price*qty + tax_amount(price*qty, rate) */
static int64_t ref_invoice_total(void)      { int64_t st = S[0]*S[1]; return st + r_tax_amount(st, S[2]); }
/* #9: clamp(average_two(a, b), lo, hi) */
static int64_t ref_clamped_average(void)    { return r_clamp(r_average_two(S[0], S[1]), S[2], S[3]); }
/* #10: abs(a - b) */
static int64_t ref_abs_diff(void)           { return r_abs(S[0] - S[1]); }
/* #11: relu(x) * scale */
static int64_t ref_scaled_relu(void)        { return r_relu(S[0]) * S[1]; }
/* #12: tax_amount(discount(price, rate), tax_rate) */
static int64_t ref_discounted_tax(void)     {
    int64_t disc = S[0] - (S[0] * S[1]) / 100;
    return r_tax_amount(disc, S[2]);
}
/* #13: percentage(income - expenses, income) — savings rate */
static int64_t ref_savings_rate(void)       { return r_percentage(S[0] - S[1], S[0]); }
/* #14: square(distance(a1,b1) + distance(a2,b2)) */
static int64_t ref_distance_metrics(void)   { int64_t s = r_distance_1d(S[0], S[1]) + r_distance_1d(S[2], S[3]); return s * s; }
/* #15: distance_1d(a, b) + midpoint(a, b) */
static int64_t ref_distance_midpoint(void)  { return r_distance_1d(S[0], S[1]) + r_midpoint(S[0], S[1]); }
/* #16: present_value(future_value(cashflow, r, n), r, n) */
static int64_t ref_pv_of_fv(void)           { return r_present_value(r_future_value(S[0], S[1], S[2]), S[1], S[2]); }
/* #17: fibonacci(n) + factorial(n) */
static int64_t ref_fib_fact_add(void)       { return r_fibonacci(S[0]) + r_factorial(S[0]); }
/* #18: gross - tax_amount(gross, rate) */
static int64_t ref_gross_minus_tax(void)    { return S[0] - r_tax_amount(S[0], S[1]); }
/* #19: compound(P, r, n) - P  (same shape as #2 but distinct entry) */
static int64_t ref_compound_minus_p(void)   { return r_compound(S[0], S[1], S[2]) - S[0]; }
/* #20: clamp(sigmoid(x), lo, hi) — same shape as #4 */
static int64_t ref_sigmoid_clamped(void)    { return r_clamp(r_sigmoid(S[0]), S[1], S[2]); }

typedef struct {
    const char *name;
    int64_t (*fn)(void);
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
};
static const int references_count = (int)(sizeof(references) / sizeof(references[0]));

int wiring_reference_compute(const char *name, int64_t *out) {
    if (!name || !out) return 0;
    for (int i = 0; i < references_count; i++) {
        if (strcmp(references[i].name, name) == 0) {
            *out = references[i].fn();
            return 1;
        }
    }
    return 0;
}
