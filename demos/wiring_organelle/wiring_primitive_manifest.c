/*
 * Wiring primitive manifest — implementation.
 * See wiring_primitive_manifest.h.
 */

#include "wiring_primitive_manifest.h"

#include <string.h>

/* All primitives are integer-typed (matching wiring_natives.c).  The manifest
 * lists each canonical name once (aliases like "abs" / "abs_val" both resolve
 * to abs_val at dispatch time but are kept under the canonical form here). */
static const WiringPrimitive g_manifest[] = {
    /* ---- Arithmetic ---- */
    { "add",            2, {PIPE_T_INT, PIPE_T_INT},               {"x","y"},           PIPE_T_INT,
      {"add","sum","plus","total","combine","accumulate", NULL} },
    { "subtract",       2, {PIPE_T_INT, PIPE_T_INT},               {"x","y"},           PIPE_T_INT,
      {"subtract","minus","less","difference","reduce","take-away", NULL} },
    { "multiply",       2, {PIPE_T_INT, PIPE_T_INT},               {"x","y"},           PIPE_T_INT,
      {"multiply","times","scaled","amplified","by","factor","gain","coefficient","product", NULL} },
    { "divide",         2, {PIPE_T_INT, PIPE_T_INT},               {"x","y"},           PIPE_T_INT,
      {"divide","quotient","split","over","ratio","per", NULL} },
    { "negate",         1, {PIPE_T_INT},                           {"x"},               PIPE_T_INT,
      {"negate","negated","invert-sign","flip-sign","opposite", NULL} },
    { "abs_val",        1, {PIPE_T_INT},                           {"x"},               PIPE_T_INT,
      {"absolute","magnitude","unsigned","abs", NULL} },
    { "square",         1, {PIPE_T_INT},                           {"x"},               PIPE_T_INT,
      {"square","squared", NULL} },
    { "cube",           1, {PIPE_T_INT},                           {"x"},               PIPE_T_INT,
      {"cube","cubed", NULL} },
    { "double_val",     1, {PIPE_T_INT},                           {"x"},               PIPE_T_INT,
      {"double","doubled","twice", NULL} },
    { "triple_val",     1, {PIPE_T_INT},                           {"x"},               PIPE_T_INT,
      {"triple","tripled","thrice", NULL} },

    /* ---- Min / max / distance ---- */
    { "min_two",        2, {PIPE_T_INT, PIPE_T_INT},               {"a","b"},           PIPE_T_INT,
      {"min","minimum","smallest","lower","least", NULL} },
    { "max_two",        2, {PIPE_T_INT, PIPE_T_INT},               {"a","b"},           PIPE_T_INT,
      {"max","maximum","largest","greater","greatest","most", NULL} },
    { "average_two",    2, {PIPE_T_INT, PIPE_T_INT},               {"a","b"},           PIPE_T_INT,
      {"average","mean","midpoint","middle","centre","center", NULL} },
    { "distance_1d",    2, {PIPE_T_INT, PIPE_T_INT},               {"a","b"},           PIPE_T_INT,
      {"distance","gap","span","apart","between", NULL} },
    { "midpoint",       2, {PIPE_T_INT, PIPE_T_INT},               {"a","b"},           PIPE_T_INT,
      {"midpoint","middle","centre","center","halfway", NULL} },
    { "mse",            2, {PIPE_T_INT, PIPE_T_INT},               {"a","b"},           PIPE_T_INT,
      {"mse","squared-error","squared-difference","ssd","l2", NULL} },

    /* ---- Bounding ---- */
    { "clamp",          3, {PIPE_T_INT, PIPE_T_INT, PIPE_T_INT},   {"x","lo","hi"},     PIPE_T_INT,
      {"clamp","clipped","bounded","constrained","pinned","limited","inside","within", NULL} },
    { "lerp",           3, {PIPE_T_INT, PIPE_T_INT, PIPE_T_INT},   {"a","b","t"},       PIPE_T_INT,
      {"lerp","interpolate","interpolated","interpolation","blend", NULL} },

    /* ---- Nonlinear ---- */
    { "sigmoid",        1, {PIPE_T_INT},                           {"x"},               PIPE_T_INT,
      {"sigmoid","logistic", NULL} },
    { "relu",           1, {PIPE_T_INT},                           {"x"},               PIPE_T_INT,
      {"relu","rectified","thresholded", NULL} },

    /* ---- Finance ---- */
    { "tax_amount",     2, {PIPE_T_INT, PIPE_T_INT},               {"amount","rate"},   PIPE_T_INT,
      {"tax","duty","levy","compute-tax", NULL} },
    { "apply_tax",      2, {PIPE_T_INT, PIPE_T_INT},               {"amount","rate"},   PIPE_T_INT,
      {"after-tax","withhold","withholding","apply-tax","net-of-tax","taxed", NULL} },
    { "percentage",     2, {PIPE_T_INT, PIPE_T_INT},               {"part","whole"},    PIPE_T_INT,
      {"percentage","percent","fraction","portion","share", NULL} },
    { "discount",       2, {PIPE_T_INT, PIPE_T_INT},               {"price","rate"},    PIPE_T_INT,
      {"discount","discounted","reduced","marked-down","sale", NULL} },
    { "markup",         2, {PIPE_T_INT, PIPE_T_INT},               {"price","rate"},    PIPE_T_INT,
      {"markup","marked-up","uplifted","raised","markup-by", NULL} },
    { "compound",       3, {PIPE_T_INT, PIPE_T_INT, PIPE_T_INT},   {"principal","rate","periods"}, PIPE_T_INT,
      {"compound","compounding","accumulating","compound-interest", NULL} },
    { "present_value",  3, {PIPE_T_INT, PIPE_T_INT, PIPE_T_INT},   {"future","rate","periods"},     PIPE_T_INT,
      {"present-value","pv","discounted-future","npv", NULL} },
    { "future_value",   3, {PIPE_T_INT, PIPE_T_INT, PIPE_T_INT},   {"present","rate","periods"},    PIPE_T_INT,
      {"future-value","fv","grown","grow", NULL} },

    /* ---- Number theory ---- */
    { "factorial",      1, {PIPE_T_INT},                           {"n"},               PIPE_T_INT,
      {"factorial","fact","Leonardo-fact", NULL} },
    { "fibonacci",      1, {PIPE_T_INT},                           {"n"},               PIPE_T_INT,
      {"fibonacci","fib","Leonardo","fib-sequence", NULL} },
    { "gcd",            2, {PIPE_T_INT, PIPE_T_INT},               {"a","b"},           PIPE_T_INT,
      {"gcd","greatest-common-divisor","common-divisor","shared-factor", NULL} },
    { "harmonic_n",     1, {PIPE_T_INT},                           {"n"},               PIPE_T_INT,
      {"harmonic","harmonic-sum","H_n","harmonic-number", NULL} },

    /* ---- Misc ---- */
    { "circle_area",    1, {PIPE_T_INT},                           {"r"},               PIPE_T_INT,
      {"circle-area","area-of-circle","disk-area", NULL} },
    { "kinetic_energy", 2, {PIPE_T_INT, PIPE_T_INT},               {"m","v"},           PIPE_T_INT,
      {"kinetic-energy","ke","energy-of-motion", NULL} },
    { "bmi",            2, {PIPE_T_INT, PIPE_T_INT},               {"weight","height"}, PIPE_T_INT,
      {"bmi","body-mass-index","mass-index", NULL} },
    { "power",          2, {PIPE_T_INT, PIPE_T_INT},               {"base","exp"},      PIPE_T_INT,
      {"power","raised-to","exponent","pow","to-the", NULL} },
};

static const int g_manifest_count = (int)(sizeof(g_manifest) / sizeof(g_manifest[0]));

const WiringPrimitive *wiring_primitive_manifest(int *out_count) {
    if (out_count) *out_count = g_manifest_count;
    return g_manifest;
}

const WiringPrimitive *wiring_primitive_find(const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < g_manifest_count; i++) {
        if (strcmp(g_manifest[i].name, name) == 0) return &g_manifest[i];
    }
    return NULL;
}
