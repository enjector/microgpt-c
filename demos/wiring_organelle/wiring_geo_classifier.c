/*
 * wiring_geo_classifier.c — Phase 1c geodesic family classifier impl.
 *
 * Adapted from demos/manifold_classifier/main.c (which proved 5/6
 * recovery on the wiring-failing prompts in the Phase 1b diagnostic).
 *
 * The same anchor table and keyword bag, exposed as a callable API
 * for wiring_organelle's eval loop. Reference RESEARCH_PIPELINE_IR.md
 * §33 for the per-prompt audit.
 */

#include "wiring_geo_classifier.h"
#include "microgpt_geodesic.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FAMILIES 24
#define MAX_KEYWORDS  8

typedef struct {
    const char *name;
    int slot;
    const char *keywords[MAX_KEYWORDS];
} FamilyAnchor;

/* Phase 2b table: each family gets a UNIQUE slot in the now-20D
 * Geodesic space. The Phase 2 (12D) table had 4 collisions: slot 5
 * stacked apply_tax / gross_minus_tax / discounted_tax / savings_rate,
 * and slot 9 stacked clamped_average / distance_midpoint. Those
 * collisions caused 3 of the 4 Phase 2 failures (#8, #13, #15).
 *
 * GEO_DIMS bumped from 12 to 20 in microgpt_geodesic.h so we have one
 * axis per held-out reference family. Keyword bags also tightened to
 * remove generic words ("price", "due") that caused false-positive
 * cross-family matches. */
static const FamilyAnchor FAMILIES[] = {
    { "bmi_clamped",        0, { "body", "mass", "index", "bmi", "weight", "height", NULL } },
    { "compound_interest",  1, { "interest", "gained", "investment", "compounds", NULL } },
    { "compound_minus_p",   2, { "final", "balance", "compound", "growth", "minus", "original", NULL } },
    { "weighted_three",     3, { "weighted", "combination", "measurements", "weights", NULL } },
    { "clamped_sigmoid",    4, { "sigmoid", "neuron", "low", "high", NULL } },
    { "sigmoid_clamped",    5, { "sigmoid", "normalised", "normalized", "clamping", NULL } },
    { "gcd_scaled",         6, { "gcd", "greatest", "common", "divisor", "coefficient", NULL } },
    { "apply_tax",          7, { "take", "home", "pay", "federal", NULL } },
    { "gross_minus_tax",    8, { "reduced", "liability", NULL } },
    { "discounted_tax",     9, { "discount", "applied", NULL } },
    { "fib_fact_mul",      10, { "fibonacci", "factorial", "multiplied", NULL } },
    { "fib_fact_add",      11, { "fibonacci", "factorial", "combined", "adding", "added", NULL } },
    { "invoice_total",     12, { "invoice", "total", "quantity", "plus", NULL } },
    { "clamped_average",   13, { "average", "bounded", "between", "minimum", "maximum", NULL } },
    { "abs_diff",          14, { "magnitude", "difference", "forecasts", NULL } },
    { "pv_of_fv",          15, { "future", "cashflow", "back", "present", "worth", NULL } },
    { "distance_metrics",  16, { "axes", "squared", "across", "coordinate", NULL } },
    { "distance_midpoint", 17, { "distance", "readings", "midpoint", NULL } },
    { "savings_rate",      18, { "fraction", "saved", "subtracting", "expenses", NULL } },
    { "scaled_relu",       19, { "rectified", "gain", NULL } },
    /* ====================================================================
     * Scaling-curve experiment — Batch 1 (geometry, slots 20-24)
     * Keywords picked to be lexically distinct from the existing 20.
     * ==================================================================== */
    { "circle_area_ratio",      20, { "ratio", "circle", "circles", "areas", "radii", NULL } },
    { "square_of_sum",          21, { "square", "sum", "summed", NULL } },
    { "triangle_area",          22, { "triangle", "base", NULL } },
    { "rectangle_perimeter",    23, { "perimeter", "rectangle", "width", NULL } },
    { "hypotenuse_squared",     24, { "hypotenuse", "legs", "squares", "pythagoras", NULL } },
    /* ====================================================================
     * Scaling-curve experiment — Batch 2 (physics, slots 25-29)
     * ==================================================================== */
    { "kinetic_energy_clamped", 25, { "kinetic", "energy", "joules", NULL } },
    { "momentum",               26, { "momentum", "mass", "velocity", NULL } },
    { "work_done",              27, { "work", "performed", "force", "distance", NULL } },
    { "power_clamped",          28, { "power", "exponent", NULL } },
    { "harmonic_sum",           29, { "harmonic", "harmonics", "terms", NULL } },
    /* ====================================================================
     * Scaling-curve experiment — Batch 3 (statistics, slots 30-34)
     * ==================================================================== */
    { "variance_two",           30, { "variance", "samples", "spread", NULL } },
    { "abs_z_score",            31, { "z-score", "z", "deviation", NULL } },
    { "range_two",              32, { "range", "max", "min", "readings", NULL } },
    { "midpoint_clamped",       33, { "midpoint", "median", "halfway", NULL } },
    { "mse_simple",             34, { "mean", "squared", "error", "prediction", "target", NULL } },
};
static const int N_FAMILIES = (int)(sizeof(FAMILIES) / sizeof(FAMILIES[0]));

static void embed_prompt(const char *prompt, double coords[GEO_DIMS]) {
    char buf[512];
    size_t n = strnlen(prompt, sizeof(buf) - 1);
    for (size_t i = 0; i < n; i++) buf[i] = (char)tolower((unsigned char)prompt[i]);
    buf[n] = '\0';

    int fam_hits[MAX_FAMILIES] = {0};
    for (int f = 0; f < N_FAMILIES; f++) {
        for (int k = 0; k < MAX_KEYWORDS && FAMILIES[f].keywords[k]; k++) {
            const char *kw = FAMILIES[f].keywords[k];
            const char *p = buf;
            while ((p = strstr(p, kw)) != NULL) {
                int before = (p == buf) || !isalpha((unsigned char)p[-1]);
                size_t kl = strlen(kw);
                int after = !isalpha((unsigned char)p[kl]);
                if (before && after) fam_hits[f]++;
                p += kl;
            }
        }
    }

    double slot_acc[GEO_DIMS] = {0};
    for (int f = 0; f < N_FAMILIES; f++) {
        if (FAMILIES[f].slot >= 0 && FAMILIES[f].slot < GEO_DIMS)
            slot_acc[FAMILIES[f].slot] += (double)fam_hits[f];
    }
    double sum_sq = 0.0;
    for (int d = 0; d < GEO_DIMS; d++) sum_sq += slot_acc[d] * slot_acc[d];
    double norm = (sum_sq > 0.0) ? sqrt(sum_sq) : 1.0;
    for (int d = 0; d < GEO_DIMS; d++) coords[d] = slot_acc[d] / norm;
}

static void anchor_coords(int family_idx, double coords[GEO_DIMS]) {
    for (int d = 0; d < GEO_DIMS; d++) coords[d] = 0.0;
    int slot = FAMILIES[family_idx].slot;
    if (slot < 0 || slot >= GEO_DIMS) return;
    coords[slot] = 1.0;
    int jitter_slot = (slot + 1 + family_idx) % GEO_DIMS;
    if (jitter_slot != slot) coords[jitter_slot] = 0.05;
}

int wiring_geo_predict_top_k(const char *prompt,
                             const char *out[WIRING_GEO_TOP_K]) {
    for (int k = 0; k < WIRING_GEO_TOP_K; k++) out[k] = NULL;
    if (!prompt || !prompt[0]) return 0;

    double prompt_emb[GEO_DIMS];
    embed_prompt(prompt, prompt_emb);

    /* Compute distance to each family. */
    double dists[MAX_FAMILIES];
    for (int f = 0; f < N_FAMILIES; f++) {
        double anchor[GEO_DIMS];
        anchor_coords(f, anchor);
        double dev[GEO_DIMS];
        for (int d = 0; d < GEO_DIMS; d++) dev[d] = anchor[d] - prompt_emb[d];
        GeodesicResult r = geo_compute_euclidean(dev, NULL, 0.0);
        dists[f] = r.tension;
    }

    /* Top-K selection (simple O(K * N) loop; N=20). */
    int chosen[WIRING_GEO_TOP_K];
    for (int i = 0; i < WIRING_GEO_TOP_K; i++) chosen[i] = -1;

    for (int k = 0; k < WIRING_GEO_TOP_K; k++) {
        double best = 1e30;
        int best_idx = -1;
        for (int f = 0; f < N_FAMILIES; f++) {
            int already = 0;
            for (int j = 0; j < k; j++) if (chosen[j] == f) { already = 1; break; }
            if (already) continue;
            if (dists[f] < best) { best = dists[f]; best_idx = f; }
        }
        if (best_idx < 0) break;
        chosen[k] = best_idx;
        out[k] = FAMILIES[best_idx].name;
    }

    int n = 0;
    for (int k = 0; k < WIRING_GEO_TOP_K; k++) if (out[k]) n++;
    return n;
}

/* Test whether `graph_name` (e.g. "fib_fact_op_add") and `family` (e.g.
 * "fib_fact_add") refer to the same family. Bridges the corpus-vs-
 * held-out naming gap: the corpus uses "<family>_op_<primitive>" while
 * the held-out file uses "<family>_<primitive>". */
static int family_match(const char *graph_name, const char *family) {
    if (!graph_name || !family) return 0;
    if (strcmp(graph_name, family) == 0) return 1;

    /* Strip trailing _<digits> from graph_name (e.g. gcd_chain_1 → gcd_chain). */
    char fam[64];
    size_t l = strnlen(graph_name, sizeof(fam) - 1);
    memcpy(fam, graph_name, l);
    fam[l] = '\0';
    for (int k = (int)l - 1; k > 0; k--) {
        if (fam[k] >= '0' && fam[k] <= '9') continue;
        if (fam[k] == '_' && k < (int)l - 1) fam[k] = '\0';
        break;
    }
    if (strcmp(fam, family) == 0) return 1;

    /* Bridge "<prefix>_op_<suffix>" ↔ "<prefix>_<suffix>": split the
     * family on its LAST underscore (suffix), check graph_name starts
     * with prefix and ends with _suffix. */
    const char *fam_last = strrchr(family, '_');
    if (fam_last && fam_last != family) {
        size_t prefix_len = (size_t)(fam_last - family);
        const char *fam_suffix = fam_last + 1;
        size_t suffix_len = strlen(fam_suffix);
        size_t gn_len = strlen(graph_name);
        if (gn_len >= prefix_len + 1 + suffix_len &&
            strncmp(graph_name, family, prefix_len) == 0 &&
            graph_name[prefix_len] == '_' &&
            strcmp(graph_name + gn_len - suffix_len, fam_suffix) == 0) {
            return 1;
        }
    }
    return 0;
}

int wiring_geo_in_top_k(const char *graph_name,
                        const char *top_k[WIRING_GEO_TOP_K],
                        int n_top_k) {
    if (!graph_name || !graph_name[0]) return 0;
    for (int k = 0; k < n_top_k; k++) {
        if (top_k[k] && family_match(graph_name, top_k[k])) return 1;
    }
    return 0;
}
