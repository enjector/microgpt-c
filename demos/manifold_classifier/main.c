/*
 * MicroGPT-C — Manifold Classifier Demo (Phase 1b diagnostic)
 *
 * Bounded experiment to answer the architectural question raised by
 * Phase 1a's negative result: when the wiring organelle produces 16/16
 * unanimous wrong answers, can a geodesic-based family classifier
 * predict the right family from the prompt?
 *
 * If YES: the bottleneck is generation (the model can't produce the
 *         right candidate even when the right family is identifiable).
 *         Manifold-learning composition is the right path forward,
 *         specifically anchor-conditional generation.
 *
 * If NO:  the manifold approach fails at the classification step. The
 *         keyword embedding is too crude or the families aren't
 *         linearly separable in 12D. Need a learned encoder.
 *
 * Method:
 *   1. Define an anchor table: ~20 template-family anchors, each a 12D
 *      coordinate. Use slot-based assignment (one-hot over 12 slots,
 *      with overflow families bucketing into the highest slots).
 *   2. Define a keyword bag per family (3-8 lowercase words).
 *   3. Load pipeline_corpus_held_out.txt (20 prompts, each annotated
 *      with `# REFERENCE: <family>`).
 *   4. For each prompt:
 *        - Lowercase + tokenise.
 *        - For each family, count keyword hits.
 *        - Project hit counts to 12D anchor space (each family contributes
 *          to its anchor slot, normalised).
 *        - Run Geodesic flat-metric distance from prompt embedding to
 *          each anchor.
 *        - Top-1 prediction = nearest anchor.
 *   5. Tally top-1 accuracy. Compare against:
 *        - the EXPECTED family (held-out file)
 *        - per-prompt outcome on the 6 prompts the wiring system fails
 *
 * Pure C99, links against microgpt_lib for Geodesic. No retraining.
 *
 * Reference: docs/research/RESEARCH_MANIFOLD_LEARNING.md §13.4 Phase 1b.
 */

#include "microgpt_geodesic.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Configuration
 * ========================================================================= */
#define MAX_FAMILIES   24
#define MAX_KEYWORDS    8
#define MAX_KW_LEN     32
#define MAX_PROMPTS    96
#define MAX_PROMPT_LEN 256
#define MAX_REF_LEN     64

/* =========================================================================
 * Family table — 20 template-family anchors covering the held-out
 * reference families. Each family has a slot index in 12D and a
 * keyword bag. Slot assignment groups semantically-related families
 * to share a slot when we exceed 12 (overflow buckets).
 * ========================================================================= */
typedef struct {
    const char *name;
    int slot;                                 /* 0..11, the anchor coord */
    const char *keywords[MAX_KEYWORDS];       /* NULL-terminated list */
} FamilyAnchor;

/* Phase 2b: 20D unique-slot table (matches wiring_geo_classifier.c). */
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
};
static const int N_FAMILIES = (int)(sizeof(FAMILIES) / sizeof(FAMILIES[0]));

/* =========================================================================
 * Held-out prompt loader — same format as wiring_organelle's loader.
 * ========================================================================= */
typedef struct {
    char prompt[MAX_PROMPT_LEN];
    char reference[MAX_REF_LEN];
} HeldOutItem;

static int load_held_out(const char *path, HeldOutItem *items, int max) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return 0;
    }
    int n = 0;
    char line[512];
    char pending_ref[MAX_REF_LEN] = {0};
    while (fgets(line, sizeof(line), fp) && n < max) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (strncmp(line, "# REFERENCE:", 12) == 0) {
            const char *p = line + 12;
            while (*p == ' ') p++;
            strncpy(pending_ref, p, sizeof(pending_ref) - 1);
            pending_ref[sizeof(pending_ref) - 1] = '\0';
            continue;
        }
        if (strncmp(line, "// ", 3) == 0) {
            strncpy(items[n].prompt, line + 3, MAX_PROMPT_LEN - 1);
            items[n].prompt[MAX_PROMPT_LEN - 1] = '\0';
            strncpy(items[n].reference, pending_ref, MAX_REF_LEN - 1);
            items[n].reference[MAX_REF_LEN - 1] = '\0';
            pending_ref[0] = '\0';
            n++;
        }
    }
    fclose(fp);
    return n;
}

/* =========================================================================
 * Embedder: lowercase + tokenise + count family keyword hits → 12D.
 * ========================================================================= */
static void embed_prompt(const char *prompt, double coords[GEO_DIMS]) {
    char buf[MAX_PROMPT_LEN];
    size_t n = strnlen(prompt, sizeof(buf) - 1);
    for (size_t i = 0; i < n; i++) {
        char c = prompt[i];
        buf[i] = (char)tolower((unsigned char)c);
    }
    buf[n] = '\0';

    /* Per-family hit count */
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

    /* Aggregate hits into 12D slots. */
    double slot_acc[GEO_DIMS] = {0};
    for (int f = 0; f < N_FAMILIES; f++) {
        if (FAMILIES[f].slot >= 0 && FAMILIES[f].slot < GEO_DIMS) {
            slot_acc[FAMILIES[f].slot] += (double)fam_hits[f];
        }
    }

    /* Normalise to unit-L2; if zero, all-zero is fine (max-distance to all anchors). */
    double sum_sq = 0.0;
    for (int d = 0; d < GEO_DIMS; d++) sum_sq += slot_acc[d] * slot_acc[d];
    double norm = (sum_sq > 0.0) ? sqrt(sum_sq) : 1.0;
    for (int d = 0; d < GEO_DIMS; d++) coords[d] = slot_acc[d] / norm;
}

/* =========================================================================
 * Anchor coords: each family's 12D anchor is its slot one-hot.
 * Overflow families share slots — Geodesic distance can't disambiguate
 * intra-slot, so we use a per-family secondary axis to break ties.
 * ========================================================================= */
static void family_anchor_coords(int family_idx, double coords[GEO_DIMS]) {
    for (int d = 0; d < GEO_DIMS; d++) coords[d] = 0.0;
    int slot = FAMILIES[family_idx].slot;
    if (slot < 0 || slot >= GEO_DIMS) return;
    coords[slot] = 1.0;
    /* Tiny secondary jitter on a different slot, indexed by family idx,
     * so two families sharing slot 5 land at different 12D points. */
    int jitter_slot = (slot + 1 + family_idx) % GEO_DIMS;
    if (jitter_slot != slot) coords[jitter_slot] = 0.05;
}

/* =========================================================================
 * Top-1 family prediction by Euclidean distance (Geodesic flat metric).
 * Returns the family index in FAMILIES[] of the nearest anchor.
 * ========================================================================= */
static int predict_family(const double prompt_emb[GEO_DIMS]) {
    double best = 1e30;
    int best_idx = -1;

    for (int f = 0; f < N_FAMILIES; f++) {
        double anchor[GEO_DIMS];
        family_anchor_coords(f, anchor);
        double dev[GEO_DIMS];
        for (int d = 0; d < GEO_DIMS; d++) dev[d] = anchor[d] - prompt_emb[d];

        GeodesicResult r = geo_compute_euclidean(dev, NULL, 0.0);
        double dist = r.tension;
        if (dist < best) { best = dist; best_idx = f; }
    }
    return best_idx;
}

/* =========================================================================
 * Main
 * ========================================================================= */
int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "pipeline_corpus_held_out.txt";

    HeldOutItem items[MAX_PROMPTS];
    int n_held = load_held_out(path, items, MAX_PROMPTS);
    if (n_held == 0) return 1;

    printf("================================================================\n");
    printf("  MicroGPT-C — Manifold Classifier Diagnostic (Phase 1b)\n");
    printf("================================================================\n\n");
    printf("Loaded %d held-out prompts from %s\n", n_held, path);
    printf("Anchor table: %d families, %d-D Geodesic flat metric\n\n",
           N_FAMILIES, GEO_DIMS);

    /* Per-prompt evaluation. */
    int n_correct = 0;
    int n_known   = 0;     /* prompts where reference is in our table */
    printf("%-3s %-22s %-22s %-6s  %s\n", "#", "REFERENCE", "PREDICTED", "MATCH",
           "PROMPT (truncated)");
    printf("---------------------------------------------------------------"
           "-----------------------------------------------------\n");

    for (int i = 0; i < n_held; i++) {
        const char *ref = items[i].reference;

        /* Find ref's family idx in our table (or -1 if absent). */
        int ref_idx = -1;
        for (int f = 0; f < N_FAMILIES; f++) {
            if (strcmp(FAMILIES[f].name, ref) == 0) { ref_idx = f; break; }
        }

        double prompt_emb[GEO_DIMS];
        embed_prompt(items[i].prompt, prompt_emb);

        int pred_idx = predict_family(prompt_emb);
        const char *pred = (pred_idx >= 0) ? FAMILIES[pred_idx].name : "(none)";

        int match = 0;
        if (ref_idx >= 0) {
            n_known++;
            /* Match if predicted family is the same OR shares a slot
             * (semantic equivalence). Strict name-match in primary metric. */
            if (pred_idx == ref_idx) {
                match = 2;  /* exact */
                n_correct++;
            } else if (pred_idx >= 0 &&
                       FAMILIES[pred_idx].slot == FAMILIES[ref_idx].slot) {
                match = 1;  /* slot-equivalent (semantic neighbour) */
            }
        }

        const char *match_str = (match == 2) ? "EXACT" : (match == 1) ? "SLOT" : "no";
        char trunc[80];
        strncpy(trunc, items[i].prompt, sizeof(trunc) - 1);
        trunc[sizeof(trunc) - 1] = '\0';
        if (strlen(trunc) > 60) { trunc[57] = '.'; trunc[58] = '.'; trunc[59] = '.'; trunc[60] = '\0'; }

        printf("%-3d %-22s %-22s %-6s  %s\n",
               i + 1, ref, pred, match_str, trunc);
    }

    printf("\n================================================================\n");
    printf("  HEADLINE\n");
    printf("================================================================\n");
    printf("Total held-out prompts: %d\n", n_held);
    printf("References in anchor table: %d\n", n_known);
    printf("Top-1 EXACT family match:   %d/%d (%.0f%%)\n",
           n_correct, n_known,
           n_known > 0 ? 100.0 * n_correct / n_known : 0.0);

    /* Highlight the 6 prompts the wiring system fails on (per Phase 1a audit). */
    static const char *failing_refs[] = {
        "bmi_clamped",
        "compound_interest",
        "weighted_three",
        "apply_tax",         /* take home pay → ref is apply_tax */
        "clamped_average",
        "fib_fact_add",
    };
    static const int n_failing = (int)(sizeof(failing_refs) / sizeof(failing_refs[0]));

    printf("\nWiring-system-failing prompts (per Phase 1a audit):\n");
    int n_failing_predicted = 0;
    for (int j = 0; j < n_failing; j++) {
        for (int i = 0; i < n_held; i++) {
            if (strcmp(items[i].reference, failing_refs[j]) == 0) {
                double prompt_emb[GEO_DIMS];
                embed_prompt(items[i].prompt, prompt_emb);
                int pred_idx = predict_family(prompt_emb);
                const char *pred = (pred_idx >= 0) ? FAMILIES[pred_idx].name : "(none)";
                int correct = (strcmp(pred, failing_refs[j]) == 0);
                if (correct) n_failing_predicted++;
                printf("  [%d] ref=%-22s  pred=%-22s  %s\n",
                       i + 1, failing_refs[j], pred,
                       correct ? "CORRECTLY classified" : "miss");
                break;
            }
        }
    }
    printf("\nGeodesic correctly classified %d/%d wiring-failing prompts.\n",
           n_failing_predicted, n_failing);

    printf("\nInterpretation:\n");
    if (n_failing_predicted >= n_failing - 1) {
        printf("  POSITIVE: manifold-based classification recovers most/all prompts\n");
        printf("  the wiring system fails on. The bottleneck IS generation, not\n");
        printf("  classification. Anchor-conditional generation is the right next step.\n");
    } else if (n_failing_predicted >= 2) {
        printf("  PARTIAL: manifold classifier recovers some but not all. Either the\n");
        printf("  keyword embedding is too crude, or some prompts genuinely have\n");
        printf("  ambiguous family signal. Need a learned encoder (EKAN train).\n");
    } else {
        printf("  NEGATIVE: manifold classifier fails on the same prompts as the\n");
        printf("  wiring system. The keyword embedding is insufficient — the\n");
        printf("  manifold thesis needs a learned encoder, not handcrafted features.\n");
    }
    printf("\n");

    return 0;
}
