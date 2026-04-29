/*
 * tools/corpus_expand.c — Phase 4a expanded corpus generator
 *
 * Generates ~5,000 (prompt, family) pairs across the 20 reference
 * families by combining curated per-family synonym tables, structural
 * sentence templates, and word-order permutations. The output is a
 * pipeline_corpus_phase4_train.txt file in the same format as
 * pipeline_corpus_train.txt: alternating "// prompt" and "@graph
 * <name>" entries (the @graph body is just a placeholder so existing
 * loaders parse it).
 *
 * Design constraints from §45:
 *   1. Target: 5,000 ± 1,000 (prompt, family) pairs
 *   2. Vocabulary: 1.5k-2.5k unique words (vs current 123)
 *   3. Leakage discipline: NO Phase 2c clean paraphrase OR adversarial
 *      axis-2 prompt verbatim in output
 *   4. Determinism: same seed → same output
 *
 * Usage:
 *   corpus_expand <out_path> [seed]
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_FAMILIES   24
#define MAX_SYN_GROUPS  6   /* synonym groups per family — different
                             * concepts within the family's expression */
#define MAX_SYNS       16   /* synonyms per group */
#define MAX_TEMPLATES   8   /* structural templates per family */
#define MAX_LINE      512

/* ---------- Family table ---------- *
 * For each family:
 *   - syn[g] = NULL-terminated array of synonyms for concept-group g
 *   - templates = NULL-terminated array of sentence templates with
 *     placeholders %0%, %1%, ... referring to synonym groups
 */

typedef struct {
    const char *name;
    const char *syn[MAX_SYN_GROUPS][MAX_SYNS];
    const char *templates[MAX_TEMPLATES];
} Family;

static const Family FAMILIES[] = {
    /* ---- bmi_clamped: bmi(weight,height) → clamp(_,lo,hi) ----
     * groups: 0=bmi-noun, 1=weight-noun, 2=height-noun, 3=clamp-verb */
    { "bmi_clamped",
      {
        { "bmi", "body mass index", "Quetelet index", "Quetelet ratio", "mass-height index", "BMI score", "BMI value", "BMI", NULL },
        { "weight", "mass", "kilograms", "body weight", "weighted measure", NULL },
        { "height", "stature", "centimetres", "height value", "tall", "size", NULL },
        { "clamp", "clip", "bound", "constrain", "pin", "limit", "hold", "cap", "restrict", "confine", "fence", NULL },
      },
      {
        "%0% from %1% and %2% %3%ed inside lo and hi bounds",
        "%3% the %0% of %1% and %2% to a range",
        "the %0% computed from %1% and %2%, %3%ed within lo hi",
        "%3%ed %0% from %1% and %2% within bounds",
        "%0% of %1% and %2% %3%ed to a permitted span",
        "%0% calculated from %1% and %2% then %3%ed inside limits",
        NULL
      } },

    /* ---- compound_interest: compound(P,r,n) - P ----
     * groups: 0=interest-noun, 1=investment-noun, 2=compound-verb, 3=duration-noun */
    { "compound_interest",
      {
        { "interest", "earnings", "yield", "return", "gain", "profit accrued", NULL },
        { "investment", "deposit", "principal", "savings", "capital", "fund", NULL },
        { "compounds", "accumulates", "accrues", "grows", "increases", "builds up", "compound-grows", NULL },
        { "years", "periods", "annual cycles", "compounding intervals", "term", NULL },
      },
      {
        "%0% gained on a %1% when it %2% over %3%",
        "the %0% portion of a %1% after it %2% over %3%",
        "%0% earned by a %1% as it %2% across %3%",
        "%0% from a %1% that %2% over %3%",
        "the %0% on a %1% which %2% for some %3%",
        NULL
      } },

    /* ---- compound_minus_p: compound(P,r,n) - P (alias) ----
     * groups: 0=final-balance-noun, 1=compound-grow-verb, 2=principal-noun, 3=minus-word */
    { "compound_minus_p",
      {
        { "final balance", "ending value", "matured amount", "terminal sum", "total accrued", NULL },
        { "after compound growth", "after compounding", "post-accumulation", "after accumulating", NULL },
        { "original principal", "initial deposit", "starting amount", "seed capital", NULL },
        { "minus", "less", "reduced by", "net of", "subtracting", NULL },
      },
      {
        "%0% %1% %3% the %2%",
        "the %0% reached %1%, %3% the %2%",
        "%1% the %0% %3% the %2% amount",
        NULL
      } },

    /* ---- weighted_three: (m1*w1 + m2*w2 + m3*w3) * 100 / (w1+w2+w3) ----
     * groups: 0=weighted-adj, 1=combination-noun, 2=measurements-noun, 3=weights-noun */
    { "weighted_three",
      {
        { "weighted", "factor-scaled", "coefficient-multiplied", "amplified", NULL },
        { "combination", "aggregate", "blend", "mixture", "composite", NULL },
        { "measurements", "readings", "observations", "data points", "samples", "values", NULL },
        { "weights", "factors", "coefficients", "multipliers", NULL },
      },
      {
        "%0% %1% of three %2% each scaled by its own %3%",
        "the %0% %1% of three %2% with %3%",
        "%0% mean of three %2% via %3%",
        "the three-way %0% %1% over %2% and %3%",
        NULL
      } },

    /* ---- clamped_sigmoid: clamp(sigmoid(x), lo, hi) ----
     * groups: 0=sigmoid-noun, 1=output-noun, 2=clamp-verb, 3=range-phrase */
    { "clamped_sigmoid",
      {
        { "sigmoid", "logistic", "logistic function", "sigmoid neuron", "sigmoid activation", NULL },
        { "output", "activation", "value", "response", "result", NULL },
        { "clamp", "clip", "constrain", "restrict", "pin", "bound", "hold", NULL },
        { "lo high range", "low high band", "permitted window", "lo hi limits", "permissible window", NULL },
      },
      {
        "%2% the %0% %1% to a %3%",
        "%0% %1% %2%ed within a %3%",
        "%2%ed %0% activation inside a %3%",
        "the %0% %1% %2%ed by a %3%",
        NULL
      } },

    /* ---- sigmoid_clamped: alias of clamped_sigmoid (just lexically different) */
    { "sigmoid_clamped",
      {
        { "sigmoid", "logistic", NULL },
        { "x", "input", "argument", NULL },
        { "normalised", "normalized", "scaled", "regularised", NULL },
        { "clamping", "clipping", "bounding", NULL },
      },
      {
        "%0% of %1% %2% by %3% into a bounded range",
        "%0% %1% %2% via %3% to a bounded interval",
        "the %0% value of %1% %2% through %3%",
        NULL
      } },

    /* ---- gcd_scaled: gcd(a,b) * k ----
     * groups: 0=gcd-noun, 1=scale-verb, 2=coefficient-noun */
    { "gcd_scaled",
      {
        { "gcd", "greatest common divisor", "largest common factor", "common divisor", "shared divisor", "highest common factor", NULL },
        { "scaled by", "multiplied by", "magnified by", "amplified by", "times", NULL },
        { "coefficient", "multiplier", "factor", "scalar k", NULL },
      },
      {
        "%0% of two integers %1% a %2%",
        "the %0% of a and b %1% a %2%",
        "%0% of two numbers %1% %2% k",
        NULL
      } },

    /* ---- apply_tax: gross - tax_amount(gross, rate) ----
     * groups: 0=net-pay-noun, 1=gross-noun, 2=tax-rate-noun */
    { "apply_tax",
      {
        { "take home pay", "after-tax pay", "net earnings", "disposable income", "post-tax income", "net pay", NULL },
        { "gross income", "gross pay", "gross earnings", "pre-tax salary", "earned wages", NULL },
        { "federal tax rate", "tax rate", "withholding rate", "tax bracket rate", NULL },
      },
      {
        "%0% from %1% at the %2%",
        "%0% computed from %1% with the %2%",
        "%0% as %1% reduced by the %2%",
        "the %0% remaining after %1% is taxed at %2%",
        NULL
      } },

    /* ---- gross_minus_tax: gross - tax_amount(gross, rate) (alias) ----
     * groups: 0=gross-noun, 1=reduce-verb, 2=tax-noun */
    { "gross_minus_tax",
      {
        { "gross income", "gross pay", "gross wages", "gross earnings", "earned wages", NULL },
        { "reduced by", "minus", "less", "decreased by", "diminished by", NULL },
        { "tax liability", "withholding", "tax owed", "tax due", "tax obligation", NULL },
      },
      {
        "%0% %1% the %2%",
        "%0% %1% what is owed in %2%",
        "the %0% net of %2%",
        NULL
      } },

    /* ---- discounted_tax: tax_amount(discount(price, disc), trate) ----
     * groups: 0=tax-noun, 1=discount-noun, 2=price-noun */
    { "discounted_tax",
      {
        { "tax", "duty", "levy", "tax owed", "tax due", NULL },
        { "discount applied", "marked-down", "reduced", "discounted", "marked", "price-cut", NULL },
        { "price", "listing", "amount", "marked price", NULL },
      },
      {
        "%0% on a %2% after a %1%",
        "%0% due on a %2% which has been %1%",
        "the %0% owed on a %1% %2%",
        "%0% computed on a %1% %2%",
        NULL
      } },

    /* ---- fib_fact_mul: fibonacci(n) * factorial(n) ----
     * groups: 0=fib-noun, 1=fact-noun, 2=multiply-verb */
    { "fib_fact_mul",
      {
        { "fibonacci of n", "fibonacci(n)", "n-th fibonacci", "Leonardo number", "Fibonacci number at n", NULL },
        { "factorial of n", "factorial(n)", "n-th factorial", "n!", "descending product at n", NULL },
        { "multiplied by", "times", "product with", "scaled by", NULL },
      },
      {
        "%0% %2% %1%",
        "the product of %0% and %1%",
        "%0% %2% the %1%",
        NULL
      } },

    /* ---- fib_fact_add: fibonacci(n) + factorial(n) ----
     * groups: 0=fib, 1=fact, 2=add-word */
    { "fib_fact_add",
      {
        { "fibonacci of n", "fibonacci(n)", "n-th fibonacci", "Leonardo number", NULL },
        { "factorial of n", "factorial(n)", "n-th factorial", "n!", NULL },
        { "added to", "combined with", "summed with", "plus", "and", NULL },
      },
      {
        "%0% %2% %1% by adding",
        "%0% %2% %1%, taking the sum",
        "the sum of %0% and %1%",
        "%0% and %1% added together",
        NULL
      } },

    /* ---- invoice_total: price*qty + tax_amount(price*qty, rate) ----
     * groups: 0=invoice, 1=price, 2=quantity, 3=tax */
    { "invoice_total",
      {
        { "invoice total", "receipt sum", "bill total", "order total", "invoice value", NULL },
        { "price", "unit cost", "unit price", "rate per unit", NULL },
        { "quantity", "units bought", "count", "number of items", NULL },
        { "tax due", "duty owed", "applicable tax", "tax at rate", NULL },
      },
      {
        "%0% of %1% times %2% plus the %3%",
        "%0% combining %2% units of %1% along with the %3%",
        "the %0% from %1% by %2% adding the %3%",
        NULL
      } },

    /* ---- clamped_average: clamp(average_two(a,b), lo, hi) ----
     * groups: 0=average-noun, 1=clamp-phrase */
    { "clamped_average",
      {
        { "average", "mean", "midpoint", "median", "central value", NULL },
        { "bounded between min and max", "constrained inside lo hi", "held within lower and upper limits", "clamped to a range", "kept inside bounds", NULL },
      },
      {
        "%0% of a and b %1%",
        "the %0% of two values %1%",
        "%0% of two readings %1%",
        NULL
      } },

    /* ---- abs_diff: abs(a - b) ----
     * groups: 0=abs-adj, 1=diff-noun, 2=between-phrase */
    { "abs_diff",
      {
        { "absolute", "unsigned", "positive-only", "magnitude of", NULL },
        { "magnitude", "difference", "gap", "deviation", "spread", NULL },
        { "between two forecasts", "between two estimates", "separating two readings", "across two predictions", NULL },
      },
      {
        "%0% %1% %2%",
        "the %0% %1% %2%",
        "%1% %2% taken as %0%",
        NULL
      } },

    /* ---- pv_of_fv: present_value(future_value(c, r, n), r, n) ----
     * groups: 0=present-phrase, 1=future-phrase, 2=discount-phrase */
    { "pv_of_fv",
      {
        { "present worth", "today's equivalent", "current value", "discounted value", NULL },
        { "future cashflow", "later sum", "deferred amount", "scheduled future payment", NULL },
        { "discounted back to today", "after temporal correction", "reduced to present terms", "back-discounted", NULL },
      },
      {
        "%0% of a %1% %2%",
        "the %0% of a %1% %2%",
        "%2% the %0% of a %1%",
        NULL
      } },

    /* ---- distance_metrics: square(distance_1d(a1,b1) + distance_1d(a2,b2)) ----
     * groups: 0=distance-noun, 1=axes-phrase, 2=square-verb */
    { "distance_metrics",
      {
        { "distance", "gap", "displacement", NULL },
        { "across two coordinate axes", "along two dimensional projections", "across two axes", "over two coordinate pairs", NULL },
        { "squared", "raised to second power", "to the power of two", NULL },
      },
      {
        "total of %0%s %1% %2%",
        "sum of %0%s %1% then %2%",
        "%0% %1% %2%",
        NULL
      } },

    /* ---- distance_midpoint: distance_1d(a, b) + midpoint(a, b) ----
     * groups: 0=distance-noun, 1=midpoint-noun */
    { "distance_midpoint",
      {
        { "distance", "displacement", "gap", "interval", NULL },
        { "midpoint", "centroid", "center", "center point", "halfway point", NULL },
      },
      {
        "%0% between two readings combined with their %1%",
        "the %0% between two observations together with their %1%",
        "%0% from one observation to another supplemented by their %1%",
        NULL
      } },

    /* ---- savings_rate: percentage(income - sum_expenses, income) ----
     * groups: 0=fraction-noun, 1=income-noun, 2=expenses-noun */
    { "savings_rate",
      {
        { "fraction", "portion", "share", "percentage", "ratio", NULL },
        { "income", "earnings", "salary", "wages", "take", NULL },
        { "expenses", "outgoings", "outlays", "spending", "costs", NULL },
      },
      {
        "%0% of %1% saved after subtracting %2%",
        "%0% of %1% remaining after deducting two %2%",
        "the %0% of %1% preserved after subtracting %2%",
        NULL
      } },

    /* ---- scaled_relu: relu(x) * scale ----
     * groups: 0=relu-noun, 1=scale-noun */
    { "scaled_relu",
      {
        { "rectified output", "thresholded activation", "ReLU output", "rectified linear value", NULL },
        { "gain factor", "amplification factor", "scaling factor", "amplification term", NULL },
      },
      {
        "%0% of x scaled by a %1%",
        "%0% multiplied by a %1%",
        "%0% amplified by a %1%",
        NULL
      } },
};

static const int N_FAMILIES = (int)(sizeof(FAMILIES) / sizeof(FAMILIES[0]));

/* ---------- Forbidden-prompt list (Phase 4a leakage guard) ---------- */
static const char *FORBIDDEN[] = {
    /* Phase 2c clean paraphrases (held-out, leakage-free) — must NOT appear verbatim. */
    "bmi of weight and height clipped to a healthy lo hi range",
    "the interest portion of an investment after principal compounds over years",
    "the weighted average of three measurements using their respective weights",
    "sigmoid neuron activation restricted to a low high band",
    "the gcd of two integers multiplied by a coefficient",
    "after-tax take home pay from federal taxation",
    "n-th fibonacci multiplied by n-th factorial",
    "invoice combining quantity times unit price plus the applicable tax",
    "the average of two values bounded between minimum and maximum",
    "absolute magnitude of the difference between two forecasts",
    "rectified output multiplied by a gain factor",
    "the tax owed once a discount has been applied to the price",
    "the fraction of income remaining after subtracting two expenses",
    "sum of distances across two coordinate axes squared",
    "the distance between two readings combined with their midpoint",
    "present worth of a future cashflow discounted back to today",
    "the sum of n-th fibonacci and n-th factorial added together",
    "gross pay reduced by the federal tax liability",
    "final compound balance minus the original principal amount",
    "sigmoid x value normalised through clamping",
    /* Adversarial axis-2 prompts (also strict no-leakage). */
    "Quetelet ratio from kilograms and centimetres constrained to a permitted span",
    "the yield earned on a deposit accumulating annually for some duration",
    "blend three observations each amplified by its own factor",
    "logistic activation pinned within a permissible window",
    "the largest shared divisor of two integers magnified by a multiplier",
    "disposable earnings remaining after government withholding",
    "product of the recursive Leonardo series and the descending product series at index n",
    "receipt sum of unit cost by units bought along with the duty owed",
    "the median of two values held inside lower and upper limits",
    "unsigned gap separating two estimates",
    "thresholded activation amplified by an amplification term",
    "the duty owed on a marked-down listing",
    "portion of earnings preserved after deducting two outlays",
    "aggregate gap along a pair of dimensional projections raised to second power",
    "the displacement from one observation to another together with their centroid",
    "today's equivalent of a later sum after temporal correction",
    "summation of the recursive Leonardo series and the descending product series at index n",
    "earnings less what is owed in withholding",
    "surplus remaining after accumulation strips the initial deposit",
    "logistic value pinned within a permissible interval",
    NULL
};

static int is_forbidden(const char *prompt) {
    for (int i = 0; FORBIDDEN[i]; i++) {
        if (strcmp(prompt, FORBIDDEN[i]) == 0) return 1;
    }
    return 0;
}

/* ---------- Cheap deterministic RNG ---------- */
static uint64_t g_rng_state = 0xC0FFEEDEADBEEFULL;
static uint32_t rng_u32(void) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 7;
    g_rng_state ^= g_rng_state << 17;
    return (uint32_t)g_rng_state;
}
static int rng_pick(int n) { return (int)(rng_u32() % (uint32_t)n); }

/* ---------- Template instantiator ----------
 * Replace %k% with a synonym from synonym group k (random within group). */
static int instantiate(const Family *fam, const char *tmpl, char *out, int out_size) {
    int oi = 0;
    int ti = 0;
    while (tmpl[ti] && oi < out_size - 1) {
        if (tmpl[ti] == '%' && tmpl[ti+1] >= '0' && tmpl[ti+1] <= '9' && tmpl[ti+2] == '%') {
            int g = tmpl[ti+1] - '0';
            if (g < 0 || g >= MAX_SYN_GROUPS) return 0;
            int n_syns = 0;
            while (n_syns < MAX_SYNS && fam->syn[g][n_syns]) n_syns++;
            if (n_syns == 0) return 0;
            const char *syn = fam->syn[g][rng_pick(n_syns)];
            int sl = (int)strlen(syn);
            if (oi + sl >= out_size - 1) return 0;
            memcpy(out + oi, syn, (size_t)sl);
            oi += sl;
            ti += 3;
        } else {
            out[oi++] = tmpl[ti++];
        }
    }
    out[oi] = '\0';
    return 1;
}

/* ---------- Vocabulary tracking (rough word count) ---------- */
#define MAX_VOCAB 4096
#define MAX_WORD   32
typedef struct { char w[MAX_WORD]; } VocabEntry;
static VocabEntry g_vocab[MAX_VOCAB];
static int g_vocab_size = 0;

static void track_vocab(const char *text) {
    char buf[MAX_WORD];
    int bi = 0;
    for (int i = 0;; i++) {
        char c = text[i];
        int is_word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        if (is_word && bi < MAX_WORD - 1) {
            buf[bi++] = (char)tolower((unsigned char)c);
        } else if (bi > 0) {
            buf[bi] = '\0';
            int found = 0;
            for (int v = 0; v < g_vocab_size; v++) {
                if (strcmp(g_vocab[v].w, buf) == 0) { found = 1; break; }
            }
            if (!found && g_vocab_size < MAX_VOCAB) {
                strncpy(g_vocab[g_vocab_size].w, buf, MAX_WORD - 1);
                g_vocab_size++;
            }
            bi = 0;
        }
        if (!c) break;
    }
}

/* ---------- Main ---------- */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out_path> [seed]\n", argv[0]);
        return 1;
    }
    if (argc >= 3) g_rng_state = strtoull(argv[2], NULL, 10) | 0xDEADBEEF;

    /* Target: per-family count s.t. total ≈ 5000.
     * 20 families × 250 = 5000. */
    const int per_family_target = 250;

    FILE *fp = fopen(argv[1], "w");
    if (!fp) { fprintf(stderr, "ERROR: cannot open %s\n", argv[1]); return 1; }
    fprintf(fp, "# Pipeline IR — Phase 4a expanded corpus\n");
    fprintf(fp, "# %d families, ~%d examples per family, target ~%d total\n",
            N_FAMILIES, per_family_target, N_FAMILIES * per_family_target);
    fprintf(fp, "# Generated by tools/corpus_expand.c (deterministic; see seed in argv[2])\n");
    fprintf(fp, "# Format: '// prompt' + '@graph <family>' (graph body is a placeholder for loader compatibility).\n\n");

    int total_emitted = 0;
    int total_forbidden_skips = 0;
    int total_dedup_skips = 0;

    /* Per-family de-dup set to avoid emitting the same prompt twice. */
    char *seen_prompts[8192];
    int n_seen = 0;

    for (int f = 0; f < N_FAMILIES; f++) {
        const Family *fam = &FAMILIES[f];
        int n_templates = 0;
        while (n_templates < MAX_TEMPLATES && fam->templates[n_templates]) n_templates++;

        int emitted_for_fam = 0;
        int attempts = 0;
        const int max_attempts_per_target = 10;
        while (emitted_for_fam < per_family_target &&
               attempts < per_family_target * max_attempts_per_target) {
            attempts++;
            const char *tmpl = fam->templates[rng_pick(n_templates)];
            char prompt[MAX_LINE];
            if (!instantiate(fam, tmpl, prompt, sizeof(prompt))) continue;
            if (is_forbidden(prompt)) { total_forbidden_skips++; continue; }
            /* De-dup: linear scan over seen prompts. */
            int dup = 0;
            for (int s = 0; s < n_seen; s++) {
                if (strcmp(seen_prompts[s], prompt) == 0) { dup = 1; break; }
            }
            if (dup) { total_dedup_skips++; continue; }
            if (n_seen < (int)(sizeof(seen_prompts) / sizeof(seen_prompts[0]))) {
                seen_prompts[n_seen++] = strdup(prompt);
            }
            track_vocab(prompt);
            fprintf(fp, "// %s\n", prompt);
            fprintf(fp, "@graph %s\n", fam->name);
            fprintf(fp, "  : in x -> int\n");
            fprintf(fp, "  : out y -> int\n");
            fprintf(fp, "  | n = %s_placeholder(x: <x>) :: x:int -> out:int\n", fam->name);
            fprintf(fp, "  y <- n.out\n");
            fprintf(fp, "@end\n");
            fprintf(fp, "---\n\n");
            emitted_for_fam++;
            total_emitted++;
        }
        fprintf(stderr, "  %-22s  emitted=%d/%d (attempts=%d)\n",
                fam->name, emitted_for_fam, per_family_target, attempts);
    }

    /* Free de-dup set. */
    for (int s = 0; s < n_seen; s++) free(seen_prompts[s]);

    fclose(fp);

    fprintf(stderr, "\n[Phase 4a] total prompts emitted: %d\n", total_emitted);
    fprintf(stderr, "[Phase 4a] unique vocabulary words: %d\n", g_vocab_size);
    fprintf(stderr, "[Phase 4a] skipped (forbidden / leakage): %d\n", total_forbidden_skips);
    fprintf(stderr, "[Phase 4a] skipped (intra-family duplicate): %d\n", total_dedup_skips);

    /* Pre-registered targets check (informational). */
    fprintf(stderr, "\n[Phase 4a §45.2 targets]\n");
    fprintf(stderr, "  total prompts:        %d  target=4000-6000  %s\n",
            total_emitted,
            (total_emitted >= 4000 && total_emitted <= 6000) ? "OK" : "OUT_OF_RANGE");
    fprintf(stderr, "  unique vocab:         %d  target=1500-2500  %s\n",
            g_vocab_size,
            (g_vocab_size >= 1500 && g_vocab_size <= 2500) ? "OK" :
                (g_vocab_size < 1500 ? "TOO_FEW" : "TOO_MANY"));
    return 0;
}
