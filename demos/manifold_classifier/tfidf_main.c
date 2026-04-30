/*
 * tfidf_main.c — Phase 3a-lite TF-IDF centroid classifier
 *
 * Pre-registered Phase 3a plan calls for an EKAN-Network classifier;
 * this is the simpler learned-encoder proxy described in §40.7. If
 * TF-IDF centroids beat the handcoded keyword bag, EKAN follow-up
 * is justified. If they don't, the 408-example corpus is too small
 * for any learned encoder and Phase 3a-full would be pure overhead.
 *
 * Mechanism:
 *   1. Build vocabulary from 408 training prompts (~1000 words).
 *   2. Compute TF-IDF feature vector per training prompt.
 *   3. Average TF-IDF features per family → family centroid.
 *   4. For each test prompt: TF-IDF, cosine similarity to each
 *      centroid, top-1 nearest = predicted family.
 *
 * No EKAN, no neural net — just a learned linear classifier whose
 * parameters are derived from the corpus.
 *
 * Usage:
 *   ./manifold_tfidf_demo <test_file>
 *   (training corpus path is hard-coded to pipeline_corpus_train.txt)
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VOCAB    4096
#define MAX_WORD_LEN   32
#define MAX_FAMILIES   48
#define MAX_FAM_NAME_LEN 64
#define MAX_PROMPT_LEN 256
#define MAX_TRAIN_PROMPTS 8192
#define MAX_TEST_PROMPTS  64

/* ---------- Vocabulary (hash-table-free, linear-search) ---------- */
typedef struct {
    char word[MAX_WORD_LEN];
    int  doc_freq;   /* number of docs containing this word */
} VocabEntry;

static VocabEntry g_vocab[MAX_VOCAB];
static int g_vocab_size = 0;

static int vocab_lookup_or_add(const char *word, int adding) {
    for (int i = 0; i < g_vocab_size; i++) {
        if (strcmp(g_vocab[i].word, word) == 0) return i;
    }
    if (!adding || g_vocab_size >= MAX_VOCAB) return -1;
    int idx = g_vocab_size++;
    strncpy(g_vocab[idx].word, word, MAX_WORD_LEN - 1);
    g_vocab[idx].word[MAX_WORD_LEN - 1] = '\0';
    g_vocab[idx].doc_freq = 0;
    return idx;
}

static int vocab_lookup(const char *word) { return vocab_lookup_or_add(word, 0); }

/* ---------- Tokenizer ---------- */
static int tokenize(const char *line, char tokens[][MAX_WORD_LEN], int max) {
    int n = 0;
    const char *p = line;
    while (*p && n < max) {
        while (*p && !isalpha((unsigned char)*p)) p++;
        if (!*p) break;
        char buf[MAX_WORD_LEN];
        int o = 0;
        while (*p && (isalpha((unsigned char)*p) || *p == '_') && o < MAX_WORD_LEN - 1) {
            buf[o++] = (char)tolower((unsigned char)*p);
            p++;
        }
        buf[o] = '\0';
        if (o > 0) {
            strncpy(tokens[n], buf, MAX_WORD_LEN);
            n++;
        }
    }
    return n;
}

/* ---------- Family table (must mirror wiring_anchor_graphs.c order) ---------- */
static const char *FAMILY_NAMES[] = {
    "bmi_clamped", "compound_interest", "compound_minus_p", "weighted_three",
    "clamped_sigmoid", "sigmoid_clamped", "gcd_scaled", "apply_tax",
    "gross_minus_tax", "discounted_tax", "fib_fact_mul", "fib_fact_add",
    "invoice_total", "clamped_average", "abs_diff", "pv_of_fv",
    "distance_metrics", "distance_midpoint", "savings_rate", "scaled_relu",
    /* Scaling-curve experiment, slots 20-39 — denominator stable at 40
     * across all measurement batches. Centroids materialize only when
     * the family has corpus_expand.c entries and the corpus is regenerated. */
    "circle_area_ratio", "square_of_sum", "triangle_area",
    "rectangle_perimeter", "hypotenuse_squared",
    "kinetic_energy_clamped", "momentum", "work_done",
    "power_clamped", "harmonic_sum",
    "variance_two", "abs_z_score", "range_two",
    "midpoint_clamped", "mse_simple",
    "lerp_clamped", "cube_then_clamp", "gcd_with_offset",
    "harmonic_clamped", "percentage_of_average",
};
static const int N_FAMILIES = sizeof(FAMILY_NAMES) / sizeof(FAMILY_NAMES[0]);

static int family_idx_of(const char *name) {
    for (int i = 0; i < N_FAMILIES; i++) {
        if (strcmp(FAMILY_NAMES[i], name) == 0) return i;
    }
    return -1;
}

/* ---------- Training prompt loader ----------
 * Reads pipeline_corpus_train.txt, extracts each (prompt, @graph_name)
 * pair. The graph name maps to a family via the FAMILY_NAMES table
 * (with `_op_` suffix bridging from the corpus naming). */
typedef struct {
    char prompt[MAX_PROMPT_LEN];
    int  family_idx;
} TrainExample;

static TrainExample g_train[MAX_TRAIN_PROMPTS];
static int g_n_train = 0;

/* Strip trailing _<digits> and bridge _op_<suffix> ↔ _<suffix>
 * (same logic as wiring_geo_classifier.c::family_match). */
static int graph_name_to_family(const char *graph_name) {
    /* Try direct match first. */
    int idx = family_idx_of(graph_name);
    if (idx >= 0) return idx;

    /* Strip trailing _<digits>. */
    char buf[MAX_FAM_NAME_LEN];
    size_t l = strnlen(graph_name, sizeof(buf) - 1);
    memcpy(buf, graph_name, l);
    buf[l] = '\0';
    for (int k = (int)l - 1; k > 0; k--) {
        if (buf[k] >= '0' && buf[k] <= '9') continue;
        if (buf[k] == '_' && k < (int)l - 1) buf[k] = '\0';
        break;
    }
    idx = family_idx_of(buf);
    if (idx >= 0) return idx;

    /* Suffix bridge: <prefix>_op_<suffix> ↔ <prefix>_<suffix>.
     * Try removing "_op_" from the middle. */
    const char *op_pos = strstr(graph_name, "_op_");
    if (op_pos) {
        char dropped[MAX_FAM_NAME_LEN];
        size_t prefix_len = (size_t)(op_pos - graph_name);
        if (prefix_len < sizeof(dropped) - 1) {
            memcpy(dropped, graph_name, prefix_len);
            dropped[prefix_len] = '_';
            strncpy(dropped + prefix_len + 1, op_pos + 4, sizeof(dropped) - prefix_len - 2);
            dropped[sizeof(dropped) - 1] = '\0';
            idx = family_idx_of(dropped);
            if (idx >= 0) return idx;
        }
    }
    return -1;
}

/* Lightweight train.txt parser: each example is a // comment line
 * followed by a @graph block, separated by blank lines. We capture
 * the "// " line as the prompt and the "@graph <name>" line as the family. */
static int load_train(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return 0;
    }
    char line[512];
    char pending_prompt[MAX_PROMPT_LEN] = {0};
    int  loaded = 0;
    int  unmapped = 0;
    while (fgets(line, sizeof(line), fp) && loaded < MAX_TRAIN_PROMPTS) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';

        if (strncmp(line, "// ", 3) == 0) {
            strncpy(pending_prompt, line + 3, MAX_PROMPT_LEN - 1);
            pending_prompt[MAX_PROMPT_LEN - 1] = '\0';
            continue;
        }
        if (strncmp(line, "@graph ", 7) == 0 && pending_prompt[0]) {
            char gname[MAX_FAM_NAME_LEN];
            const char *src = line + 7;
            while (*src == ' ') src++;
            int o = 0;
            while (*src && *src != ' ' && *src != '\n' && *src != '\t' && o < MAX_FAM_NAME_LEN - 1) {
                gname[o++] = *src++;
            }
            gname[o] = '\0';
            int fam_idx = graph_name_to_family(gname);
            if (fam_idx >= 0) {
                strncpy(g_train[loaded].prompt, pending_prompt, MAX_PROMPT_LEN);
                g_train[loaded].family_idx = fam_idx;
                loaded++;
            } else {
                unmapped++;
            }
            pending_prompt[0] = '\0';
        }
    }
    fclose(fp);
    g_n_train = loaded;
    fprintf(stderr, "[tfidf] loaded %d (prompt, family) training pairs (%d unmapped graph names skipped)\n",
            loaded, unmapped);
    return loaded;
}

/* ---------- TF-IDF feature extraction ---------- */
static void build_vocabulary(void) {
    /* Pass 1: collect unique words, mark doc-frequency.
     * "Document" = a single training prompt. */
    for (int t = 0; t < g_n_train; t++) {
        char tokens[64][MAX_WORD_LEN];
        int n_tok = tokenize(g_train[t].prompt, tokens, 64);
        /* Track which words appeared in this doc to avoid double-counting df. */
        int seen[MAX_VOCAB] = {0};
        for (int i = 0; i < n_tok; i++) {
            int idx = vocab_lookup_or_add(tokens[i], 1);
            if (idx >= 0 && !seen[idx]) {
                g_vocab[idx].doc_freq++;
                seen[idx] = 1;
            }
        }
    }
    fprintf(stderr, "[tfidf] vocabulary: %d unique words across %d training prompts\n",
            g_vocab_size, g_n_train);
}

/* Compute TF-IDF feature vector for one prompt (length g_vocab_size). */
static void prompt_tfidf(const char *prompt, double *out) {
    for (int i = 0; i < g_vocab_size; i++) out[i] = 0.0;
    char tokens[64][MAX_WORD_LEN];
    int n_tok = tokenize(prompt, tokens, 64);
    if (n_tok == 0) return;

    /* TF (raw count, no normalization yet). */
    for (int i = 0; i < n_tok; i++) {
        int idx = vocab_lookup(tokens[i]);
        if (idx >= 0) out[idx] += 1.0;
    }
    /* TF normalisation (relative frequency in doc) and IDF weighting. */
    double inv_n = 1.0 / (double)n_tok;
    for (int i = 0; i < g_vocab_size; i++) {
        if (out[i] == 0.0) continue;
        double tf = out[i] * inv_n;
        double idf = log(((double)g_n_train + 1.0) / ((double)g_vocab[i].doc_freq + 1.0)) + 1.0;
        out[i] = tf * idf;
    }
    /* L2 normalize. */
    double sum_sq = 0.0;
    for (int i = 0; i < g_vocab_size; i++) sum_sq += out[i] * out[i];
    if (sum_sq > 0.0) {
        double inv = 1.0 / sqrt(sum_sq);
        for (int i = 0; i < g_vocab_size; i++) out[i] *= inv;
    }
}

/* ---------- Family centroids ---------- */
static double g_centroid[MAX_FAMILIES][MAX_VOCAB];
static int    g_fam_count[MAX_FAMILIES];

static void build_centroids(void) {
    for (int f = 0; f < N_FAMILIES; f++) {
        for (int v = 0; v < g_vocab_size; v++) g_centroid[f][v] = 0.0;
        g_fam_count[f] = 0;
    }
    double *feat = (double *)calloc((size_t)g_vocab_size, sizeof(double));
    for (int t = 0; t < g_n_train; t++) {
        prompt_tfidf(g_train[t].prompt, feat);
        int f = g_train[t].family_idx;
        for (int v = 0; v < g_vocab_size; v++) g_centroid[f][v] += feat[v];
        g_fam_count[f]++;
    }
    free(feat);
    /* Average and L2-renormalise each centroid. */
    for (int f = 0; f < N_FAMILIES; f++) {
        if (g_fam_count[f] == 0) continue;
        double inv_n = 1.0 / (double)g_fam_count[f];
        double sum_sq = 0.0;
        for (int v = 0; v < g_vocab_size; v++) {
            g_centroid[f][v] *= inv_n;
            sum_sq += g_centroid[f][v] * g_centroid[f][v];
        }
        if (sum_sq > 0.0) {
            double inv = 1.0 / sqrt(sum_sq);
            for (int v = 0; v < g_vocab_size; v++) g_centroid[f][v] *= inv;
        }
    }
    /* Telemetry. */
    fprintf(stderr, "[tfidf] family centroids built. Per-family training counts:\n");
    for (int f = 0; f < N_FAMILIES; f++) {
        fprintf(stderr, "  %-20s  n=%d\n", FAMILY_NAMES[f], g_fam_count[f]);
    }
}

/* ---------- Top-1 prediction by cosine similarity ---------- */
static int predict_family(const char *prompt) {
    double *feat = (double *)calloc((size_t)g_vocab_size, sizeof(double));
    prompt_tfidf(prompt, feat);
    int best_idx = -1;
    double best_score = -1e30;
    for (int f = 0; f < N_FAMILIES; f++) {
        if (g_fam_count[f] == 0) continue;
        double dot = 0.0;
        for (int v = 0; v < g_vocab_size; v++) dot += feat[v] * g_centroid[f][v];
        if (dot > best_score) { best_score = dot; best_idx = f; }
    }
    free(feat);
    return best_idx;
}

/* ---------- Test prompt loader ---------- */
typedef struct {
    char prompt[MAX_PROMPT_LEN];
    char reference[MAX_FAM_NAME_LEN];
} TestItem;

static int load_test(const char *path, TestItem *items, int max) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return 0;
    }
    int n = 0;
    char line[512];
    char pending_ref[MAX_FAM_NAME_LEN] = {0};
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
            strncpy(items[n].reference, pending_ref, sizeof(items[n].reference) - 1);
            items[n].reference[sizeof(items[n].reference) - 1] = '\0';
            pending_ref[0] = '\0';
            n++;
        }
    }
    fclose(fp);
    return n;
}

/* ---------- Main ---------- */
int main(int argc, char **argv) {
    const char *test_path  = (argc > 1) ? argv[1] : "pipeline_corpus_held_out.txt";
    const char *train_path = (argc > 2) ? argv[2] : "pipeline_corpus_train.txt";

    printf("================================================================\n");
    printf("  MicroGPT-C — Phase 3a-lite TF-IDF Centroid Classifier\n");
    printf("================================================================\n\n");

    if (!load_train(train_path)) return 1;
    build_vocabulary();
    build_centroids();

    TestItem items[MAX_TEST_PROMPTS];
    int n_test = load_test(test_path, items, MAX_TEST_PROMPTS);
    if (n_test == 0) {
        fprintf(stderr, "ERROR: no test prompts loaded from %s\n", test_path);
        return 1;
    }
    printf("\nLoaded %d test prompts from %s\n", n_test, test_path);
    printf("Trained on %d (prompt, family) pairs from %s\n", g_n_train, train_path);
    printf("Vocabulary: %d unique words\n\n", g_vocab_size);

    int n_correct = 0;
    int n_known   = 0;
    printf("%-3s %-20s %-20s %-6s  %s\n", "#", "REFERENCE", "PREDICTED", "MATCH", "PROMPT (truncated)");
    printf("------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < n_test; i++) {
        int ref_idx = family_idx_of(items[i].reference);
        if (ref_idx >= 0) n_known++;
        int pred_idx = predict_family(items[i].prompt);
        const char *pred_name = (pred_idx >= 0) ? FAMILY_NAMES[pred_idx] : "(none)";
        const char *match_str;
        if (ref_idx >= 0 && pred_idx == ref_idx) { match_str = "EXACT"; n_correct++; }
        else match_str = "no";
        char trunc[80];
        strncpy(trunc, items[i].prompt, sizeof(trunc) - 1);
        trunc[sizeof(trunc) - 1] = '\0';
        if (strlen(trunc) > 60) { trunc[57] = '.'; trunc[58] = '.'; trunc[59] = '.'; trunc[60] = '\0'; }
        printf("%-3d %-20s %-20s %-6s  %s\n", i + 1, items[i].reference, pred_name, match_str, trunc);
    }

    printf("\n================================================================\n");
    printf("  HEADLINE\n");
    printf("================================================================\n");
    printf("Top-1 EXACT family match: %d/%d (%.0f%%)\n",
           n_correct, n_known,
           n_known > 0 ? 100.0 * n_correct / n_known : 0.0);
    return 0;
}
