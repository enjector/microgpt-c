/*
 * tools/e15_eval.c — Experiment E15 held-out evaluation driver.
 *
 * Loads up to 4 checkpoints (monolithic + 3 OPA organelles), runs each
 * on a held-out TSV of "<state>\t<solution>\n" positions, and reports
 * solve rate / solution length / latency per architecture.
 *
 * The deterministic verifier (klotski OR puzzle15) replays the
 * generated move sequence on the encoded state and checks goal
 * reachability.  Solve = goal reached within the move budget.
 *
 * Compile-time-macro caveat (E09 §3.4): like e15_train, this binary
 * uses the compile-time engine macros (N_EMBD, N_HEAD, ...) for the
 * matmul shapes.  Two binaries (e15_mono_eval at 900K config,
 * e15_opa_eval at 300K config) are built — one per checkpoint family.
 *
 * The "monolithic" arm runs greedy decode from "<state>|" on the
 * monolithic checkpoint, then verifies.
 *
 * The "OPA" arm runs greedy decode on each of the 3 organelles
 * (planner / player / judge — each role-tagged at training) and picks
 * the longest valid prefix from any candidate.  This is the
 * "coordination is the intelligence" mechanism: 3 specialists +
 * deterministic verifier filter → robust answer.
 *
 * Usage:
 *   e15_mono_eval --task klotski|puzzle15
 *                 --ckpt checkpoints/<task>_mono.ckpt
 *                 --vocab checkpoints/<task>_mono.vocab
 *                 --heldout build/<task>_heldout_large.tsv
 *                 --max-moves 200
 *                 --out results/<task>_mono_eval.csv
 *
 *   e15_opa_eval --task klotski|puzzle15
 *                --planner-ckpt ... --player-ckpt ... --judge-ckpt ...
 *                --planner-vocab ... --player-vocab ... --judge-vocab ...
 *                --heldout ... --max-moves 200
 *                --out ...
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include "microgpt.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ========================================================================
 * Verifiers — replay a move string on a state and check goal-reachability.
 * Mirrors the logic in tools/{klotski_a_star.c, puzzle15_a_star.c} but
 * standalone (no popen) so we can verify in-process.
 * ======================================================================== */

/* --- Klotski (4 rows × 5 cols, '.', 'A'..'G').
 *     Move tokens: "<id><dir>" e.g. "AD", separated by ','. */
#define KLOTSKI_ROWS 4
#define KLOTSKI_COLS 5
#define KLOTSKI_CELLS (KLOTSKI_ROWS * KLOTSKI_COLS)

typedef char KlotskiBoard[KLOTSKI_CELLS];

static int klotski_cell(int r, int c) { return r * KLOTSKI_COLS + c; }

static void klotski_dir_delta(char d, int *dr, int *dc) {
    *dr = 0; *dc = 0;
    if (d == 'U') *dr = -1;
    else if (d == 'D') *dr = 1;
    else if (d == 'L') *dc = -1;
    else if (d == 'R') *dc = 1;
}

static int klotski_can_move(const KlotskiBoard b, char block_id, char direction) {
    int dr, dc;
    klotski_dir_delta(direction, &dr, &dc);
    if (dr == 0 && dc == 0) return 0;
    for (int r = 0; r < KLOTSKI_ROWS; r++)
        for (int c = 0; c < KLOTSKI_COLS; c++) {
            if (b[klotski_cell(r, c)] != block_id) continue;
            int nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= KLOTSKI_ROWS || nc < 0 || nc >= KLOTSKI_COLS) return 0;
            char dst = b[klotski_cell(nr, nc)];
            if (dst != '.' && dst != block_id) return 0;
        }
    return 1;
}

static void klotski_apply_move(KlotskiBoard b, char block_id, char direction) {
    int dr, dc;
    klotski_dir_delta(direction, &dr, &dc);
    int positions[KLOTSKI_CELLS][2];
    int n = 0;
    for (int r = 0; r < KLOTSKI_ROWS; r++)
        for (int c = 0; c < KLOTSKI_COLS; c++)
            if (b[klotski_cell(r, c)] == block_id) {
                positions[n][0] = r;
                positions[n][1] = c;
                n++;
            }
    for (int i = 0; i < n; i++)
        b[klotski_cell(positions[i][0], positions[i][1])] = '.';
    for (int i = 0; i < n; i++)
        b[klotski_cell(positions[i][0] + dr, positions[i][1] + dc)] = block_id;
}

static int klotski_is_goal(const KlotskiBoard b) {
    return b[klotski_cell(KLOTSKI_ROWS - 2, 1)] == 'A' &&
           b[klotski_cell(KLOTSKI_ROWS - 2, 2)] == 'A' &&
           b[klotski_cell(KLOTSKI_ROWS - 1, 1)] == 'A' &&
           b[klotski_cell(KLOTSKI_ROWS - 1, 2)] == 'A';
}

/* Returns the length of the *longest valid prefix* (in moves) for which
 *   the puzzle reaches the goal, or 0 if the goal is never reached.
 * Out param *out_moves_consumed: total moves successfully applied
 *   before either reaching goal or hitting an invalid move. */
static int klotski_verify(const char *state, const char *solution,
                          int max_moves, int *out_moves_consumed) {
    KlotskiBoard b;
    if (strlen(state) != KLOTSKI_CELLS) return 0;
    memcpy(b, state, KLOTSKI_CELLS);
    if (klotski_is_goal(b)) {
        if (out_moves_consumed) *out_moves_consumed = 0;
        return 1;
    }
    int moves = 0;
    const char *p = solution;
    while (*p && moves < max_moves) {
        /* Each token = "<id><dir>", optionally followed by ','. */
        char id = *p++;
        if (!*p) break;
        char dir = *p++;
        /* Allow valid block ids and directions only. */
        if (!(id >= 'A' && id <= 'G') ||
            !(dir == 'U' || dir == 'D' || dir == 'L' || dir == 'R')) {
            break;
        }
        if (!klotski_can_move(b, id, dir)) {
            break;
        }
        klotski_apply_move(b, id, dir);
        moves++;
        if (klotski_is_goal(b)) {
            if (out_moves_consumed) *out_moves_consumed = moves;
            return 1;
        }
        if (*p == ',') p++;
        else if (*p == '\0') break;
        else break;  /* malformed separator */
    }
    if (out_moves_consumed) *out_moves_consumed = moves;
    return 0;
}

/* --- 15-puzzle (4×4 hex board, blank = '0').
 *     Move letters: U/D/L/R indicating the direction the BLANK moves. */
#define P15_N 4
#define P15_NN 16

typedef struct {
    unsigned char tile[P15_NN];
    int blank;
} P15Board;

static int p15_parse(const char *state, P15Board *b) {
    if (strlen(state) != P15_NN) return 0;
    b->blank = -1;
    for (int i = 0; i < P15_NN; i++) {
        char c = state[i];
        int v = -1;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') v = 10 + (c - 'A');
        else return 0;
        b->tile[i] = (unsigned char)v;
        if (v == 0) b->blank = i;
    }
    return b->blank >= 0;
}

static int p15_apply(P15Board *b, char dir) {
    int dr = 0, dc = 0;
    switch (dir) {
        case 'U': dr = -1; break;
        case 'D': dr = 1; break;
        case 'L': dc = -1; break;
        case 'R': dc = 1; break;
        default: return 0;
    }
    int r = b->blank / P15_N, c = b->blank % P15_N;
    int nr = r + dr, nc = c + dc;
    if (nr < 0 || nr >= P15_N || nc < 0 || nc >= P15_N) return 0;
    int ni = nr * P15_N + nc;
    b->tile[b->blank] = b->tile[ni];
    b->tile[ni] = 0;
    b->blank = ni;
    return 1;
}

static int p15_is_goal(const P15Board *b) {
    for (int i = 0; i < P15_NN - 1; i++)
        if (b->tile[i] != (unsigned char)(i + 1)) return 0;
    return b->tile[P15_NN - 1] == 0;
}

static int p15_verify(const char *state, const char *solution,
                      int max_moves, int *out_moves_consumed) {
    P15Board b;
    if (!p15_parse(state, &b)) return 0;
    if (p15_is_goal(&b)) {
        if (out_moves_consumed) *out_moves_consumed = 0;
        return 1;
    }
    int moves = 0;
    for (const char *p = solution; *p && moves < max_moves; p++) {
        if (*p == ',' || *p == ' ') continue;
        if (!p15_apply(&b, *p)) break;
        moves++;
        if (p15_is_goal(&b)) {
            if (out_moves_consumed) *out_moves_consumed = moves;
            return 1;
        }
    }
    if (out_moves_consumed) *out_moves_consumed = moves;
    return 0;
}

/* ========================================================================
 * Vocab persistence — re-load the vocab written by e15_train --vocab-save.
 * ======================================================================== */
static int load_vocab(const char *path, Vocab *vocab) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t vs = 0;
    if (fscanf(f, "%zu\n", &vs) != 1) { fclose(f); return -1; }
    vocab->vocab_size = vs;
    vocab->chars = (char *)malloc(vs + 1);
    if (!vocab->chars) { fclose(f); return -1; }
    for (size_t i = 0; i < vs; i++) {
        int c1 = fgetc(f);
        if (c1 == EOF) { fclose(f); free(vocab->chars); return -1; }
        if (c1 == '\\') {
            int c2 = fgetc(f);
            if (c2 == 'n') vocab->chars[i] = '\n';
            else if (c2 == '\\') vocab->chars[i] = '\\';
            else vocab->chars[i] = (char)c2;
            /* consume trailing '\n' */
            fgetc(f);
        } else {
            vocab->chars[i] = (char)c1;
            fgetc(f);
        }
    }
    vocab->chars[vs] = '\0';
    vocab->bos_id = (vs > 0) ? (vs - 1) : 0;
    fclose(f);
    return 0;
}

/* ========================================================================
 * Generation — greedy decode from a prefix until a terminator or budget.
 * ======================================================================== */
typedef struct {
    Model *model;
    const Vocab *vocab;
    int n_layer;
    int block_size;
} ModelHandle;

/* Greedy-decode up to `max_gen` tokens starting from the prefix.  Writes
 * the generated characters (after the '|' separator if one is found in
 * the prefix) into `out` (nul-terminated).  Returns the number of chars
 * generated.  Terminates on:
 *   - emitting BOS (EOS sentinel)
 *   - hitting newline
 *   - max_gen reached
 *   - block_size budget reached */
static size_t greedy_generate(ModelHandle *h, const char *prefix,
                              char *out, size_t out_cap, size_t max_gen) {
    const Vocab *vocab = h->vocab;
    Model *model = h->model;
    int nl = h->n_layer;
    int bs = h->block_size;

    /* Allocate per-layer KV caches. */
    const MicrogptConfig *cfg = model_config(model);
    scalar_t **keys = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    scalar_t **values = (scalar_t **)calloc((size_t)nl, sizeof(scalar_t *));
    size_t *cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    for (int L = 0; L < nl; L++) {
        keys[L]   = kv_cache_alloc(cfg);
        values[L] = kv_cache_alloc(cfg);
    }
    scalar_t *logits = (scalar_t *)calloc(vocab->vocab_size,
                                          sizeof(scalar_t));

    /* Tokenize prefix. */
    size_t prefix_buf[2048];
    size_t prefix_len = strlen(prefix);
    size_t n_tok = tokenize(prefix, prefix_len, vocab, prefix_buf,
                            sizeof(prefix_buf) / sizeof(prefix_buf[0]));
    /* tokenize prepends BOS and appends BOS (EOS sentinel).  We want
     * to feed the BOS + prefix chars only, stripping the trailing
     * BOS so the next prediction comes after the prefix's last char. */
    if (n_tok > 0 && prefix_buf[n_tok - 1] == vocab->bos_id) n_tok--;

    /* Prime KV by running forward on each prefix token. */
    size_t pos = 0;
    for (size_t i = 0; i < n_tok && (int)pos < bs; i++) {
        forward_inference(model, prefix_buf[i], pos, keys, values,
                          cache_len, logits);
        pos++;
    }

    /* Greedy decode. */
    size_t out_n = 0;
    for (size_t step = 0; step < max_gen && (int)pos < bs - 1; step++) {
        /* Greedy: argmax. */
        size_t best = 0;
        scalar_t bv = logits[0];
        for (size_t k = 1; k < vocab->vocab_size; k++) {
            if (logits[k] > bv) { bv = logits[k]; best = k; }
        }
        if (best == vocab->bos_id) break;  /* EOS sentinel */
        if (best >= vocab->vocab_size) break;
        char c = vocab->chars[best];
        if (c == '\n') break;
        if (out_n + 1 >= out_cap) break;
        out[out_n++] = c;
        forward_inference(model, best, pos, keys, values, cache_len, logits);
        pos++;
    }
    out[out_n] = '\0';

    free(logits);
    for (int L = 0; L < nl; L++) {
        kv_cache_free(keys[L]);
        kv_cache_free(values[L]);
    }
    free(keys); free(values); free(cache_len);
    return out_n;
}

/* ========================================================================
 * Held-out reading
 * ======================================================================== */
typedef struct {
    char **states;
    char **solutions;
    size_t n;
} Heldout;

static int load_heldout(const char *path, Heldout *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t cap = 64;
    out->states = (char **)malloc(cap * sizeof(char *));
    out->solutions = (char **)malloc(cap * sizeof(char *));
    out->n = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        size_t llen = strlen(line);
        while (llen > 0 && (line[llen - 1] == '\n' || line[llen - 1] == '\r'))
            line[--llen] = '\0';
        if (llen == 0) continue;
        char *tab = strchr(line, '\t');
        if (out->n >= cap) {
            cap *= 2;
            out->states = (char **)realloc(out->states, cap * sizeof(char *));
            out->solutions = (char **)realloc(out->solutions, cap * sizeof(char *));
        }
        if (tab) {
            *tab = '\0';
            out->states[out->n] = strdup(line);
            out->solutions[out->n] = strdup(tab + 1);
        } else {
            out->states[out->n] = strdup(line);
            out->solutions[out->n] = strdup("");
        }
        out->n++;
    }
    fclose(f);
    return 0;
}

static void free_heldout(Heldout *h) {
    for (size_t i = 0; i < h->n; i++) {
        free(h->states[i]);
        free(h->solutions[i]);
    }
    free(h->states);
    free(h->solutions);
    h->n = 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */
typedef int (*verify_fn)(const char *state, const char *solution,
                         int max_moves, int *out_moves);

typedef struct {
    Model *model;
    Vocab vocab;
    char role_tag[8];  /* "", "P:", "M:", "J:" */
} OrgHandle;

static int load_one(const char *ckpt_path, const char *vocab_path,
                    const char *role_tag, OrgHandle *out) {
    if (load_vocab(vocab_path, &out->vocab) != 0) {
        fprintf(stderr, "e15_eval: cannot read vocab '%s'\n", vocab_path);
        return -1;
    }
    MicrogptConfig cfg = microgpt_default_config();
    cfg.n_embd = N_EMBD; cfg.n_head = N_HEAD; cfg.n_layer = N_LAYER;
    cfg.block_size = BLOCK_SIZE; cfg.mlp_dim = MLP_DIM;
    cfg.batch_size = 1;
    /* Allocate transient Adam buffers (we don't use them). */
    /* checkpoint_load reads vocab_size from the file's header; we pass
     * our own vocab_size and check it matches. */
    /* Allocate temp m/v buffers sized for the model's nparams.  Since
     * we don't know nparams up front, use a generous overestimate
     * based on cfg. */
    /* Use the formula: roughly 2*vs*ne + bs*ne + nlayer*(4*ne^2 + 2*ne*md). */
    size_t ne = (size_t)cfg.n_embd, md = (size_t)cfg.mlp_dim,
           blk = (size_t)cfg.block_size, vs = out->vocab.vocab_size;
    size_t np_est = 2 * vs * ne + blk * ne +
                    (size_t)cfg.n_layer * (4 * ne * ne + 2 * ne * md);
    scalar_t *m_buf = (scalar_t *)calloc(np_est + 16, sizeof(scalar_t));
    scalar_t *v_buf = (scalar_t *)calloc(np_est + 16, sizeof(scalar_t));
    int step_out = 0;
    out->model = checkpoint_load(ckpt_path, out->vocab.vocab_size, &cfg,
                                 m_buf, v_buf, &step_out);
    free(m_buf); free(v_buf);
    if (!out->model) {
        fprintf(stderr, "e15_eval: checkpoint_load('%s') failed\n", ckpt_path);
        free(out->vocab.chars);
        return -1;
    }
    strncpy(out->role_tag, role_tag, sizeof(out->role_tag) - 1);
    out->role_tag[sizeof(out->role_tag) - 1] = '\0';
    fprintf(stdout, "[e15_eval] loaded ckpt=%s vocab=%zu params=%zu role='%s' step=%d\n",
            ckpt_path, out->vocab.vocab_size, model_num_params(out->model),
            out->role_tag, step_out);
    return 0;
}

int main(int argc, char **argv) {
    const char *task = NULL;
    const char *heldout_path = NULL;
    const char *out_csv = NULL;
    int max_moves = 200;
    int n_limit = 0;  /* 0 = all */
    int verbose = 0;
    const char *mono_ckpt = NULL, *mono_vocab = NULL;
    const char *planner_ckpt = NULL, *planner_vocab = NULL;
    const char *player_ckpt = NULL, *player_vocab = NULL;
    const char *judge_ckpt = NULL, *judge_vocab = NULL;
    int mode_opa = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--task") && i + 1 < argc) task = argv[++i];
        else if (!strcmp(argv[i], "--heldout") && i + 1 < argc) heldout_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_csv = argv[++i];
        else if (!strcmp(argv[i], "--max-moves") && i + 1 < argc) max_moves = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) n_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt") && i + 1 < argc) mono_ckpt = argv[++i];
        else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) mono_vocab = argv[++i];
        else if (!strcmp(argv[i], "--planner-ckpt") && i + 1 < argc) { planner_ckpt = argv[++i]; mode_opa = 1; }
        else if (!strcmp(argv[i], "--planner-vocab") && i + 1 < argc) planner_vocab = argv[++i];
        else if (!strcmp(argv[i], "--player-ckpt") && i + 1 < argc) { player_ckpt = argv[++i]; mode_opa = 1; }
        else if (!strcmp(argv[i], "--player-vocab") && i + 1 < argc) player_vocab = argv[++i];
        else if (!strcmp(argv[i], "--judge-ckpt") && i + 1 < argc) { judge_ckpt = argv[++i]; mode_opa = 1; }
        else if (!strcmp(argv[i], "--judge-vocab") && i + 1 < argc) judge_vocab = argv[++i];
        else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "e15_eval: unknown arg '%s'\n", argv[i]); return 2; }
    }
    if (!task || !heldout_path) {
        fprintf(stderr,
            "usage: e15_eval --task klotski|puzzle15 --heldout <tsv> "
            "[--out <csv>] [--max-moves N] [--limit N] [-v]\n"
            "  monolithic: --ckpt <ckpt> --vocab <vocab>\n"
            "  OPA:        --planner-ckpt --planner-vocab "
            "--player-ckpt --player-vocab --judge-ckpt --judge-vocab\n");
        return 2;
    }
    verify_fn verify;
    if (!strcmp(task, "klotski")) verify = klotski_verify;
    else if (!strcmp(task, "puzzle15")) verify = p15_verify;
    else { fprintf(stderr, "e15_eval: unknown task '%s'\n", task); return 2; }

    Heldout heldout = {0};
    if (load_heldout(heldout_path, &heldout) != 0) {
        fprintf(stderr, "e15_eval: cannot read heldout '%s'\n", heldout_path);
        return 1;
    }
    fprintf(stdout, "[e15_eval] task=%s heldout=%s positions=%zu max_moves=%d\n",
            task, heldout_path, heldout.n, max_moves);

    /* Load organelles. */
    OrgHandle mono = {0}, planner = {0}, player = {0}, judge = {0};
    OrgHandle *orgs[4]; int n_orgs = 0;
    if (mode_opa) {
        if (!planner_ckpt || !planner_vocab ||
            !player_ckpt  || !player_vocab  ||
            !judge_ckpt   || !judge_vocab) {
            fprintf(stderr, "e15_eval: OPA mode requires all 3 organelles\n");
            free_heldout(&heldout);
            return 2;
        }
        if (load_one(planner_ckpt, planner_vocab, "P:", &planner) != 0 ||
            load_one(player_ckpt,  player_vocab,  "M:", &player)  != 0 ||
            load_one(judge_ckpt,   judge_vocab,   "J:", &judge)   != 0) {
            free_heldout(&heldout);
            return 1;
        }
        orgs[n_orgs++] = &planner;
        orgs[n_orgs++] = &player;
        orgs[n_orgs++] = &judge;
    } else {
        if (!mono_ckpt || !mono_vocab) {
            fprintf(stderr, "e15_eval: monolithic mode requires --ckpt + --vocab\n");
            free_heldout(&heldout);
            return 2;
        }
        if (load_one(mono_ckpt, mono_vocab, "", &mono) != 0) {
            free_heldout(&heldout);
            return 1;
        }
        orgs[n_orgs++] = &mono;
    }

    FILE *csv = NULL;
    if (out_csv) {
        csv = fopen(out_csv, "wb");
        if (!csv) {
            fprintf(stderr, "e15_eval: cannot open out '%s'\n", out_csv);
        } else {
            fprintf(csv, "idx,state,oracle_sol,arch,gen_sol,solved,moves_consumed,latency_ms\n");
        }
    }

    /* Evaluate. */
    size_t n_eval = heldout.n;
    if (n_limit > 0 && (size_t)n_limit < n_eval) n_eval = (size_t)n_limit;

    size_t solved = 0;
    long total_moves = 0;
    long max_moves_solved = 0;
    double total_latency_ms = 0;
    double max_latency_ms = 0;
    double latencies[heldout.n + 1];
    size_t lat_n = 0;

    char gen[2048];
    char prefix[2048];

    clock_t eval_start = clock();
    for (size_t i = 0; i < n_eval; i++) {
        const char *state = heldout.states[i];
        const char *oracle = heldout.solutions[i];

        clock_t t0 = clock();
        int best_solved = 0;
        int best_moves = -1;
        char best_gen[2048] = "";
        const char *best_arch = mode_opa ? "OPA" : "MONO";

        for (int k = 0; k < n_orgs; k++) {
            OrgHandle *org = orgs[k];
            ModelHandle h;
            h.model = org->model;
            h.vocab = &org->vocab;
            h.n_layer = N_LAYER;
            h.block_size = BLOCK_SIZE;
            /* Build prefix: "<role_tag><state>|" */
            snprintf(prefix, sizeof(prefix), "%s%s|", org->role_tag, state);
            size_t max_gen_tokens = BLOCK_SIZE - strlen(prefix) - 4;
            if (max_gen_tokens > 800) max_gen_tokens = 800;
            greedy_generate(&h, prefix, gen, sizeof(gen), max_gen_tokens);
            int mc = 0;
            int ok = verify(state, gen, max_moves, &mc);
            /* Pick the FIRST organelle whose output reaches the goal —
             * if none, accumulate the longest valid prefix as a tie-breaker. */
            if (ok && !best_solved) {
                best_solved = 1;
                best_moves = mc;
                strncpy(best_gen, gen, sizeof(best_gen) - 1);
                best_gen[sizeof(best_gen) - 1] = '\0';
                /* For OPA, keep the planner's first valid solution; for
                 * monolithic this is the only one anyway. */
                break;
            }
            if (!best_solved && mc > best_moves) {
                best_moves = mc;
                strncpy(best_gen, gen, sizeof(best_gen) - 1);
                best_gen[sizeof(best_gen) - 1] = '\0';
            }
        }

        clock_t t1 = clock();
        double lat_ms = 1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC;
        latencies[lat_n++] = lat_ms;
        total_latency_ms += lat_ms;
        if (lat_ms > max_latency_ms) max_latency_ms = lat_ms;

        if (best_solved) {
            solved++;
            total_moves += best_moves;
            if (best_moves > max_moves_solved) max_moves_solved = best_moves;
        }

        if (verbose && i < 5) {
            fprintf(stdout,
                "[%zu] state=%s oracle=%s gen=%s solved=%d moves=%d lat=%.1fms\n",
                i, state, oracle, best_gen, best_solved, best_moves, lat_ms);
        }
        if (csv) {
            fprintf(csv, "%zu,%s,%s,%s,%s,%d,%d,%.1f\n",
                i, state, oracle, best_arch, best_gen, best_solved,
                best_moves, lat_ms);
        }
    }
    double eval_sec = (double)(clock() - eval_start) / CLOCKS_PER_SEC;

    /* Compute p99 latency. */
    /* Simple insertion sort (n is small). */
    for (size_t i = 1; i < lat_n; i++) {
        double key = latencies[i];
        size_t j = i;
        while (j > 0 && latencies[j - 1] > key) {
            latencies[j] = latencies[j - 1];
            j--;
        }
        latencies[j] = key;
    }
    double p99 = lat_n > 0 ? latencies[(size_t)(0.99 * (lat_n - 1))] : 0;
    double mean_lat = lat_n > 0 ? total_latency_ms / (double)lat_n : 0;
    double mean_moves = solved > 0 ? (double)total_moves / (double)solved : 0;
    double solve_rate = 100.0 * (double)solved / (double)n_eval;

    fprintf(stdout,
        "\n[e15_eval] RESULT task=%s arch=%s heldout=%zu/%zu\n"
        "  solved          = %zu (%.1f%%)\n"
        "  mean_moves(solved) = %.1f\n"
        "  max_moves_solved   = %ld\n"
        "  mean_latency       = %.2f ms\n"
        "  p99_latency        = %.2f ms\n"
        "  total_eval_wall    = %.2f s\n",
        task, mode_opa ? "OPA" : "MONO", n_eval, heldout.n,
        solved, solve_rate, mean_moves, max_moves_solved,
        mean_lat, p99, eval_sec);

    if (csv) fclose(csv);

    /* Cleanup. */
    if (mono.model) { model_free(mono.model); free(mono.vocab.chars); }
    if (planner.model) { model_free(planner.model); free(planner.vocab.chars); }
    if (player.model)  { model_free(player.model);  free(player.vocab.chars); }
    if (judge.model)   { model_free(judge.model);   free(judge.vocab.chars); }
    free_heldout(&heldout);
    return 0;
}
