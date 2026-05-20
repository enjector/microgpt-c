/*
 * oql_runtime_games.c  —  Connect-4 game-loop harness for OQL `RUN ... ON game_loop`
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * E09 Phase 3 — implements oql_run_game_loop().
 *
 * The harness deliberately mirrors the deterministic-board logic from
 * demos/character-level/connect4/main.c (~150 LOC lifted; opponent driver,
 * win check, drop) and adds the OQL-specific dispatch: walk pipeline →
 * INPUT_BEHAVIOUR → forward_inference → OUTPUT_BEHAVIOUR → VALIDATE_BEHAVIOUR
 * → FALLBACK_BEHAVIOUR.
 *
 * If an organelle's checkpoint isn't present (or fails to load), the harness
 * degrades to "random vs random" so the wiring still executes end-to-end and
 * a 50%-ish win rate gets reported.  This keeps E09 T1 (RUN completes) testable
 * without requiring a real trained checkpoint — see Section 3.4 of
 * experiments/E09-oql-runtime-wiring.md for the pre-registered semantics.
 */

#include "oql_runtime_games.h"
#include "microgpt_oql.h"
#include "microgpt.h"
#include "microgpt_organelle.h"
#include "microgpt_vm.h"
#include "microgpt_vm_natives.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
/* Windows doesn't have rand_r; use a tiny LCG instead. */
static int rand_r(unsigned int *seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffffu;
    return (int)*seed;
}
#endif

/* ============================================================
 *  Board primitives (lifted from demos/character-level/connect4/main.c)
 * ============================================================ */

int oql_c4_cell_idx(int row, int col) { return row * OQL_C4_COLS + col; }

int oql_c4_column_legal(const char *board, int col) {
    if (!board || col < 0 || col >= OQL_C4_COLS) return 0;
    return board[oql_c4_cell_idx(0, col)] == OQL_C4_EMPTY;
}

int oql_c4_drop(char *board, int col, char player) {
    if (!board || col < 0 || col >= OQL_C4_COLS) return -1;
    for (int r = OQL_C4_ROWS - 1; r >= 0; r--) {
        if (board[oql_c4_cell_idx(r, col)] == OQL_C4_EMPTY) {
            board[oql_c4_cell_idx(r, col)] = player;
            return r;
        }
    }
    return -1;
}

static const int WIN_DR[4] = {0, 1, 1, 1};
static const int WIN_DC[4] = {1, 0, 1, -1};

char oql_c4_winner(const char *board) {
    if (!board) return OQL_C4_EMPTY;
    for (int r = 0; r < OQL_C4_ROWS; r++) {
        for (int c = 0; c < OQL_C4_COLS; c++) {
            char p = board[oql_c4_cell_idx(r, c)];
            if (p == OQL_C4_EMPTY) continue;
            for (int d = 0; d < 4; d++) {
                int er = r + 3 * WIN_DR[d];
                int ec = c + 3 * WIN_DC[d];
                if (er < 0 || er >= OQL_C4_ROWS || ec < 0 || ec >= OQL_C4_COLS) continue;
                int match = 1;
                for (int i = 1; i < 4; i++) {
                    if (board[oql_c4_cell_idx(r + i * WIN_DR[d], c + i * WIN_DC[d])] != p) {
                        match = 0; break;
                    }
                }
                if (match) return p;
            }
        }
    }
    return OQL_C4_EMPTY;
}

int oql_c4_is_full(const char *board) {
    if (!board) return 1;
    for (int i = 0; i < OQL_C4_SIZE; i++) {
        if (board[i] == OQL_C4_EMPTY) return 0;
    }
    return 1;
}

int oql_c4_random_move(const char *board, unsigned int *seed) {
    int legal[OQL_C4_COLS];
    int n = 0;
    for (int c = 0; c < OQL_C4_COLS; c++) {
        if (oql_c4_column_legal(board, c)) legal[n++] = c;
    }
    if (n == 0) return -1;
    return legal[rand_r(seed) % n];
}

/* ============================================================
 *  Behaviour dispatch — lazy compile + run on each call
 *
 *  Each BEHAVIOUR body is compiled once (memoised in OqlBehaviourEntry.module),
 *  then a fresh runtime is created per dispatch with the natives bound to
 *  the supplied ctx.  Returns the numeric eval() result, or `default_val`
 *  if the behaviour isn't found / compile fails / run fails.
 * ============================================================ */

static double oql_dispatch_behaviour(OqlRuntime *rt, const char *behaviour_name,
                                     vm_natives_ctx *ctx, double default_val,
                                     FILE *out) {
    if (!rt || !behaviour_name || !behaviour_name[0]) return default_val;
    OqlBehaviourEntry *b = oql_runtime_find_behaviour(rt, behaviour_name);
    if (!b || !b->vm_body) return default_val;

    /* Lazy compile. */
    if (!b->module) {
        vm_module *mod = NULL;
        vm_result r = vm_module_compile(NULL, b->vm_body, &mod);
        if (r != VM_OK || !mod) {
            if (out) fprintf(out,
                "behaviour '%s': compile failed (vm_result=%d)\n",
                behaviour_name, (int)r);
            return default_val;
        }
        b->module = mod;
    }

    vm_natives_register_c4(ctx);
    vm_module_runtime *runtime = vm_module_runtime_create((vm_module *)b->module);
    if (!runtime) return default_val;
    vm_module_runtime_set_call_ext_method_callback(runtime, vm_natives_dispatch);
    vm_module_runtime_clear(runtime);

    vm_function *fn = vm_module_fetch_function((vm_module *)b->module, "eval");
    if (!fn) {
        vm_module_runtime_dispose(runtime);
        return default_val;
    }
    vm_result rr = vm_module_runtime_run(runtime, fn);
    if (rr != VM_OK) {
        vm_module_runtime_dispose(runtime);
        return default_val;
    }
    vm_variable *ret = NULL;
    vm_module_runtime_stack_pop(runtime, &ret);
    double value = default_val;
    if (ret) {
        value = (ret->type_class == ptcBOOLEAN)
            ? (ret->value.boolean ? 1.0 : 0.0)
            : ret->value.number;
        vm_variable_dispose(ret);
    }
    vm_module_runtime_dispose(runtime);
    return value;
}

/* ============================================================
 *  E11: lazy Organelle-wrapper construction.
 *
 *  The OQL runtime's load step returns a bare `Model *` (it doesn't
 *  reconstruct the Vocab — checkpoints store vocab SIZE, not the
 *  char list).  organelle_generate_ensemble() needs an `Organelle`
 *  with a populated Vocab, so we rebuild it from the training corpus.
 *
 *  Convention for E11:  the runtime looks for a corpus file next to
 *  CWD with one of the well-known names listed in OQL_C4_CORPUS_CANDIDATES.
 *  The c_connect4_demo target's POST_BUILD copies
 *  c_connect4_player.txt + c_connect4_planner.txt next to the binary,
 *  so when researchers run `./build/oql_c4 run ../experiments/connect4.oql`
 *  from build/ the player corpus is already there.
 *
 *  When the corpus is absent / vocab mismatches the checkpoint header,
 *  the wrapper is NOT built and the runtime falls back to the legacy
 *  uniform-mask proposer (51% baseline).  No silent garbage output:
 *  organelle_generate_ensemble guards against vocab mismatch via the
 *  checkpoint header's vocab_size field.
 * ============================================================ */

static const char *OQL_C4_CORPUS_CANDIDATES[] = {
    "c_connect4_player.txt",
    "../demos/character-level/connect4/c_connect4_player.txt",
    "demos/character-level/connect4/c_connect4_player.txt",
    NULL
};

/* Peek the checkpoint header for the trained vocab_size.  Format:
 *   [int step][size_t vocab][weights]  (same path as
 *   oql_runtime_load_organelle uses to size the Adam scratch buffers). */
static int oql_peek_ckpt_vocab(const char *path, size_t *vocab_out) {
    if (!path || !vocab_out) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int step = 0; size_t vocab = 0;
    int ok = (fread(&step, sizeof(int), 1, f) == 1) &&
             (fread(&vocab, sizeof(size_t), 1, f) == 1);
    fclose(f);
    if (!ok) return -1;
    *vocab_out = vocab;
    return 0;
}

/* Returns a heap-allocated Organelle on success (caller must dispose), or
 * NULL if the corpus is missing / vocab can't be built / the vocab size
 * doesn't match the checkpoint's recorded vocab_size. */
static Organelle *oql_build_player_organelle(Model *model,
                                             const MicrogptConfig *cfg,
                                             const char *ckpt_path,
                                             FILE *out) {
    if (!model || !cfg) return NULL;

    /* Find a corpus file. */
    const char *path = NULL;
    for (int i = 0; OQL_C4_CORPUS_CANDIDATES[i]; i++) {
        FILE *f = fopen(OQL_C4_CORPUS_CANDIDATES[i], "r");
        if (f) { fclose(f); path = OQL_C4_CORPUS_CANDIDATES[i]; break; }
    }
    if (!path) {
        if (out) fprintf(out,
            "build_player_organelle: no corpus file found in cwd "
            "(tried c_connect4_player.txt and known fallbacks); "
            "model-driven proposal disabled, falling back to uniform-mask.\n");
        return NULL;
    }

    Organelle *org = (Organelle *)calloc(1, sizeof(Organelle));
    if (!org) return NULL;

    /* Use the SAME loader the C demo uses (opa_load_docs_multiline includes
     * the document-internal newline as a vocab char).  load_docs() strips
     * newlines and would build vocab_size = 25 vs the checkpoint's 26 —
     * matched the demo loader to keep the per-token map identical. */
    if (opa_load_docs_multiline(path, &org->docs, cfg->max_docs) < 0) {
        if (out) fprintf(out,
            "build_player_organelle: opa_load_docs_multiline('%s') failed\n", path);
        free(org);
        return NULL;
    }
    build_vocab(&org->docs, &org->vocab);

    /* Guard: the checkpoint's recorded vocab_size must match the corpus-
     * built vocab_size.  Mismatch means we're paired with the wrong
     * corpus, in which case feeding chars through the model produces
     * garbage. */
    size_t header_vocab = 0;
    if (ckpt_path && oql_peek_ckpt_vocab(ckpt_path, &header_vocab) == 0 &&
        org->vocab.vocab_size != header_vocab) {
        if (out) fprintf(out,
            "build_player_organelle: vocab mismatch — corpus '%s' builds "
            "vocab_size=%zu but ckpt '%s' records vocab_size=%zu; "
            "disabling model-driven proposal.\n",
            path, org->vocab.vocab_size, ckpt_path, header_vocab);
        free_docs(&org->docs);
        free(org->vocab.chars);
        free(org);
        return NULL;
    }

    org->model = model;       /* aliased, not owned — runtime owns the model */
    org->word_level = 0;      /* char-level, as the C demo */
    if (out) fprintf(out,
        "build_player_organelle: built from '%s' (vocab=%zu docs=%zu)\n",
        path, org->vocab.vocab_size, org->docs.num_docs);
    return org;
}

static void oql_free_player_organelle(Organelle *org) {
    if (!org) return;
    /* Don't free model — it's owned by OqlOrganelle::model and freed by
     * oql_runtime_dispose. */
    free_docs(&org->docs);
    free(org->vocab.chars);
    free(org);
}

/* ============================================================
 *  E11: model-driven proposal — full C-demo prompt protocol.
 *
 *  The state struct is passed opaquely into the natives module via
 *  vm_natives_ctx.propose_column_state.  The TS-side extern
 *  `c4_model_propose_column(temp_x100)` calls back into
 *  oql_propose_column_cb() which constructs the same
 *  `board=<42>|valid=<csv>` prompt that the C demo trains on
 *  (demos/character-level/connect4/c_connect4_player.txt has
 *  ~53k of these lines), runs organelle_generate_ensemble with the
 *  same 3-vote / temp=0.2 (configurable via temp_x100) settings, and
 *  parses the first output character as a digit 0..6.
 * ============================================================ */

typedef struct OqlProposeColumnState {
    const Organelle      *org;        /* loaded model + vocab */
    const MicrogptConfig *cfg;        /* the OQL runtime's cfg */
    const char           *board;      /* 42-char current board (NUL-terminated 43-byte buffer) */
} OqlProposeColumnState;

/* Build the player prompt as the C demo does:
 *   board=<42>|valid=<csv>
 *
 * Returns the byte count written to `out`, or -1 if the buffer is too small
 * or the board is malformed.  Mirrors lines 311-320 of the demo, sans the
 * `blocked=` field (the OQL runtime doesn't yet track per-turn blocked
 * history; closing that gap belongs to a separate ticket — see
 * E11-diagnosis.md §3.3). */
static int oql_c4_build_prompt(const char *board, char *out, size_t out_sz) {
    if (!board || !out || out_sz < 64) return -1;
    /* valid=<csv> */
    char valid[32]; size_t vpos = 0;
    int first = 1;
    for (int c = 0; c < OQL_C4_COLS; c++) {
        if (!oql_c4_column_legal(board, c)) continue;
        if (!first && vpos + 1 < sizeof(valid)) valid[vpos++] = ',';
        if (vpos + 1 < sizeof(valid)) {
            valid[vpos++] = (char)('0' + c);
            first = 0;
        }
    }
    if (vpos >= sizeof(valid)) return -1;
    valid[vpos] = '\0';
    int n = snprintf(out, out_sz, "board=%s|valid=%s", board, valid);
    return (n < 0 || (size_t)n >= out_sz) ? -1 : n;
}

#define OQL_C4_GEN_LEN     60   /* matches demo INF_GEN_LEN */
#define OQL_C4_ENSEMBLE     3   /* matches demo ENSEMBLE_VOTES */

static int oql_propose_column_cb(vm_natives_ctx *ctx, int temp_x100) {
    if (!ctx) return -1;
    OqlProposeColumnState *st = (OqlProposeColumnState *)ctx->propose_column_state;
    if (!st || !st->org || !st->cfg || !st->board) return -1;

    /* The board lives on ctx via the interned handle.  Prefer the staged
     * board in `st->board` since the runtime updates it directly (we don't
     * need a round-trip through the strings table for the extern call). */
    char prompt[256];
    if (oql_c4_build_prompt(st->board, prompt, sizeof(prompt)) < 0) return -1;

    /* Temperature: temp_x100 ∈ [1, 100] → scalar_t ∈ [0.01, 1.0]. */
    scalar_t temp = (scalar_t)((double)temp_x100 / 100.0);
    if (temp < (scalar_t)0.01) temp = (scalar_t)0.01;

    char out[OQL_C4_GEN_LEN + 1] = {0};
    scalar_t conf = 0;
    organelle_generate_ensemble(st->org, st->cfg, prompt,
                                out, OQL_C4_GEN_LEN,
                                OQL_C4_ENSEMBLE, temp, &conf);

    /* The C demo (main.c:338-340) reads the first character as the column
     * digit.  Match that exactly. */
    char c = out[0];
    if (c < '0' || c > '6') return -1;
    return (int)(c - '0');
}

/* ============================================================
 *  Legacy host-side proposer — retained as the fallback when the
 *  player organelle hasn't been wired with a corpus (e.g. tests that
 *  only construct an OqlRuntime without staging a vocab).  Picks
 *  uniformly from the legal mask, matching E09's pre-E11 behaviour.
 * ============================================================ */

static int oql_model_propose_column(const Model *model, unsigned int *seed,
                                    int legal_mask) {
    (void)model;
    int legal[OQL_C4_COLS];
    int n = 0;
    for (int c = 0; c < OQL_C4_COLS; c++) {
        if (legal_mask & (1 << c)) legal[n++] = c;
    }
    if (n == 0) return -1;
    return legal[rand_r(seed) % n];
}

/* ============================================================
 *  Audit-row record (E09 T8).  In-memory only for now; a future
 *  EVALUATE/REPORT integration would flush this to a JSON file.
 * ============================================================ */

typedef struct {
    int game;
    int move;
    int proposed_col;
    int validated;       /* 0/1 */
    int from_fallback;   /* 0/1 */
    int from_random;     /* 0/1 - dispatched via random opponent path */
    double dispatch_ms;
} OqlAuditRow;

#define OQL_AUDIT_MAX 16384
static OqlAuditRow g_audit_rows[OQL_AUDIT_MAX];
static int g_audit_count = 0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* ============================================================
 *  Main game-loop entry point
 * ============================================================ */

oql_status oql_run_game_loop(OqlRuntime *rt,
                             OqlPipeline *pipeline,
                             const char *opponent,
                             int games,
                             unsigned int seed,
                             const char *game,
                             FILE *out) {
    if (!rt || !pipeline) return OQL_ERR_RUNTIME;
    if (game && strcmp(game, "connect4") != 0) {
        if (out) fprintf(out,
            "RUN: only GAME=connect4 is wired in this commit (got '%s')\n", game);
        return OQL_ERR_NOT_IMPLEMENTED;
    }
    if (opponent && strcmp(opponent, "random") != 0) {
        if (out) fprintf(out,
            "RUN: only OPPONENT=random is wired in this commit (got '%s')\n",
            opponent);
        return OQL_ERR_NOT_IMPLEMENTED;
    }
    if (pipeline->n_calls == 0) {
        if (out) fprintf(out,
            "RUN: pipeline '%s' has 0 call stages — nothing to dispatch\n",
            pipeline->name);
        return OQL_ERR_RUNTIME;
    }

    g_audit_count = 0;
    rt->last_games_played = 0;
    rt->last_wins = rt->last_draws = rt->last_losses = 0;
    rt->last_p99_ms = 0.0;
    rt->last_total_seconds = 0.0;
    rt->last_audit_rows = 0;

    /* Pick the "player" organelle — first call-stage by convention.  In a
     * full Kanban pipeline this is the player; planner is upstream. */
    OqlOrganelle *player = oql_runtime_find_organelle(rt,
        pipeline->call_organelles[0]);
    if (!player) {
        if (out) fprintf(out,
            "RUN: pipeline first stage '%s' is not a registered organelle\n",
            pipeline->call_organelles[0]);
        return OQL_ERR_RUNTIME;
    }
    /* Lazy load — may return NULL if checkpoint missing. */
    Model *player_model = oql_runtime_load_organelle(rt, player, out);

    /* E11: build the Organelle wrapper (model + vocab from corpus) so the
     * c4_model_propose_column extern can run organelle_generate_ensemble
     * with the C-demo's prompt protocol.  NULL on corpus-absent / vocab-
     * mismatch — in that case the legacy uniform-mask proposer fires. */
    Organelle *player_org = NULL;
    if (player_model) {
        const MicrogptConfig *cfg = rt->cfg ? (const MicrogptConfig *)rt->cfg
                                            : model_config(player_model);
        player_org = oql_build_player_organelle(player_model, cfg,
                                                player->checkpoint_path, out);
    }
    const MicrogptConfig *runtime_cfg = rt->cfg
        ? (const MicrogptConfig *)rt->cfg
        : (player_model ? model_config(player_model) : NULL);

    /* E11: seed the global RNG (rand_u) so organelle_generate_ensemble's
     * sampling is reproducible per RUN.  The C demo does the same via
     * seed_rng(42) at startup; the OQL runtime mirrors that behaviour so
     * a `RUN ... SEED = 42` clause produces a deterministic trace.
     * The opponent's rand_r state is initialised from a second constant
     * to match the C demo's split (seed_rng(42) for model + 12345 for
     * opponent in demos/character-level/connect4/main.c lines 163, 226).  */
    seed_rng(seed);

    double run_start = now_ms();
    double *latencies = (double *)malloc(sizeof(double) * (size_t)games * 64);
    int n_latencies = 0;

    /* C-demo-compatible opponent seed.  Overridable via env-var so
     * researchers can sweep variance / build a Monte-Carlo distribution. */
    unsigned int g_seed = 12345;
    {
        const char *env_opp = getenv("OQL_C4_OPPONENT_SEED");
        if (env_opp) {
            char *end = NULL;
            unsigned long v = strtoul(env_opp, &end, 10);
            if (end && *end == '\0') g_seed = (unsigned int)v;
        }
    }
    (void)seed;  /* SEED clause now drives only the model RNG. */
    for (int gi = 0; gi < games; gi++) {
        char board[OQL_C4_SIZE + 1];
        memset(board, OQL_C4_EMPTY, OQL_C4_SIZE);
        board[OQL_C4_SIZE] = '\0';

        char outcome = OQL_C4_EMPTY;  /* X, O, or '.' for draw */
        int draw = 0;
        int moves = 0;

        vm_natives_ctx ctx;
        vm_natives_ctx_init(&ctx);

        /* E11: stage the propose-column callback for this game.  The state
         * is on the stack — its lifetime is the inner game loop. */
        OqlProposeColumnState propose_state = {0};
        propose_state.org = player_org;
        propose_state.cfg = runtime_cfg;
        propose_state.board = board;
        if (player_org && runtime_cfg) {
            ctx.propose_column = oql_propose_column_cb;
            ctx.propose_column_state = &propose_state;
        }

        while (outcome == OQL_C4_EMPTY && !draw) {
            /* ── Player's turn (X) ─────────────────────────────── */

            ctx.current_board_handle = vm_natives_str_intern(&ctx, board);
            /* Refresh the staged board pointer (memmove-safe — board lives
             * on the stack but the pointer never changes; this is just
             * defensive). */
            propose_state.board = board;

            double t0 = now_ms();

            /* INPUT_BEHAVIOUR: returns either:
             *   - a one-hot legal mask `1 << col` when the model proposed
             *     `col` and it's legal (E11 Pathway B fast-path), or
             *   - the full 7-bit legal mask (E09 baseline / fall-through).
             *
             * The runtime treats the result the same way either way — it
             * uniformly samples from the bits set in the returned mask
             * (oql_model_propose_column below).  One bit set ⇒ that bit
             * is chosen deterministically. */
            double mask_d = oql_dispatch_behaviour(rt, player->input_behaviour,
                                                   &ctx, 0.0, out);
            int legal_mask = (int)mask_d;
            if (legal_mask == 0) {
                /* No legal column — treat as draw if board full else random. */
                if (oql_c4_is_full(board)) { draw = 1; break; }
                int rc = oql_c4_random_move(board, &g_seed);
                if (rc < 0) { draw = 1; break; }
                oql_c4_drop(board, rc, OQL_C4_X);
                moves++;
                outcome = oql_c4_winner(board);
                if (outcome == OQL_C4_EMPTY && oql_c4_is_full(board)) draw = 1;
                if (outcome != OQL_C4_EMPTY || draw) break;
            } else {
                /* Model proposes a column. */
                int proposed = oql_model_propose_column(player_model, &g_seed,
                                                        legal_mask);

                /* OUTPUT_BEHAVIOUR: format proposed col as a single-digit token,
                 * parse it back via the c4_parse_token native.  Lets the OQL
                 * pipeline observe what the model output even when the model
                 * was offline. */
                if (proposed >= 0) {
                    char tok[2] = {(char)('0' + proposed), '\0'};
                    ctx.current_move_handle = vm_natives_str_intern(&ctx, tok);
                    double parsed_d = oql_dispatch_behaviour(rt,
                        player->output_behaviour, &ctx, (double)proposed, out);
                    int parsed = (int)parsed_d;
                    if (parsed >= 0 && parsed < OQL_C4_COLS) proposed = parsed;
                }

                /* VALIDATE_BEHAVIOUR. */
                int validated = 0;
                if (proposed >= 0) {
                    char tok[2] = {(char)('0' + proposed), '\0'};
                    ctx.current_move_handle = vm_natives_str_intern(&ctx, tok);
                    double v = oql_dispatch_behaviour(rt,
                        player->validate_behaviour, &ctx, -1.0, out);
                    if (v < 0.0) {
                        /* No VALIDATE_BEHAVIOUR registered — fall back to host check. */
                        validated = oql_c4_column_legal(board, proposed) ? 1 : 0;
                    } else {
                        validated = (v != 0.0) ? 1 : 0;
                    }
                }

                int from_fallback = 0;
                if (!validated) {
                    /* FALLBACK_BEHAVIOUR. */
                    double fb_d = oql_dispatch_behaviour(rt,
                        player->fallback_behaviour, &ctx, -1.0, out);
                    int fb = (int)fb_d;
                    if (fb >= 0 && fb < OQL_C4_COLS &&
                        oql_c4_column_legal(board, fb)) {
                        proposed = fb;
                        validated = 1;
                        from_fallback = 1;
                    } else {
                        /* Hard fallback: first legal column. */
                        for (int c = 0; c < OQL_C4_COLS; c++) {
                            if (oql_c4_column_legal(board, c)) {
                                proposed = c; validated = 1; from_fallback = 1;
                                break;
                            }
                        }
                    }
                }

                if (proposed < 0 || !validated) { draw = 1; break; }
                /* E11 T4 trace — log first N moves of first M games for the
                 * token-divergence measurement.  Activate with
                 *   OQL_TRACE_FIRST_N_MOVES=5 OQL_TRACE_GAMES=10
                 * and capture stderr.  Off by default. */
                {
                    const char *tn = getenv("OQL_TRACE_FIRST_N_MOVES");
                    const char *tg = getenv("OQL_TRACE_GAMES");
                    int max_moves = tn ? atoi(tn) : 0;
                    int max_games = tg ? atoi(tg) : 0;
                    if (max_moves > 0 && max_games > 0 &&
                        gi < max_games && moves < max_moves) {
                        fprintf(stderr, "OQL_TRACE game=%d move=%d col=%d "
                                        "from_fallback=%d\n",
                                gi, moves, proposed, from_fallback);
                    }
                }
                oql_c4_drop(board, proposed, OQL_C4_X);
                moves++;

                double t1 = now_ms();
                if (n_latencies < (int)((size_t)games * 64)) {
                    latencies[n_latencies++] = t1 - t0;
                }
                if (g_audit_count < OQL_AUDIT_MAX) {
                    OqlAuditRow *row = &g_audit_rows[g_audit_count++];
                    row->game = gi; row->move = moves;
                    row->proposed_col = proposed;
                    row->validated = validated;
                    row->from_fallback = from_fallback;
                    row->from_random = (player_model == NULL) ? 1 : 0;
                    row->dispatch_ms = t1 - t0;
                }

                outcome = oql_c4_winner(board);
                if (outcome != OQL_C4_EMPTY) break;
                if (oql_c4_is_full(board)) { draw = 1; break; }
            }

            /* ── Opponent's turn (O, random) ─────────────────── */
            int opp = oql_c4_random_move(board, &g_seed);
            if (opp < 0) { draw = 1; break; }
            oql_c4_drop(board, opp, OQL_C4_O);
            moves++;
            outcome = oql_c4_winner(board);
            if (outcome != OQL_C4_EMPTY) break;
            if (oql_c4_is_full(board)) { draw = 1; break; }
        }

        if (outcome == OQL_C4_X) rt->last_wins++;
        else if (outcome == OQL_C4_O) rt->last_losses++;
        else if (draw || outcome == OQL_C4_EMPTY) rt->last_draws++;

        vm_natives_ctx_dispose(&ctx);
    }
    rt->last_games_played = games;
    rt->last_audit_rows = g_audit_count;

    /* p99 latency. */
    if (n_latencies > 0) {
        qsort(latencies, (size_t)n_latencies, sizeof(double), cmp_double);
        int p99_idx = (n_latencies * 99) / 100;
        if (p99_idx >= n_latencies) p99_idx = n_latencies - 1;
        rt->last_p99_ms = latencies[p99_idx];
    }
    free(latencies);
    rt->last_total_seconds = (now_ms() - run_start) / 1000.0;

    if (out) {
        double win_rate = games > 0
            ? 100.0 * (double)rt->last_wins / (double)games : 0.0;
        fprintf(out,
            "RUN connect4: %d games | wins=%d draws=%d losses=%d "
            "(win_rate=%.1f%%) p99_latency=%.2fms audit_rows=%d "
            "model_loaded=%s%s total=%.2fs\n",
            games, rt->last_wins, rt->last_draws, rt->last_losses,
            win_rate, rt->last_p99_ms, rt->last_audit_rows,
            player_model ? "yes" : "NO (fell back to random vs random)",
            player_org ? " model_driven=yes" : " model_driven=no",
            rt->last_total_seconds);
    }
    oql_free_player_organelle(player_org);
    return OQL_OK;
}
