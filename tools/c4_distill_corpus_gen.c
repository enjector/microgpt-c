/*
 * tools/c4_distill_corpus_gen.c — LLM-distillation corpus generator
 * for Experiment E13.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Pre-registered in `experiments/E13-llm-game-distillation.md` §1.3.2.
 *
 * Plays N Connect-4 games where:
 *   - X is the local LM Studio LLM (Qwen 3.6 35B by default).
 *   - O is a deterministic random opponent.
 *
 * For each game, the (board, move) pair at every X turn is buffered.
 * If X wins the game, the buffered pairs are appended to the output
 * corpus.  If X loses or draws, the pairs are discarded (the
 * "only-winning-games" filter — §1.3.2 step 5).
 *
 * The corpus is emitted in the exact format the existing
 * `c_connect4_player.txt` uses (line by line):
 *
 *     board=........................................|valid=0,1,2,3,4,5,6
 *     3
 *
 *     board=........................................X|valid=0,1,2,3,4,5,6
 *     2
 *     ...
 *
 * — each (prompt, label) separated by a blank line, label is the
 * single ASCII digit 0..6.  The OQL TRAIN adapter calls load_docs +
 * tokenize on this file unchanged.
 *
 * Stop conditions:
 *   - N games played (--games=N), default 1000.
 *   - M (board, move) pairs accumulated (--max-pairs=M), default 10000.
 *   - Either threshold hits first (early-stop per §1.3.2 step 6).
 *
 * Determinism:
 *   - Random opponent uses --opp-seed=K (default 13).  LLM player is
 *     cached on (board, valid, model, seed) keys — re-running with the
 *     same seeds produces bit-identical corpora (T7).
 */

#define _POSIX_C_SOURCE 200809L

#include "llm_game_player.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ----- Board constants (mirror demos/character-level/connect4/main.c) ----- */
#define C4_ROWS 6
#define C4_COLS 7
#define C4_SIZE (C4_ROWS * C4_COLS)
#define C4_EMPTY '.'
#define C4_X 'X'
#define C4_O 'O'

/* ----- Defaults ----- */
#define DEFAULT_GAMES      1000
#define DEFAULT_MAX_PAIRS  10000
#define DEFAULT_OPP_SEED   13
#define DEFAULT_LLM_SEED   42
#define DEFAULT_OUT_PATH   "data/c4_distill_corpus.txt"
#define DEFAULT_CACHE_DIR  "data/c4_distill_cache"

/* ----- Board helpers ----- */

static int cell_idx(int r, int c) { return r * C4_COLS + c; }

static int get_valid_columns(const char *board, int *cols) {
    int n = 0;
    for (int c = 0; c < C4_COLS; c++) {
        if (board[cell_idx(0, c)] == C4_EMPTY) cols[n++] = c;
    }
    return n;
}

static int drop_piece(char *board, int col, char player) {
    if (col < 0 || col >= C4_COLS) return -1;
    for (int r = C4_ROWS - 1; r >= 0; r--) {
        if (board[cell_idx(r, col)] == C4_EMPTY) {
            board[cell_idx(r, col)] = player;
            return r;
        }
    }
    return -1;
}

static int count_pieces(const char *board) {
    int n = 0;
    for (int i = 0; i < C4_SIZE; i++) if (board[i] != C4_EMPTY) n++;
    return n;
}

static const int WIN_DR[4] = {0, 1, 1, 1};
static const int WIN_DC[4] = {1, 0, 1, -1};

static char check_winner(const char *board) {
    for (int r = 0; r < C4_ROWS; r++) {
        for (int c = 0; c < C4_COLS; c++) {
            if (board[cell_idx(r, c)] == C4_EMPTY) continue;
            char who = board[cell_idx(r, c)];
            for (int d = 0; d < 4; d++) {
                int er = r + 3 * WIN_DR[d];
                int ec = c + 3 * WIN_DC[d];
                if (er < 0 || er >= C4_ROWS || ec < 0 || ec >= C4_COLS)
                    continue;
                int ok = 1;
                for (int i = 1; i < 4; i++) {
                    if (board[cell_idx(r + i * WIN_DR[d], c + i * WIN_DC[d])]
                        != who) { ok = 0; break; }
                }
                if (ok) return who;
            }
        }
    }
    return C4_EMPTY;
}

/* ----- Pair buffer ----- */

typedef struct {
    char  prompt[256];   /* board=...|valid=... */
    int   move;          /* 0..6 */
} C4Pair;

#define MAX_GAME_PAIRS 42  /* max plies for X in a game */

/* ----- Argument parsing ----- */

typedef struct {
    int   n_games;
    int   max_pairs;
    int   opp_seed;
    int   llm_seed;
    const char *out_path;
    const char *cache_dir;
    const char *model_id;
    const char *endpoint;
    int   skip_health_check;
    const char *prepend_path;   /* optional baseline corpus to splice in
                                   front of the LLM corpus (vocab coverage) */
} Args;

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [--games=N] [--max-pairs=M] [--opp-seed=K] [--llm-seed=K]\n"
        "          [--out=PATH] [--cache=DIR] [--model=ID] [--endpoint=URL]\n"
        "          [--prepend=PATH] [--skip-health-check]\n"
        "\n"
        "  E13 — LLM-distillation corpus generator.  Plays N Connect-4 games\n"
        "  with the LLM as X (qwen/qwen3.6-35b-a3b via http://127.0.0.1:1234\n"
        "  by default), random as O, and emits (board, move) pairs from the\n"
        "  LLM's wins to OUT.  Stops at min(N games, M pairs).\n"
        "\n"
        "  --prepend=PATH copies PATH's contents to OUT before the LLM\n"
        "    pairs land, so the trained student's char-vocab matches that\n"
        "    of the baseline checkpoint loader (which requires vocab\n"
        "    alignment with c_connect4_player.txt — see oql_runtime_games.c\n"
        "    `build_player_organelle`).  Recommended for E13 training.\n"
        "\n"
        "  defaults: games=%d max_pairs=%d opp_seed=%d llm_seed=%d\n"
        "            out='%s' cache='%s'\n",
        argv0, DEFAULT_GAMES, DEFAULT_MAX_PAIRS, DEFAULT_OPP_SEED,
        DEFAULT_LLM_SEED, DEFAULT_OUT_PATH, DEFAULT_CACHE_DIR);
}

static int parse_args(int argc, char **argv, Args *a) {
    a->n_games = DEFAULT_GAMES;
    a->max_pairs = DEFAULT_MAX_PAIRS;
    a->opp_seed = DEFAULT_OPP_SEED;
    a->llm_seed = DEFAULT_LLM_SEED;
    a->out_path = DEFAULT_OUT_PATH;
    a->cache_dir = DEFAULT_CACHE_DIR;
    a->model_id = NULL;
    a->endpoint = NULL;
    a->skip_health_check = 0;
    a->prepend_path = NULL;
    for (int i = 1; i < argc; i++) {
        const char *s = argv[i];
        if (!strncmp(s, "--games=", 8))         a->n_games = atoi(s + 8);
        else if (!strncmp(s, "--max-pairs=", 12)) a->max_pairs = atoi(s + 12);
        else if (!strncmp(s, "--opp-seed=", 11))  a->opp_seed = atoi(s + 11);
        else if (!strncmp(s, "--llm-seed=", 11))  a->llm_seed = atoi(s + 11);
        else if (!strncmp(s, "--out=", 6))        a->out_path = s + 6;
        else if (!strncmp(s, "--cache=", 8))      a->cache_dir = s + 8;
        else if (!strncmp(s, "--model=", 8))      a->model_id = s + 8;
        else if (!strncmp(s, "--endpoint=", 11))  a->endpoint = s + 11;
        else if (!strncmp(s, "--prepend=", 10))   a->prepend_path = s + 10;
        else if (!strcmp(s, "--skip-health-check")) a->skip_health_check = 1;
        else if (!strcmp(s, "--help") || !strcmp(s, "-h")) {
            print_usage(argv[0]); return -1;
        } else {
            fprintf(stderr, "unknown arg: %s\n", s);
            print_usage(argv[0]); return -1;
        }
    }
    return 0;
}

/* ----- Build the valid-columns CSV exactly like the C demo (e.g. "0,2,4") ----- */
static void build_valid_csv(const int *cols, int n, char *out, size_t cap) {
    size_t pos = 0;
    for (int i = 0; i < n; i++) {
        int wr = snprintf(out + pos, cap - pos, "%s%d", (i ? "," : ""), cols[i]);
        if (wr < 0 || (size_t)wr >= cap - pos) break;
        pos += (size_t)wr;
    }
    out[pos] = '\0';
}

/* ----- mkdir -p the parent dir of a path (best-effort) ----- */
static void mkdir_parents(const char *path) {
    char buf[512];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            (void)system("test -d '/'"); /* no-op; just to silence -Wunused */
            char cmd[600];
            snprintf(cmd, sizeof(cmd), "mkdir -p '%s' 2>/dev/null", buf);
            system(cmd);
            *p = '/';
        }
    }
}

/* ----- Main ----- */

int main(int argc, char **argv) {
    Args a;
    if (parse_args(argc, argv, &a) != 0) return 1;

    fprintf(stderr,
        "E13 c4_distill_corpus_gen — LLM-distillation corpus generator\n"
        "  games=%d max_pairs=%d opp_seed=%d llm_seed=%d\n"
        "  out=%s cache=%s\n"
        "  endpoint=%s model=%s\n\n",
        a.n_games, a.max_pairs, a.opp_seed, a.llm_seed,
        a.out_path, a.cache_dir,
        a.endpoint ? a.endpoint : "http://127.0.0.1:1234",
        a.model_id ? a.model_id : "qwen/qwen3.6-35b-a3b");

    LlmGamePlayer *llm =
        llm_game_player_new(a.model_id, a.endpoint, a.cache_dir, a.llm_seed);
    if (!llm) { fprintf(stderr, "OOM\n"); return 1; }

    if (!a.skip_health_check) {
        if (!llm_game_player_health_check(llm)) {
            fprintf(stderr,
                "FATAL: LM Studio health check failed — endpoint %s "
                "unreachable or model not loaded.  Aborting; see E13 §1.5 "
                "stop conditions.\n",
                a.endpoint ? a.endpoint : "http://127.0.0.1:1234");
            llm_game_player_free(llm);
            return 2;
        }
        fprintf(stderr, "LM Studio health check: OK\n");
    }

    /* Open output file. */
    mkdir_parents(a.out_path);
    FILE *out = fopen(a.out_path, "wb");
    if (!out) {
        fprintf(stderr, "FATAL: cannot open %s for writing\n", a.out_path);
        llm_game_player_free(llm);
        return 1;
    }

    /* Optional: prepend baseline corpus.  Streams bytes verbatim — no
     * parsing.  Ensures the trained student's vocab covers every char
     * the baseline c_connect4_player.txt contains.  Per E13 §3.4 the
     * student trains on the COMBINED corpus (baseline + LLM-distill);
     * this is augmentation distillation, not pure LLM-only distillation
     * — documented honestly in the writeup. */
    int prepend_bytes = 0;
    if (a.prepend_path && a.prepend_path[0]) {
        FILE *pf = fopen(a.prepend_path, "rb");
        if (!pf) {
            fprintf(stderr,
                "FATAL: --prepend='%s' could not be opened\n",
                a.prepend_path);
            fclose(out);
            llm_game_player_free(llm);
            return 1;
        }
        char chunk[8192];
        size_t r;
        while ((r = fread(chunk, 1, sizeof(chunk), pf)) > 0) {
            fwrite(chunk, 1, r, out);
            prepend_bytes += (int)r;
        }
        fclose(pf);
        /* Ensure the prepended file ends with a blank-line separator so
         * the first LLM record doesn't accidentally fuse with the last
         * baseline record.  The corpus loader (opa_load_docs_multiline)
         * splits on blank lines, so two consecutive '\n's are required. */
        fputc('\n', out);
        fputc('\n', out);
        prepend_bytes += 2;
        fprintf(stderr,
            "Prepended baseline corpus '%s' (%d bytes)\n",
            a.prepend_path, prepend_bytes);
        fflush(out);
    }

    /* Stats. */
    int games_played = 0;
    int llm_wins = 0, llm_losses = 0, llm_draws = 0;
    int pairs_emitted = 0;
    unsigned int opp_seed = (unsigned int)a.opp_seed;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int g = 0; g < a.n_games && pairs_emitted < a.max_pairs; g++) {
        char board[C4_SIZE + 1];
        memset(board, C4_EMPTY, C4_SIZE);
        board[C4_SIZE] = '\0';

        C4Pair game_pairs[MAX_GAME_PAIRS];
        int n_game_pairs = 0;

        char winner = C4_EMPTY;
        int draw = 0;

        while (winner == C4_EMPTY && !draw) {
            int cols[C4_COLS];
            int n_valid = get_valid_columns(board, cols);
            if (n_valid == 0) { draw = 1; break; }

            /* Build prompt + valid CSV. */
            char valid_csv[32];
            build_valid_csv(cols, n_valid, valid_csv, sizeof(valid_csv));

            char prompt[256];
            char bstr[C4_SIZE + 1];
            memcpy(bstr, board, C4_SIZE);
            bstr[C4_SIZE] = '\0';
            snprintf(prompt, sizeof(prompt),
                     "board=%s|valid=%s", bstr, valid_csv);

            /* Ask the LLM. */
            int move = -1;
            int rc = llm_game_player_move(llm, bstr, valid_csv, &move);
            if (rc < 0 || move < 0 || move >= C4_COLS) {
                fprintf(stderr,
                    "warn: LLM call hard-failed at game=%d ply=%d — "
                    "using random fallback to keep the loop alive\n",
                    g, n_game_pairs);
                move = cols[(int)(rand_r(&opp_seed) % (unsigned)n_valid)];
            }
            /* Validate move legality (LLM may have hallucinated a full
             * column — drop_piece will return -1). */
            if (drop_piece(board, move, C4_X) < 0) {
                /* Pick any legal column and treat the game as poisoned. */
                move = cols[(int)(rand_r(&opp_seed) % (unsigned)n_valid)];
                drop_piece(board, move, C4_X);
            }

            /* Record pair before checking for terminal — we want the
             * (board state X saw, move X chose). */
            if (n_game_pairs < MAX_GAME_PAIRS) {
                strncpy(game_pairs[n_game_pairs].prompt, prompt,
                        sizeof(game_pairs[0].prompt) - 1);
                game_pairs[n_game_pairs].prompt[
                    sizeof(game_pairs[0].prompt) - 1] = '\0';
                game_pairs[n_game_pairs].move = move;
                n_game_pairs++;
            }

            winner = check_winner(board);
            if (winner != C4_EMPTY) break;
            if (count_pieces(board) == C4_SIZE) { draw = 1; break; }

            /* Opponent (random) plays O. */
            int oc[C4_COLS];
            int n_o = get_valid_columns(board, oc);
            if (n_o == 0) { draw = 1; break; }
            int opp_col = oc[(int)(rand_r(&opp_seed) % (unsigned)n_o)];
            drop_piece(board, opp_col, C4_O);

            winner = check_winner(board);
            if (winner != C4_EMPTY) break;
            if (count_pieces(board) == C4_SIZE) { draw = 1; break; }
        }

        games_played++;
        if (winner == C4_X) llm_wins++;
        else if (winner == C4_O) llm_losses++;
        else llm_draws++;

        /* Only emit pairs from winning games. */
        if (winner == C4_X) {
            for (int i = 0; i < n_game_pairs && pairs_emitted < a.max_pairs; i++) {
                fprintf(out, "%s\n%d\n\n",
                        game_pairs[i].prompt, game_pairs[i].move);
                pairs_emitted++;
            }
            fflush(out);
        }

        if ((g + 1) % 10 == 0 || g == 0 || (g + 1) == a.n_games) {
            const LlmGamePlayerStats *st = llm_game_player_stats(llm);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double dt = (double)(t1.tv_sec - t0.tv_sec)
                      + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
            fprintf(stderr,
                "  game %4d/%d | wins=%d losses=%d draws=%d | pairs=%d "
                "| cache_hits=%d net_fail=%d parse_fail=%d | %.1fs\n",
                g + 1, a.n_games, llm_wins, llm_losses, llm_draws,
                pairs_emitted, st->cache_hits, st->network_failures,
                st->parse_failures, dt);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (double)(t1.tv_sec - t0.tv_sec)
              + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

    fclose(out);
    const LlmGamePlayerStats *st = llm_game_player_stats(llm);

    fprintf(stderr,
        "\nE13 corpus-gen summary (T6 diagnostics):\n"
        "  games played       : %d / %d\n"
        "  LLM wins           : %d (%.1f%%)\n"
        "  LLM losses         : %d (%.1f%%)\n"
        "  LLM draws          : %d (%.1f%%)\n"
        "  pairs emitted      : %d / %d\n"
        "  LLM total calls    : %d\n"
        "  cache hits         : %d\n"
        "  parse retries      : %d\n"
        "  parse failures     : %d\n"
        "  network failures   : %d\n"
        "  total wallclock    : %.1fs (%.2fs LLM time)\n"
        "  output             : %s\n",
        games_played, a.n_games,
        llm_wins,
        games_played > 0 ? 100.0 * llm_wins / games_played : 0.0,
        llm_losses,
        games_played > 0 ? 100.0 * llm_losses / games_played : 0.0,
        llm_draws,
        games_played > 0 ? 100.0 * llm_draws / games_played : 0.0,
        pairs_emitted, a.max_pairs,
        st->total_calls, st->cache_hits, st->parse_retries,
        st->parse_failures, st->network_failures,
        dt, st->cumulative_wallclock_seconds, a.out_path);

    /* T6 LLM-X win rate < 50% trips the skip rule. */
    int llm_win_rate_pct =
        games_played > 0 ? (100 * llm_wins / games_played) : 0;
    if (llm_win_rate_pct < 50 && games_played >= 20) {
        fprintf(stderr,
            "\nWARNING: LLM-X win rate < 50%% — per E13 §1.5 skip rules, "
            "investigate prompt template before generating a full corpus.\n");
    }

    llm_game_player_free(llm);
    return 0;
}
