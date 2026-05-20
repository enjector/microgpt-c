/*
 * tools/llm_game_player.c — LLM-as-Connect-4-teacher bridge (Experiment E13).
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.  See header for the
 * pre-registration cross-reference and surface contract.
 *
 * Implementation strategy:
 *   - Transport is `curl` invoked via `popen("curl ... | cat")` — zero
 *     new build dependencies and curl is universally available on the
 *     macOS / Linux targets the rest of the project supports.
 *   - JSON request body is hand-assembled (a handful of fields, all
 *     well-known).  JSON response parsing is a one-shot string search
 *     for `"content":"…"`.  This is intentionally fragile and
 *     intentionally tiny — the only consumer is one OpenAI-compatible
 *     LM Studio endpoint, and the response shape is fixed by LM Studio's
 *     contract.  Any deviation surfaces immediately as a parse failure.
 *   - Cache is one tiny text file per (board, model, seed) hash.  Cache
 *     hit returns the cached single-digit move in microseconds; cache
 *     miss takes 0.5-5s depending on the LLM.
 *
 * Robustness:
 *   - Two retry budget per logical call: first try low-T, second try
 *     with an explicit "respond with exactly one digit 0-6" reminder.
 *   - Hard failure returns -1 — corpus generation drops the game.
 */

#define _POSIX_C_SOURCE 200809L

#include "llm_game_player.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* ============================================================
 * Configuration
 * ============================================================ */

#define LLMGP_DEFAULT_ENDPOINT "http://127.0.0.1:1234"
#define LLMGP_DEFAULT_MODEL    "qwen/qwen3.6-35b-a3b"
#define LLMGP_RESPONSE_CAP     8192   /* bytes of JSON we tolerate */
#define LLMGP_PROMPT_CAP       2048   /* bytes of board prompt */
#define LLMGP_BUF_CAP          16384  /* total scratch */

struct LlmGamePlayer {
    char model_id[128];
    char endpoint_url[256];
    char cache_dir[512];
    int  seed;
    LlmGamePlayerStats stats;
};

/* ============================================================
 * Tiny utilities
 * ============================================================ */

/* 64-bit FNV-1a — small, no allocations, deterministic.  Used for the
 * cache key only; collisions just trigger a recompute. */
static unsigned long long llmgp_fnv1a(const char *s, size_t len) {
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Append a JSON-escaped C string to a buffer.  Escapes only the
 * characters that LM Studio's content field could realistically
 * receive: `\`, `"`, `\n`, `\r`, `\t`.  Returns 0 on success, -1 on
 * overflow. */
static int llmgp_json_escape_append(char *dst, size_t cap, size_t *pos,
                                    const char *src) {
    for (; *src; src++) {
        unsigned char c = (unsigned char)*src;
        const char *esc = NULL;
        char buf[8] = {0};
        switch (c) {
        case '"':  esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\n': esc = "\\n";  break;
        case '\r': esc = "\\r";  break;
        case '\t': esc = "\\t";  break;
        default:
            if (c < 0x20) {
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                esc = buf;
            } else {
                /* fall-through: copy single byte */
            }
        }
        if (esc) {
            size_t n = strlen(esc);
            if (*pos + n >= cap) return -1;
            memcpy(dst + *pos, esc, n);
            *pos += n;
        } else {
            if (*pos + 1 >= cap) return -1;
            dst[(*pos)++] = (char)c;
        }
    }
    return 0;
}

/* ============================================================
 * Construction / teardown
 * ============================================================ */

LlmGamePlayer *llm_game_player_new(const char *model_id,
                                   const char *endpoint_url,
                                   const char *cache_dir,
                                   int seed) {
    LlmGamePlayer *p = (LlmGamePlayer *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    strncpy(p->model_id, model_id ? model_id : LLMGP_DEFAULT_MODEL,
            sizeof(p->model_id) - 1);
    strncpy(p->endpoint_url, endpoint_url ? endpoint_url : LLMGP_DEFAULT_ENDPOINT,
            sizeof(p->endpoint_url) - 1);
    if (cache_dir && cache_dir[0]) {
        strncpy(p->cache_dir, cache_dir, sizeof(p->cache_dir) - 1);
        /* Best-effort mkdir; if it already exists, fine. */
        mkdir(p->cache_dir, 0755);
    }
    p->seed = seed;
    return p;
}

void llm_game_player_free(LlmGamePlayer *p) {
    if (!p) return;
    free(p);
}

const LlmGamePlayerStats *llm_game_player_stats(const LlmGamePlayer *p) {
    return p ? &p->stats : NULL;
}

/* ============================================================
 * Curl transport
 * ============================================================ */

/* Drain a FILE* (typically from popen) into a heap buffer.  Caps at
 * LLMGP_RESPONSE_CAP so a runaway response can't OOM us.  Returns the
 * number of bytes read or -1 on error.
 *
 * Uses fgetc() in a loop — fread() against a popen pipe can return
 * 0 mid-stream on short reads on macOS, which silently truncated
 * responses during E13 smoke testing.  fgetc() loops on the same
 * underlying read but returns EOF only when the child closes the
 * pipe, which is what we want. */
static long llmgp_drain(FILE *fp, char *out, size_t cap) {
    size_t n = 0;
    int c;
    while (n + 1 < cap && (c = fgetc(fp)) != EOF) {
        out[n++] = (char)c;
    }
    out[n] = '\0';
    return (long)n;
}

/* Run a `curl` command and capture stdout. */
static long llmgp_run_curl(const char *cmd, char *out, size_t cap) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;
    long n = llmgp_drain(fp, out, cap);
    int rc = pclose(fp);
    if (rc != 0 && n == 0) return -1;
    return n;
}

/* ============================================================
 * Health check
 * ============================================================ */

int llm_game_player_health_check(const LlmGamePlayer *p) {
    if (!p) return 0;
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "curl -sS -m 10 %s/v1/models 2>/dev/null",
             p->endpoint_url);
    char buf[LLMGP_RESPONSE_CAP];
    long n = llmgp_run_curl(cmd, buf, sizeof(buf));
    if (n <= 0) {
        fprintf(stderr,
                "llm_game_player: health check failed — endpoint %s "
                "unreachable\n", p->endpoint_url);
        return 0;
    }
    if (!strstr(buf, p->model_id)) {
        fprintf(stderr,
                "llm_game_player: model '%s' not found in /v1/models "
                "response (first 200 chars: %.200s)\n",
                p->model_id, buf);
        return 0;
    }
    return 1;
}

/* ============================================================
 * Prompt rendering
 * ============================================================ */

/* Pretty-print the board into a 6x7 grid prefixed by column headers.
 * Returns 0 on success, -1 on overflow. */
static int llmgp_render_board_grid(const char *board, char *dst, size_t cap) {
    if (strlen(board) < 42) return -1;
    size_t pos = 0;
    int n = snprintf(dst + pos, cap - pos, "  0 1 2 3 4 5 6\n");
    if (n < 0 || (size_t)n >= cap - pos) return -1;
    pos += (size_t)n;
    for (int r = 0; r < 6; r++) {
        if (pos + 2 >= cap) return -1;
        dst[pos++] = ' '; dst[pos++] = ' ';
        for (int c = 0; c < 7; c++) {
            if (pos + 2 >= cap) return -1;
            dst[pos++] = board[r * 7 + c];
            dst[pos++] = ' ';
        }
        if (pos + 1 >= cap) return -1;
        dst[pos++] = '\n';
    }
    dst[pos] = '\0';
    return 0;
}

/* Build the system+user messages.  Iteration history:
 *   v1 (initial): "Output the column number 0-6"
 *     — LLM frequently responded with prose / multiple digits.
 *   v2 (after iteration on 20 fixed boards): explicit constraint set,
 *     ASCII board, "Reply with exactly one digit and nothing else."
 *   v3 (current): adds the legal-columns list so the LLM can't pick a
 *     full column.  Documented in experiments/E13 §3.1. */
static int llmgp_build_prompt(const char *board,
                              const char *valid_columns,
                              int with_strict_reminder,
                              char *user, size_t user_cap) {
    char grid[256];
    if (llmgp_render_board_grid(board, grid, sizeof(grid)) != 0) return -1;
    const char *strict =
        with_strict_reminder
        ? "\nIMPORTANT: respond with a single ASCII digit between 0 and 6 only. "
          "No explanation, no words, no punctuation."
        : "";
    int n = snprintf(
        user, user_cap,
        "You are playing Connect-4 as X against an opponent O. "
        "The goal is to get four X in a row horizontally, vertically, "
        "or diagonally before O does.\n\n"
        "Board (X = you, O = opponent, . = empty; row 0 is top):\n"
        "%s\n"
        "Legal columns this turn: %s\n\n"
        "Pick the best column for X to drop into. Respond with exactly "
        "one digit (the column index from the legal set above)."
        "%s",
        grid, valid_columns, strict);
    if (n < 0 || (size_t)n >= user_cap) return -1;
    return 0;
}

/* ============================================================
 * JSON request + response
 * ============================================================ */

/* Build the JSON request body for /v1/chat/completions.  `temperature`
 * is the OpenAI-compatible field (0.0..2.0); we send 0.0-0.2 for
 * determinism. */
static int llmgp_build_request_json(const LlmGamePlayer *p,
                                    const char *user_message,
                                    double temperature,
                                    int max_tokens,
                                    char *out, size_t cap) {
    size_t pos = 0;
    int n;
    /* `reasoning_effort=none` disables Qwen 3.6 35B's thinking tokens,
     * so the model emits the answer directly into `content` rather than
     * spending the budget on `reasoning_content`.  Measured during
     * prompt iteration (E13 §3.1): with reasoning enabled and a 16-token
     * cap, the model fills the budget with reasoning and `content`
     * comes back empty (finish_reason=length); with `reasoning_effort
     * =none` the model returns `"3"` directly in ~0.15s. */
    n = snprintf(out + pos, cap - pos,
                 "{\"model\":\"%s\",\"temperature\":%.2f,"
                 "\"reasoning_effort\":\"none\","
                 "\"max_tokens\":%d,\"messages\":[{\"role\":\"user\","
                 "\"content\":\"",
                 p->model_id, temperature, max_tokens);
    if (n < 0 || (size_t)n >= cap - pos) return -1;
    pos += (size_t)n;
    if (llmgp_json_escape_append(out, cap, &pos, user_message) != 0) return -1;
    n = snprintf(out + pos, cap - pos, "\"}]}");
    if (n < 0 || (size_t)n >= cap - pos) return -1;
    pos += (size_t)n;
    return 0;
}

/* Crude JSON content extractor — finds the first `"content":"…"`
 * substring and copies the unescaped value into `dst`.  Stops at the
 * first unescaped `"`.  Handles `\"`, `\\`, `\n`, `\r`, `\t`.  Anything
 * else is copied verbatim.  LM Studio pretty-prints with arbitrary
 * whitespace between `:` and the opening `"` of the value, so we
 * tolerate any amount of whitespace there. */
static int llmgp_extract_content(const char *body, char *dst, size_t cap) {
    /* Find `"content"` then skip past `:` + whitespace + `"`. */
    const char *p = strstr(body, "\"content\"");
    if (!p) return -1;
    p += strlen("\"content\"");
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (*p != ':') return -1;
    p++;
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (*p != '"') return -1;
    p++;
    size_t out = 0;
    while (*p && out + 1 < cap) {
        if (*p == '\\' && p[1]) {
            switch (p[1]) {
            case '"':  dst[out++] = '"';  p += 2; continue;
            case '\\': dst[out++] = '\\'; p += 2; continue;
            case 'n':  dst[out++] = '\n'; p += 2; continue;
            case 'r':  dst[out++] = '\r'; p += 2; continue;
            case 't':  dst[out++] = '\t'; p += 2; continue;
            case '/':  dst[out++] = '/';  p += 2; continue;
            case 'u': {
                /* Skip \uXXXX — we don't expect any in a digit reply,
                 * but for forward-compat we emit a '?' placeholder so
                 * the parser still terminates. */
                if (p[2] && p[3] && p[4] && p[5]) {
                    dst[out++] = '?';
                    p += 6;
                    continue;
                }
                return -1;
            }
            default:
                dst[out++] = p[1]; p += 2; continue;
            }
        }
        if (*p == '"') break;
        dst[out++] = *p++;
    }
    dst[out] = '\0';
    return 0;
}

/* Make one /v1/chat/completions call.  Returns 0 + populates `out_content`
 * on success; -1 on transport/parse failure. */
static int llmgp_call(const LlmGamePlayer *p,
                      const char *user_message,
                      double temperature,
                      char *out_content,
                      size_t out_cap,
                      double *out_seconds) {
    char body[LLMGP_BUF_CAP];
    /* max_tokens=32 is plenty for a single-digit reply but leaves a few
     * tokens of headroom for the rare model that prefixes whitespace. */
    if (llmgp_build_request_json(p, user_message, temperature, 32, body,
                                 sizeof(body)) != 0) {
        fprintf(stderr, "llm_game_player: request JSON overflow\n");
        return -1;
    }

    /* Write body to a tempfile so we can pass --data-binary @file
     * (avoiding shell-escaping all of the JSON).  Include a static
     * counter in the filename so back-to-back calls within the same
     * second don't collide. */
    static unsigned long s_call_seq = 0;
    s_call_seq++;
    char tmp_path[256];
    snprintf(tmp_path, sizeof(tmp_path), "/tmp/llmgp_req_%d_%lx_%lu.json",
             (int)getpid(), (unsigned long)time(NULL), s_call_seq);
    FILE *tf = fopen(tmp_path, "wb");
    if (!tf) {
        fprintf(stderr, "llm_game_player: cannot write %s\n", tmp_path);
        return -1;
    }
    fwrite(body, 1, strlen(body), tf);
    fclose(tf);

    /* Drop curl's stderr; write the body to a sibling tempfile so we can
     * read it back via plain fopen() without worrying about pipe buffer
     * boundaries — earlier versions used `popen("curl ... | cat")` and
     * silently truncated responses after the first ~200 bytes on macOS. */
    char resp_path[256];
    snprintf(resp_path, sizeof(resp_path), "%s.resp", tmp_path);
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "curl -sS -m 60 -X POST -H 'Content-Type: application/json' "
             "--data-binary @%s -o %s %s/v1/chat/completions 2>/dev/null",
             tmp_path, resp_path, p->endpoint_url);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Run curl and ignore its stdout — we read the body from -o file
     * because popen pipes were truncating responses. */
    char scratch[64];
    (void)llmgp_run_curl(cmd, scratch, sizeof(scratch));

    /* Slurp the response file. */
    char resp[LLMGP_RESPONSE_CAP];
    long n = 0;
    FILE *rf = fopen(resp_path, "rb");
    if (rf) {
        size_t got = fread(resp, 1, sizeof(resp) - 1, rf);
        resp[got] = '\0';
        n = (long)got;
        fclose(rf);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    unlink(tmp_path);
    unlink(resp_path);

    if (out_seconds) {
        *out_seconds = (double)(t1.tv_sec - t0.tv_sec)
                     + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    }

    if (n <= 0) {
        fprintf(stderr, "llm_game_player: empty/erroneous response (n=%ld)\n", n);
        return -1;
    }
    if (llmgp_extract_content(resp, out_content, out_cap) != 0) {
        fprintf(stderr,
            "llm_game_player: parse failed; resp size=%ld; full body follows:\n"
            "----------------\n%s\n----------------\n",
            n, resp);
        return -1;
    }
    return 0;
}

/* ============================================================
 * Move parsing
 * ============================================================ */

/* Extract the first digit 0-6 in `content`, validate against
 * `valid_columns`.  Returns the column or -1 on no match. */
static int llmgp_parse_move(const char *content, const char *valid_columns) {
    for (const char *p = content; *p; p++) {
        if (*p >= '0' && *p <= '6') {
            int col = *p - '0';
            /* Check it's in the valid set. */
            for (const char *v = valid_columns; *v; v++) {
                if (*v == *p) return col;
            }
            /* Allow the LLM to choose any digit 0-6 even if it's not
             * in valid_columns — the corpus generator will drop the
             * pair anyway if drop_piece fails.  But for cleanliness
             * keep it constrained when we can. */
        }
    }
    return -1;
}

/* Fallback: pick the centre column if legal, else the first legal one. */
static int llmgp_fallback_col(const char *valid_columns) {
    /* Prefer 3 (centre) if present. */
    for (const char *v = valid_columns; *v; v++) {
        if (*v == '3') return 3;
    }
    /* Else the first digit in the list. */
    for (const char *v = valid_columns; *v; v++) {
        if (*v >= '0' && *v <= '6') return *v - '0';
    }
    return -1;
}

/* ============================================================
 * Cache
 * ============================================================ */

static void llmgp_cache_path(const LlmGamePlayer *p,
                             const char *board,
                             const char *valid_columns,
                             char *out, size_t cap) {
    /* Hash (board || valid || model || seed) together. */
    char key[256];
    snprintf(key, sizeof(key), "%s|%s|%s|%d",
             board, valid_columns, p->model_id, p->seed);
    unsigned long long h = llmgp_fnv1a(key, strlen(key));
    snprintf(out, cap, "%s/%016llx.txt", p->cache_dir, h);
}

static int llmgp_cache_load(const LlmGamePlayer *p,
                            const char *board,
                            const char *valid_columns,
                            int *out_move) {
    if (!p->cache_dir[0]) return 0;
    char path[768];
    llmgp_cache_path(p, board, valid_columns, path, sizeof(path));
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) { fclose(f); return 0; }
    fclose(f);
    if (v < 0 || v > 6) return 0;
    *out_move = v;
    return 1;
}

static void llmgp_cache_store(const LlmGamePlayer *p,
                              const char *board,
                              const char *valid_columns,
                              int move) {
    if (!p->cache_dir[0]) return;
    char path[768];
    llmgp_cache_path(p, board, valid_columns, path, sizeof(path));
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%d\n", move);
    fclose(f);
}

/* ============================================================
 * Public move entry-point
 * ============================================================ */

int llm_game_player_move(LlmGamePlayer *p,
                         const char *board_string,
                         const char *valid_columns,
                         int *out_move) {
    if (!p || !board_string || !valid_columns || !out_move) return -1;
    p->stats.total_calls++;

    /* 1. Cache lookup. */
    if (llmgp_cache_load(p, board_string, valid_columns, out_move)) {
        p->stats.cache_hits++;
        return 0;
    }

    /* 2. First attempt — friendly prompt, low temperature. */
    char user[LLMGP_PROMPT_CAP];
    char content[LLMGP_RESPONSE_CAP];
    double secs = 0.0;

    if (llmgp_build_prompt(board_string, valid_columns, 0,
                           user, sizeof(user)) != 0) {
        p->stats.parse_failures++;
        *out_move = llmgp_fallback_col(valid_columns);
        return (*out_move >= 0) ? 1 : -1;
    }
    int rc = llmgp_call(p, user, 0.0, content, sizeof(content), &secs);
    p->stats.cumulative_wallclock_seconds += secs;
    if (rc != 0) {
        /* One retry on transport / parse failure. */
        rc = llmgp_call(p, user, 0.0, content, sizeof(content), &secs);
        p->stats.cumulative_wallclock_seconds += secs;
        if (rc != 0) {
            p->stats.network_failures++;
            *out_move = llmgp_fallback_col(valid_columns);
            return (*out_move >= 0) ? 1 : -1;
        }
    }

    int col = llmgp_parse_move(content, valid_columns);

    /* 3. On parse failure, retry once with the strict reminder. */
    if (col < 0) {
        p->stats.parse_retries++;
        if (llmgp_build_prompt(board_string, valid_columns, 1,
                               user, sizeof(user)) == 0) {
            rc = llmgp_call(p, user, 0.0, content, sizeof(content), &secs);
            p->stats.cumulative_wallclock_seconds += secs;
            if (rc == 0) {
                col = llmgp_parse_move(content, valid_columns);
            }
        }
    }

    if (col < 0) {
        p->stats.parse_failures++;
        *out_move = llmgp_fallback_col(valid_columns);
        if (*out_move < 0) return -1;
        /* Cache the fallback so subsequent re-runs are deterministic. */
        llmgp_cache_store(p, board_string, valid_columns, *out_move);
        p->stats.cache_writes++;
        return 1;
    }

    *out_move = col;
    llmgp_cache_store(p, board_string, valid_columns, col);
    p->stats.cache_writes++;
    return 0;
}
