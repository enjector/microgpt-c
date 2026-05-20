/*
 * tools/llm_game_player.h — LLM-as-Connect-4-teacher bridge (Experiment E13).
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Pre-registered in `experiments/E13-llm-game-distillation.md` §1.3.1.
 *
 * Talks to a local LM Studio endpoint (OpenAI-compatible
 * `/v1/chat/completions`) via a `curl` subprocess — no new build deps.
 * The LLM is invoked **only at design time** (corpus generation);
 * the trained student organelle never calls back at inference (E13 T9,
 * RESEARCH_OPA_DIRECTIONS.md §10 hard-lock).
 *
 *  - llm_game_player_new() — heap-allocates a player with a deterministic
 *    cache directory (one JSON file per board hash).
 *  - llm_game_player_move() — given a 42-char board string + a comma-
 *    separated valid-columns list, returns the LLM's chosen column
 *    (0..6).  Cache-first; on miss, POSTs to the endpoint.
 *  - llm_game_player_health_check() — pings the endpoint and confirms
 *    the configured model id appears in `/v1/models`.
 *
 * Pure C99.  No threading, no SIMD — the LLM call is the bottleneck and
 * adds 100s of ms per move.  Single global retry budget per session.
 *
 * Engine surface invariants (E13 T9 / E08 T5):
 *   - This file lives in tools/ — never linked against `microgpt_lib_*`
 *     variants.  Compiles into the `c4_distill_corpus_gen` driver only.
 *   - Includes <stdlib.h>, <stdio.h>, <string.h>, <unistd.h> — no engine
 *     headers (no microgpt.h / microgpt_vm.h).
 */
#ifndef LLM_GAME_PLAYER_H
#define LLM_GAME_PLAYER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlmGamePlayer LlmGamePlayer;

/* Diagnostic counters — read after a session for T6 stats.  Mirrors
 * the per-session totals the spec asks for: total calls, cache hits,
 * retries, parse failures, network failures.  Owned by the player;
 * read directly. */
typedef struct {
    int total_calls;
    int cache_hits;
    int cache_writes;
    int parse_retries;    /* second-attempt LLM calls after first parse failure */
    int parse_failures;   /* even retry failed → fell back to centre column */
    int network_failures; /* curl returned non-zero / no body */
    double cumulative_wallclock_seconds;
} LlmGamePlayerStats;

/* Create a new player.  `cache_dir` may be NULL (no caching).
 * `endpoint_url` defaults to `http://127.0.0.1:1234` if NULL.
 * `model_id` defaults to `qwen/qwen3.6-35b-a3b` if NULL.
 * `seed` is hashed into cache keys so the same (board, seed) is
 * deterministic across runs (T7).  Returns NULL on OOM. */
LlmGamePlayer *llm_game_player_new(const char *model_id,
                                   const char *endpoint_url,
                                   const char *cache_dir,
                                   int seed);

void llm_game_player_free(LlmGamePlayer *p);

/* Ping `/v1/models` on the configured endpoint.  Returns 1 if the
 * configured `model_id` appears in the response, 0 otherwise.
 * On failure (network, parse) returns 0 and writes a diagnostic to
 * stderr.  Must be called once before any move calls. */
int llm_game_player_health_check(const LlmGamePlayer *p);

/* Ask the LLM for a Connect-4 move.
 *
 * Inputs:
 *   board_string  : 42-char "X/O/." board, row-major (top-left first).
 *                   The format used by `demos/character-level/connect4/main.c`.
 *   valid_columns : zero-or-more digits 0..6 separated by ',' (e.g. "0,2,4").
 *                   The LLM's output is constrained to this set; out-of-set
 *                   moves trigger a retry then a fallback.
 *
 * Output:
 *   *out_move : column 0..6 on success, or -1 on hard failure
 *               (network down with no retry left).
 *
 * Returns:
 *   0  on cache hit or successful LLM call (out_move populated).
 *   1  on successful fallback (e.g. parse failure → centre column from
 *      `valid_columns`; out_move populated, caller should still count it).
 *  -1  on hard failure — endpoint unreachable, retry budget exhausted,
 *      no valid column to fall back to.
 */
int llm_game_player_move(LlmGamePlayer *p,
                         const char *board_string,
                         const char *valid_columns,
                         int *out_move);

/* Return a pointer to the player's stats struct (read-only). */
const LlmGamePlayerStats *llm_game_player_stats(const LlmGamePlayer *p);

#ifdef __cplusplus
}
#endif

#endif /* LLM_GAME_PLAYER_H */
