/*
 * tools/oracle_corpus_source.h — Experiment E15 oracle bridge.
 *
 * Design-time-only bridge to a deterministic-solver binary (A* /
 * BFS).  Invokes the oracle via popen, caches results under a
 * directory keyed by (oracle_path + count + seed + difficulty), and
 * produces JSON-line (state, solution) pairs.
 *
 * Like E12's LLM bridge, the trained organelle MUST NOT call the
 * oracle at inference.  T8 hard-lock: no new build deps beyond libc.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#ifndef ORACLE_CORPUS_SOURCE_H
#define ORACLE_CORPUS_SOURCE_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *oracle_binary;   /* path to the solver binary (e.g. "build/puzzle15_a_star") */
    const char *cache_dir;       /* directory; created on demand */
    int         seed;
    int         count;
    const char *difficulty;      /* "mixed" | "easy" | "medium" | "hard" */
    int         verbose;
} OracleSource;

typedef struct {
    int   cache_hit;
    long  emitted;
    long  bytes_received;
    double wall_seconds;
} OracleEmitStats;

/* Emit (state, solution) pairs from the oracle.
 *
 * If `cache_dir/<hash>.jsonl` exists, slurp it and return; otherwise
 * invoke `oracle_binary --count N --seed S --difficulty K --quiet`
 * via popen, capture stdout, write it to the cache file atomically,
 * and return the same payload as a heap-allocated buffer.
 *
 * `out` receives a heap-allocated buffer the caller frees.  Returns 0
 * on success, nonzero on failure. */
int  oracle_emit(const OracleSource *src,
                 char **out, size_t *out_len,
                 OracleEmitStats *stats,
                 FILE *log);

/* Compute the cache path for the given (oracle_binary, count, seed,
 * difficulty) tuple.  Returns a heap-allocated path the caller frees,
 * or NULL on error. */
char *oracle_cache_path(const OracleSource *src);

/* Compute Jaccard token-overlap between two whitespace-tokenised
 * strings (or, for our purposes, between two board-state strings
 * treated character-by-character).  Used by the T7 leakage audit. */
double oracle_jaccard_state(const char *a, const char *b);

/* Split a buffer of JSON-line {"state":"...","solution":"..."} pairs
 * into a flat array of (state, solution) string pairs.  The state /
 * solution pointers point INTO the original buffer; mutating it
 * invalidates them.  Returns the number of pairs extracted.  `pairs`
 * is heap-allocated; caller frees.  Returns -1 on error. */
typedef struct {
    const char *state;
    size_t      state_len;
    const char *solution;
    size_t      solution_len;
    int         moves;
} OraclePair;

int  oracle_parse_jsonl(char *buf, size_t buf_len,
                        OraclePair **out_pairs, size_t *out_n);

#ifdef __cplusplus
}
#endif

#endif /* ORACLE_CORPUS_SOURCE_H */
