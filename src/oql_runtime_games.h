/*
 * oql_runtime_games.h  —  reusable game-loop harnesses for OQL `RUN ... ON game_loop`
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * E09 Phase 3 extraction.  The original Connect-4 game-loop lives in
 * demos/character-level/connect4/main.c (~500 LOC). The OQL runtime needs the
 * same logic — column legality / drop / win-check — to drive a pipeline of
 * OqlOrganelles.  This header lifts the deterministic board helpers into a
 * shared module that both the C demo (eventually) and the OQL runtime can call.
 *
 * The functions here are intentionally minimal: they don't know about the OQL
 * runtime; they just operate on `char board[42]` and column indices.  The
 * actual game-loop orchestration (pipeline walk, behaviour dispatch, opponent
 * driver, metric collection) lives in oql_runtime_games.c so the test harness
 * can exercise the full loop end-to-end.
 */

#ifndef OQL_RUNTIME_GAMES_H
#define OQL_RUNTIME_GAMES_H

#include "microgpt_oql.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  Connect-4 board primitives
 * ============================================================ */

#define OQL_C4_ROWS 6
#define OQL_C4_COLS 7
#define OQL_C4_SIZE (OQL_C4_ROWS * OQL_C4_COLS)
#define OQL_C4_EMPTY '.'
#define OQL_C4_X     'X'
#define OQL_C4_O     'O'

/* Index helper: (row, col) -> board[r * COLS + c]. */
int oql_c4_cell_idx(int row, int col);

/* Returns 1 if `col` has a free slot on `board`, else 0. */
int oql_c4_column_legal(const char *board, int col);

/* Drop `player` (`X` or `O`) into `col`; returns the row landed on, or -1
 * if the column is full / out of range. Mutates `board`. */
int oql_c4_drop(char *board, int col, char player);

/* Check for a 4-in-a-row winner; returns 'X', 'O', or `.` if no winner. */
char oql_c4_winner(const char *board);

/* Returns 1 if board is full (i.e. draw if no winner). */
int oql_c4_is_full(const char *board);

/* Random opponent — picks a uniformly random legal column.  Uses *seed
 * as a thread-local PRNG state. Returns -1 if no legal column. */
int oql_c4_random_move(const char *board, unsigned int *seed);

/* ============================================================
 *  Game-loop entry point — invoked by oql_exec_run_runtime.
 *
 *  Walks the pipeline (linear-chain or IR-driven), per game:
 *   - for each call(organelle) stage:
 *     * lazy-load the organelle's checkpoint via oql_runtime_load_organelle,
 *     * dispatch INPUT_BEHAVIOUR  → board encoding / mask,
 *     * forward-inference          → next-token logits,
 *     * sample                     → token,
 *     * dispatch OUTPUT_BEHAVIOUR  → proposed column,
 *     * dispatch VALIDATE_BEHAVIOUR → 0/1,
 *     * if !valid: dispatch FALLBACK_BEHAVIOUR → fallback column,
 *     * drop piece;
 *   - opponent (random) plays;
 *   - until terminal.
 *
 *  Records win / draw / loss counts, per-move latency p99, and an
 *  audit-row counter into rt->last_*.  Returns OQL_OK on completion
 *  even when wins == 0 (a successful run reporting 0% is still a valid
 *  run — see E09 T1 vs T2).
 * ============================================================ */
oql_status oql_run_game_loop(OqlRuntime *rt,
                             OqlPipeline *pipeline,
                             const char *opponent,
                             int games,
                             unsigned int seed,
                             const char *game,
                             FILE *out);

#ifdef __cplusplus
}
#endif

#endif /* OQL_RUNTIME_GAMES_H */
