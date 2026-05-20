/*
 * microgpt_vm_natives.c  —  VM extern-table extensions for OQL BEHAVIOUR bodies.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * E08 Phase 2 — extern table extension only.  No new VM opcodes; the C
 * primitives register into a verb_context (parser-blessed) and dispatch
 * through the existing opCALL_EXT_METHOD path via a shared callback.
 */

#include "microgpt_vm_natives.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 *  Internal: native entry record
 * ============================================================ */

typedef double (*vm_natives_fn)(vm_natives_ctx *ctx,
                                int argc, const double *argv);

typedef struct {
    const char *name;
    int arity;             /* expected argument count (popped from stack) */
    vm_natives_fn fn;
} vm_natives_entry;

/* Single bound context (single-threaded — same trade-off the OQL & VM
 * parsers make for their global YY_INPUT state). */
static vm_natives_ctx   *g_natives_ctx = NULL;

/* Static dispatch table — populated at registration time. */
#define VM_NATIVES_MAX 32
static vm_natives_entry  g_natives_table[VM_NATIVES_MAX];
static int               g_natives_count = 0;

/* ============================================================
 *  Lifecycle
 * ============================================================ */

void vm_natives_ctx_init(vm_natives_ctx *ctx) {
    if (!ctx) return;
    ctx->strings.strings  = NULL;
    ctx->strings.count    = 0;
    ctx->strings.capacity = 0;
    ctx->current_board_handle = -1;
    ctx->current_move_handle  = -1;
    ctx->last_entropy = 0.0;
    ctx->centre_column = 3; /* Connect-4 default */
    ctx->propose_column = NULL;       /* E11: set by runtime after lazy load */
    ctx->propose_column_state = NULL;
}

void vm_natives_ctx_dispose(vm_natives_ctx *ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->strings.count; i++) {
        free(ctx->strings.strings[i]);
    }
    free(ctx->strings.strings);
    ctx->strings.strings  = NULL;
    ctx->strings.count    = 0;
    ctx->strings.capacity = 0;
    if (g_natives_ctx == ctx) g_natives_ctx = NULL;
}

/* ============================================================
 *  String interning
 * ============================================================ */

int vm_natives_str_intern(vm_natives_ctx *ctx, const char *s) {
    if (!ctx || !s) return -1;
    vm_natives_str_table *t = &ctx->strings;
    if (t->count == t->capacity) {
        size_t new_cap = t->capacity ? t->capacity * 2 : 8;
        char **resized = (char **)realloc(t->strings, new_cap * sizeof(char *));
        if (!resized) return -1;
        t->strings = resized;
        t->capacity = new_cap;
    }
    size_t n = strlen(s) + 1;
    char *copy = (char *)malloc(n);
    if (!copy) return -1;
    memcpy(copy, s, n);
    t->strings[t->count] = copy;
    return (int)(t->count++);
}

const char *vm_natives_str_lookup(const vm_natives_ctx *ctx, int handle) {
    if (!ctx) return NULL;
    if (handle < 0) return NULL;
    if ((size_t)handle >= ctx->strings.count) return NULL;
    return ctx->strings.strings[handle];
}

/* ============================================================
 *  Connect-4 board parsing
 * ============================================================ */

#define C4_ROWS 6
#define C4_COLS 7
#define C4_SIZE (C4_ROWS * C4_COLS)
#define C4_EMPTY '.'

static int c4_legal_mask_from_board(const char *board) {
    if (!board) return 0;
    if (strlen(board) < (size_t)C4_SIZE) return 0;
    int mask = 0;
    for (int c = 0; c < C4_COLS; c++) {
        if (board[c] == C4_EMPTY) mask |= (1 << c);
    }
    return mask;
}

/* ============================================================
 *  Native primitives (one C function each — uniform signature)
 * ============================================================ */

static double n_c4_legal_column_mask(vm_natives_ctx *ctx,
                                     int argc, const double *argv) {
    (void)argc; (void)argv;
    if (!ctx) return 0.0;
    return (double)c4_legal_mask_from_board(
        vm_natives_str_lookup(ctx, ctx->current_board_handle));
}

static double n_c4_column_is_legal(vm_natives_ctx *ctx,
                                   int argc, const double *argv) {
    if (!ctx || argc < 1) return 0.0;
    int col = (int)argv[0];
    if (col < 0 || col >= C4_COLS) return 0.0;
    int mask = c4_legal_mask_from_board(
        vm_natives_str_lookup(ctx, ctx->current_board_handle));
    return (mask & (1 << col)) ? 1.0 : 0.0;
}

static double n_c4_parse_token(vm_natives_ctx *ctx,
                               int argc, const double *argv) {
    (void)argc; (void)argv;
    if (!ctx) return -1.0;
    const char *tok = vm_natives_str_lookup(ctx, ctx->current_move_handle);
    if (!tok || !tok[0]) return -1.0;
    char c = tok[0];
    if (c < '0' || c > '6') return -1.0;
    return (double)(c - '0');
}

static double n_c4_centre_col(vm_natives_ctx *ctx,
                              int argc, const double *argv) {
    (void)argc; (void)argv;
    if (!ctx) return 3.0;
    return (double)ctx->centre_column;
}

static double n_c4_last_entropy(vm_natives_ctx *ctx,
                                int argc, const double *argv) {
    (void)argc; (void)argv;
    if (!ctx) return 0.0;
    return ctx->last_entropy;
}

static double n_c4_token_handle(vm_natives_ctx *ctx,
                                int argc, const double *argv) {
    if (!ctx || argc < 1) return -1.0;
    int c = (int)argv[0];
    if (c < 0 || c > 9) return -1.0;
    char buf[2] = {(char)('0' + c), '\0'};
    return (double)vm_natives_str_intern(ctx, buf);
}

static double n_c4_board_handle_from_str(vm_natives_ctx *ctx,
                                         int argc, const double *argv) {
    (void)ctx;
    if (argc < 1) return -1.0;
    return argv[0];
}

/* E11: c4_model_propose_column(temp_x100)
 *
 *   Asks the host (via the runtime-installed callback) to run a real
 *   model-driven move proposal against the current board.  Returns the
 *   proposed column in [0, 6] or -1.0 if either:
 *     - no callback is installed (e.g. checkpoint absent / running under
 *       a unit test that doesn't wire the runtime);
 *     - the callback returned -1 (the model produced an unparseable
 *       token or proposed an illegal column).
 *
 *   The TS-side signature is `declare function c4_model_propose_column(t: number): number;`.
 *   The temperature argument is integer × 100 (e.g. 20 = 0.2) because the
 *   VM ABI is number-only and the OQL runtime keeps the wire format
 *   integer-stable.  Clamped to [1, 100] at the host before being divided
 *   back to a scalar_t temperature.
 */
static double n_c4_model_propose_column(vm_natives_ctx *ctx,
                                        int argc, const double *argv) {
    if (!ctx) return -1.0;
    if (!ctx->propose_column) return -1.0;
    int temp_x100 = (argc >= 1) ? (int)argv[0] : 20;
    if (temp_x100 < 1)   temp_x100 = 1;
    if (temp_x100 > 100) temp_x100 = 100;
    int col = ctx->propose_column(ctx, temp_x100);
    if (col < 0 || col > 6) return -1.0;
    return (double)col;
}

/* ============================================================
 *  Registration helper — adds the (name, arity, fn) entry to the dispatch
 *  table.  The TS-side `declare function NAME(...)` covers the parser
 *  validation — no verb registration needed.
 * ============================================================ */

static void vm_natives_add(const char *name, int arity, vm_natives_fn fn) {
    if (g_natives_count >= VM_NATIVES_MAX) return;
    g_natives_table[g_natives_count].name  = name;
    g_natives_table[g_natives_count].arity = arity;
    g_natives_table[g_natives_count].fn    = fn;
    g_natives_count++;
}

/* ============================================================
 *  Public registration entry point
 * ============================================================ */

int vm_natives_register_c4(vm_natives_ctx *ctx) {
    if (!ctx) return 0;
    g_natives_ctx   = ctx;
    g_natives_count = 0;  /* reset for fresh registration */

    vm_natives_add("c4_legal_column_mask",     0, n_c4_legal_column_mask);
    vm_natives_add("c4_column_is_legal",       1, n_c4_column_is_legal);
    /* number-typed twin of c4_column_is_legal (returns 0/1).  Researchers
     * default to the number variant in BEHAVIOUR bodies that mix the check
     * with arithmetic — keeps the VM IL all-numeric and avoids the
     * boolean-result-in-if-test pattern (which the current VM verifier
     * compiles surprisingly; see RESEARCH note in microgpt_vm_natives.c
     * header). */
    vm_natives_add("c4_column_is_legal_n",     1, n_c4_column_is_legal);
    vm_natives_add("c4_parse_token",           0, n_c4_parse_token);
    vm_natives_add("c4_centre_col",            0, n_c4_centre_col);
    vm_natives_add("c4_last_entropy",          0, n_c4_last_entropy);
    vm_natives_add("c4_token_handle",          1, n_c4_token_handle);
    vm_natives_add("c4_board_handle_from_str", 1, n_c4_board_handle_from_str);
    /* E11: single new extern — full board+prompt-protocol model proposal. */
    vm_natives_add("c4_model_propose_column",  1, n_c4_model_propose_column);
    return g_natives_count;
}

/* ============================================================
 *  Runtime dispatcher — popping per declared arity (NOT draining stack)
 *
 *  Walks g_natives_table for a name match; if found, pops `arity` numbers
 *  off the runtime stack, invokes the C primitive with the bound ctx, and
 *  pushes the numeric result.  Pops are LIFO so we reverse the arg list to
 *  restore declaration order.
 * ============================================================ */

void vm_natives_dispatch(struct vm_module_runtime_t *runtime,
                         vm_function *function) {
    if (!runtime || !function || !function->name) return;

    /* Find entry. */
    vm_natives_entry *entry = NULL;
    for (int i = 0; i < g_natives_count; i++) {
        if (strcmp(g_natives_table[i].name, function->name) == 0) {
            entry = &g_natives_table[i];
            break;
        }
    }
    if (!entry) {
        fprintf(stderr, "[vm_natives] unknown native: %s\n", function->name);
        vm_module_runtime_stack_push_number(runtime, 0.0);
        return;
    }

    double args[16];
    int    argc = entry->arity;
    if (argc > 16) argc = 16;

    /* Pop args in LIFO order; the VM pushed them left-to-right, so the
     * topmost element is the last declared arg.  Reverse into argv. */
    for (int i = argc - 1; i >= 0; i--) {
        vm_variable *var = NULL;
        if (vm_module_runtime_stack_pop(runtime, &var) != VM_OK || !var) {
            args[i] = 0.0;
            continue;
        }
        args[i] = var->value.number;
        vm_variable_dispose(var);
    }

    double ret = entry->fn(g_natives_ctx, argc, args);

    /* Push back as the type the function's TS signature declared.  This
     * matches the existing declare_function test callbacks, which push a
     * boolean for `: boolean`-typed externs and a number otherwise. */
    if (function->return_type_class == ptcBOOLEAN) {
        vm_module_runtime_stack_push_boolean(runtime, ret != 0.0);
    } else {
        vm_module_runtime_stack_push_number(runtime, ret);
    }
}
