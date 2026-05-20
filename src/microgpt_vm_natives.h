/*
 * microgpt_vm_natives.h  —  VM extern-table extensions for OQL BEHAVIOUR bodies.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 *
 * Provides:
 *   - A small "behaviour context" (vm_natives_ctx) that lets externs share
 *     host-side state with the VM: a string-handle table for non-numeric
 *     I/O, and the current Connect-4-shaped mailbox (board + move + entropy).
 *   - A pack of Connect-4-flavoured extern callbacks dispatched through a
 *     single vm_call_ext_method_callback at runtime.
 *
 *  T3 invariant — ZERO new VM opcodes.  Every primitive is plain C dispatched
 *  via the existing opCALL_EXT_METHOD path.  The VM's bytecode, runtime, and
 *  verifier are entirely untouched by E08.
 *
 * How the parser blesses extern references
 * ----------------------------------------
 * The VM's TS frontend already supports `declare function NAME(...): RET`
 * (see src/microgpt_vm.y, productions `declare_function_header`).  Each
 * declaration registers a function-stub in the module that emits a single
 * `opCALL_EXT_METHOD` and returns.  Therefore the BEHAVIOUR body is
 * responsible for declaring every native it calls — exactly as
 * tests/resources/tests/vm/declare_function ts files already do today — our
 * runtime dispatcher routes the calls.
 *
 * E08-deferred (Phase 5): widen the dispatcher to cover Mastermind /
 * Pentago / 8-puzzle; the cross-demo union enumerated in
 * docs/research/BEHAVIOUR_CATALOGUE.md §4.2.
 *
 * Authoring caveat — single-return idiom
 * --------------------------------------
 * The existing VM treats `return` as a STACK_PUSH+marker only — it does NOT
 * actually exit the function at runtime.  Multiple `return X` statements
 * all execute in source order and the LAST value pushed wins.  Therefore
 * every BEHAVIOUR body should follow the conditions3-shape pattern: assign
 * to a single result variable inside any conditional branches, then return
 * that variable ONCE at the end.  Example:
 *
 *   function eval(): number {
 *     var col    = c4_parse_token();
 *     var result = 0;
 *     if (col >= 0) {
 *       result = c4_column_is_legal_n(col);
 *     }
 *     return result;   // single return; do NOT put `return X` inside the if
 *   }
 *
 * Fixing the VM's RETURN opcode to actually short-circuit is a separate
 * VM-extension proposal (would require either an opcode-semantic patch in
 * vm_module_runtime_run or a new opEXIT opcode — out of scope for E08's
 * T3 zero-new-opcodes lock).  See experiments/E08-oql-behaviours.md §3.9.
 */

#ifndef MICROGPT_VM_NATIVES_H
#define MICROGPT_VM_NATIVES_H

#include "microgpt_vm.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  String-handle interning table
 *
 *  The VM's extern ABI exchanges numbers only.  Strings cross the boundary
 *  as integer handles indexed into this host-owned table.
 * ============================================================ */

typedef struct {
    char **strings;     /* heap-cloned interned strings, owned */
    size_t count;
    size_t capacity;
} vm_natives_str_table;

/* ============================================================
 *  Behaviour context (single-process; one mailbox per registration)
 *
 *  The host stages mailbox values then runs the VM behaviour.  The VM body
 *  reads via registered externs; the externs in turn read from the mailbox.
 * ============================================================ */

typedef struct vm_natives_ctx {
    vm_natives_str_table strings;

    /* Connect-4 mailbox — set by the host between behaviour invocations. */
    int   current_board_handle;     /* handle into strings; -1 if unset */
    int   current_move_handle;      /* handle into strings; -1 if unset */
    double last_entropy;            /* last model-output entropy ∈ [0, log V] */
    int   centre_column;            /* preferred FALLBACK column (default 3) */
} vm_natives_ctx;

/* Lifecycle. */
void vm_natives_ctx_init(vm_natives_ctx *ctx);
void vm_natives_ctx_dispose(vm_natives_ctx *ctx);

/* String handle helpers. */
int         vm_natives_str_intern(vm_natives_ctx *ctx, const char *s);
const char *vm_natives_str_lookup(const vm_natives_ctx *ctx, int handle);

/* ============================================================
 *  Connect-4 native pack — binds the given ctx into a global slot and
 *  populates the dispatcher's name → (arity, fn) table.
 *
 *  Names callable from VM TS via `declare function`:
 *    c4_legal_column_mask()            → number  (7-bit mask: bit c = col c legal)
 *    c4_column_is_legal(col)           → boolean
 *    c4_column_is_legal_n(col)         → number  (0/1 variant of the above)
 *    c4_parse_token()                  → number  (parses current_move → 0..6, or -1)
 *    c4_centre_col()                   → number  (the FALLBACK preference)
 *    c4_last_entropy()                 → number
 *    c4_token_handle(c)                → number  (intern "<digit>"; returns handle)
 *    c4_board_handle_from_str(h)       → number  (identity passthrough)
 *
 *  Returns the number of natives registered.  Single-threaded — same
 *  trade-off as the OQL & VM parsers' global YY_INPUT state.
 * ============================================================ */

int vm_natives_register_c4(vm_natives_ctx *ctx);

/* ============================================================
 *  Runtime dispatcher — install via vm_module_runtime_set_call_ext_method_callback
 *
 *  Pops exactly arity numbers off the runtime stack, invokes the C primitive,
 *  and pushes the numeric result.  Distinct from the engine's
 *  _ext_method_dispatch in that it consults a name-keyed table built by
 *  vm_natives_register_c4 and pops exactly the declared arity rather than
 *  draining the stack.
 * ============================================================ */

void vm_natives_dispatch(struct vm_module_runtime_t *runtime,
                         vm_function *function);

#ifdef __cplusplus
}
#endif

#endif /* MICROGPT_VM_NATIVES_H */
