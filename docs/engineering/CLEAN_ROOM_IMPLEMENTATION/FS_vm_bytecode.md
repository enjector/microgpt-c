# FS_vm_bytecode — Functional / Format Specification

**Document ID:** FS-VM-001
**Version:** 1.0
**Status:** DRAFT
**Last updated:** 2026-04-30
**Source of truth:** `src/microgpt_vm.h` (opcode enum, public types), `src/microgpt_vm.c` (compiler, runtime, verifier passes), `src/microgpt_vm.l` and `microgpt_vm.y` (lexer/parser).

---

## RFC 2119

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL in this document are to be interpreted as described in RFC 2119.

## 1. Format overview

The MicroGPT-C VM consumes a TypeScript-flavoured surface language and lowers it to a register-and-stack hybrid bytecode for execution by a runtime. This document specifies:

- The **surface language tokens** and grammar (lexer / parser inputs).
- The **opcode set** and the meaning of each instruction.
- The **instruction descriptor** that the runtime walks.
- The **module image** that can be (eventually) serialised to disk.

The serialisation of compiled bytecode to a file is **not** required for V1.0 — the engine compiles source on `vm_engine_load(e, src)` and dispatches via in-memory instructions. A future on-disk format is recorded as `GAP-VM-001` in `TRACEABILITY.md`.

## 2. Lexer / parser

The lexer is implemented with Flex (`src/microgpt_vm.l`); the parser with Bison ≥ 3.0 (`src/microgpt_vm.y`). Pre-generated output (`microgpt_vm_parser.l.c`, `microgpt_vm_parser.tab.c`, `microgpt_vm_parser.tab.h`) is committed so a build that lacks a recent Bison still succeeds.

The grammar uses `%define api.prefix` (Bison ≥ 3.0). macOS's stock Bison 2.3 cannot regenerate the parser. CMake auto-falls back to the pre-generated sources.

Surface-language tokens (informal, derived from `microgpt_vm.l`):

| Token class | Examples |
|---|---|
| Keywords | `let`, `const`, `function`, `if`, `else`, `while`, `for`, `return`, `true`, `false`, `null` |
| Type tokens | `number`, `string`, `boolean`, `void`, `object` |
| Identifiers | `[A-Za-z_][A-Za-z0-9_]*` |
| Number literal | Decimal with optional fraction (`0`, `1.5`, `-3`) |
| String literal | Double-quoted, with C-style escape sequences |
| Operators | `+ - * / % ** = == != <= >= < > && || !` |
| Punctuation | `( ) { } [ ] ; , . :` |

The grammar matches a simplified TypeScript: function declarations, top-level statements, expression statements, control flow (`if`/`while`/`for`), variable declarations, function calls. Object property access (`obj.prop`) and indexing (`arr[i]`) are supported. Closures, classes, and async functions are NOT supported in V1.0.

## 3. Compilation pipeline

```
source ─▶ Flex ─▶ Bison ─▶ AST ─▶ vm_module_generator ─▶ instructions ─▶
        ─▶ 6-pass verifier ─▶ vm_module ready for vm_module_runtime
```

Each pass is implemented in `microgpt_vm.c`. The 6-pass verifier:

1. **Symbol resolution** — every identifier reference resolves to a declaration.
2. **Type-class check** — operands of arithmetic ops are `ptcNUMBER`; conditions are coercible to boolean.
3. **Function arity check** — calls match declared arity (or signal a usage error).
4. **Label uniqueness** — `opLABEL` targets are unique within a function.
5. **Jump validity** — every `opJUMP*` target exists in the function's label table.
6. **Stack-balance invariant** — `opSTACK_PUSH` and `opSTACK_POP` are paired across each control-flow path.

## 4. Opcode set

Defined in `vm_instruction_opcode_t` in `microgpt_vm.h`. The full enum (tag values are stable across V1.x):

```
opNOP, opINC, opDEC, opADD, opMUL, opDIV, opEXP, opPOW, opSUB, opNEG, opNOT,
opSET_VAR, opCREATE_SET_VAR, opGET_OBJ_VAR, opSET_OBJ_VAR,
opSTACK_POP, opSTACK_PUSH,
opCALL_METHOD, opCALL_OBJ_METHOD, opCALL_EXT_METHOD,
opRETURN,
opCONDITION_GTE, opCONDITION_LTE, opCONDITION_GT, opCONDITION_LT,
opCONDITION_NE,  opCONDITION_EQ,  opCONDITION_TRUE,
opCONDITIONAL_AND, opCONDITIONAL_OR,
opJUMP_IF_TRUE, opJUMP_IF_FALSE, opJUMP,
opLABEL, opYIELD, opXPATH, opJSON, opCOMMENT
```

| Opcode | Effect |
|---|---|
| `opNOP` | No operation. |
| `opINC` / `opDEC` | Increment / decrement a named variable in place. |
| `opADD` / `opSUB` / `opMUL` / `opDIV` / `opEXP` / `opPOW` / `opNEG` | Arithmetic on the runtime stack: pop operands, push result. |
| `opNOT` | Logical NOT on the top of the stack. |
| `opSET_VAR` | Assign top-of-stack to an existing variable by name. |
| `opCREATE_SET_VAR` | Declare a new variable and assign it. |
| `opGET_OBJ_VAR` / `opSET_OBJ_VAR` | Object property accessors. |
| `opSTACK_PUSH` / `opSTACK_POP` | Push / pop the runtime stack. |
| `opCALL_METHOD` | Call a function defined in the loaded module. |
| `opCALL_OBJ_METHOD` | Call a method on an object value. |
| `opCALL_EXT_METHOD` | Call an external (native) C function registered via `vm_engine_register_fn`. |
| `opRETURN` | Return from the current function. |
| `opCONDITION_GT` / `LT` / `GTE` / `LTE` / `EQ` / `NE` | Comparison; push boolean. |
| `opCONDITION_TRUE` | Truthiness test on top of stack. |
| `opCONDITIONAL_AND` / `OR` | Boolean AND/OR. |
| `opJUMP_IF_TRUE` / `JUMP_IF_FALSE` / `JUMP` | Control flow; target is a label name. |
| `opLABEL` | Label target (no-op at execution). |
| `opYIELD` | Co-routine yield (cooperative scheduling hook; reserved). |
| `opXPATH` | XPath evaluation against an XML value (deferred — see `GAP-VM-002`). |
| `opJSON` | JSON access (deferred — see `GAP-VM-002`). |
| `opCOMMENT` | Source-level comment, executed as a no-op. |

`vm_instruction_opcode_to_string(op)` is REQUIRED to round-trip an opcode tag to its source-level name (e.g. `opADD → "opADD"`). This is used by `vm_engine_dump_il`.

## 5. Instruction descriptor

Each instruction is represented by a `vm_instruction` struct (defined in `microgpt_vm.c`; the public surface exposes the opcode enum only). At minimum it carries:

- The opcode tag.
- A primary operand (variable name or label string), heap-allocated.
- An optional secondary operand.
- A line / column annotation for diagnostic messages.

Function bodies are flat arrays of instructions.

## 6. Module image (compiled state)

A `vm_module` holds:

- A list of `vm_function`s, each with a name, parameter list, instruction array, and label-to-index map.
- A symbol table mapping global names to `vm_variable`s.
- A pool of `vm_variable`s pre-allocated for the runtime stack.

The runtime (`vm_module_runtime`) walks each function's instruction array using a small interpreter loop. The dispatch rate (per `bench_microgpt_vm.c`) is 3.7–5.8 M ops/s single-threaded.

## 7. External function ABI

Native C functions are registered via:

```c
typedef double (*vm_native_fn)(int argc, const double *argv);
void vm_engine_register_fn(vm_engine *e, const char *name, vm_native_fn fn);
```

The signature constraints:

- Inputs are `argc` doubles in `argv[]`.
- The return is a single `double`.
- Side effects are permitted but the function MUST NOT call back into the same engine on the same thread (re-entrancy is undefined in V1.0).

This signature is the primary reason `pipeline_execute_vm` in the Pipeline IR module restricts itself to INT/FLOAT-typed pipelines (see `BS_pipeline_ir.md`).

## 8. Error codes

Returned by the high-level engine API and by `verb_*` helpers. Distinct namespaces:

| Range | Owner | Examples |
|---|---|---|
| `VM_OK` (0), `VM_UNKNOWN` (1), `VM_FAIL` (2) | Generic VM result | Returned by most engine APIs |
| `2300..2306` | Verb dispatch errors | `RESULT_CORE_VERB_ERROR_NO_MATCH`, `RESULT_CORE_VERB_ERROR_INCORRECT_USAGE`, `RESULT_CORE_VERB_ERROR_EXEC_NOT_ENOUGH_PARAMS` |
| `vm_engine_load` non-zero | Compile-time error | Use `vm_engine_last_error(e)` |
| `vm_engine_run` non-zero | Runtime error | Use `vm_engine_last_error(e)` |

## 9. Surface-language example

```typescript
function fib(n: number): number {
  if (n < 2) {
    return n;
  }
  return fib(n - 1) + fib(n - 2);
}

function main(): number {
  return fib(10);
}
```

Compilation produces an IL trace inspectable via `vm_engine_dump_il(e)`. Running `main` returns `55.0` accessible through `vm_engine_result_number(e)`.

## 10. Versioning

The opcode enum is **append-only** within version 1.x. New opcodes MAY be added to the end; existing tag values MUST remain stable. A breaking change (re-numbering, removal) bumps to 2.0 and SHALL be marked by a new module magic if/when an on-disk format is introduced.

`opXPATH` and `opJSON` are reserved tags whose runtime semantics are deferred (`GAP-VM-002`); they SHALL NOT be removed; if and when they ship, the implementation SHALL match the runtime semantics described in this document at that time.

## 11. Test vectors

`tests/test_microgpt_vm.c` exercises:

- Lexer / parser for the supported grammar.
- Compiler — every opcode is emitted by at least one fixture program.
- Verifier — each of the 6 passes has a positive and negative test.
- Runtime — `bench_microgpt_vm.c` measures dispatch speed.

`resources/vm/` holds source fixtures consumed by the test harness; CMake copies these next to the test binary at build time.

## 12. Cross-references

- `BS_vm.md` for the runtime invariants (precondition / postcondition contracts on each opcode).
- `TDD_vm.md` for the implementation strategy.
- `BS_pipeline_ir.md` (REQ-PIPE-008) for the VM-backed dispatch path.
- `FRD.md` REQ-VM-001 .. REQ-VM-006.

## 13. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
