# TDD_vm — Technical Design Document (Virtual Machine)

**Document ID:** TDD-VM-001
**Version:** 1.0
**Status:** DRAFT
**Paired BS:** `BS_vm.md`
**FS:** `FS_vm_bytecode.md`
**Sources:** `src/microgpt_vm.h`, `src/microgpt_vm.c`, `src/microgpt_vm.l`, `src/microgpt_vm.y`, pre-generated `src/microgpt_vm_parser.{l.c,tab.c,tab.h}`.

## 1. Overview

The VM is a TypeScript-flavoured surface language compiled through Flex / Bison ≥ 3.0 to bytecode and executed by an interpreter loop. It is a self-contained module — the only dependency is on the core engine's containers (`vm_list`, `vm_map`, `vm_queue`, `vm_string_buffer`) which are header-only inlines in `microgpt_vm.h`.

The VM exists primarily so that learned C / TypeScript code (the `c_vm_codegen`, `w_vm_codegen_*`, `c_vm_compose` demos) can be executed at training-evaluation time, and so the Pipeline IR can use a VM-backed dispatch path (`pipeline_execute_vm`) for INT/FLOAT-typed pipelines.

## 2. Architecture

```
   surface source (TypeScript-ish)
            │
            ▼
        Flex lexer (microgpt_vm.l)
            │ tokens
            ▼
        Bison parser (microgpt_vm.y, %define api.prefix)
            │ AST
            ▼
        vm_module_generator
            │ instructions (vm_instruction[])
            ▼
        6-pass verifier
            │ verified vm_module
            ▼
        vm_module_runtime (interpreter loop)
            │ runtime stack of vm_variable
            ▼
        vm_engine_result_{number,string,bool}
```

When Flex / Bison ≥ 3.0 are not available at build time, CMake falls back to the committed pre-generated parser sources (`microgpt_vm_parser.l.c`, `microgpt_vm_parser.tab.c`, `microgpt_vm_parser.tab.h`).

## 3. Data flow

`vm_engine_load(e, source)`:
1. Lex / parse the source into an AST.
2. Walk AST → emit `vm_instruction[]` per function.
3. Run the 6-pass verifier (symbol resolution → type-class → arity → label uniqueness → jump validity → stack balance).
4. On success, `vm_module` holds the compiled state and is executable.

`vm_engine_run(e, fn_name)`:
1. Resolve `fn_name` in the module's function table.
2. Push a new runtime frame with parameter `vm_variable`s.
3. Loop over instructions, dispatching on opcode (see `FS_vm_bytecode.md` § 4).
4. On `opRETURN`, pop frame and yield the return value to `vm_engine_result_*`.

## 4. Key data structures

### 4.1 Containers (header-only)

`vm_list` — growable `void *` array with per-item type metadata.
`vm_map` — string-keyed dictionary; open-addressed hash table.
`vm_queue` — singly-linked FIFO.
`vm_string_buffer` — growable C-string builder with `printf`-style append.

These exist as static inlines in `microgpt_vm.h` so user code that includes the header gets the containers for free.

### 4.2 `vm_variable`

A tagged union holding `bool / double / char* / void*` plus type tag (`vm_param_type_class`) and lifecycle flags (`is_register`, `is_constant`, `is_used`, `is_preallocated`). Optional debug name when `_DEBUG_TRACE` is defined.

### 4.3 `vm_instruction`

Internal struct (defined in `microgpt_vm.c`) with:
- Opcode tag (`vm_instruction_opcode`).
- Primary operand (variable name or label), heap-allocated.
- Optional secondary operand.
- Source line / column for diagnostics.

### 4.4 `vm_function`, `vm_module`, `vm_engine`

`vm_function` holds a name, parameter list, instruction array, label-to-index map. `vm_module` holds an ordered list of functions and a global symbol table. `vm_engine` is the public-facing handle owning a module + runtime.

## 5. Algorithms

### 5.1 Lexer / parser

Standard Flex / Bison patterns; `microgpt_vm.l` defines tokens for keywords, identifiers, numbers, strings, operators, punctuation. `microgpt_vm.y` declares the grammar with `%define api.prefix` (Bison ≥ 3.0 only). The grammar is a simplified subset of TypeScript covering function declarations, expression statements, control flow, variable declarations, calls, object accessors.

### 5.2 Code generator

A recursive walk of the AST emits opcodes per construct:
- Variable declaration → `opCREATE_SET_VAR`.
- Assignment → `opSET_VAR` or `opSET_OBJ_VAR`.
- Binary expression → operands pushed via `opSTACK_PUSH`, then `opADD/SUB/...`, leaving the result on the stack.
- Comparison → `opCONDITION_*` writes a boolean.
- `if`/`while`/`for` → `opLABEL`s plus `opJUMP_IF_*` for branches.
- Function call → `opCALL_METHOD` / `_OBJ_METHOD` / `_EXT_METHOD`.
- `return` → `opRETURN`.

### 5.3 6-pass verifier

Each pass is straight-line over the `vm_instruction[]` array:
1. **Symbols**: every identifier resolves to a declaration in scope.
2. **Type-class**: arithmetic operands are `ptcNUMBER`; conditions are coercible to boolean.
3. **Arity**: calls match declared parameter count.
4. **Label uniqueness**: `opLABEL` targets do not duplicate within a function.
5. **Jump validity**: every `opJUMP*` target is a known label.
6. **Stack balance**: `opSTACK_PUSH` and `opSTACK_POP` are balanced across each control-flow path (a small dataflow analysis).

### 5.4 Runtime interpreter

A switch-dispatched loop over `vm_instruction *pc`. Each opcode mutates a small register file plus the runtime stack of `vm_variable *`. Native function calls (`opCALL_EXT_METHOD`) marshal stack operands to a `double[]` and call the registered `vm_native_fn`; the return is a single `double` placed back on the stack.

The `verb_*` API is a higher-level dispatch layer (sentence → verb match → exec) layered over the VM; it is used by the experimental verb-context demos but is not load-bearing for the core platform.

## 6. Concurrency model

A `vm_engine` is single-threaded. Concurrent execution requires multiple engines, each with its own loaded module. Re-entrancy is undefined: a registered native must NOT call back into the same engine on the same thread.

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Surface language | TypeScript-ish | Lua / Lisp / custom DSL | The codegen demos target real-world-looking code; TypeScript-shape is what the corpus contains. |
| Parser generator | Flex / Bison | Hand-rolled | Standard tools, well-understood; mac OS ships an old Bison so we commit pre-generated sources as a fallback. |
| Bytecode form | In-memory only | On-disk binary | The compile-time cost is small enough (M tok/s grammar, small functions) that re-compiling per run is acceptable; on-disk is reserved for future releases. |
| Runtime model | Stack-and-register hybrid | Pure stack / pure register | Hybrid handles both the common arithmetic patterns (stack) and named-variable lookups (register) efficiently. |
| Native ABI | `double(int argc, const double *argv)` | Variadic / typed | Keeps the ABI fixed and trivially wrappable; complex types must marshal to / from doubles. The Pipeline IR's `pipeline_execute_vm` inherits this constraint. |

## 8. Known limitations

- No closures, classes, async, or generators.
- Native ABI is double-only; STRING / LIST / TENSOR / RECORD types in the Pipeline IR cannot dispatch through `pipeline_execute_vm`.
- `opXPATH` and `opJSON` are reserved opcode tags whose runtime semantics are deferred (see `FS_vm_bytecode.md` `GAP-VM-002`).
- No on-disk bytecode form; `vm_engine_load` is the only way in.
- `vm_engine_dump_il` writes to stdout only.

## 9. References

- `docs/research/RESEARCH_VM_CODEGEN.md`.
- Standard Flex / Bison documentation.
- The `c_vm_codegen`, `w_vm_codegen_*`, `c_vm_compose` demos for end-to-end exercise.

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
