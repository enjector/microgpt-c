# The VM Organelle — Complete Specification

## Spear Summary

**Point:** The VM is a self-contained compiler and runtime that turns TypeScript-like scripts *and LaTeX math* into executable bytecode in under 5 000 lines of pure C99.

**Picture:** It's like a universal power adapter — you plug in either a `.ts`-style program or a handwritten `\sum` equation and the same 35-instruction engine runs them both at native speed.

**Proof:** 45 tests pass across 55 compiler and 34 runtime fixtures including nested `\sum` loops, multi-function call chains, and conditional branching — all compiled and executed through one `switch`-dispatch loop.

**Push:** If you need to embed scriptable logic or mathematical expressions inside the MicroGPT-C training pipeline, grab `microgpt_vm.h`, link `microgpt_vm_lib`, and you're done in three API calls.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Compilation Pipeline](#3-compilation-pipeline)
4. [Bytecode Reference](#4-bytecode-reference)
5. [API Reference](#5-api-reference)
6. [Internals](#6-internals)
7. [Constraints & Limitations](#7-constraints--limitations)
8. [Examples & Test Fixtures](#8-examples--test-fixtures)
9. [Known Issues](#9-known-issues)
10. [Roadmap](#10-roadmap)
11. [Source Files](#11-source-files)

---

## 1. Overview

The VM organelle is a **stack-based virtual machine** that compiles a TypeScript-like language (plus LaTeX mathematical notation) into a 35-opcode intermediate language (IL) and executes it. It is implemented in pure C99 with Flex/Bison for parsing. The entire system is self-contained in two files: one header (`microgpt_vm.h`, 1 470 lines) and one implementation (`microgpt_vm.c`, 3 675 lines), plus the lexer/grammar definitions.

### Dual Syntax Support

The same engine accepts two input styles that compile to identical bytecode:

```typescript
// TypeScript-like syntax
function add(x: number, y: number): number {
    return x + y;
}
```

```latex
// LaTeX mathematical notation
f(x, y) = x^2 + y^3
e() = \sum_{i=0}^{20} i + 3
```

---

## 2. Architecture

### Pipeline Overview

```
Source Code → Flex Lexer → Bison Parser → Code Generator → Post-Processing → Verifier → Runtime → Result
```

### Component Hierarchy

```
vm_engine                  ← High-level C API (opaque handle)
  ├── vm_module            ← Compilation unit
  │     ├── vm_function    ← Bytecode container per function
  │     │     ├── vm_instruction   ← Single opcode + 3 operands + metadata
  │     │     ├── vm_variable      ← Tagged union (boolean | number | string | other)
  │     │     ├── vm_type_trait    ← Compile-time type propagation
  │     │     ├── labels           ← cmap<name → instruction index>
  │     │     ├── registers        ← Compiler-generated temporaries (_reg0, _reg1...)
  │     │     ├── constants        ← Compile-time literal values
  │     │     └── symbols          ← String lifecycle tracking
  │     └── errors         ← sequence<vm_module_error*>
  ├── vm_module_runtime    ← Stack machine execution context
  │     └── stack[100]     ← Fixed-size evaluation stack
  └── native_fns[]        ← Registered C callback table
```

### Core Data Types

#### `vm_module` — Compilation Unit

| Field | Type | Purpose |
|-------|------|---------|
| `functions` | `cmap<name → vm_function*>` | O(1) function lookup by name |
| `functions_list` | `sequence<vm_function*>` | Declaration-order traversal |
| `errors` | `sequence<vm_module_error*>` | Parse/compile diagnostics with line info |
| `verb_context_` | `verb_context*` | Back-pointer to engine for native dispatch |

#### `vm_function` — Bytecode Container

| Field | Type | Purpose |
|-------|------|---------|
| `instructions` | `sequence<vm_instruction*>` | The IL bytecode |
| `parameters` | `sequence<char*>` | Parameter names |
| `labels` | `cmap<name → index>` | Jump targets |
| `variables` | `cmap<name → vm_variable*>` | Local symbol table |
| `registers` | `cmap<name → vm_variable*>` | Compiler-generated temporaries |
| `constants` | `cmap<name → vm_variable*>` | Compile-time values |
| `symbols` | `sequence<char*>` | All tracked strings (lifecycle) |
| `return_type` | `char*` | Declared return type name |
| `return_type_class` | `vm_param_type_class` | Resolved return type class |

#### `vm_instruction` — Single Opcode

```c
typedef struct vm_instruction_t {
    vm_instruction_opcode opcode;

    char *param1;                      // Operand 1 string
    char *param1_type_name;            // Type annotation
    vm_param_type_class param1_type_class;  // Resolved type
    bool param1_is_constant;           // Compile-time literal?
    bool param1_is_register;           // Compiler temporary?
    bool param1_is_resolved;           // Pre-allocated?

    // param2 and param3: same shape as param1

    size_t meta_source_line_number;    // Source line for diagnostics
} vm_instruction;
```

#### `vm_variable` — Tagged Value

| Field | Purpose |
|-------|---------|
| `type_class` | `ptcNONE`, `ptcBOOLEAN`, `ptcNUMBER`, `ptcSTRING`, `ptcOTHER` |
| `value` | Union: `boolean` (bool), `number` (double), `string` (char*), `other` (void*) |
| `is_register` | Temporary — allocated by codegen |
| `is_constant` | Immutable after first assignment |
| `is_used` | Referenced during execution |
| `is_preallocated` | Pool-managed; skip `free()` |

### Support Layer Data Structures

The header includes a self-contained C99 support layer (no external dependencies):

| Type | Implementation | Purpose |
|------|---------------|---------|
| `sequence` | Growable `void*` array with per-item type metadata | Function lists, instruction lists, parameters |
| `cmap` | Open-addressing hash map (DJB2, power-of-2 capacity) | Function lookup, symbol tables, variables |
| `queue` | Singly-linked FIFO | Code fragment management during codegen |
| `string_buffer` | Growable `char*` builder | IL dump, error messages |
| `verb_context` | String-keyed function dispatch table | Native function registration |

### Memory Model

#### Ownership Rules

1. **Engine owns Module and Runtime** — `vm_engine_dispose()` frees both
2. **Module owns Functions** — `vm_module_dispose()` walks `functions` cmap
3. **Function owns Instructions, Variables, Symbols** — `vm_function_dispose()` frees all sub-resources
4. **Strings are cloned on store** — `cmap_set` clones keys; `string_clone()` for all heap strings

#### Allocation Primitives

- `xmemory_new(T)` → `calloc(1, sizeof(T))` — zero-initialised allocation
- `xmemory_free(p)` → `free(p); p = NULL` — NULL-safe with auto-null
- `string_clone(s)` → `malloc + memcpy` — heap-allocated copy

#### Stack Machine

```
┌──────────┐  ← stack_size (grows up, max 100)
│ var[n-1]  │
│ var[n-2]  │
│   ...     │
│ var[0]    │
└──────────┘  ← base
```

Push operations create new `vm_variable` instances. Pop returns a pointer. The stack is reset between function calls via `vm_module_runtime_clear()`.

---

## 3. Compilation Pipeline

### 3.1 Lexer (`microgpt_vm.l`)

Flex scanner with prefix `vm_module_parser_`. Uses a custom `YY_INPUT` macro that reads from an **in-memory string buffer** (no file I/O).

#### Token Categories

| Category | Tokens |
|----------|--------|
| **Keywords** | `function`, `return`, `if`, `else`, `var`, `while`, `for`, `yield`, `declare` |
| **Literals** | `NUMBER` (ints/decimals), `STRING` (single/double quoted), `BOOLEAN` (`true`/`false`) |
| **Identifiers** | `NAME` — `[a-zA-Z_][0-9a-zA-Z_]*` |
| **Operators** | `+`, `-`, `*`, `/`, `^`, `++`, `--`, `+=`, `-=` |
| **Comparisons** | `>=`, `<=`, `>`, `<`, `!=`, `==` |
| **Logical** | `&&`, `\|\|` |
| **LaTeX** | `\sum_`, `\frac`, `\pow`, `α`/`\alpha` |
| **Comments** | `//...` and `/* ... */` |

#### Input Mechanism

```c
#undef YY_INPUT
#define YY_INPUT(buf, result, max_size) \
{ \
    int c = vm_module_parser_char_fetch_next(_vm_ctx_current_parser); \
    result = (c == 0) ? YY_NULL : (buf[0] = c, 1); \
}
```

The lexer tracks `vm_module_parser_state_input_line_number` and `vm_module_parser_state_input_line_column` for error reporting. Newlines reset the column counter.

### 3.2 Parser (`microgpt_vm.y`)

Bison grammar with prefix `vm_module_parser_`. Produces **no AST** — grammar actions directly emit bytecode via macros.

#### Grammar Structure

```
code
  └── comments functions comments

functions
  └── function*
        ├── TypeScript function:  function_header '(' params ')' ':' type '{' statements '}'
        ├── Declare function:     'declare' function_header '(' params ')' ':' type ';'
        └── LaTeX math function:  math_function_header '=' math_expression

statement
  ├── var_declaration ';'         → opCREATE_SET_VAR
  ├── assignment ';'              → opSET_VAR
  ├── return expression ';'       → opSTACK_PUSH + opRETURN
  ├── function_call ';'           → opCALL_METHOD / opCALL_EXT_METHOD
  ├── if / else                   → opJUMP_IF_FALSE / opJUMP
  ├── while                       → opLABEL + opJUMP_IF_FALSE + opJUMP
  ├── for                         → opLABEL + opJUMP_IF_FALSE + deferred increment + opJUMP
  └── increment_expression ';'    → opINC / opDEC / opADD / opSUB

expression
  ├── literals (NUMBER, STRING, BOOLEAN, NAME)
  ├── binary ops (+, -, *, /, ^)  → opADD / opSUB / opMUL / opDIV / opEXP
  ├── unary (!, -)                → opNOT / opNEG
  ├── function calls NAME '(' args ')'
  ├── property access NAME '.' NAME
  └── parenthesised '(' expression ')'
```

#### Code Emission Macros

```c
#define emit(op, p1, p2, p3)    vm_module_generator_function_emit(gen, op, p1, p2, p3)
#define function_begin(name)    vm_module_generator_function_begin(gen, name)
#define function_end()          vm_module_generator_function_end(gen)
#define create_register()       vm_module_generator_tmp_register_create(gen)
#define create_label()          vm_module_generator_label_create(gen)
#define labels_push(lbl)        vm_module_generator_label_push(gen, lbl)
#define labels_pop()            vm_module_generator_label_pop(gen)
#define track(s)                vm_module_generator_symbol_track(gen, s)
#define trait_type_set(v,t,c)   vm_module_generator_trait_type_set(gen, v, t, c)
#define trait_type_link(s,d)    vm_module_generator_trait_type_link(gen, s, d)
```

#### LaTeX Math Compilation

`\sum_{var=start}^{end} body` compiles into:

```
CREATE_SET_VAR  var   start            (loop init)
LABEL           loop_begin
CONDITION_LT    var   end   _reg       (loop condition)
JUMP_IF_FALSE   _reg  loop_end
ADD             accum body  accum      (loop body)
INC             var                    (increment)
JUMP            loop_begin
LABEL           loop_end
```

### 3.3 Code Generator

Entry point: `vm_module_generator_create(module)`.

#### Function Lifecycle

1. `function_begin(name)` — creates `vm_function`, adds to module, emits `opLABEL name`
2. Grammar actions call `emit(opcode, p1, p2, p3)` for each statement
3. `function_end()` — runs post-processing, adds function to module's lookup map

#### Register Allocation

- Named `_reg0`, `_reg1`, ... (prefix `_reg` + counter)
- Counter resets per function via `reset_registers()`
- Registers are pre-allocated as `vm_variable` during the interning pass

#### Label Management

- `create_label()` generates `_label0`, `_label1`, ...
- `labels_push(label)` pushes onto stack for nested constructs
- `labels_pop()` retrieves innermost label for `JUMP` targets

#### Deferred Code Fragments

For `if`/`else` and `for` loops, the generator uses a `queue` of code fragments:

- `emit_defer_push_begin()` — start capturing instructions into a deferred queue
- `emit_defer_push_end()` — stop capturing
- `emit_defer_pop()` — flush deferred instructions into the main instruction stream

### 3.4 Post-Processing (3 passes)

#### Pass 1: Jump Resolution (`_vm_post_processing_resolve_jumps`)

Converts label-name jumps to instruction-index jumps:
1. Build `labels` map: `name → instruction_index` for all `opLABEL` instructions
2. For each `opJUMP`/`opJUMP_IF_TRUE`/`opJUMP_IF_FALSE`, replace label name with resolved index

#### Pass 2: Variable Interning (`_vm_post_processing_intern_variables`)

Pre-allocates variables referenced in instructions:
1. Walk all instructions in each function
2. For each operand with `is_resolved = false`, look up in `variables`/`registers`/`constants`
3. Create `vm_variable` instances, set `is_resolved = true`

#### Pass 3: Call Method Pre-calculation (`_vm_post_processing_call_method_pre_calc`)

For `opCALL_METHOD` instructions:
1. Resolve target function by name from module
2. Store function pointer directly in instruction for fast dispatch

### 3.5 Verifier (6 passes)

| Pass | Function | What It Checks |
|------|----------|---------------|
| 1 | `verify_not_null` | All instruction operands are non-NULL where required |
| 2 | `check_instruction_params` | Operand types match opcode expectations |
| 3 | `check_function_call_params` | Argument count matches parameter count |
| 4 | `check_call_function_declared` | Called functions exist in the module |
| 5 | `check_call_var_declared` | Referenced variables exist in scope |
| 6 | `check_function_return_value_type` | Return type matches function signature |

Errors are reported via `vm_module_error_add()` with line and column information.

### 3.6 Type Traits

The `vm_type_trait` system tracks type information during compilation:

```c
typedef struct vm_type_trait_t {
    char *type;              // "number", "string", "boolean"
    bool is_constant;        // Compile-time literal?
    vm_param_type_class type_class;
} vm_type_trait;
```

Traits are linked between variables via `trait_type_link(source, dest)`, propagating type information through the codegen pipeline. This enables the verifier to check type compatibility at call sites.

---

## 4. Bytecode Reference

### Instruction Format

Every bytecode instruction is a `vm_instruction` struct with an opcode, three string operands (each with type metadata and resolution flags), and source-line tracking.

### 4.1 Arithmetic (7 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opADD` | `dest`, `lhs`, `rhs` | `dest = lhs + rhs` (numeric or string concat) |
| `opSUB` | `dest`, `lhs`, `rhs` | `dest = lhs - rhs` |
| `opMUL` | `dest`, `lhs`, `rhs` | `dest = lhs * rhs` |
| `opDIV` | `dest`, `lhs`, `rhs` | `dest = lhs / rhs` |
| `opEXP` | `dest`, `base`, `exp` | `dest = base ^ exp` |
| `opPOW` | `dest`, `base`, `exp` | Synonym for `opEXP` |
| `opNEG` | `dest`, `src`, — | `dest = -src` |

### 4.2 Unary / Increment (3 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opINC` | `var`, —, — | `var += 1` |
| `opDEC` | `var`, —, — | `var -= 1` |
| `opNOT` | `dest`, `src`, — | `dest = !src` |

### 4.3 Variable Operations (4 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opCREATE_SET_VAR` | `name`, `value`, `type` | Declare new variable and assign |
| `opSET_VAR` | `name`, `value`, — | Assign to existing variable |
| `opGET_OBJ_VAR` | `dest`, `obj`, `property` | Read object property |
| `opSET_OBJ_VAR` | `obj`, `property`, `value` | Write object property |

### 4.4 Stack Operations (2 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opSTACK_PUSH` | `value`, —, — | Push value onto evaluation stack |
| `opSTACK_POP` | `dest`, —, — | Pop top of stack into dest |

### 4.5 Function Calls (3 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opCALL_METHOD` | `name`, —, — | Call function defined in same module |
| `opCALL_OBJ_METHOD` | `obj`, `method`, — | Call method on object |
| `opCALL_EXT_METHOD` | `name`, —, — | Call registered native/external function |

### 4.6 Control Flow (4 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opJUMP` | `label`, —, — | Unconditional jump to label |
| `opJUMP_IF_TRUE` | `cond`, `label`, — | Jump if condition is truthy |
| `opJUMP_IF_FALSE` | `cond`, `label`, — | Jump if condition is falsy |
| `opLABEL` | `name`, —, — | Jump target (no-op at runtime) |

### 4.7 Comparison (6 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opCONDITION_EQ` | `dest`, `lhs`, `rhs` | `dest = (lhs == rhs)` |
| `opCONDITION_NE` | `dest`, `lhs`, `rhs` | `dest = (lhs != rhs)` |
| `opCONDITION_GT` | `dest`, `lhs`, `rhs` | `dest = (lhs > rhs)` |
| `opCONDITION_LT` | `dest`, `lhs`, `rhs` | `dest = (lhs < rhs)` |
| `opCONDITION_GTE` | `dest`, `lhs`, `rhs` | `dest = (lhs >= rhs)` |
| `opCONDITION_LTE` | `dest`, `lhs`, `rhs` | `dest = (lhs <= rhs)` |

### 4.8 Logical (3 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opCONDITION_TRUE` | `dest`, `src`, — | `dest = bool(src)` |
| `opCONDITIONAL_AND` | `dest`, `lhs`, `rhs` | `dest = lhs && rhs` |
| `opCONDITIONAL_OR` | `dest`, `lhs`, `rhs` | `dest = lhs \|\| rhs` |

### 4.9 Miscellaneous (3 opcodes)

| Opcode | Params | Semantics |
|--------|--------|-----------|
| `opRETURN` | —, —, — | Return from function (top of stack = return value) |
| `opNOP` | —, —, — | No operation |
| `opCOMMENT` | `line`, `text`, — | Source comment preserved in IL (no effect) |

### 4.10 Reserved / Experimental (3 opcodes)

| Opcode | Params | Status |
|--------|--------|--------|
| `opYIELD` | —, —, — | Not implemented |
| `opXPATH` | `expr`, —, — | Legacy; not implemented |
| `opJSON` | `expr`, —, — | Legacy; not implemented |

### IL Dump Example

Source:
```typescript
function main(): number {
    var x: number = 10;
    var y: number = 20;
    return x + y;
}
```

Compiled IL (via `vm_engine_dump_il()`):
```
  0)  LABEL           main
  1)  CREATE_SET_VAR  x               10              number
  2)  CREATE_SET_VAR  y               20              number
  3)  ADD             _reg0           x               y
  4)  STACK_PUSH      _reg0
  5)  RETURN
```

- **Line 0**: Label marks the function entry point
- **Lines 1–2**: Declare typed variables with initial values
- **Line 3**: Add `x` and `y`, store in compiler-generated register `_reg0`
- **Line 4**: Push the result onto the evaluation stack
- **Line 5**: Return (runtime pops the stack for the return value)

---

## 5. API Reference

### 5.1 High-Level Engine API

#### Lifecycle

```c
#include "microgpt_vm.h"

vm_engine *e = vm_engine_create();

int err = vm_engine_load(e, source_code);
if (err != 0) {
    fprintf(stderr, "Error: %s\n", vm_engine_last_error(e));
}

vm_engine_run(e, "main");

double num      = vm_engine_result_number(e);
const char *str = vm_engine_result_string(e);
int    flag     = vm_engine_result_bool(e);

vm_engine_dispose(e);
```

#### Function Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `vm_engine_create` | `vm_engine* (void)` | Allocate engine. Returns NULL on OOM. |
| `vm_engine_dispose` | `void (vm_engine*)` | Free engine, module, and runtime. |
| `vm_engine_load` | `int (vm_engine*, const char* source)` | Parse + compile. Returns 0 on success. |
| `vm_engine_run` | `int (vm_engine*, const char* fn_name)` | Execute function. Returns 0 on success. |
| `vm_engine_register_fn` | `void (vm_engine*, const char* name, vm_native_fn fn)` | Register native C callback. |
| `vm_engine_result_number` | `double (const vm_engine*)` | Last numeric return value. |
| `vm_engine_result_string` | `const char* (const vm_engine*)` | Last string return value. |
| `vm_engine_result_bool` | `int (const vm_engine*)` | Last boolean return value (0/1). |
| `vm_engine_last_error` | `const char* (const vm_engine*)` | Error string after failed load/run. |
| `vm_engine_dump_il` | `void (const vm_engine*)` | Print compiled IL to stdout. |

### 5.2 Native Function Callbacks

```c
typedef double (*vm_native_fn)(int argc, const double *argv);

static double native_square(int argc, const double *argv) {
    return (argc > 0) ? argv[0] * argv[0] : 0.0;
}

vm_engine *e = vm_engine_create();
vm_engine_register_fn(e, "square", native_square);  // Must be before load()

vm_engine_load(e,
    "function main(): number {\n"
    "    return square(7);\n"
    "}\n");

vm_engine_run(e, "main");
printf("7² = %f\n", vm_engine_result_number(e));  // 49.0
vm_engine_dispose(e);
```

> **Note**: Native functions must be registered *before* `vm_engine_load()` so the parser can resolve them.

### 5.3 Low-Level Module API

```c
// Direct compilation (bypassing vm_engine)
verb_context *ctx = verb_context_create();
vm_module *module = NULL;
result r = vm_module_compile(ctx, source_code, &module);

if (r == RESULT_OK && sequence_count(module->errors) == 0) {
    vm_function *fn = vm_module_fetch_function(module, "main");

    char *il = vm_module_to_string(module);
    puts(il);
    string_free(il);
}

vm_module_dispose(module);
verb_context_dispose(ctx);
```

### 5.4 Runtime Execution

```c
vm_module_runtime *rt = vm_module_runtime_create(module);

vm_function *fn = vm_module_fetch_function(module, "main");
fn->instruction_pointer = 0;

bool ok = vm_module_runtime_run(rt, fn);

vm_variable *result = NULL;
vm_module_runtime_stack_pop(rt, &result);

vm_module_runtime_dispose(rt);
```

### 5.5 Inspecting Functions

```c
vm_function *fn = vm_module_fetch_function(module, "main");

sequence_foreach_of(fn->instructions, vm_instruction*, instr) {
    printf("%s %s %s %s\n",
        vm_instruction_opcode_to_string(instr->opcode),
        instr->param1 ? instr->param1 : "",
        instr->param2 ? instr->param2 : "",
        instr->param3 ? instr->param3 : "");
}

char *vars = vm_function_variables_to_string(fn);
puts(vars);
string_free(vars);
```

### 5.6 Error Handling

```c
int err = vm_engine_load(e, bad_source);
if (err != 0) {
    const char *msg = vm_engine_last_error(e);
    // msg: "Parse errors: [L3:10] unexpected token ..."
}
```

Parse errors include line and column information from the Bison parser.

### 5.7 Build and Link

```cmake
add_library(microgpt_vm_lib STATIC src/microgpt_vm.c ${VM_PARSER_SOURCES})
target_include_directories(microgpt_vm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE microgpt_vm_lib m)
```

Requires Flex and Bison at build time for parser generation.

---

## 6. Internals

### 6.1 `microgpt_vm.c` Layout

```
Section                   ~Lines   Entry Points
────────────────────────────────────────────────────
Engine API                 250     vm_engine_*
Module                     100     vm_module_*
Compiler                    10     vm_module_compile
Code Generator             550     vm_module_generator_*
Post-Processing            400     jump resolution, variable interning
Verifier                   400     type checks, call validation
Module Parser              200     Flex/Bison entry point
Runtime                    700     vm_module_runtime_run  ← largest
Functions                  200     vm_function_*
Instructions               250     vm_instruction_*
Variables                  350     vm_variable_*
Type Traits                100     vm_type_trait_*
```

### 6.2 Runtime Dispatch Loop

The largest function (~700 lines) — a `switch` dispatch loop:

```c
while (ip < instruction_count) {
    instruction = instructions[ip];
    switch (instruction->opcode) {
        case opCREATE_SET_VAR: ...
        case opSET_VAR:        ...
        case opADD:            ...
        // ... 35 cases
    }
    ip++;
}
```

#### Type Dispatch Macro

Arithmetic opcodes use an `op()` macro that pattern-matches on operand type classes:

```c
op(ptcNUMBER, ptcNUMBER, ptcNUMBER, {
    var3->value.number = var1->value.number + var2->value.number;
})
op(ptcSTRING, ptcSTRING, ptcSTRING, {
    // string concatenation
})
op(ptcNUMBER, ptcSTRING, ptcSTRING, {
    // number-to-string coercion + concat
})
```

#### External Method Dispatch

When `opCALL_EXT_METHOD` is encountered:
1. Pop arguments from the stack
2. Invoke `vm_call_ext_method_callback` (registered via `vm_module_runtime_set_call_ext_method_callback`)
3. The callback pushes its return value onto the stack

#### Debug Tracing

When `_DEBUG_TRACE` is defined, each instruction is printed before execution via `vm_instruction_with_var_to_string()`.

### 6.3 Iteration Macros

```c
sequence_foreach_of(func->instructions, vm_instruction*, instr) { ... }
cmap_foreach_of(module->functions, name, vm_function*, fn)     { ... }
queue_foreach_of(gen->code_fragments, code_fragment*, frag)    { ... }
```

---

## 7. Constraints & Limitations

### Language Constraints

| Constraint | Detail |
|-----------|--------|
| **No closures** | Functions cannot capture variables from enclosing scope |
| **No arrays** | No native array type (opaque verb-based arrays only) |
| **No structs/records** | Only primitive types: `number`, `string`, `boolean` |
| **No module system** | No `import`/`export` — single compilation unit only |
| **No `break`/`continue`** | Loops have no early-exit mechanism |
| **No `else if` optimisation** | Nested `else if` generates suboptimal jump chains |
| **No negative literals** | `-42` is parsed as unary minus on `42` (`opNEG`) |
| **Single Greek letter** | Only `α`/`\alpha` is supported in LaTeX notation |
| **No garbage collection** | Lifecycle is manual `dispose()` — caller must free |

### Runtime Constraints

| Constraint | Value | Impact |
|-----------|-------|--------|
| **Maximum stack depth** | 100 (`VM_MAX_STACK`) | No overflow check — exceeding causes buffer overwrite |
| **Single-threaded parser** | Global `_vm_ctx_current_parser` | Cannot parse concurrently |
| **Fixed operand count** | 3 per instruction | All opcodes use exactly 3 string operands |
| **No tail-call optimisation** | — | Deep recursion will exhaust the stack |
| **Static return buffer** | `vm_variable_to_string` uses `static char[64]` | Not thread-safe |

### Type System Constraints

| Constraint | Detail |
|-----------|--------|
| **4 primitive types** | `boolean`, `number` (double), `string` (char*), `other` (opaque void*) |
| **Type inference via traits** | Type propagation is heuristic, not formally verified |
| **No type coercion errors** | `string + number` silently coerces — no type error |
| **No generics** | No parameterised types |

---

## 8. Examples & Test Fixtures

Test fixtures are in `tests/resources/tests/vm/`. Compiler tests use `.ts` → `.il` golden file comparison. Runtime tests use `.ts` → `.output` expected value comparison.

### 8.1 Arithmetic — `runtime_math1.ts`

```typescript
function main() {
    var amount = 100;
    amount = amount + 1.23 - 1;
    var amount_is_correct = amount == 100.23;
    var total = (10 * 17.5) + 100 - 5 - (2 + 3);
    var sum = 100 - 2000;
    var value1 = 100 / 2;
    var value2 = 100 / 2.5;
    var value3 = 2.3 * 1.2;
    var value4 = 10 * (2.3 * 1.2);
}
```

**Expected output:**
```
amount=100.23
amount_is_correct=true
total=265
sum=-1900
value1=50
value2=40
value3=2.76
value4=27.6
```

### 8.2 Conditionals — `runtime_conditions1.ts`

```typescript
function main() {
    var amount = 100;
    var a = 0; var b = 0; var c = 0; var d = 0; var e = 0;
    var a1 = 0; var b1 = 0; var f = 0; var g = 0;

    if (amount > 10) {
        a = 1;
        if (amount >= 100) { b = 1; }
        else { b1 = 1; }
    } else { a1 = 1; }

    if (amount - 1 == 99) { c = 1; }
    if (amount < 200) { if (amount <= 100) { d = 1; } }
    if (amount > 100) { e = 1; }
    if (amount < 1000) { f = 1; f = f + 1; }
    else { g = 1; }
}
```

**Expected output:**
```
amount=100  a=1  b=1  c=1  d=1  e=0  a1=0  b1=0  f=2  g=0
```

### 8.3 Strings — `runtime_strings1.ts`

```typescript
function main(): string {
    var firstname = "Bob";
    var lastname = "Fish";
    var greeting = "Hello " + firstname + " " + lastname;
    var age = 18;
    var greeting2 = "You are " + age;
    var fullname = firstname;
    fullname = fullname + ", ";
    fullname = fullname + lastname;
    var greeting3 = "You are " + age + " is " + fullname;
    return "Reply " + greeting3;
}
```

**Expected output:**
```
firstname=Bob
lastname=Fish
greeting=Hello Bob Fish
age=18
greeting2=You are 18
fullname=Bob, Fish
greeting3=You are 18 is Bob, Fish
```

### 8.4 Loops — `runtime_loop2.ts`

```typescript
function main() {
    var count = 0;
    for (var i = 0; i < 10; i++) {
        count++;
    }
    var total = 0;
    for (var j = 0; j < 10; j++) {
        total += j;
    }
}
```

### 8.5 Multi-Function Calls — `runtime_simple1.ts`

```typescript
function calculate_interest3(amount: number, rate: number): number {
    return amount * rate;
}
function add_bonus(amount: number): number {
    return amount + 100;
}
function main(): number {
    var amount = 10;
    amount = 11;
    amount = amount + 120;
    var total = add_bonus(calculate_interest3(amount, 2));
    return total;
}
```

### 8.6 LaTeX Math — `runtime_latex_math1.ts`

```latex
a1() = (123 + 1 + (2 * 2))
a(x) = x + 1
b(x, y) = x + y;
c(x, y) = x ^ 2 + y ^ 3
d(\alpha) = \alpha + 1
e() = \sum_{i=0}^{20} i + 3;
f() = \sum_{i=0}^{20} (i + 3) + i
g() = \sum_{i=0}^{10} \sum_{j=0}^{5} i + j
h(n) = \sum_{i=0}^{n} \sum_{j=0}^{n} i + j
k(n) = \sum_{i=0}^{10} \sum_{j=0}^{(n+1)} i + j
m(x, n) = \sum_{i=0}^{10} \sum_{j=0}^{(n+1)} i + j * x
```

**Expected output:**
```
result_a1=128     result_ax=6       result_b=8
result_c=31       result_d=10       result_e=250
result_f=440      result_g=1375     result_h=250
result_k=1815     result_m=2640
```

### 8.7 External Functions — `gen_body2.ts`

```typescript
declare function debug(value: number) : void;

function get_rate(country: string): number {
    return 1.34;
}
function calculate_interest2(amount: number, country: string): number {
    return amount * get_rate() * global_rate;
}
function main() {
    var amount = 100;
    amount = amount + 100;
    var interest = calculate_interest2(amount, "UK");
    debug(interest);
}
```

### 8.8 While Loops — `gen_loop1.ts`

```typescript
function main() {
    var total = 100;
    while (total > 0) {
        total = total - 1;
    }
}
```

### 8.9 LaTeX Fractions — `gen_arithmetic1.tex`

```latex
f1(x) = 1
f2(x) = x + 1
f3(x, y) = x + y
f4(x, y) = x^2 + y^3
MSE(n) = \frac{1}{n}
```

### Test Coverage Summary

| Suite | Fixture Count | Description |
|-------|--------------|-------------|
| Compiler tests | 55 `.ts` → `.il` pairs | IL output matches golden bytecode |
| Runtime tests | 34 `.ts` + `.output` pairs | Execution results match expected values |
| LaTeX tests | 3 `.tex` files | Mathematical expression compilation |
| **Total** | **92 fixtures** | |

---

## 9. Known Issues

### Critical

| ID | Issue | Impact |
|----|-------|--------|
| **ISS-1** | Global parser state (`_vm_ctx_current_parser`, `vm_tmp_var`, `vm_tmp_label`) | Two threads cannot parse simultaneously; nested `vm_engine_load()` corrupts state |
| **ISS-2** | No stack overflow check (`VM_MAX_STACK = 100`) | Push past 100 causes buffer overwrite |

### High

| ID | Issue | Status |
|----|-------|--------|
| **ISS-3** | String null-terminator missing in lexer | ✅ Fixed |
| **ISS-4** | `RESULT_OK` defined twice in `microgpt_vm.c` | Open |
| **ISS-5** | Duplicate `#include` directives in `.l` and `.y` | ✅ Fixed |

### Medium

| ID | Issue | Detail |
|----|-------|--------|
| **ISS-6** | `opEXP` and `opPOW` are synonyms | One should be removed or aliased |
| **ISS-7** | `vm_module_runtime_run` is 700+ lines | Should be split into per-opcode handlers |
| **ISS-8** | String ownership unclear | No consistent `const char*` convention |
| **ISS-9** | Comment regex simplified | Block comments need separate Flex start condition for robustness |
| **ISS-15** | Duplicate merge calls in `vm_module_generator_defer_pop` | Likely copy-paste error |
| **ISS-16** | `vm_variable_to_string` uses `static char[64]` | Not thread-safe |

### Low

| ID | Issue | Detail |
|----|-------|--------|
| **ISS-10** | `vm_tmp_var` reused without clearing | Fragile if rules reordered |
| **ISS-11** | Dead opcodes: `opYIELD`, `opXPATH`, `opJSON` | Declared but no runtime implementation |
| **ISS-12** | LaTeX support limited to `\alpha` only | β, γ, δ, θ, π not tokenised |
| **ISS-13** | Negative number literals not supported | `-42` parsed as `NEG 42` |
| **ISS-14** | Debug printf in `vm_function_dispose` | ✅ Fixed |

---

## 10. Roadmap

### Phase 1 — Housekeeping (Near-Term)

| Item | Description |
|------|-------------|
| `vm_` prefix rename | Rename support functions to avoid namespace collisions |
| `const char*` consistency | Change input parameters from `char*` to `const char*` |
| Unified error codes | Replace `bool`/`result`/`int` with single `vm_result` enum |
| Remove dead opcodes | `opYIELD`, `opXPATH`, `opJSON` |
| Consolidate `RESULT_OK` | Move to header, remove both `.c` defines |
| Global parser state | Document thread-safety limitation at minimum |

### Phase 2 — Engine Improvements (Medium-Term)

| Item | Complexity |
|------|-----------|
| Dynamic stack (replace fixed 100) | Medium |
| `break`/`continue` for loops | Medium |
| String interning (pool identical constants) | Low |
| Register allocator (reuse out-of-scope regs) | Medium |
| Better error recovery (multi-error reporting) | Medium |
| Negative number literals | Low |
| Multi-character Greek letters | Low |
| `else if` chain optimisation | Medium |

### Phase 3 — Advanced Features (Long-Term)

| Item | Complexity |
|------|-----------|
| Computed-goto dispatch (~30% speedup) | Medium |
| WASM compilation target | High |
| Expanded type system (arrays, records) | High |
| Module system (`import`/`export`) | High |
| Closures (scope chains) | High |
| Garbage collector | High |
| Debug protocol (breakpoint/step/inspect) | High |
| Parser rewrite (recursive descent) | High |

---

## 11. Source Files

| File | Lines | Role |
|------|-------|------|
| [`microgpt_vm.h`](../../src/microgpt_vm.h) | 1 470 | Self-contained header: types, enums, prototypes, support layer |
| [`microgpt_vm.c`](../../src/microgpt_vm.c) | 3 675 | All implementation: engine, compiler, generator, runtime |
| [`microgpt_vm.l`](../../src/microgpt_vm.l) | 147 | Flex lexer (in-memory string input, no files) |
| [`microgpt_vm.y`](../../src/microgpt_vm.y) | 478 | Bison grammar (direct bytecode emission, no AST) |
| [`microgpt_vm_parser.l.c`](../../src/microgpt_vm_parser.l.c) | — | Generated lexer (Flex output) |
| [`microgpt_vm_parser.tab.c`](../../src/microgpt_vm_parser.tab.c) | — | Generated parser (Bison output) |

---

## 12. Verb System Enhancement — Design Specification

### 12.1 Background and Motivation

The VM currently has two separate mechanisms for native function dispatch:

1. **`vm_engine_register_fn`** — A simple `name → callback` registration used by the high-level engine API. The callback signature is `double (*vm_native_fn)(int argc, const double *argv)` — numeric-only, no string or opaque handle support.

2. **`verb_context` stub** (`microgpt_vm.h` lines 600–662) — A minimal implementation providing `verb_register`, `verb_find`, `verb_exists`, and the `verb(fn)` / `verb_arg(name)` macros. **Missing**: `verb_compile`, `verb_exec`, `verb_unregister`, `verb_enum`, `verb_definition_dispose`, `verb_compiled` struct, error codes, and the `definition_params`/`name_length` fields from the full verb definition.

The `opCALL_EXT_METHOD` runtime handler (line 2633 of `microgpt_vm.c`) uses a raw `vm_call_ext_method_callback` function pointer — it does **not** route through the verb system at all.

Meanwhile, the **full verb system** (`experiments/engines/microgpt-verb/main/src/verb.cpp`, 756 lines) provides:

- **`verb_compile`** (416 lines) — Sentence parser supporting quoted strings, JSON objects/arrays, XML, and space-delimited parameters
- **`verb_exec`** — Command dispatch: compile → longest-match verb lookup → invoke function → return response
- **`verb_unregister`** — Dynamic verb removal
- **`verb_enum`** — Verb discovery (iterate all registered verbs)
- **`verb_definition_dispose`** — Proper cleanup of verb definitions including name, params, and param_list
- **`verb_result`** — Human-readable error code strings
- **Error codes** — `RESULT_CORE_VERB_ERROR_NO_MATCH`, `_ALREADY_REGISTERED`, `_ARGUMENT_NULL`, `_INCORRECT_USAGE`, `_ARGUMENT_PARSING`, etc.
- **22 unit tests** covering all paths

### 12.2 Gap Analysis: VM Stub vs Full Verb System

| Feature | VM Stub (current) | Full Verb System | Gap |
|---------|-------------------|------------------|-----|
| `verb_register` | ✅ Basic (strtok-based param split) | ✅ Full (usage info generation) | Missing `definition_params`, `name_length` |
| `verb_find` | ✅ Simple cmap lookup | ✅ Returns `verb_definition*` | Signature mismatch (stub returns `void*`) |
| `verb_exists` | ✅ | ✅ | — |
| `verb_compile` | ❌ Missing | ✅ 416-line parser | **Critical gap** |
| `verb_exec` | ❌ Missing | ✅ Compile + dispatch + error handling | **Critical gap** |
| `verb_unregister` | ❌ Missing | ✅ With cleanup | Missing |
| `verb_enum` | ❌ Missing | ✅ Returns cmap | Missing |
| `verb_definition_dispose` | ❌ Missing | ✅ Full cleanup chain | **Memory leak** |
| `verb_compiled` struct | ❌ Missing | ✅ Holds parse result + args | **Critical gap** |
| Error codes | ❌ Missing | ✅ 7 codes | Missing |
| `definition_params` field | ❌ Missing | ✅ Usage string e.g. `<name> <message>` | Missing |
| `name_length` field | ❌ Missing | ✅ Cached for O(1) prefix match | Missing |
| Quoted string params | ❌ | ✅ Single + double quotes | Missing |
| JSON params | ❌ | ✅ Objects + arrays with nesting | Missing |
| XML params | ❌ | ✅ Tag matching + short tags | Missing |
| `opCALL_EXT_METHOD` verb routing | ❌ Raw callback | N/A | **Integration gap** |

### 12.3 Proposed Changes

#### Design Principle

Port the verb system **into** the existing `microgpt_vm.h`/`.c` files as inline/static functions — keeping the VM self-contained (no external dependencies). The verb system is pure C with no C++ features (the `.cpp` extension in the source tree is misleading — the code is C compatible).

#### Phase 1: Complete the Verb Stub (~250 lines)

**[MODIFY] `microgpt_vm.h`** — Replace the stub verb section (lines 600–662) with the complete verb system:

```c
/* ── verb_context (full DSL dispatch layer) ── */

// Error codes
#define RESULT_CORE_VERB_ERROR              2299
#define RESULT_CORE_VERB_ERROR_NO_MATCH     2300
#define RESULT_CORE_VERB_ERROR_ALREADY_REGISTERED 2301
#define RESULT_CORE_VERB_ERROR_ARGUMENT_NULL 2302
#define RESULT_CORE_VERB_ERROR_INCORRECT_USAGE 2303
#define RESULT_CORE_VERB_ERROR_ARGUMENT_PARSING 2304
#define RESULT_CORE_VERB_ERROR_EXEC_NOT_ENOUGH_PARAMS 2306

// Full verb_definition with all fields
typedef struct verb_definition_t {
    const char *name;
    const char *definition_params;     // NEW: "<param1> <param2>" usage string
    size_t name_length;                // NEW: cached strlen for prefix match
    sequence *param_list;
    size_t params_count;
    verb_function function;
    void *fcontext;
    struct verb_definition_t *next_verb_definition; // legacy list pointer
} verb_definition;

// verb_compiled — parse result holder
typedef struct verb_compiled_t {
    int result;
    verb_definition *verb_definition_;
    cmap *verb_arg_list;
    char *sentence_values;
    void *context;
} verb_compiled;
```

New functions to add:
- `verb_compile()` — Port the 416-line parser from `verb.cpp` as a `static inline` or as a regular function in `microgpt_vm.c`
- `verb_exec()` — Compile + dispatch + error formatting
- `verb_unregister()` — Remove + dispose verb by name
- `verb_enum()` — Return verb definition map
- `verb_definition_dispose()` — Full cleanup chain
- `verb_compile_dispose()` — Free compiled sentence + arg map
- `verb_result()` — Error code → string

**[MODIFY] `microgpt_vm.c`** — Wire `opCALL_EXT_METHOD` to use `verb_exec`:

```c
case opCALL_EXT_METHOD:
    // Build a verb sentence from function name + stack args
    // Call verb_exec(module->verb_context_, sentence, &response)
    // Push response back onto stack as vm_variable
    break;
```

#### Phase 2: Add Verb Unit Tests (~10 new test cases)

**[NEW] `tests/resources/tests/vm/runtime/runtime_verb1.ts`** — Script using declared external functions:

```typescript
declare function square(x: number): number;
declare function greet(name: string): string;

function main(): number {
    var result = square(7);
    var message = greet("World");
    return result;
}
```

**[MODIFY] `tests/test_microgpt_vm.c`** — Add verb-specific test functions:

| Test | What It Validates |
|------|-------------------|
| `test_verb_register_and_find` | Register verb, find it, check params_count |
| `test_verb_register_duplicate` | Second register returns `ALREADY_REGISTERED` |
| `test_verb_exec_simple` | Execute `"greet World"`, check response |
| `test_verb_exec_missing_params` | Execute with no params → `INCORRECT_USAGE` + usage string |
| `test_verb_exec_not_found` | Execute unregistered verb → `NO_MATCH` |
| `test_verb_unregister` | Register, unregister, verify gone |
| `test_verb_exec_quoted_params` | Execute with `'quoted string'` params |
| `test_verb_exec_with_vm_script` | Full integration: register verb + load .ts + run → verb callback fires |
| `test_verb_enum` | Register 3 verbs, enumerate, verify all present |
| `test_verb_definition_dispose` | Register + dispose → no memory leaks |

#### Phase 3: Wire `opCALL_EXT_METHOD` Through Verb System

Currently the `opCALL_EXT_METHOD` handler calls a raw function pointer:

```c
runtime->call_ext_method_callback(runtime, function);
```

The enhancement routes this through `verb_exec`:

1. When `opCALL_EXT_METHOD` fires, pop arguments from the stack
2. Build a verb sentence: `"function_name arg1 arg2 ..."`
3. Call `verb_exec(module->verb_context_, sentence, &response)`
4. Parse response back into a `vm_variable` and push onto stack
5. If `verb_exec` fails, set runtime error

This preserves backward compatibility — the `vm_call_ext_method_callback` path is kept as a fallback when no verb_context is set.

### 12.4 Verb Compile Parser — What Gets Ported

The `verb_compile` function is the most complex piece (416 lines). It handles:

| Feature | Lines in verb.cpp | Difficulty |
|---------|-------------------|-----------|
| Longest-match verb lookup | 278–298 | Low — iterate cmap, prefix match |
| Space-delimited params | 346–348, 546–548 | Low — split on spaces |
| Single-quoted strings | 380–383, 413–415 | Low — track quote state |
| Double-quoted strings | 385–388, 417–418 | Low — same pattern |
| JSON arrays `[...]` | 390–392, 421–454 | Medium — nested depth tracking |
| JSON objects `{...}` | 395–398, 458–493 | Medium — same pattern |
| XML tags `<.../...>` | 400–404, 495–544 | High — tag matching, short tags, depth |
| Quote stripping | 594–601 | Low — adjust start/end pointers |
| Error: partial quotes | 566–571 | Low — check depth at EOL |
| Error: missing params | 642–660 | Low — count vs expected |

**Decision:** Port the full parser including XML support. The ~416 lines translate cleanly to C99 (the verb.cpp is already C-compatible code despite the .cpp extension — the only C++ construct is `nullptr` which becomes `NULL`).

### 12.5 Implementation Estimate

| Phase | Files | Lines Changed | Effort |
|-------|-------|--------------|--------|
| Phase 1: Complete verb stub | `microgpt_vm.h`, `microgpt_vm.c` | ~550 added, ~60 replaced | Medium |
| Phase 2: Unit tests | `test_microgpt_vm.c`, 1 new `.ts` fixture | ~200 added | Low |
| Phase 3: Wire `opCALL_EXT_METHOD` | `microgpt_vm.c` | ~40 changed | Low |
| **Total** | 4 files | ~790 lines | **~4 hours** |

### 12.6 Verification Plan

1. **Build** — `cmake --build build` passes with no warnings
2. **Existing tests pass** — All 92 existing fixtures (55 compiler + 34 runtime + 3 LaTeX) unchanged
3. **New verb tests pass** — 10 new test cases all green
4. **Integration test** — `.ts` script calls declared external function → verb callback fires → return value available
5. **Memory check** — No leaks in verb lifecycle (register → exec → unregister → dispose)

### 12.7 Connection to ORGANELLE_GENERALISATION_VM.md

This enhancement directly enables **Phase 1** of the `vm_compose` pipeline described in `ORGANELLE_GENERALISATION_VM.md`:

- The **Opaque Array Handle** model (§2.2) requires `verb_exec` to dispatch array operations from VM scripts to native C functions
- The `vm_wiringgen` corpus (Phase 1) needs scripts that call registered verbs like `rolling_mean(input_signal, 10)` — this only works with a full verb system
- The deterministic syntax gate (§3) uses `vm_module_parser` — but the *semantic* gate needs `verb_exists` to validate function names at compile time
- The **vm_compose pipeline** (§3) replaces `gcc -fsyntax-only` with `vm_module_compile` — the verb system ensures declared external functions are properly resolved

### 12.8 Implementation Status ✅

**Completed:** February 2026

#### Phase 1: Full Verb System — Complete

Ported ~470 lines from `microgpt-verb/verb.cpp` into `microgpt_vm.c`:

| Function | Purpose |
|----------|---------|
| `verb_register` | Register verb with param parsing and usage string |
| `verb_compile` | 280-line sentence parser (quoted strings, JSON, XML) |
| `verb_exec` | Compile → dispatch → error handling |
| `verb_unregister` | Remove verb from context |
| `verb_definition_dispose` / `verb_context_dispose` / `verb_compile_dispose` | Memory cleanup |
| `verb_result` | Error code → string |

Header additions to `microgpt_vm.h`: `verb_compiled` struct, 6 error codes, `cmap_remove`, `SENTENCE_MESSAGE_MAX_SIZE`.

#### Phase 2: Unit Tests — Complete

| Suite | Tests | Status |
|-------|-------|--------|
| `vm_verb_tests` | 10 (register, exec, unregister, quoted, enum, etc.) | ✅ |
| `vm_verb_opaque_handle_tests` | 4 (create_signal, rolling_mean, full pipeline, error handling) | ✅ |
| **All VM tests** | **59 passed, 0 failed** | ✅ |

#### Phase 3: Opaque Handle Worked Example — Complete

This is the concrete example requested by the gap analysis: *"how `rolling_mean(input_signal, 10)` would be wired as a native callback returning an opaque handle"*.

**Architecture** (as described in `ORGANELLE_GENERALISATION_VM.md` §2.2):
- Native C functions allocate arrays and return integer handle IDs as strings
- The VM (or verb caller) passes handles between verbs as opaque scalars
- All array manipulation happens inside native C callbacks — the VM never sees `arr[i]`

**Verbs implemented in test suite** (`test_microgpt_vm.c`):
```
create_signal <length>        → allocates double[length], returns handle "0"
rolling_mean  <handle> <window> → computes smoothed signal, returns new handle "1"
signal_value_at <handle> <index> → reads value at index, returns "9.000000"
signal_length <handle>         → returns element count
```

**Example pipeline** (from `should_opaque_handle_full_pipeline`):
```
create_signal 20       → handle "0"  (raw signal [1..20])
rolling_mean 0 5       → handle "1"  (smoothed signal)
signal_value_at 1 10   → "9.000000"  (mean of values 7,8,9,10,11)
```

#### Benchmark Results

| Benchmark | Throughput |
|-----------|-----------|
| `benchmark_simple_verb_call_single_param` | **2,665,293/s** |
| `benchmark_opaque_handle_rolling_mean` (3 verb calls per iteration) | **260,525/s** |
| For comparison: `benchmark_simple1` (VM bytecode runtime) | 1,128,142/s |

