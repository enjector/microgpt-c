# libpipeline_ir

A standalone C99 library implementing the **Pipeline IR**: a typed
directed dataflow graph (`Pipeline`, `PipelineNode`, `PipelineEdge`,
`PipelineType`) with a strict verifier, a tolerant parser, a graph
repair pass, a round-trip-safe `@graph...@end` text format, and a
GraphViz DOT renderer.  Zero dependencies beyond libc.

The library was extracted from
[MicroGPT-C](https://github.com/enjector/microgpt-c)'s `src/` tree by
[Experiment E02](../../experiments/E02-pipeline-ir-library.md) so that
neurosymbolic systems built outside that project can use the same
deterministic post-hoc Judge surface — including frontier-LLM tool
calls — without depending on the MicroGPT-C transformer engine.

## Status

| Field | Value |
|---|---|
| Version | 0.1.0 (pre-v1, experimental) |
| Language | C99, libc + libm only |
| License | MIT (see `../LICENSE` in the parent repo) |
| Header | `#include <pipeline_ir/pipeline_ir.h>` |
| Library | `libpipeline_ir.a` (static, ~60 KB stripped on -O3) |
| Test gate | All 55 in-tree unit tests pass against the extracted library |

ABI stability: the macros `PIPELINE_IR_API_VERSION_{MAJOR,MINOR,PATCH}`
in the public header track semver.  Pre-1.0, the API is "experimental
but functional for the symbols listed below"; the v1.0 cut will lock
the surface.

## Build

### Standalone (just the library + examples)

```bash
cmake -S libs/pipeline_ir -B build_pir -DCMAKE_BUILD_TYPE=Release
cmake --build build_pir --parallel
./build_pir/examples/pipeline_ir_example_custom_generator
```

### From a consumer's CMake project via FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
  pipeline_ir
  GIT_REPOSITORY https://github.com/enjector/microgpt-c.git
  GIT_TAG        main
  SOURCE_SUBDIR  libs/pipeline_ir
)
set(PIPELINE_IR_BUILD_EXAMPLES OFF CACHE BOOL "")
FetchContent_MakeAvailable(pipeline_ir)

add_executable(my_judge src/main.c)
target_link_libraries(my_judge PRIVATE pipeline_ir)
```

### From a vendored copy

```cmake
add_subdirectory(third_party/pipeline_ir)
target_link_libraries(my_target PRIVATE pipeline_ir)
```

## Usage example

Build a graph from a C struct, verify it, print the canonical text +
DOT form:

```c
#include <pipeline_ir/pipeline_ir.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    Pipeline *p = pipeline_create("demo");

    /* Signature: (x:int, y:int) -> (result:int) */
    const char *sig_in_names[]  = { "x", "y" };
    PipelineType *sig_in_types[] = {
        pipeline_type_int(), pipeline_type_int()
    };
    const char *sig_out_names[] = { "result" };
    PipelineType *sig_out_types[] = { pipeline_type_int() };
    pipeline_set_signature(p,
        2, sig_in_names, sig_in_types,
        1, sig_out_names, sig_out_types);

    /* add(a:int, b:int) -> sum:int */
    const char *add_in_names[]   = { "a", "b" };
    PipelineType *add_in_types[]  = {
        pipeline_type_int(), pipeline_type_int()
    };
    const char *add_out_names[]  = { "sum" };
    PipelineType *add_out_types[] = { pipeline_type_int() };
    pipeline_add_node(p, "add1", "add",
        2, add_in_names, add_in_types,
        1, add_out_names, add_out_types);

    pipeline_connect_signature_in (p, "x", "add1", "a");
    pipeline_connect_signature_in (p, "y", "add1", "b");
    pipeline_connect_signature_out(p, "add1", "sum", "result");

    if (pipeline_verify(p) != PIPE_OK) {
        fprintf(stderr, "verify FAILED: %s\n", pipeline_last_error());
        pipeline_free(p);
        return 1;
    }

    char *txt = pipeline_render_text(p);
    printf("%s\n", txt);
    free(txt);
    pipeline_free(p);
    return 0;
}
```

See [`examples/custom_generator/main.c`](examples/custom_generator/main.c)
for a runnable version.

## Parsing LLM-emitted graphs

The intended frontier-LLM use case:

```c
const char *llm_output = receive_from_model();          /* @graph...@end */

Pipeline *p = pipeline_parse_text(llm_output);          /* strict */
if (!p)
    p = pipeline_parse_text_tolerant(llm_output);       /* dedupe + auto-promote */

PipelineRepairReport rep = {0};
pipeline_repair(p, &rep);                               /* drop fragments */

int rc = pipeline_verify(p);                            /* certify */
/* PIPE_OK => safe to dispatch; PIPE_ERR_* => actionable error in pipeline_last_error() */
```

Each layer is a pure subtraction or deduplication pass; nothing
fabricates structure that the model didn't emit.

## Public API stability classification

| Category | Symbol(s) | Stability |
|---|---|---|
| Versioning | `PIPELINE_IR_API_VERSION_MAJOR`, `_MINOR`, `_PATCH` | Stable |
| Type constructors | `pipeline_type_void/int/float/string/any/list/tensor/record` | Stable |
| Type ops | `pipeline_type_clone`, `pipeline_type_free`, `pipeline_type_equal`, `pipeline_type_format` | Stable |
| Lifecycle | `pipeline_create`, `pipeline_free` | Stable |
| Builder | `pipeline_add_node`, `pipeline_add_subgraph`, `pipeline_connect`, `pipeline_set_signature`, `pipeline_connect_signature_{in,out}` | Stable |
| Config | `pipeline_node_set_config_{int,float,string}` | Stable |
| Verifier | `pipeline_verify`, `pipeline_verify_partial` | Stable |
| Repair | `pipeline_repair`, `PipelineRepairReport` | Stable |
| Errors | `pipeline_last_error`, `PIPE_OK`, `PIPE_ERR_*` | Stable |
| Values | `PipelineValue`, `pipeline_value_clear` | Stable |
| Execute (callback) | `pipeline_execute`, `PipelineDispatchFn` | Stable |
| Execute (VM) | `pipeline_execute_vm` | Experimental — depends on opt-in `pipeline_ir_vm.c` TU + caller-supplied `vm_engine`; ABI may evolve |
| Text I/O | `pipeline_render_text`, `pipeline_parse_text`, `pipeline_parse_text_tolerant` | Stable (text format frozen by `INV-PIPE-002`) |
| DOT renderer | `pipeline_render_dot` | Stable |
| Internal | symbols in `src/pipeline_ir_internal.h` (`mgpt_pipe_set_err`, `mgpt_pipe_find_incoming_edge`, `MGPT_PIPE_SIG_*_NODE`) | Internal — not installed, no stability guarantee |

"Stable" here means: the symbol is part of the v0.1 contract and will
not be removed or have its signature changed without a major-version
bump.  "Experimental" means: the symbol is shipped but may change
shape before v1.0.  "Internal" means: do not link against it from
user code.

## Threading

All `pipeline_*` functions are re-entrant per `Pipeline*` (no shared
mutable global state in the IR core); however, `pipeline_last_error()`
returns a thread-local buffer on most platforms but a static buffer
when `_Thread_local`/`__thread` is unavailable.  Treat it as
single-thread-only for portability.

## Repository layout

```
libs/pipeline_ir/
├── include/pipeline_ir/
│   └── pipeline_ir.h            # public ABI
├── src/
│   ├── pipeline_ir.c            # IR + verifier + parsers + repair + DOT
│   ├── pipeline_ir_internal.h   # private; NOT installed
│   └── pipeline_ir_vm.c         # opt-in VM dispatcher (requires vm_engine)
├── examples/
│   ├── CMakeLists.txt
│   └── custom_generator/
│       └── main.c               # ~140 LOC working example
├── CMakeLists.txt               # builds pipeline_ir target
└── README.md                    # this file
```

## See also

- `experiments/E02-pipeline-ir-library.md` — the pre-registered
  experiment that produced this extraction; Section 3 holds the
  measurement results (T1–T6).
- `docs/research/RESEARCH_PIPELINE_IR.md` (parent repo) — the full
  history of the IR's design, including the Phase 1–17 evolution and
  the leakage audit that re-grounded the wiring-organelle claims.
- `tests/test_microgpt_pipeline.c` (parent repo) — the 55-test
  acceptance suite that gates extraction work.
