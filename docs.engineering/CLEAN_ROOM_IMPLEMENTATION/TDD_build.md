# TDD_build — Technical Design Document (Build system + feature flags)

**Document ID:** TDD-BUILD-001
**Version:** 1.0
**Status:** DRAFT
**Sources:** `CMakeLists.txt`, `bootstrap.sh`, `bootstrap.bat`, `docs/BUILD_OPTIONS.md`.

## 1. Overview

The build system uses CMake ≥ 3.10. It produces:
- A core static library `microgpt_lib` with the default feature flags.
- One additional `microgpt_lib_<md5>` static library variant per unique combination of compile-time `-D` overrides used by demos.
- One executable per registered `add_demo(...)` invocation.
- Test binaries (`test_*`) registered via `add_test`.
- Benchmark binaries (`bench_*`) without assertions.

The compile-time architecture model (`N_EMBD`, `N_LAYER`, etc. as macros) makes the variant scheme necessary — different demos compile against different macros, and the constant-folding requires distinct object files. The library-variant cache ensures two demos with identical macro sets share one library target.

## 2. Architecture

```
   bootstrap.sh / .bat
        │
        ▼
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
        │
        ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ root CMakeLists.txt (≈ 2,200 lines)                               │
   │                                                                    │
   │  • feature options (MICROGPT_SIMD, _USE_FLOAT, _BLAS, _METAL, ...)│
   │  • base library: microgpt_lib                                      │
   │  • _microgpt_lib_for_defines(OUT_VAR, DEFS...)                    │
   │       — sorts DEFS, MD5-hashes, creates                            │
   │         microgpt_lib_<hash> if absent, returns target name         │
   │  • add_demo(NAME ... SOURCE ... [THREADS METAL BLAS]              │
   │             [COPY data1 data2 ...] [DEFINES K=V ...])              │
   │       — links against the right variant, adds POST_BUILD copy steps│
   │  • per-test add_test() registrations                               │
   └───────────────────────────────────────────────────────────────────┘
        │
        ▼
   build/
        ├── microgpt_lib(.a / .lib)
        ├── microgpt_lib_<hash>(.a / .lib) for each demo's macro set
        ├── test_microgpt, test_microgpt_msa, test_microgpt_pipeline, ...
        ├── bench_microgpt, bench_microgpt_msa, ...
        └── <demo binaries> + their copied data files
```

## 3. Feature flags

CMake `option()`s map directly to compile-time `-D` definitions on the library variant:

| Option | Default | Effect |
|---|---|---|
| `MICROGPT_SIMD` | ON | `-march=native` (gcc/clang), `/arch:AVX2` (MSVC) |
| `MICROGPT_USE_FLOAT` | ON | `scalar_t = float` (32-bit) |
| `MICROGPT_HEAD_PARALLEL` | OFF | Parallelise attention heads at inference |
| `MICROGPT_PAGED_KV` | OFF | Demand-paged KV cache |
| `QUANTIZATION_INT8` | OFF | INT8 quantised weight storage |
| `MICROGPT_BLAS` | OFF | BLAS-accelerated `lin_fwd` / `lin_bwd` (Accelerate / OpenBLAS / MKL) |
| `MICROGPT_METAL` | OFF | Apple Metal GPU bridge |
| `MICROGPT_ATTN_RES` | OFF | Block Attention Residuals |
| `MICROGPT_ATTN_SINK` | OFF | DeepSeek-V4 attention sink (`ATTN_SINK_LOGIT` magnitude) |
| `MICROGPT_QK_NORM` | OFF | DeepSeek-V4 Q/K RMSNorm pre-dot |
| `MICROGPT_PARTIAL_ROPE` | OFF | DeepSeek-V4 partial RoPE (`ROPE_DIMS`, `ROPE_BASE`) |
| `ENABLE_TURBOQUANT` | OFF | TurboQuant 4-bit KV |
| `ENABLE_ROTORQUANT` | OFF | RotorQuant 4-bit KV |
| `MSA_POOL_MODE` | 0 | MSA pool weighting (0..3) |

Architecture macros (`N_EMBD`, `N_HEAD`, `N_LAYER`, `BLOCK_SIZE`, `MLP_DIM`, `NUM_STEPS`, etc.) are passed via `add_demo(... DEFINES ...)`, NOT via top-level CMake options — each demo bakes its own architecture into its library variant.

## 4. Library variant cache

`_microgpt_lib_for_defines(OUT_VAR DEF1 DEF2 ...)`:

1. If `DEFS` is empty, return `microgpt_lib` (the base target).
2. `list(SORT _defs)` for canonical ordering.
3. `string(MD5 _hash "${_raw_suffix}")` to fingerprint.
4. `set(_tgt "microgpt_lib_${_hash}")`.
5. If the target already exists, return it.
6. Otherwise:
   - Create a new STATIC library with the same source files.
   - Propagate base feature flags (`INTERFACE_COMPILE_DEFINITIONS`).
   - Add per-demo `target_compile_definitions(...)` for each `DEFS` entry.
   - Propagate compile options (e.g. `-march=native`).

The MD5 hash makes the variant identifier deterministic and short. Re-running cmake with no changes creates no new variants.

## 5. `add_demo` macro

```cmake
add_demo(NAME      target_name
         SOURCE    path/to/main.c
         [THREADS]                    # link pthreads (POSIX)
         [METAL]                      # link Metal + Foundation; copy shader
         [BLAS]                       # link BLAS framework
         [COPY     file1 file2 ...]   # POST_BUILD: copy data files next to binary
         [DEFINES  N_EMBD=128 N_LAYER=4 ...])
```

Behaviour:
1. Pick the right library variant via `_microgpt_lib_for_defines`.
2. `add_executable(... ${SOURCES})`.
3. `target_link_libraries(... PRIVATE ${variant} m)`.
4. If `THREADS` and not MSVC, link `Threads::Threads`.
5. If `METAL` and Metal framework is available, link Metal + Foundation, copy `microgpt_metal.metal` next to the binary.
6. If `BLAS` and BLAS is available, link the BLAS implementation.
7. For each `COPY` file, emit a `POST_BUILD` step that copies the file from source to next to the built binary (handles cross-platform path differences).

This keeps each demo's `main.c` minimal and lets users add new demos without writing CMake plumbing.

## 6. CI

`.github/workflows/cmake-multi-platform.yml` runs:
- Ubuntu 22.04 with `gcc` and `clang`.
- macOS-latest with `clang` (Apple).
- Windows-latest with MSVC `cl`.

For each, it builds the project, runs `test_microgpt`, runs `bench_microgpt`, and verifies they exit 0. CI does NOT run all 90+ targets — `test_microgpt` is the smoke test; subsystem-specific tests (`test_microgpt_msa`, etc.) run only locally and on demand.

## 7. Trade-offs considered

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Build tool | CMake | Bazel / Meson / plain Make | CMake has the largest C99 ecosystem footprint; the multi-platform CI is straightforward. |
| Library variant scheme | Hashed-by-defines | One library + runtime branches | Runtime branching loses the constant-fold benefit; hashing is the cleanest way to keep `add_demo` callers terse. |
| Default `scalar_t` | `float` | `double` | Float32 is faster on ARM NEON (4-wide vs 2-wide) and halves memory; double is opt-in for research / gradient comparisons. |
| Default `MICROGPT_SIMD` | ON | OFF | `-march=native` gives auto-vectorisation a chance; users targeting non-native CPUs override on the command line. |
| Pre-generated parser commit | Yes (Flex/Bison output) | Require Flex/Bison ≥ 3.0 at build | macOS ships Bison 2.3 which can't parse the grammar; pre-generated sources keep the build reproducible. |

## 8. Known limitations

- The variant cache fingerprints by sorted `-D` define-set MD5, but does not capture optimisation level changes or compiler-version skew. Building debug + release in the same `build/` directory may create variants whose names don't reflect the build type.
- POST_BUILD copy steps run per-build; large data files re-copy on every incremental build (timestamp-based caching mitigates but doesn't eliminate this).
- Cross-compilation has been validated for Linux × x86-64 / ARM64 only; SBCs are documented as future work, not validated.

## 9. References

- `docs/BUILD_OPTIONS.md` — user-facing flag documentation.
- CMake documentation for `add_library`, `add_executable`, `target_compile_definitions`, `add_custom_command(POST_BUILD)`.

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-04-30 | Initial extraction. |
