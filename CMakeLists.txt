cmake_minimum_required(VERSION 3.10)
project(microgpt C)

set(CMAKE_C_STANDARD 99)

# Default to Release when no config is specified (single-config generators)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# High optimisation for Release
if(MSVC)
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /Ot" CACHE STRING "" FORCE)
else()
  set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -funroll-loops -DNDEBUG" CACHE STRING "" FORCE)
endif()

# ==================================================================
#  Feature options
# ==================================================================

option(MICROGPT_SIMD           "Enable SIMD/vectorised codegen"                     ON)
option(MICROGPT_USE_FLOAT      "Use float (32-bit) instead of double (64-bit)"      ON)
option(MICROGPT_HEAD_PARALLEL  "Parallelise attention heads per inference call"      OFF)
option(MICROGPT_PAGED_KV       "Use demand-paged KV cache"                          OFF)
option(QUANTIZATION_INT8       "Store weights as 8-bit integers with per-matrix scales" OFF)
option(MICROGPT_BLAS           "Use BLAS (Accelerate / OpenBLAS / MKL)"             OFF)
option(MICROGPT_METAL          "Use Metal GPU acceleration (macOS only)"            OFF)

# ==================================================================
#  Core library  (single OBJECT library shared by all targets)
# ==================================================================

add_library(microgpt_lib STATIC src/microgpt.c src/microgpt_organelle.c)
target_include_directories(microgpt_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- Compile-time feature flags ---
if(MICROGPT_USE_FLOAT)
  target_compile_definitions(microgpt_lib PUBLIC MICROGPT_USE_FLOAT=1)
endif()
if(MICROGPT_SIMD)
  target_compile_definitions(microgpt_lib PUBLIC MICROGPT_SIMD=1)
  if(MSVC)
    target_compile_options(microgpt_lib PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
  else()
    target_compile_options(microgpt_lib PRIVATE $<$<CONFIG:Release>:-march=native>)
  endif()
endif()
if(QUANTIZATION_INT8)
  target_compile_definitions(microgpt_lib PUBLIC QUANTIZATION_INT8=1 QUANTISATION_INT8=1)
endif()
if(MICROGPT_PAGED_KV)
  target_compile_definitions(microgpt_lib PUBLIC MICROGPT_PAGED_KV=1)
endif()
if(MICROGPT_HEAD_PARALLEL)
  target_compile_definitions(microgpt_lib PUBLIC MICROGPT_HEAD_PARALLEL=1)
endif()

# ==================================================================
#  Optional: BLAS acceleration
# ==================================================================

if(MICROGPT_BLAS)
  if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
      message(STATUS "Found Apple Accelerate.framework — BLAS enabled")
      target_compile_definitions(microgpt_lib PUBLIC MICROGPT_BLAS=1)
    else()
      message(WARNING "Accelerate.framework not found — disabling BLAS")
      set(MICROGPT_BLAS OFF)
    endif()
  endif()
endif()

# ==================================================================
#  Optional: Metal GPU acceleration (macOS only)
# ==================================================================

if(MICROGPT_METAL)
  if(APPLE)
    enable_language(OBJC)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
      message(STATUS "Found Metal.framework — GPU acceleration enabled")
      target_compile_definitions(microgpt_lib PUBLIC MICROGPT_METAL=1)
    else()
      message(WARNING "Metal.framework not found — disabling Metal GPU")
      set(MICROGPT_METAL OFF)
    endif()
  else()
    message(WARNING "Metal GPU is only available on macOS — disabling")
    set(MICROGPT_METAL OFF)
  endif()
endif()

# ==================================================================
#  Threading (needed by demos that use multi-threaded training)
# ==================================================================

if(NOT MSVC)
  find_package(Threads REQUIRED)
endif()

# ==================================================================
#  Helper: create a library variant with overridden compile defs
# ==================================================================
#
# Each demo that uses non-default architecture dimensions (N_EMBD,
# N_LAYER, BLOCK_SIZE, …) needs its OWN compiled copy of microgpt.c
# so the compiler can constant-fold the macro values.  We cache
# library targets by their sorted defines so identical configs share
# one target.
#
set(_MICROGPT_LIB_VARIANTS "" CACHE INTERNAL "")

function(_microgpt_lib_for_defines OUT_VAR)
  # Collect remaining ARGN as the defines list
  set(_defs ${ARGN})
  if(NOT _defs)
    set(${OUT_VAR} microgpt_lib PARENT_SCOPE)
    return()
  endif()
  # Build a unique suffix from the sorted defines
  list(SORT _defs)
  string(REPLACE ";" "_" _raw_suffix "${_defs}")
  string(MD5 _hash "${_raw_suffix}")
  set(_tgt "microgpt_lib_${_hash}")

  # Reuse existing target if already created
  if(TARGET ${_tgt})
    set(${OUT_VAR} ${_tgt} PARENT_SCOPE)
    return()
  endif()

  # Create a new STATIC library with the same sources + features
  add_library(${_tgt} STATIC src/microgpt.c src/microgpt_organelle.c)
  target_include_directories(${_tgt} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

  # Propagate all feature flags from the base target
  get_target_property(_base_defs microgpt_lib INTERFACE_COMPILE_DEFINITIONS)
  if(_base_defs)
    target_compile_definitions(${_tgt} PUBLIC ${_base_defs})
  endif()

  # Add the arch-specific overrides
  foreach(_d ${_defs})
    target_compile_definitions(${_tgt} PUBLIC ${_d})
  endforeach()

  # Propagate compile options
  get_target_property(_base_opts microgpt_lib COMPILE_OPTIONS)
  if(_base_opts)
    target_compile_options(${_tgt} PRIVATE ${_base_opts})
  endif()

  set(${OUT_VAR} ${_tgt} PARENT_SCOPE)
endfunction()

# ==================================================================
#  Helper: add a demo executable
# ==================================================================
#
#  add_demo(NAME target_name
#           SOURCE path/to/main.c
#           [THREADS]                  -- link pthreads
#           [METAL]                    -- link Metal + copy shader
#           [BLAS]                     -- link BLAS framework
#           [COPY file1 file2 ...]     -- copy data files to build dir
#           [DEFINES DEF1=val ...]     -- override architecture macros
#
function(add_demo)
  cmake_parse_arguments(DEMO "THREADS;METAL;BLAS" "NAME;SOURCE" "COPY;DEFINES" ${ARGN})

  set(DEMO_SOURCES ${DEMO_SOURCE})
  if(DEMO_METAL AND MICROGPT_METAL)
    list(APPEND DEMO_SOURCES src/microgpt_metal.m)
  endif()

  # Pick the right library variant for the requested defines
  _microgpt_lib_for_defines(_LIB_TGT ${DEMO_DEFINES})

  add_executable(${DEMO_NAME} ${DEMO_SOURCES})
  target_link_libraries(${DEMO_NAME} PRIVATE ${_LIB_TGT})

  if(NOT MSVC)
    target_link_libraries(${DEMO_NAME} PRIVATE m)
  endif()

  if(DEMO_THREADS AND NOT MSVC)
    target_link_libraries(${DEMO_NAME} PRIVATE Threads::Threads)
  endif()

  if(DEMO_BLAS AND MICROGPT_BLAS AND APPLE)
    target_link_libraries(${DEMO_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
  endif()

  if(DEMO_METAL AND MICROGPT_METAL)
    target_link_libraries(${DEMO_NAME} PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_metal.metal"
        "$<TARGET_FILE_DIR:${DEMO_NAME}>/microgpt_metal.metal"
      COMMENT "Copying microgpt_metal.metal for ${DEMO_NAME}"
    )
  endif()

  # Copy data files
  foreach(F ${DEMO_COPY})
    get_filename_component(FNAME ${F} NAME)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/${F}"
        "$<TARGET_FILE_DIR:${DEMO_NAME}>/${FNAME}"
      COMMENT "Copying ${FNAME} to build output"
    )
  endforeach()
endfunction()

# ==================================================================
#  Foundation: Character-level name generation  (uses defaults)
# ==================================================================

add_demo(
  NAME   c_names_demo
  SOURCE demos/character-level/names/main.c
  COPY   demos/character-level/names/c_names.txt
)

# ==================================================================
#  Foundation: Character-level Shakespeare generation
# ==================================================================

add_demo(
  NAME    c_shakespeare_demo
  SOURCE  demos/character-level/shakespeare/main.c
  THREADS
  COPY    demos/character-level/shakespeare/c_shakespeare.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=512
          NUM_STEPS=30000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=200 MAX_DOCS=200000 MAX_DOC_LEN=512
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1
)

add_demo(
  NAME    w_shakespeare_demo
  SOURCE  demos/word-level/shakespeare/main.c
  METAL BLAS
  COPY    demos/word-level/shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=10000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1
)


# Shakespeare char-level optimisation A/B benchmarks (2K steps, batch=4 for quick comparison)
# Must match demo's hardcoded config: N_EMBD=128, N_LAYER=4 (~1M params)
add_demo(
  NAME    c_optim_shk_baseline
  SOURCE  demos/character-level/shakespeare/main.c
  THREADS
  COPY    demos/character-level/shakespeare/c_shakespeare.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=512
          NUM_STEPS=2000 LEARNING_RATE=0.001 BATCH_SIZE=4
          MAX_VOCAB=200 MAX_DOCS=200000 MAX_DOC_LEN=512
)
add_demo(
  NAME    c_optim_shk_labelsmooth
  SOURCE  demos/character-level/shakespeare/main.c
  THREADS
  COPY    demos/character-level/shakespeare/c_shakespeare.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=512
          NUM_STEPS=2000 LEARNING_RATE=0.001 BATCH_SIZE=4
          MAX_VOCAB=200 MAX_DOCS=200000 MAX_DOC_LEN=512
          LABEL_SMOOTH=0.1
)
add_demo(
  NAME    c_optim_shk_weightdecay
  SOURCE  demos/character-level/shakespeare/main.c
  THREADS
  COPY    demos/character-level/shakespeare/c_shakespeare.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=512
          NUM_STEPS=2000 LEARNING_RATE=0.001 BATCH_SIZE=4
          MAX_VOCAB=200 MAX_DOCS=200000 MAX_DOC_LEN=512
          WEIGHT_DECAY=0.01
)
add_demo(
  NAME    c_optim_shk_combined
  SOURCE  demos/character-level/shakespeare/main.c
  THREADS
  COPY    demos/character-level/shakespeare/c_shakespeare.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=512
          NUM_STEPS=2000 LEARNING_RATE=0.001 BATCH_SIZE=4
          MAX_VOCAB=200 MAX_DOCS=200000 MAX_DOC_LEN=512
          GRAD_CLIP=1.0 WEIGHT_DECAY=0.01 LABEL_SMOOTH=0.1
)

# Shakespeare word-level optimisation A/B benchmarks
# Word-level: N_EMBD=48, N_LAYER=1, ~250K params — vocab dominates
add_demo(
  NAME    w_optim_shk_baseline
  SOURCE  demos/word-level/shakespeare/main.c
  THREADS
  COPY    demos/word-level/shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
)
add_demo(
  NAME    w_optim_shk_labelsmooth
  SOURCE  demos/word-level/shakespeare/main.c
  THREADS
  COPY    demos/word-level/shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
          LABEL_SMOOTH=0.1
)
add_demo(
  NAME    w_optim_shk_weightdecay
  SOURCE  demos/word-level/shakespeare/main.c
  THREADS
  COPY    demos/word-level/shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
          WEIGHT_DECAY=0.01
)


# ==================================================================
#  Organelle: VM code generation (constrained DSL experiment)
# ==================================================================

add_demo(
  NAME    c_vm_codegen
  SOURCE  demos/character-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/character-level/vm_codegen/c_vm_functions_combined.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=15000 LEARNING_RATE=0.0005 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1
)
target_link_libraries(c_vm_codegen PRIVATE microgpt_vm_lib)

# ==================================================================
#  Organelle: VM codegen V2 (organelle word API refactor)
# ==================================================================

add_demo(
  NAME    c_vm_codegen_v2
  SOURCE  demos/character-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/character-level/vm_codegen/c_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
)
target_link_libraries(c_vm_codegen_v2 PRIVATE microgpt_vm_lib)

# ==================================================================
#  Organelle: VM codegen V2 word-level
# ==================================================================
add_demo(
  NAME    w_vm_codegen_v2
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
)
target_link_libraries(w_vm_codegen_v2 PRIVATE microgpt_vm_lib)

# ==================================================================
#  Organelle: VM compose pipeline (generate → validate → retry)
# ==================================================================

add_demo(
  NAME    c_vm_compose
  SOURCE  demos/character-level/vm_compose/main.c
  METAL BLAS
  COPY    demos/character-level/vm_codegen/c_vm_functions_combined.txt
          demos/character-level/vm_compose/test_intents.txt
          demos/character-level/vm_compose/test_signatures.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=15000 LEARNING_RATE=0.0005 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
)
target_link_libraries(c_vm_compose PRIVATE microgpt_vm_lib)



# ==================================================================
#  Organelle: 8-Puzzle multi-organelle pipeline
# ==================================================================

# ==================================================================
#  Organelle: OPA Code Composition Pipeline
# ==================================================================

add_demo(
  NAME    c99_compose
  SOURCE  demos/character-level/c99_compose/main.c
  THREADS METAL BLAS
  COPY    demos/character-level/c99_compose/c_planner.txt
          demos/character-level/c99_compose/c_judge.txt
          demos/character-level/c99_compose/c_registry.txt
          demos/character-level/c99_compose/test_intents.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=6 BLOCK_SIZE=128 MLP_DIM=512
          NUM_STEPS=50000 LEARNING_RATE=0.0005 BATCH_SIZE=8 WARMUP_STEPS=2500
          MAX_VOCAB=200 MAX_DOCS=5000 MAX_DOC_LEN=256
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1
)



add_demo(
  NAME    c_puzzle8_demo
  SOURCE  demos/character-level/puzzle8/main.c
  THREADS
  COPY    demos/character-level/puzzle8/c_puzzle8_strategist.txt
          demos/character-level/puzzle8/c_puzzle8_mover.txt
          demos/character-level/puzzle8/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8/c_puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Organelle: 8-Puzzle Verification Targets (Phase 5)
# ==================================================================

# 460K Bare Model (Verify scaling removed the need for scaffolding)
add_demo(
  NAME    c_puzzle8_bare
  SOURCE  demos/character-level/puzzle8/main.c
  THREADS
  COPY    demos/character-level/puzzle8/c_puzzle8_strategist.txt
          demos/character-level/puzzle8/c_puzzle8_mover.txt
          demos/character-level/puzzle8/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8/c_puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
          DISABLE_PIPELINE_ASSISTS=1
)

# 64K Assisted (Contrast: Scaffolding is a crutch for low capacity)
add_demo(
  NAME    c_puzzle8_64k_assisted
  SOURCE  demos/character-level/puzzle8/main.c
  THREADS
  COPY    demos/character-level/puzzle8/c_puzzle8_strategist.txt
          demos/character-level/puzzle8/c_puzzle8_mover.txt
          demos/character-level/puzzle8/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8/c_puzzle8_judge.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=10000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
          STRATEGIST_CKPT="puzzle8_strategist_64k.ckpt"
          MOVER_CKPT="puzzle8_mover_64k.ckpt"
          DETECTOR_CKPT="puzzle8_detector_64k.ckpt"
          DETOUR_MOVER_CKPT="puzzle8_detour_mover_64k.ckpt"
          JUDGE_CKPT="puzzle8_judge_64k.ckpt"
)

# 64K Bare (Contrast: 64K fails without crutch)
add_demo(
  NAME    c_puzzle8_64k_bare
  SOURCE  demos/character-level/puzzle8/main.c
  THREADS
  COPY    demos/character-level/puzzle8/c_puzzle8_strategist.txt
          demos/character-level/puzzle8/c_puzzle8_mover.txt
          demos/character-level/puzzle8/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8/c_puzzle8_judge.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=10000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
          DISABLE_PIPELINE_ASSISTS=1
          STRATEGIST_CKPT="puzzle8_strategist_64k.ckpt"
          MOVER_CKPT="puzzle8_mover_64k.ckpt"
          DETECTOR_CKPT="puzzle8_detector_64k.ckpt"
          DETOUR_MOVER_CKPT="puzzle8_detour_mover_64k.ckpt"
          JUDGE_CKPT="puzzle8_judge_64k.ckpt"
)

# ==================================================================
#  Organelle: 8-Puzzle with Reasoning Trace capture (comparison variant)
# ==================================================================

add_demo(
  NAME    c_puzzle8_reasoning_demo
  SOURCE  demos/character-level/puzzle8_reasoning/main.c
  THREADS
  COPY    demos/character-level/puzzle8_reasoning/c_puzzle8_strategist.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover_enriched.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: 8-Puzzle with Enriched Corpus (Phase 4 A/B comparison)
# ==================================================================

add_demo(
  NAME    c_puzzle8_reasoning_enriched_demo
  SOURCE  demos/character-level/puzzle8_reasoning/main.c
  THREADS
  COPY    demos/character-level/puzzle8_reasoning/c_puzzle8_strategist.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover_enriched.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
          USE_ENRICHED_CORPUS=1
)

# ==================================================================
#  Organelle: 8-Puzzle with Combined Corpus (Phase 4b A/B comparison)
# ==================================================================

add_demo(
  NAME    c_puzzle8_reasoning_combined_demo
  SOURCE  demos/character-level/puzzle8_reasoning/main.c
  THREADS
  COPY    demos/character-level/puzzle8_reasoning/c_puzzle8_strategist.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover_enriched.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover_combined.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=192 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=192
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
          USE_ENRICHED_CORPUS=2
)

# ==================================================================
#  Organelle: 8-Puzzle bare-model tests (Phase 5 — scaffolding removal)
# ==================================================================

add_demo(
  NAME    c_puzzle8_reasoning_standard_bare
  SOURCE  demos/character-level/puzzle8_reasoning/main.c
  THREADS
  COPY    demos/character-level/puzzle8_reasoning/c_puzzle8_strategist.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover_enriched.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
          DISABLE_PIPELINE_ASSISTS=1
)

add_demo(
  NAME    c_puzzle8_reasoning_combined_bare
  SOURCE  demos/character-level/puzzle8_reasoning/main.c
  THREADS
  COPY    demos/character-level/puzzle8_reasoning/c_puzzle8_strategist.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover_enriched.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_mover_combined.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_detector.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_detour_mover.txt
          demos/character-level/puzzle8_reasoning/c_puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=192 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=192
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
          USE_ENRICHED_CORPUS=2 DISABLE_PIPELINE_ASSISTS=1
)

# ==================================================================
#  Organelle: Tic-Tac-Toe multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    c_tictactoe_demo
  SOURCE  demos/character-level/tictactoe/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_planner.txt
          demos/character-level/tictactoe/c_tictactoe_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Organelle: Connect-4 multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    c_connect4_demo
  SOURCE  demos/character-level/connect4/main.c
  THREADS
  COPY    demos/character-level/connect4/c_connect4_planner.txt
          demos/character-level/connect4/c_connect4_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Organelle: Lights Out multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    c_lightsout_demo
  SOURCE  demos/character-level/lightsout/main.c
  THREADS
  COPY    demos/character-level/lightsout/c_lightsout_planner.txt
          demos/character-level/lightsout/c_lightsout_player.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=256
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=20000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)
# ==================================================================

add_demo(
  NAME    c_mastermind_demo
  SOURCE  demos/character-level/mastermind/main.c
  THREADS
  COPY    demos/character-level/mastermind/c_mastermind_planner.txt
          demos/character-level/mastermind/c_mastermind_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)
# ==================================================================

add_demo(
  NAME    c_klotski_demo
  SOURCE  demos/character-level/klotski/main.c
  THREADS
  COPY    demos/character-level/klotski/c_klotski_planner.txt
          demos/character-level/klotski/c_klotski_player.txt
  DEFINES N_EMBD=32 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=128
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)
# ==================================================================

add_demo(
  NAME    c_sudoku_demo
  SOURCE  demos/character-level/sudoku/main.c
  THREADS
  COPY    demos/character-level/sudoku/c_sudoku_planner.txt
          demos/character-level/sudoku/c_sudoku_player.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=256
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=20000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)
# ==================================================================

add_demo(
  NAME    c_othello_demo
  SOURCE  demos/character-level/othello/main.c
  THREADS
  COPY    demos/character-level/othello/c_othello_planner.txt
          demos/character-level/othello/c_othello_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)
# ==================================================================

add_demo(
  NAME    c_hex_demo
  SOURCE  demos/character-level/hex/main.c
  THREADS
  COPY    demos/character-level/hex/c_hex_planner.txt
          demos/character-level/hex/c_hex_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Organelle: Hex 5x5 variant (smaller board, same topology features)
# ==================================================================

add_demo(
  NAME    c_hex5_demo
  SOURCE  demos/character-level/hex/main.c
  THREADS
  COPY    demos/character-level/hex/c_hex5_planner.txt
          demos/character-level/hex/c_hex5_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          HEX_GRID=5
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Organelle: Hex 5x5 Word-Level (word tokenisation variant)
# ==================================================================

add_demo(
  NAME    w_hex5_demo
  SOURCE  demos/word-level/hex/main_word.c
  THREADS
  COPY    demos/word-level/hex/w_hex5_planner.txt
          demos/word-level/hex/w_hex5_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=600 MAX_DOCS=10000 MAX_DOC_LEN=128
          HEX_GRID=5 WORD_VOCAB_SIZE=512
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Organelle: Pentago multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    c_pentago_demo
  SOURCE  demos/character-level/pentago/main.c
  THREADS
  COPY    demos/character-level/pentago/c_pentago_planner.txt
          demos/character-level/pentago/c_pentago_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Organelle: Red Donkey (Huarong Dao) multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    c_reddonkey_demo
  SOURCE  demos/character-level/reddonkey/main.c
  THREADS
  COPY    demos/character-level/reddonkey/c_reddonkey_planner.txt
          demos/character-level/reddonkey/c_reddonkey_player.txt
  DEFINES N_EMBD=32 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=128
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Organelle: Red Donkey 5×4 classic Huarong Dao layout
# ==================================================================

add_demo(
  NAME    c_rd54_demo
  SOURCE  demos/character-level/reddonkey/main.c
  THREADS
  COPY    demos/character-level/reddonkey/c_rd54_planner.txt
          demos/character-level/reddonkey/c_rd54_player.txt
  DEFINES N_EMBD=32 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=128
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          RD_ROWS=5 RD_COLS=4
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  K-15: BLOCK_SIZE=256 variants (testing wider context window)
# ==================================================================

add_demo(
  NAME    c_hex_256_demo
  SOURCE  demos/character-level/hex/main.c
  THREADS
  COPY    demos/character-level/hex/c_hex_planner.txt
          demos/character-level/hex/c_hex_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=256 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=256
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

add_demo(
  NAME    c_rd54_256_demo
  SOURCE  demos/character-level/reddonkey/main.c
  THREADS
  COPY    demos/character-level/reddonkey/c_rd54_planner.txt
          demos/character-level/reddonkey/c_rd54_player.txt
  DEFINES N_EMBD=32 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=128
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=256
          RD_ROWS=5 RD_COLS=4
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)
# ==================================================================
#
#  Quick targets for comparing training optimisation techniques.
#  Each trains a single organelle on the tictactoe corpus (game data)
#  with 5000 steps — fast enough for iterative A/B testing.
#

# Baseline — no optimisations
add_demo(
  NAME    c_optim_bench_baseline
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
)

# Experiment 1: Gradient Clipping (||g|| <= 1.0)
add_demo(
  NAME    c_optim_bench_gradclip
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0
)

# Experiment 2: Weight Decay (AdamW, wd=0.01)
add_demo(
  NAME    c_optim_bench_weightdecay
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          WEIGHT_DECAY=0.01
)

# Experiment 3: Tuned Adam Hyperparameters (standard values)
add_demo(
  NAME    c_optim_bench_adamtuned
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.0006 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          BETA1=0.9 BETA2=0.999
)

# Experiment 4: Label Smoothing (alpha=0.1)
add_demo(
  NAME    c_optim_bench_labelsmooth
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          LABEL_SMOOTH=0.1
)

# Experiment 5: Combined (grad clip + weight decay + label smoothing)
add_demo(
  NAME    c_optim_bench_combined
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 WEIGHT_DECAY=0.01 LABEL_SMOOTH=0.1
)

# Connect-4 benchmarks (larger board, more complex strategy)
# Uses the same architecture as the real connect4_demo (N_EMBD=96, ~460K params)

add_demo(
  NAME    c_optim_c4_baseline
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/connect4/c_connect4_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          BENCH_CORPUS=\"connect4_player.txt\"
)

add_demo(
  NAME    c_optim_c4_gradclip
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/connect4/c_connect4_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          BENCH_CORPUS=\"connect4_player.txt\"
          GRAD_CLIP=1.0
)

add_demo(
  NAME    c_optim_c4_weightdecay
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/connect4/c_connect4_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          BENCH_CORPUS=\"connect4_player.txt\"
          WEIGHT_DECAY=0.01
)

add_demo(
  NAME    c_optim_c4_adamtuned
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/connect4/c_connect4_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.0006 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          BENCH_CORPUS=\"connect4_player.txt\"
          BETA1=0.9 BETA2=0.999
)

add_demo(
  NAME    c_optim_c4_labelsmooth
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/connect4/c_connect4_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          BENCH_CORPUS=\"connect4_player.txt\"
          LABEL_SMOOTH=0.1
)

add_demo(
  NAME    c_optim_c4_combined
  SOURCE  demos/character-level/optim_bench/main.c
  THREADS
  COPY    demos/character-level/connect4/c_connect4_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
          BENCH_CORPUS=\"connect4_player.txt\"
          GRAD_CLIP=1.0 WEIGHT_DECAY=0.01 LABEL_SMOOTH=0.1
)

# ==================================================================
#  Organelle: EuroMillions Lottery prediction pipeline
# ==================================================================

add_demo(
  NAME    c_lottery_demo
  SOURCE  demos/character-level/lottery/main.c
  THREADS
  COPY    demos/character-level/lottery/c_lottery_analyser.txt
          demos/character-level/lottery/c_lottery_predictor.txt
          demos/character-level/lottery/c_euromillions-draw-history.csv
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=256
          NUM_STEPS=50000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=8000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1
)
# ==================================================================


# ==================================================================
#  Organelle: Transfer Learning — TTT → Othello (cross-game)
# ==================================================================

add_demo(
  NAME    c_tictactoe_transfer_othello
  SOURCE  demos/character-level/tictactoe_transfer_othello/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_player.txt
          demos/character-level/othello/c_othello_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1
)

# ==================================================================
#  Organelle: Transfer Learning — Same-game (Planner → Player)
# ==================================================================

add_demo(
  NAME    c_tictactoe_transfer_planner_to_player
  SOURCE  demos/character-level/tictactoe_transfer_planner_to_player/main.c
  THREADS
  COPY    demos/character-level/tictactoe/c_tictactoe_planner.txt
          demos/character-level/tictactoe/c_tictactoe_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1
)

# ==================================================================
#  Tests
# ==================================================================

enable_testing()
add_executable(test_microgpt tests/test_microgpt.c)
target_link_libraries(test_microgpt PRIVATE microgpt_lib)
if(NOT MSVC)
  target_link_libraries(test_microgpt PRIVATE m)
endif()
add_test(NAME microgpt_tests COMMAND test_microgpt WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# ==================================================================
#  Benchmarks
# ==================================================================

add_executable(bench_microgpt tests/bench_microgpt.c)
target_link_libraries(bench_microgpt PRIVATE microgpt_lib)
if(NOT MSVC)
  target_link_libraries(bench_microgpt PRIVATE m)
endif()

# ==================================================================
#  Organelle Pipeline Tests & Benchmarks
# ==================================================================

add_executable(test_microgpt_organelle tests/test_microgpt_organelle.c)
target_link_libraries(test_microgpt_organelle PRIVATE microgpt_lib)
if(NOT MSVC)
  target_link_libraries(test_microgpt_organelle PRIVATE m)
endif()
add_test(NAME organelle_tests COMMAND test_microgpt_organelle WORKING_DIRECTORY ${CMAKE_BINARY_DIR})



# ==================================================================
#  Virtual Machine Engine
# ==================================================================

find_package(FLEX)
find_package(BISON)

set(VM_PARSER_SOURCES)
# Grammar uses %define api.prefix which requires Bison >= 3.0
# macOS ships Bison 2.3 which cannot parse this syntax
if(FLEX_FOUND AND BISON_FOUND AND BISON_VERSION VERSION_GREATER_EQUAL "3.0")
    flex_target(vm_scanner
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_vm.l
        ${CMAKE_CURRENT_BINARY_DIR}/microgpt_vm_parser.l.c)
    bison_target(vm_parser
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_vm.y
        ${CMAKE_CURRENT_BINARY_DIR}/microgpt_vm_parser.tab.c)
    add_flex_bison_dependency(vm_scanner vm_parser)
    list(APPEND VM_PARSER_SOURCES
        ${FLEX_vm_scanner_OUTPUTS}
        ${BISON_vm_parser_OUTPUTS})
else()
    if(BISON_FOUND AND BISON_VERSION VERSION_LESS "3.0")
        message(STATUS "Bison ${BISON_VERSION} found but >= 3.0 required — using pre-generated parser sources")
    else()
        message(STATUS "Flex/Bison not found — using pre-generated parser sources")
    endif()
    list(APPEND VM_PARSER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_vm_parser.l.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_vm_parser.tab.c)
endif()

add_library(microgpt_vm_lib STATIC src/microgpt_vm.c ${VM_PARSER_SOURCES})
target_include_directories(microgpt_vm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_microgpt_vm tests/test_microgpt_vm.c)
target_link_libraries(test_microgpt_vm PRIVATE microgpt_vm_lib)
target_include_directories(test_microgpt_vm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
if(NOT MSVC)
  target_link_libraries(test_microgpt_vm PRIVATE m)
endif()

# Copy test resources dynamically for test_microgpt_vm
add_custom_command(TARGET test_microgpt_vm POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/resources/tests/vm"
    "$<TARGET_FILE_DIR:test_microgpt_vm>/resources/vm"
  COMMENT "Copying VM test resources"
)
add_test(NAME microgpt_vm_tests COMMAND test_microgpt_vm WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(bench_microgpt_vm tests/bench_microgpt_vm.c)
target_link_libraries(bench_microgpt_vm PRIVATE microgpt_vm_lib)
target_include_directories(bench_microgpt_vm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
if(NOT MSVC)
  target_link_libraries(bench_microgpt_vm PRIVATE m)
endif()
