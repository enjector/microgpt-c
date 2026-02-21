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
  NAME   names_demo
  SOURCE demos/character-level/names/main.c
  COPY   demos/character-level/names/names.txt
)

# ==================================================================
#  Foundation: Character-level Shakespeare generation
# ==================================================================

add_demo(
  NAME    shakespeare_demo
  SOURCE  demos/character-level/shakespeare/main.c
  THREADS
  COPY    demos/character-level/shakespeare/shakespeare.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=512
          NUM_STEPS=30000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=200 MAX_DOCS=200000 MAX_DOC_LEN=512
)

# ==================================================================
#  Organelle: C code generation
# ==================================================================

add_demo(
  NAME    c_codegen
  SOURCE  experiments/organelles/c_codegen/main.c
  THREADS METAL BLAS
  COPY    experiments/organelles/c_codegen/c_functions.txt
  DEFINES N_EMBD=128 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=512 MLP_DIM=512
          NUM_STEPS=50000 LEARNING_RATE=0.0003 BATCH_SIZE=16
          MAX_VOCAB=200 MAX_DOCS=5000 MAX_DOC_LEN=1024
)

# ==================================================================
#  Organelle: C wiring generation
# ==================================================================

add_demo(
  NAME    c_wiringgen
  SOURCE  experiments/organelles/c_wiringgen/main.c
  THREADS METAL BLAS
  COPY    experiments/organelles/c_wiringgen/c_wiring.txt
  DEFINES N_EMBD=128 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=512 MLP_DIM=512
          NUM_STEPS=50000 LEARNING_RATE=0.0003 BATCH_SIZE=16
          MAX_VOCAB=200 MAX_DOCS=5000 MAX_DOC_LEN=1024
)

# ==================================================================
#  Organelle: 8-Puzzle multi-organelle pipeline
# ==================================================================

# ==================================================================
#  Organelle: OPA Code Composition Pipeline
# ==================================================================

add_demo(
  NAME    c_compose
  SOURCE  experiments/organelles/c_compose/main.c
  THREADS METAL BLAS
  COPY    experiments/organelles/c_compose/c_planner.txt
          experiments/organelles/c_compose/c_judge.txt
          experiments/organelles/c_compose/c_registry.txt
          experiments/organelles/c_compose/test_intents.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=6 BLOCK_SIZE=128 MLP_DIM=512
          NUM_STEPS=50000 LEARNING_RATE=0.0005 BATCH_SIZE=8 WARMUP_STEPS=2500
          MAX_VOCAB=200 MAX_DOCS=5000 MAX_DOC_LEN=256
)

# ==================================================================
#  Organelle: OPA Code Composition Pipeline v3 (with Syntax Judge)
# ==================================================================

add_demo(
  NAME    c_compose_v3
  SOURCE  experiments/organelles/c_compose_v3/main.c
  THREADS METAL BLAS
  COPY    experiments/organelles/c_compose/c_planner.txt
          experiments/organelles/c_compose/c_judge.txt
          experiments/organelles/c_compose/c_registry.txt
          experiments/organelles/c_compose/test_intents.txt
          experiments/organelles/c_wiringgen/c_wiring.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=6 BLOCK_SIZE=512 MLP_DIM=512
          NUM_STEPS=50000 LEARNING_RATE=0.0005 BATCH_SIZE=8 WARMUP_STEPS=2500
          MAX_VOCAB=200 MAX_DOCS=5000 MAX_DOC_LEN=256
          WIRING_N_EMBD=128 WIRING_N_HEAD=8 WIRING_N_LAYER=6
          WIRING_BLOCK_SIZE=512 WIRING_MLP_DIM=512 WIRING_NUM_STEPS=20000
          WIRING_MAX_DOC_LEN=1024 GEN_LEN=600
)

add_demo(
  NAME    puzzle8_demo
  SOURCE  experiments/organelles/puzzle8/main.c
  THREADS
  COPY    experiments/organelles/puzzle8/puzzle8_strategist.txt
          experiments/organelles/puzzle8/puzzle8_mover.txt
          experiments/organelles/puzzle8/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8/puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: 8-Puzzle Verification Targets (Phase 5)
# ==================================================================

# 460K Bare Model (Verify scaling removed the need for scaffolding)
add_demo(
  NAME    puzzle8_bare
  SOURCE  experiments/organelles/puzzle8/main.c
  THREADS
  COPY    experiments/organelles/puzzle8/puzzle8_strategist.txt
          experiments/organelles/puzzle8/puzzle8_mover.txt
          experiments/organelles/puzzle8/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8/puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          DISABLE_PIPELINE_ASSISTS=1
)

# 64K Assisted (Contrast: Scaffolding is a crutch for low capacity)
add_demo(
  NAME    puzzle8_64k_assisted
  SOURCE  experiments/organelles/puzzle8/main.c
  THREADS
  COPY    experiments/organelles/puzzle8/puzzle8_strategist.txt
          experiments/organelles/puzzle8/puzzle8_mover.txt
          experiments/organelles/puzzle8/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8/puzzle8_judge.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=10000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          STRATEGIST_CKPT="puzzle8_strategist_64k.ckpt"
          MOVER_CKPT="puzzle8_mover_64k.ckpt"
          DETECTOR_CKPT="puzzle8_detector_64k.ckpt"
          DETOUR_MOVER_CKPT="puzzle8_detour_mover_64k.ckpt"
          JUDGE_CKPT="puzzle8_judge_64k.ckpt"
)

# 64K Bare (Contrast: 64K fails without crutch)
add_demo(
  NAME    puzzle8_64k_bare
  SOURCE  experiments/organelles/puzzle8/main.c
  THREADS
  COPY    experiments/organelles/puzzle8/puzzle8_strategist.txt
          experiments/organelles/puzzle8/puzzle8_mover.txt
          experiments/organelles/puzzle8/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8/puzzle8_judge.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=10000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
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
  NAME    puzzle8_reasoning_demo
  SOURCE  experiments/organelles/puzzle8_reasoning/main.c
  THREADS
  COPY    experiments/organelles/puzzle8_reasoning/puzzle8_strategist.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover_enriched.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: 8-Puzzle with Enriched Corpus (Phase 4 A/B comparison)
# ==================================================================

add_demo(
  NAME    puzzle8_reasoning_enriched_demo
  SOURCE  experiments/organelles/puzzle8_reasoning/main.c
  THREADS
  COPY    experiments/organelles/puzzle8_reasoning/puzzle8_strategist.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover_enriched.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          USE_ENRICHED_CORPUS=1
)

# ==================================================================
#  Organelle: 8-Puzzle with Combined Corpus (Phase 4b A/B comparison)
# ==================================================================

add_demo(
  NAME    puzzle8_reasoning_combined_demo
  SOURCE  experiments/organelles/puzzle8_reasoning/main.c
  THREADS
  COPY    experiments/organelles/puzzle8_reasoning/puzzle8_strategist.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover_enriched.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover_combined.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=192 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=192
          USE_ENRICHED_CORPUS=2
)

# ==================================================================
#  Organelle: 8-Puzzle bare-model tests (Phase 5 — scaffolding removal)
# ==================================================================

add_demo(
  NAME    puzzle8_reasoning_standard_bare
  SOURCE  experiments/organelles/puzzle8_reasoning/main.c
  THREADS
  COPY    experiments/organelles/puzzle8_reasoning/puzzle8_strategist.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover_enriched.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=128
          DISABLE_PIPELINE_ASSISTS=1
)

add_demo(
  NAME    puzzle8_reasoning_combined_bare
  SOURCE  experiments/organelles/puzzle8_reasoning/main.c
  THREADS
  COPY    experiments/organelles/puzzle8_reasoning/puzzle8_strategist.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover_enriched.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_mover_combined.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_detector.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_detour_mover.txt
          experiments/organelles/puzzle8_reasoning/puzzle8_judge.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=192 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=60000 MAX_DOC_LEN=192
          USE_ENRICHED_CORPUS=2 DISABLE_PIPELINE_ASSISTS=1
)

# ==================================================================
#  Organelle: Tic-Tac-Toe multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    tictactoe_demo
  SOURCE  experiments/organelles/tictactoe/main.c
  THREADS
  COPY    experiments/organelles/tictactoe/tictactoe_planner.txt
          experiments/organelles/tictactoe/tictactoe_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Connect-4 multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    connect4_demo
  SOURCE  experiments/organelles/connect4/main.c
  THREADS
  COPY    experiments/organelles/connect4/connect4_planner.txt
          experiments/organelles/connect4/connect4_player.txt
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Lights Out multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    lightsout_demo
  SOURCE  experiments/organelles/lightsout/main.c
  THREADS
  COPY    experiments/organelles/lightsout/lightsout_planner.txt
          experiments/organelles/lightsout/lightsout_player.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=256
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=20000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Mastermind multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    mastermind_demo
  SOURCE  experiments/organelles/mastermind/main.c
  THREADS
  COPY    experiments/organelles/mastermind/mastermind_planner.txt
          experiments/organelles/mastermind/mastermind_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Klotski simplified multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    klotski_demo
  SOURCE  experiments/organelles/klotski/main.c
  THREADS
  COPY    experiments/organelles/klotski/klotski_planner.txt
          experiments/organelles/klotski/klotski_player.txt
  DEFINES N_EMBD=32 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=128
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Sudoku 4x4 multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    sudoku_demo
  SOURCE  experiments/organelles/sudoku/main.c
  THREADS
  COPY    experiments/organelles/sudoku/sudoku_planner.txt
          experiments/organelles/sudoku/sudoku_player.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=256
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=20000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Othello 6x6 multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    othello_demo
  SOURCE  experiments/organelles/othello/main.c
  THREADS
  COPY    experiments/organelles/othello/othello_planner.txt
          experiments/organelles/othello/othello_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Hex 7x7 multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    hex_demo
  SOURCE  experiments/organelles/hex/main.c
  THREADS
  COPY    experiments/organelles/hex/hex_planner.txt
          experiments/organelles/hex/hex_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Pentago multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    pentago_demo
  SOURCE  experiments/organelles/pentago/main.c
  THREADS
  COPY    experiments/organelles/pentago/pentago_planner.txt
          experiments/organelles/pentago/pentago_player.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Red Donkey (Huarong Dao) multi-organelle pipeline
# ==================================================================

add_demo(
  NAME    reddonkey_demo
  SOURCE  experiments/organelles/reddonkey/main.c
  THREADS
  COPY    experiments/organelles/reddonkey/reddonkey_planner.txt
          experiments/organelles/reddonkey/reddonkey_player.txt
  DEFINES N_EMBD=32 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=128 MLP_DIM=128
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: EuroMillions Lottery prediction pipeline
# ==================================================================

add_demo(
  NAME    lottery_demo
  SOURCE  experiments/organelles/lottery/main.c
  THREADS
  COPY    experiments/organelles/lottery/lottery_analyser.txt
          experiments/organelles/lottery/lottery_predictor.txt
          experiments/organelles/lottery/euromillions-draw-history.csv
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=256
          NUM_STEPS=50000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=8000 MAX_DOC_LEN=128
)

# ==================================================================
#  Organelle: Market Regime Detection pipeline
# ==================================================================

add_demo(
  NAME    markets_demo
  SOURCE  experiments/organelles/markets/main.c
  THREADS
  COPY    experiments/organelles/markets/market_analyser.txt
          experiments/organelles/markets/market_regime.txt
          experiments/organelles/markets/market_rotator.txt
          experiments/organelles/markets/market_data.csv
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=3 BLOCK_SIZE=128 MLP_DIM=512
          NUM_STEPS=50000 LEARNING_RATE=0.0003 BATCH_SIZE=8
          MAX_VOCAB=80 MAX_DOCS=8000 MAX_DOC_LEN=128
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

add_executable(bench_microgpt_organelle tests/bench_microgpt_organelle.c)
target_link_libraries(bench_microgpt_organelle PRIVATE microgpt_lib)
if(NOT MSVC)
  target_link_libraries(bench_microgpt_organelle PRIVATE m)
endif()

