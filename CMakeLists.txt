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
option(MICROGPT_ATTN_RES       "Block Attention Residuals (learned depth-attention)" OFF)
option(ENABLE_TURBOQUANT       "Enable TurboQuant for memory compression (MSA)" OFF)
option(ENABLE_ROTORQUANT       "Enable RotorQuant for memory compression (MSA)" OFF)

# ==================================================================
#  Sub-libraries — extracted standalone modules
# ==================================================================
#
# libpipeline_ir lives in libs/pipeline_ir/ as a self-contained C99
# library; the parent microgpt_lib links it so existing demos that
# `#include "microgpt_pipeline.h"` keep working via the shim header at
# src/microgpt_pipeline.h.  Experiment E02 documents the extraction.
add_subdirectory(libs/pipeline_ir)

# ==================================================================
#  Core library  (single OBJECT library shared by all targets)
# ==================================================================

add_library(microgpt_lib STATIC src/microgpt.c src/microgpt_organelle.c src/microgpt_msa.c src/microgpt_turboquant.c src/microgpt_rotorquant.c src/microgpt_vr.c src/microgpt_geodesic.c src/microgpt_eml.c)
target_include_directories(microgpt_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(microgpt_lib PUBLIC pipeline_ir)

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
if(MICROGPT_ATTN_RES)
  target_compile_definitions(microgpt_lib PUBLIC MICROGPT_ATTN_RES=1)
endif()
if(ENABLE_TURBOQUANT)
  target_compile_definitions(microgpt_lib PUBLIC ENABLE_TURBOQUANT=1)
endif()
if(ENABLE_ROTORQUANT)
  target_compile_definitions(microgpt_lib PUBLIC ENABLE_ROTORQUANT=1)
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

  # Create a new STATIC library with the same sources + features.
  # The Pipeline IR lives in libs/pipeline_ir/ as a separate target;
  # variant libs reuse the SAME pipeline_ir target — the IR has no
  # demo-specific macros so a single compiled copy is correct.
  add_library(${_tgt} STATIC src/microgpt.c src/microgpt_organelle.c src/microgpt_msa.c src/microgpt_turboquant.c src/microgpt_rotorquant.c src/microgpt_vr.c src/microgpt_geodesic.c src/microgpt_eml.c)
  target_include_directories(${_tgt} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(${_tgt} PUBLIC pipeline_ir)

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
#  EML Organelle demo
#  Loads a snapped EML tree (recovered offline by the EML SR trainer in
#  the companion eml research repo) and evaluates it in pure C99 against
#  a noisy elementary target.  See docs/research/RESEARCH_EML_ORGANELLE.md.
# ==================================================================

add_demo(
  NAME   eml_organelle_demo
  SOURCE demos/character-level/eml_organelle/main.c
)
# The demo includes c_eml_tree.h, which lives next to main.c.
target_include_directories(eml_organelle_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/eml_organelle)

# ==================================================================
#  EML Quant Boundary demo — boundary map of where the EML organelle
#  works on quant-flavored data and where it doesn't.  Mirrors the
#  lottery negative control's pattern: 3 positive + 3 negative cases.
#  See docs/research/RESEARCH_EML_ORGANELLE.md §Boundary Map.
# ==================================================================

add_demo(
  NAME   eml_quant_boundary_demo
  SOURCE demos/character-level/eml_quant_boundary/main.c
)
target_include_directories(eml_quant_boundary_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/eml_quant_boundary)

# ==================================================================
#  EML Hybrid OPA Pipeline demo — neuro-symbolic composition.
#  Tiny neural direction-predictor + two EML deterministic transforms
#  routed through Kanban + cycle detector.  Reuses the c_eml_logprice.h
#  and c_eml_compound.h headers from the boundary demo dir.
# ==================================================================

add_demo(
  NAME    eml_hybrid_pipeline_demo
  SOURCE  demos/character-level/eml_hybrid_pipeline/main.c
  THREADS
  COPY    demos/character-level/eml_hybrid_pipeline/c_directions.txt
  DEFINES N_EMBD=24 N_HEAD=2 N_LAYER=2 BLOCK_SIZE=32 MLP_DIM=48
          NUM_STEPS=600 LEARNING_RATE=0.01 BATCH_SIZE=4 MAX_DOC_LEN=64
)
target_include_directories(eml_hybrid_pipeline_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/eml_hybrid_pipeline
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/eml_quant_boundary)

# ==================================================================
#  EML Sensor Calibration demo — PWJ pipeline with EML Worker.
#  Photodiode-style log-amp.  Worker = depth-3 EML log tree (exact),
#  Judge = deterministic range check, Planner = tiny neural that
#  sequences MEASURE / REPORT / CALIBRATE actions.  See
#  docs/research/RESEARCH_EML_ORGANELLE.md §Sensor Calibration.
# ==================================================================

add_demo(
  NAME    eml_sensor_calibration_demo
  SOURCE  demos/character-level/eml_sensor_calibration/main.c
  THREADS
  COPY    demos/character-level/eml_sensor_calibration/c_calibration_planner.txt
  DEFINES N_EMBD=24 N_HEAD=2 N_LAYER=2 BLOCK_SIZE=32 MLP_DIM=48
          NUM_STEPS=600 LEARNING_RATE=0.01 BATCH_SIZE=4 MAX_DOC_LEN=64
)
target_include_directories(eml_sensor_calibration_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/eml_sensor_calibration
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/eml_quant_boundary)

# ==================================================================
#  E04 — Neural + EML hybrid demo on synthetic pendulum data.
#  Implements the end-to-end pipeline pre-registered in
#  experiments/E04-eml-neural-hybrid.md:
#    noisy (L, theta_obs) → neural regime classifier
#                       → per-regime EML node
#                       → IR-verified bounds judge
#                       → (period, regime, sympy audit)
#  Includes a same-architecture pure-neural baseline for the T4
#  contrast. The EML tree headers are placeholders pending offline
#  retraining; see c_eml_smallangle.h header docstring.
# ==================================================================

add_demo(
  NAME    eml_neural_hybrid_demo
  SOURCE  demos/character-level/eml_neural_hybrid/main.c
  THREADS
  DEFINES N_EMBD=32 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=16 MLP_DIM=64
          NUM_STEPS=3000 LEARNING_RATE=0.005 BATCH_SIZE=16 MAX_DOC_LEN=16
)
target_include_directories(eml_neural_hybrid_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/eml_neural_hybrid)

# ==================================================================
#  Attention Sink A/B benchmark (DeepSeek-V4 §2.3.3 port evaluation)
#  Two binaries with identical seeds & data; only ATTN_SINK differs.
#  See docs/research/RESEARCH_DEEPSEEK_V4_PORTING_ATTENTION_SINK.md
# ==================================================================

add_demo(
  NAME    bench_attn_sink_baseline
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES BENCH_NUM_STEPS=600 BENCH_BATCH_SIZE=8
)

add_demo(
  NAME    bench_attn_sink_on
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES BENCH_NUM_STEPS=600 BENCH_BATCH_SIZE=8
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
)

# Deeper config: 4 layers, BLOCK_SIZE=64 — long-context probe exercises sink.
add_demo(
  NAME    bench_attn_sink_deep_baseline
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
)

add_demo(
  NAME    bench_attn_sink_deep_on
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
)

# Sink-magnitude sensitivity sweep on the deeper config.
add_demo(
  NAME    bench_attn_sink_deep_on_strong
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=0.0
)

add_demo(
  NAME    bench_attn_sink_deep_on_weak
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-3.0
)

# ==================================================================
#  Q/K RMSNorm pre-dot A/B benchmark (DeepSeek-V4 §2.3.3 port)
#  Reuses the same A/B harness as the attention-sink benchmark.
#  See docs/research/RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md
# ==================================================================

# Tiny config (negative control)
add_demo(
  NAME    bench_qk_norm_baseline
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES BENCH_NUM_STEPS=600 BENCH_BATCH_SIZE=8
)

add_demo(
  NAME    bench_qk_norm_on
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES BENCH_NUM_STEPS=600 BENCH_BATCH_SIZE=8
          MICROGPT_QK_NORM=1
)

# Deep config (where QK norm should matter most)
add_demo(
  NAME    bench_qk_norm_deep_baseline
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
)

add_demo(
  NAME    bench_qk_norm_deep_on
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_QK_NORM=1
)

# Stress: high learning rate to surface attention-logit blow-up — the
# regime V4 cites as the main motivation. Without QK norm we expect the
# baseline to either NaN or train to a worse loss; with QK norm it should
# remain stable.
add_demo(
  NAME    bench_qk_norm_stress_baseline
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=800 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          LEARNING_RATE=0.02
)

add_demo(
  NAME    bench_qk_norm_stress_on
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=800 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          LEARNING_RATE=0.02
          MICROGPT_QK_NORM=1
)

# Combined: QK norm + attention sink. The two should compose cleanly.
add_demo(
  NAME    bench_qk_norm_deep_with_sink
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_QK_NORM=1 MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
)

# ==================================================================
#  MSA Sliding-Window Recency A/B benchmark (DeepSeek-V4 §2.3.3 port)
#  See docs/research/RESEARCH_DEEPSEEK_V4_MSA_SLIDING_WINDOW_RECENCY.md
# ==================================================================

# Deep config — block_size=64, long held-out sequence = 512 tokens,
# so the model crosses 8× the active-cache size and triggers many
# MSA chunking events.
add_demo(
  NAME    bench_msa_sliding_baseline
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=512
)

add_demo(
  NAME    bench_msa_sliding_on
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=512
          BENCH_MSA_SLIDING_WINDOW=1 MSA_WIN=16
)

# Window-size sensitivity sweep
add_demo(
  NAME    bench_msa_sliding_on_w8
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=512
          BENCH_MSA_SLIDING_WINDOW=1 MSA_WIN=8
)

add_demo(
  NAME    bench_msa_sliding_on_w32
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=512
          BENCH_MSA_SLIDING_WINDOW=1 MSA_WIN=32
)

# Longer-context scaling test: 2048-token held-out sequence (32× block_size)
add_demo(
  NAME    bench_msa_sliding_long_baseline
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
)

# ==================================================================
#  MSA CSA-style learnable pooling A/B sweep (DeepSeek-V4 §2.3.1 port)
#  Reuses the same long-context benchmark as the sliding-window paper.
#  See docs/research/RESEARCH_DEEPSEEK_V4_MSA_CSA_LEARNABLE_POOL.md
# ==================================================================
# Mode 0: uniform mean pool (baseline, existing behaviour)
add_demo(
  NAME    bench_msa_csa_mean
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          MSA_POOL_MODE=0
)

# Mode 1: linear ramp (newest token weighted ~2× oldest)
add_demo(
  NAME    bench_msa_csa_linear
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          MSA_POOL_MODE=1
)

# Mode 2: exponential recency (last few tokens dominate the pool)
add_demo(
  NAME    bench_msa_csa_exp
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          MSA_POOL_MODE=2
)

# Mode 3: content-aware (softmax of cosine-to-anchor)
add_demo(
  NAME    bench_msa_csa_content
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          MSA_POOL_MODE=3
)

# ==================================================================
#  MSA Lightning Indexer + Top-K A/B sweep (DeepSeek-V4 §2.3.1 port)
#  Reuses the same long-context benchmark.
#  See docs/research/RESEARCH_DEEPSEEK_V4_LIGHTNING_INDEXER_TOPK.md
# ==================================================================
# Baseline: existing top-1 cosine retrieval.
add_demo(
  NAME    bench_msa_indexer_baseline
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_ROUTING_MODE=0
)

# Lightning indexer at K=1 — isolates the scoring change from top-K.
add_demo(
  NAME    bench_msa_indexer_k1
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_ROUTING_MODE=1 BENCH_MSA_TOPK=1
)

# Lightning indexer at K=2.
add_demo(
  NAME    bench_msa_indexer_k2
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_ROUTING_MODE=1 BENCH_MSA_TOPK=2
)

# Lightning indexer at K=4.
add_demo(
  NAME    bench_msa_indexer_k4
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_ROUTING_MODE=1 BENCH_MSA_TOPK=4
)

# Lightning indexer at K=8 — half the active cache becomes pool-derived.
add_demo(
  NAME    bench_msa_indexer_k8
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_ROUTING_MODE=1 BENCH_MSA_TOPK=8
)

# Combined: Lightning Indexer K=4 + content-aware pool (mode 3).
add_demo(
  NAME    bench_msa_indexer_k4_csa
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_ROUTING_MODE=1 BENCH_MSA_TOPK=4 MSA_POOL_MODE=3
)

add_demo(
  NAME    bench_msa_sliding_long_on
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_SLIDING_WINDOW=1 MSA_WIN=32
)

# Run the full unit-test suite with QK_NORM=1 to verify the backward.
# The gradient_direction_reduces_loss and training_reduces_loss tests
# in test_microgpt.c are sensitive to a broken Q/K backward.
add_executable(test_microgpt_qk_norm tests/test_microgpt.c)
target_compile_definitions(test_microgpt_qk_norm PRIVATE MICROGPT_QK_NORM=1)
_microgpt_lib_for_defines(_QK_NORM_LIB MICROGPT_QK_NORM=1)
target_link_libraries(test_microgpt_qk_norm PRIVATE ${_QK_NORM_LIB})
if(NOT MSVC)
  target_link_libraries(test_microgpt_qk_norm PRIVATE m)
endif()
add_test(NAME microgpt_qk_norm_tests COMMAND test_microgpt_qk_norm
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# ==================================================================
#  Partial RoPE A/B benchmark (DeepSeek-V4 §2.3.3 port)
#  See docs/research/RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md
# ==================================================================
# Tiny config (negative control)
add_demo(
  NAME    bench_rope_baseline
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES BENCH_NUM_STEPS=600 BENCH_BATCH_SIZE=8
)

add_demo(
  NAME    bench_rope_on
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES BENCH_NUM_STEPS=600 BENCH_BATCH_SIZE=8
          MICROGPT_PARTIAL_ROPE=1
)

# Deep config — main standalone result.
add_demo(
  NAME    bench_rope_deep_baseline
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
)

add_demo(
  NAME    bench_rope_deep_on
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_PARTIAL_ROPE=1
)

# RoPE + attention sink — composition test
add_demo(
  NAME    bench_rope_deep_with_sink
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_PARTIAL_ROPE=1 MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
)

# RoPE + Q/K RMSNorm
add_demo(
  NAME    bench_rope_deep_with_qknorm
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_PARTIAL_ROPE=1 MICROGPT_QK_NORM=1
)

# Full V4-stack: RoPE + sink + Q/K RMSNorm — best expected combination
add_demo(
  NAME    bench_rope_deep_full_stack
  SOURCE  tests/bench_microgpt_attn_sink.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_PROBE_LEN=64
          MICROGPT_PARTIAL_ROPE=1 MICROGPT_QK_NORM=1
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
)

# Sliding-window MSA benchmark with RoPE on (the unblock-test from
# the sliding-window paper). Uses the long 2048-token harness.
add_demo(
  NAME    bench_rope_msa_sliding_baseline
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          MICROGPT_PARTIAL_ROPE=1
)

add_demo(
  NAME    bench_rope_msa_sliding_on
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_SLIDING_WINDOW=1 MSA_WIN=32
          MICROGPT_PARTIAL_ROPE=1
)

# ==================================================================
#  Sliding-window MSA with RoPE re-rotation on injection
#  Unblock-fix experiment from RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md §6.4.
#  See docs/research/RESEARCH_DEEPSEEK_V4_MSA_ROPE_REROTATE.md
# ==================================================================
# Same target, same flags, same MSA_WIN — but the bench now calls
# msa_recency_inject_rope() which re-rotates each cached K by the
# delta (new_pos − orig_pos) to restore relative-position semantics.
# This target shares its compile config with bench_rope_msa_sliding_on
# above; the change is *purely in the .c file* (already merged via
# the Edit above), so we add a separately-named copy here just so the
# binary is independently identifiable in the test results, and so a
# build with the prior bench_rope_msa_sliding_on still measures the
# pre-fix behaviour for direct comparison.
#
# (Note: because the Edit makes the new behaviour the default when
# both MICROGPT_PARTIAL_ROPE and BENCH_MSA_SLIDING_WINDOW are on, the
# existing bench_rope_msa_sliding_on target now ALSO uses the rope-
# aware injection. That binary's binary-name is preserved but its
# semantics have changed.)
add_demo(
  NAME    bench_rope_msa_sliding_rerotate
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_SLIDING_WINDOW=1 MSA_WIN=32
          MICROGPT_PARTIAL_ROPE=1
)

# ==================================================================
#  MSA pool rotation fix — un-rotate-before-pool, re-rotate-on-expand.
#  Closes the cross-regime gap (RoPE-on absolute MSA PPL was higher than
#  RoPE-off because mean-pool of rotated K vectors is undefined).
#  See docs/research/RESEARCH_DEEPSEEK_V4_MSA_POOL_ROPE_REROTATE.md
#
#  Both targets below use the rope-aware pool wrappers automatically
#  via #ifdef MICROGPT_PARTIAL_ROPE, since the bench source now calls
#  msa_pool_chunk_rope and msa_expand_context_rope when the flag is on.
#  These targets exist as separately-named binaries to make the
#  cross-regime comparison reproducible from the command line.
# ==================================================================
add_demo(
  NAME    bench_rope_msa_pool_rotated_baseline
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          MICROGPT_PARTIAL_ROPE=1
)

add_demo(
  NAME    bench_rope_msa_pool_rotated_sliding
  SOURCE  tests/bench_microgpt_msa_sliding.c
  COPY    demos/character-level/names/c_names.txt
  DEFINES N_EMBD=64 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=64 MLP_DIM=128
          BENCH_NUM_STEPS=1500 BENCH_BATCH_SIZE=16 BENCH_LONG_SEQ_LEN=2048
          BENCH_MSA_SLIDING_WINDOW=1 MSA_WIN=32
          MICROGPT_PARTIAL_ROPE=1
)

# Full unit-test suite with PARTIAL_ROPE=1 to validate the RoPE backward.
add_executable(test_microgpt_rope tests/test_microgpt.c)
target_compile_definitions(test_microgpt_rope PRIVATE MICROGPT_PARTIAL_ROPE=1)
_microgpt_lib_for_defines(_ROPE_LIB MICROGPT_PARTIAL_ROPE=1)
target_link_libraries(test_microgpt_rope PRIVATE ${_ROPE_LIB})
if(NOT MSVC)
  target_link_libraries(test_microgpt_rope PRIVATE m)
endif()
add_test(NAME microgpt_rope_tests COMMAND test_microgpt_rope
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

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

# AttnRes A/B benchmark: same config with Block Attention Residuals enabled
add_demo(
  NAME    c_optim_shk_attnres
  SOURCE  demos/character-level/shakespeare/main.c
  THREADS
  COPY    demos/character-level/shakespeare/c_shakespeare.txt
  DEFINES N_EMBD=128 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=512
          NUM_STEPS=2000 LEARNING_RATE=0.001 BATCH_SIZE=4
          MAX_VOCAB=200 MAX_DOCS=200000 MAX_DOC_LEN=512
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1
          MICROGPT_ATTN_RES=1 ATTN_RES_BLOCK_SIZE=2
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

# VM codegen A/B: N_LAYER=4 (deep — no AttnRes)
add_demo(
  NAME    w_vm_codegen_deep
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
)
target_link_libraries(w_vm_codegen_deep PRIVATE microgpt_vm_lib)

# Deep codegen + full V4 stack A/B comparison.
# This is the deep config sweet spot where the V4 papers measured
# the headline -8.7% PPL win. Codegen is densely relative-positional
# (RoPE), has prompt-vs-body asymmetry (sinks), and benefits from
# stable attention magnitudes (Q/K RMSNorm). Both targets fresh-train
# with NUM_STEPS=5000 from distinct checkpoints for clean A/B.
add_demo(
  NAME    w_vm_codegen_deep_v4baseline
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
          CHECKPOINT_FILE="w_vm_codegen_deep_v4baseline.ckpt"
)
target_link_libraries(w_vm_codegen_deep_v4baseline PRIVATE microgpt_vm_lib)

add_demo(
  NAME    w_vm_codegen_deep_v4
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
          MICROGPT_PARTIAL_ROPE=1
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
          MICROGPT_QK_NORM=1
          CHECKPOINT_FILE="w_vm_codegen_deep_v4.ckpt"
)
target_link_libraries(w_vm_codegen_deep_v4 PRIVATE microgpt_vm_lib)

# Single-flag ablation to identify which V4 port is the culprit for the
# 30-percentage-point regression observed with the full stack on
# w_vm_codegen_deep_v4. Hypothesis (per commit e92129d): RoPE removes
# absolute-position structure that grammar-rigid codegen relies on, so
# RoPE-only should be the biggest hurter while sink-only and qknorm-only
# may be near-neutral.
add_demo(
  NAME    w_vm_codegen_deep_v4_rope_only
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
          MICROGPT_PARTIAL_ROPE=1
          CHECKPOINT_FILE="w_vm_codegen_deep_v4_rope.ckpt"
)
target_link_libraries(w_vm_codegen_deep_v4_rope_only PRIVATE microgpt_vm_lib)

add_demo(
  NAME    w_vm_codegen_deep_v4_sink_only
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
          CHECKPOINT_FILE="w_vm_codegen_deep_v4_sink.ckpt"
)
target_link_libraries(w_vm_codegen_deep_v4_sink_only PRIVATE microgpt_vm_lib)

add_demo(
  NAME    w_vm_codegen_deep_v4_qknorm_only
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
          MICROGPT_QK_NORM=1
          CHECKPOINT_FILE="w_vm_codegen_deep_v4_qknorm.ckpt"
)
target_link_libraries(w_vm_codegen_deep_v4_qknorm_only PRIVATE microgpt_vm_lib)

# Sink + Q/K RMSNorm only (no RoPE) — the "safe-for-codegen" V4 subset
# if the hypothesis holds. Should match or beat baseline.
add_demo(
  NAME    w_vm_codegen_deep_v4_sink_qknorm
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
          MICROGPT_QK_NORM=1
          CHECKPOINT_FILE="w_vm_codegen_deep_v4_sink_qknorm.ckpt"
)
target_link_libraries(w_vm_codegen_deep_v4_sink_qknorm PRIVATE microgpt_vm_lib)

# VM codegen A/B: N_LAYER=4 + AttnRes (block=2)
add_demo(
  NAME    w_vm_codegen_attnres
  SOURCE  demos/word-level/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/word-level/vm_codegen/w_vm_functions_pretok.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          WORD_VOCAB_SIZE=900
          MICROGPT_ATTN_RES=1 ATTN_RES_BLOCK_SIZE=2
)
target_link_libraries(w_vm_codegen_attnres PRIVATE microgpt_vm_lib)

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
  DEFINES N_EMBD=48 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=192
          NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=10000 MAX_DOC_LEN=128
          GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01
)

# ==================================================================
#  Memory Sparse Attention (MSA) Integration Demos
# ==================================================================

add_demo(
  NAME    msa_latent_handoff
  SOURCE  demos/msa/latent_handoff/main.c
  THREADS
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=1 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=500 MAX_DOC_LEN=128
)
target_include_directories(msa_latent_handoff PRIVATE src)

add_demo(
  NAME    msa_fraud_guardian
  SOURCE  demos/msa/fraud_guardian/main.c
  THREADS
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=1 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=500 MAX_DOC_LEN=128
)
target_include_directories(msa_fraud_guardian PRIVATE src)

add_demo(
  NAME    msa_semantic_companion
  SOURCE  demos/msa/semantic_companion/main.c
  THREADS
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=1 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=500 MAX_DOC_LEN=128
)
target_include_directories(msa_semantic_companion PRIVATE src)

add_demo(
  NAME    msa_context_extender
  SOURCE  demos/msa/context_extender/main.c
  THREADS
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=1 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=500 MAX_DOC_LEN=128
)
target_include_directories(msa_context_extender PRIVATE src)

add_demo(
  NAME    msa_pipeline_bench
  SOURCE  demos/msa/pipeline_bench/main.c
  THREADS
  DEFINES N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
          NUM_STEPS=1 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=50 MAX_DOCS=500 MAX_DOC_LEN=128
)
target_include_directories(msa_pipeline_bench PRIVATE src)

add_demo(
  NAME    msa_infinite_shakespeare
  SOURCE  demos/msa/infinite_shakespeare/main.c
  THREADS
  COPY    demos/msa/infinite_shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=1 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
)
target_include_directories(msa_infinite_shakespeare PRIVATE src)

# Same demo with the V4 active-attention stack (RoPE + sink + Q/K RMSNorm)
# and the rope-aware MSA pool/expand wrappers. Fresh training (5000 steps)
# since the legacy checkpoint isn't compatible with the rope-rotated
# attention path. Separate checkpoint filename so it doesn't collide.
# See docs/research/RESEARCH_DEEPSEEK_V4_PARTIAL_ROPE.md and
# RESEARCH_DEEPSEEK_V4_MSA_POOL_ROPE_REROTATE.md.
add_demo(
  NAME    msa_infinite_shakespeare_v4
  SOURCE  demos/msa/infinite_shakespeare/main.c
  THREADS
  COPY    demos/msa/infinite_shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
          MICROGPT_PARTIAL_ROPE=1
          MICROGPT_ATTN_SINK=1 ATTN_SINK_LOGIT=-1.0
          MICROGPT_QK_NORM=1
          CHECKPOINT_FILE="w_shakespeare_v4.ckpt"
)
target_include_directories(msa_infinite_shakespeare_v4 PRIVATE src)

# Apples-to-apples baseline: same arch and step count as the V4 variant,
# but with all V4 flags off. Distinct checkpoint name so it can be built
# and run alongside the v4 variant for direct comparison.
add_demo(
  NAME    msa_infinite_shakespeare_v4_baseline
  SOURCE  demos/msa/infinite_shakespeare/main.c
  THREADS
  COPY    demos/msa/infinite_shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=8
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
          CHECKPOINT_FILE="w_shakespeare_v4baseline.ckpt"
)
target_include_directories(msa_infinite_shakespeare_v4_baseline PRIVATE src)

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
#  Manifold Classifier Diagnostic — Phase 1b
#  Standalone demo testing whether geodesic-based family classification
#  can predict the right family for the prompts where the wiring
#  system unanimously fails. Confirms whether the bottleneck is
#  generation (manifold thesis valid) or classification (manifold
#  thesis weak).
# ==================================================================

add_demo(
  NAME   manifold_classifier_demo
  SOURCE demos/manifold_classifier/main.c
  COPY   demos/wiring_organelle/pipeline_corpus_held_out.txt
)

# Phase 3a-lite — TF-IDF centroid classifier (learned baseline). See
# RESEARCH_PIPELINE_IR.md §40.7 for the scoping rationale.
add_demo(
  NAME   manifold_tfidf_demo
  SOURCE demos/manifold_classifier/tfidf_main.c
  COPY   demos/wiring_organelle/pipeline_corpus_held_out.txt
         demos/wiring_organelle/pipeline_corpus_adversarial.txt
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

add_executable(test_microgpt_msa tests/test_microgpt_msa.c)
target_link_libraries(test_microgpt_msa PRIVATE microgpt_lib)
target_include_directories(test_microgpt_msa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt_msa PRIVATE m)
endif()
add_test(NAME microgpt_msa_tests COMMAND test_microgpt_msa WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# EML Organelle — snapped-tree evaluator (microgpt_eml.{h,c}).
# See docs/research/RESEARCH_EML_ORGANELLE.md for the integration story.
add_executable(test_microgpt_eml tests/test_microgpt_eml.c)
target_link_libraries(test_microgpt_eml PRIVATE microgpt_lib)
target_include_directories(test_microgpt_eml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt_eml PRIVATE m)
endif()
add_test(NAME microgpt_eml_tests COMMAND test_microgpt_eml WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Graph Pipeline IR — Phase 1 (RESEARCH_PIPELINE_IR.md).
# pipeline_ir_vm.c is the opt-in VM-backed dispatcher exposed by the
# libs/pipeline_ir/ subtree; consumers compile it into their own
# target alongside a vm_engine implementation (here: microgpt_vm.c).
add_executable(test_microgpt_pipeline
  tests/test_microgpt_pipeline.c
  ${PIPELINE_IR_VM_SOURCE}
  src/microgpt_vm.c
  src/microgpt_vm_parser.l.c
  src/microgpt_vm_parser.tab.c)
target_link_libraries(test_microgpt_pipeline PRIVATE microgpt_lib)
target_include_directories(test_microgpt_pipeline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tests
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${PIPELINE_IR_INTERNAL_INCLUDE_DIR})
if(NOT MSVC)
  target_link_libraries(test_microgpt_pipeline PRIVATE m)
endif()
add_test(NAME microgpt_pipeline_tests COMMAND test_microgpt_pipeline WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Vietoris-Rips persistent cohomology engine — Phase 1a manifold-learning lift.
# Lifted from sibling /Users/user/dev/projects/microgpt-c (C99 implementation,
# fixed-point at VR_MAX_DIMS=12, VR_MAX_PTS=64).
add_executable(test_microgpt_vr tests/test_microgpt_vr.c)
target_link_libraries(test_microgpt_vr PRIVATE microgpt_lib)
target_include_directories(test_microgpt_vr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt_vr PRIVATE m)
endif()
add_test(NAME microgpt_vr_tests COMMAND test_microgpt_vr WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Geodesic Riemannian solver — Phase 1b manifold-learning lift.
# Lifted from sibling. C99, GEO_DIMS=12 fixed-point.
add_executable(test_microgpt_geodesic tests/test_microgpt_geodesic.c)
target_link_libraries(test_microgpt_geodesic PRIVATE microgpt_lib)
target_include_directories(test_microgpt_geodesic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt_geodesic PRIVATE m)
endif()
add_test(NAME microgpt_geodesic_tests COMMAND test_microgpt_geodesic WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# EKAN B-spline basis — Phase 1b manifold-learning lift.
# Header-only fixed-point cubic B-spline.
add_executable(test_microgpt_ekan tests/test_microgpt_ekan.c)
target_link_libraries(test_microgpt_ekan PRIVATE microgpt_lib)
target_include_directories(test_microgpt_ekan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt_ekan PRIVATE m)
endif()
add_test(NAME microgpt_ekan_tests COMMAND test_microgpt_ekan WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# EKAN autoencoder network — Phase 1b manifold-learning lift.
# Multi-layer KAN with Fourier activations + Adam.
add_executable(test_microgpt_ekan_network tests/test_microgpt_ekan_network.c)
target_link_libraries(test_microgpt_ekan_network PRIVATE microgpt_lib)
target_include_directories(test_microgpt_ekan_network PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt_ekan_network PRIVATE m)
endif()
add_test(NAME microgpt_ekan_network_tests COMMAND test_microgpt_ekan_network WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Stream B of the compositional generator fix — type-directed search +
# primitive manifest. Links the pipeline IR (in microgpt_lib) plus the
# opt-in VM-backed dispatcher and the VM library, mirroring how
# test_microgpt_pipeline brings them in.
add_executable(test_microgpt_wiring_compositional
  tests/test_microgpt_wiring_compositional.c
  demos/wiring_organelle/wiring_primitive_manifest.c
  demos/wiring_organelle/wiring_compositional_search.c
  demos/wiring_organelle/wiring_arg_binder.c
  demos/wiring_organelle/wiring_geo_classifier.c
  demos/wiring_organelle/wiring_natives.c
  ${PIPELINE_IR_VM_SOURCE}
  src/microgpt_vm.c
  src/microgpt_vm_parser.l.c
  src/microgpt_vm_parser.tab.c)
target_link_libraries(test_microgpt_wiring_compositional PRIVATE microgpt_lib)
target_include_directories(test_microgpt_wiring_compositional PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tests
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle
  ${PIPELINE_IR_INTERNAL_INCLUDE_DIR})
if(NOT MSVC)
  target_link_libraries(test_microgpt_wiring_compositional PRIVATE m)
endif()
add_test(NAME microgpt_wiring_compositional_tests
         COMMAND test_microgpt_wiring_compositional
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Phase 5 held-out harness — runs the type-directed compositional search
# over the 30-prompt leakage-audited held-out set, dispatches via the VM,
# scores against the reference oracle, and prints per-axis stats.
add_executable(wiring_phase5_harness
  demos/wiring_organelle/wiring_phase5_harness.c
  demos/wiring_organelle/wiring_primitive_manifest.c
  demos/wiring_organelle/wiring_compositional_search.c
  demos/wiring_organelle/wiring_arg_binder.c
  demos/wiring_organelle/wiring_geo_classifier.c
  demos/wiring_organelle/wiring_references.c
  ${PIPELINE_IR_VM_SOURCE}
  src/microgpt_vm.c
  src/microgpt_vm_parser.l.c
  src/microgpt_vm_parser.tab.c)
target_link_libraries(wiring_phase5_harness PRIVATE microgpt_lib)
target_include_directories(wiring_phase5_harness PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle
  ${PIPELINE_IR_INTERNAL_INCLUDE_DIR})
if(NOT MSVC)
  target_link_libraries(wiring_phase5_harness PRIVATE m)
endif()
add_custom_command(TARGET wiring_phase5_harness POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_compositional_test.txt
          $<TARGET_FILE_DIR:wiring_phase5_harness>/pipeline_corpus_compositional_test.txt)

# Pipeline IR corpus generator (Phase 3a) — produces hand-curated
# (prompt, graph) training pairs by building Pipeline IR objects
# programmatically and rendering to text. See
# docs/research/RESEARCH_PIPELINE_IR.md §12.
add_executable(pipeline_corpus_gen tools/pipeline_corpus_gen.c)
target_link_libraries(pipeline_corpus_gen PRIVATE microgpt_lib)
target_include_directories(pipeline_corpus_gen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(pipeline_corpus_gen PRIVATE m)
endif()

# Phase 4a — expanded corpus generator (synonym multiplication +
# template variation + word-order permutations). See
# RESEARCH_PIPELINE_IR.md §45 for the pre-registered design.
add_executable(corpus_expand tools/corpus_expand.c)
if(NOT MSVC)
  target_link_libraries(corpus_expand PRIVATE m)
endif()

# Phase 5a — parser re-eval tool. Reads a wiring_organelle_demo log and
# re-parses each "best output" graph through both strict and tolerant
# parsers; reports the verify/fidelity delta. No retrain needed.
add_executable(reeval_parser tools/reeval_parser.c)
target_link_libraries(reeval_parser PRIVATE microgpt_lib)
target_include_directories(reeval_parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(reeval_parser PRIVATE m)
endif()

# ──────────────────────────────────────────────────────────────────
# E13 — LLM-as-Connect-4-teacher distillation corpus generator.
#
# Pre-registered in experiments/E13-llm-game-distillation.md §1.3.
# Standalone CLI: plays N Connect-4 games with the LLM as X, random
# as O, and emits (board, move) pairs from the LLM's wins to a
# corpus file the OQL TRAIN adapter consumes unchanged.
#
# No engine link, no microgpt_lib variant — talks to LM Studio via
# curl subprocess only.  Zero new build deps (T8).
# ──────────────────────────────────────────────────────────────────
add_executable(c4_distill_corpus_gen
    tools/c4_distill_corpus_gen.c
    tools/llm_game_player.c)
target_include_directories(c4_distill_corpus_gen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools)
# Smoke test — generator runs end-to-end with zero failures.
add_test(NAME pipeline_corpus_smoke
         COMMAND pipeline_corpus_gen ${CMAKE_BINARY_DIR}/_corpus_smoke.txt
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Wiring Organelle demo (Pipeline IR Phase 3c) — train a small word-level
# Transformer on the templated corpus, evaluate novel-prompt generation.
# See docs/research/RESEARCH_PIPELINE_IR.md §13.
#
# NOTE: V4 active-attention stack (PARTIAL_ROPE/ATTN_SINK/QK_NORM) is
# DELIBERATELY NOT enabled — codegen is grammar-rigid and the V4 stack
# regresses such tasks by 30pp per the prior ablation.
add_demo(
  NAME    wiring_organelle_demo
  SOURCE  demos/wiring_organelle/main.c
  THREADS
  # Phase 9 attempted 1.49M params (128-emb / 6-layer / 8000 steps) and
  # regressed: 35% → 30% correct due to overfitting on 272 examples.
  # Phase 10 reverts to the 540K sweet spot (Phase 8 config) and adds
  # argument-order paraphrases to the corpus instead. See §23 of
  # RESEARCH_PIPELINE_IR.md.
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=384 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=1500 MAX_DOCS=5000 MAX_DOC_LEN=4096
)
# Phase 6 — register wiring native primitives (~40 C functions matching
# w_vm_functions.txt) as the dispatch backend for pipeline_execute().
# Phase 7 — reference-answer correctness suite for held-out NL prompts.
target_sources(wiring_organelle_demo PRIVATE
  demos/wiring_organelle/wiring_natives.c
  demos/wiring_organelle/wiring_references.c
  demos/wiring_organelle/wiring_geo_classifier.c
  demos/wiring_organelle/wiring_anchor_graphs.c
  demos/wiring_organelle/wiring_fragments.c
  demos/wiring_organelle/wiring_primitive_manifest.c
  demos/wiring_organelle/wiring_compositional_search.c
  demos/wiring_organelle/wiring_arg_binder.c)
target_include_directories(wiring_organelle_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle)
# Generate the wiring + planner corpus next to the binary as a POST_BUILD step.
# Phase 15: 4-arg invocation also emits pipeline_corpus_planner.txt for the
# planner organelle (template-family classifier).
add_custom_command(TARGET wiring_organelle_demo POST_BUILD
  COMMAND pipeline_corpus_gen
          $<TARGET_FILE_DIR:wiring_organelle_demo>/pipeline_corpus_train.txt
          $<TARGET_FILE_DIR:wiring_organelle_demo>/pipeline_corpus_val.txt
          $<TARGET_FILE_DIR:wiring_organelle_demo>/pipeline_corpus_planner.txt
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_held_out.txt
          $<TARGET_FILE_DIR:wiring_organelle_demo>/pipeline_corpus_held_out.txt
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_composition.txt
          $<TARGET_FILE_DIR:wiring_organelle_demo>/pipeline_corpus_composition.txt
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_scaling_heldout.txt
          $<TARGET_FILE_DIR:wiring_organelle_demo>/pipeline_corpus_scaling_heldout.txt
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_scaling_heldout_v2.txt
          $<TARGET_FILE_DIR:wiring_organelle_demo>/pipeline_corpus_scaling_heldout_v2.txt
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_scaling_heldout_v3.txt
          $<TARGET_FILE_DIR:wiring_organelle_demo>/pipeline_corpus_scaling_heldout_v3.txt
  COMMAND sh
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_held_out_leakage.sh
          $<TARGET_FILE_DIR:wiring_organelle_demo>
  DEPENDS pipeline_corpus_gen
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_held_out.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_composition.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_scaling_heldout.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_scaling_heldout_v2.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/demos/wiring_organelle/pipeline_corpus_scaling_heldout_v3.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_held_out_leakage.sh
  COMMENT "Generating Pipeline IR corpus (wiring + planner), copying held-out + composition, leakage check"
)
# Phase 16 attempted family-prefixed wiring training (--prefix-family arg)
# but regressed 80% → 75% — vocab grew 1051 → 1211 tokens, distribution
# shift hurt verify rate. Reverted; canonical baseline = Phase 15. The
# CLI-side --prefix-family option in pipeline_corpus_gen.c is kept
# (additive flag, no-op when not passed) for future experiments.

add_executable(test_microgpt_turboquant tests/test_microgpt_turboquant.c)
target_link_libraries(test_microgpt_turboquant PRIVATE microgpt_lib)
target_include_directories(test_microgpt_turboquant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt_turboquant PRIVATE m)
endif()
add_test(NAME microgpt_turboquant_tests COMMAND test_microgpt_turboquant WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(test_microgpt_rotorquant tests/test_microgpt_rotorquant.c)
target_link_libraries(test_microgpt_rotorquant PRIVATE microgpt_lib)
target_include_directories(test_microgpt_rotorquant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt_rotorquant PRIVATE m)
endif()
add_test(NAME microgpt_rotorquant_tests COMMAND test_microgpt_rotorquant WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# ==================================================================
#  Benchmarks
# ==================================================================

add_executable(bench_microgpt tests/bench_microgpt.c)
target_link_libraries(bench_microgpt PRIVATE microgpt_lib)
if(NOT MSVC)
  target_link_libraries(bench_microgpt PRIVATE m)
endif()

add_executable(bench_microgpt_msa tests/bench_microgpt_msa.c)
target_link_libraries(bench_microgpt_msa PRIVATE microgpt_lib)
target_include_directories(bench_microgpt_msa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(bench_microgpt_msa PRIVATE m)
endif()

add_executable(bench_microgpt_turboquant tests/bench_microgpt_turboquant.c)
target_link_libraries(bench_microgpt_turboquant PRIVATE microgpt_lib)
target_include_directories(bench_microgpt_turboquant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(bench_microgpt_turboquant PRIVATE m)
endif()

add_executable(bench_microgpt_rotorquant tests/bench_microgpt_rotorquant.c)
target_link_libraries(bench_microgpt_rotorquant PRIVATE microgpt_lib)
target_include_directories(bench_microgpt_rotorquant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(bench_microgpt_rotorquant PRIVATE m)
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

add_executable(bench_ssd tests/bench_ssd.c)
target_link_libraries(bench_ssd PRIVATE microgpt_lib)
if(NOT MSVC)
  target_link_libraries(bench_ssd PRIVATE m)
endif()



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

add_library(microgpt_vm_lib STATIC
    src/microgpt_vm.c
    src/microgpt_vm_natives.c
    ${VM_PARSER_SOURCES})
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

# ==================================================================
#  OQL Engine — Experiment E07 (Organelle Query Language)
# ==================================================================
#
# See experiments/E07-oql-dsl.md and docs/research/OQL_GRAMMAR_REFERENCE.md.
# Mirrors the VM's Flex/Bison wiring: regenerate if Bison >= 3.0 is found,
# otherwise fall back to the committed pre-generated parser sources.

set(OQL_PARSER_SOURCES)
if(FLEX_FOUND AND BISON_FOUND AND BISON_VERSION VERSION_GREATER_EQUAL "3.0")
    flex_target(oql_scanner
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_oql.l
        ${CMAKE_CURRENT_BINARY_DIR}/microgpt_oql_parser.l.c)
    bison_target(oql_parser
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_oql.y
        ${CMAKE_CURRENT_BINARY_DIR}/microgpt_oql_parser.tab.c
        DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/microgpt_oql_parser.tab.h)
    add_flex_bison_dependency(oql_scanner oql_parser)
    list(APPEND OQL_PARSER_SOURCES
        ${FLEX_oql_scanner_OUTPUTS}
        ${BISON_oql_parser_OUTPUTS})
else()
    list(APPEND OQL_PARSER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_oql_parser.l.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/microgpt_oql_parser.tab.c)
endif()

add_library(microgpt_oql_lib STATIC
    src/microgpt_oql.c
    src/oql_runtime_games.c
    src/oql_runtime_train.c
    ${OQL_PARSER_SOURCES})
target_include_directories(microgpt_oql_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR})
# pipeline_verify lives in microgpt_lib (microgpt_pipeline.c); link it in.
# microgpt_vm_lib provides vm_module_compile / vm_module_dispose used by
# the E09 runtime (oql_execute_with_runtime + behaviour dispatch).
target_link_libraries(microgpt_oql_lib PUBLIC microgpt_lib microgpt_vm_lib)

add_executable(oql tools/oql_runner.c)
target_link_libraries(oql PRIVATE microgpt_oql_lib)
target_include_directories(oql PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(oql PRIVATE m)
endif()

# ──────────────────────────────────────────────────────────────────
#  e12_generate — LLM-as-corpus-source bridge (Experiment E12).
#
#  Reads an OQL script containing a CREATE CORPUS … FROM LLM …
#  statement and actually invokes the LM Studio endpoint via curl
#  (subprocess) to produce the materialised corpus file.  Filters
#  each emission through pipeline_ir.verify() (if VERIFY_VIA is set)
#  and the Jaccard audit against a held-out (if AUDIT_AGAINST is set).
#
#  Tool-only — never linked into any organelle runtime binary; the
#  trained wiring organelle MUST NOT call the LLM at inference.
#  T8 hard-lock holds: zero new build deps beyond curl.
# ──────────────────────────────────────────────────────────────────
add_executable(e12_generate
    tools/e12_generate.c
    tools/llm_corpus_source.c)
target_link_libraries(e12_generate PRIVATE microgpt_oql_lib)
target_include_directories(e12_generate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tools)
if(NOT MSVC)
    target_link_libraries(e12_generate PRIVATE m)
endif()


# ──────────────────────────────────────────────────────────────────
#  oql_c4 — Connect-4-dims variant of the `oql` CLI.
#
#  E09 Pathway A: the engine's matmul hot loops constant-fold N_EMBD,
#  N_HEAD, etc. as compile-time #define macros (CLAUDE.md "Compile-time
#  architecture").  Loading a Connect-4 checkpoint (trained with
#  N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384) into the
#  default OQL binary (N_EMBD=16) would silently produce garbage —
#  the matmul shapes don't match.
#
#  This variant builds a SECOND oql_runtime_games.o + oql_runner.o pair
#  against the Connect-4-shaped microgpt_lib variant, so the OQL CLI
#  can load c_connect4_player.ckpt without dimension mismatch.
#
#  Researchers run:
#      ./build/c_connect4_demo                   # trains + saves the ckpt
#      cp c_connect4_player.ckpt build/checkpoints/   # (one-time copy)
#      ./build/oql_c4 run ../experiments/connect4.oql
# ──────────────────────────────────────────────────────────────────
set(_C4_DEFINES
    N_EMBD=96 N_HEAD=8 N_LAYER=4 BLOCK_SIZE=128 MLP_DIM=384
    NUM_STEPS=25000 LEARNING_RATE=0.001 BATCH_SIZE=8
    MAX_VOCAB=50 MAX_DOCS=5000 MAX_DOC_LEN=128
    GRAD_CLIP=1.0 LABEL_SMOOTH=0.1 WEIGHT_DECAY=0.01)
_microgpt_lib_for_defines(_C4_LIB ${_C4_DEFINES})

# Rebuild oql_runtime_games.c + microgpt_oql.c with the C4 macros so
# oql_default_cfg() returns matching dims for checkpoint_load.
add_executable(oql_c4
    tools/oql_runner.c
    src/oql_runtime_games.c
    src/oql_runtime_train.c
    src/microgpt_oql.c
    ${OQL_PARSER_SOURCES})
target_link_libraries(oql_c4 PRIVATE ${_C4_LIB} microgpt_vm_lib)
target_include_directories(oql_c4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(oql_c4 PRIVATE ${_C4_DEFINES})
if(NOT MSVC)
    target_link_libraries(oql_c4 PRIVATE m)
endif()

# ──────────────────────────────────────────────────────────────────
#  oql_wiring — Wiring-dims variant of the `oql` CLI (Experiment E12).
#
#  Same compile-time-macro discipline as oql_c4: the wiring organelle
#  was trained at N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=384 MLP_DIM=384,
#  so a TRAIN dispatch via the OQL CLI must compile against the same
#  macros for matmul dimensions to match.  Researchers run:
#      ./build/oql_wiring run ../experiments/E12-train.oql
# ──────────────────────────────────────────────────────────────────
set(_WIRING_DEFINES
    N_EMBD=96 N_HEAD=4 N_LAYER=4 BLOCK_SIZE=384 MLP_DIM=384
    NUM_STEPS=5000 LEARNING_RATE=0.001 BATCH_SIZE=16
    MAX_VOCAB=1500 MAX_DOCS=5000 MAX_DOC_LEN=4096)
_microgpt_lib_for_defines(_WIRING_LIB ${_WIRING_DEFINES})
add_executable(oql_wiring
    tools/oql_runner.c
    src/oql_runtime_games.c
    src/oql_runtime_train.c
    src/microgpt_oql.c
    ${OQL_PARSER_SOURCES})
target_link_libraries(oql_wiring PRIVATE ${_WIRING_LIB} microgpt_vm_lib)
target_include_directories(oql_wiring PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(oql_wiring PRIVATE ${_WIRING_DEFINES})
if(NOT MSVC)
    target_link_libraries(oql_wiring PRIVATE m)
endif()

# ──────────────────────────────────────────────────────────────────
#  e12_eval_v2 — Phase 5 evaluator for the LLM-trained wiring organelle.
#
#  Loads a wiring-dim checkpoint, rebuilds the WordVocab from the same
#  training corpus, generates completions for each pipeline_corpus_scaling_
#  heldout_v2.txt prompt, parses+verifies via libpipeline_ir, and reports
#  the T4 (verified rate) headline.
# ──────────────────────────────────────────────────────────────────
add_executable(e12_eval_v2 tools/e12_eval_v2.c)
target_link_libraries(e12_eval_v2 PRIVATE ${_WIRING_LIB})
target_include_directories(e12_eval_v2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/pipeline_ir/include)
target_compile_definitions(e12_eval_v2 PRIVATE ${_WIRING_DEFINES})
if(NOT MSVC)
    target_link_libraries(e12_eval_v2 PRIVATE m)
endif()

add_executable(test_microgpt_oql tests/test_microgpt_oql.c)
# microgpt_vm_lib is wired in so the E08 connect4 worked-example test can
# compile + run each parsed CREATE BEHAVIOUR's VM body — see
# tests/test_microgpt_oql.c::test_e08_connect4_behaviours.
target_link_libraries(test_microgpt_oql PRIVATE microgpt_oql_lib microgpt_vm_lib)
target_include_directories(test_microgpt_oql PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(test_microgpt_oql PRIVATE m)
endif()
add_test(NAME microgpt_oql_tests COMMAND test_microgpt_oql WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# ──────────────────────────────────────────────────────────────────
#  test_microgpt_oql_train — E10 loss-curve fidelity smoke test.
#
#  Links against the *default* microgpt_lib variant (no DEFINES) so
#  the engine macros (N_EMBD=16 N_LAYER=1 ...) match names_demo's
#  config exactly.  This avoids the E09 §3.4 compile-time-macro
#  silent-failure mode that bit the Connect-4 OQL CLI: the smoke
#  test trains a names-shaped model in OQL and a names-shaped model
#  in C with byte-identical hyperparameters, and asserts the
#  per-step loss curves agree within the pre-reg ±10% relative budget.
#
#  Copies c_names.txt next to the test binary so it can be run
#  from build/ as `./test_microgpt_oql_train`.
# ──────────────────────────────────────────────────────────────────
add_executable(test_microgpt_oql_train tests/test_microgpt_oql_train.c)
target_link_libraries(test_microgpt_oql_train PRIVATE microgpt_oql_lib)
target_include_directories(test_microgpt_oql_train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(test_microgpt_oql_train PRIVATE m)
endif()
add_custom_command(TARGET test_microgpt_oql_train POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/names/c_names.txt"
        "$<TARGET_FILE_DIR:test_microgpt_oql_train>/c_names.txt"
    COMMENT "Copying c_names.txt for test_microgpt_oql_train"
)
add_test(NAME microgpt_oql_train_tests
         COMMAND test_microgpt_oql_train
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Long-horizon (5000-step) loss-curve benchmark — kept out of ctest
# (3-10s runtime) but registered so engineers can re-measure when the
# adapter changes.  Run from build/ as `./bench_microgpt_oql_train`.
add_executable(bench_microgpt_oql_train tests/bench_microgpt_oql_train.c)
target_link_libraries(bench_microgpt_oql_train PRIVATE microgpt_oql_lib)
target_include_directories(bench_microgpt_oql_train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(bench_microgpt_oql_train PRIVATE m)
endif()
add_custom_command(TARGET bench_microgpt_oql_train POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/demos/character-level/names/c_names.txt"
        "$<TARGET_FILE_DIR:bench_microgpt_oql_train>/c_names.txt"
    COMMENT "Copying c_names.txt for bench_microgpt_oql_train"
)

# ==================================================================
#  TurboQuant Integration Evaluation Targets
# ==================================================================

# 1) Shakespeare Word-Level + Infinite MSA (Baseline FP32)
add_demo(
  NAME    tq_shakespeare_base
  SOURCE  demos/turbo_quant/shakespeare/main.c
  THREADS
  COPY    demos/turbo_quant/shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=0 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
)

# 2) Shakespeare Word-Level + Infinite MSA + TurboQuant (Compressed)
add_demo(
  NAME    tq_shakespeare_tq
  SOURCE  demos/turbo_quant/shakespeare/main.c
  THREADS
  COPY    demos/turbo_quant/shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=0 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
          ENABLE_TURBOQUANT=1
)

# 3) VM Codegen + Infinite MSA (Baseline FP32)
add_demo(
  NAME    tq_vm_codegen_base
  SOURCE  demos/turbo_quant/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/turbo_quant/vm_codegen/c_vm_functions_combined.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=0 LEARNING_RATE=0.0005 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
)
target_link_libraries(tq_vm_codegen_base PRIVATE microgpt_vm_lib)

# 4) VM Codegen + Infinite MSA + TurboQuant (Compressed)
add_demo(
  NAME    tq_vm_codegen_tq
  SOURCE  demos/turbo_quant/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/turbo_quant/vm_codegen/c_vm_functions_combined.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=0 LEARNING_RATE=0.0005 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          ENABLE_TURBOQUANT=1
)
target_link_libraries(tq_vm_codegen_tq PRIVATE microgpt_vm_lib)

# 4b) VM Codegen + TurboQuant — Q/K RMSNorm A/B comparison.
#     Both targets fresh-train (NUM_STEPS=5000), differ only by the
#     MICROGPT_QK_NORM flag. Q/K RMSNorm is the V4 port specifically
#     recommended for noisy-K regimes — exactly what 4-bit dequantised
#     K vectors produce. See docs/research/RESEARCH_DEEPSEEK_V4_QK_RMSNORM_PREDOT.md
#     and the codegen-fit analysis in the conversation transcript.
add_demo(
  NAME    tq_vm_codegen_tq_v4baseline
  SOURCE  demos/turbo_quant/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/turbo_quant/vm_codegen/c_vm_functions_combined.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.0005 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          ENABLE_TURBOQUANT=1
          CHECKPOINT_FILE="tq_vm_codegen_v4baseline.ckpt"
          TRAINING_LOG="tq_vm_codegen_v4baseline.log"
)
target_link_libraries(tq_vm_codegen_tq_v4baseline PRIVATE microgpt_vm_lib)

add_demo(
  NAME    tq_vm_codegen_tq_qknorm
  SOURCE  demos/turbo_quant/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/turbo_quant/vm_codegen/c_vm_functions_combined.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=5000 LEARNING_RATE=0.0005 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          ENABLE_TURBOQUANT=1
          MICROGPT_QK_NORM=1
          CHECKPOINT_FILE="tq_vm_codegen_qknorm.ckpt"
          TRAINING_LOG="tq_vm_codegen_qknorm.log"
)
target_link_libraries(tq_vm_codegen_tq_qknorm PRIVATE microgpt_vm_lib)

# 5) Standalone KV-cache compression demo (no corpus needed)
add_executable(tq_kv_cache_demo demos/turbo_quant/simple/main.c)
target_link_libraries(tq_kv_cache_demo PRIVATE microgpt_lib)
target_include_directories(tq_kv_cache_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(tq_kv_cache_demo PRIVATE m)
endif()

# ==================================================================
#  RotorQuant Integration Evaluation Targets
# ==================================================================

# 1) Shakespeare Word-Level + Infinite MSA (Baseline FP32)
add_demo(
  NAME    rq_shakespeare_base
  SOURCE  demos/rotor_quant/shakespeare/main.c
  THREADS
  COPY    demos/turbo_quant/shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=0 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
)

# 2) Shakespeare Word-Level + Infinite MSA + RotorQuant (Compressed)
add_demo(
  NAME    rq_shakespeare_rq
  SOURCE  demos/rotor_quant/shakespeare/main.c
  THREADS
  COPY    demos/turbo_quant/shakespeare/w_shakespeare.txt
  DEFINES N_EMBD=48 N_HEAD=4 N_LAYER=1 BLOCK_SIZE=64 MLP_DIM=192
          NUM_STEPS=0 LEARNING_RATE=0.001 BATCH_SIZE=16
          MAX_VOCAB=5000 MAX_DOCS=200000 MAX_DOC_LEN=512
          ENABLE_ROTORQUANT=1
)

# 3) VM Codegen + Infinite MSA (Baseline FP32)
add_demo(
  NAME    rq_vm_codegen_base
  SOURCE  demos/rotor_quant/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/turbo_quant/vm_codegen/c_vm_functions_combined.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=0 LEARNING_RATE=0.0005 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
)
target_link_libraries(rq_vm_codegen_base PRIVATE microgpt_vm_lib)

# 4) VM Codegen + Infinite MSA + RotorQuant (Compressed)
add_demo(
  NAME    rq_vm_codegen_rq
  SOURCE  demos/rotor_quant/vm_codegen/main.c
  THREADS METAL BLAS
  COPY    demos/turbo_quant/vm_codegen/c_vm_functions_combined.txt
  DEFINES N_EMBD=96 N_HEAD=4 N_LAYER=2 BLOCK_SIZE=256 MLP_DIM=384
          NUM_STEPS=0 LEARNING_RATE=0.0005 BATCH_SIZE=16
          MAX_VOCAB=1200 MAX_DOCS=5000 MAX_DOC_LEN=512
          ENABLE_ROTORQUANT=1
)
target_link_libraries(rq_vm_codegen_rq PRIVATE microgpt_vm_lib)

# 5) Standalone KV-cache compression demo (no corpus needed)
add_executable(rq_kv_cache_demo demos/rotor_quant/simple/main.c)
target_link_libraries(rq_kv_cache_demo PRIVATE microgpt_lib)
target_include_directories(rq_kv_cache_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(rq_kv_cache_demo PRIVATE m)
endif()
