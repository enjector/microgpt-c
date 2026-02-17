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

# ---- Core library (object library for shared compilation) ----
add_library(microgpt_lib OBJECT src/microgpt.c)
target_include_directories(microgpt_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- SIMD auto-vectorisation (ON by default) ----
# Disable with: cmake -DMICROGPT_SIMD=OFF ..
option(MICROGPT_SIMD "Enable SIMD/vectorized codegen for faster training/inference" ON)
if(MICROGPT_SIMD)
  if(MSVC)
    target_compile_options(microgpt_lib PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
  else()
    target_compile_options(microgpt_lib PRIVATE $<$<CONFIG:Release>:-march=native>)
  endif()
  target_compile_definitions(microgpt_lib PRIVATE MICROGPT_SIMD=1)
endif()

# ---- Optional: INT8 weight quantisation ----
# cmake -DQUANTIZATION_INT8=ON ..
option(QUANTIZATION_INT8 "Store weights as 8-bit integers with per-matrix scales" OFF)
if(QUANTIZATION_INT8)
  target_compile_definitions(microgpt_lib PRIVATE QUANTIZATION_INT8=1 QUANTISATION_INT8=1)
endif()

# ---- Example: Character-level name generation ----
add_executable(names_demo examples/names/main.c $<TARGET_OBJECTS:microgpt_lib>)
target_include_directories(names_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(names_demo PRIVATE m)
endif()
add_custom_command(TARGET names_demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/names/names.txt"
    "$<TARGET_FILE_DIR:names_demo>/names.txt"
  COMMENT "Copying names.txt to build output"
)

# ---- Example: Word-level Shakespeare generation ----
# A separate OBJECT library is needed because the Shakespeare demo overrides
# architecture defines (N_LAYER, N_HEAD, etc.) that change struct layout.
# Both the library and the demo must be compiled with the same defines.
add_library(microgpt_lib_shakespeare OBJECT src/microgpt.c)
target_include_directories(microgpt_lib_shakespeare PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(microgpt_lib_shakespeare PRIVATE
  N_EMBD=32
  N_HEAD=2
  N_LAYER=1
  BLOCK_SIZE=16
  BATCH_SIZE=16
  NUM_STEPS=10000
  LEARNING_RATE=0.01
  MAX_VOCAB=7000
  MAX_DOCS=200000
  MAX_DOC_LEN=128
)
if(MICROGPT_SIMD)
  if(MSVC)
    target_compile_options(microgpt_lib_shakespeare PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
  else()
    target_compile_options(microgpt_lib_shakespeare PRIVATE $<$<CONFIG:Release>:-march=native>)
  endif()
endif()
if(QUANTIZATION_INT8)
  target_compile_definitions(microgpt_lib_shakespeare PRIVATE QUANTIZATION_INT8=1 QUANTISATION_INT8=1)
endif()

add_executable(shakespeare_demo examples/shakespeare/main.c $<TARGET_OBJECTS:microgpt_lib_shakespeare>)
target_include_directories(shakespeare_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(shakespeare_demo PRIVATE
  N_EMBD=32
  N_HEAD=2
  N_LAYER=1
  BLOCK_SIZE=16
  BATCH_SIZE=16
  NUM_STEPS=10000
  LEARNING_RATE=0.01
  MAX_VOCAB=7000
  MAX_DOCS=200000
  MAX_DOC_LEN=128
)
if(NOT MSVC)
  target_link_libraries(shakespeare_demo PRIVATE m)
endif()
add_custom_command(TARGET shakespeare_demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/shakespeare/shakespeare.txt"
    "$<TARGET_FILE_DIR:shakespeare_demo>/shakespeare.txt"
  COMMENT "Copying shakespeare.txt to build output"
)

# Run from project root so data files are found, or pass path via argv in future.

# ---- Tests ----
enable_testing()
add_executable(test_microgpt tests/test_microgpt.c $<TARGET_OBJECTS:microgpt_lib>)
target_include_directories(test_microgpt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(test_microgpt PRIVATE m)
endif()
add_test(NAME microgpt_tests COMMAND test_microgpt WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# ---- Benchmarks ----
add_executable(bench_microgpt tests/bench_microgpt.c $<TARGET_OBJECTS:microgpt_lib>)
target_include_directories(bench_microgpt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
  target_link_libraries(bench_microgpt PRIVATE m)
endif()
