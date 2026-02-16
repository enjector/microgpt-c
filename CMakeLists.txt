cmake_minimum_required(VERSION 3.10)
project(microgpt C CXX)

set(CMAKE_C_STANDARD 99)

# Default to Release when no config is specified (single-config generators)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# High optimisation for Release
if(MSVC)
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /Ot" CACHE STRING "" FORCE)
else()
  set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
endif()

add_executable(microgpt main.c microgpt.c)
target_include_directories(microgpt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Copy training data next to the executable so it can be found at runtime
add_custom_command(TARGET microgpt POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/data/names.txt"
    "$<TARGET_FILE_DIR:microgpt>/names.txt"
  COMMENT "Copying names.txt to build output"
)

# Link math library on non-MSVC platforms
if(NOT MSVC)
  target_link_libraries(microgpt PRIVATE m)
endif()

# Optional: INT8 weight quantization (smaller memory, same training; good for demo / embedded)
# cmake -DQUANTIZATION_INT8=ON ..
option(QUANTIZATION_INT8 "Store weights as 8-bit integers with per-matrix scales" OFF)
if(QUANTIZATION_INT8)
  target_compile_definitions(microgpt PRIVATE QUANTIZATION_INT8=1)
  target_compile_definitions(microgpt PRIVATE QUANTISATION_INT8=1)  # British spelling alias
endif()

# Optional: SIMD-friendly codegen (auto-vectorization of dot products, norms, etc.)
# cmake -DMICROGPT_SIMD=ON ..
option(MICROGPT_SIMD "Enable SIMD/vectorized codegen (e.g. AVX2) for faster training/inference" OFF)
if(MICROGPT_SIMD)
  if(MSVC)
    target_compile_options(microgpt PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
  else()
    # -march=native uses best ISA on build host (SSE/AVX/AVX2); use -march=haswell etc. for portability
    target_compile_options(microgpt PRIVATE $<$<CONFIG:Release>:-march=native>)
  endif()
  target_compile_definitions(microgpt PRIVATE MICROGPT_SIMD=1)
endif()

# Run from project root so names.txt is found, or pass path via argv in future.
