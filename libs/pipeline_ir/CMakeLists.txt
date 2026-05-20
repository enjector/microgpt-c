# ==================================================================
#  libpipeline_ir — Standalone Pipeline IR + verifier library
# ==================================================================
#
# Pure C99, libc-only.  Builds as a STATIC archive `pipeline_ir.a`
# (linker name: libpipeline_ir.a on POSIX, pipeline_ir.lib on MSVC).
#
# Extracted from MicroGPT-C per Experiment E02 — see
#   experiments/E02-pipeline-ir-library.md
# for the pre-registered targets and rationale.
#
# CONSUMERS
# ---------
# In-tree (MicroGPT-C parent repo):
#     # automatically picked up by the top-level CMakeLists.txt
#     target_link_libraries(my_target PRIVATE pipeline_ir)
#
# External (out-of-tree) project using FetchContent:
#     include(FetchContent)
#     FetchContent_Declare(
#       pipeline_ir
#       GIT_REPOSITORY https://github.com/enjector/microgpt-c.git
#       GIT_TAG        main
#       SOURCE_SUBDIR  libs/pipeline_ir
#     )
#     FetchContent_MakeAvailable(pipeline_ir)
#     target_link_libraries(my_target PRIVATE pipeline_ir)
#
# External using add_subdirectory (vendored copy):
#     add_subdirectory(third_party/pipeline_ir)
#     target_link_libraries(my_target PRIVATE pipeline_ir)
#
# The library exposes one public include directory
# (`libs/pipeline_ir/include/`).  Public header path:
#   #include <pipeline_ir/pipeline_ir.h>
#
# The internal header `src/pipeline_ir_internal.h` is NOT installed
# and not part of any public include directory — it is shared between
# `pipeline_ir.c` and the opt-in `pipeline_ir_vm.c` only.

cmake_minimum_required(VERSION 3.10)

# Allow standalone configuration (cmake -S libs/pipeline_ir -B build_pir)
# as well as add_subdirectory() from the parent.
if(NOT DEFINED PROJECT_NAME OR NOT PROJECT_NAME)
  project(pipeline_ir C)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
#  Core target — IR + verifier + parsers + repair + DOT renderer
# ------------------------------------------------------------------
add_library(pipeline_ir STATIC
  src/pipeline_ir.c
)
target_include_directories(pipeline_ir PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# Internal header is reachable from within src/ only.
target_include_directories(pipeline_ir PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(NOT MSVC)
  # C99 + warnings.  -O2 is inherited from the parent's
  # CMAKE_C_FLAGS_RELEASE; libpipeline_ir itself doesn't override it.
  target_compile_options(pipeline_ir PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Export an alias so callers in either the parent tree or external
# trees can `target_link_libraries(foo PRIVATE pipeline_ir::pipeline_ir)`
# without caring whether they consumed via add_subdirectory or
# FetchContent.
add_library(pipeline_ir::pipeline_ir ALIAS pipeline_ir)

# ------------------------------------------------------------------
#  Opt-in VM dispatcher
# ------------------------------------------------------------------
# pipeline_ir_vm.c implements pipeline_execute_vm() which depends on
# a vm_engine implementation supplied by the consumer (the MicroGPT-C
# parent provides one via src/microgpt_vm.c).  Building the file
# requires the consumer to also drop in a compatible "microgpt_vm.h"
# on the include path.  We expose the SOURCE PATH as a cache variable
# so consumers can add it to their target sources directly.
set(PIPELINE_IR_VM_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline_ir_vm.c"
    CACHE INTERNAL
    "Absolute path to opt-in pipeline_ir_vm.c; add to consumer target sources to enable pipeline_execute_vm()")

# Internal include directory (only `pipeline_ir_vm.c` needs it).  Also
# exposed so the MicroGPT-C parent test target can find
# pipeline_ir_internal.h.
set(PIPELINE_IR_INTERNAL_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    CACHE INTERNAL
    "Path to libs/pipeline_ir/src/ for the opt-in VM dispatcher TU")

# ------------------------------------------------------------------
#  Examples (opt-in)
# ------------------------------------------------------------------
# Each example is a self-contained executable that links ONLY
# pipeline_ir.  Useful as smoke tests for the public ABI from an
# external-consumer perspective and as starting templates for new
# integrations.  Default ON when libpipeline_ir is the top-level
# project; default OFF when consumed via add_subdirectory or
# FetchContent (so embedders don't get extra binaries by accident).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_pir_examples_default ON)
else()
  set(_pir_examples_default ON)  # also ON in-tree so CI smoke-tests them
endif()
option(PIPELINE_IR_BUILD_EXAMPLES
       "Build libpipeline_ir example binaries (custom_generator, ...)"
       ${_pir_examples_default})
if(PIPELINE_IR_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
