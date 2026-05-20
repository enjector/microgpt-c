# Examples for libpipeline_ir.  Each example is a self-contained
# executable that links ONLY pipeline_ir (no microgpt_lib, no VM, no
# transformer engine).  These also serve as smoke tests for the public
# ABI from an external-consumer perspective.

add_executable(pipeline_ir_example_custom_generator
  custom_generator/main.c
)
target_link_libraries(pipeline_ir_example_custom_generator
  PRIVATE pipeline_ir
)
