/*
 * MicroGPT-C VM Engine — Unit Tests
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include "../src/microgpt_vm.h"
#include "test.h"

// --- Added dynamic file loader ---
#include <stdio.h>
#include <stdlib.h>

static char *load_file_content(const char *filepath) {
  FILE *f = fopen(filepath, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *string = malloc(fsize + 1);
  fread(string, 1, fsize, f);
  fclose(f);
  string[fsize] = 0;
  return string;
}
// ---------------------------------

// --- vm_array_ops_tests.h ---
/*
 * vm_array_ops_tests.h
 *
 * Copyright 2026 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include "test.h"

extern enx_test_case_t vm_array_ops_tests[];

// --- vm_array_ops_tests.c ---
/*
 * vm_array_ops_tests.c
 *
 * Copyright 2026 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enx_test(should_orchestrate_opaque_arrays_in_vm) {
  // Test disabled: verb.h dependency removed
}

enx_test_case_t vm_array_ops_tests[] = {
    enx_test_case(should_orchestrate_opaque_arrays_in_vm), enx_test_case_end()};

// --- vm_module_compiler_tests.h ---
/*
 * vm_module_compiler_tests.h
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include "test.h"

void _TEST_VM_MODULE_COMPILER_MODULE_COMPARE_ASSERT_FATAL(
    const char *target_name);

#define _TEST_VM_MODULE_COMPILER_CODE(name)                                    \
  enx_test(should_successfully_compile_##name) {                               \
    _TEST_VM_MODULE_COMPILER_MODULE_COMPARE_ASSERT_FATAL(#name);               \
  }

#define TEST_VM_MODULE_COMPILER_CODE(name) _TEST_VM_MODULE_COMPILER_CODE(name)

#define _TEST_VM_MODULE_COMPILER_enx_test_case(name)                           \
  {"should_successfully_compile_" #name, should_successfully_compile_##name}

#define TEST_VM_MODULE_COMPILER_enx_test_case(name)                            \
  _TEST_VM_MODULE_COMPILER_enx_test_case(name)

extern enx_test_case_t vm_module_compiler_tests[];

// --- vm_module_compiler_tests.c ---
/*
 * vm_module_compiler_tests.c
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TEST_VM_MODULE_COMPILER_CODE(gen_body1)
TEST_VM_MODULE_COMPILER_CODE(gen_body2)
TEST_VM_MODULE_COMPILER_CODE(gen_body3)
TEST_VM_MODULE_COMPILER_CODE(gen_body4)
TEST_VM_MODULE_COMPILER_CODE(gen_body5)
TEST_VM_MODULE_COMPILER_CODE(gen_math1)
TEST_VM_MODULE_COMPILER_CODE(gen_math2)
TEST_VM_MODULE_COMPILER_CODE(gen_fn_empty)
TEST_VM_MODULE_COMPILER_CODE(gen_fn_arg_ret_calc1)
TEST_VM_MODULE_COMPILER_CODE(gen_fn_arg_ret_calc2)
TEST_VM_MODULE_COMPILER_CODE(gen_loop1)
TEST_VM_MODULE_COMPILER_CODE(gen_loop2)
TEST_VM_MODULE_COMPILER_CODE(gen_loop3)
TEST_VM_MODULE_COMPILER_CODE(gen_loop4)
TEST_VM_MODULE_COMPILER_CODE(gen_loop5)
TEST_VM_MODULE_COMPILER_CODE(gen_loop6)
TEST_VM_MODULE_COMPILER_CODE(gen_condition1)
TEST_VM_MODULE_COMPILER_CODE(gen_condition2)
TEST_VM_MODULE_COMPILER_CODE(gen_condition3)
TEST_VM_MODULE_COMPILER_CODE(gen_condition4)
TEST_VM_MODULE_COMPILER_CODE(gen_latex_math_where)
TEST_VM_MODULE_COMPILER_CODE(gen_latex_math1)

enx_test_case_t vm_module_compiler_tests[] = {
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_body1),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_body2),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_body3),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_body4),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_body5),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_math1),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_math2),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_fn_empty),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_fn_arg_ret_calc1),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_fn_arg_ret_calc2),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_loop1),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_loop2),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_loop3),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_loop4),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_loop5),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_loop6),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_condition1),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_condition2),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_condition3),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_condition4),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_latex_math_where),
    TEST_VM_MODULE_COMPILER_enx_test_case(gen_latex_math1),
    enx_test_case_end()};

void _TEST_VM_MODULE_COMPILER_MODULE_COMPARE_ASSERT_FATAL(
    const char *target_name) {
  char source_path[512];
  snprintf(source_path, sizeof(source_path), "resources/vm/compiler/%s.ts",
           target_name);
  char expected_path[512];
  snprintf(expected_path, sizeof(expected_path), "resources/vm/compiler/%s.il",
           target_name);

  char *source = load_file_content(source_path);
  enx_assert_ptr_not_null(source);

  char *expected = load_file_content(expected_path);

  vm_module *module = NULL;
  result r = vm_module_compile(NULL, source, &module);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);
  char *actual = vm_module_to_string(module);

  if (!string_equals(expected, actual)) {
    char filename_il[512];
#ifdef FIX_TEST_DATA
    snprintf(filename_il, sizeof(filename_il), "resources/vm/compiler/%s.il",
             target_name);
#else
    snprintf(filename_il, sizeof(filename_il),
             "resources/vm/compiler/%s_il.txt", target_name);
#endif
    printf("\n****************************************************************"
           "****************\n* "
           "Expected:\n*******************************************************"
           "*************************\n%s|\n**********************************"
           "**********************************************\n* "
           "Actual:\n*********************************************************"
           "***********************\n%s|\n",
           expected, actual);

#ifdef FIX_TEST_DATA
    file_write_text(filename_il, actual, string_length(actual));
#endif
#ifndef FIX_TEST_DATA
    enx_assert_fail();
#endif
  }

  string_free(actual);
  free(source);
  if (expected)
    free(expected);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

// --- vm_module_declare_function_tests.h ---
/*
 * vm_module_declare_function_tests.h
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include "test.h"

extern enx_test_case_t vm_module_declare_function_tests[];

// --- vm_module_declare_function_tests.c ---
/*
 * vm_module_declare_function_tests.c
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

// #define _DEBUG_IL

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
test_vm_call_ext_method_callback(struct vm_module_runtime_t *runtime,
                                 vm_function *function) {
  if (string_equals(function->name, "is_temperature_hot") ||
      string_equals(function->name, "is_pressure_high")) {
    vm_variable *var_param = NULL;
    result r = vm_module_runtime_stack_pop(runtime, &var_param);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
    enx_assert_ptr_not_null(var_param);
    enx_assert_equal_int(ptcNUMBER, var_param->type_class);

    vm_module_runtime_stack_push_boolean(runtime, true);

    vm_variable_dispose(var_param);

    //    } else
    //
    // if (string_equals(function->name, "eval")) {
    //     vm_variable* var_param_a = NULL;
    //     vm_variable* var_param_b = NULL;
    //     result r = vm_module_runtime_stack_pop(runtime, &var_param_a);
    //     TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
    //     enx_assert_ptr_not_null(var_param_a);
    //     enx_assert_equal_int(ptcNUMBER, var_param_a->type_class);

    //     r = vm_module_runtime_stack_pop(runtime, &var_param_b);
    //     TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
    //     enx_assert_ptr_not_null(var_param_b);
    //     enx_assert_equal_int(ptcNUMBER, var_param_b->type_class);

    //     vm_module_runtime_stack_push_number(runtime,
    //     var_param_a->value.number + var_param_b->value.number);

    //     vm_variable_dispose(var_param_a);
    //     vm_variable_dispose(var_param_b);
  } else if (string_equals(function->name, "simple_add")) {
    vm_variable *var_param_a = NULL;
    vm_variable *var_param_b = NULL;
    result r = vm_module_runtime_stack_pop(runtime, &var_param_a);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
    enx_assert_ptr_not_null(var_param_a);
    enx_assert_equal_int(ptcNUMBER, var_param_a->type_class);

    r = vm_module_runtime_stack_pop(runtime, &var_param_b);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
    enx_assert_ptr_not_null(var_param_b);
    enx_assert_equal_int(ptcNUMBER, var_param_b->type_class);

    vm_module_runtime_stack_push_number(runtime, var_param_a->value.number +
                                                     var_param_b->value.number);

    vm_variable_dispose(var_param_a);
    vm_variable_dispose(var_param_b);
  } else {
    printf("Unknown function: %s\n", function->name);
    assert(false);
  }
}

enx_test(should_successfully_use_declare_function_condition1) {
  // Compile function
  vm_module *module = NULL;
  char *source = load_file_content(
      "resources/vm/declare_function/declare_function_condition1.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  enx_assert_equal_size(0, sequence_count(module->errors));

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  vm_module_runtime_set_call_ext_method_callback(
      runtime, test_vm_call_ext_method_callback);

  vm_function *function = vm_module_fetch_function(module, "eval");
  enx_assert_ptr_not_null(function);

  double temperature = 32.2;
  double pressure = 233;

  vm_module_runtime_stack_push_number(runtime, temperature);
  vm_module_runtime_stack_push_number(runtime, pressure);

  r = vm_module_runtime_run(runtime, function);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

  //    vm_compile_result r = vm_function_compile(function_condition1_ts,
  //    expected_return_type_class);
  // enx_assert_equal_int(expected_return_type_class,
  // r.function->return_type_class);

  // Check results
  vm_variable *return_var = NULL;
  r = vm_module_runtime_stack_pop(runtime, &return_var);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

  enx_assert_equal_int(ptcBOOLEAN, return_var->type_class);
  enx_assert_equal_bool(true, return_var->value.boolean);

  vm_variable_dispose(return_var);
  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test(should_successfully_use_declare_function_calculation1) {
  // Compile function
  vm_module *module = NULL;
  char *source = load_file_content(
      "resources/vm/declare_function/declare_function_calculation1.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  enx_assert_equal_size(0, sequence_count(module->errors));

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  vm_module_runtime_set_call_ext_method_callback(
      runtime, test_vm_call_ext_method_callback);

  vm_module_runtime_clear(runtime);

  vm_function *function = vm_module_fetch_function(module, "eval");
  enx_assert_ptr_not_null(function);

  r = vm_module_runtime_run(runtime, function);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

  // Check results
  vm_variable *return_var = NULL;
  result rs = vm_module_runtime_stack_pop(runtime, &return_var);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, rs);
  enx_assert_ptr_not_null(return_var);

  enx_assert_equal_int(ptcNUMBER, return_var->type_class);
  enx_assert_equal_double(300, return_var->value.number);

  vm_variable_dispose(return_var);
  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test_case_t vm_module_declare_function_tests[] = {
    enx_test_case(should_successfully_use_declare_function_condition1),
    enx_test_case(should_successfully_use_declare_function_calculation1),
    enx_test_case_end()};

// --- vm_module_function_call_tests.h ---
/*
 * vm_module_function_call_tests.h
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include "test.h"

void _TEST_VM_FUNCTION_MODULE_COMPARE_ASSERT_FATAL(const char *name);

#define _TEST_VM_FUNCTION_CODE(name)                                           \
  enx_test(should_successfully_execute_##name) {                               \
    _TEST_VM_FUNCTION_MODULE_COMPARE_ASSERT_FATAL(#name);                      \
  }

#define TEST_VM_FUNCTION_CODE(name) _TEST_VM_FUNCTION_CODE(name)

#define _TEST_VM_FUNCTION_enx_test_case(name)                                  \
  {"should_successfully_eval_" #name, should_successfully_execute_##name}

#define TEST_VM_FUNCTION_enx_test_case(name)                                   \
  _TEST_VM_FUNCTION_enx_test_case(name)

extern enx_test_case_t vm_module_function_call_tests[];

// --- vm_module_function_call_tests.c ---
/*
 * vm_module_function_call_tests.c
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

// #define _DEBUG_IL

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enx_test(should_successfully_function_condition1) {
  // Compile function
  vm_module *module = NULL;
  char *source =
      load_file_content("resources/vm/function/function_condition1.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);
  enx_assert_equal_size(0, sequence_count(module->errors));

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  vm_function *function = vm_module_fetch_function(module, "eval");
  enx_assert_ptr_not_null(function);

  r = vm_module_runtime_run(runtime, function);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

  //    vm_compile_result r = vm_function_compile(function_condition1_ts,
  //    expected_return_type_class);
  // enx_assert_equal_int(expected_return_type_class,
  // r.function->return_type_class);

  // Check results
  vm_variable *return_var = NULL;
  r = vm_module_runtime_stack_pop(runtime, &return_var);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

  enx_assert_equal_int(ptcBOOLEAN, return_var->type_class);
  enx_assert_equal_bool(false, return_var->value.boolean);

  vm_variable_dispose(return_var);
  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test(should_successfully_function_condition2) {
  // Compile function
  vm_module *module = NULL;
  char *source =
      load_file_content("resources/vm/function/function_condition2.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  // Tests
  const double inputs[] = {10, 50, 51, 60, 99, 100, 150};
  const bool expected_outputs[] = {false, false, true, true,
                                   true,  false, false};
  const size_t inputs_count = sizeof(inputs) / sizeof(double);
  const size_t expected_outputs_count = sizeof(expected_outputs) / sizeof(bool);
  enx_assert_equal_size(inputs_count, expected_outputs_count);

  for (size_t i = 0; i < inputs_count; i++) {
    vm_module_runtime_clear(runtime);

    const double input = inputs[i];
    const bool expected_output = expected_outputs[i];

    vm_module_runtime_stack_push_number(runtime, input);

    vm_function *function = vm_module_fetch_function(module, "eval");
    enx_assert_ptr_not_null(function);

    r = vm_module_runtime_run(runtime, function);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

    // Check results
    vm_variable *return_var = NULL;
    result rs = vm_module_runtime_stack_pop(runtime, &return_var);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, rs);
    enx_assert_ptr_not_null(return_var);

    enx_assert_equal_int(ptcBOOLEAN, return_var->type_class);
    enx_assert_equal_bool(expected_output, return_var->value.boolean);

    vm_variable_dispose(return_var);
  }

  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test(should_successfully_function_condition3) {
  // Compile function
  vm_module *module = NULL;
  char *source =
      load_file_content("resources/vm/function/function_condition3.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);
  enx_assert_equal_size(0, sequence_count(module->errors));

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  // Tests
  const char *inputs_name[] = {"Fred", "Anissa", "Sarah", "Bob", "Nicole"};
  const double inputs_age[] = {10, 18, 60, -1, 20};
  const bool expected_outputs[] = {true, true, false, true, false};
  const size_t inputs_name_count = sizeof(inputs_name) / sizeof(char *);
  const size_t inputs_age_count = sizeof(inputs_age) / sizeof(double);
  const size_t expected_outputs_count = sizeof(expected_outputs) / sizeof(bool);
  enx_assert_equal_size(inputs_name_count, expected_outputs_count);
  enx_assert_equal_size(inputs_age_count, expected_outputs_count);

  for (size_t i = 0; i < inputs_name_count; i++) {
    vm_module_runtime_clear(runtime);

    const char *input_name = inputs_name[i];
    const double input_age = inputs_age[i];
    const bool expected_output = expected_outputs[i];

    vm_module_runtime_stack_push_number(runtime, input_age);
    vm_module_runtime_stack_push_string(runtime, input_name);

    vm_function *function = vm_module_fetch_function(module, "eval");
    enx_assert_ptr_not_null(function);

    r = vm_module_runtime_run(runtime, function);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

    // Check results
    vm_variable *return_var = NULL;
    result rs = vm_module_runtime_stack_pop(runtime, &return_var);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, rs);
    enx_assert_ptr_not_null(return_var);

    enx_assert_equal_int(ptcBOOLEAN, return_var->type_class);
    enx_assert_equal_bool(expected_output, return_var->value.boolean);

    vm_variable_dispose(return_var);
  }

  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test(should_successfully_function_condition4) {
  // Compile function
  vm_module *module = NULL;
  char *source =
      load_file_content("resources/vm/function/function_condition4.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);
  enx_assert_equal_size(0, sequence_count(module->errors));

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  // Tests
  const double inputs[] = {1, 2, 3};
  const bool expected_outputs[] = {false, true, false};
  const size_t inputs_count = sizeof(inputs) / sizeof(double);
  const size_t expected_outputs_count = sizeof(expected_outputs) / sizeof(bool);
  enx_assert_equal_size(inputs_count, expected_outputs_count);

  for (size_t i = 0; i < inputs_count; i++) {
    vm_module_runtime_clear(runtime);

    const double input = inputs[i];
    const bool expected_output = expected_outputs[i];

    vm_module_runtime_stack_push_number(runtime, input);

    vm_function *function = vm_module_fetch_function(module, "eval");
    enx_assert_ptr_not_null(function);

    r = vm_module_runtime_run(runtime, function);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

    // Check results
    vm_variable *return_var = NULL;
    result rs = vm_module_runtime_stack_pop(runtime, &return_var);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, rs);
    enx_assert_ptr_not_null(return_var);

    enx_assert_equal_int(ptcBOOLEAN, return_var->type_class);
    enx_assert_equal_bool(expected_output, return_var->value.boolean);

    vm_variable_dispose(return_var);
  }

  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test(should_successfully_function_calculation1) {
  // Compile function
  vm_module *module = NULL;
  char *source =
      load_file_content("resources/vm/function/function_calculation1.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);
  enx_assert_equal_size(0, sequence_count(module->errors));

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  vm_module_runtime_clear(runtime);

  vm_function *function = vm_module_fetch_function(module, "eval");
  enx_assert_ptr_not_null(function);

  r = vm_module_runtime_run(runtime, function);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

  // Check results
  vm_variable *return_var = NULL;
  result rs = vm_module_runtime_stack_pop(runtime, &return_var);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, rs);
  enx_assert_ptr_not_null(return_var);

  enx_assert_equal_int(ptcNUMBER, return_var->type_class);
  enx_assert_equal_bool(102, return_var->value.number);

  vm_variable_dispose(return_var);
  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test(should_successfully_function_calculation2) {
  // Compile function
  vm_module *module = NULL;
  char *source =
      load_file_content("resources/vm/function/function_calculation2.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);
  enx_assert_equal_size(0, sequence_count(module->errors));

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  // Tests
  const double inputs_loan_amount[] = {1000, 2000, 4000, 8000, 16000};
  const double inputs_interest_rate[] = {1.2, 2.4, 3.6, 4.8, 6.0};
  const double inputs_duration_months[] = {12, 24, 36, 48, 60};
  const size_t inputs_loan_amount_count =
      sizeof(inputs_loan_amount) / sizeof(double);
  const size_t inputs_interest_rate_count =
      sizeof(inputs_interest_rate) / sizeof(double);
  const size_t inputs_duration_months_count =
      sizeof(inputs_duration_months) / sizeof(double);
  enx_assert_equal_size(inputs_interest_rate_count, inputs_interest_rate_count);
  enx_assert_equal_size(inputs_duration_months_count,
                        inputs_interest_rate_count);

  for (size_t i = 0; i < inputs_loan_amount_count; i++) {
    vm_module_runtime_clear(runtime);

    const double input_loan_amount = inputs_loan_amount[i];
    const double input_interest_rate = inputs_interest_rate[i];
    const double input_duration_months = inputs_duration_months[i];

    vm_module_runtime_stack_push_number(runtime, input_duration_months);
    vm_module_runtime_stack_push_number(runtime, input_interest_rate);
    vm_module_runtime_stack_push_number(runtime, input_loan_amount);

    vm_function *function = vm_module_fetch_function(module, "eval");
    enx_assert_ptr_not_null(function);

    r = vm_module_runtime_run(runtime, function);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

    // Check results
    vm_variable *return_var = NULL;
    result rs = vm_module_runtime_stack_pop(runtime, &return_var);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, rs);
    enx_assert_ptr_not_null(return_var);

    enx_assert_equal_int(ptcNUMBER, return_var->type_class);

    double expected_output =
        (input_loan_amount * input_interest_rate * (1 + input_interest_rate) *
         input_duration_months) /
        ((1 + input_interest_rate) * input_duration_months - 1);

    enx_assert_equal_double(expected_output, return_var->value.number);

    vm_variable_dispose(return_var);
  }

  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test(should_successfully_function_calculation3) {
  // Compile function
  vm_module *module = NULL;
  char *source =
      load_file_content("resources/vm/function/function_calculation3.ts");
  enx_assert_ptr_not_null(source);
  result r = vm_module_compile(NULL, source, &module);
  free(source);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);
  enx_assert_equal_size(0, sequence_count(module->errors));

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  // Run
  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  // Tests
  const char *inputs_name[] = {"Fred", "Anissa", "Sarah", "Bob", "Nicole"};
  const double inputs_stars[] = {10, 18, 60, -1, 20};
  const char *expected_outputs[] = {
      "Hello Fred you have 10 stars", "Hello Anissa you have 18 stars",
      "Hello Sarah you have 60 stars", "Hello Bob you have -1 stars",
      "Hello Nicole you have 20 stars"};
  const size_t inputs_name_count = sizeof(inputs_name) / sizeof(char *);
  const size_t inputs_stars_count = sizeof(inputs_stars) / sizeof(double);
  const size_t expected_outputs_count =
      sizeof(expected_outputs) / sizeof(char *);
  enx_assert_equal_size(inputs_name_count, expected_outputs_count);
  enx_assert_equal_size(inputs_stars_count, expected_outputs_count);

  for (size_t i = 0; i < inputs_name_count; i++) {
    const char *input_name = inputs_name[i];
    const double input_stars = inputs_stars[i];
    const char *expected_output = expected_outputs[i];

    vm_module_runtime_stack_push_number(runtime, input_stars);
    vm_module_runtime_stack_push_string(runtime, input_name);

    vm_function *function = vm_module_fetch_function(module, "eval");
    enx_assert_ptr_not_null(function);

    r = vm_module_runtime_run(runtime, function);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

    // Check results
    vm_variable *return_var = NULL;
    result rs = vm_module_runtime_stack_pop(runtime, &return_var);
    TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, rs);
    enx_assert_ptr_not_null(return_var);

    enx_assert_equal_int(ptcSTRING, return_var->type_class);
    enx_assert_equal_string(expected_output, return_var->value.string);

    vm_variable_dispose(return_var);

    vm_module_runtime_clear(runtime);
  }

  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

enx_test_case_t vm_module_function_call_tests[] = {
    enx_test_case(should_successfully_function_condition1),
    enx_test_case(should_successfully_function_condition2),
    enx_test_case(should_successfully_function_condition3),
    enx_test_case(should_successfully_function_condition4),
    enx_test_case(should_successfully_function_calculation1),
    enx_test_case(should_successfully_function_calculation2),
    enx_test_case(should_successfully_function_calculation3),
    enx_test_case_end()};

// --- vm_module_runtime_tests.h ---
/*
 * vm_module_runtime_tests.h
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include "test.h"

void _TEST_VM_MODULE_RUNTIME_MODULE_COMPARE_ASSERT_FATAL(const char *name);

#define _TEST_VM_MODULE_RUNTIME_CODE(name)                                     \
  enx_test(should_successfully_execute_##name) {                               \
    _TEST_VM_MODULE_RUNTIME_MODULE_COMPARE_ASSERT_FATAL(#name);                \
  }

#define TEST_VM_MODULE_RUNTIME_CODE(name) _TEST_VM_MODULE_RUNTIME_CODE(name)

#define _TEST_VM_MODULE_RUNTIME_enx_test_case(name)                            \
  {"should_successfully_convert_" #name, should_successfully_execute_##name}

#define TEST_VM_MODULE_RUNTIME_enx_test_case(name)                             \
  _TEST_VM_MODULE_RUNTIME_enx_test_case(name)

extern enx_test_case_t vm_module_runtime_tests[];

// --- vm_module_runtime_tests.c ---
/*
 * vm_tests_runtime.c
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

// #define _DEBUG_IL
// #define FIX_TEST_DATA // TODO: fix fo latex output required

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TEST_VM_MODULE_RUNTIME_CODE(runtime_math1)
TEST_VM_MODULE_RUNTIME_CODE(runtime_math2)
TEST_VM_MODULE_RUNTIME_CODE(runtime_functions1)
TEST_VM_MODULE_RUNTIME_CODE(runtime_conditions1)
TEST_VM_MODULE_RUNTIME_CODE(runtime_conditions2)
TEST_VM_MODULE_RUNTIME_CODE(runtime_conditions3)
TEST_VM_MODULE_RUNTIME_CODE(runtime_conditions4)
TEST_VM_MODULE_RUNTIME_CODE(runtime_loop1)
TEST_VM_MODULE_RUNTIME_CODE(runtime_loop2)
TEST_VM_MODULE_RUNTIME_CODE(runtime_loop3)
TEST_VM_MODULE_RUNTIME_CODE(runtime_strings1)
TEST_VM_MODULE_RUNTIME_CODE(runtime_latex_math1)

enx_test_case_t vm_module_runtime_tests[] = {
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_math1),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_math2),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_functions1),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_conditions1),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_conditions2),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_conditions3),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_conditions4),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_loop1),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_loop2),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_loop3),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_strings1),
    TEST_VM_MODULE_RUNTIME_enx_test_case(runtime_latex_math1),
    enx_test_case_end()};

void _TEST_VM_MODULE_RUNTIME_MODULE_COMPARE_ASSERT_FATAL(const char *name) {
  char source_path[512];
  snprintf(source_path, sizeof(source_path), "resources/vm/runtime/%s.ts",
           name);
  char expected_path[512];
  snprintf(expected_path, sizeof(expected_path),
           "resources/vm/runtime/%s.output", name);

  char *source = load_file_content(source_path);
  enx_assert_ptr_not_null(source);

  char *expected = load_file_content(expected_path);

  vm_module *module = NULL;
  result r = vm_module_compile(NULL, source, &module);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
  enx_assert_ptr_not_null(module);

#ifdef _DEBUG_IL
  char *il = vm_module_to_string(module);
  puts(il);
  xmemory_free(il);
#endif

  sequence_foreach_of(module->errors, vm_module_error *, error) {
    printf("ERROR: %zu:%s\n", error->source_line_number, error->message);
  }
  enx_assert_equal_size(0, sequence_count(module->errors));

  vm_module_runtime *runtime = vm_module_runtime_create(module);
  enx_assert_ptr_not_null(runtime);

  vm_function *function = vm_module_fetch_function(module, "main");
  enx_assert_ptr_not_null(function);

  r = vm_module_runtime_run(runtime, function);
  TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);

  /// IL
  char *module_il = vm_module_to_string(module);
  char filename_il[512];
  snprintf(filename_il, sizeof(filename_il),
           "../enjector-core/test/resources/vm/runtime/%s_il.txt", name);
#ifdef FIX_TEST_DATA
  file_write_text(filename_il, module_il, string_length(module_il));
#endif
  string_free(module_il);
  //////

  char *actual = vm_function_variables_to_string(function);

  if (!string_equals(expected, actual)) {
#ifdef FIX_TEST_DATA
    char filename_output[512];
    snprintf(filename_output, sizeof(filename_output),
             "../enjector-core/test/resources/vm/runtime/%s.txt", output_name);
    file_write_text(filename_output, actual, string_length(actual));
#endif

    vm_variable *return_var = NULL;
    r = vm_module_runtime_stack_pop(runtime, &return_var);
    char *var_return_str = "(void)";

    if (r != RESULT_CORE_VM_EMPTY_STACK) {
      TEST_ASSERT_EQUAL_RESULT_FATAL(RESULT_OK, r);
      enx_assert_ptr_not_null(return_var);
      var_return_str = vm_variable_to_string(return_var);
    }

    printf("*******************************************************************"
           "*************\n* "
           "Expected:\n********************************************************"
           "************************\n%s|\n************************************"
           "********************************************\n* "
           "Actual:\n**********************************************************"
           "**********************\n%s\nreturn=%s|\n",
           expected, actual, var_return_str);
    enx_assert_fail();
  }

  string_free(actual);
  free((void *)source);
  if (expected)
    free((void *)expected);

  vm_module_runtime_dispose(runtime);
  vm_module_dispose(module);

  xmemory_report_exit_on_leaks();
}

// --- vm_ts_wiring_tests.h ---
#ifndef ENJECTOR_VM_TS_WIRING_TESTS_H
#define ENJECTOR_VM_TS_WIRING_TESTS_H

#include "test.h"

extern enx_test_case_t vm_ts_wiring_tests[];

#endif // ENJECTOR_VM_TS_WIRING_TESTS_H

// --- vm_ts_wiring_tests.c ---

// Removed file.h
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enx_test(should_compile_entire_ts_wiring_corpus) {
  char *corpus_code = NULL;
  size_t length = 0;

  // Read the generated TS wiring corpus
  FILE *f = fopen("tests/resources/ts_wiring.txt", "rb");
  if (!f) {
    f = fopen("../tests/resources/ts_wiring.txt", "rb");
  }
  if (!f) {
    f = fopen("../../tests/resources/ts_wiring.txt", "rb");
  }
  if (!f) {
    printf("Failed to open ts_wiring.txt. Ensure it exists.\n");
    enx_assert_fail();
  }

  fseek(f, 0, SEEK_END);
  length = ftell(f);
  fseek(f, 0, SEEK_SET);

  corpus_code = malloc(length + 1);
  fread(corpus_code, 1, length, f);
  corpus_code[length] = '\0';
  fclose(f);

  // Compile the entire corpus as a single module
  vm_module *module = NULL;
  result r = vm_module_compile(NULL, corpus_code, &module);

  if (r != RESULT_OK) {
    printf("COMPILATION FAILED.\n");
  }

  enx_assert_equal_int(RESULT_OK, r);
  enx_assert_ptr_not_null(module);

  vm_module_dispose(module);
  free(corpus_code);
}

enx_test_case_t vm_ts_wiring_tests[] = {
    enx_test_case(should_compile_entire_ts_wiring_corpus), enx_test_case_end()};

// --- vm_verb_tests ---
/*
 * vm_verb_tests - Unit tests for the full verb system
 *
 * Copyright 2026 Enjector Software, Ltd. All Rights Reserved.
 */

/* Test verb callbacks */
static verb(test_greet_verb) {
  const char *name = verb_arg("name");
  if (name) {
    char buf[256];
    snprintf(buf, sizeof(buf), "Hello %s", name);
    return string_clone(buf);
  }
  return string_clone("Hello");
}

static verb(test_echo_verb) {
  const char *message = verb_arg("message");
  return message ? string_clone(message) : string_clone("");
}

static verb(test_add_verb) {
  const char *a = verb_arg("a");
  const char *b = verb_arg("b");
  char buf[64];
  snprintf(buf, sizeof(buf), "%d", atoi(a) + atoi(b));
  return string_clone(buf);
}

static verb(test_noparams_verb) { return string_clone("connected"); }

/* 1. Register and find */
enx_test(should_verb_register_and_find) {
  verb_context *ctx = verb_context_create();

  int r = verb_register(ctx, "greet", "name", test_greet_verb, NULL);
  enx_assert_equal_int(r, RESULT_OK);

  verb_definition *def = verb_find(ctx, "greet");
  enx_assert_ptr_not_null(def);
  enx_assert_equal_size(def->params_count, 1);
  enx_assert_equal_string("greet", def->name);

  verb_context_dispose(ctx);
}

/* 2. Duplicate registration */
enx_test(should_verb_register_duplicate) {
  verb_context *ctx = verb_context_create();

  int r = verb_register(ctx, "echo", "message", test_echo_verb, NULL);
  enx_assert_equal_int(r, RESULT_OK);

  r = verb_register(ctx, "echo", "message", test_echo_verb, NULL);
  enx_assert_equal_int(r, RESULT_CORE_VERB_ERROR_ALREADY_REGISTERED);

  verb_context_dispose(ctx);
}

/* 3. Execute simple verb */
enx_test(should_verb_exec_simple) {
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "echo", "message", test_echo_verb, NULL);

  char *response = NULL;
  int r = verb_exec(ctx, "echo hello", &response);
  enx_assert_equal_int(r, RESULT_OK);
  enx_assert_ptr_not_null(response);
  enx_assert_equal_string(response, "hello");
  free(response);

  verb_context_dispose(ctx);
}

/* 4. Execute with missing params → usage string */
enx_test(should_verb_exec_missing_params) {
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "greet", "name", test_greet_verb, NULL);

  char *response = NULL;
  int r = verb_exec(ctx, "greet", &response);
  enx_assert_equal_int(r, RESULT_CORE_VERB_ERROR_INCORRECT_USAGE);
  enx_assert_ptr_not_null(response);
  /* Response should contain "Usage: greet <name>" */
  enx_assert_true(strstr(response, "Usage:") != NULL);
  free(response);

  verb_context_dispose(ctx);
}

/* 5. Execute unregistered verb */
enx_test(should_verb_exec_not_found) {
  verb_context *ctx = verb_context_create();

  char *response = NULL;
  int r = verb_exec(ctx, "unknown_cmd", &response);
  enx_assert_equal_int(r, RESULT_CORE_VERB_ERROR_NO_MATCH);

  verb_context_dispose(ctx);
}

/* 6. Unregister verb */
enx_test(should_verb_unregister) {
  verb_context *ctx = verb_context_create();

  int r = verb_register(ctx, "echo", "message", test_echo_verb, NULL);
  enx_assert_equal_int(r, RESULT_OK);
  enx_assert_true(verb_exists(ctx, "echo"));

  r = verb_unregister(ctx, "echo");
  enx_assert_equal_int(r, RESULT_OK);
  enx_assert_false(verb_exists(ctx, "echo"));

  verb_context_dispose(ctx);
}

/* 7. Execute with quoted string params */
enx_test(should_verb_exec_quoted_params) {
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "greet", "name", test_greet_verb, NULL);

  char *response = NULL;
  int r = verb_exec(ctx, "greet 'Fred Smith'", &response);
  enx_assert_equal_int(r, RESULT_OK);
  enx_assert_ptr_not_null(response);
  enx_assert_equal_string(response, "Hello Fred Smith");
  free(response);

  verb_context_dispose(ctx);
}

/* 8. Enumerate registered verbs */
enx_test(should_verb_enum) {
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "echo", "message", test_echo_verb, NULL);
  verb_register(ctx, "greet", "name", test_greet_verb, NULL);
  verb_register(ctx, "connect", "", test_noparams_verb, NULL);

  cmap *m = verb_enum(ctx);
  enx_assert_ptr_not_null(m);
  enx_assert_true(m->count == 3);

  verb_context_dispose(ctx);
}

/* 9. No-params verb execution */
enx_test(should_verb_exec_noparams) {
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "connect", "", test_noparams_verb, NULL);

  char *response = NULL;
  int r = verb_exec(ctx, "connect", &response);
  enx_assert_equal_int(r, RESULT_OK);
  enx_assert_ptr_not_null(response);
  enx_assert_equal_string(response, "connected");
  free(response);

  verb_context_dispose(ctx);
}

/* 10. Two-param verb execution */
enx_test(should_verb_exec_two_params) {
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "add", "a b", test_add_verb, NULL);

  char *response = NULL;
  int r = verb_exec(ctx, "add 10 20", &response);
  enx_assert_equal_int(r, RESULT_OK);
  enx_assert_ptr_not_null(response);
  enx_assert_equal_string(response, "30");
  free(response);

  verb_context_dispose(ctx);
}

enx_test_case_t vm_verb_tests[] = {
    enx_test_case(should_verb_register_and_find),
    enx_test_case(should_verb_register_duplicate),
    enx_test_case(should_verb_exec_simple),
    enx_test_case(should_verb_exec_missing_params),
    enx_test_case(should_verb_exec_not_found),
    enx_test_case(should_verb_unregister),
    enx_test_case(should_verb_exec_quoted_params),
    enx_test_case(should_verb_enum),
    enx_test_case(should_verb_exec_noparams),
    enx_test_case(should_verb_exec_two_params),
    enx_test_case_end()};

// --- vm_verb_opaque_handle_tests ---
/*
 * Opaque array handle worked example
 * (ORGANELLE_GENERALISATION_VM.md §2.2)
 *
 * Demonstrates: create_signal → rolling_mean → signal_value_at → signal_dispose
 * using ptcOTHER (void*) handles managed by verb callbacks.
 */

#include <math.h>

/* Opaque signal struct — the verb callbacks own these allocations */
typedef struct {
  double *data;
  size_t length;
} signal_handle;

static signal_handle *_signal_create(size_t length) {
  signal_handle *s = (signal_handle *)malloc(sizeof(signal_handle));
  s->data = (double *)calloc(length, sizeof(double));
  s->length = length;
  return s;
}

static void _signal_free(signal_handle *s) {
  if (s) {
    free(s->data);
    free(s);
  }
}

/* Handle registry (simple array for test isolation) */
#define MAX_HANDLES 64
static signal_handle *_handle_registry[MAX_HANDLES];
static size_t _handle_count = 0;

static size_t _handle_store(signal_handle *s) {
  size_t id = _handle_count++;
  _handle_registry[id] = s;
  return id;
}

static signal_handle *_handle_get(size_t id) {
  return (id < _handle_count) ? _handle_registry[id] : NULL;
}

static void _handle_registry_reset(void) {
  for (size_t i = 0; i < _handle_count; i++) {
    _signal_free(_handle_registry[i]);
    _handle_registry[i] = NULL;
  }
  _handle_count = 0;
}

/* ── Signal verbs ── */

/* create_signal <length> → returns handle ID as string */
static verb(verb_create_signal) {
  const char *len_str = verb_arg("length");
  size_t length = (size_t)atoi(len_str);

  signal_handle *s = _signal_create(length);
  /* Fill with sample data: 1.0, 2.0, 3.0, ... */
  for (size_t i = 0; i < length; i++)
    s->data[i] = (double)(i + 1);

  size_t id = _handle_store(s);
  char buf[32];
  snprintf(buf, sizeof(buf), "%zu", id);
  return string_clone(buf);
}

/* rolling_mean <handle> <window> → returns new handle ID */
static verb(verb_rolling_mean) {
  const char *handle_str = verb_arg("handle");
  const char *window_str = verb_arg("window");

  size_t src_id = (size_t)atoi(handle_str);
  size_t window = (size_t)atoi(window_str);

  signal_handle *src = _handle_get(src_id);
  if (!src)
    return string_clone("ERROR: invalid handle");

  signal_handle *dst = _signal_create(src->length);
  for (size_t i = 0; i < src->length; i++) {
    double sum = 0;
    size_t count = 0;
    for (size_t j = (i >= window ? i - window + 1 : 0); j <= i; j++) {
      sum += src->data[j];
      count++;
    }
    dst->data[i] = sum / (double)count;
  }

  size_t dst_id = _handle_store(dst);
  char buf[32];
  snprintf(buf, sizeof(buf), "%zu", dst_id);
  return string_clone(buf);
}

/* signal_value_at <handle> <index> → returns value as string */
static verb(verb_signal_value_at) {
  const char *handle_str = verb_arg("handle");
  const char *index_str = verb_arg("index");

  size_t id = (size_t)atoi(handle_str);
  size_t idx = (size_t)atoi(index_str);

  signal_handle *s = _handle_get(id);
  if (!s || idx >= s->length)
    return string_clone("ERROR: out of bounds");

  char buf[64];
  snprintf(buf, sizeof(buf), "%.6f", s->data[idx]);
  return string_clone(buf);
}

/* signal_length <handle> → returns length as string */
static verb(verb_signal_length) {
  const char *handle_str = verb_arg("handle");
  size_t id = (size_t)atoi(handle_str);

  signal_handle *s = _handle_get(id);
  if (!s)
    return string_clone("ERROR: invalid handle");

  char buf[32];
  snprintf(buf, sizeof(buf), "%zu", s->length);
  return string_clone(buf);
}

/* ── Tests ── */

/* 11. Create signal → verify handle and length */
enx_test(should_opaque_handle_create_signal) {
  _handle_registry_reset();
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "create_signal", "length", verb_create_signal, NULL);
  verb_register(ctx, "signal_length", "handle", verb_signal_length, NULL);

  char *response = NULL;
  int r = verb_exec(ctx, "create_signal 10", &response);
  enx_assert_equal_int(r, RESULT_OK);
  enx_assert_ptr_not_null(response);
  enx_assert_equal_string(response, "0"); /* first handle = 0 */
  free(response);

  /* Verify length */
  r = verb_exec(ctx, "signal_length 0", &response);
  enx_assert_equal_int(r, RESULT_OK);
  enx_assert_equal_string(response, "10");
  free(response);

  verb_context_dispose(ctx);
  _handle_registry_reset();
}

/* 12. rolling_mean(input, 3) → verify smoothed values */
enx_test(should_opaque_handle_rolling_mean) {
  _handle_registry_reset();
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "create_signal", "length", verb_create_signal, NULL);
  verb_register(ctx, "rolling_mean", "handle window", verb_rolling_mean, NULL);
  verb_register(ctx, "signal_value_at", "handle index", verb_signal_value_at,
                NULL);

  /* Create source signal [1,2,3,4,5] */
  char *response = NULL;
  verb_exec(ctx, "create_signal 5", &response);
  free(response);

  /* Apply rolling_mean with window=3 */
  verb_exec(ctx, "rolling_mean 0 3", &response);
  enx_assert_equal_string(response, "1"); /* second handle = 1 */
  free(response);

  /* Verify: index 0 = 1.0 (only itself) */
  verb_exec(ctx, "signal_value_at 1 0", &response);
  enx_assert_equal_string(response, "1.000000");
  free(response);

  /* Verify: index 2 = (1+2+3)/3 = 2.0 */
  verb_exec(ctx, "signal_value_at 1 2", &response);
  enx_assert_equal_string(response, "2.000000");
  free(response);

  /* Verify: index 4 = (3+4+5)/3 = 4.0 */
  verb_exec(ctx, "signal_value_at 1 4", &response);
  enx_assert_equal_string(response, "4.000000");
  free(response);

  verb_context_dispose(ctx);
  _handle_registry_reset();
}

/* 13. Full pipeline: create → transform → read → dispose */
enx_test(should_opaque_handle_full_pipeline) {
  _handle_registry_reset();
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "create_signal", "length", verb_create_signal, NULL);
  verb_register(ctx, "rolling_mean", "handle window", verb_rolling_mean, NULL);
  verb_register(ctx, "signal_value_at", "handle index", verb_signal_value_at,
                NULL);
  verb_register(ctx, "signal_length", "handle", verb_signal_length, NULL);

  /* Pipeline: create_signal(20) → rolling_mean(h, 5) → read values */
  char *h1 = NULL, *h2 = NULL, *resp = NULL;

  verb_exec(ctx, "create_signal 20", &h1);
  enx_assert_equal_string(h1, "0");

  /* Build sentence dynamically: "rolling_mean 0 5" */
  char sentence[128];
  snprintf(sentence, sizeof(sentence), "rolling_mean %s 5", h1);
  verb_exec(ctx, sentence, &h2);
  enx_assert_equal_string(h2, "1");

  /* Verify smoothed signal length */
  snprintf(sentence, sizeof(sentence), "signal_length %s", h2);
  verb_exec(ctx, sentence, &resp);
  enx_assert_equal_string(resp, "20");
  free(resp);

  /* Verify smoothed signal value at index 10 */
  /* raw[10]=11, window=5 → mean(7,8,9,10,11) = 9.0 */
  snprintf(sentence, sizeof(sentence), "signal_value_at %s 10", h2);
  verb_exec(ctx, sentence, &resp);
  enx_assert_equal_string(resp, "9.000000");
  free(resp);

  free(h1);
  free(h2);
  verb_context_dispose(ctx);
  _handle_registry_reset();
}

/* 14. Invalid handle returns error */
enx_test(should_opaque_handle_invalid_returns_error) {
  _handle_registry_reset();
  verb_context *ctx = verb_context_create();

  verb_register(ctx, "signal_value_at", "handle index", verb_signal_value_at,
                NULL);

  char *response = NULL;
  int r = verb_exec(ctx, "signal_value_at 999 0", &response);
  enx_assert_equal_int(r, RESULT_CORE_VERB_ERROR);
  enx_assert_ptr_not_null(response);
  enx_assert_true(strstr(response, "ERROR:") != NULL);
  free(response);

  verb_context_dispose(ctx);
  _handle_registry_reset();
}

enx_test_case_t vm_verb_opaque_handle_tests[] = {
    enx_test_case(should_opaque_handle_create_signal),
    enx_test_case(should_opaque_handle_rolling_mean),
    enx_test_case(should_opaque_handle_full_pipeline),
    enx_test_case(should_opaque_handle_invalid_returns_error),
    enx_test_case_end()};

// --- main.c ---
/*
 * main.c
 *
 * Copyright 2024 Enjector Software, Ltd. All Rights Reserved.
 *
 */

#include "test.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//

//

static test_suite tests[] = {

    enx_test_case(vm_module_compiler_tests),
    enx_test_case(vm_module_declare_function_tests),
    enx_test_case(vm_module_function_call_tests),
    // enx_test_case(vm_module_function_tests),
    enx_test_case(vm_module_runtime_tests), enx_test_case(vm_array_ops_tests),
    //   enx_test_case( vm_eval_tests ),

    enx_test_case(vm_verb_tests), enx_test_case(vm_verb_opaque_handle_tests),
    enx_test_case(vm_ts_wiring_tests),

    enx_test_case_end()};

int main(int argc, const char *argv[]) {
  // system_enable_catch_exceptions();

  core_result_to_string_register();
  xmemory_report_clear(); // Ignore memory used by the result registration

  log_set_enable_debug(false);
  log_set_enable_info(true);
  log_set_enable_warn(false);
  log_set_enable_error(true);

  bool result = test_suite_run(tests);

  // #ifdef _WIN32    // For running via Visual Studio
  //     printf("End, press key to close\n");
  //     getchar();
  // #endif
  result_to_string_register_clear();

  // Return 0 for success and 1 fo failure
  return result ? 0 : 1;
}
