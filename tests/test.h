/*
 * MicroGPT-C — Test Framework Header
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ENX_TEST_H
#define ENX_TEST_H

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define enx_assert_equal_int(a, b)                                             \
  do {                                                                         \
    if ((int)(a) != (int)(b)) {                                                \
      printf("ASSERT FAILED: %d != %d at %s:%d\n", (int)(a), (int)(b),         \
             __FILE__, __LINE__);                                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define enx_assert_true(cond)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("ASSERT FAILED: expected true at %s:%d\n", __FILE__, __LINE__);   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define enx_assert_false(cond)                                                 \
  do {                                                                         \
    if ((cond)) {                                                              \
      printf("ASSERT FAILED: expected false at %s:%d\n", __FILE__, __LINE__);  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define enx_assert_ptr_not_null(ptr)                                           \
  do {                                                                         \
    if ((ptr) == NULL) {                                                       \
      printf("ASSERT FAILED: pointer is null at %s:%d\n", __FILE__, __LINE__); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define enx_assert_equal_ptr(a, b)                                             \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      printf("ASSERT FAILED: pointers not equal at %s:%d\n", __FILE__,         \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define enx_assert_equal_size(a, b)                                            \
  do {                                                                         \
    if ((size_t)(a) != (size_t)(b)) {                                          \
      printf("ASSERT FAILED: %zu != %zu at %s:%d\n", (size_t)(a), (size_t)(b), \
             __FILE__, __LINE__);                                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define enx_assert_equal_string(a, b)                                          \
  do {                                                                         \
    if (strcmp((a), (b)) != 0) {                                               \
      printf("ASSERT FAILED: '%s' != '%s' at %s:%d\n", (a), (b), __FILE__,     \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define enx_assert_equal_bool(a, b) enx_assert_equal_int((int)(a), (int)(b))
#define enx_assert_equal_double(a, b)                                          \
  do {                                                                         \
    if ((double)(a) != (double)(b)) {                                          \
      printf("ASSERT FAILED: %f != %f at %s:%d\n", (double)(a), (double)(b),   \
             __FILE__, __LINE__);                                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define enx_assert_fail()                                                      \
  do {                                                                         \
    printf("ASSERT FAILED: explicit fail at %s:%d\n", __FILE__, __LINE__);     \
    exit(1);                                                                   \
  } while (0)

#define TEST_ASSERT_EQUAL_RESULT_FATAL(a, b) enx_assert_equal_int(a, b)
#define TEST_ASSERT_EQUAL_STRING_FATAL(a, b) enx_assert_equal_string(a, b)
#define TEST_ASSERT_EQUAL_INT_FATAL(a, b) enx_assert_equal_int(a, b)

typedef struct {
  const char *name;
  void (*func)(void);
} enx_test_case_t;

typedef struct {
  const char *name;
  enx_test_case_t *cases;
} test_suite_t;

typedef test_suite_t test_suite;

#define enx_test(name) void name(void)
#define enx_test_case(arr) {#arr, arr}
#define enx_test_case_end() {NULL, NULL}

static inline bool test_suite_run(test_suite *suites) {
  int passed = 0;
  int failed = 0;
  for (int i = 0; suites[i].name != NULL; i++) {
    printf("Running suite: %s\n", suites[i].name);
    for (int j = 0; suites[i].cases[j].func != NULL; j++) {
      // printf("  Running test: %s\n", suites[i].cases[j].name);
      suites[i].cases[j].func();
      passed++;
    }
  }
  printf("All tests finished. Passed: %d, Failed: %d\n", passed, failed);
  return failed == 0;
}

// Stubs for debug/diagnostics functions (no-ops in test builds)
#define core_result_to_string_register() (void)0
#define result_to_string_register_clear() (void)0
#define vm_memory_report_clear() (void)0
#define vm_memory_report_exit_on_leaks() (void)0
#define log_set_enable_debug(x) (void)(x)
#define log_set_enable_info(x) (void)(x)
#define log_set_enable_warn(x) (void)(x)
#define log_set_enable_error(x) (void)(x)

#endif // ENX_TEST_H
