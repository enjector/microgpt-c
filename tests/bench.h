/*
 * bench.h  —  Lightweight benchmark harness for VM engine tests.
 *
 * Header-only. Provides benchmark fixture definitions and a suite runner
 * that measures wall-clock throughput (transactions per second).
 *
 * Usage:
 *   benchmark my_benchmarks[] = {
 *       BENCHMARK_CASE(bench_compile),
 *       {BENCHMARK_END}
 *   };
 *   benchmark_suite suites[] = {
 *       BENCHMARK_CASE(my_benchmarks),
 *       {BENCHMARK_END}
 *   };
 *   benchmark_suite_run(suites, NULL);
 *
 * Copyright 2024 Enjector Software Ltd. MIT License.
 */

#ifndef BENCH_H
#define BENCH_H

#include "microgpt_vm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ── Portable high-resolution timing ── */
#ifdef _WIN32
#include <windows.h>
/* Provide clock_gettime / CLOCK_MONOTONIC shim for MSVC */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
static inline int clock_gettime(int clk_id, struct timespec *tp) {
  (void)clk_id;
  LARGE_INTEGER freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  tp->tv_sec = (long)(count.QuadPart / freq.QuadPart);
  tp->tv_nsec =
      (long)((count.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
  return 0;
}
#endif

/* ── Types ── */

/** A single benchmark: a name and a function that returns an iteration count
 *  (or BENCHMARK_RESULT_CORE_FAILED / _IGNORE). */
typedef struct benchmark_t {
  const char *name;
  int (*benchmark_function)();
} benchmark;

/** A named group of benchmarks. */
typedef struct benchmark_suite_t {
  const char *name;
  benchmark *benchmarks;
} benchmark_suite;

/* ── Constants ── */

#define BENCHMARK_RESULT_CORE_FAILED -1
#define BENCHMARK_RESULT_CORE_IGNORE -2
#define BENCHMARK_END NULL

/** Shorthand to create a benchmark_suite or benchmark entry from an array. */
#define BENCHMARK_CASE(function) {#function, function}

/* ── Runner ── */

/**
 * Execute all benchmark suites, print per-benchmark throughput, and
 * emit a Markdown summary table to stdout.
 *
 * @param fixtures         NULL-terminated array of benchmark_suite.
 * @param optional_output_path  Reserved (unused).
 * @return true if all benchmarks passed.
 */
static inline bool benchmark_suite_run(benchmark_suite fixtures[],
                                       const char *optional_output_path) {
  (void)optional_output_path;
  VM_ASSERT_NOT_NULL(fixtures);
  bool all_passed = true;
  int i = 0;
  benchmark_suite *fixture;

  vm_string_buffer *summary = vm_string_buffer_create_empty();
  vm_string_buffer_append(summary, "Benchmark Run\n");
  vm_string_buffer_append(summary, "| Area | Name | Transactions per Second |\n");
  vm_string_buffer_append(summary, "| --- | --- | --- |\n");

  while ((fixture = &fixtures[i++]) && fixture->name != BENCHMARK_END) {
    VM_ASSERT_NOT_NULL(fixture);
    printf("Benchmark: %s\n", fixture->name);
    int j = 0;
    benchmark *bench;

    while ((bench = &fixture->benchmarks[j++]) &&
           bench->name != BENCHMARK_END) {
      VM_ASSERT_NOT_NULL(bench);

      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);

      const int vm_result = bench->benchmark_function();

      switch (vm_result) {
      case BENCHMARK_RESULT_CORE_IGNORE:
        all_passed = true;
        break;

      case BENCHMARK_RESULT_CORE_FAILED:
        printf("\tFAILED : %s\n", bench->name);
        vm_string_buffer_append_format(summary, "| %s | %s | ? |\n", fixture->name,
                                    bench->name);
        all_passed = false;
        break;

      default: {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed =
            (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        size_t rate = 0;
        if (elapsed > 0) {
          rate = (size_t)(vm_result / elapsed);
        }

        printf("\tSUCCESS: %s: %zu per second (count:=%d, elapsed:=%.4fs)\n",
               bench->name, rate, vm_result, elapsed);

        char rate_formatted[64] = "0";
        if (rate > 0)
          snprintf(rate_formatted, sizeof(rate_formatted), "%zu", rate);

        vm_string_buffer_append_format(summary, "| %s | %s | %s |\n",
                                    fixture->name, bench->name, rate_formatted);
        break;
      }
      }
    }
    printf("\n");
  }

  if (all_passed) {
    printf("All benchmark tests succeeded\n");
  } else {
    printf("Some benchmark tests failed\n");
  }

  printf("\nBenchmark Summary (Markdown):\n\n");
  puts((char *)summary->data);
  printf("\n\n");

  vm_string_buffer_free(summary);
  return all_passed;
}

#endif /* BENCH_H */
