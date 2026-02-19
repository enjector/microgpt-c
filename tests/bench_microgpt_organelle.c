/*
 * MicroGPT-C Organelle Pipeline Benchmarks
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Benchmarks for OPA library operations: kanban state management,
 * cycle detection, and pipe-string parsing.
 * Zero dependencies — uses only clock() for timing.
 *
 * Build:  cmake --build build --target bench_microgpt_organelle
 * Run:    ./build/bench_microgpt_organelle
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Timing helpers ---- */

static scalar_t elapsed_ms(clock_t start) {
  return (scalar_t)(clock() - start) / (scalar_t)CLOCKS_PER_SEC * 1000.0;
}

#define BENCH_HEADER(name)                                                     \
  printf("  %-40s ", name);                                                    \
  fflush(stdout)

#define BENCH_RESULT(ms, metric, unit)                                         \
  printf("%8.2f ms  |  %10.1f %s\n", (ms), (metric), (unit))

/* ==================================================================== */
/*                        BENCHMARKS                                     */
/* ==================================================================== */

static void bench_kanban_ops(void) {
  BENCH_HEADER("kanban add_blocked + is_blocked");

  OpaKanban kb;
  int iters = 500000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    opa_kanban_init(&kb, 5);
    opa_kanban_add_blocked(&kb, "up");
    opa_kanban_add_blocked(&kb, "down");
    opa_kanban_add_blocked(&kb, "left");
    opa_kanban_is_blocked(&kb, "up");
    opa_kanban_is_blocked(&kb, "right");
    opa_kanban_add_last(&kb, "up");
    opa_kanban_add_last(&kb, "down");
    opa_kanban_clear_blocked(&kb);
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (scalar_t)iters / (ms / 1000.0), "cycles/s");
}

static void bench_cycle_detection(void) {
  BENCH_HEADER("cycle detect + record");

  OpaCycleDetector cd;
  int iters = 1000000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    if (i % 100 == 0)
      opa_cycle_init(&cd);
    opa_cycle_record(&cd, i % 4);
    opa_cycle_detected(&cd, (i + 1) % 4);
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (scalar_t)iters / (ms / 1000.0), "ops/s");
}

static void bench_pipe_string_parse(void) {
  BENCH_HEADER("pipe-string extract + starts_with");

  int iters = 500000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    char buf[128];
    strcpy(buf, "board=XO_OX_XOX|empties=3|blocked=up,down|move=left");
    opa_extract_pipe_value(buf, "blocked");
    opa_pipe_starts_with(buf, "board=");
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (scalar_t)iters / (ms / 1000.0), "parses/s");
}

/* ==================================================================== */
/*                            MAIN                                       */
/* ==================================================================== */

int main(void) {
  printf("\n=== MicroGPT-C Organelle Pipeline Benchmarks ===\n\n");

  printf("[OpaKanban]\n");
  bench_kanban_ops();

  printf("\n[OpaCycleDetector]\n");
  bench_cycle_detection();

  printf("\n[Pipe-String Parsing]\n");
  bench_pipe_string_parse();

  printf("\n=== Done ===\n\n");
  return 0;
}
