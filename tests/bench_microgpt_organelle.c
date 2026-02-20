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
/*                    VALID-MOVE FILTER                                   */
/* ==================================================================== */

static void bench_valid_filter(void) {
  BENCH_HEADER("valid_filter + valid_fallback");

  OpaKanban kb;
  int iters = 500000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    opa_valid_filter("3", "0,1,2,3,4,5,6,7,8");
    opa_valid_filter("9", "0,1,2,3,4,5,6,7,8");
    opa_kanban_init(&kb, 3);
    opa_kanban_add_blocked(&kb, "0");
    opa_kanban_add_blocked(&kb, "2");
    char fb[16];
    opa_valid_fallback(&kb, "0,1,2,3,4", fb, sizeof(fb));
  }
  scalar_t ms = elapsed_ms(t0);
  BENCH_RESULT(ms / 1000.0, (scalar_t)iters / (ms / 1000.0), "cycles/s");
}

/* ==================================================================== */
/*                   MULTI-LINE CORPUS LOADER                             */
/* ==================================================================== */

/* Helper to write a temp file for benchmarking */
static const char *bench_write_temp(const char *name, const char *content) {
  FILE *f = fopen(name, "wb");
  if (f) {
    fputs(content, f);
    fclose(f);
  }
  return name;
}

static void bench_multiline_loader(void) {
  BENCH_HEADER("opa_load_docs_multiline (50 docs)");

  /* Create a corpus with 50 short documents */
  char corpus[4096];
  int pos = 0;
  for (int d = 0; d < 50; d++) {
    int n = snprintf(corpus + pos, sizeof(corpus) - (size_t)pos,
                     "prompt %d\nresponse %d\n\n", d, d);
    if (n < 0 || (size_t)n >= sizeof(corpus) - (size_t)pos) {
      /* Truncation or error: stop appending to avoid overflow */
      break;
    }
    pos += n;
  }
  if ((size_t)pos >= sizeof(corpus)) {
    corpus[sizeof(corpus) - 1] = '\0';
  } else {
    corpus[pos] = '\0';
  }
  bench_write_temp("_bench_ml.txt", corpus);

  int iters = 5000;
  clock_t t0 = clock();
  for (int i = 0; i < iters; i++) {
    Docs docs = {0};
    opa_load_docs_multiline("_bench_ml.txt", &docs, 100);
    free_docs(&docs);
  }
  scalar_t ms = elapsed_ms(t0);
  remove("_bench_ml.txt");
  BENCH_RESULT(ms / iters, (scalar_t)iters / (ms / 1000.0), "loads/s");
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

  printf("\n[Valid-Move Filter]\n");
  bench_valid_filter();

  printf("\n[Multi-Line Corpus Loader]\n");
  bench_multiline_loader();

  printf("\n=== Done ===\n\n");
  return 0;
}
