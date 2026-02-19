/*
 * MicroGPT-C Organelle Pipeline Unit Tests
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Tests for the OPA library: OpaKanban, OpaCycleDetector,
 * pipe-string helpers, and multi-line corpus loader.
 * Zero dependencies — uses the same assertion harness as test_microgpt.c.
 *
 * Build:  cmake --build build --target test_microgpt_organelle
 * Run:    ./build/test_microgpt_organelle
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Minimal test harness (same as test_microgpt.c) ---- */

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name)                                                             \
  static void test_##name(void);                                               \
  static void run_##name(void) {                                               \
    g_tests_run++;                                                             \
    printf("  %-50s ", #name);                                                 \
    fflush(stdout);                                                            \
    test_##name();                                                             \
    printf("PASS\n");                                                          \
    fflush(stdout);                                                            \
    g_tests_passed++;                                                          \
  }                                                                            \
  static void test_##name(void)

#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL\n");                                                        \
      fprintf(stderr, "    Assertion failed: %s\n    at %s:%d\n", #cond,       \
              __FILE__, __LINE__);                                             \
      g_tests_failed++;                                                        \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_GT(a, b) ASSERT((a) > (b))
#define ASSERT_LT(a, b) ASSERT((a) < (b))
#define ASSERT_STREQ(a, b) ASSERT(strcmp((a), (b)) == 0)

#define RUN(name) run_##name()

/* ---- Helper: create a temp file with given content ---- */

static const char *write_temp_file(const char *name, const char *content) {
  FILE *f = fopen(name, "wb");
  if (f) {
    fputs(content, f);
    fclose(f);
  }
  return name;
}

/* ==================================================================== */
/*                        OpaKanban TESTS                                */
/* ==================================================================== */

TEST(kanban_init_zeroes_state) {
  OpaKanban kb;
  opa_kanban_init(&kb, 5);
  ASSERT_STREQ(kb.blocked, "");
  ASSERT_STREQ(kb.last, "");
  ASSERT_EQ(kb.stalls, 0);
  ASSERT_EQ(kb.replans, 0);
  ASSERT_EQ(kb.max_history, 5);
}

TEST(kanban_add_blocked_single) {
  OpaKanban kb;
  opa_kanban_init(&kb, 5);
  opa_kanban_add_blocked(&kb, "up");
  ASSERT_STREQ(kb.blocked, "up");
  ASSERT_EQ(opa_kanban_is_blocked(&kb, "up"), 1);
  ASSERT_EQ(opa_kanban_is_blocked(&kb, "down"), 0);
}

TEST(kanban_add_blocked_multiple) {
  OpaKanban kb;
  opa_kanban_init(&kb, 5);
  opa_kanban_add_blocked(&kb, "up");
  opa_kanban_add_blocked(&kb, "down");
  opa_kanban_add_blocked(&kb, "left");
  ASSERT_EQ(opa_kanban_is_blocked(&kb, "up"), 1);
  ASSERT_EQ(opa_kanban_is_blocked(&kb, "down"), 1);
  ASSERT_EQ(opa_kanban_is_blocked(&kb, "left"), 1);
  ASSERT_EQ(opa_kanban_is_blocked(&kb, "right"), 0);
}

TEST(kanban_add_blocked_no_duplicates) {
  OpaKanban kb;
  opa_kanban_init(&kb, 5);
  opa_kanban_add_blocked(&kb, "up");
  opa_kanban_add_blocked(&kb, "up");
  opa_kanban_add_blocked(&kb, "up");
  /* Should still just be "up", not "up,up,up" */
  ASSERT_STREQ(kb.blocked, "up");
}

TEST(kanban_clear_blocked) {
  OpaKanban kb;
  opa_kanban_init(&kb, 5);
  opa_kanban_add_blocked(&kb, "up");
  opa_kanban_add_blocked(&kb, "down");
  opa_kanban_clear_blocked(&kb);
  ASSERT_STREQ(kb.blocked, "");
  ASSERT_EQ(opa_kanban_is_blocked(&kb, "up"), 0);
  ASSERT_EQ(opa_kanban_is_blocked(&kb, "down"), 0);
}

TEST(kanban_add_last_basic) {
  OpaKanban kb;
  opa_kanban_init(&kb, 3);
  opa_kanban_add_last(&kb, "up");
  ASSERT_STREQ(kb.last, "up");
  opa_kanban_add_last(&kb, "down");
  ASSERT_STREQ(kb.last, "up,down");
  opa_kanban_add_last(&kb, "left");
  ASSERT_STREQ(kb.last, "up,down,left");
}

TEST(kanban_add_last_drops_oldest) {
  OpaKanban kb;
  opa_kanban_init(&kb, 3);
  opa_kanban_add_last(&kb, "up");
  opa_kanban_add_last(&kb, "down");
  opa_kanban_add_last(&kb, "left");
  /* max_history=3, adding 4th should drop "up" */
  opa_kanban_add_last(&kb, "right");
  ASSERT_STREQ(kb.last, "down,left,right");
}

TEST(kanban_add_last_disabled_when_zero) {
  OpaKanban kb;
  opa_kanban_init(&kb, 0);
  opa_kanban_add_last(&kb, "up");
  ASSERT_STREQ(kb.last, "");
}

TEST(kanban_stalls_and_replans) {
  OpaKanban kb;
  opa_kanban_init(&kb, 5);
  kb.stalls = 3;
  kb.replans = 2;
  ASSERT_EQ(kb.stalls, 3);
  ASSERT_EQ(kb.replans, 2);
}

/* ==================================================================== */
/*                     OpaCycleDetector TESTS                             */
/* ==================================================================== */

TEST(cycle_init_empty) {
  OpaCycleDetector cd;
  opa_cycle_init(&cd);
  ASSERT_EQ(cd.len, 0);
  ASSERT_EQ(opa_cycle_detected(&cd, 0), 0);
}

TEST(cycle_no_false_positive_short) {
  OpaCycleDetector cd;
  opa_cycle_init(&cd);
  opa_cycle_record(&cd, 0);
  opa_cycle_record(&cd, 1);
  ASSERT_EQ(opa_cycle_detected(&cd, 0), 0);
}

TEST(cycle_detects_ab_pattern) {
  OpaCycleDetector cd;
  opa_cycle_init(&cd);
  opa_cycle_record(&cd, 0); /* A */
  opa_cycle_record(&cd, 1); /* B */
  opa_cycle_record(&cd, 0); /* A */
  /* Proposing B should detect cycle: A,B,A,B */
  ASSERT_EQ(opa_cycle_detected(&cd, 1), 1);
  /* Proposing A or C should not */
  ASSERT_EQ(opa_cycle_detected(&cd, 0), 0);
  ASSERT_EQ(opa_cycle_detected(&cd, 2), 0);
}

TEST(cycle_other_returns_partner) {
  OpaCycleDetector cd;
  opa_cycle_init(&cd);
  opa_cycle_record(&cd, 0);
  opa_cycle_record(&cd, 1);
  opa_cycle_record(&cd, 0);
  int other = opa_cycle_other(&cd, 1);
  ASSERT_EQ(other, 0);
}

TEST(cycle_record_wraps_window) {
  OpaCycleDetector cd;
  opa_cycle_init(&cd);
  for (int i = 0; i < OPA_CYCLE_WINDOW + 4; i++)
    opa_cycle_record(&cd, i % 4);
  ASSERT_EQ(cd.len, OPA_CYCLE_WINDOW + 4);
  int last_idx = (cd.len - 1) % OPA_CYCLE_WINDOW;
  ASSERT(cd.history[last_idx] >= 0 && cd.history[last_idx] <= 3);
}

/* ==================================================================== */
/*                     PIPE-STRING HELPER TESTS                           */
/* ==================================================================== */

TEST(extract_pipe_value_basic) {
  char buf[128];
  strcpy(buf, "board=XO_|empties=6|move=3");
  const char *val = opa_extract_pipe_value(buf, "board");
  ASSERT_NE(val, NULL);
  ASSERT_STREQ(val, "XO_");
}

TEST(extract_pipe_value_last_field) {
  char buf[128];
  strcpy(buf, "board=XO_|empties=6|move=3");
  const char *val = opa_extract_pipe_value(buf, "move");
  ASSERT_NE(val, NULL);
  ASSERT_STREQ(val, "3");
}

TEST(extract_pipe_value_missing_key) {
  char buf[128];
  strcpy(buf, "board=XO_|empties=6");
  const char *val = opa_extract_pipe_value(buf, "missing");
  ASSERT_EQ(val, NULL);
}

TEST(pipe_starts_with_match) {
  ASSERT_EQ(opa_pipe_starts_with("action=up|", "action="), 1);
}

TEST(pipe_starts_with_no_match) {
  ASSERT_EQ(opa_pipe_starts_with("action=up|", "board="), 0);
}

/* ==================================================================== */
/*                   MULTI-LINE CORPUS LOADER TESTS                       */
/* ==================================================================== */

TEST(load_docs_multiline_basic) {
  write_temp_file("_test_ml.txt",
                  "prompt one\nresponse one\n\nprompt two\nresponse two\n");
  Docs docs = {0};
  int rc = opa_load_docs_multiline("_test_ml.txt", &docs, 100);
  ASSERT_EQ(rc, 0);
  ASSERT_EQ(docs.num_docs, 2);
  ASSERT(docs.doc_lens[0] > 0);
  ASSERT(docs.doc_lens[1] > 0);
  free_docs(&docs);
  remove("_test_ml.txt");
}

TEST(load_docs_multiline_missing) {
  Docs docs = {0};
  int rc = opa_load_docs_multiline("_nonexistent_xyz.txt", &docs, 100);
  ASSERT_NE(rc, 0);
}

/* ==================================================================== */
/*                              MAIN                                     */
/* ==================================================================== */

int main(void) {
  printf("\n=== MicroGPT-C Organelle Pipeline Tests ===\n");

  printf("\n[OpaKanban]\n");
  RUN(kanban_init_zeroes_state);
  RUN(kanban_add_blocked_single);
  RUN(kanban_add_blocked_multiple);
  RUN(kanban_add_blocked_no_duplicates);
  RUN(kanban_clear_blocked);
  RUN(kanban_add_last_basic);
  RUN(kanban_add_last_drops_oldest);
  RUN(kanban_add_last_disabled_when_zero);
  RUN(kanban_stalls_and_replans);

  printf("\n[OpaCycleDetector]\n");
  RUN(cycle_init_empty);
  RUN(cycle_no_false_positive_short);
  RUN(cycle_detects_ab_pattern);
  RUN(cycle_other_returns_partner);
  RUN(cycle_record_wraps_window);

  printf("\n[Pipe-String Helpers]\n");
  RUN(extract_pipe_value_basic);
  RUN(extract_pipe_value_last_field);
  RUN(extract_pipe_value_missing_key);
  RUN(pipe_starts_with_match);
  RUN(pipe_starts_with_no_match);

  printf("\n[Multi-Line Corpus Loader]\n");
  RUN(load_docs_multiline_basic);
  RUN(load_docs_multiline_missing);

  /* Summary */
  printf("\n=== Results: %d/%d passed", g_tests_passed, g_tests_run);
  if (g_tests_failed > 0)
    printf(" (%d FAILED)", g_tests_failed);
  printf(" ===\n\n");

  return g_tests_failed > 0 ? 1 : 0;
}
