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
/*                     VALID-MOVE FILTER TESTS                            */
/* ==================================================================== */

TEST(valid_filter_match) { ASSERT_EQ(opa_valid_filter("3", "0,2,3,5"), 1); }

TEST(valid_filter_no_match) { ASSERT_EQ(opa_valid_filter("4", "0,2,3,5"), 0); }

TEST(valid_filter_empty_list) {
  /* empty list = no constraint, returns 1 */
  ASSERT_EQ(opa_valid_filter("4", ""), 1);
  ASSERT_EQ(opa_valid_filter("4", NULL), 1);
}

TEST(valid_filter_single_item) {
  ASSERT_EQ(opa_valid_filter("3", "3"), 1);
  ASSERT_EQ(opa_valid_filter("4", "3"), 0);
}

TEST(valid_filter_direction_strings) {
  ASSERT_EQ(opa_valid_filter("up", "up,down,left"), 1);
  ASSERT_EQ(opa_valid_filter("right", "up,down,left"), 0);
}

TEST(valid_fallback_picks_first_unblocked) {
  OpaKanban kb;
  opa_kanban_init(&kb, 3);
  opa_kanban_add_blocked(&kb, "0");
  opa_kanban_add_blocked(&kb, "2");
  char fb[16];
  int found = opa_valid_fallback(&kb, "0,2,3,5", fb, sizeof(fb));
  ASSERT_EQ(found, 1);
  ASSERT_STREQ(fb, "3");
}

TEST(valid_fallback_all_blocked) {
  OpaKanban kb;
  opa_kanban_init(&kb, 3);
  opa_kanban_add_blocked(&kb, "0");
  opa_kanban_add_blocked(&kb, "1");
  char fb[16];
  int found = opa_valid_fallback(&kb, "0,1", fb, sizeof(fb));
  ASSERT_EQ(found, 0);
}

/* ==================================================================== */
/*              MULTI-LINE LOADER EDGE CASES                              */
/* ==================================================================== */

TEST(load_docs_multiline_single_doc) {
  /* A corpus with no blank lines should produce exactly one document */
  write_temp_file("_test_ml_single.txt", "line one\nline two\nline three\n");
  Docs docs = {0};
  int rc = opa_load_docs_multiline("_test_ml_single.txt", &docs, 100);
  ASSERT_EQ(rc, 0);
  ASSERT_EQ(docs.num_docs, 1);
  ASSERT(docs.doc_lens[0] > 0);
  free_docs(&docs);
  remove("_test_ml_single.txt");
}

TEST(load_docs_multiline_trailing_newlines) {
  /* Multiple trailing newlines should not create extra empty docs */
  write_temp_file("_test_ml_trail.txt", "doc one\n\n\n\ndoc two\n\n\n\n\n");
  Docs docs = {0};
  int rc = opa_load_docs_multiline("_test_ml_trail.txt", &docs, 100);
  ASSERT_EQ(rc, 0);
  ASSERT_EQ(docs.num_docs, 2);
  free_docs(&docs);
  remove("_test_ml_trail.txt");
}

TEST(load_docs_multiline_max_docs_limit) {
  /* max_docs should cap the number of documents loaded */
  write_temp_file("_test_ml_max.txt", "doc1\n\ndoc2\n\ndoc3\n\ndoc4\n\ndoc5\n");
  Docs docs = {0};
  int rc = opa_load_docs_multiline("_test_ml_max.txt", &docs, 2);
  ASSERT_EQ(rc, 0);
  ASSERT_EQ(docs.num_docs, 2);
  free_docs(&docs);
  remove("_test_ml_max.txt");
}

TEST(load_docs_multiline_empty_file) {
  /* An empty file should produce zero documents, not crash */
  write_temp_file("_test_ml_empty.txt", "");
  Docs docs = {0};
  int rc = opa_load_docs_multiline("_test_ml_empty.txt", &docs, 100);
  ASSERT_EQ(rc, 0);
  ASSERT_EQ(docs.num_docs, 0);
  free_docs(&docs);
  remove("_test_ml_empty.txt");
}

/* ==================================================================== */
/*                     ORGANELLE LIFECYCLE TESTS                          */
/* ==================================================================== */

TEST(organelle_free_null_safe) {
  /* organelle_free(NULL) must be a no-op, not a crash */
  organelle_free(NULL);
  /* If we reach here, the test passed */
}

/* ==================================================================== */
/*                   VALID-FILTER EDGE CASES                              */
/* ==================================================================== */

TEST(valid_filter_empty_action) {
  /* Empty action should never match anything */
  ASSERT_EQ(opa_valid_filter("", "0,2,3"), 0);
  ASSERT_EQ(opa_valid_filter(NULL, "0,2,3"), 0);
}

TEST(valid_filter_no_partial_substring_match) {
  /* "up" should NOT match "updown" — requires exact field match */
  ASSERT_EQ(opa_valid_filter("up", "updown,left,right"), 0);
  ASSERT_EQ(opa_valid_filter("updown", "up,left,right"), 0);
}

TEST(valid_fallback_empty_csv) {
  OpaKanban kb;
  opa_kanban_init(&kb, 3);
  char fb[16];
  int found = opa_valid_fallback(&kb, "", fb, sizeof(fb));
  ASSERT_EQ(found, 0);
  found = opa_valid_fallback(&kb, NULL, fb, sizeof(fb));
  ASSERT_EQ(found, 0);
}

/* ==================================================================== */
/*                   ENSEMBLE VOTING CONSTANTS                            */
/* ==================================================================== */

TEST(ensemble_constants_valid) {
  /* OPA_MAX_VOTES must be positive */
  ASSERT_GT(OPA_MAX_VOTES, 0);
  /* OPA_TEMP_JITTER must be non-negative */
  ASSERT(OPA_TEMP_JITTER >= 0.0);
}

/* ==================================================================== */
/*                     REASONING TRACE TESTS                              */
/* ==================================================================== */

TEST(trace_init_empty) {
  OpaTrace tr;
  opa_trace_init(&tr, 12);
  ASSERT_EQ(tr.num_steps, 0);
  ASSERT_EQ(tr.initial_metric, 12);
  ASSERT_EQ(tr.final_metric, 12);
  ASSERT_EQ(tr.solved, 0);
}

TEST(trace_record_single) {
  OpaTrace tr;
  opa_trace_init(&tr, 10);
  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 10, 9, "", 1);
  ASSERT_EQ(tr.num_steps, 1);
  ASSERT_STREQ(tr.steps[0].action, "up");
  ASSERT_EQ(tr.steps[0].outcome, OPA_STEP_ACCEPTED);
  ASSERT_EQ(tr.steps[0].metric_before, 10);
  ASSERT_EQ(tr.steps[0].metric_after, 9);
  ASSERT_EQ(tr.steps[0].from_model, 1);
  ASSERT_EQ(tr.steps[0].step, 1);
}

TEST(trace_record_sequence) {
  OpaTrace tr;
  opa_trace_init(&tr, 12);
  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 12, 11, "", 1);
  opa_trace_record(&tr, "right", OPA_STEP_ACCEPTED, 11, 10, "", 1);
  opa_trace_record(&tr, "up", OPA_STEP_REJECTED, 10, -1, "up", 1);
  opa_trace_record(&tr, "left", OPA_STEP_STALL, 10, 10, "", 0);
  ASSERT_EQ(tr.num_steps, 4);
  ASSERT_STREQ(tr.steps[0].action, "up");
  ASSERT_STREQ(tr.steps[1].action, "right");
  ASSERT_STREQ(tr.steps[2].action, "up");
  ASSERT_STREQ(tr.steps[3].action, "left");
  ASSERT_EQ(tr.steps[2].metric_after, -1);
  ASSERT_STREQ(tr.steps[2].blocked_snapshot, "up");
  ASSERT_EQ(tr.steps[3].from_model, 0);
}

TEST(trace_record_overflow) {
  OpaTrace tr;
  opa_trace_init(&tr, 20);
  for (int i = 0; i < OPA_TRACE_MAX_STEPS + 10; i++) {
    opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 20 - i, 19 - i, "", 1);
  }
  /* Should clamp at max, not overflow */
  ASSERT_EQ(tr.num_steps, OPA_TRACE_MAX_STEPS);
}

TEST(trace_finalise) {
  OpaTrace tr;
  opa_trace_init(&tr, 12);
  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 12, 11, "", 1);
  opa_trace_finalise(&tr, 0, 1);
  ASSERT_EQ(tr.final_metric, 0);
  ASSERT_EQ(tr.solved, 1);
}

TEST(trace_count_outcomes) {
  OpaTrace tr;
  opa_trace_init(&tr, 10);
  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 10, 9, "", 1);
  opa_trace_record(&tr, "right", OPA_STEP_REJECTED, 9, -1, "", 1);
  opa_trace_record(&tr, "left", OPA_STEP_ACCEPTED, 9, 8, "", 1);
  opa_trace_record(&tr, "down", OPA_STEP_STALL, 8, 8, "", 0);
  opa_trace_record(&tr, "up", OPA_STEP_REPLAN, 8, 8, "", 0);
  opa_trace_record(&tr, "right", OPA_STEP_CYCLE_BREAK, 8, 7, "", 1);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_ACCEPTED), 2);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_REJECTED), 1);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_STALL), 1);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_REPLAN), 1);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_CYCLE_BREAK), 1);
}

TEST(trace_has_recovery) {
  /* md increases (regression) then decreases (recovery) */
  OpaTrace tr;
  opa_trace_init(&tr, 10);
  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 10, 9, "", 1);
  opa_trace_record(&tr, "left", OPA_STEP_ACCEPTED, 9, 11, "",
                   1); /* regression */
  opa_trace_record(&tr, "down", OPA_STEP_ACCEPTED, 11, 8, "",
                   1); /* recovery! */
  ASSERT_EQ(opa_trace_has_recovery(&tr), 1);
}

TEST(trace_no_false_recovery) {
  /* Monotonically decreasing trace — no recovery */
  OpaTrace tr;
  opa_trace_init(&tr, 10);
  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 10, 9, "", 1);
  opa_trace_record(&tr, "right", OPA_STEP_ACCEPTED, 9, 8, "", 1);
  opa_trace_record(&tr, "down", OPA_STEP_ACCEPTED, 8, 7, "", 1);
  ASSERT_EQ(opa_trace_has_recovery(&tr), 0);
}

TEST(trace_to_corpus_format) {
  OpaTrace tr;
  opa_trace_init(&tr, 10);
  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 10, 9, "", 1);
  opa_trace_record(&tr, "right", OPA_STEP_REJECTED, 9, -1, "up", 1);
  opa_trace_finalise(&tr, 9, 0);

  char buf[2048];
  int len = opa_trace_to_corpus(&tr, buf, sizeof(buf));
  ASSERT_GT(len, 0);
  /* Header line should be present */
  ASSERT(strstr(buf, "TRACE|") != NULL);
  ASSERT(strstr(buf, "initial=10") != NULL);
  ASSERT(strstr(buf, "solved=0") != NULL);
  ASSERT(strstr(buf, "steps=2") != NULL);
  /* Step lines should contain action and outcome */
  ASSERT(strstr(buf, "up") != NULL);
  ASSERT(strstr(buf, "accepted") != NULL);
  ASSERT(strstr(buf, "rejected") != NULL);
  ASSERT(strstr(buf, "model") != NULL);
}

TEST(trace_to_corpus_empty) {
  OpaTrace tr;
  opa_trace_init(&tr, 0);
  opa_trace_finalise(&tr, 0, 1);
  char buf[256];
  int len = opa_trace_to_corpus(&tr, buf, sizeof(buf));
  ASSERT_GT(len, 0);
  ASSERT(strstr(buf, "TRACE|") != NULL);
  ASSERT(strstr(buf, "steps=0") != NULL);
}

TEST(trace_write_file) {
  OpaTrace tr;
  opa_trace_init(&tr, 10);
  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 10, 9, "", 1);
  opa_trace_finalise(&tr, 9, 0);

  const char *path = "_test_trace_output.txt";
  remove(path); /* ensure clean start */
  int rc = opa_trace_write(&tr, path);
  ASSERT_EQ(rc, 0);

  /* Verify file exists and has content */
  FILE *f = fopen(path, "r");
  ASSERT_NE(f, NULL);
  char fbuf[2048];
  size_t nread = fread(fbuf, 1, sizeof(fbuf) - 1, f);
  fbuf[nread] = '\0';
  fclose(f);
  ASSERT_GT((int)nread, 0);
  ASSERT(strstr(fbuf, "TRACE|") != NULL);
  remove(path);
}

TEST(trace_pipeline_simulation) {
  /* Simulate a full pipeline: accept, reject, stall, replan, cycle-break, solve
   */
  OpaTrace tr;
  opa_trace_init(&tr, 12);

  opa_trace_record(&tr, "up", OPA_STEP_ACCEPTED, 12, 11, "", 1);
  opa_trace_record(&tr, "right", OPA_STEP_ACCEPTED, 11, 10, "", 1);
  opa_trace_record(&tr, "up", OPA_STEP_REJECTED, 10, -1, "up", 1);
  opa_trace_record(&tr, "left", OPA_STEP_STALL, 10, 10, "", 0);
  opa_trace_record(&tr, "down", OPA_STEP_STALL, 10, 10, "", 0);
  opa_trace_record(&tr, "up", OPA_STEP_REPLAN, 10, 10, "", 0);
  opa_trace_record(&tr, "right", OPA_STEP_ACCEPTED, 10, 11, "",
                   1); /* regression */
  opa_trace_record(&tr, "down", OPA_STEP_CYCLE_BREAK, 11, 9, "up,left", 1);
  opa_trace_record(&tr, "left", OPA_STEP_ACCEPTED, 9, 0, "", 1); /* solved! */
  opa_trace_finalise(&tr, 0, 1);

  /* Verify counts */
  ASSERT_EQ(tr.num_steps, 9);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_ACCEPTED), 4);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_REJECTED), 1);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_STALL), 2);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_REPLAN), 1);
  ASSERT_EQ(opa_trace_count(&tr, OPA_STEP_CYCLE_BREAK), 1);

  /* Verify recovery (regression at step 7, then progress at step 8) */
  ASSERT_EQ(opa_trace_has_recovery(&tr), 1);

  /* Verify solved */
  ASSERT_EQ(tr.solved, 1);
  ASSERT_EQ(tr.final_metric, 0);
  ASSERT_EQ(tr.initial_metric, 12);

  /* Verify corpus serialisation */
  char buf[4096];
  int len = opa_trace_to_corpus(&tr, buf, sizeof(buf));
  ASSERT_GT(len, 0);
  ASSERT(strstr(buf, "TRACE|") != NULL);
  ASSERT(strstr(buf, "solved=1") != NULL);
  ASSERT(strstr(buf, "cycle_break") != NULL);
  ASSERT(strstr(buf, "replan") != NULL);
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
  RUN(load_docs_multiline_single_doc);
  RUN(load_docs_multiline_trailing_newlines);
  RUN(load_docs_multiline_max_docs_limit);
  RUN(load_docs_multiline_empty_file);

  printf("\n[Organelle Lifecycle]\n");
  RUN(organelle_free_null_safe);

  printf("\n[Valid-Move Filter]\n");
  RUN(valid_filter_match);
  RUN(valid_filter_no_match);
  RUN(valid_filter_empty_list);
  RUN(valid_filter_single_item);
  RUN(valid_filter_direction_strings);
  RUN(valid_filter_empty_action);
  RUN(valid_filter_no_partial_substring_match);
  RUN(valid_fallback_picks_first_unblocked);
  RUN(valid_fallback_all_blocked);
  RUN(valid_fallback_empty_csv);

  printf("\n[Ensemble Voting]\n");
  RUN(ensemble_constants_valid);

  printf("\n[Reasoning Trace]\n");
  RUN(trace_init_empty);
  RUN(trace_record_single);
  RUN(trace_record_sequence);
  RUN(trace_record_overflow);
  RUN(trace_finalise);
  RUN(trace_count_outcomes);
  RUN(trace_has_recovery);
  RUN(trace_no_false_recovery);
  RUN(trace_to_corpus_format);
  RUN(trace_to_corpus_empty);
  RUN(trace_write_file);
  RUN(trace_pipeline_simulation);

  /* Summary */
  printf("\n=== Results: %d/%d passed", g_tests_passed, g_tests_run);
  if (g_tests_failed > 0)
    printf(" (%d FAILED)", g_tests_failed);
  printf(" ===\n\n");

  return g_tests_failed > 0 ? 1 : 0;
}
