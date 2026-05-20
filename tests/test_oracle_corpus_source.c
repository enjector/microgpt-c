/*
 * tests/test_oracle_corpus_source.c — Experiment E15 unit tests for the
 * oracle corpus-source adapter.
 *
 * Exercises the JSON-line parser, bigram-Jaccard, and cache-path
 * derivation without invoking the underlying solver binary (which is
 * tested separately via the --self-test mode of puzzle15_a_star and
 * klotski_a_star).
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#include "../tools/oracle_corpus_source.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENX_ABORT_ON_FAIL 1

/* ── Mini test harness (no test.h dep — keeps the binary self-contained) ── */

static int g_passed = 0;
static int g_failed = 0;
#define T(expr) do { if (expr) g_passed++; else { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); g_failed++; \
} } while (0)

static void test_parse_jsonl_minimal(void) {
    char buf[] =
        "{\"state\":\"abc\",\"solution\":\"UD\",\"moves\":2}\n"
        "{\"state\":\"xyz\",\"solution\":\"LR\",\"moves\":1}\n";
    OraclePair *pairs = NULL;
    size_t n = 0;
    int rc = oracle_parse_jsonl(buf, sizeof(buf) - 1, &pairs, &n);
    T(rc == 0);
    T(n == 2);
    T(pairs);
    if (n >= 2) {
        T(strcmp(pairs[0].state, "abc") == 0);
        T(strcmp(pairs[0].solution, "UD") == 0);
        T(pairs[0].moves == 2);
        T(strcmp(pairs[1].state, "xyz") == 0);
        T(strcmp(pairs[1].solution, "LR") == 0);
        T(pairs[1].moves == 1);
    }
    free(pairs);
}

static void test_parse_jsonl_with_extra_fields(void) {
    /* puzzle15 emits "md", klotski emits "scramble_depth" — both should
     * be tolerated; we only need state and solution. */
    char buf[] =
        "{\"state\":\"127356409eb8dfac\",\"solution\":\"LURDDL\","
        "\"moves\":6,\"md\":10}\n"
        "{\"state\":\"BDGE.B.F.CAA..CAA...\",\"solution\":\"AR\","
        "\"moves\":1,\"scramble_depth\":10}\n";
    OraclePair *pairs = NULL;
    size_t n = 0;
    int rc = oracle_parse_jsonl(buf, sizeof(buf) - 1, &pairs, &n);
    T(rc == 0);
    T(n == 2);
    if (n >= 2) {
        T(strcmp(pairs[0].state, "127356409eb8dfac") == 0);
        T(strcmp(pairs[0].solution, "LURDDL") == 0);
        T(strcmp(pairs[1].state, "BDGE.B.F.CAA..CAA...") == 0);
        T(strcmp(pairs[1].solution, "AR") == 0);
    }
    free(pairs);
}

static void test_parse_jsonl_skips_malformed_lines(void) {
    /* A line without a "{" is silently skipped; a line missing
     * "solution" is also skipped (state alone isn't a usable pair). */
    char buf[] =
        "# this is a comment\n"
        "{\"state\":\"abc\",\"solution\":\"UD\"}\n"
        "{\"state\":\"orphan\"}\n"
        "{\"state\":\"def\",\"solution\":\"LR\"}\n"
        "\n";
    OraclePair *pairs = NULL;
    size_t n = 0;
    int rc = oracle_parse_jsonl(buf, sizeof(buf) - 1, &pairs, &n);
    T(rc == 0);
    T(n == 2);  /* the two valid lines */
    if (n >= 2) {
        T(strcmp(pairs[0].state, "abc") == 0);
        T(strcmp(pairs[1].state, "def") == 0);
    }
    free(pairs);
}

static void test_jaccard_identical_states_is_one(void) {
    double j = oracle_jaccard_state("abc123def456", "abc123def456");
    T(j > 0.999);
}

static void test_jaccard_disjoint_alphabets_low(void) {
    double j = oracle_jaccard_state("aaaa", "bbbb");
    /* Both have one bigram each ("aa" and "bb"); no overlap. */
    T(j < 0.1);
}

static void test_jaccard_partial_overlap(void) {
    double j = oracle_jaccard_state("abcd", "abef");
    /* a:{ab,bc,cd}, b:{ab,be,ef}; intersection={ab}; union=5; j=0.2. */
    T(j > 0.15 && j < 0.25);
}

static void test_cache_path_deterministic(void) {
    OracleSource a = {
        .oracle_binary = "build/puzzle15_a_star",
        .cache_dir = "/tmp/__e15_test_cache",
        .seed = 1337,
        .count = 100,
        .difficulty = "mixed",
        .verbose = 0
    };
    OracleSource b = a;
    char *pa = oracle_cache_path(&a);
    char *pb = oracle_cache_path(&b);
    T(pa && pb);
    if (pa && pb) T(strcmp(pa, pb) == 0);
    /* Changing seed must change the cache file. */
    b.seed = 4242;
    char *pc = oracle_cache_path(&b);
    T(pc);
    if (pa && pc) T(strcmp(pa, pc) != 0);
    free(pa); free(pb); free(pc);
}

int main(void) {
    test_parse_jsonl_minimal();
    test_parse_jsonl_with_extra_fields();
    test_parse_jsonl_skips_malformed_lines();
    test_jaccard_identical_states_is_one();
    test_jaccard_disjoint_alphabets_low();
    test_jaccard_partial_overlap();
    test_cache_path_deterministic();
    fprintf(stderr, "oracle_corpus_source tests: %d passed, %d failed\n",
            g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
