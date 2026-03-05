/*
 * SSD-Inspired Optimisation Benchmarks for MicroGPT-C
 *
 * Measures:
 *   1. ensemble_old:  N calls to organelle_generate() (original approach)
 *   2. ensemble_new:  prefix KV cache sharing (new approach)
 *   3. speculative:   draft+target speculative decoding
 *   4. baseline:      single organelle_generate() call
 *
 * Build with default compile-time config (same as test_microgpt_organelle).
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"
#include "microgpt_organelle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Timing helper ── */
static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Create a test organelle from a corpus string ── */
static Organelle *make_bench_organelle(const MicrogptConfig *cfg,
                                       const char *corpus) {
  Organelle *org = (Organelle *)calloc(1, sizeof(Organelle));
  if (!org)
    return NULL;

  Docs docs = {0};
  docs.data = (char *)malloc(strlen(corpus) + 1);
  strcpy(docs.data, corpus);
  docs.lines = (char **)malloc(sizeof(char *));
  docs.doc_lens = (size_t *)malloc(sizeof(size_t));
  docs.lines[0] = docs.data;
  docs.doc_lens[0] = strlen(corpus);
  docs.num_docs = 1;

  build_vocab(&docs, &org->vocab);
  org->model = model_create(org->vocab.vocab_size, cfg);
  org->docs = docs;
  return org;
}

/* ── OLD ensemble: calls organelle_generate N times (baseline) ── */
static void ensemble_old(const Organelle *org, const MicrogptConfig *cfg,
                         const char *prompt, char *output, int max_len,
                         int n_votes, scalar_t base_temp) {
  char candidates[7][128];
  int vote_counts[7];
  int unique = 0;

  for (int v = 0; v < n_votes; v++) {
    scalar_t jitter =
        OPA_TEMP_JITTER * ((scalar_t)v - (scalar_t)n_votes / (scalar_t)2.0);
    scalar_t temp = base_temp + jitter;
    if (temp < (scalar_t)0.01)
      temp = (scalar_t)0.01;

    char buf[128];
    int gen_len = max_len < 127 ? max_len : 127;
    organelle_generate(org, cfg, prompt, buf, gen_len, temp);

    int found = 0;
    for (int u = 0; u < unique; u++) {
      if (strcmp(candidates[u], buf) == 0) {
        vote_counts[u]++;
        found = 1;
        break;
      }
    }
    if (!found && unique < 7) {
      strncpy(candidates[unique], buf, 127);
      candidates[unique][127] = '\0';
      vote_counts[unique] = 1;
      unique++;
    }
  }

  int best_idx = 0;
  for (int u = 1; u < unique; u++) {
    if (vote_counts[u] > vote_counts[best_idx])
      best_idx = u;
  }
  strncpy(output, candidates[best_idx], (size_t)max_len);
  output[max_len] = '\0';
}

int main(void) {
  MicrogptConfig cfg = microgpt_default_config();
  seed_rng(42);

  printf("\n");
  printf("================================================================\n");
  printf("  SSD-Inspired Optimisation Benchmarks\n");
  printf("================================================================\n");
  microgpt_print_config("Benchmark Config", &cfg);

  /* Create test organelles with a more interesting corpus */
  const char *corpus = "STATE|board=XO_OX__X_\nACTION|move=5\n"
                       "STATE|board=XOOOXXXOX\nACTION|move=3\n"
                       "STATE|board=_________\nACTION|move=0\n";
  Organelle *org = make_bench_organelle(&cfg, corpus);
  Organelle *draft = make_bench_organelle(&cfg, corpus);
  if (!org || !draft) {
    fprintf(stderr, "ERROR: failed to create test organelles\n");
    return 1;
  }

  const char *prompt = "hel";
  int max_len = 5;
  int n_votes = 5;
  scalar_t temp = (scalar_t)0.5;
  int n_iters = 500;

  char output[128];

  printf("Benchmark parameters:\n");
  printf("  prompt:    \"%s\" (%zu chars)\n", prompt, strlen(prompt));
  printf("  max_len:   %d\n", max_len);
  printf("  n_votes:   %d\n", n_votes);
  printf("  temp:      %.2f\n", (double)temp);
  printf("  n_iters:   %d\n", n_iters);
  printf("\n");

  /* ── Benchmark 1: Single generate (baseline) ── */
  {
    double t0 = now_sec();
    for (int i = 0; i < n_iters; i++) {
      organelle_generate(org, &cfg, prompt, output, max_len, temp);
    }
    double elapsed = now_sec() - t0;
    double per_call_us = (elapsed / n_iters) * 1e6;
    printf("%-40s %8.1f µs/call  (%d iters, %.3fs total)\n",
           "1. Single generate (baseline):", per_call_us, n_iters, elapsed);
  }

  /* ── Benchmark 2: Old ensemble (N separate generate calls) ── */
  {
    double t0 = now_sec();
    for (int i = 0; i < n_iters; i++) {
      ensemble_old(org, &cfg, prompt, output, max_len, n_votes, temp);
    }
    double elapsed = now_sec() - t0;
    double per_call_us = (elapsed / n_iters) * 1e6;
    printf("%-40s %8.1f µs/call  (%d iters, %.3fs total)\n",
           "2. OLD ensemble (5 votes, no cache):", per_call_us, n_iters,
           elapsed);
  }

  /* ── Benchmark 3: New ensemble (prefix KV cache sharing) ── */
  {
    scalar_t conf;
    double t0 = now_sec();
    for (int i = 0; i < n_iters; i++) {
      organelle_generate_ensemble(org, &cfg, prompt, output, max_len, n_votes,
                                  temp, &conf);
    }
    double elapsed = now_sec() - t0;
    double per_call_us = (elapsed / n_iters) * 1e6;
    printf("%-40s %8.1f µs/call  (%d iters, %.3fs total)\n",
           "3. NEW ensemble (5 votes, prefix cache):", per_call_us, n_iters,
           elapsed);
  }

  /* ── Benchmark 4: Speculative decoding (spec_k=4) ── */
  {
    int total_accepted = 0, total_drafted = 0;
    double t0 = now_sec();
    for (int i = 0; i < n_iters; i++) {
      int acc = 0, dft = 0;
      organelle_generate_speculative(draft, org, &cfg, prompt, output, max_len,
                                     temp, 4, &acc, &dft);
      total_accepted += acc;
      total_drafted += dft;
    }
    double elapsed = now_sec() - t0;
    double per_call_us = (elapsed / n_iters) * 1e6;
    double accept_rate =
        total_drafted > 0 ? 100.0 * total_accepted / total_drafted : 0;
    printf("%-40s %8.1f µs/call  (%d iters, %.3fs total)\n",
           "4. Speculative decode (k=4):", per_call_us, n_iters, elapsed);
    printf("   Acceptance rate: %.1f%% (%d/%d tokens)\n", accept_rate,
           total_accepted, total_drafted);
  }

  /* ── Benchmark 5: Speculative decoding (spec_k=2) ── */
  {
    int total_accepted = 0, total_drafted = 0;
    double t0 = now_sec();
    for (int i = 0; i < n_iters; i++) {
      int acc = 0, dft = 0;
      organelle_generate_speculative(draft, org, &cfg, prompt, output, max_len,
                                     temp, 2, &acc, &dft);
      total_accepted += acc;
      total_drafted += dft;
    }
    double elapsed = now_sec() - t0;
    double per_call_us = (elapsed / n_iters) * 1e6;
    double accept_rate =
        total_drafted > 0 ? 100.0 * total_accepted / total_drafted : 0;
    printf("%-40s %8.1f µs/call  (%d iters, %.3fs total)\n",
           "5. Speculative decode (k=2):", per_call_us, n_iters, elapsed);
    printf("   Acceptance rate: %.1f%% (%d/%d tokens)\n", accept_rate,
           total_accepted, total_drafted);
  }

  /* ── Summary ── */
  printf("\n");

  organelle_free(org);
  organelle_free(draft);

  printf("================================================================\n");
  printf("  Benchmark complete\n");
  printf(
      "================================================================\n\n");

  return 0;
}
