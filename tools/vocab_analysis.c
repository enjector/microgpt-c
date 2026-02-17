/*
 * vocab_analysis.c — Measure <unk> ratio at different max_words settings
 *
 * Build manually:
 *   cc -O2 -DN_EMBD=64 -DN_HEAD=4 -DMLP_DIM=256 -DN_LAYER=2 -DBLOCK_SIZE=32 \
 *      -DBATCH_SIZE=1 -DNUM_STEPS=1 -DLEARNING_RATE=0.001 -DMAX_VOCAB=50000 \
 *      -DMAX_DOCS=1 -DMAX_DOC_LEN=128 \
 *      -I src tools/vocab_analysis.c src/microgpt.c -lm -o build/vocab_analysis
 */

#define _CRT_SECURE_NO_WARNINGS 1
#include "microgpt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  size_t text_len;
  char *text = load_file("shakespeare.txt", &text_len);
  if (!text) {
    fprintf(stderr, "Cannot open shakespeare.txt (run from build dir)\n");
    return 1;
  }
  printf("Shakespeare: %.1f KB, %zu bytes\n\n", (double)text_len / 1024.0,
         text_len);

  /* Test different vocabulary sizes */
  size_t vocab_sizes[] = {500,  1000,  2000,  3000,  5000,
                          7500, 10000, 15000, 20000, 30000};
  int n_tests = 10;

  printf("%-12s %-12s %-12s %-12s %-12s %-12s\n", "max_words", "vocab_size",
         "total_tok", "unk_count", "unk_pct", "model_params");
  printf("%-12s %-12s %-12s %-12s %-12s %-12s\n", "----------", "----------",
         "---------", "---------", "-------", "------------");

  size_t *tokens = (size_t *)malloc(text_len * sizeof(size_t));

  for (int t = 0; t < n_tests; t++) {
    size_t max_w = vocab_sizes[t];
    WordVocab wv;
    memset(&wv, 0, sizeof(wv));
    if (build_word_vocab(text, text_len, max_w, &wv) != 0) {
      printf("build_word_vocab failed for max_words=%zu\n", max_w);
      continue;
    }

    size_t total = tokenize_words(text, text_len, &wv, tokens, text_len);
    size_t unk_count = 0;
    for (size_t i = 0; i < total; i++) {
      if (tokens[i] == wv.unk_id)
        unk_count++;
    }
    double unk_pct = 100.0 * (double)unk_count / (double)total;

    /* Calculate model params */
    size_t np = wv.vocab_size * N_EMBD        /* wte */
                + (size_t)BLOCK_SIZE * N_EMBD /* wpe */
                + wv.vocab_size * N_EMBD      /* lm_head */
                +
                (size_t)N_LAYER * ((size_t)N_EMBD * N_EMBD * 4 /* wq+wk+wv+wo */
                                   + (size_t)MLP_DIM * N_EMBD  /* fc1 */
                                   + (size_t)N_EMBD * MLP_DIM  /* fc2 */
                                  );

    printf("%-12zu %-12zu %-12zu %-12zu %-11.1f%% %-12zu\n", max_w,
           wv.vocab_size, total, unk_count, unk_pct, np);

    free_word_vocab(&wv);
  }

  free(tokens);
  free(text);
  return 0;
}
