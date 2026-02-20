/*
 * MicroGPT-C — C Code Generation Example (c_codegen)
 * Copyright (c) 2026 Ajay Soni, Enjector Software Ltd. MIT License.
 *
 * Trains a character-level GPT on a corpus of numerical recipes-style C
 * functions with descriptive comment headers, then generates new C code
 * conditioned on natural language prompts.
 *
 * Each function in the training file is separated by a blank line. The model
 * learns the mapping: comment -> function body. At inference time,
 * you provide a prompt like "FFT with windowing" and the model generates
 * the corresponding C code.
 *
 * Build:
 *   cmake --build build --target c_codegen
 *   ./build/c_codegen
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "microgpt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MICROGPT_METAL
#include "microgpt_metal.h"
#endif

#define CODEGEN_TEMP 0.3
#define GEN_LEN 400 /* characters to generate per sample */
#define CHECKPOINT_FILE "c_codegen.ckpt"
#define TRAINING_LOG "c_codegen.ckpt.log"

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

/* ---- Load multi-line documents separated by blank lines ---- */
static int load_docs_multiline(const char *path, Docs *docs, int max_docs) {
  FILE *f = fopen(path, "r");
  if (!f)
    return -1;

  /* Read entire file into docs->data */
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (fsize <= 0 || fsize > 50 * 1024 * 1024) {
    fclose(f);
    return -1;
  }

  docs->data = (char *)malloc((size_t)fsize + 1);
  if (!docs->data) {
    fclose(f);
    return -1;
  }
  size_t rd = fread(docs->data, 1, (size_t)fsize, f);
  fclose(f);
  docs->data[rd] = '\0';

  /* Allocate index arrays */
  docs->lines = (char **)malloc((size_t)max_docs * sizeof(char *));
  docs->doc_lens = (size_t *)malloc((size_t)max_docs * sizeof(size_t));
  if (!docs->lines || !docs->doc_lens) {
    free(docs->data);
    return -1;
  }

  /* Scan buffer: split on blank lines (two consecutive newlines) */
  docs->num_docs = 0;
  char *p = docs->data;
  while (*p && docs->num_docs < (size_t)max_docs) {
    /* Skip blank lines between documents */
    while (*p == '\n' || *p == '\r')
      p++;
    if (!*p)
      break;

    /* Mark start of this document block */
    char *start = p;

    /* Advance to the next blank line (or EOF) */
    while (*p) {
      if (*p == '\n') {
        /* Check if next char is also a newline (blank line separator) */
        char *next = p + 1;
        if (*next == '\r')
          next++;
        if (*next == '\n' || *next == '\0')
          break;
      }
      p++;
    }

    size_t len = (size_t)(p - start);
    /* Trim trailing newlines from this block */
    while (len > 0 && (start[len - 1] == '\n' || start[len - 1] == '\r'))
      len--;

    if (len > 0) {
      docs->lines[docs->num_docs] = start;
      docs->doc_lens[docs->num_docs] = len;
      docs->num_docs++;
    }

    /* Skip past the blank line separator */
    if (*p)
      p++;
  }
  return 0;
}

int main(void) {
  unsigned int train_seed = 42;
  unsigned int infer_seed = (unsigned int)time(NULL);
  srand(train_seed);
  seed_rng(train_seed);

  /* Runtime configuration — C codegen-specific overrides */
  MicrogptConfig cfg = microgpt_default_config();
  cfg.n_embd = 128;
  cfg.n_head = 4;
  cfg.mlp_dim = 512;
  cfg.n_layer = 4;
  cfg.block_size = 512;
  cfg.batch_size = 16;
  cfg.num_steps = 50000;
  cfg.learning_rate = 0.0003;
  cfg.max_vocab = 200;
  cfg.max_docs = 5000;
  cfg.max_doc_len = 1024;
  microgpt_print_config("MicroGPT-C - C Code Generation Demo", &cfg);

  const int nl = cfg.n_layer;

#ifdef MICROGPT_METAL
  if (metal_init() != 0) {
    fprintf(stderr, "WARNING: Metal GPU init failed, falling back to CPU\n");
  }
#endif

  /* ---- Load C functions corpus (blank-line separated) ---- */
  Docs docs = {0};
  if (load_docs_multiline("c_functions.txt", &docs, cfg.max_docs) != 0) {
    fprintf(stderr, "Cannot open c_functions.txt\n");
    return 1;
  }
  shuffle_docs(&docs);
  printf("loaded %zu C functions (multi-line documents)\n", docs.num_docs);

  size_t total_chars = 0;
  for (size_t i = 0; i < docs.num_docs; i++)
    total_chars += docs.doc_lens[i];
  printf("total characters: %zu (%.1f KB)\n", total_chars,
         (scalar_t)total_chars / 1024.0);

  /* Open training log (append mode — accumulates across runs) */
  FILE *logf = fopen(TRAINING_LOG, "a");
  if (logf) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    fprintf(logf, "\n========================================\n");
    fprintf(logf, "Run: %04d-%02d-%02d %02d:%02d:%02d\n", t->tm_year + 1900,
            t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    fprintf(logf, "========================================\n");
    fprintf(logf, "Corpus: %zu functions | %zu chars (%.1f KB)\n",
            docs.num_docs, total_chars, (scalar_t)total_chars / 1024.0);
  }

  /* Build character vocabulary */
  Vocab vocab = {0};
  build_vocab(&docs, &vocab);
  printf("vocab: %zu characters\n", vocab.vocab_size);
  if (logf)
    fprintf(logf, "Vocab: %zu characters\n", vocab.vocab_size);

  if (vocab.vocab_size > (size_t)cfg.max_vocab) {
    fprintf(stderr, "vocab_size %zu exceeds max_vocab %d\n", vocab.vocab_size,
            cfg.max_vocab);
    free_docs(&docs);
    free(vocab.chars);
    return 1;
  }

  printf("N_EMBD=%d BLOCK_SIZE=%d N_LAYER=%d N_HEAD=%d\n\n", cfg.n_embd,
         cfg.block_size, cfg.n_layer, cfg.n_head);

  /* ---- Create or load model ---- */
  size_t nparams;
  scalar_t *grad_buffer, *m_buf, *v_buf;
  Model *model = NULL;
  int trained = 0;

  {
    Model *tmp = model_create(vocab.vocab_size, &cfg);
    if (!tmp) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
    nparams = model_num_params(tmp);
    model_free(tmp);
  }

  m_buf = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  v_buf = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  grad_buffer = (scalar_t *)calloc(nparams, sizeof(scalar_t));
  if (!grad_buffer || !m_buf || !v_buf) {
    fprintf(stderr, "OOM (optimiser)\n");
    return 1;
  }

  /* Try loading checkpoint */
  int resume_step = 0;
  model = checkpoint_load(CHECKPOINT_FILE, vocab.vocab_size, &cfg, m_buf, v_buf,
                          &resume_step);
  if (model) {
    printf("loaded checkpoint '%s' (trained %d steps) — skipping training\n\n",
           CHECKPOINT_FILE, resume_step);
    trained = 1;
  } else {
    model = model_create(vocab.vocab_size, &cfg);
    if (!model) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
  }

  int nthreads = mgpt_default_threads(cfg.batch_size);
#ifdef MICROGPT_METAL
  if (metal_available()) {
    nthreads = 1; /* GPU handles parallelism — avoid command queue contention */
    printf("[Metal] Single-threaded mode (GPU handles compute)\n");
  }
#endif
  printf("params: %zu | batch %d | steps %d | lr %.4f | threads %d\n\n",
         nparams, cfg.batch_size, cfg.num_steps, (scalar_t)cfg.learning_rate,
         nthreads);
  if (logf) {
    fprintf(logf,
            "Architecture: N_EMBD=%d N_LAYER=%d N_HEAD=%d BLOCK_SIZE=%d\n",
            cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.block_size);
    fprintf(logf, "Params: %zu\n", nparams);
    fprintf(logf, "Training: batch=%d steps=%d lr=%.4f threads=%d\n",
            cfg.batch_size, cfg.num_steps, (scalar_t)cfg.learning_rate,
            nthreads);
    fprintf(logf, "Seeds: train=%u infer=%u\n", train_seed, infer_seed);
    fprintf(logf, "Inference: temp=%.2f gen_len=%d\n", (scalar_t)CODEGEN_TEMP,
            GEN_LEN);
    fflush(logf);
  }

  /* ---- Allocate per-thread resources ---- */
  TrainWorker *workers =
      (TrainWorker *)calloc((size_t)nthreads, sizeof(TrainWorker));
  mgpt_thread_t *threads =
      (mgpt_thread_t *)calloc((size_t)nthreads, sizeof(mgpt_thread_t));
  mgpt_thread_trampoline_t *tramps = (mgpt_thread_trampoline_t *)calloc(
      (size_t)nthreads, sizeof(mgpt_thread_trampoline_t));
  if (!workers || !threads || !tramps) {
    fprintf(stderr, "OOM\n");
    return 1;
  }

  for (int t = 0; t < nthreads; t++) {
    workers[t].model = model;
    workers[t].docs = &docs;
    workers[t].vocab = &vocab;
    workers[t].grads = (scalar_t *)calloc(nparams, sizeof(scalar_t));
    workers[t].keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    workers[t].values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
    workers[t].cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
    workers[t].token_buf =
        (size_t *)malloc(((size_t)cfg.block_size + 2) * sizeof(size_t));
    if (!workers[t].grads || !workers[t].keys || !workers[t].values ||
        !workers[t].cache_len || !workers[t].token_buf) {
      fprintf(stderr, "OOM\n");
      return 1;
    }
    for (int L = 0; L < nl; L++) {
      workers[t].keys[L] = kv_cache_alloc(&cfg);
      workers[t].values[L] = kv_cache_alloc(&cfg);
      if (!workers[t].keys[L] || !workers[t].values[L]) {
        fprintf(stderr, "OOM (thread KV cache)\n");
        return 1;
      }
    }
  }

  /* KV cache for inference */
  scalar_t **inf_keys = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  scalar_t **inf_values = (scalar_t **)malloc((size_t)nl * sizeof(scalar_t *));
  size_t *inf_cache_len = (size_t *)calloc((size_t)nl, sizeof(size_t));
  for (int L = 0; L < nl; L++) {
    inf_keys[L] = kv_cache_alloc(&cfg);
    inf_values[L] = kv_cache_alloc(&cfg);
  }

  /* ---- Training ---- */
  if (!trained) {
    if (logf)
      fprintf(logf, "\n--- Training ---\n");
    size_t tokens_trained = 0;
    time_t t0 = time(NULL);
    scalar_t best_loss = 1e9; /* track best loss for checkpoint saving */
    char best_ckpt_name[256];

    for (int step = 0; step < cfg.num_steps; step++) {
      int batches_per_thread = cfg.batch_size / nthreads;
      int remainder = cfg.batch_size % nthreads;
      int cursor = 0;

      for (int t = 0; t < nthreads; t++) {
        workers[t].batch_start = cursor;
        int count = batches_per_thread + (t < remainder ? 1 : 0);
        workers[t].batch_end = cursor + count;
        cursor += count;
        workers[t].rng_seed = (unsigned int)(step * nthreads + t + 1);
      }

      for (int t = 0; t < nthreads; t++)
        mgpt_thread_create(&threads[t], &tramps[t], train_worker_run,
                           &workers[t]);
      for (int t = 0; t < nthreads; t++)
        mgpt_thread_join(threads[t]);

      scalar_t batch_loss = 0;
      size_t batch_positions = 0;
      memset(grad_buffer, 0, nparams * sizeof(scalar_t));

      for (int t = 0; t < nthreads; t++) {
        batch_loss += workers[t].loss;
        batch_positions += workers[t].positions;
        const scalar_t *tg = workers[t].grads;
        for (size_t i = 0; i < nparams; i++)
          grad_buffer[i] += tg[i];
      }
      tokens_trained += batch_positions;

      scalar_t mean_loss = batch_loss / (scalar_t)batch_positions;
      for (size_t i = 0; i < nparams; i++)
        grad_buffer[i] /= (scalar_t)batch_positions;
      adam_step(model, grad_buffer, m_buf, v_buf, step);

      if ((step + 1) % 500 == 0 || step == 0) {
        scalar_t elapsed = difftime(time(NULL), t0);
        if (elapsed < 1.0)
          elapsed = 1.0;
        scalar_t steps_per_sec = (step + 1) / elapsed;
        scalar_t eta_sec = (cfg.num_steps - step - 1) / steps_per_sec;
        int el_m = (int)elapsed / 60, el_s = (int)elapsed % 60;
        int eta_m = (int)eta_sec / 60, eta_s = (int)eta_sec % 60;
        printf("step %5d / %d | loss %.4f | %dm%02ds elapsed, ETA %dm%02ds",
               step + 1, cfg.num_steps, mean_loss, el_m, el_s, eta_m, eta_s);

        /* Save best checkpoint when loss improves */
        if (mean_loss < best_loss && step > 0) {
          best_loss = mean_loss;
          snprintf(best_ckpt_name, sizeof(best_ckpt_name),
                   "c_codegen_best_step%d_loss%.4f.ckpt", step + 1, mean_loss);
          if (checkpoint_save(model, m_buf, v_buf, step + 1, best_ckpt_name) ==
              0)
            printf(" [BEST -> %s]", best_ckpt_name);
          /* Also save as the standard checkpoint for inference */
          checkpoint_save(model, m_buf, v_buf, step + 1, CHECKPOINT_FILE);
        }
        printf("\n");

        /* Log step to training log */
        if (logf) {
          fprintf(logf, "step %5d / %d | loss %.4f", step + 1, cfg.num_steps,
                  mean_loss);
          if (mean_loss < best_loss || (mean_loss == best_loss && step > 0))
            fprintf(logf, " [BEST]");
          fprintf(logf, "\n");
          fflush(logf);
        }
      }
    }

    scalar_t train_sec = difftime(time(NULL), t0);
    if (train_sec < 1.0)
      train_sec = 1.0;
    printf("\nTraining: %.1fs | %.0f steps/s | %.1fk tok/s\n", train_sec,
           (scalar_t)cfg.num_steps / train_sec,
           (scalar_t)tokens_trained / train_sec / 1000.0);

    if (checkpoint_save(model, m_buf, v_buf, cfg.num_steps, CHECKPOINT_FILE) ==
        0)
      printf("final checkpoint saved to '%s'\n", CHECKPOINT_FILE);
    else
      fprintf(stderr, "warning: failed to save checkpoint\n");
    printf("best training loss: %.4f (saved in %s)\n", best_loss,
           best_ckpt_name);

    if (logf) {
      fprintf(logf, "\nTraining complete: %.1fs | %.0f steps/s | %.1fk tok/s\n",
              train_sec, (scalar_t)cfg.num_steps / train_sec,
              (scalar_t)tokens_trained / train_sec / 1000.0);
      fprintf(logf, "Best loss: %.4f (%s)\n", best_loss, best_ckpt_name);
      fprintf(logf, "Final checkpoint: %s\n", CHECKPOINT_FILE);
      fflush(logf);
    }
  } else {
    if (logf)
      fprintf(logf, "Loaded checkpoint: %s (trained %d steps)\n",
              CHECKPOINT_FILE, resume_step);
  }

/* ---- Confidence scoring helpers ---- */
/* Compute softmax probability for a specific token given raw logits */
#define CONFIDENCE_HIGH 0.60   /* above = confident, show result */
#define CONFIDENCE_MEDIUM 0.30 /* above = partial confidence */
#define CONFIDENCE_LOW 0.15    /* below = refuse to generate */

  /* ---- Generate C code from prompts ---- */
  seed_rng(infer_seed);
  printf("\n--- generated C functions (with confidence) ---\n");
  scalar_t *logits_buf =
      (scalar_t *)malloc((size_t)cfg.max_vocab * sizeof(scalar_t));

  /* Inference timing accumulators */
  int total_inf_tokens = 0;
  double total_inf_time = 0.0;

  /* Prompts to test — these are fed as context, then the model generates */
  const char *prompts[] = {
      /* ---- NOVEL COMPOSITIONS (0 corpus matches, verified) ---- */
      /* Simple array ops — model knows loops+arrays but not these concepts */
      "/* create array of square numbers */",
      "/* reverse array in place */",
      "/* create array of cube numbers */",
      "/* fill array with powers of two */",
      "/* element-wise multiply two arrays */",
      "/* absolute difference of two arrays */",
      /* Math compositions — model knows arithmetic but not these combos */
      "/* sum of squared differences */",
      "/* root mean square value */",
      "/* weighted median of array */",
      "/* compute array range (max minus min) */",
      /* ---- CONTROLS — exact corpus matches (should work) ---- */
      "/* compute average of array */",
      "/* FFT with Hamming window */",
      "/* Black-Scholes call option price */",
      "/* compute MACD histogram */",
      "/* bubble sort ascending */",
  };
  int num_prompts = (int)(sizeof(prompts) / sizeof(prompts[0]));

  for (int s = 0; s < num_prompts; s++) {
    for (int L = 0; L < nl; L++)
      inf_cache_len[L] = 0;

    const char *prompt = prompts[s];
    size_t prompt_len = strlen(prompt);

    /* Feed prompt character-by-character to build context,
       tracking prompt recognition confidence */
    size_t token = vocab.bos_id;
    size_t pos = 0;
    scalar_t prompt_conf_sum = 0.0;
    int prompt_conf_count = 0;

    /* Start inference timer */
    struct timespec inf_start, inf_end;
    clock_gettime(CLOCK_MONOTONIC, &inf_start);

    /* Process initial BOS */
    forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                      logits_buf);
    pos++;

    /* Feed each prompt character and measure prediction confidence */
    for (size_t pi = 0;
         pi < prompt_len && pos < (size_t)(cfg.block_size - GEN_LEN); pi++) {
      /* Find token ID for this character */
      size_t next_token = vocab.bos_id; /* fallback */
      for (size_t c = 0; c < vocab.vocab_size; c++) {
        if (vocab.chars[c] == (unsigned char)prompt[pi]) {
          next_token = c;
          break;
        }
      }

      /* Compute softmax probability for this character from current logits */
      {
        scalar_t max_val = logits_buf[0];
        for (size_t c = 1; c < vocab.vocab_size; c++)
          if (logits_buf[c] > max_val)
            max_val = logits_buf[c];
        scalar_t sum = 0;
        for (size_t c = 0; c < vocab.vocab_size; c++)
          sum += exp(logits_buf[c] - max_val);
        scalar_t prob = exp(logits_buf[next_token] - max_val) / sum;
        prompt_conf_sum += prob;
        prompt_conf_count++;
      }

      token = next_token;
      forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                        logits_buf);
      pos++;
    }

    /* Feed a newline after the comment to start the function */
    for (size_t c = 0; c < vocab.vocab_size; c++) {
      if (vocab.chars[c] == '\n') {
        token = c;
        break;
      }
    }
    forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                      logits_buf);
    pos++;

    /* Compute prompt recognition confidence */
    scalar_t prompt_confidence =
        prompt_conf_count > 0 ? prompt_conf_sum / prompt_conf_count : 0.0;

    /* Display prompt with confidence indicator */
    const char *conf_icon = prompt_confidence >= CONFIDENCE_HIGH     ? "HIGH"
                            : prompt_confidence >= CONFIDENCE_MEDIUM ? "MEDIUM"
                            : prompt_confidence >= CONFIDENCE_LOW    ? "LOW"
                                                                     : "NONE";
    const char *conf_sym = prompt_confidence >= CONFIDENCE_HIGH     ? "==="
                           : prompt_confidence >= CONFIDENCE_MEDIUM ? "==-"
                           : prompt_confidence >= CONFIDENCE_LOW    ? "=--"
                                                                    : "---";

    printf("\n[%s] confidence: %.0f%% (%s)  %s\n", conf_sym,
           prompt_confidence * 100.0, conf_icon, prompt);

    /* If confidence is too low, warn and skip or generate anyway */
    if (prompt_confidence < CONFIDENCE_LOW) {
      printf("  >> prompt not recognized — skipping generation\n");
      continue;
    }

    /* Generate code, tracking per-character confidence */
    int gen_count = GEN_LEN;
    if ((int)pos + gen_count > cfg.block_size)
      gen_count = cfg.block_size - (int)pos;

    scalar_t gen_conf_sum = 0.0;
    int gen_conf_count = 0;
    char *gen_buf = (char *)malloc((size_t)GEN_LEN + 1);
    int gen_len = 0;

    for (int g = 0; g < gen_count; g++) {
      /* Compute softmax to get probability of sampled token */
      scalar_t max_val = logits_buf[0];
      for (size_t c = 1; c < vocab.vocab_size; c++)
        if (logits_buf[c] > max_val)
          max_val = logits_buf[c];
      scalar_t sum = 0;
      for (size_t c = 0; c < vocab.vocab_size; c++)
        sum += exp(logits_buf[c] - max_val);

      token = sample_token(logits_buf, vocab.vocab_size, CODEGEN_TEMP);
      if (token == vocab.bos_id)
        break;

      /* Track confidence for this generated character */
      scalar_t prob = exp(logits_buf[token] - max_val) / sum;
      gen_conf_sum += prob;
      gen_conf_count++;

      gen_buf[gen_len++] = (char)vocab.chars[token];

      forward_inference(model, token, pos, inf_keys, inf_values, inf_cache_len,
                        logits_buf);
      pos++;
    }
    gen_buf[gen_len] = '\0';

    /* Compute generation confidence */
    scalar_t gen_confidence =
        gen_conf_count > 0 ? gen_conf_sum / gen_conf_count : 0.0;

    /* Stop inference timer */
    clock_gettime(CLOCK_MONOTONIC, &inf_end);
    double inf_elapsed = (inf_end.tv_sec - inf_start.tv_sec) +
                         (inf_end.tv_nsec - inf_start.tv_nsec) / 1e9;
    int prompt_tokens = (int)prompt_len + 2; /* +BOS +newline */
    int total_tokens = prompt_tokens + gen_len;
    double tok_per_sec = inf_elapsed > 0 ? total_tokens / inf_elapsed : 0;
    total_inf_tokens += total_tokens;
    total_inf_time += inf_elapsed;

    /* Print generated code */
    printf("%s\n", gen_buf);
    printf(
        "  >> generation confidence: %.0f%% | %d tok in %.3fs (%.0f tok/s)\n",
        gen_confidence * 100.0, total_tokens, inf_elapsed, tok_per_sec);
    free(gen_buf);

    /* Log inference result */
    if (logf) {
      fprintf(logf, "  prompt: %s\n", prompt);
      fprintf(logf,
              "  confidence: %.0f%% | gen_conf: %.0f%% | %d tok | %.0f tok/s\n",
              prompt_confidence * 100.0, gen_confidence * 100.0, total_tokens,
              tok_per_sec);
    }
  }

  /* ---- Inference summary ---- */
  printf("\n--- inference summary ---\n");
  printf(
      "prompts: %d | total tokens: %d | total time: %.3fs | avg: %.0f tok/s\n",
      num_prompts, total_inf_tokens, total_inf_time,
      total_inf_time > 0 ? total_inf_tokens / total_inf_time : 0);

  if (logf) {
    fprintf(logf, "\n--- Inference Summary ---\n");
    fprintf(logf,
            "prompts: %d | total_tokens: %d | time: %.3fs | avg: %.0f tok/s\n",
            num_prompts, total_inf_tokens, total_inf_time,
            total_inf_time > 0 ? total_inf_tokens / total_inf_time : 0);
    fprintf(logf, "========================================\n\n");
    fclose(logf);
    logf = NULL;
    printf("training log appended to '%s'\n", TRAINING_LOG);
  }

  /* ---- Cleanup ---- */
  for (int t = 0; t < nthreads; t++) {
    for (int L = 0; L < nl; L++) {
      kv_cache_free(workers[t].keys[L]);
      kv_cache_free(workers[t].values[L]);
    }
    free(workers[t].keys);
    free(workers[t].values);
    free(workers[t].cache_len);
    free(workers[t].token_buf);
    free(workers[t].grads);
  }
  free(workers);
  free(threads);
  free(tramps);
  for (int L = 0; L < nl; L++) {
    kv_cache_free(inf_keys[L]);
    kv_cache_free(inf_values[L]);
  }
  free(inf_keys);
  free(inf_values);
  free(inf_cache_len);
  free(logits_buf);
  free(grad_buffer);
  free(m_buf);
  free(v_buf);
  model_free(model);
  free_docs(&docs);
  free(vocab.chars);
  return 0;
}
