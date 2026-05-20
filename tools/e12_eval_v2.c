/*
 * tools/e12_eval_v2.c — Experiment E12 Phase 5 evaluator.
 *
 * Loads a wiring-dim checkpoint trained on the LLM-generated corpus via
 * the OQL TRAIN dispatch (which uses CHAR-level Vocab + load_docs +
 * tokenize), rebuilds the same Vocab from the training corpus, runs
 * auto-regressive sampling on each prompt in
 * pipeline_corpus_scaling_heldout_v2.txt, parses + verifies via
 * libpipeline_ir, and reports the verify rate.
 *
 * Headline metric = T4 = verified / 20.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1

#include "microgpt.h"
#include "pipeline_ir/pipeline_ir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_OUT 4096
#define MAX_PROMPTS 256

static int load_heldout_prompts(const char *path,
                                char prompts[MAX_PROMPTS][512],
                                char references[MAX_PROMPTS][64]) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return 0; }
    char *body = (char *)malloc((size_t)n + 1);
    if (!body) { fclose(f); return 0; }
    fread(body, 1, (size_t)n, f);
    body[n] = '\0';
    fclose(f);

    int count = 0;
    char *p = body;
    char cur_ref[64] = "";
    while (p && *p && count < MAX_PROMPTS) {
        char *eol = strchr(p, '\n');
        size_t llen = eol ? (size_t)(eol - p) : strlen(p);
        if (llen > 14 && strncmp(p, "# REFERENCE: ", 13) == 0) {
            size_t copy = llen - 13;
            if (copy >= sizeof(cur_ref)) copy = sizeof(cur_ref) - 1;
            memcpy(cur_ref, p + 13, copy);
            cur_ref[copy] = '\0';
        } else if (llen > 3 && p[0] == '/' && p[1] == '/' && p[2] == ' ') {
            size_t copy = llen - 3;
            if (copy >= 511) copy = 511;
            memcpy(prompts[count], p + 3, copy);
            prompts[count][copy] = '\0';
            strncpy(references[count], cur_ref, 63);
            references[count][63] = '\0';
            count++;
        }
        if (!eol) break;
        p = eol + 1;
    }
    free(body);
    return count;
}

static void generate_char(const Model *model, const MicrogptConfig *cfg,
                          const Vocab *vocab,
                          const char *prompt, char *output, size_t max_out,
                          scalar_t temperature, int max_chars) {
    int nl_layers = cfg->n_layer;
    scalar_t **keys   = (scalar_t **)malloc((size_t)nl_layers * sizeof(scalar_t *));
    scalar_t **values = (scalar_t **)malloc((size_t)nl_layers * sizeof(scalar_t *));
    size_t   *cache_len = (size_t   *)calloc((size_t)nl_layers, sizeof(size_t));
    for (int l = 0; l < nl_layers; l++) {
        keys[l]   = kv_cache_alloc(cfg);
        values[l] = kv_cache_alloc(cfg);
    }
    scalar_t *logits = (scalar_t *)malloc(vocab->vocab_size * sizeof(scalar_t));
    int pos = 0;
    int out_pos = 0;
    output[0] = '\0';

    forward_inference(model, vocab->bos_id, pos, keys, values, cache_len, logits);
    pos++;

    size_t plen = strlen(prompt);
    size_t *prompt_ids = (size_t *)malloc(plen * sizeof(size_t));
    size_t n_prompt = tokenize(prompt, plen, vocab, prompt_ids, plen);
    for (size_t i = 0; i < n_prompt && pos < cfg->block_size - 1; i++) {
        forward_inference(model, prompt_ids[i], pos, keys, values, cache_len, logits);
        pos++;
    }
    free(prompt_ids);

    for (int g = 0; g < max_chars && pos < cfg->block_size - 1; g++) {
        size_t token = sample_token(logits, vocab->vocab_size, temperature);
        if (token == vocab->bos_id) break;
        if (token < vocab->vocab_size) {
            unsigned char ch = vocab->chars[token];
            if (out_pos < (int)max_out - 1) {
                output[out_pos++] = (char)ch;
                output[out_pos] = '\0';
            }
        }
        forward_inference(model, token, pos, keys, values, cache_len, logits);
        pos++;
        if (out_pos >= 4 && strstr(output, "@end") != NULL) break;
    }

    free(logits);
    for (int l = 0; l < nl_layers; l++) {
        kv_cache_free(keys[l]);
        kv_cache_free(values[l]);
    }
    free(keys); free(values); free(cache_len);
}

static MicrogptConfig default_cfg(void) {
    MicrogptConfig cfg = microgpt_default_config();
    cfg.n_embd = N_EMBD;
    cfg.n_head = N_HEAD;
    cfg.mlp_dim = MLP_DIM;
    cfg.n_layer = N_LAYER;
    cfg.block_size = BLOCK_SIZE;
    return cfg;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <checkpoint> <train_corpus_for_vocab> <heldout_v2>\n",
            argv[0]);
        return 1;
    }
    const char *ckpt_path = argv[1];
    const char *train_path = argv[2];
    const char *heldout_path = argv[3];

    Docs docs;
    memset(&docs, 0, sizeof(docs));
    if (load_docs(train_path, &docs, MAX_DOCS) != 0) {
        fprintf(stderr, "load_docs failed for %s\n", train_path);
        return 1;
    }
    Vocab vocab;
    memset(&vocab, 0, sizeof(vocab));
    build_vocab(&docs, &vocab);
    fprintf(stderr, "[eval] vocab built: %zu chars (bos=%zu)\n",
            vocab.vocab_size, vocab.bos_id);

    MicrogptConfig cfg = default_cfg();
    size_t ne = (size_t)cfg.n_embd;
    size_t bs = (size_t)cfg.block_size;
    size_t md = (size_t)cfg.mlp_dim;
    size_t nlay = (size_t)cfg.n_layer;
    size_t scratch_n = vocab.vocab_size * ne * 2 + bs * ne
                     + nlay * (4 * ne * ne + 2 * md * ne) + 1024;
    scalar_t *m = (scalar_t *)calloc(scratch_n, sizeof(scalar_t));
    scalar_t *v = (scalar_t *)calloc(scratch_n, sizeof(scalar_t));
    int step = 0;
    Model *model = checkpoint_load(ckpt_path, vocab.vocab_size, &cfg, m, v, &step);
    free(m); free(v);
    if (!model) {
        fprintf(stderr, "checkpoint_load failed for %s (vocab=%zu)\n",
                ckpt_path, vocab.vocab_size);
        free(docs.data); free(docs.lines); free(docs.doc_lens);
        free(vocab.chars);
        return 1;
    }
    fprintf(stderr, "[eval] checkpoint loaded: step=%d vocab=%zu\n",
            step, vocab.vocab_size);

    char prompts[MAX_PROMPTS][512];
    char refs[MAX_PROMPTS][64];
    int n_prompts = load_heldout_prompts(heldout_path, prompts, refs);
    if (n_prompts == 0) {
        fprintf(stderr, "no prompts in %s\n", heldout_path);
        model_free(model);
        free(docs.data); free(docs.lines); free(docs.doc_lens);
        free(vocab.chars);
        return 1;
    }
    fprintf(stderr, "[eval] loaded %d held-out prompts\n", n_prompts);

    seed_rng(42);
    int well_formed = 0;
    int parsed = 0;
    int verified = 0;
    char gen[MAX_OUT];
    for (int i = 0; i < n_prompts; i++) {
        char primed[1024];
        snprintf(primed, sizeof(primed), "// %s\n", prompts[i]);
        generate_char(model, &cfg, &vocab, primed, gen, sizeof(gen),
                      (scalar_t)0.3f, 800);

        int has_graph = strstr(gen, "@graph") != NULL;
        int has_end = strstr(gen, "@end") != NULL;
        if (has_graph && has_end) well_formed++;
        const char *gstart = strstr(gen, "@graph");
        const char *gend = strstr(gen, "@end");
        if (gstart && gend && gend > gstart) {
            char gbuf[MAX_OUT];
            size_t glen = (size_t)(gend - gstart) + 4;
            if (glen >= sizeof(gbuf)) glen = sizeof(gbuf) - 1;
            memcpy(gbuf, gstart, glen);
            gbuf[glen] = '\0';
            Pipeline *p = pipeline_parse_text_tolerant(gbuf);
            if (p) {
                parsed++;
                PipelineRepairReport rep;
                pipeline_repair(p, &rep);
                if (pipeline_verify(p) == 0) verified++;
                pipeline_free(p);
            }
        }
        fprintf(stderr, "[%2d/%2d] ref=%-30s wf=%d parsed=%d verified-running=%d\n",
                i + 1, n_prompts, refs[i],
                (has_graph && has_end) ? 1 : 0, parsed, verified);
        if (i < 5) {
            fprintf(stderr, "        prompt: %s\n", prompts[i]);
            fprintf(stderr, "        gen[:240]: %.240s\n", gen);
        }
    }

    fprintf(stderr, "\n[E12 Phase 5 - v2 sealed held-out]\n");
    fprintf(stderr, "  prompts:      %d\n", n_prompts);
    fprintf(stderr, "  well-formed:  %d / %d = %.0f%%\n",
            well_formed, n_prompts, 100.0 * well_formed / n_prompts);
    fprintf(stderr, "  parsed:       %d / %d = %.0f%%\n",
            parsed, n_prompts, 100.0 * parsed / n_prompts);
    fprintf(stderr, "  verified (T4): %d / %d = %.0f%%\n",
            verified, n_prompts, 100.0 * verified / n_prompts);

    model_free(model);
    free(docs.data); free(docs.lines); free(docs.doc_lens);
    free(vocab.chars);
    return 0;
}
