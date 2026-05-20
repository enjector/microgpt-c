/*
 * tools/llm_corpus_source.h — Experiment E12 LLM bridge.
 *
 * Design-time-only bridge to a local LM Studio endpoint.  T8 hard-lock:
 * zero new build deps beyond curl.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#ifndef LLM_CORPUS_SOURCE_H
#define LLM_CORPUS_SOURCE_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *model_id;
    const char *endpoint_url;
    const char *cache_dir;
    int         seed;
    int         max_retries;
    int         verbose;
} LlmSource;

typedef struct {
    int   cache_hit;
    int   curl_attempts;
    long  bytes_received;
} LlmEmitStats;

int  llm_health_check(const LlmSource *src, FILE *out);
int  llm_emit(const LlmSource *src,
              const char *prompt,
              char **out, size_t *out_len,
              LlmEmitStats *stats,
              FILE *log);
double llm_jaccard_bow(const char *a, const char *b);
int  llm_json_extract(const char *body, const char *field, char **out);
char *llm_cache_path(const LlmSource *src, const char *prompt);

#ifdef __cplusplus
}
#endif

#endif /* LLM_CORPUS_SOURCE_H */
