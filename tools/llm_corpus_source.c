/*
 * tools/llm_corpus_source.c — Experiment E12 LLM bridge.
 *
 * Pure C99 + curl subprocess.  T8 hard-locked: zero new build deps.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#define _POSIX_C_SOURCE 200809L

#include "llm_corpus_source.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define LLM_DEFAULT_ENDPOINT "http://127.0.0.1:1234"
#define LLM_DEFAULT_CACHE_DIR ".oql_llm_cache"
#define LLM_DEFAULT_MAX_RETRIES 3
#define LLM_BUF_INIT 65536

static const char *eff_endpoint(const LlmSource *s) {
    return (s && s->endpoint_url && s->endpoint_url[0]) ? s->endpoint_url
                                                        : LLM_DEFAULT_ENDPOINT;
}

static const char *eff_cache_dir(const LlmSource *s) {
    return (s && s->cache_dir && s->cache_dir[0]) ? s->cache_dir
                                                  : LLM_DEFAULT_CACHE_DIR;
}

static int eff_max_retries(const LlmSource *s) {
    if (!s) return LLM_DEFAULT_MAX_RETRIES;
    return (s->max_retries > 0) ? s->max_retries : LLM_DEFAULT_MAX_RETRIES;
}

/* ============================================================
 *  Tolerant JSON extractor.  Supports path syntax
 *  "key1.0.key2.content" where numeric tokens index arrays.
 *  Not a full JSON parser; sufficient for OpenAI chat completions.
 * ============================================================ */

static const char *skip_ws(const char *p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static const char *find_key(const char *p, const char *key) {
    int depth = 0;
    size_t klen = strlen(key);
    while (*p) {
        p = skip_ws(p);
        if (*p == '{' || *p == '[') { depth++; p++; continue; }
        if (*p == '}' || *p == ']') {
            depth--; p++;
            if (depth < 0) return NULL;
            continue;
        }
        if (*p == '"') {
            const char *q = p + 1;
            const char *end = q;
            while (*end && (*end != '"' || *(end-1) == '\\')) end++;
            if (!*end) return NULL;
            size_t flen = (size_t)(end - q);
            const char *after = skip_ws(end + 1);
            int is_key = (*after == ':');
            if (depth == 1 && is_key && flen == klen &&
                memcmp(q, key, klen) == 0) {
                return skip_ws(after + 1);
            }
            p = end + 1;
            continue;
        }
        p++;
    }
    return NULL;
}

static const char *index_array(const char *p, int idx) {
    p = skip_ws(p);
    if (*p != '[') return NULL;
    p++;
    int cur = 0;
    int depth = 0;
    while (*p) {
        p = skip_ws(p);
        if (cur == idx && depth == 0) return p;
        if (*p == '{' || *p == '[') { depth++; p++; continue; }
        if (*p == '}' || *p == ']') {
            if (depth == 0) return NULL;
            depth--; p++; continue;
        }
        if (*p == '"') {
            p++;
            while (*p && (*p != '"' || *(p-1) == '\\')) p++;
            if (*p) p++;
            continue;
        }
        if (*p == ',' && depth == 0) { cur++; p++; continue; }
        p++;
    }
    return NULL;
}

static int copy_json_string(const char *p, char **out) {
    if (!p || *p != '"') return -1;
    p++;
    size_t cap = 256, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return -1;
    while (*p && *p != '"') {
        if (len + 4 >= cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return -1; }
            buf = nb;
        }
        if (*p == '\\' && p[1]) {
            char e = p[1];
            switch (e) {
            case 'n': buf[len++] = '\n'; break;
            case 't': buf[len++] = '\t'; break;
            case 'r': buf[len++] = '\r'; break;
            case '"': buf[len++] = '"';  break;
            case '\\':buf[len++] = '\\'; break;
            case '/': buf[len++] = '/';  break;
            case 'u': {
                if (isxdigit((unsigned char)p[2]) && isxdigit((unsigned char)p[3]) &&
                    isxdigit((unsigned char)p[4]) && isxdigit((unsigned char)p[5])) {
                    buf[len++] = '?';
                    p += 4;
                } else {
                    buf[len++] = 'u';
                }
                break;
            }
            default: buf[len++] = e; break;
            }
            p += 2;
            continue;
        }
        buf[len++] = *p++;
    }
    buf[len] = '\0';
    *out = buf;
    return 0;
}

int llm_json_extract(const char *body, const char *field, char **out) {
    if (!body || !field || !out) return -1;
    *out = NULL;
    const char *p = skip_ws(body);
    if (*p != '{') return -1;
    char path[256];
    strncpy(path, field, sizeof(path) - 1);
    path[sizeof(path) - 1] = '\0';
    char *save = NULL;
    for (char *tok = strtok_r(path, ".", &save); tok;
         tok = strtok_r(NULL, ".", &save)) {
        int is_num = 1;
        for (const char *t = tok; *t; t++) {
            if (!isdigit((unsigned char)*t)) { is_num = 0; break; }
        }
        if (is_num) {
            p = index_array(p, atoi(tok));
        } else {
            p = find_key(p, tok);
        }
        if (!p) return -1;
        p = skip_ws(p);
    }
    return copy_json_string(p, out);
}

/* ============================================================
 *  Cache (FNV-1a 64-bit hash -> hex filename)
 * ============================================================ */

static uint64_t fnv1a_64(const char *s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 0x100000001b3ULL;
    }
    return h;
}

char *llm_cache_path(const LlmSource *src, const char *prompt) {
    if (!src || !prompt) return NULL;
    char key[2048];
    snprintf(key, sizeof(key), "%s|%d|%s",
             src->model_id ? src->model_id : "",
             src->seed,
             prompt);
    uint64_t h = fnv1a_64(key);
    const char *dir = eff_cache_dir(src);
    mkdir(dir, 0755);
    char *path = (char *)malloc(strlen(dir) + 32);
    if (!path) return NULL;
    sprintf(path, "%s/%016llx.txt", dir, (unsigned long long)h);
    return path;
}

static char *slurp_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    if (len_out) *len_out = got;
    return buf;
}

static int spit_file(const char *path, const char *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t w = fwrite(data, 1, len, f);
    fclose(f);
    return (w == len) ? 0 : -1;
}

static char *json_escape(const char *s) {
    size_t n = strlen(s);
    size_t cap = n * 2 + 16;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (j + 8 >= cap) {
            cap *= 2;
            char *nb = (char *)realloc(out, cap);
            if (!nb) { free(out); return NULL; }
            out = nb;
        }
        if (c == '"') { out[j++] = '\\'; out[j++] = '"'; }
        else if (c == '\\') { out[j++] = '\\'; out[j++] = '\\'; }
        else if (c == '\n') { out[j++] = '\\'; out[j++] = 'n'; }
        else if (c == '\r') { out[j++] = '\\'; out[j++] = 'r'; }
        else if (c == '\t') { out[j++] = '\\'; out[j++] = 't'; }
        else if (c < 0x20)  { j += snprintf(out + j, 8, "\\u%04x", c); }
        else                { out[j++] = (char)c; }
    }
    out[j] = '\0';
    return out;
}

int llm_health_check(const LlmSource *src, FILE *out) {
    if (!src || !src->model_id) return -1;
    const char *endpoint = eff_endpoint(src);
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "curl -sS --fail --max-time 10 %s/v1/models 2>&1",
             endpoint);
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    size_t cap = 8192, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { pclose(p); return -1; }
    int c;
    while ((c = fgetc(p)) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); pclose(p); return -1; }
            buf = nb;
        }
        buf[len++] = (char)c;
    }
    buf[len] = '\0';
    int rc = pclose(p);
    if (rc != 0) {
        if (out) fprintf(out,
            "llm_health_check: curl returned %d, endpoint %s unreachable\n",
            rc, endpoint);
        free(buf);
        return -1;
    }
    int found = strstr(buf, src->model_id) != NULL;
    if (out) {
        fprintf(out,
            "llm_health_check: endpoint=%s model=%s available=%s (%zu bytes)\n",
            endpoint, src->model_id, found ? "yes" : "NO", len);
    }
    free(buf);
    return found ? 0 : -1;
}

static int do_one_curl(const LlmSource *src, const char *prompt,
                       char **body_out, size_t *body_len_out, FILE *log) {
    const char *endpoint = eff_endpoint(src);
    char *esc = json_escape(prompt);
    if (!esc) return -1;
    size_t payload_cap = strlen(esc) + strlen(src->model_id ? src->model_id : "") + 512;
    char *payload = (char *)malloc(payload_cap);
    if (!payload) { free(esc); return -1; }
    snprintf(payload, payload_cap,
             "{\"model\":\"%s\","
             "\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
             "\"max_tokens\":16384,"
             "\"temperature\":0.2,"
             "\"seed\":%d,"
             "\"stream\":false}",
             src->model_id ? src->model_id : "",
             esc, src->seed);
    free(esc);

    char tmp_path[256];
    snprintf(tmp_path, sizeof(tmp_path), "/tmp/llm_payload_%d_%lld.json",
             (int)getpid(), (long long)time(NULL));
    if (spit_file(tmp_path, payload, strlen(payload)) != 0) {
        free(payload);
        return -1;
    }
    free(payload);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "curl -sS --fail --max-time 300 -X POST "
             "-H 'Content-Type: application/json' "
             "--data @%s %s/v1/chat/completions 2>&1",
             tmp_path, endpoint);
    FILE *p = popen(cmd, "r");
    if (!p) { unlink(tmp_path); return -1; }
    size_t cap = LLM_BUF_INIT, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { pclose(p); unlink(tmp_path); return -1; }
    int c;
    while ((c = fgetc(p)) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); pclose(p); unlink(tmp_path); return -1; }
            buf = nb;
        }
        buf[len++] = (char)c;
    }
    buf[len] = '\0';
    int rc = pclose(p);
    unlink(tmp_path);
    if (rc != 0) {
        if (log) fprintf(log, "[llm] curl rc=%d body_len=%zu\n", rc, len);
        free(buf);
        return -1;
    }
    *body_out = buf;
    if (body_len_out) *body_len_out = len;
    return 0;
}

int llm_emit(const LlmSource *src,
             const char *prompt,
             char **out, size_t *out_len,
             LlmEmitStats *stats,
             FILE *log) {
    if (!src || !prompt || !out) return -1;
    *out = NULL;
    if (out_len) *out_len = 0;
    if (stats) { stats->cache_hit = 0; stats->curl_attempts = 0; stats->bytes_received = 0; }

    char *cache_path = llm_cache_path(src, prompt);
    if (cache_path) {
        size_t flen = 0;
        char *cached = slurp_file(cache_path, &flen);
        if (cached && flen > 0) {
            if (stats) stats->cache_hit = 1;
            *out = cached;
            if (out_len) *out_len = flen;
            if (log && src->verbose) {
                fprintf(log, "[llm] cache HIT  %s (%zu bytes)\n", cache_path, flen);
            }
            free(cache_path);
            return 0;
        }
        free(cached);
    }

    int max = eff_max_retries(src);
    char *content = NULL;
    for (int attempt = 1; attempt <= max; attempt++) {
        if (stats) stats->curl_attempts = attempt;
        char *body = NULL;
        size_t body_len = 0;
        int rc = do_one_curl(src, prompt, &body, &body_len, log);
        if (rc != 0) {
            if (log) fprintf(log, "[llm] attempt %d/%d curl failed\n", attempt, max);
            continue;
        }
        if (stats) stats->bytes_received = (long)body_len;
        rc = llm_json_extract(body, "choices.0.message.content", &content);
        if (rc != 0 || !content || !*content) {
            /* Qwen3/Gemma4 thinking models emit useful output in reasoning_content
             * when max_tokens runs out before content begins.  Fall back. */
            free(content); content = NULL;
            rc = llm_json_extract(body, "choices.0.message.reasoning_content", &content);
        }
        free(body);
        if (rc == 0 && content && content[0]) break;
        if (log) fprintf(log, "[llm] attempt %d/%d parse failed\n", attempt, max);
        free(content); content = NULL;
    }
    if (!content) {
        free(cache_path);
        return -1;
    }
    size_t clen = strlen(content);
    if (cache_path) {
        spit_file(cache_path, content, clen);
        if (log && src->verbose) {
            fprintf(log, "[llm] cache MISS %s (%zu bytes)\n", cache_path, clen);
        }
        free(cache_path);
    }
    *out = content;
    if (out_len) *out_len = clen;
    return 0;
}

#define MAX_TOKENS 256
#define MAX_TOKLEN 64

static int tokenize(const char *s, char tokens[MAX_TOKENS][MAX_TOKLEN]) {
    int n = 0;
    int j = 0;
    char buf[MAX_TOKLEN];
    for (; *s && n < MAX_TOKENS; s++) {
        char c = (char)tolower((unsigned char)*s);
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            if (j < MAX_TOKLEN - 1) buf[j++] = c;
        } else {
            if (j >= 2) {
                buf[j] = '\0';
                int dup = 0;
                for (int k = 0; k < n; k++) {
                    if (strcmp(tokens[k], buf) == 0) { dup = 1; break; }
                }
                if (!dup) {
                    strncpy(tokens[n], buf, MAX_TOKLEN - 1);
                    tokens[n][MAX_TOKLEN - 1] = '\0';
                    n++;
                }
            }
            j = 0;
        }
    }
    if (j >= 2 && n < MAX_TOKENS) {
        buf[j] = '\0';
        int dup = 0;
        for (int k = 0; k < n; k++) {
            if (strcmp(tokens[k], buf) == 0) { dup = 1; break; }
        }
        if (!dup) {
            strncpy(tokens[n], buf, MAX_TOKLEN - 1);
            tokens[n][MAX_TOKLEN - 1] = '\0';
            n++;
        }
    }
    return n;
}

double llm_jaccard_bow(const char *a, const char *b) {
    if (!a || !b) return 0.0;
    char ta[MAX_TOKENS][MAX_TOKLEN];
    char tb[MAX_TOKENS][MAX_TOKLEN];
    int na = tokenize(a, ta);
    int nb = tokenize(b, tb);
    if (na == 0 || nb == 0) return 0.0;
    int isect = 0;
    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            if (strcmp(ta[i], tb[j]) == 0) { isect++; break; }
        }
    }
    int u = na + nb - isect;
    return (u > 0) ? ((double)isect / (double)u) : 0.0;
}
