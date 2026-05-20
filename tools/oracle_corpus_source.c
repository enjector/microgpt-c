/*
 * tools/oracle_corpus_source.c — Experiment E15 oracle bridge.
 *
 * Pure C99 + popen.  T8 hard-locked: zero new build deps.
 *
 * Copyright (c) 2026 Ajay Soni.  MIT License.
 */

#define _POSIX_C_SOURCE 200809L

#include "oracle_corpus_source.h"

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

#define ORACLE_DEFAULT_CACHE "/.oql_oracle_cache"
#define ORACLE_BUF_INIT 65536

/* ─── FNV-1a 64-bit hash → hex filename ──────────────────────────────── */

static uint64_t fnv1a_64(const char *s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 0x100000001b3ULL;
    }
    return h;
}

static const char *eff_cache_dir(const OracleSource *s) {
    return (s && s->cache_dir && s->cache_dir[0]) ? s->cache_dir
                                                  : ".oql_oracle_cache";
}

char *oracle_cache_path(const OracleSource *src) {
    if (!src) return NULL;
    char key[1024];
    snprintf(key, sizeof(key), "%s|%d|%d|%s",
             src->oracle_binary ? src->oracle_binary : "",
             src->count, src->seed,
             src->difficulty ? src->difficulty : "mixed");
    uint64_t h = fnv1a_64(key);
    const char *dir = eff_cache_dir(src);
    /* Ensure the cache directory exists.  mkdir is idempotent if we
     * accept EEXIST. */
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        return NULL;
    }
    char *out = (char *)malloc(strlen(dir) + 32);
    if (!out) return NULL;
    sprintf(out, "%s/%016llx.jsonl", dir, (unsigned long long)h);
    return out;
}

/* ─── Slurp a file (caller frees buf) ───────────────────────────────── */

static int slurp(const char *path, char **out_buf, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return -1; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    *out_buf = buf;
    if (out_len) *out_len = got;
    return 0;
}

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* ─── popen oracle invocation ───────────────────────────────────────── */

static int run_oracle(const OracleSource *src, char **out_buf, size_t *out_len,
                      FILE *log) {
    if (!src || !src->oracle_binary) return -1;
    /* Build the command line.  We deliberately do NOT use shell
     * metacharacters in any argument value (count, seed are ints;
     * difficulty is a controlled enum), so popen-with-shell is safe
     * here.  Still single-quote the binary path to defend against
     * paths containing spaces. */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "'%s' --count %d --seed %d --difficulty %s --quiet",
             src->oracle_binary,
             src->count > 0 ? src->count : 10,
             src->seed,
             (src->difficulty && src->difficulty[0]) ? src->difficulty : "mixed");
    if (log) fprintf(log, "[oracle] exec: %s\n", cmd);

    FILE *p = popen(cmd, "r");
    if (!p) {
        if (log) fprintf(log, "[oracle] popen failed: %s\n", strerror(errno));
        return -1;
    }
    size_t cap = ORACLE_BUF_INIT, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { pclose(p); return -1; }
    while (!feof(p) && !ferror(p)) {
        if (len + 8192 >= cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); pclose(p); return -1; }
            buf = nb;
        }
        size_t got = fread(buf + len, 1, cap - len - 1, p);
        len += got;
        if (got == 0) break;
    }
    int rc = pclose(p);
    buf[len] = '\0';
    if (rc != 0) {
        if (log) fprintf(log, "[oracle] non-zero exit (%d)\n", rc);
        free(buf);
        return -1;
    }
    *out_buf = buf;
    if (out_len) *out_len = len;
    return 0;
}

/* ─── Atomic write via temp + rename ─────────────────────────────────── */

static int atomic_write(const char *path, const char *data, size_t len) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    if (fwrite(data, 1, len, f) != len) {
        fclose(f); unlink(tmp); return -1;
    }
    if (fclose(f) != 0) { unlink(tmp); return -1; }
    if (rename(tmp, path) != 0) { unlink(tmp); return -1; }
    return 0;
}

/* ─── Public API ─────────────────────────────────────────────────────── */

int oracle_emit(const OracleSource *src,
                char **out, size_t *out_len,
                OracleEmitStats *stats,
                FILE *log) {
    if (!src || !out) return -1;
    *out = NULL;
    if (out_len) *out_len = 0;
    if (stats) memset(stats, 0, sizeof(*stats));

    char *cache_path = oracle_cache_path(src);
    if (!cache_path) return -1;

    if (file_exists(cache_path)) {
        char *buf = NULL; size_t n = 0;
        if (slurp(cache_path, &buf, &n) == 0) {
            *out = buf;
            if (out_len) *out_len = n;
            if (stats) {
                stats->cache_hit = 1;
                stats->bytes_received = (long)n;
            }
            if (log) fprintf(log,
                "[oracle] CACHE HIT: %s (%zu bytes)\n", cache_path, n);
            free(cache_path);
            return 0;
        }
        /* Cache file present but unreadable -- fall through and regenerate. */
    }

    clock_t t0 = clock();
    char *buf = NULL; size_t n = 0;
    int rc = run_oracle(src, &buf, &n, log);
    if (rc != 0 || !buf) { free(cache_path); return -1; }
    double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
    if (stats) {
        stats->cache_hit = 0;
        stats->bytes_received = (long)n;
        stats->wall_seconds = el;
    }
    if (log) fprintf(log,
        "[oracle] FRESH: %zu bytes in %.2fs (writing %s)\n",
        n, el, cache_path);
    /* Persist atomically. */
    if (atomic_write(cache_path, buf, n) != 0) {
        if (log) fprintf(log, "[oracle] WARNING: cache write failed (%s)\n",
                         strerror(errno));
        /* Still return the data — caching is best-effort. */
    }
    free(cache_path);
    *out = buf;
    if (out_len) *out_len = n;
    return 0;
}

/* ─── JSON-line parser ────────────────────────────────────────────────
 *
 * Tolerant parser: extracts "state":"..." and "solution":"..." from
 * each non-empty line.  Other fields ("moves", "md", etc.) are
 * passed through transparently — we only look for these two keys.
 * The buffer is mutated in place (newlines and inner quotes are
 * substituted with NULs as we slice the strings out). */

static char *find_quoted_value(char *p, const char *key, size_t *out_len) {
    size_t klen = strlen(key);
    char target[64];
    snprintf(target, sizeof(target), "\"%s\":\"", key);
    size_t tlen = strlen(target);
    char *start = strstr(p, target);
    if (!start) return NULL;
    start += tlen;
    char *end = start;
    while (*end && *end != '"') {
        if (*end == '\\' && end[1]) end += 2;
        else end++;
    }
    if (*end != '"') return NULL;
    *out_len = (size_t)(end - start);
    *end = '\0';
    (void)klen;
    return start;
}

int oracle_parse_jsonl(char *buf, size_t buf_len,
                       OraclePair **out_pairs, size_t *out_n) {
    if (!buf || !out_pairs || !out_n) return -1;
    *out_pairs = NULL;
    *out_n = 0;

    /* Count newlines as upper bound. */
    size_t max_lines = 1;
    for (size_t i = 0; i < buf_len; i++) if (buf[i] == '\n') max_lines++;
    OraclePair *pairs = (OraclePair *)calloc(max_lines, sizeof(OraclePair));
    if (!pairs) return -1;
    size_t n = 0;

    char *line = buf;
    char *end = buf + buf_len;
    while (line < end) {
        /* Find end-of-line. */
        char *nl = memchr(line, '\n', (size_t)(end - line));
        if (nl) *nl = '\0';
        else {
            /* Last line without trailing newline -- treat as terminated. */
            /* (buf already has a trailing nul from oracle_emit.) */
        }
        if (*line == '{') {
            size_t st_len = 0, sol_len = 0;
            char *st = find_quoted_value(line, "state", &st_len);
            char *sol = find_quoted_value(line, "solution", &sol_len);
            if (st && sol) {
                pairs[n].state        = st;
                pairs[n].state_len    = st_len;
                pairs[n].solution     = sol;
                pairs[n].solution_len = sol_len;
                /* Optional moves field. */
                char *mv = strstr(line, "\"moves\":");
                pairs[n].moves = mv ? atoi(mv + 8) : -1;
                n++;
            }
        }
        line = nl ? nl + 1 : end;
    }
    *out_pairs = pairs;
    *out_n = n;
    return 0;
}

/* ─── Jaccard similarity ─────────────────────────────────────────────
 *
 * For state strings (16-char hex for puzzle15, 20-char letters for
 * klotski), the natural "token" is the character bigram.  We compute
 * Jaccard over the set of bigrams.  This is what the T7 leakage
 * audit uses: if a held-out state's bigram set overlaps a training
 * state's bigram set with Jaccard >= 0.7, we treat them as
 * effectively-the-same position and reject. */

static int build_bigrams(const char *s, int (*table)[256]) {
    /* table is a 256x256 indicator; we just zero it and set entries. */
    int total = 0;
    for (size_t i = 0; s[i] && s[i + 1]; i++) {
        int a = (unsigned char)s[i];
        int b = (unsigned char)s[i + 1];
        if (!table[a][b]) { table[a][b] = 1; total++; }
    }
    return total;
}

double oracle_jaccard_state(const char *a, const char *b) {
    if (!a || !b) return 0.0;
    static int ta[256][256];
    static int tb[256][256];
    memset(ta, 0, sizeof(ta));
    memset(tb, 0, sizeof(tb));
    int na = build_bigrams(a, ta);
    int nb = build_bigrams(b, tb);
    if (na == 0 && nb == 0) return 1.0;
    int inter = 0, uni = 0;
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++) {
            int va = ta[i][j], vb = tb[i][j];
            if (va || vb) uni++;
            if (va && vb) inter++;
        }
    return uni > 0 ? (double)inter / (double)uni : 0.0;
}
