/*
 * wiring_fragments.c — Phase 3b fragment composition implementation.
 *
 * Algorithm (simple, no learning):
 *   1. Scan prompt for keyword hits per fragment.
 *   2. Pick the top-2 (or top-3) fragments by hit count.
 *   3. Order them by the prompt-position of their first matched
 *      keyword (earlier word → earlier in the chain).
 *   4. Build the composed @graph: collect the fragment's non-chain
 *      args into the input signature; for each fragment, emit a
 *      body line whose first arg is either a fresh `<input>` (first
 *      fragment) or `<prev>.out` (subsequent fragments).
 *
 * Output type-matching is trivial here: every fragment's output is
 * `int` and every chain-input position expects `int`. If the
 * primitive registry is later extended with non-int types, the
 * type-checker in `pipeline_verify()` will catch mismatches.
 */

#include "wiring_fragments.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FRAGMENTS    16
#define MAX_KEYWORDS     12
#define MAX_ARGS          4

typedef struct {
    const char *name;
    const char *keywords[MAX_KEYWORDS];
    const char *primitive;            /* native primitive name */
    const char *arg_names[MAX_ARGS];  /* ordered argument names */
    int          n_args;
    int          chain_arg_idx;       /* index into arg_names[] that
                                        * accepts the predecessor's
                                        * output when chained (always
                                        * 0 in this fragment table —
                                        * the first arg is the data
                                        * input, the rest are config). */
} Fragment;

/* Fragment table: one entry per reusable sub-DAG. Keyword bags
 * tightened to be lexically distinct from each other to keep the
 * top-K selection unambiguous. */
static const Fragment FRAGMENTS[] = {
    {
        "clamp_step",
        { "clamp", "clipped", "bounded", "constrained", "pinned", "limited", "inside", "within", NULL },
        "clamp",
        { "x", "lo", "hi" }, 3, 0
    },
    {
        "markup_step",
        { "markup", "marked-up", "uplifted", "raised", NULL },
        "markup",
        { "price", "rate" }, 2, 0
    },
    {
        "discount_step",
        { "discount", "discounted", "reduced", "marked-down", "marked", "down", NULL },
        "discount",
        { "price", "rate" }, 2, 0
    },
    {
        "tax_step",
        { "tax", "taxed", "duty", "levy", NULL },
        "tax_amount",
        { "amount", "rate" }, 2, 0
    },
    {
        "apply_tax_step",
        { "after-tax", "withhold", "withholding", NULL },
        "apply_tax",
        { "amount", "rate" }, 2, 0
    },
    {
        "compound_step",
        { "compound", "compounds", "compounding", "accumulating", "accumulates", NULL },
        "compound",
        { "principal", "rate", "periods" }, 3, 0
    },
    {
        "subtract_principal_step",
        { "minus", "less", "interest", "yield", "earned", NULL },
        "subtract",
        { "x", "y" }, 2, 0
    },
    {
        "percentage_step",
        { "percentage", "percent", "fraction", "portion", "share", NULL },
        "percentage",
        { "part", "whole" }, 2, 0
    },
    {
        "multiply_step",
        { "multiplied", "times", "scaled", "amplified", "magnified", "by", "factor", "gain", "coefficient", NULL },
        "multiply",
        { "x", "y" }, 2, 0
    },
    {
        "abs_diff_step",
        { "absolute", "difference", "magnitude", "unsigned", "gap", "between", NULL },
        "subtract",  /* Stage one of abs(a-b); composer special-cases this below */
        { "x", "y" }, 2, 0
    },
    {
        "average_step",
        { "average", "mean", "midpoint", NULL },
        "average_two",
        { "a", "b" }, 2, 0
    },
    {
        "gcd_step",
        { "gcd", "greatest", "common", "divisor", "shared", NULL },
        "gcd",
        { "a", "b" }, 2, 0
    },
    {
        "relu_step",
        { "relu", "rectified", "thresholded", NULL },
        "relu",
        { "x" }, 1, 0
    },
    {
        "sigmoid_step",
        { "sigmoid", "logistic", NULL },
        "sigmoid",
        { "x" }, 1, 0
    },
    {
        "fib_fact_mul_step",
        { "fibonacci", "factorial", "Leonardo", NULL },
        "_fib_fact_mul",  /* Composer special-cases this fused step. */
        { "n" }, 1, 0
    },
};
static const int N_FRAGMENTS = (int)(sizeof(FRAGMENTS) / sizeof(FRAGMENTS[0]));

/* ---------- Keyword hit detection ---------- */

static int word_char(char c) {
    return isalnum((unsigned char)c) || c == '-' || c == '_';
}

/* Find first byte-offset of `kw` in lowercased prompt, with whole-word
 * boundary at both ends. Returns -1 if not found. */
static int find_keyword_pos(const char *lc_prompt, const char *kw) {
    size_t kl = strlen(kw);
    if (kl == 0) return -1;
    const char *p = lc_prompt;
    while ((p = strstr(p, kw)) != NULL) {
        int before_ok = (p == lc_prompt) || !word_char(p[-1]);
        int after_ok  = !word_char(p[kl]);
        if (before_ok && after_ok) return (int)(p - lc_prompt);
        p += kl;
    }
    return -1;
}

typedef struct {
    int frag_idx;
    int hit_count;
    int first_pos;  /* byte offset of the earliest keyword hit */
} FragHit;

static int fragment_hits(const char *prompt, FragHit *out, int max_out) {
    /* Lowercase the prompt once. */
    char lc[512];
    size_t n = strnlen(prompt, sizeof(lc) - 1);
    for (size_t i = 0; i < n; i++) {
        lc[i] = (char)tolower((unsigned char)prompt[i]);
    }
    lc[n] = '\0';

    int n_hits = 0;
    for (int f = 0; f < N_FRAGMENTS; f++) {
        int count = 0;
        int first = -1;
        /* Lowercase each keyword on the fly. */
        for (int k = 0; k < MAX_KEYWORDS && FRAGMENTS[f].keywords[k]; k++) {
            char kw_lc[64];
            const char *kw = FRAGMENTS[f].keywords[k];
            size_t kl = strnlen(kw, sizeof(kw_lc) - 1);
            for (size_t i = 0; i < kl; i++) kw_lc[i] = (char)tolower((unsigned char)kw[i]);
            kw_lc[kl] = '\0';
            int pos = find_keyword_pos(lc, kw_lc);
            if (pos >= 0) {
                count++;
                if (first < 0 || pos < first) first = pos;
            }
        }
        if (count > 0 && n_hits < max_out) {
            out[n_hits].frag_idx = f;
            out[n_hits].hit_count = count;
            out[n_hits].first_pos = first;
            n_hits++;
        }
    }
    return n_hits;
}

static int cmp_by_hit_desc(const void *a, const void *b) {
    const FragHit *fa = (const FragHit *)a;
    const FragHit *fb = (const FragHit *)b;
    if (fb->hit_count != fa->hit_count) return fb->hit_count - fa->hit_count;
    return fa->first_pos - fb->first_pos;
}

static int cmp_by_pos_asc(const void *a, const void *b) {
    const FragHit *fa = (const FragHit *)a;
    const FragHit *fb = (const FragHit *)b;
    return fa->first_pos - fb->first_pos;
}

/* ---------- Graph composition ---------- */

/* Append a body line for `frag`, where first_input_ref is what to put
 * in the chain-arg slot (e.g. "<x1>" for first fragment, "n0.out" for
 * subsequent). Returns 0 on success. */
static int emit_fragment(char **w, char *end, int node_idx,
                         const Fragment *frag,
                         const char *first_input_ref,
                         const char *const *other_input_refs) {
    /* Special case: abs_diff_step is implemented as subtract + abs_val.
     * Emit two lines and adjust node_idx accordingly via caller. */
    if (strcmp(frag->primitive, "subtract") == 0 &&
        strcmp(frag->name, "abs_diff_step") == 0) {
        int n = snprintf(*w, (size_t)(end - *w),
            "  | n%d_sub = subtract(x: %s, y: %s) :: x:int, y:int -> out:int\n"
            "  | n%d_abs = abs_val(x: n%d_sub.out) :: x:int -> out:int\n",
            node_idx, first_input_ref, other_input_refs[0],
            node_idx, node_idx);
        if (n < 0 || n >= end - *w) return -1;
        *w += n;
        return 0;
    }
    /* Special case: fib_fact_mul_step is fibonacci + factorial + multiply. */
    if (strcmp(frag->primitive, "_fib_fact_mul") == 0) {
        int n = snprintf(*w, (size_t)(end - *w),
            "  | n%d_fib = fibonacci(x: %s) :: x:int -> out:int\n"
            "  | n%d_fact = factorial(x: %s) :: x:int -> out:int\n"
            "  | n%d_mul = multiply(x: n%d_fib.out, y: n%d_fact.out) :: x:int, y:int -> out:int\n",
            node_idx, first_input_ref,
            node_idx, first_input_ref,
            node_idx, node_idx, node_idx);
        if (n < 0 || n >= end - *w) return -1;
        *w += n;
        return 0;
    }
    /* General case: emit a single primitive call with the given inputs. */
    int written;
    if (frag->n_args == 1) {
        written = snprintf(*w, (size_t)(end - *w),
            "  | n%d = %s(%s: %s) :: %s:int -> out:int\n",
            node_idx, frag->primitive,
            frag->arg_names[0], first_input_ref,
            frag->arg_names[0]);
    } else if (frag->n_args == 2) {
        written = snprintf(*w, (size_t)(end - *w),
            "  | n%d = %s(%s: %s, %s: %s) :: %s:int, %s:int -> out:int\n",
            node_idx, frag->primitive,
            frag->arg_names[0], first_input_ref,
            frag->arg_names[1], other_input_refs[0],
            frag->arg_names[0], frag->arg_names[1]);
    } else if (frag->n_args == 3) {
        written = snprintf(*w, (size_t)(end - *w),
            "  | n%d = %s(%s: %s, %s: %s, %s: %s) :: %s:int, %s:int, %s:int -> out:int\n",
            node_idx, frag->primitive,
            frag->arg_names[0], first_input_ref,
            frag->arg_names[1], other_input_refs[0],
            frag->arg_names[2], other_input_refs[1],
            frag->arg_names[0], frag->arg_names[1], frag->arg_names[2]);
    } else {
        return -1;  /* Not supported. */
    }
    if (written < 0 || written >= end - *w) return -1;
    *w += written;
    return 0;
}

/* Output reference for a fragment's emitted node — the LAST internal
 * node name + ".out". Mirrors the special-cases in emit_fragment. */
static int frag_output_name(int node_idx, const Fragment *frag, char *buf, size_t bs) {
    if (strcmp(frag->name, "abs_diff_step") == 0) {
        return snprintf(buf, bs, "n%d_abs.out", node_idx);
    }
    if (strcmp(frag->primitive, "_fib_fact_mul") == 0) {
        return snprintf(buf, bs, "n%d_mul.out", node_idx);
    }
    return snprintf(buf, bs, "n%d.out", node_idx);
}

/* Composition entry point. */
int wiring_compose_for_prompt(const char *prompt, char *out_buf, size_t out_size) {
    if (!prompt || !out_buf || out_size < 256) return 0;

    FragHit hits[MAX_FRAGMENTS];
    int n_hits = fragment_hits(prompt, hits, MAX_FRAGMENTS);
    if (n_hits < 2) return 0;  /* Not a composition. */

    /* Pick top-K by hit count (allow up to 3 for richer chains). */
    qsort(hits, (size_t)n_hits, sizeof(FragHit), cmp_by_hit_desc);
    int K = (n_hits >= 3 && hits[2].hit_count >= 1) ? 3 : 2;

    /* Defensive: cap at 3 and require each of the top-K to have ≥1 hit. */
    if (K > 3) K = 3;
    if (K > n_hits) K = n_hits;

    /* Order the chosen fragments by prompt position (earlier first). */
    qsort(hits, (size_t)K, sizeof(FragHit), cmp_by_pos_asc);

    /* Build the composed @graph text.
     *   : in <first-frag arg 0> -> int
     *   : in <first-frag arg 1...> -> int
     *   : in <subsequent fragments' non-chain args> -> int
     *   : out y -> int
     *   | <body lines>
     *   y <- <last fragment output>
     *   @end
     *
     * We give each input a unique name based on its position in the
     * combined input signature. Arg names from each fragment can
     * collide (e.g. two fragments both have an "x" arg), so we
     * disambiguate by suffixing with the input index. */
    char input_decls[1024] = {0};
    char body[2048] = {0};
    char *body_w = body;
    char *body_end = body + sizeof(body);

    int input_count = 0;
    char input_refs[16][32];  /* "<input_0>" etc. */

    for (int slot = 0; slot < K; slot++) {
        const Fragment *frag = &FRAGMENTS[hits[slot].frag_idx];
        char first_in_ref[64];
        const char *other_in_ref_ptrs[MAX_ARGS];
        char other_in_buf[MAX_ARGS][64];

        int chain_args_consumed = 0;
        if (slot == 0) {
            /* First fragment: every arg is a fresh input. */
            int reservation_n = frag->n_args;
            for (int a = 0; a < reservation_n; a++) {
                if (input_count >= 16) return 0;
                snprintf(input_refs[input_count], sizeof(input_refs[0]),
                         "<%s_%d>", frag->arg_names[a], input_count);
                /* Append to input_decls. */
                size_t cur_len = strlen(input_decls);
                int n = snprintf(input_decls + cur_len,
                                 sizeof(input_decls) - cur_len,
                                 "  : in %s_%d -> int\n",
                                 frag->arg_names[a], input_count);
                if (n < 0 || n >= (int)(sizeof(input_decls) - cur_len)) return 0;
                input_count++;
            }
            snprintf(first_in_ref, sizeof(first_in_ref), "%s", input_refs[input_count - reservation_n]);
            for (int a = 1; a < reservation_n; a++) {
                snprintf(other_in_buf[a-1], sizeof(other_in_buf[a-1]), "%s",
                         input_refs[input_count - reservation_n + a]);
                other_in_ref_ptrs[a-1] = other_in_buf[a-1];
            }
            chain_args_consumed = 1;
        } else {
            /* Subsequent: chain arg comes from previous fragment's output. */
            char prev_out[64];
            const Fragment *prev = &FRAGMENTS[hits[slot - 1].frag_idx];
            frag_output_name(slot - 1, prev, prev_out, sizeof(prev_out));
            snprintf(first_in_ref, sizeof(first_in_ref), "%s", prev_out);
            chain_args_consumed = 1;
            /* Other args are fresh inputs. */
            for (int a = 1; a < frag->n_args; a++) {
                if (input_count >= 16) return 0;
                snprintf(input_refs[input_count], sizeof(input_refs[0]),
                         "<%s_%d>", frag->arg_names[a], input_count);
                size_t cur_len = strlen(input_decls);
                int n = snprintf(input_decls + cur_len,
                                 sizeof(input_decls) - cur_len,
                                 "  : in %s_%d -> int\n",
                                 frag->arg_names[a], input_count);
                if (n < 0 || n >= (int)(sizeof(input_decls) - cur_len)) return 0;
                snprintf(other_in_buf[a-1], sizeof(other_in_buf[a-1]), "%s",
                         input_refs[input_count]);
                other_in_ref_ptrs[a-1] = other_in_buf[a-1];
                input_count++;
            }
        }
        (void)chain_args_consumed;

        /* Emit the fragment body. */
        if (emit_fragment(&body_w, body_end, slot, frag, first_in_ref, other_in_ref_ptrs) != 0) {
            return 0;
        }
    }

    /* Final output reference = last fragment's output. */
    char last_out[64];
    frag_output_name(K - 1, &FRAGMENTS[hits[K - 1].frag_idx], last_out, sizeof(last_out));

    /* Render the full @graph. */
    int n = snprintf(out_buf, out_size,
        "@graph composed_phase3b\n"
        "%s"
        "  : out y -> int\n"
        "%s"
        "  y <- %s\n"
        "@end\n",
        input_decls, body, last_out);
    if (n < 0 || n >= (int)out_size) return 0;
    return 1;
}
