/*
 * Argument-to-port binder — implementation.
 * See wiring_arg_binder.h.
 *
 * Algorithm:
 *   1. Tokenise the prompt into content words (lower-cased, hyphen-
 *      normalised, simple punctuation stripped).
 *   2. For each input port across the chosen primitives (outer first,
 *      then inners in input-port order), find the first prompt noun
 *      that matches one of the port's keywords (port_keywords[ip] or
 *      the port name itself).  Match is case-insensitive whole-word.
 *   3. If a noun has already been bound to a previous port, alias —
 *      both ports read the SAME signature input.  Otherwise allocate
 *      a fresh signature input and remember the noun → slot mapping.
 *   4. Ports with no matching noun fall through to positional
 *      arg_<n> slots (the legacy V1.0.4 behaviour, preserved as a
 *      fallback).
 */

#include "wiring_arg_binder.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BINDER_MAX_TOKENS 64
#define BINDER_TOKEN_LEN  WIRING_ARG_NAME_LEN

/* ── Tokeniser ──────────────────────────────────────────── */

/* Lower + hyphen → space + alnum/_ filter; tokens delimited by whitespace. */
static int tokenise_prompt(const char *prompt,
                           char tokens[BINDER_MAX_TOKENS][BINDER_TOKEN_LEN]) {
    int n = 0;
    const char *p = prompt;
    while (*p && n < BINDER_MAX_TOKENS) {
        while (*p && !(isalpha((unsigned char)*p) || *p == '_' || isdigit((unsigned char)*p))) p++;
        if (!*p) break;
        char *out = tokens[n];
        size_t len = 0;
        while (*p && (isalnum((unsigned char)*p) || *p == '_' || *p == '-') && len + 1 < BINDER_TOKEN_LEN) {
            char c = *p++;
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
            if (c == '-') c = '_';
            out[len++] = c;
        }
        out[len] = '\0';
        if (len >= 1) n++;
    }
    return n;
}

/* Compare token vs keyword, both with hyphen→underscore normalisation. */
static int token_matches_kw(const char *tok, const char *kw) {
    if (!tok || !kw) return 0;
    /* Re-normalise on the fly. */
    size_t i = 0;
    while (tok[i] && kw[i]) {
        char a = tok[i]; if (a == '-') a = '_';
        char b = kw[i];  if (b == '-') b = '_';
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
        i++;
    }
    return tok[i] == '\0' && kw[i] == '\0';
}

/* Score: 1 if this port's keyword set contains `tok`, 0 otherwise.
 * Falls back to matching the canonical port name if port_keywords[ip][0]
 * is NULL. */
static int port_accepts_token(const WiringPrimitive *prim, int ip, const char *tok) {
    if (!prim || ip < 0 || ip >= prim->n_inputs) return 0;
    /* Custom keyword set first. */
    int has_custom = 0;
    for (int k = 0; k < WIRING_PRIM_MAX_PORT_KEYWORDS; k++) {
        const char *kw = prim->port_keywords[ip][k];
        if (!kw) break;
        has_custom = 1;
        if (token_matches_kw(tok, kw)) return 1;
    }
    /* Fallback: match the port name itself. */
    if (!has_custom) {
        const char *pn = prim->input_names[ip];
        if (pn && token_matches_kw(tok, pn)) return 1;
    }
    return 0;
}

/* ── Binder ─────────────────────────────────────────────── */

int wiring_arg_bind(const char *prompt,
                    int outer_idx,
                    const int *inner_picks,
                    WiringBindResult *result) {
    if (!prompt || !inner_picks || !result) return 0;
    memset(result, 0, sizeof(*result));

    int n_manifest = 0;
    const WiringPrimitive *manifest = wiring_primitive_manifest(&n_manifest);
    if (outer_idx < 0 || outer_idx >= n_manifest) return 0;

    /* Tokenise once. */
    char tokens[BINDER_MAX_TOKENS][BINDER_TOKEN_LEN];
    int n_tokens = tokenise_prompt(prompt, tokens);

    const WiringPrimitive *outer = &manifest[outer_idx];

    /* "Used" flag per token — once a noun is bound to a port, prefer
     * other unused nouns for subsequent ports. (Repeated nouns are
     * still allowed via the noun→slot map below.) */
    int used[BINDER_MAX_TOKENS] = {0};

    /* Noun → sig_in slot map: list of unique nouns bound so far. */
    char nouns[WIRING_ARG_MAX_SIG_INS][WIRING_ARG_NAME_LEN];
    int  noun_slot[WIRING_ARG_MAX_SIG_INS];
    int  n_unique_nouns = 0;

    /* Walk ports in this order: (outer ports) then (inner ports per
     * outer-input that has an inner). The order matches build_graph_for_outer. */

    /* Build the linear list of (node_idx, n_inputs, prim*). */
    typedef struct { int node_idx; int n_inputs; const WiringPrimitive *prim; } Node;
    Node nodes[WIRING_ARG_MAX_NODES];
    int n_nodes = 0;

    /* Inner nodes first (matches build_graph_for_outer order). */
    for (int ip = 0; ip < outer->n_inputs; ip++) {
        if (inner_picks[ip] >= 0 && n_nodes < WIRING_ARG_MAX_NODES) {
            nodes[n_nodes].node_idx = 1 + ip;   /* synthetic node id */
            nodes[n_nodes].prim = &manifest[inner_picks[ip]];
            nodes[n_nodes].n_inputs = nodes[n_nodes].prim->n_inputs;
            n_nodes++;
        }
    }
    /* Outer last. */
    if (n_nodes < WIRING_ARG_MAX_NODES) {
        nodes[n_nodes].node_idx = 0;
        nodes[n_nodes].prim = outer;
        nodes[n_nodes].n_inputs = outer->n_inputs;
        n_nodes++;
    }

    /* For each port across all nodes (in the order above), pick the
     * earliest prompt token that the port accepts. The outer's port whose
     * input is wired to an inner's output is SKIPPED (no signature
     * input needed for it — handled by build_graph_for_outer). */

    /* Pre-compute which outer ports take an inner. */
    int outer_port_takes_inner[WIRING_PRIM_MAX_INPUTS] = {0};
    for (int ip = 0; ip < outer->n_inputs; ip++) {
        outer_port_takes_inner[ip] = (inner_picks[ip] >= 0) ? 1 : 0;
    }

    int n_bind = 0;
    for (int ni = 0; ni < n_nodes; ni++) {
        const WiringPrimitive *prim = nodes[ni].prim;
        int is_outer = (nodes[ni].node_idx == 0);
        for (int ip = 0; ip < prim->n_inputs; ip++) {
            if (is_outer && outer_port_takes_inner[ip]) continue;  /* wired from inner */

            /* For an inner, bind its arguments by inheriting the OUTER's
             * port-keyword for the slot the inner feeds. This is the
             * fix for the duplicate-inner misrouting bug: `subtract(abs_val(x), abs_val(y))`
             * needs the two abs_val inputs to bind to different prompt
             * nouns ("x" vs "y"), even though both abs_val ports are
             * conventionally "x". The outer's port_ip ("x" or "y") is
             * the right disambiguator.
             *
             * For inners with arity >1 (e.g. compound(P, r, n) feeding
             * an outer's input), only the FIRST input inherits the
             * outer's port-keyword; the rest fall back to the inner's
             * own port keywords. */
            const WiringPrimitive *port_kw_source = prim;
            int port_kw_index = ip;
            if (!is_outer && ip == 0) {
                /* Find which outer port this inner feeds. */
                int outer_port_for_this_inner = nodes[ni].node_idx - 1;
                if (outer_port_for_this_inner >= 0 &&
                    outer_port_for_this_inner < outer->n_inputs) {
                    port_kw_source = outer;
                    port_kw_index = outer_port_for_this_inner;
                }
            }

            /* Look for a matching noun. */
            const char *picked_noun = NULL;
            int picked_token_idx = -1;
            /* Pass 1: prefer an unused noun. */
            for (int t = 0; t < n_tokens; t++) {
                if (used[t]) continue;
                if (port_accepts_token(port_kw_source, port_kw_index, tokens[t])) {
                    picked_noun = tokens[t];
                    picked_token_idx = t;
                    break;
                }
            }
            /* Pass 2 (R2 unification): allow re-using a noun that is
             * already bound to another port — so e.g. "x squared and y"
             * gives `square(x)` and `y`, but "x and x" reuses the same
             * sig input on both. */
            if (!picked_noun) {
                for (int t = 0; t < n_tokens; t++) {
                    if (port_accepts_token(port_kw_source, port_kw_index, tokens[t])) {
                        picked_noun = tokens[t];
                        picked_token_idx = t;
                        break;
                    }
                }
            }

            if (n_bind >= (int)(sizeof(result->bindings) / sizeof(result->bindings[0]))) return 0;
            WiringArgBinding *b = &result->bindings[n_bind++];
            b->node_idx = nodes[ni].node_idx;
            b->port_idx = ip;

            if (picked_noun) {
                used[picked_token_idx] = 1;
                /* Find or insert in the unique-noun map. */
                int slot = -1;
                for (int k = 0; k < n_unique_nouns; k++) {
                    if (strcmp(nouns[k], picked_noun) == 0) { slot = noun_slot[k]; break; }
                }
                if (slot < 0) {
                    if (n_unique_nouns >= WIRING_ARG_MAX_SIG_INS) return 0;
                    strncpy(nouns[n_unique_nouns], picked_noun, WIRING_ARG_NAME_LEN - 1);
                    nouns[n_unique_nouns][WIRING_ARG_NAME_LEN - 1] = '\0';
                    slot = n_unique_nouns;
                    noun_slot[n_unique_nouns] = slot;
                    /* Materialise the sig-in name (use the noun verbatim). */
                    strncpy(result->sig_in_names[slot], picked_noun, WIRING_ARG_NAME_LEN - 1);
                    result->sig_in_names[slot][WIRING_ARG_NAME_LEN - 1] = '\0';
                    n_unique_nouns++;
                }
                b->sig_in_idx = slot;
                strncpy(b->noun, picked_noun, WIRING_ARG_NAME_LEN - 1);
                b->noun[WIRING_ARG_NAME_LEN - 1] = '\0';
            } else {
                /* Fallback: positional arg_<n>. */
                if (n_unique_nouns >= WIRING_ARG_MAX_SIG_INS) return 0;
                int slot = n_unique_nouns;
                snprintf(result->sig_in_names[slot], WIRING_ARG_NAME_LEN, "arg_%d", slot);
                noun_slot[n_unique_nouns] = slot;
                nouns[n_unique_nouns][0] = '\0';  /* no noun key */
                n_unique_nouns++;
                b->sig_in_idx = slot;
                b->noun[0] = '\0';
            }
        }
    }

    result->n_bindings = n_bind;
    result->n_sig_inputs = n_unique_nouns;
    return 1;
}
