/*
 * Type-directed compositional search — implementation.
 * See wiring_compositional_search.h.
 *
 * V1.0.6 (Phase 6 of compositional fix) — three pre-registered changes
 * (RESEARCH_DISCLOSURE.md §4) over the V1.0.4 greedy beam=1 algorithm:
 *
 *   H1. Beam width 2 over the outer pick: try the two highest-scoring
 *       primitives whose output is INT, build a graph for each, choose
 *       the one whose primitive set covers more of the prompt's content
 *       words.
 *
 *   H2. Drop the name-dedup pass on inner picks. Earlier code zeroed
 *       inner_picks[] when the same primitive was picked for multiple
 *       outer input ports; this killed compositions like
 *       `subtract(square(x), square(y))`. The verifier still catches
 *       genuine type errors.
 *
 *   H3. Geodesic-classifier tie-break. When two manifest scores tie at
 *       the outer pick, consult wiring_geo_predict_top_k() for a family
 *       hint and prefer the primitive whose name appears as a substring
 *       of any top-K family name.
 *
 * Each improvement is gated by a compile-time flag (WIRING_BEAM,
 * WIRING_KEEP_DUPS, WIRING_USE_GEO) defaulting to ON. Turning all three
 * off recovers the V1.0.4 greedy behaviour for ablation.
 */

#include "wiring_compositional_search.h"
#include "wiring_primitive_manifest.h"
#include "wiring_arg_binder.h"
#include "wiring_geo_classifier.h"
#include "microgpt_pipeline.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef WIRING_BEAM
#define WIRING_BEAM 2
#endif
#ifndef WIRING_KEEP_DUPS
#define WIRING_KEEP_DUPS 1
#endif
#ifndef WIRING_USE_GEO
#define WIRING_USE_GEO 1
#endif

#define MAX_LOWER_BUF 4096

/* ── keyword matching helpers ──────────────────────────────── */

static void to_lower_copy(const char *src, char *dst, size_t cap) {
    size_t i = 0;
    while (src[i] && i + 1 < cap) {
        char c = src[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c == '-') c = ' ';  /* normalise hyphens to spaces */
        dst[i] = c;
        i++;
    }
    dst[i] = '\0';
}

/* Whole-word case-insensitive substring search.  Both `prompt_lc` and
 * `keyword` should already be lower-case + hyphen-normalised. */
static int prompt_contains_keyword(const char *prompt_lc, const char *keyword_lc) {
    size_t k_len = strlen(keyword_lc);
    if (k_len == 0) return 0;
    const char *p = prompt_lc;
    while ((p = strstr(p, keyword_lc)) != NULL) {
        char before = (p == prompt_lc) ? ' ' : *(p - 1);
        char after  = p[k_len];
        int b_ok = (before == ' ' || before == '\t' || before == '.' || before == ','
                    || before == ';' || before == ':' || before == '!' || before == '?'
                    || before == '\'' || before == '"' || before == '(');
        int a_ok = (after == '\0' || after == ' ' || after == '\t' || after == '.'
                    || after == ',' || after == ';' || after == ':' || after == '!'
                    || after == '?' || after == '\'' || after == '"' || after == ')'
                    || after == 's' /* allow simple plurals */);
        if (b_ok && a_ok) return 1;
        p += k_len;
    }
    return 0;
}

static int score_primitive(const WiringPrimitive *prim, const char *prompt_lc) {
    int score = 0;
    for (int k = 0; k < WIRING_PRIM_MAX_KEYWORDS; k++) {
        const char *kw = prim->keywords[k];
        if (!kw) break;
        char kw_lc[64];
        to_lower_copy(kw, kw_lc, sizeof(kw_lc));
        if (prompt_contains_keyword(prompt_lc, kw_lc)) score++;
    }
    return score;
}

/* Pick the highest-scoring primitive whose output is `desired`.  Returns
 * its index in the manifest, or -1 if no primitive matches.  `excluded`
 * is an optional NULL-terminated array of primitive *names* to skip
 * (used to avoid picking the same primitive at outer + inner positions). */
static int pick_best_primitive(const WiringPrimitive *manifest, int n_manifest,
                               const char *prompt_lc, PipelineTypeKind desired,
                               const char **excluded, int *out_score) {
    int best = -1, best_score = 0;
    for (int i = 0; i < n_manifest; i++) {
        if (manifest[i].output_type != desired) continue;
        int skip = 0;
        if (excluded) {
            for (int e = 0; excluded[e]; e++) {
                if (strcmp(excluded[e], manifest[i].name) == 0) { skip = 1; break; }
            }
        }
        if (skip) continue;
        int s = score_primitive(&manifest[i], prompt_lc);
        if (s > best_score) {
            best_score = s;
            best = i;
        }
    }
    if (out_score) *out_score = best_score;
    return best;
}

/* Generate a unique node id given a base name and a counter. */
static char *fresh_node_id(const char *base, int n, char *buf, size_t cap) {
    snprintf(buf, cap, "%s_%d", base, n);
    return buf;
}

/* Type constructor for kinds we use. */
static PipelineType *make_type(PipelineTypeKind k) {
    switch (k) {
    case PIPE_T_INT:   return pipeline_type_int();
    case PIPE_T_FLOAT: return pipeline_type_float();
    default:           return pipeline_type_any();
    }
}

/* Find the smallest character offset of any of `prim`'s keywords inside
 * `prompt_lc`. Returns INT32_MAX-equivalent (large number) if none match. */
static int earliest_keyword_pos(const WiringPrimitive *prim, const char *prompt_lc) {
    int best = 1 << 30;
    for (int k = 0; k < WIRING_PRIM_MAX_KEYWORDS; k++) {
        const char *kw = prim->keywords[k];
        if (!kw) break;
        char kw_lc[64];
        to_lower_copy(kw, kw_lc, sizeof(kw_lc));
        const char *hit = strstr(prompt_lc, kw_lc);
        if (hit) {
            int pos = (int)(hit - prompt_lc);
            if (pos < best) best = pos;
        }
    }
    return best;
}

/* ── Phase 6: top-N outer pick (beam) ────────────────────────
 *
 * Returns up to `n` outer candidates in `out_idx[]`, ordered by
 * descending score. `*out_count` is populated with the number actually
 * written (≤ n). Ties are broken by EARLIEST-keyword-position in the
 * prompt (English heads its compositions left-to-right — "the X of Y"
 * means X is the outer), then by manifest order, then optionally by
 * the geo-classifier hint when WIRING_USE_GEO is enabled. */
static void pick_top_n_primitives(const WiringPrimitive *manifest, int n_manifest,
                                  const char *prompt_lc, const char *original_prompt,
                                  PipelineTypeKind desired,
                                  int n, int *out_idx, int *out_count) {
    int scores[64]; int idx[64]; int earliest[64];
    int total = 0;
    for (int i = 0; i < n_manifest && total < 64; i++) {
        if (manifest[i].output_type != desired) continue;
        int s = score_primitive(&manifest[i], prompt_lc);
        if (s == 0) continue;
        scores[total] = s;
        idx[total] = i;
        earliest[total] = earliest_keyword_pos(&manifest[i], prompt_lc);
        total++;
    }
#if WIRING_USE_GEO
    /* Stream F (Phase 6b): manifest-driven prior. For each candidate,
     * count how many of its PORT keywords appear in the prompt; bump
     * the score by that many. Per-port keywords are domain-specific
     * nouns (e.g. bmi has weight/height/mass/cm) and matching them
     * shifts the tie-break toward primitives whose argument vocabulary
     * the prompt actually uses. This replaces the V1.0.6 substring
     * match against the legacy FAMILIES table — that table was tuned
     * for the Phase-13-leaked anchors and didn't transfer.
     *
     * Also retained: the legacy substring-match against the geo
     * classifier's top-K family hint, as an additional +1 nudge. */
    if (original_prompt) {
        for (int i = 0; i < total; i++) {
            const WiringPrimitive *prim = &manifest[idx[i]];
            int port_kw_hits = 0;
            for (int ip = 0; ip < prim->n_inputs; ip++) {
                for (int k = 0; k < WIRING_PRIM_MAX_PORT_KEYWORDS; k++) {
                    const char *kw = prim->port_keywords[ip][k];
                    if (!kw) break;
                    char kw_lc[64];
                    to_lower_copy(kw, kw_lc, sizeof(kw_lc));
                    if (prompt_contains_keyword(prompt_lc, kw_lc)) {
                        port_kw_hits++;
                        break;  /* count each port at most once */
                    }
                }
            }
            scores[i] += port_kw_hits;
        }
        const char *top_k[WIRING_GEO_TOP_K] = {0};
        int k = wiring_geo_predict_top_k(original_prompt, top_k);
        for (int i = 0; i < total; i++) {
            const char *pname = manifest[idx[i]].name;
            for (int kk = 0; kk < k; kk++) {
                if (top_k[kk] && pname && strstr(top_k[kk], pname)) {
                    scores[i] += 1;
                    break;
                }
            }
        }
    }
#else
    (void)original_prompt;
#endif
    /* Sort by score desc, then earliest-keyword-pos asc on tie. */
    for (int i = 1; i < total; i++) {
        for (int j = i; j > 0; j--) {
            int swap = 0;
            if (scores[j] > scores[j-1]) swap = 1;
            else if (scores[j] == scores[j-1] && earliest[j] < earliest[j-1]) swap = 1;
            if (!swap) break;
            int ts = scores[j]; scores[j] = scores[j-1]; scores[j-1] = ts;
            int ti = idx[j]; idx[j] = idx[j-1]; idx[j-1] = ti;
            int te = earliest[j]; earliest[j] = earliest[j-1]; earliest[j-1] = te;
        }
    }
    int take = total < n ? total : n;
    for (int i = 0; i < take; i++) out_idx[i] = idx[i];
    *out_count = take;
}

/* Discover the inner picks for an outer (factored out so the binder
 * path can use them too). Writes to inner_picks_out (length
 * WIRING_PRIM_MAX_INPUTS). */
static void discover_inner_picks(const WiringPrimitive *manifest, int n_manifest,
                                 int outer_idx, const char *prompt_lc,
                                 int *inner_picks_out) {
    const WiringPrimitive *outer = &manifest[outer_idx];
    for (int i = 0; i < WIRING_PRIM_MAX_INPUTS; i++) inner_picks_out[i] = -1;
    const char *exclude[3] = { outer->name, NULL, NULL };
    for (int ip = 0; ip < outer->n_inputs; ip++) {
        int s = 0;
        int inner_idx = pick_best_primitive(manifest, n_manifest, prompt_lc,
                                            outer->input_types[ip], exclude, &s);
        if (inner_idx >= 0 && s > 0) inner_picks_out[ip] = inner_idx;
    }

#if !WIRING_KEEP_DUPS
    {
        int best_score_per_inner[64] = {0};
        int best_port_per_inner[64];
        for (int i = 0; i < 64; i++) best_port_per_inner[i] = -1;
        for (int ip = 0; ip < outer->n_inputs; ip++) {
            int idx = inner_picks_out[ip];
            if (idx < 0) continue;
            int s = score_primitive(&manifest[idx], prompt_lc);
            if (s > best_score_per_inner[idx]) {
                if (best_port_per_inner[idx] >= 0)
                    inner_picks_out[best_port_per_inner[idx]] = -1;
                best_score_per_inner[idx] = s;
                best_port_per_inner[idx] = ip;
            } else {
                inner_picks_out[ip] = -1;
            }
        }
    }
#endif
}

/* ── Phase 6: build a graph from a chosen outer + inner picks ────── */
static Pipeline *build_graph_for_outer(
        const WiringPrimitive *manifest, int n_manifest,
        int outer_idx, const char *prompt_lc,
        const char *original_prompt,
        int *inner_picks_out,           /* output: WIRING_PRIM_MAX_INPUTS slots */
        int *signature_in_count_out) {
    (void)original_prompt;
    const WiringPrimitive *outer = &manifest[outer_idx];

    int inner_picks[WIRING_PRIM_MAX_INPUTS];
    discover_inner_picks(manifest, n_manifest, outer_idx, prompt_lc, inner_picks);
    for (int i = 0; i < WIRING_PRIM_MAX_INPUTS; i++) inner_picks_out[i] = inner_picks[i];

    /* ── Build the Pipeline IR graph. ── */
    Pipeline *p = pipeline_create("composed");
    if (!p) return NULL;

    /* First pass: count signature inputs (one per outer-input that has no inner). */
    int n_sig_in = 0;
    int n_sig_in_per_outer_in[WIRING_PRIM_MAX_INPUTS];
    int sub_sig_in_per_outer_in[WIRING_PRIM_MAX_INPUTS];  /* sig inputs from inner's args */
    int n_inner_used = 0;
    for (int ip = 0; ip < outer->n_inputs; ip++) {
        if (inner_picks[ip] < 0) {
            n_sig_in_per_outer_in[ip] = 1;
            sub_sig_in_per_outer_in[ip] = 0;
            n_sig_in += 1;
        } else {
            n_sig_in_per_outer_in[ip] = 0;
            const WiringPrimitive *inner = &manifest[inner_picks[ip]];
            sub_sig_in_per_outer_in[ip] = inner->n_inputs;
            n_sig_in += inner->n_inputs;
            n_inner_used++;
        }
    }
    (void)n_sig_in_per_outer_in;
    (void)sub_sig_in_per_outer_in;

    if (n_sig_in == 0 || n_sig_in > 16) { pipeline_free(p); return NULL; }

    /* Build signature inputs.  Names are arg_0, arg_1, ... — kept simple. */
    const char *sig_in_names[16];
    PipelineType *sig_in_types[16];
    char sig_in_storage[16][16];
    for (int i = 0; i < n_sig_in; i++) {
        snprintf(sig_in_storage[i], sizeof(sig_in_storage[i]), "arg_%d", i);
        sig_in_names[i] = sig_in_storage[i];
        sig_in_types[i] = pipeline_type_int();
    }
    const char *sig_out_names[1] = {"y"};
    PipelineType *sig_out_types[1] = { pipeline_type_int() };
    if (pipeline_set_signature(p, n_sig_in, sig_in_names, sig_in_types,
                               1, sig_out_names, sig_out_types) != 0) {
        pipeline_free(p);
        return NULL;
    }

    /* Add inner nodes first (for topo cleanliness, though pipeline accepts
     * any order). */
    char inner_node_ids[WIRING_PRIM_MAX_INPUTS][32];
    int sig_cursor = 0;
    for (int ip = 0; ip < outer->n_inputs; ip++) {
        if (inner_picks[ip] < 0) continue;
        const WiringPrimitive *inner = &manifest[inner_picks[ip]];
        fresh_node_id("inner", ip, inner_node_ids[ip], sizeof(inner_node_ids[ip]));

        const char *in_names[WIRING_PRIM_MAX_INPUTS];
        PipelineType *in_types[WIRING_PRIM_MAX_INPUTS];
        for (int k = 0; k < inner->n_inputs; k++) {
            in_names[k] = inner->input_names[k] ? inner->input_names[k] : "in";
            in_types[k] = make_type(inner->input_types[k]);
        }
        const char *out_names[1] = {"out"};
        PipelineType *out_types[1] = { make_type(inner->output_type) };
        if (pipeline_add_node(p, inner_node_ids[ip], inner->name,
                              inner->n_inputs, in_names, in_types,
                              1, out_names, out_types) < 0) {
            pipeline_free(p);
            return NULL;
        }

        for (int k = 0; k < inner->n_inputs; k++) {
            if (sig_cursor >= n_sig_in) { pipeline_free(p); return NULL; }
            if (pipeline_connect_signature_in(p, sig_in_storage[sig_cursor],
                                              inner_node_ids[ip], in_names[k]) != 0) {
                pipeline_free(p);
                return NULL;
            }
            sig_cursor++;
        }
    }

    /* Add the outer node. */
    {
        const char *in_names[WIRING_PRIM_MAX_INPUTS];
        PipelineType *in_types[WIRING_PRIM_MAX_INPUTS];
        for (int k = 0; k < outer->n_inputs; k++) {
            in_names[k] = outer->input_names[k] ? outer->input_names[k] : "in";
            in_types[k] = make_type(outer->input_types[k]);
        }
        const char *out_names[1] = {"out"};
        PipelineType *out_types[1] = { make_type(outer->output_type) };
        if (pipeline_add_node(p, "outer", outer->name,
                              outer->n_inputs, in_names, in_types,
                              1, out_names, out_types) < 0) {
            pipeline_free(p);
            return NULL;
        }

        /* Wire each outer input. */
        for (int ip = 0; ip < outer->n_inputs; ip++) {
            if (inner_picks[ip] >= 0) {
                /* Connect inner.out → outer.<port_ip>. */
                if (pipeline_connect(p, inner_node_ids[ip], "out",
                                     "outer", in_names[ip]) != 0) {
                    pipeline_free(p);
                    return NULL;
                }
            } else {
                /* Connect signature input → outer.<port_ip>. */
                if (sig_cursor >= n_sig_in) { pipeline_free(p); return NULL; }
                if (pipeline_connect_signature_in(p, sig_in_storage[sig_cursor],
                                                  "outer", in_names[ip]) != 0) {
                    pipeline_free(p);
                    return NULL;
                }
                sig_cursor++;
            }
        }
    }

    /* outer.out → sig.y */
    if (pipeline_connect_signature_out(p, "outer", "out", "y") != 0) {
        pipeline_free(p);
        return NULL;
    }

    /* Verify. */
    if (pipeline_verify(p) != PIPE_OK) {
        pipeline_free(p);
        return NULL;
    }

    /* Pass the inner-pick set back so the caller can populate the report. */
    for (int ip = 0; ip < WIRING_PRIM_MAX_INPUTS; ip++)
        inner_picks_out[ip] = inner_picks[ip];
    *signature_in_count_out = n_sig_in;
    return p;
}

/* ── Phase 6b: build a graph using the argument binder (Stream D). ──
 *
 * Signature inputs are noun-keyed: each unique prompt noun bound to any
 * port becomes one signature input.  Two ports that bound the same noun
 * share the same signature input — this eliminates the V1.0.6 duplicate-
 * inner misrouting failure mode.  Unbound ports fall back to positional
 * arg_<n> slots (legacy V1.0.4 behaviour). */
static Pipeline *build_graph_with_binder(
        const WiringPrimitive *manifest, int n_manifest,
        int outer_idx, const int *inner_picks,
        const WiringBindResult *bindings,
        int *signature_in_count_out) {
    (void)n_manifest;
    const WiringPrimitive *outer = &manifest[outer_idx];

    Pipeline *p = pipeline_create("composed");
    if (!p) return NULL;

    /* Build the signature input list straight from the binder. */
    int n_sig_in = bindings->n_sig_inputs;
    if (n_sig_in <= 0 || n_sig_in > 16) { pipeline_free(p); return NULL; }

    const char *sig_in_names[16];
    PipelineType *sig_in_types[16];
    for (int i = 0; i < n_sig_in; i++) {
        sig_in_names[i] = bindings->sig_in_names[i];
        sig_in_types[i] = pipeline_type_int();
    }
    const char *sig_out_names[1] = {"y"};
    PipelineType *sig_out_types[1] = { pipeline_type_int() };
    if (pipeline_set_signature(p, n_sig_in, sig_in_names, sig_in_types,
                               1, sig_out_names, sig_out_types) != 0) {
        pipeline_free(p);
        return NULL;
    }

    /* Add each inner node and wire its inputs from the binder. */
    char inner_node_ids[WIRING_PRIM_MAX_INPUTS][32];
    for (int ip = 0; ip < outer->n_inputs; ip++) {
        if (inner_picks[ip] < 0) continue;
        const WiringPrimitive *inner = &manifest[inner_picks[ip]];
        snprintf(inner_node_ids[ip], sizeof(inner_node_ids[ip]), "inner_%d", ip);

        const char *in_names[WIRING_PRIM_MAX_INPUTS];
        PipelineType *in_types[WIRING_PRIM_MAX_INPUTS];
        for (int k = 0; k < inner->n_inputs; k++) {
            in_names[k] = inner->input_names[k] ? inner->input_names[k] : "in";
            in_types[k] = make_type(inner->input_types[k]);
        }
        const char *out_names[1] = {"out"};
        PipelineType *out_types[1] = { make_type(inner->output_type) };
        if (pipeline_add_node(p, inner_node_ids[ip], inner->name,
                              inner->n_inputs, in_names, in_types,
                              1, out_names, out_types) < 0) {
            pipeline_free(p);
            return NULL;
        }
        for (int k = 0; k < inner->n_inputs; k++) {
            int slot = -1;
            for (int _i = 0; _i < bindings->n_bindings; _i++) {
                if (bindings->bindings[_i].node_idx == (1 + ip) &&
                    bindings->bindings[_i].port_idx == k) {
                    slot = bindings->bindings[_i].sig_in_idx; break;
                }
            }
            if (slot < 0 || slot >= n_sig_in) { pipeline_free(p); return NULL; }
            if (pipeline_connect_signature_in(p, sig_in_names[slot],
                                              inner_node_ids[ip], in_names[k]) != 0) {
                pipeline_free(p);
                return NULL;
            }
        }
    }

    /* Add the outer node and wire its inputs. */
    {
        const char *in_names[WIRING_PRIM_MAX_INPUTS];
        PipelineType *in_types[WIRING_PRIM_MAX_INPUTS];
        for (int k = 0; k < outer->n_inputs; k++) {
            in_names[k] = outer->input_names[k] ? outer->input_names[k] : "in";
            in_types[k] = make_type(outer->input_types[k]);
        }
        const char *out_names[1] = {"out"};
        PipelineType *out_types[1] = { make_type(outer->output_type) };
        if (pipeline_add_node(p, "outer", outer->name,
                              outer->n_inputs, in_names, in_types,
                              1, out_names, out_types) < 0) {
            pipeline_free(p);
            return NULL;
        }
        for (int ip = 0; ip < outer->n_inputs; ip++) {
            if (inner_picks[ip] >= 0) {
                if (pipeline_connect(p, inner_node_ids[ip], "out",
                                     "outer", in_names[ip]) != 0) {
                    pipeline_free(p);
                    return NULL;
                }
            } else {
                int slot = -1;
                for (int _i = 0; _i < bindings->n_bindings; _i++) {
                    if (bindings->bindings[_i].node_idx == 0 &&
                        bindings->bindings[_i].port_idx == ip) {
                        slot = bindings->bindings[_i].sig_in_idx; break;
                    }
                }
                if (slot < 0 || slot >= n_sig_in) { pipeline_free(p); return NULL; }
                if (pipeline_connect_signature_in(p, sig_in_names[slot],
                                                  "outer", in_names[ip]) != 0) {
                    pipeline_free(p);
                    return NULL;
                }
            }
        }
    }
    if (pipeline_connect_signature_out(p, "outer", "out", "y") != 0) {
        pipeline_free(p);
        return NULL;
    }
    if (pipeline_verify(p) != PIPE_OK) { pipeline_free(p); return NULL; }
    *signature_in_count_out = n_sig_in;
    return p;
}

/* ── Top-level entry point: beam search over outer candidates. ── */

Pipeline *wiring_compositional_search(const char *prompt,
                                      WiringComposeReport *report) {
    if (report) memset(report, 0, sizeof(*report));
    if (!prompt) return NULL;

    int n_manifest = 0;
    const WiringPrimitive *manifest = wiring_primitive_manifest(&n_manifest);
    if (n_manifest <= 0) return NULL;

    char prompt_lc[MAX_LOWER_BUF];
    to_lower_copy(prompt, prompt_lc, sizeof(prompt_lc));

    /* H1 — beam: take the top-N outer candidates. */
    int outer_top[WIRING_BEAM];
    int outer_count = 0;
    pick_top_n_primitives(manifest, n_manifest, prompt_lc, prompt,
                          PIPE_T_INT, WIRING_BEAM, outer_top, &outer_count);
    if (outer_count == 0) return NULL;

    /* For each outer candidate, build a verified graph and score it by
     * the number of distinct primitive names in the graph that have at
     * least one keyword hit on the prompt — i.e. the "prompt-coverage"
     * heuristic. The candidate with the highest coverage wins. Ties
     * broken by node count (prefer fewer nodes — Occam). */
    Pipeline *best_p = NULL;
    int best_outer = -1;
    int best_inner[WIRING_PRIM_MAX_INPUTS];
    int best_sig_in = 0;
    int best_score = -1;
    int best_nodes = 1 << 30;
    for (int b = 0; b < outer_count; b++) {
        int inner_picks[WIRING_PRIM_MAX_INPUTS];
        int sig_in = 0;

        /* Discover inner picks once per outer candidate. */
        discover_inner_picks(manifest, n_manifest, outer_top[b], prompt_lc, inner_picks);

        /* Run the binder (Stream D). */
        WiringBindResult bindings;
        Pipeline *p = NULL;
        if (wiring_arg_bind(prompt, outer_top[b], inner_picks, &bindings)) {
            p = build_graph_with_binder(manifest, n_manifest,
                                        outer_top[b], inner_picks,
                                        &bindings, &sig_in);
        }
        /* Fallback to the legacy positional builder if binder failed. */
        if (!p) {
            (void)bindings;
            p = build_graph_for_outer(manifest, n_manifest,
                                      outer_top[b], prompt_lc, prompt,
                                      inner_picks, &sig_in);
        }
        if (!p) continue;

        /* Score: total keyword + port-keyword hits across outer + distinct inners.
         * This rewards primitives whose vocabulary the prompt actually uses,
         * not just graphs with more nodes. Fixes the lerp-vs-max_two-with-
         * inner case where the bigger graph misroutes despite the simpler
         * graph being semantically right. */
        int seen[64] = {0};
        int n_inner = 0;
        int coverage = score_primitive(&manifest[outer_top[b]], prompt_lc);
        /* Add port-keyword hits for the outer. */
        {
            const WiringPrimitive *po = &manifest[outer_top[b]];
            for (int ip = 0; ip < po->n_inputs; ip++) {
                for (int k = 0; k < WIRING_PRIM_MAX_PORT_KEYWORDS; k++) {
                    const char *kw = po->port_keywords[ip][k];
                    if (!kw) break;
                    char kw_lc[64];
                    to_lower_copy(kw, kw_lc, sizeof(kw_lc));
                    if (prompt_contains_keyword(prompt_lc, kw_lc)) { coverage++; break; }
                }
            }
        }
        for (int ip = 0; ip < manifest[outer_top[b]].n_inputs; ip++) {
            int pi = inner_picks[ip];
            if (pi >= 0 && pi < 64 && !seen[pi]) {
                seen[pi] = 1;
                n_inner++;
                coverage += score_primitive(&manifest[pi], prompt_lc);
            }
        }
        int n_nodes = 1 + n_inner;
        if (coverage > best_score ||
            (coverage == best_score && n_nodes < best_nodes)) {
            if (best_p) pipeline_free(best_p);
            best_p = p;
            best_outer = outer_top[b];
            best_sig_in = sig_in;
            best_nodes = n_nodes;
            best_score = coverage;
            for (int i = 0; i < WIRING_PRIM_MAX_INPUTS; i++)
                best_inner[i] = inner_picks[i];
        } else {
            pipeline_free(p);
        }
    }

    if (!best_p) return NULL;

    if (report) {
        report->verified = 1;
        report->signature_in_count = best_sig_in;
        report->signature_out_count = 1;
        int idx = 0;
        const WiringPrimitive *outer = &manifest[best_outer];
        for (int ip = 0; ip < outer->n_inputs; ip++) {
            if (best_inner[ip] >= 0 && idx < WIRING_COMPOSE_MAX_NODES) {
                report->primitive_names[idx++] = manifest[best_inner[ip]].name;
            }
        }
        if (idx < WIRING_COMPOSE_MAX_NODES) {
            report->primitive_names[idx++] = outer->name;
        }
        report->n_nodes_used = idx;
        /* Phase 6b Stream E — copy signature input names from the verified
         * graph so the harness can remap inputs by noun. */
        for (int i = 0; i < best_p->n_sig_in &&
                       i < WIRING_COMPOSE_MAX_SIG_INS; i++) {
            const char *nm = best_p->signature_in[i].name
                                 ? best_p->signature_in[i].name : "";
            strncpy(report->signature_in_names[i], nm,
                    WIRING_COMPOSE_NAME_LEN - 1);
            report->signature_in_names[i][WIRING_COMPOSE_NAME_LEN - 1] = '\0';
        }
    }

    return best_p;
}

char *wiring_compositional_search_render(const char *prompt,
                                         Pipeline **pipeline_out,
                                         WiringComposeReport *report) {
    Pipeline *p = wiring_compositional_search(prompt, report);
    if (!p) return NULL;
    char *text = pipeline_render_text(p);
    if (pipeline_out) {
        *pipeline_out = p;
    } else {
        pipeline_free(p);
    }
    return text;
}
