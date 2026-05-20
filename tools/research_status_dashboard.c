/* tools/research_status_dashboard.c
 *
 * Pure-C99 pre-registration block extractor + status classifier.
 * Pre-registered as Experiment E05 (experiments/E05-prereg-methodology-public.md)
 * and Experiment 7.2 (RESEARCH_OPA_DIRECTIONS.md §8.2).
 *
 * Walks a list of markdown source files (passed on argv) and emits:
 *   - STATUS_DASHBOARD.md   — markdown table of {file, heading, status, line}
 *   - STATUS_DASHBOARD.json — machine-readable sidecar
 *
 * Block model:
 *   A "pre-reg block" is a Markdown section (## or ### heading) whose body
 *   contains at least one canonical pre-reg marker. The block's name is the
 *   heading text. Its body extends from the heading to the next sibling-or-
 *   higher heading.
 *
 *   Markers (case-sensitive substrings of trimmed lines):
 *     "**Pre-registered targets"            "**Pre-registered skip rule"
 *     "**Pre-registered hypothes"           "**Pre-registered prediction"
 *     "**Pre-registered §"                  "**Hypothesis (locked"
 *     "**Hypothesis** (locked"              "> **H1 (locked"  (and H2/H3/...)
 *     "- **Pre-registered"                  "### N.M Pre-registered"
 *     "### N.M Pre-reg"                     "### N.M Hypotheses"
 *     "### N.M Skip rule"                   "### N.M Aggregate target"
 *
 *   This is intentionally permissive: any of these markers anywhere in a
 *   section is enough to classify the section as a pre-reg block. The
 *   marker count is reported in the sidecar JSON for transparency.
 *
 * Status classification (first-match wins, evaluated on heading + body):
 *   1. heading or body says "CANCELLED" in normative position  → CANCELLED
 *   2. body says "EXCEEDED" / "exceeded prediction"            → EXCEEDED
 *   3. body says "PARTIALLY-RESOLVED" / "PARTIAL"              → PARTIAL
 *   4. body says "FALSIFIED" / "falsified"                     → FALSIFIED
 *   5. body says "RESOLVED" / "CONFIRMED" / "design goal met"  → PASS
 *   6. otherwise                                                → PROPOSAL-LOCKED
 *
 * Unparsed legacy formats (e.g. "H_main / H_alt" tables in
 * wiring_scaling_curve.md) are surfaced as PARSER-MISS records at the
 * end of the dashboard so they remain visible without being silently
 * absorbed by the parser. See E05 §2.1 skip rule.
 *
 * Pure C99, zero deps. No malloc churn — uses static buffers sized for the
 * known corpus (largest source file is RESEARCH_PIPELINE_IR.md at ~3900 lines).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_BLOCKS         4096
#define MAX_LINE           4096
#define MAX_HEADING        512
#define MAX_BLOCK_TEXT     32768
#define MAX_PATH           512
#define MAX_MISSES         256

typedef enum {
    ST_PROPOSAL_LOCKED = 0,
    ST_PASS,
    ST_FALSIFIED,
    ST_PARTIAL,
    ST_CANCELLED,
    ST_EXCEEDED,
    ST__COUNT
} Status;

static const char *status_name(Status s) {
    switch (s) {
        case ST_PROPOSAL_LOCKED: return "PROPOSAL-LOCKED";
        case ST_PASS:            return "PASS";
        case ST_FALSIFIED:       return "FALSIFIED";
        case ST_PARTIAL:         return "PARTIAL";
        case ST_CANCELLED:       return "CANCELLED";
        case ST_EXCEEDED:        return "EXCEEDED";
        default:                 return "UNKNOWN";
    }
}

typedef struct {
    char        file[MAX_PATH];
    int         line;           /* heading line, 1-indexed */
    int         heading_level;  /* 2 for ##, 3 for ### */
    char        heading[MAX_HEADING];
    char        body[MAX_BLOCK_TEXT];
    int         markers;        /* count of pre-reg markers in this block */
    Status      status;
    /* Section path: tracks the level-2 parent section's index so sibling
     * level-3 Outcome sections can be associated for status classification. */
    int         parent_level2_idx;
} PreregBlock;

typedef struct {
    char        file[MAX_PATH];
    int         line;
    char        snippet[160];
} ParserMiss;

static PreregBlock blocks[MAX_BLOCKS];
static int         n_blocks = 0;
static ParserMiss  misses[MAX_MISSES];
static int         n_misses = 0;
static int         status_counts[ST__COUNT] = {0};

/* Trim trailing whitespace in place. */
static void rtrim(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' ||
                     s[n - 1] == ' '  || s[n - 1] == '\t')) {
        s[--n] = '\0';
    }
}

/* Strip leading whitespace and return pointer into the same buffer. */
static const char *ltrim(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

/* Case-insensitive substring search (returns 1 if found). */
static int icontains(const char *hay, const char *needle) {
    size_t nl = strlen(needle);
    if (nl == 0) return 1;
    for (const char *p = hay; *p; p++) {
        size_t i;
        for (i = 0; i < nl; i++) {
            char a = p[i];
            char b = needle[i];
            if (a == '\0') break;
            if (tolower((unsigned char)a) != tolower((unsigned char)b)) break;
        }
        if (i == nl) return 1;
    }
    return 0;
}

/* Case-sensitive substring search. */
static int contains(const char *hay, const char *needle) {
    return strstr(hay, needle) != NULL;
}

/* Is the trimmed line a Markdown heading line?
 * Returns 0 if no, else 2/3/4 for ## / ### / ####, capped at 4.
 */
static int heading_level(const char *line) {
    int level = 0;
    while (line[level] == '#') level++;
    if (level >= 2 && level <= 6 && line[level] == ' ') {
        return level > 4 ? 4 : level;
    }
    return 0;
}

/* Return 1 if the trimmed line is a pre-reg marker. */
static int is_prereg_marker(const char *line) {
    const char *t = ltrim(line);

    /* Bolded forms. */
    if (strncmp(t, "**Pre-registered targets", 24) == 0) return 1;
    if (strncmp(t, "**Pre-registered skip rule", 26) == 0) return 1;
    if (strncmp(t, "**Pre-registered hypothes", 25) == 0) return 1;
    if (strncmp(t, "**Pre-registered prediction", 27) == 0) return 1;
    if (strncmp(t, "**Pre-registered §", 19) == 0) return 1;

    /* Bullet forms. */
    if (strncmp(t, "- **Pre-registered hypothes", 27) == 0) return 1;
    if (strncmp(t, "- **Pre-registered targets", 26) == 0) return 1;
    if (strncmp(t, "- **Pre-registered skip rule", 28) == 0) return 1;
    if (strncmp(t, "- **Pre-registered §", 21) == 0) return 1;
    if (strncmp(t, "- **Pre-registered outcome", 26) == 0) return 1;

    /* Hypothesis (locked …) forms. */
    if (strncmp(t, "**Hypothesis (locked", 20) == 0) return 1;
    if (strncmp(t, "**Hypothesis** (locked", 22) == 0) return 1;

    /* Blockquote H1/H2/H3 (locked ...) form used in experiments/E05*. */
    if (strncmp(t, "> **H", 5) == 0 &&
        (t[5] >= '0' && t[5] <= '9') &&
        contains(t, "(locked")) {
        return 1;
    }

    /* Inline sub-section heading forms — sub-section header that names a
     * pre-reg sub-block. These are level-3 or 4 headings whose text
     * names the canonical pre-reg sub-parts: "Pre-registered targets",
     * "Hypotheses (in falsifiable form)", "Skip rule", "Aggregate target",
     * "Disposition logic", "No-regression invariants".
     */
    int hl = heading_level(t);
    if (hl >= 3 && hl <= 4) {
        const char *htxt = t + hl + 1; /* skip "### " */
        if (contains(htxt, "Pre-registered targets")) return 1;
        if (contains(htxt, "Pre-registered hypothes")) return 1;
        if (contains(htxt, "Pre-registered prediction")) return 1;
        if (contains(htxt, "Pre-registered Phase")) return 1; /* §6.3 form */
        if (contains(htxt, "Pre-reg target")) return 1;
        if (contains(htxt, "Hypotheses (in falsifiable form)")) return 1;
        if (contains(htxt, "Hypotheses (verbatim")) return 1;
        if (contains(htxt, "Pre-registered Phase")) return 1;
    }

    return 0;
}

/* Detect a legacy-format pre-reg miss — a level-2 heading whose text
 * explicitly says "Pre-registered hypotheses" without using the
 * canonical level-3 sub-block layout in the next 20 lines. The wiring
 * scaling_curve.md document uses this shape.
 */
static int is_legacy_h_main_heading(const char *line) {
    const char *t = ltrim(line);
    int hl = heading_level(t);
    if (hl != 2) return 0;
    return contains(t, "Pre-registered hypothes");
}

/* Status classifier — heading first, then body.
 *
 * The classifier uses **outcome markers**: phrases that indicate a
 * measurement was actually made (not just that one was forecast). A
 * pre-reg block without an outcome marker stays PROPOSAL-LOCKED even
 * if the body discusses falsification as a future possibility.
 *
 * Outcome markers are detected via the rollup pass: a sibling level-3
 * "Outcome" / "Disposition" / "Result" / "Hypothesis review" section
 * gets appended to the block's body during rollup. Their content is
 * what the classifier reads to assign a non-PROPOSAL status.
 */
static int has_outcome_marker(const PreregBlock *b) {
    return icontains(b->body, "--- sibling outcome ---") ||
           icontains(b->body, "--- parent level-2 body ---") ||
           icontains(b->heading, "outcome") ||
           icontains(b->heading, "result") ||
           icontains(b->heading, "vs measurement") ||
           icontains(b->heading, "predictions (locked");
}

static Status classify_status(const PreregBlock *b) {
    /* Rule 1: CANCELLED — heading-level signal is authoritative. */
    if (icontains(b->heading, "cancelled")) return ST_CANCELLED;

    int has_outcome = has_outcome_marker(b);

    /* CANCELLED via skip-rule firing — requires outcome marker. */
    if (has_outcome &&
        (icontains(b->body, "**CANCELLED** per") ||
         icontains(b->body, "is cancelled.") ||
         icontains(b->body, "cancelled per the") ||
         icontains(b->body, "skip condition fires") ||
         icontains(b->body, "**cancelled** at this scale") ||
         icontains(b->body, "is **cancelled**"))) {
        return ST_CANCELLED;
    }

    /* Rule 2: EXCEEDED — needs outcome */
    if (has_outcome &&
        (icontains(b->body, "exceeded prediction") ||
         icontains(b->body, "exceeded the escalation") ||
         icontains(b->body, "exceeded the upper bound") ||
         icontains(b->body, "both targets exceeded") ||
         icontains(b->body, "** — exceeded"))) return ST_EXCEEDED;

    /* Rule 3: PARTIAL — needs outcome */
    if (icontains(b->heading, "partially-resolved") ||
        icontains(b->heading, "partial")) return ST_PARTIAL;
    if (has_outcome &&
        (icontains(b->body, "partially-resolved") ||
         icontains(b->body, "partially confirmed") ||
         icontains(b->body, "**partially**") ||
         icontains(b->body, "stays partially-resolved"))) return ST_PARTIAL;

    /* Rule 4: FALSIFIED — needs outcome AND explicit falsification verdict */
    if (icontains(b->heading, "falsified") ||
        icontains(b->heading, "negative result")) return ST_FALSIFIED;
    if (has_outcome &&
        (icontains(b->body, "is falsified") ||
         icontains(b->body, "**falsified**") ||
         icontains(b->body, "are falsified") ||
         icontains(b->body, "h_main is falsified") ||
         icontains(b->body, "hypothesis is **falsified**") ||
         icontains(b->body, "miss by") ||
         icontains(b->body, "missed by wide margin"))) return ST_FALSIFIED;

    /* Rule 5: PASS — needs outcome AND explicit success verdict */
    if (has_outcome &&
        (icontains(b->body, "→ resolved") ||
         icontains(b->body, "**resolved**.") ||
         icontains(b->body, "h1 confirmed") ||
         icontains(b->body, "h2 confirmed") ||
         icontains(b->body, "h3 confirmed") ||
         icontains(b->body, "**confirmed**") ||
         icontains(b->body, "design goal met") ||
         icontains(b->body, "both targets met") ||
         icontains(b->body, "(pass, both") ||
         icontains(b->body, "all six pre-registered targets meet") ||
         icontains(b->body, "**pass**") ||
         icontains(b->body, "— pass") ||
         icontains(b->body, "— shipped"))) return ST_PASS;

    /* Default: PROPOSAL-LOCKED. */
    return ST_PROPOSAL_LOCKED;
}

/* Append a line to a block's body, truncating if necessary. */
static void append_body(PreregBlock *b, const char *s) {
    size_t cur = strlen(b->body);
    size_t add = strlen(s);
    if (cur + add + 2 >= MAX_BLOCK_TEXT) {
        add = MAX_BLOCK_TEXT - cur - 2;
        if ((ssize_t)add <= 0) return;
    }
    memcpy(b->body + cur, s, add);
    b->body[cur + add] = '\n';
    b->body[cur + add + 1] = '\0';
}

/* Open a new section (heading). Returns the block index, or -1 if no
 * room.
 */
static int open_block(const char *path, int line_no, int hl,
                      const char *heading_text, int parent_level2_idx) {
    if (n_blocks >= MAX_BLOCKS) {
        fprintf(stderr, "WARN: MAX_BLOCKS hit at %s:%d\n", path, line_no);
        return -1;
    }
    PreregBlock *b = &blocks[n_blocks];
    memset(b, 0, sizeof *b);
    strncpy(b->file, path, sizeof b->file - 1);
    b->file[sizeof b->file - 1] = '\0';
    b->line = line_no;
    b->heading_level = hl;
    b->parent_level2_idx = parent_level2_idx;
    strncpy(b->heading, heading_text, sizeof b->heading - 1);
    b->heading[sizeof b->heading - 1] = '\0';
    return n_blocks++;
}

/* Process one markdown file. Returns number of marker-bearing blocks
 * added; misses are emitted as a side effect.
 */
static int process_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "WARN: cannot open %s: skipping\n", path);
        return 0;
    }

    char line[MAX_LINE];
    int  line_no = 0;
    /* Section stack: track the currently-open level-2 and level-3 blocks
     * so a marker at level 3 attaches to the deepest open block, while a
     * marker at level 4 still attaches to its level-3 parent. */
    int cur_block_idx = -1;
    int cur_level2_idx = -1;
    int cur_level = 0;

    while (fgets(line, sizeof line, f)) {
        line_no++;
        rtrim(line);

        int hl = heading_level(line);
        if (hl >= 2 && hl <= 3) {
            /* Open a new section. Close any previous section first by
             * leaving cur_block_idx pointing at the new block. */
            const char *htxt = line + hl + 1;
            int parent = (hl == 3) ? cur_level2_idx : -1;
            cur_block_idx = open_block(path, line_no, hl, htxt, parent);
            cur_level = hl;
            if (hl == 2) cur_level2_idx = cur_block_idx;

            /* If the heading itself is a pre-reg marker (e.g.
             * "### 3.2 Pre-registered targets"), count it. The new
             * block self-marks. */
            if (is_prereg_marker(line)) {
                blocks[cur_block_idx].markers++;
            }

            /* Legacy form: ## heading directly named "Pre-registered hypotheses"
             * with no canonical level-3 sub-blocks → surface as miss. */
            if (is_legacy_h_main_heading(line) && n_misses < MAX_MISSES) {
                strncpy(misses[n_misses].file, path, sizeof misses[n_misses].file - 1);
                misses[n_misses].file[sizeof misses[n_misses].file - 1] = '\0';
                misses[n_misses].line = line_no;
                strncpy(misses[n_misses].snippet, line, sizeof misses[n_misses].snippet - 1);
                misses[n_misses].snippet[sizeof misses[n_misses].snippet - 1] = '\0';
                n_misses++;
            }
            continue;
        }
        if (hl == 1) {
            /* Top-level # heading — reset the stack. */
            cur_block_idx = -1;
            cur_level = 0;
            continue;
        }

        /* Body line. Attach to current section. */
        if (cur_block_idx >= 0) {
            append_body(&blocks[cur_block_idx], line);
            if (is_prereg_marker(line)) {
                blocks[cur_block_idx].markers++;
            }
        }
        (void)cur_level;
    }

    fclose(f);
    return 0; /* total added is tracked via the global n_blocks/markers */
}

static const char *default_files[] = {
    "docs/research/RESEARCH_PIPELINE_IR.md",
    "docs/research/RESEARCH_MANIFOLD_LEARNING.md",
    "docs/research/RESEARCH_OPA_DIRECTIONS.md",
    "docs/research/RESEARCH_OPENMYTHOS_CROSS_POLLINATION.md",
    "docs/research/RESEARCH_EML_ORGANELLE.md",
    "docs/research/ORGANELLE_STATE.md",
    "docs/research/wiring_scaling_curve.md",
    "docs/research/wiring_scaling_curve_phase3.md",
    "docs/research/wiring_scaling_post_phase3.md",
    "docs/research/wiring_scaling_v3_deep_negative.md",
    "docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md",
    "experiments/E05-prereg-methodology-public.md",
    NULL
};

/* Before pruning: for each marker-bearing level-3 block, scan sibling
 * level-3 blocks under the same level-2 parent looking for "Outcome",
 * "Disposition", "What's still failing" sections, and append their
 * bodies so the status classifier can read across the experiment.
 *
 * We also propagate the parent's heading into the child's heading field
 * (separated by " | ") so the classifier can read parent-level signals
 * like "## 2. Cancelled phases ..." without confounding the body.
 */
static void rollup_outcomes(void) {
    int orig_n = n_blocks; /* freeze the pre-rollup count */
    for (int i = 0; i < orig_n; i++) {
        PreregBlock *b = &blocks[i];
        if (b->markers == 0) continue;
        if (b->heading_level != 3) continue;
        if (b->parent_level2_idx < 0) continue;
        for (int j = 0; j < orig_n; j++) {
            if (j == i) continue;
            PreregBlock *o = &blocks[j];
            if (o->heading_level != 3) continue;
            if (o->parent_level2_idx != b->parent_level2_idx) continue;
            if (strcmp(o->file, b->file) != 0) continue;
            if (icontains(o->heading, "outcome") ||
                icontains(o->heading, "disposition") ||
                icontains(o->heading, "what's still") ||
                icontains(o->heading, "hypothesis review") ||
                icontains(o->heading, "result")) {
                append_body(b, "--- sibling outcome ---");
                append_body(b, o->body);
            }
        }
        /* Propagate the parent's heading text into the child's
         * heading-classifier-input field by appending it. We DON'T
         * append the parent's body — that re-included the conditional
         * skip-rule "falsified" language and produced false positives.
         */
        int p = b->parent_level2_idx;
        if (p >= 0 && p < orig_n) {
            size_t cur = strlen(b->heading);
            size_t avail = sizeof b->heading - cur - 5;
            if (avail > 8) {
                /* "<child> | <parent>" form. */
                strncat(b->heading, " | ", avail);
                strncat(b->heading, blocks[p].heading, avail - 3);
            }
        }
    }

    /* Second pass: for RESEARCH_PIPELINE_IR.md style files where the
     * pre-reg and its outcome are SIBLING level-2 sections (e.g. §42 pre-reg,
     * §43 outcome; §45 pre-reg, §46 outcome; §47 pre-reg + outcome combined),
     * also pull in the next-and-following-level-2 sections from the same
     * file whose headings contain "outcome", "result", "Phase 3b", etc.
     */
    for (int i = 0; i < orig_n; i++) {
        PreregBlock *b = &blocks[i];
        if (b->markers == 0) continue;
        if (b->heading_level != 3) continue;
        /* Only attempt for blocks whose name contains "Pre-registered" — these
         * are the §N.M sub-sections inside §N where §(N+1) is the outcome. */
        if (!icontains(b->heading, "pre-registered")) continue;
        for (int j = 0; j < orig_n; j++) {
            if (j == i) continue;
            PreregBlock *o = &blocks[j];
            if (strcmp(o->file, b->file) != 0) continue;
            if (o->line <= b->line) continue;
            if (o->heading_level != 2) continue;
            /* Walk forward at most 200 lines. */
            if (o->line - b->line > 200) continue;
            if (icontains(o->heading, "outcome") ||
                icontains(o->heading, "phase 4 ") ||
                icontains(o->heading, "phase 3b") ||
                icontains(o->heading, "phase 6d") ||
                icontains(o->heading, "results vs") ||
                icontains(o->heading, " result")) {
                append_body(b, "--- sibling level-2 outcome ---");
                append_body(b, o->heading);
                append_body(b, o->body);
                break;
            }
        }
    }
}

/* Compact a block array down to marker-bearing blocks only. */
static void prune_non_prereg_blocks(void) {
    int j = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (blocks[i].markers > 0) {
            if (i != j) blocks[j] = blocks[i];
            j++;
        }
    }
    n_blocks = j;
}

static void write_markdown(const char *out_path) {
    FILE *f = fopen(out_path, "w");
    if (!f) { perror(out_path); exit(2); }

    fprintf(f, "# Research pre-registration status dashboard\n\n");
    fprintf(f, "> Generated by `tools/research_status_dashboard.{c,sh}`. Pre-registered ");
    fprintf(f, "as Experiment E05 (`experiments/E05-prereg-methodology-public.md`) and ");
    fprintf(f, "Experiment 7.2 (`docs/research/RESEARCH_OPA_DIRECTIONS.md` §8.2).\n\n");
    fprintf(f, "Status legend: **PROPOSAL-LOCKED** — pre-reg committed, outcome not yet measured. ");
    fprintf(f, "**PASS** — measured outcome at or above pre-reg target. ");
    fprintf(f, "**FALSIFIED** — measured outcome below floor. ");
    fprintf(f, "**PARTIAL** — partially-resolved per disposition logic. ");
    fprintf(f, "**CANCELLED** — pre-reg skip rule fired. ");
    fprintf(f, "**EXCEEDED** — measured outcome above the upper bound of the prediction interval.\n\n");

    fprintf(f, "## Summary\n\n");
    fprintf(f, "| Status | Count |\n");
    fprintf(f, "|---|---:|\n");
    for (int i = 0; i < ST__COUNT; i++) {
        fprintf(f, "| %s | %d |\n", status_name((Status)i), status_counts[i]);
    }
    fprintf(f, "| **Total** | **%d** |\n\n", n_blocks);
    fprintf(f, "Parser misses (legacy-format pre-reg blocks NOT silently absorbed): **%d**\n\n", n_misses);

    fprintf(f, "## Pre-registered blocks\n\n");
    fprintf(f, "| # | Status | Markers | File | Line | Block name (nearest heading) |\n");
    fprintf(f, "|---:|---|---:|---|---:|---|\n");
    for (int i = 0; i < n_blocks; i++) {
        const PreregBlock *b = &blocks[i];
        char esc_heading[MAX_HEADING];
        size_t j = 0;
        for (size_t k = 0; b->heading[k] && j < sizeof esc_heading - 2; k++) {
            if (b->heading[k] == '|') esc_heading[j++] = '/';
            else                       esc_heading[j++] = b->heading[k];
        }
        esc_heading[j] = '\0';
        fprintf(f, "| %d | %s | %d | `%s` | %d | %s |\n",
                i + 1, status_name(b->status), b->markers,
                b->file, b->line, esc_heading);
    }

    if (n_misses > 0) {
        fprintf(f, "\n## Parser misses (legacy-format pre-reg blocks)\n\n");
        fprintf(f, "These blocks use a pre-canonical pre-reg format and are surfaced ");
        fprintf(f, "here so they can be canonicalised in a follow-up commit. Per E05 ");
        fprintf(f, "§2.1's skip rule, the parser is NOT widened to silently absorb them.\n\n");
        fprintf(f, "| File | Line | Snippet |\n");
        fprintf(f, "|---|---:|---|\n");
        for (int i = 0; i < n_misses; i++) {
            char esc[160];
            size_t j = 0;
            for (size_t k = 0; misses[i].snippet[k] && j < sizeof esc - 2; k++) {
                if (misses[i].snippet[k] == '|') esc[j++] = '/';
                else                              esc[j++] = misses[i].snippet[k];
            }
            esc[j] = '\0';
            fprintf(f, "| `%s` | %d | %s |\n", misses[i].file, misses[i].line, esc);
        }
    }

    fclose(f);
}

/* Minimal JSON escaper for the sidecar. */
static void json_escape(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n",  f); break;
            case '\r': fputs("\\r",  f); break;
            case '\t': fputs("\\t",  f); break;
            default:
                if (c < 0x20) fprintf(f, "\\u%04x", c);
                else          fputc(c, f);
                break;
        }
    }
    fputc('"', f);
}

static void write_json(const char *out_path) {
    FILE *f = fopen(out_path, "w");
    if (!f) { perror(out_path); exit(2); }

    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"research_status_dashboard.v1\",\n");
    fprintf(f, "  \"generator\": \"tools/research_status_dashboard.c\",\n");
    fprintf(f, "  \"summary\": {\n");
    fprintf(f, "    \"total\": %d,\n", n_blocks);
    fprintf(f, "    \"misses\": %d,\n", n_misses);
    for (int i = 0; i < ST__COUNT; i++) {
        fprintf(f, "    \"%s\": %d%s\n",
                status_name((Status)i),
                status_counts[i],
                (i == ST__COUNT - 1) ? "" : ",");
    }
    fprintf(f, "  },\n");

    fprintf(f, "  \"blocks\": [\n");
    for (int i = 0; i < n_blocks; i++) {
        const PreregBlock *b = &blocks[i];
        fprintf(f, "    {");
        fprintf(f, "\"file\": ");      json_escape(f, b->file);
        fprintf(f, ", \"line\": %d",    b->line);
        fprintf(f, ", \"heading_level\": %d", b->heading_level);
        fprintf(f, ", \"markers\": %d", b->markers);
        fprintf(f, ", \"status\": ");   json_escape(f, status_name(b->status));
        fprintf(f, ", \"heading\": "); json_escape(f, b->heading);
        fprintf(f, "}%s\n", (i == n_blocks - 1) ? "" : ",");
    }
    fprintf(f, "  ],\n");

    fprintf(f, "  \"misses\": [\n");
    for (int i = 0; i < n_misses; i++) {
        fprintf(f, "    {");
        fprintf(f, "\"file\": ");      json_escape(f, misses[i].file);
        fprintf(f, ", \"line\": %d",    misses[i].line);
        fprintf(f, ", \"snippet\": "); json_escape(f, misses[i].snippet);
        fprintf(f, "}%s\n", (i == n_misses - 1) ? "" : ",");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

int main(int argc, char **argv) {
    const char *md_out  = "STATUS_DASHBOARD.md";
    const char *json_out = "STATUS_DASHBOARD.json";

    int first_file = 1;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--md=", 5) == 0) { md_out = argv[i] + 5; first_file = i + 1; }
        else if (strncmp(argv[i], "--json=", 7) == 0) { json_out = argv[i] + 7; first_file = i + 1; }
        else break;
    }

    const char **files = (const char **)(argv + first_file);
    int n_files = argc - first_file;

    if (n_files == 0) {
        for (int i = 0; default_files[i]; i++) {
            process_file(default_files[i]);
        }
    } else {
        for (int i = 0; i < n_files; i++) {
            process_file(files[i]);
        }
    }

    /* Roll up sibling Outcome / Disposition sections into each marker
     * bearing block's body, so the status classifier can read them. */
    rollup_outcomes();

    /* Compact away sections that didn't show any markers. */
    prune_non_prereg_blocks();

    /* Classify. */
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].status = classify_status(&blocks[i]);
        status_counts[blocks[i].status]++;
    }

    write_markdown(md_out);
    write_json(json_out);

    fprintf(stdout, "extracted %d pre-reg blocks, %d misses\n", n_blocks, n_misses);
    fprintf(stdout, "wrote %s and %s\n", md_out, json_out);
    return 0;
}
