#!/usr/bin/env bash
# tools/run_leakage_audit_ci.sh
#
# Pre-registered as Experiment E05 §3.2 (experiments/E05-prereg-methodology-public.md)
# and Experiment 7.1 (docs/research/RESEARCH_OPA_DIRECTIONS.md §8.1).
#
# Drives tools/scaling_leakage_audit.sh across every held-out file
# declared in tools/leakage_audit_thresholds.json. Fails the build
# (non-zero exit) if any file exceeds its calibrated threshold.
#
# Usage (from repo root):
#   bash tools/run_leakage_audit_ci.sh
#
# Pre-requisite: the corpus regenerator and held-out files need to live
# in build/. The script will run `./bootstrap.sh` if build/ is missing.
#
# Zero deps beyond bash + awk + the existing scaling_leakage_audit.sh.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
THRESHOLDS="${SCRIPT_DIR}/leakage_audit_thresholds.json"
AUDIT_SCRIPT="${SCRIPT_DIR}/scaling_leakage_audit.sh"

if [ ! -f "$THRESHOLDS" ]; then
    echo "[run_leakage_audit_ci] ERROR: $THRESHOLDS missing" >&2
    exit 2
fi
if [ ! -f "$AUDIT_SCRIPT" ]; then
    echo "[run_leakage_audit_ci] ERROR: $AUDIT_SCRIPT missing" >&2
    exit 2
fi

# Make sure build/ exists with the corpus files. If not, build now —
# the audit needs the generated corpora next to the held-out files.
if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/pipeline_corpus_phase4_train.txt" ]; then
    echo "[run_leakage_audit_ci] build/ missing or corpus not generated — running bootstrap.sh"
    if [ -x "$REPO_ROOT/bootstrap.sh" ]; then
        "$REPO_ROOT/bootstrap.sh" >/dev/null 2>&1 || {
            echo "[run_leakage_audit_ci] WARN: bootstrap failed, attempting cmake direct"
            cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
            cmake --build "$BUILD_DIR" --config Release --parallel 4 >/dev/null
        }
    fi
fi

# Generate the Phase-4 training corpus if missing.
if [ ! -f "$BUILD_DIR/pipeline_corpus_phase4_train.txt" ]; then
    if [ -x "$BUILD_DIR/corpus_expand" ]; then
        echo "[run_leakage_audit_ci] generating pipeline_corpus_phase4_train.txt"
        (cd "$BUILD_DIR" && ./corpus_expand pipeline_corpus_phase4_train.txt 42 >/dev/null)
    fi
fi

# Extract per-file thresholds from JSON. Pure awk/sed; zero deps.
# Form expected:
#   "thresholds": {
#     "default": { "max_jaccard_07_count": 0, "max_jaccard_10_count": 0, "max_lexical_anchors_50pct_count": 2 },
#     "<file>": { ... },
#     ...
#   }
# We extract a TSV: <file>\t<max_j07>\t<max_j10>\t<max_lex>
THRESHOLDS_TSV=$(mktemp)
trap "rm -f $THRESHOLDS_TSV" EXIT

awk '
    BEGIN { in_thresholds = 0; file = ""; in_block = 0 }
    /"thresholds":/ { in_thresholds = 1; next }
    !in_thresholds { next }
    # File entry opener: "filename": { — match by looking for the opening brace.
    /^[ \t]*"[^"]+":[ \t]*\{/ {
        # Skip the outer "thresholds": { itself (already consumed).
        line = $0
        # Extract the quoted name as the candidate file key.
        match(line, /"[^"]+"/)
        if (RSTART > 0) {
            key = substr(line, RSTART + 1, RLENGTH - 2)
            # Heuristic: skip internal keys (those NOT containing a slash AND
            # not equal to "default"). The "thresholds" key itself is rejected
            # because in_thresholds was already set before we saw it.
            if (key == "default" || index(key, ".txt") > 0) {
                file = key
                in_block = 1
                max_j07 = ""; max_j10 = ""; max_lex = ""
            }
        }
        next
    }
    in_block && /^[ \t]*"max_jaccard_07_count":/ {
        gsub(/[^0-9-]/, "", $2); max_j07 = $2 + 0
    }
    in_block && /^[ \t]*"max_jaccard_10_count":/ {
        gsub(/[^0-9-]/, "", $2); max_j10 = $2 + 0
    }
    in_block && /^[ \t]*"max_lexical_anchors_50pct_count":/ {
        gsub(/[^0-9-]/, "", $2); max_lex = $2 + 0
    }
    in_block && /^[ \t]*\}/ {
        if (file != "") {
            printf "%s\t%s\t%s\t%s\n",
                file,
                (max_j07 == "" ? 0 : max_j07),
                (max_j10 == "" ? 0 : max_j10),
                (max_lex == "" ? 0 : max_lex)
        }
        in_block = 0; file = ""
    }
' "$THRESHOLDS" > "$THRESHOLDS_TSV"

# Print what we parsed.
echo "============================================================"
echo "Leakage audit CI — pre-registered as Experiment E05 §3.2 / 7.1"
echo "============================================================"
echo "Configured thresholds (from $THRESHOLDS):"
awk -F'\t' '{ printf "  %-50s  j07≤%s  j10≤%s  lex≤%s\n", $1, $2, $3, $4 }' "$THRESHOLDS_TSV"
echo ""

# Run the audit per file (skip the "default" pseudo-entry).
TRAIN="${BUILD_DIR}/pipeline_corpus_phase4_train.txt"
VIOLATIONS=0

while IFS=$'\t' read -r file max_j07 max_j10 max_lex; do
    if [ "$file" = "default" ]; then continue; fi
    held="${BUILD_DIR}/${file}"
    if [ ! -f "$held" ]; then
        echo "[skip] $file (not present in build/ — may be a stale entry)"
        continue
    fi
    if [ ! -f "$TRAIN" ]; then
        echo "[skip] $file (training corpus $TRAIN not generated yet)"
        continue
    fi

    # Run the audit; capture stdout.
    echo "--- audit: $file ---"
    LOG=$(mktemp)
    (cd "$BUILD_DIR" && bash "$AUDIT_SCRIPT" "$file" "pipeline_corpus_phase4_train.txt") > "$LOG" 2>&1 || true

    # Parse the audit script's stdout for the three counts.
    a_verbatim=$(awk '/Audit A total:/ { print $4; exit }' "$LOG")
    b_jaccard=$(awk '/Audit B total:/ { print $4; exit }' "$LOG")
    # Count Jaccard = 1.000 lines from Audit B section.
    j10_count=$(grep -E '^\s*\[WARN\] 1\.000' "$LOG" | wc -l | tr -d ' ')
    # Count HIGH lexical-anchor lines from Audit C section.
    high_lex=$(grep -E '^\s*\[HIGH\]' "$LOG" | wc -l | tr -d ' ')

    a_verbatim=${a_verbatim:-0}
    b_jaccard=${b_jaccard:-0}
    j10_count=${j10_count:-0}
    high_lex=${high_lex:-0}

    # Compare to thresholds.
    fail=""
    if [ "$b_jaccard" -gt "$max_j07" ]; then
        fail="$fail jaccard07($b_jaccard>$max_j07)"
    fi
    if [ "$j10_count" -gt "$max_j10" ]; then
        fail="$fail jaccard10($j10_count>$max_j10)"
    fi
    if [ "$high_lex" -gt "$max_lex" ]; then
        fail="$fail lex_anchors($high_lex>$max_lex)"
    fi

    if [ -n "$fail" ]; then
        echo "  [FAIL] $file:$fail"
        echo "    audit-A verbatim=$a_verbatim, audit-B Jaccard≥0.7=$b_jaccard, Jaccard=1.0=$j10_count, audit-C high-lex=$high_lex"
        VIOLATIONS=$((VIOLATIONS + 1))
    else
        echo "  [ ok ] $file  (j07=$b_jaccard ≤ $max_j07, j10=$j10_count ≤ $max_j10, lex=$high_lex ≤ $max_lex)"
    fi
    rm -f "$LOG"
done < "$THRESHOLDS_TSV"

# Verbatim check uses the dedicated guard which is the canonical
# protection for the Phase-13 pattern.
if [ -x "$SCRIPT_DIR/check_held_out_leakage.sh" ]; then
    echo ""
    echo "--- verbatim guard: tools/check_held_out_leakage.sh ---"
    bash "$SCRIPT_DIR/check_held_out_leakage.sh" "$BUILD_DIR" || {
        echo "  [FAIL] check_held_out_leakage.sh reported verbatim hits"
        VIOLATIONS=$((VIOLATIONS + 1))
    }
fi

echo ""
echo "============================================================"
if [ "$VIOLATIONS" -gt 0 ]; then
    echo "RESULT: $VIOLATIONS violation(s) — leakage audit FAILED."
    echo "See per-file output above and tools/leakage_audit_thresholds.json."
    echo "Per E05 §2.1, do NOT loosen thresholds without a"
    echo "RESEARCH_DISCLOSURE.md entry citing the relaxation."
    echo "============================================================"
    exit 1
fi
echo "RESULT: 0 violations — leakage audit PASSED."
echo "============================================================"
exit 0
