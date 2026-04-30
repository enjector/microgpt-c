#!/usr/bin/env bash
# Wiring organelle full eval scoreboard — Tier 0 harness.
# Runs the five §44.6 evals from RESEARCH_PIPELINE_IR.md and prints a
# Markdown table of headline scores. Run from build/ (or any dir that
# contains the binaries + corpus .txt files copied by add_demo()).
#
# Usage:
#   cd build && bash ../tools/wiring_eval_all.sh > tier_N.md
#
# Exit non-zero if any binary or corpus is missing.

set -u
set -o pipefail

WIRING="./wiring_organelle_demo"
TFIDF="./manifold_tfidf_demo"

for f in "$WIRING" "$TFIDF" \
         pipeline_corpus_held_out.txt \
         pipeline_corpus_composition.txt \
         pipeline_corpus_adversarial.txt \
         pipeline_corpus_phase4_train.txt; do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: missing $f (run from build/ after cmake --build)" >&2
        exit 1
    fi
done

# Run an eval, capture stdout, extract one or more headline scores.
# $1 = label, $2 = command, $3 = newline-separated "metric_name|grep_pattern" pairs.
run_eval() {
    local label="$1"
    local cmd="$2"
    local metrics="$3"
    local out
    out=$(mktemp)
    if ! eval "$cmd" > "$out" 2>&1; then
        echo "| $label | ERROR (non-zero exit) | \`$cmd\` |"
        rm -f "$out"
        return 1
    fi
    local cell=""
    while IFS='|' read -r metric_name pat; do
        [[ -z "$metric_name" ]] && continue
        local line score
        line=$(grep -m1 -E "$pat" "$out" || true)
        if [[ -z "$line" ]]; then
            score="NO-MATCH"
        else
            score=$(echo "$line" | grep -oE '[0-9]+/[0-9]+ \([0-9]+%\)' | head -1)
            score="${score:-PARSE-FAIL}"
        fi
        if [[ -z "$cell" ]]; then
            cell="**$metric_name**: $score"
        else
            cell="$cell<br>$metric_name: $score"
        fi
    done <<< "$metrics"
    echo "| $label | $cell | \`$cmd\` |"
    rm -f "$out"
}

echo "# Wiring scoreboard"
echo
echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ") UTC"
echo "Commit: $(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo
echo "| Eval | Score | Command |"
echo "|---|---|---|"

# For wiring evals, report two metrics:
#   strict-verified [HEADLINE] — graph well-formed and passes verifier
#   correct on all 5 inputs    — Phase 8: end-to-end correct on every input
# §44.2's headline numbers correspond to the second metric.
WIRING_METRICS="HEADLINE (strict-verified)|Best-of-[0-9]+ strict-verified:.*\\[HEADLINE\\]
correct on all inputs|Best-of-[0-9]+ correct on all"

run_eval "Phase 2c clean (anchor)" \
         "$WIRING --clean-only" \
         "$WIRING_METRICS"

run_eval "Phase 3b composition" \
         "$WIRING --composition" \
         "$WIRING_METRICS"

run_eval "Wiring transformer alone" \
         "$WIRING --no-anchor --clean-only" \
         "$WIRING_METRICS"

TFIDF_METRICS="Top-1 EXACT|Top-1 EXACT family match:"

run_eval "TF-IDF adversarial axis-2" \
         "$TFIDF pipeline_corpus_adversarial.txt pipeline_corpus_phase4_train.txt" \
         "$TFIDF_METRICS"

run_eval "TF-IDF no-regression (Phase 2c clean)" \
         "$TFIDF pipeline_corpus_held_out.txt pipeline_corpus_phase4_train.txt" \
         "$TFIDF_METRICS"
