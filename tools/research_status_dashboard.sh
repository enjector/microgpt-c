#!/usr/bin/env bash
# tools/research_status_dashboard.sh
#
# Build (if needed) and run the pre-registration block extractor.
# Pre-registered as Experiment E05 §3.1 (experiments/E05-prereg-methodology-public.md)
# and Experiment 7.2 (docs/research/RESEARCH_OPA_DIRECTIONS.md §8.2).
#
# Walks every docs/research/RESEARCH_*.md, docs/research/wiring_*.md,
# docs/research/ORGANELLE_STATE.md, docs/engineering/.../RESEARCH_DISCLOSURE.md,
# and experiments/E0?.md, emitting:
#   STATUS_DASHBOARD.md
#   STATUS_DASHBOARD.json
# at the repo root.
#
# Usage:
#   bash tools/research_status_dashboard.sh
#   bash tools/research_status_dashboard.sh --md=path.md --json=path.json
#
# Zero deps beyond a C99 compiler and bash.

set -eu

# Walk to repo root from the script location.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Locate or build the parser binary.
BIN="${REPO_ROOT}/build/research_status_dashboard"
SRC="${REPO_ROOT}/tools/research_status_dashboard.c"

if [ ! -f "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    mkdir -p "${REPO_ROOT}/build"
    CC="${CC:-cc}"
    echo "[research_status_dashboard] compiling parser..." >&2
    "$CC" -std=c99 -O2 -Wall -Wextra -o "$BIN" "$SRC"
fi

# Gather the input file list deterministically.
# (Glob sort order is the locale-default lex sort.)
FILES=()
for pattern in \
    "docs/research/RESEARCH_*.md" \
    "docs/research/wiring_*.md" \
    "docs/research/ORGANELLE_STATE.md" \
    "docs/engineering/CLEAN_ROOM_IMPLEMENTATION/RESEARCH_DISCLOSURE.md" \
    "experiments/E0?-*.md"; do
    for f in $pattern; do
        if [ -f "$f" ]; then FILES+=("$f"); fi
    done
done

# Pass-through args (--md=, --json=) come before the file list.
declare -a EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --md=*|--json=*) EXTRA+=("$1"); shift ;;
        *) break ;;
    esac
done

if [ ${#EXTRA[@]} -gt 0 ]; then
    "$BIN" "${EXTRA[@]}" "${FILES[@]}"
else
    "$BIN" "${FILES[@]}"
fi

echo "[research_status_dashboard] walked ${#FILES[@]} files" >&2
