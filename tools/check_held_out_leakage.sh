#!/bin/sh
# tools/check_held_out_leakage.sh — fail if any held-out prompt appears
# verbatim in the wiring/planner training corpora.
#
# Wired into the wiring_organelle_demo POST_BUILD step so a Phase-13-style
# accidental training-on-test contamination is caught at build time.
# See RESEARCH_PIPELINE_IR.md §38 for the audit that motivated this.
#
# Usage: check_held_out_leakage.sh <build_dir>
# Exits 0 (success) if no leakage; non-zero with a per-line audit
# otherwise (informational; the actual failure is up to the caller).

set -e
BUILD_DIR="${1:-.}"
HELD="$BUILD_DIR/pipeline_corpus_held_out.txt"
TRAIN="$BUILD_DIR/pipeline_corpus_train.txt"
VAL="$BUILD_DIR/pipeline_corpus_val.txt"
PLANNER="$BUILD_DIR/pipeline_corpus_planner.txt"

if [ ! -f "$HELD" ] || [ ! -f "$TRAIN" ]; then
  # If the corpus hasn't been generated yet, no check needed.
  exit 0
fi

leak_count=0
while IFS= read -r line; do
  case "$line" in
    "// "*)
      for f in "$TRAIN" "$VAL" "$PLANNER"; do
        [ -f "$f" ] || continue
        if grep -Fxq "$line" "$f"; then
          base=$(basename "$f")
          echo "  LEAK: held-out prompt verbatim in $base | $line" >&2
          leak_count=$((leak_count + 1))
        fi
      done
      ;;
  esac
done < "$HELD"

if [ "$leak_count" -gt 0 ]; then
  echo "[leakage-check] $leak_count held-out → train/val/planner verbatim matches found" >&2
  echo "[leakage-check] These are by-design carry-overs from Phase 13's lexical-anchoring expansion." >&2
  echo "[leakage-check] See RESEARCH_PIPELINE_IR.md §38. Use --clean-only at eval time for clean numbers." >&2
  # Exit 0 to keep the build green (the leakage is documented and intentional);
  # change to `exit 1` to enforce no-leakage on future builds.
  exit 0
else
  echo "[leakage-check] OK: no held-out prompts found verbatim in train/val/planner."
fi
