#!/usr/bin/env bash
# tools/e15_eval_all.sh — Experiment E15 held-out evaluation driver.
#
# Runs Phase 6 of E15: for each task in {klotski, puzzle15} and each
# architecture in {monolithic, OPA}, runs the held-out evaluation
# and writes a per-run CSV + a summary log.  Emits a compact comparison
# table on stdout suitable for pasting into the Section 3 writeup.
#
# Usage:   ./tools/e15_eval_all.sh
# Reads:   build/{klotski,puzzle15}_heldout_large.tsv,
#          checkpoints/{task}_{role}_e15.{ckpt,vocab}
# Writes:  results/{task}_{arch}_eval.{csv,log}

set -e
cd "$(dirname "$0")/.."

MAX_MOVES="${E15_MAX_MOVES:-200}"
LIMIT="${E15_LIMIT:-0}"
LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ]; then LIMIT_ARG="--limit $LIMIT"; fi

echo "============================================================"
echo "  Experiment E15 Phase 6 — held-out evaluation"
echo "============================================================"
echo

declare -A SOLVE
declare -A MEAN_MOVES
declare -A P99

for task in klotski puzzle15; do
    heldout="build/${task}_heldout_large.tsv"
    if [ ! -f "$heldout" ]; then
        echo "[skip] $task: $heldout missing"
        continue
    fi

    # Monolithic arm
    mono_ckpt="checkpoints/${task}_mono_e15.ckpt"
    mono_vocab="checkpoints/${task}_mono_e15.vocab"
    if [ -f "$mono_ckpt" ] && [ -f "$mono_vocab" ]; then
        out_log="results/${task}_mono_eval.log"
        out_csv="results/${task}_mono_eval.csv"
        echo "[run] $task mono ..."
        ./build/e15_mono_eval --task "$task" --ckpt "$mono_ckpt" --vocab "$mono_vocab" \
            --heldout "$heldout" --max-moves "$MAX_MOVES" --out "$out_csv" $LIMIT_ARG \
            > "$out_log" 2>&1
        # Extract solve rate / mean moves / p99.
        sr=$(grep -E "^  solved" "$out_log" | sed -E 's/.*\(([0-9.]+)%\)/\1/')
        mm=$(grep -E "^  mean_moves" "$out_log" | sed -E 's/.* = ([0-9.]+).*/\1/')
        p99=$(grep -E "^  p99_latency" "$out_log" | sed -E 's/.* = ([0-9.]+).*/\1/')
        SOLVE["${task}_mono"]=$sr
        MEAN_MOVES["${task}_mono"]=$mm
        P99["${task}_mono"]=$p99
        echo "      solve=${sr}% mean_moves=${mm} p99=${p99}ms"
    else
        echo "[skip] $task mono: checkpoint missing ($mono_ckpt)"
    fi

    # OPA arm
    planner="checkpoints/${task}_planner_e15"
    player="checkpoints/${task}_player_e15"
    judge="checkpoints/${task}_judge_e15"
    if [ -f "${planner}.ckpt" ] && [ -f "${player}.ckpt" ] && [ -f "${judge}.ckpt" ]; then
        out_log="results/${task}_opa_eval.log"
        out_csv="results/${task}_opa_eval.csv"
        echo "[run] $task opa ..."
        ./build/e15_opa_eval --task "$task" \
            --planner-ckpt "${planner}.ckpt" --planner-vocab "${planner}.vocab" \
            --player-ckpt  "${player}.ckpt"  --player-vocab  "${player}.vocab" \
            --judge-ckpt   "${judge}.ckpt"   --judge-vocab   "${judge}.vocab" \
            --heldout "$heldout" --max-moves "$MAX_MOVES" --out "$out_csv" $LIMIT_ARG \
            > "$out_log" 2>&1
        sr=$(grep -E "^  solved" "$out_log" | sed -E 's/.*\(([0-9.]+)%\)/\1/')
        mm=$(grep -E "^  mean_moves" "$out_log" | sed -E 's/.* = ([0-9.]+).*/\1/')
        p99=$(grep -E "^  p99_latency" "$out_log" | sed -E 's/.* = ([0-9.]+).*/\1/')
        SOLVE["${task}_opa"]=$sr
        MEAN_MOVES["${task}_opa"]=$mm
        P99["${task}_opa"]=$p99
        echo "      solve=${sr}% mean_moves=${mm} p99=${p99}ms"
    else
        echo "[skip] $task opa: at least one organelle missing"
    fi
done

echo
echo "============================================================"
echo "  COMPARISON TABLE (T5 headline)"
echo "============================================================"
printf "%-12s | %10s | %10s | %10s\n" "task" "mono %" "opa %" "margin pp"
echo "-------------+------------+------------+------------"
for task in klotski puzzle15; do
    mono="${SOLVE[${task}_mono]:--}"
    opa="${SOLVE[${task}_opa]:--}"
    if [ "$mono" != "-" ] && [ "$opa" != "-" ]; then
        margin=$(awk -v a="$opa" -v b="$mono" 'BEGIN { printf "%.1f", a - b }')
    else
        margin="-"
    fi
    printf "%-12s | %10s | %10s | %10s\n" "$task" "$mono" "$opa" "$margin"
done
echo
echo "Done.  See results/*_eval.{log,csv} for per-position records."
