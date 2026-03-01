#!/bin/bash
# ============================================================================
# MicroGPT-C — Reasoning Trace Experiment (Automated A/B Test)
# ============================================================================
#
# This script runs the full 4-phase reasoning trace experiment:
#
#   Phase 1+2: Train standard model, solve 30 puzzles, capture traces
#              and generate enriched corpus
#   Phase 3:   Build combined corpus (standard + enriched)
#   Phase 4:   Train combined model, solve 30 puzzles
#   Phase 5:   Verify scaffolding removal (460K bare model)
#   Phase 6:   Perform capacity contrast (64K assisted vs bare)
#   Phase 7:   Print full scorecard and verification verdict
#
# Usage:
#   cd microgpt-c
#   ./experiments/organelles/puzzle8_reasoning/run_experiment.sh
#
# Prerequisites:
#   cmake -S . -B build   (configure first)
#
# ============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/organelles/puzzle8_reasoning"
RESULTS_DIR="$BUILD_DIR/reasoning_experiment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}"
echo "============================================================================"
echo "  MicroGPT-C — Reasoning Trace Experiment"
echo "  Automated A/B Test: Standard vs Combined Corpus"
echo "============================================================================"
echo -e "${NC}"

# ---- Step 0: Build targets ----

echo -e "${YELLOW}[Step 0] Building experiment targets...${NC}"

cd "$PROJECT_ROOT"
cmake -S . -B build > /dev/null 2>&1

echo "  Building puzzle8_reasoning_demo (standard mode)..."
cmake --build build --target puzzle8_reasoning_demo 2>&1 | grep -E "Built target|error" || true

echo "  Building puzzle8_reasoning_combined_demo (combined mode)..."
cmake --build build --target puzzle8_reasoning_combined_demo 2>&1 | grep -E "Built target|error" || true

echo "  Building bare and 64K verification targets..."
cmake --build build --target puzzle8_bare 2>&1 | grep -E "Built target|error" || true
cmake --build build --target puzzle8_64k_assisted 2>&1 | grep -E "Built target|error" || true
cmake --build build --target puzzle8_64k_bare 2>&1 | grep -E "Built target|error" || true

echo -e "${GREEN}  ✓ Build complete${NC}"
echo

# ---- Step 1+2: Run standard model and capture traces ----

echo -e "${YELLOW}[Phase 1+2] Running STANDARD model (trace capture + enriched corpus generation)...${NC}"
echo "  This trains 5 organelles × 25K steps, then solves 30 puzzles."
echo "  Expected time: ~2-3 minutes"
echo

mkdir -p "$RESULTS_DIR"

cd "$BUILD_DIR"
./puzzle8_reasoning_demo > "$RESULTS_DIR/standard_output.txt" 2>&1

# Extract results
STD_SOLVE=$(grep "^Overall:" "$RESULTS_DIR/standard_output.txt" | grep -oE '[0-9]+ / [0-9]+')
STD_RATE=$(grep "^Overall:" "$RESULTS_DIR/standard_output.txt" | grep -oE '[0-9]+%' | head -1)
STD_EASY=$(grep "^EASY" "$RESULTS_DIR/standard_output.txt" | awk '{print $2}')
STD_MEDIUM=$(grep "^MEDIUM" "$RESULTS_DIR/standard_output.txt" | awk '{print $2}')
STD_HARD=$(grep "^HARD" "$RESULTS_DIR/standard_output.txt" | awk '{print $2}')
STD_PARSE=$(grep "Parse errors:" "$RESULTS_DIR/standard_output.txt" | awk '{print $NF}')
STD_CYCLES=$(grep "Cycle breaks:" "$RESULTS_DIR/standard_output.txt" | awk '{print $NF}')
STD_TRACES=$(grep "Traces written:" "$RESULTS_DIR/standard_output.txt" | awk '{print $NF}')
STD_RECOVERY=$(grep "Recovery traces:" "$RESULTS_DIR/standard_output.txt" | sed 's/.*: *//')
STD_TIME=$(grep "Pipeline time:" "$RESULTS_DIR/standard_output.txt" | awk '{print $NF}')

echo -e "${GREEN}  ✓ Standard model complete: ${STD_SOLVE} solved (${STD_RATE})${NC}"
echo "    Traces: ${STD_TRACES}, Recovery: ${STD_RECOVERY}"
echo

# ---- Step 3: Build combined corpus ----

echo -e "${YELLOW}[Phase 3] Building combined corpus...${NC}"

ENRICHED_FILE="$BUILD_DIR/puzzle8_mover_enriched.txt"
STANDARD_FILE="$EXPERIMENT_DIR/puzzle8_mover.txt"
COMBINED_FILE="$EXPERIMENT_DIR/puzzle8_mover_combined.txt"

if [ ! -f "$ENRICHED_FILE" ]; then
  echo -e "${RED}  ✗ ERROR: Enriched corpus not found at $ENRICHED_FILE${NC}"
  echo "  Phase 1+2 should have generated this file."
  exit 1
fi

cat "$STANDARD_FILE" "$ENRICHED_FILE" > "$COMBINED_FILE"

STD_LINES=$(wc -l < "$STANDARD_FILE" | tr -d ' ')
ENR_LINES=$(wc -l < "$ENRICHED_FILE" | tr -d ' ')
CMB_LINES=$(wc -l < "$COMBINED_FILE" | tr -d ' ')

echo -e "${GREEN}  ✓ Combined corpus: ${CMB_LINES} lines (standard: ${STD_LINES} + enriched: ${ENR_LINES})${NC}"

# Copy to build dir so the combined target can find it
cp "$COMBINED_FILE" "$BUILD_DIR/puzzle8_mover_combined.txt"

# Rebuild combined target to pick up new corpus
echo "  Rebuilding combined target..."
cd "$PROJECT_ROOT"
cmake --build build --target puzzle8_reasoning_combined_demo 2>&1 | grep -E "Built target|error" || true
echo

# ---- Step 4: Run combined model ----

echo -e "${YELLOW}[Phase 4] Running COMBINED model (standard + enriched corpus)...${NC}"
echo "  This trains the mover on the combined corpus, then solves 30 puzzles."
echo "  Context window: 192 (vs 128 for standard)"
echo "  Expected time: ~3-5 minutes"
echo

cd "$BUILD_DIR"
./puzzle8_reasoning_combined_demo > "$RESULTS_DIR/combined_output.txt" 2>&1

# Extract results
CMB_SOLVE=$(grep "^Overall:" "$RESULTS_DIR/combined_output.txt" | grep -oE '[0-9]+ / [0-9]+')
CMB_RATE=$(grep "^Overall:" "$RESULTS_DIR/combined_output.txt" | grep -oE '[0-9]+%' | head -1)
CMB_EASY=$(grep "^EASY" "$RESULTS_DIR/combined_output.txt" | awk '{print $2}')
CMB_MEDIUM=$(grep "^MEDIUM" "$RESULTS_DIR/combined_output.txt" | awk '{print $2}')
CMB_HARD=$(grep "^HARD" "$RESULTS_DIR/combined_output.txt" | awk '{print $2}')
CMB_PARSE=$(grep "Parse errors:" "$RESULTS_DIR/combined_output.txt" | awk '{print $NF}')
CMB_CYCLES=$(grep "Cycle breaks:" "$RESULTS_DIR/combined_output.txt" | awk '{print $NF}')
CMB_TRACES=$(grep "Traces written:" "$RESULTS_DIR/combined_output.txt" | awk '{print $NF}')
CMB_RECOVERY=$(grep "Recovery traces:" "$RESULTS_DIR/combined_output.txt" | sed 's/.*: *//')
CMB_TIME=$(grep "Pipeline time:" "$RESULTS_DIR/combined_output.txt" | awk '{print $NF}')

echo -e "${GREEN}  ✓ Combined model complete: ${CMB_SOLVE} solved (${CMB_RATE})${NC}"
echo

# ---- Step 5: Verification — Bare Model (460K) ----

echo -e "${YELLOW}[Phase 5] Verifying 460K BARE model (no scaffolding)...${NC}"
cd "$BUILD_DIR"
./puzzle8_bare > "$RESULTS_DIR/bare_460k_output.txt" 2>&1
BARE_460_SOLVE=$(grep "^Overall:" "$RESULTS_DIR/bare_460k_output.txt" | grep -oE '[0-9]+ / [0-9]+')
BARE_460_RATE=$(grep "^Overall:" "$RESULTS_DIR/bare_460k_output.txt" | grep -oE '[0-9]+%' | head -1)
BARE_460_CYCLES=$(grep "Cycle breaks:" "$RESULTS_DIR/bare_460k_output.txt" | awk '{print $NF}')
echo -e "${GREEN}  ✓ 460K Bare complete: ${BARE_460_SOLVE} solved (Cycles: ${BARE_460_CYCLES})${NC}"
echo

# ---- Step 6: Verification — Capacity Contrast (64K) ----

echo -e "${YELLOW}[Phase 6] Verifying 64K model (Assisted vs Bare contrast)...${NC}"
echo "  Running 64K Assisted..."
./puzzle8_64k_assisted > "$RESULTS_DIR/assisted_64k_output.txt" 2>&1
ASST_64_SOLVE=$(grep "^Overall:" "$RESULTS_DIR/assisted_64k_output.txt" | grep -oE '[0-9]+ / [0-9]+')
ASST_64_CYCLES=$(grep "Cycle breaks:" "$RESULTS_DIR/assisted_64k_output.txt" | awk '{print $NF}')

echo "  Running 64K Bare..."
./puzzle8_64k_bare > "$RESULTS_DIR/bare_64k_output.txt" 2>&1
BARE_64_SOLVE=$(grep "^Overall:" "$RESULTS_DIR/bare_64k_output.txt" | grep -oE '[0-9]+ / [0-9]+')
BARE_64_CYCLES=$(grep "Cycle breaks:" "$RESULTS_DIR/bare_64k_output.txt" | awk '{print $NF}')

echo -e "${GREEN}  ✓ 64K contrast complete: Assisted ${ASST_64_SOLVE} vs Bare ${BARE_64_SOLVE}${NC}"
echo

# ---- Step 7: Print scorecard ----

echo -e "${BOLD}${CYAN}"
echo "============================================================================"
echo "  REASONING TRACE EXPERIMENT — A/B SCORECARD"
echo "============================================================================"
echo -e "${NC}"

printf "${BOLD}%-22s  %-16s  %-16s${NC}\n" "Metric" "Standard" "Combined"
printf "%-22s  %-16s  %-16s\n" "──────────────────────" "────────────────" "────────────────"
printf "%-22s  %-16s  %-16s\n" "Corpus size" "${STD_LINES} lines" "${CMB_LINES} lines"
printf "%-22s  %-16s  %-16s\n" "Context (BLOCK_SIZE)" "128" "192"
printf "%-22s  %-16s  %-16s\n" "" "" ""
printf "%-22s  %-16s  %-16s\n" "Overall solve rate" "${STD_SOLVE} (${STD_RATE})" "${CMB_SOLVE} (${CMB_RATE})"
printf "%-22s  %-16s  %-16s\n" "  Easy band" "${STD_EASY}" "${CMB_EASY}"
printf "%-22s  %-16s  %-16s\n" "  Medium band" "${STD_MEDIUM}" "${CMB_MEDIUM}"
printf "%-22s  %-16s  %-16s\n" "  Hard band" "${STD_HARD}" "${CMB_HARD}"
printf "%-22s  %-16s  %-16s\n" "" "" ""
printf "%-22s  %-16s  %-16s\n" "Parse errors" "${STD_PARSE}" "${CMB_PARSE}"
printf "%-22s  %-16s  %-16s\n" "Cycle breaks" "${STD_CYCLES}" "${CMB_CYCLES}"
printf "%-22s  %-16s  %-16s\n" "Recovery traces" "${STD_RECOVERY}" "${CMB_RECOVERY}"
printf "%-22s  %-16s  %-16s\n" "Pipeline time" "${STD_TIME}" "${CMB_TIME}"

echo
echo -e "${BOLD}${CYAN}"
echo "============================================================================"
echo "  VERIFICATION — SCALING vs SCAFFOLDING"
echo "============================================================================"
echo -e "${NC}"

printf "${BOLD}%-22s  %-16s  %-16s  %-16s${NC}\n" "Metric" "64K Assisted" "64K Bare" "460K Bare"
printf "%-22s  %-16s  %-16s  %-16s\n" "──────────────────────" "────────────────" "────────────────" "────────────────"
printf "%-22s  %-16s  %-16s  %-16s\n" "Solve Rate" "${ASST_64_SOLVE}" "${BARE_64_SOLVE}" "${BARE_460_SOLVE}"
printf "%-22s  %-16s  %-16s  %-16s\n" "Cycle Breaks" "${ASST_64_CYCLES}" "${BARE_64_CYCLES}" "${BARE_460_CYCLES}"

# Capacity check
BARE_460_SOLVED_N=$(echo "$BARE_460_SOLVE" | awk -F/ '{print $1}' | tr -d ' ')
ASST_64_SOLVED_N=$(echo "$ASST_64_SOLVE" | awk -F/ '{print $1}' | tr -d ' ')
BARE_64_SOLVED_N=$(echo "$BARE_64_SOLVE" | awk -F/ '{print $1}' | tr -d ' ')

echo -e "${BOLD}Scaling Verdict:${NC}"
if [ "$BARE_460_SOLVED_N" -ge "$STD_SOLVED_N" ]; then
  echo -e "  ${GREEN}✓ VERIFIED: Scaling removed the need for scaffolding.${NC}"
  echo -e "  ${GREEN}  460K Bare model matches or exceeds Assisted performance.${NC}"
else
  echo -e "  ${RED}✗ FAILED: 460K Bare is weaker than Assisted.${NC}"
  echo -e "  ${RED}  Scaffolding is still providing essential logic.${NC}"
fi

echo -e "${BOLD}Scaffolding Verdict:${NC}"
if [ "$ASST_64_SOLVED_N" -gt "$BARE_64_SOLVED_N" ]; then
  echo -e "  ${GREEN}✓ VERIFIED: Scaffolding acts as a 'capacity bridge' for 64K.${NC}"
  echo -e "  ${GREEN}  64K model requires the oscillation breaker to solve puzzles (+$(( ASST_64_SOLVED_N - BARE_64_SOLVED_N )) solved)${NC}"
else
  echo -e "  ${YELLOW}= INCONCLUSIVE: Scaffolding didn't help the 64K model much.${NC}"
fi

echo
echo -e "${CYAN}Results saved to: ${RESULTS_DIR}/${NC}"
echo "  standard_output.txt  — full standard model output"
echo "  combined_output.txt  — full combined model output"
echo
echo -e "${BOLD}Trace file: ${BUILD_DIR}/puzzle8_reasoning_traces.txt${NC}"
echo -e "${BOLD}Enriched corpus: ${BUILD_DIR}/puzzle8_mover_enriched.txt${NC}"
echo
echo "============================================================================"
