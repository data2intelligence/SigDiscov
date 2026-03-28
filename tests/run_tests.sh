#!/bin/bash
# Automated regression test suite for SigDiscov Moran's I
#
# Usage:
#   Direct:  ./tests/run_tests.sh [path/to/morans_i_mkl]
#   SLURM:   sbatch tests/run_tests_slurm.sh
#
# Prerequisites: morans_i_mkl binary must be built first.
# Tests use the built-in toy example with fixed seeds for reproducibility.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="${1:-${PROJECT_DIR}/morans_i_mkl}"
OUTPUT_DIR="${SCRIPT_DIR}/output"
EXPECTED_DIR="${SCRIPT_DIR}/expected"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

cleanup() {
    if [ "$FAIL" -eq 0 ]; then
        rm -rf "$OUTPUT_DIR"
    else
        echo "Test output preserved in $OUTPUT_DIR for inspection"
    fi
}

compare_file() {
    local test_name="$1"
    local actual="$2"
    local expected="$3"

    if [ ! -f "$actual" ]; then
        echo -e "${RED}FAIL${NC}: ${test_name} - output file not created: ${actual}"
        ((FAIL++))
        return
    fi

    if [ ! -f "$expected" ]; then
        echo -e "${RED}SKIP${NC}: ${test_name} - no expected file: ${expected}"
        echo "  (Run with --generate-expected to create golden files)"
        return
    fi

    # Compare with tolerance for floating point values
    if diff -q "$actual" "$expected" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}: ${test_name}"
        ((PASS++))
    else
        echo -e "${RED}FAIL${NC}: ${test_name}"
        echo "  Diff:"
        diff --brief "$actual" "$expected" || true
        diff -u "$expected" "$actual" | head -20
        ((FAIL++))
    fi
}

# Generate expected files mode
if [ "${1:-}" = "--generate-expected" ]; then
    BINARY="${2:-${PROJECT_DIR}/morans_i_mkl}"
    echo "=== Generating expected test outputs ==="
    echo "Binary: $BINARY"

    if [ ! -x "$BINARY" ]; then
        echo "Error: Binary not found or not executable: $BINARY"
        exit 1
    fi

    mkdir -p "$EXPECTED_DIR"

    # Test 1: Toy example without permutations
    echo "Generating: toy_basic..."
    "$BINARY" --run-toy-example -o "${EXPECTED_DIR}/toy_basic" -t 1

    # Test 2: Toy example with permutations (fixed seed)
    echo "Generating: toy_perm..."
    "$BINARY" --run-toy-example -o "${EXPECTED_DIR}/toy_perm" \
        --run-perm --num-perm 100 --perm-seed 42 \
        --perm-out-z --perm-out-p -t 1

    echo "=== Expected files generated in ${EXPECTED_DIR} ==="
    ls -la "$EXPECTED_DIR"/toy_*
    exit 0
fi

# Main test execution
echo "=== SigDiscov Regression Test Suite ==="
echo "Binary: $BINARY"
echo "Date: $(date)"

if [ ! -x "$BINARY" ]; then
    echo "Error: Binary not found or not executable: $BINARY"
    echo "Build first with: make"
    exit 1
fi

trap cleanup EXIT
mkdir -p "$OUTPUT_DIR"

# -------------------------------------------------------
# Test 1: Toy example - basic (no permutations)
# -------------------------------------------------------
echo ""
echo "--- Test 1: Toy example (basic) ---"
"$BINARY" --run-toy-example -o "${OUTPUT_DIR}/toy_basic" -t 1

for f in toy_basic_toy_2D_X_calc_Znorm.tsv \
         toy_basic_toy_2D_observed_I_full.tsv \
         toy_basic_toy_2D_theoretical_I_full.tsv; do
    compare_file "toy_basic/$f" "${OUTPUT_DIR}/${f}" "${EXPECTED_DIR}/${f}"
done

# -------------------------------------------------------
# Test 2: Toy example - with permutations (fixed seed)
# -------------------------------------------------------
echo ""
echo "--- Test 2: Toy example (permutations, seed=42) ---"
"$BINARY" --run-toy-example -o "${OUTPUT_DIR}/toy_perm" \
    --run-perm --num-perm 100 --perm-seed 42 \
    --perm-out-z --perm-out-p -t 1

for f in toy_perm_toy_2D_observed_I_full.tsv \
         toy_perm_zscores_lower_tri.tsv \
         toy_perm_pvalues_lower_tri.tsv; do
    compare_file "toy_perm/$f" "${OUTPUT_DIR}/${f}" "${EXPECTED_DIR}/${f}"
done

# -------------------------------------------------------
# Test 3: Verify toy example produces expected Moran's I signs
# -------------------------------------------------------
echo ""
echo "--- Test 3: Moran's I value sanity checks ---"

OBSERVED="${OUTPUT_DIR}/toy_basic_toy_2D_observed_I_full.tsv"
if [ -f "$OBSERVED" ]; then
    # Gene0 (row gradient) autocorrelation should be positive (> 0.5)
    gene0_auto=$(awk 'NR==2{print $2}' "$OBSERVED")
    if (( $(echo "$gene0_auto > 0.5" | bc -l) )); then
        echo -e "${GREEN}PASS${NC}: Gene0 autocorrelation ($gene0_auto > 0.5)"
        ((PASS++))
    else
        echo -e "${RED}FAIL${NC}: Gene0 autocorrelation ($gene0_auto <= 0.5, expected > 0.5)"
        ((FAIL++))
    fi

    # Gene3 (checkerboard) autocorrelation should be negative
    gene3_auto=$(awk 'NR==5{print $5}' "$OBSERVED")
    if (( $(echo "$gene3_auto < 0" | bc -l) )); then
        echo -e "${GREEN}PASS${NC}: Gene3 (checkerboard) autocorrelation ($gene3_auto < 0)"
        ((PASS++))
    else
        echo -e "${RED}FAIL${NC}: Gene3 (checkerboard) autocorrelation ($gene3_auto >= 0, expected < 0)"
        ((FAIL++))
    fi
else
    echo -e "${RED}FAIL${NC}: Cannot run sanity checks - observed file missing"
    ((FAIL+=2))
fi

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "=== Test Summary ==="
echo -e "Passed: ${GREEN}${PASS}${NC}"
echo -e "Failed: ${RED}${FAIL}${NC}"

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}TESTS FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}ALL TESTS PASSED${NC}"
    exit 0
fi
