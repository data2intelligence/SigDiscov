#!/bin/bash
# SigDiscov Example: Run on real Visium data and verify output
#
# This script runs SigDiscov on a 50-gene subset of a real Visium dataset
# (3,813 spots) and verifies the output matches the expected result.
#
# Usage:
#   Docker:      bash examples/run_example.sh docker
#   Singularity: bash examples/run_example.sh singularity sigdiscov.sif
#   Native:      bash examples/run_example.sh native ./build/morans_i_mkl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="${SCRIPT_DIR}/expression.tsv"
EXPECTED="${SCRIPT_DIR}/expected_output.tsv"
OUTPUT="${SCRIPT_DIR}/test_output"

MODE="${1:-docker}"
shift || true

# Clean previous test output
rm -f "${OUTPUT}"*

echo "=== SigDiscov Example Verification ==="
echo "Input:    ${INPUT} (50 genes x 3,813 Visium spots)"
echo "Expected: ${EXPECTED}"
echo ""

case "${MODE}" in
  docker)
    echo "Running with Docker..."
    docker run --rm -v "${SCRIPT_DIR}:/data" psychemistz/sigdiscov \
      -i /data/expression.tsv -o /data/test_output -r 3 -p 0 -b 1 -g 1 -s 0
    ;;
  singularity)
    SIF="${1:?Usage: run_example.sh singularity <path-to-sif>}"
    echo "Running with Singularity..."
    singularity exec --bind "${SCRIPT_DIR}" "${SIF}" \
      morans_i_mkl -i "${INPUT}" -o "${OUTPUT}" -r 3 -p 0 -b 1 -g 1 -s 0
    ;;
  native)
    BIN="${1:?Usage: run_example.sh native <path-to-binary>}"
    echo "Running with native binary..."
    "${BIN}" -i "${INPUT}" -o "${OUTPUT}" -r 3 -p 0 -b 1 -g 1 -s 0
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: run_example.sh {docker|singularity|native} [extra-arg]"
    exit 1
    ;;
esac

echo ""
echo "=== Verifying output ==="

# Find the output file (the tool appends a suffix like _all_pairs_moran_i_raw.tsv)
ACTUAL=""
for f in "${OUTPUT}"_all_pairs_moran_i_raw.tsv "${OUTPUT}.tsv" "${OUTPUT}"; do
  [ -f "$f" ] && ACTUAL="$f" && break
done

if [ -z "${ACTUAL}" ]; then
  echo "FAIL: No output file produced."
  ls -la "${SCRIPT_DIR}"/test_output* 2>/dev/null || true
  exit 1
fi

if diff -q "${EXPECTED}" "${ACTUAL}" > /dev/null 2>&1; then
  echo "PASS: Output matches expected result exactly."
else
  echo "Output differs from expected. Checking numerical tolerance..."
  # Allow tiny floating-point differences (< 1e-6)
  paste "${EXPECTED}" "${ACTUAL}" | awk -F'\t' '{
    for (i = 1; i <= NF/2; i++) {
      diff = $i - $(i + NF/2)
      if (diff < 0) diff = -diff
      if (diff > 1e-6) { print "FAIL: line " NR ", col " i ": expected " $i " got " $(i+NF/2) " (diff=" diff ")"; fail=1 }
    }
  } END { if (!fail) print "PASS: Output matches within tolerance (1e-6)." }'
fi

# Clean up
rm -f "${OUTPUT}"*
