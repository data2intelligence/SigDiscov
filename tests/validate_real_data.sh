#!/bin/bash
#SBATCH --job-name=sigdiscov_validate
#SBATCH --partition=norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# Configurable via environment variables
# SLURM copies scripts to a temp dir, so BASH_SOURCE won't resolve to the repo.
# Default to SLURM_SUBMIT_DIR (the directory where sbatch was invoked).
PROJECT_DIR="${SIGDISCOV_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$PROJECT_DIR"

INPUT="${SIGDISCOV_INPUT:-/data/parks34/projects/0sigdiscov/archive/moran_i/datasets/visium/vst/1_vst.tsv}"
EXPECTED="${SIGDISCOV_EXPECTED:-/data/parks34/projects/0sigdiscov/archive/moran_i/datasets/1_spatial_sig_vst_inhouse_s0_r3.tsv}"
OUTPUT_DIR="${PROJECT_DIR}/tests/output"
OUTPUT_PREFIX="${OUTPUT_DIR}/validate_1_vst"

mkdir -p "$OUTPUT_DIR"

module load intel/2024.0.1.46 || { echo "ERROR: Intel module not available"; exit 1; }

echo "=== Building ==="
make clean && make
echo ""

echo "=== Running Moran's I (Visium, -s 0 -r 3, all pairs) ==="
time ./build/morans_i_mkl \
    -i "$INPUT" \
    -o "$OUTPUT_PREFIX" \
    -p 0 -r 3 -s 0 \
    -b 1 -g 1 \
    -t 8

echo ""

ACTUAL="${OUTPUT_PREFIX}_all_pairs_moran_i_raw.tsv"

if [ ! -f "$ACTUAL" ]; then
    echo "FAIL: Output file not created: $ACTUAL"
    exit 1
fi

echo "=== Comparing output ==="
python3 "${PROJECT_DIR}/tests/compare_tsv.py" "$EXPECTED" "$ACTUAL"

echo ""
echo "=== Validation complete ==="
