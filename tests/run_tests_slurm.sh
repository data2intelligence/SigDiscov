#!/bin/bash
#SBATCH --job-name=sigdiscov_tests
#SBATCH --partition=norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=tests/output/slurm_test_%j.log

# SLURM wrapper to build and run the regression test suite.
#
# Usage:
#   sbatch tests/run_tests_slurm.sh [--generate-expected]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load Intel oneAPI
module load intel/2024.0.1.46

# Build
echo "=== Building ==="
make clean && make
echo ""

# Run tests
if [ "${1:-}" = "--generate-expected" ]; then
    echo "=== Generating expected outputs ==="
    bash tests/run_tests.sh --generate-expected ./morans_i_mkl
else
    echo "=== Running tests ==="
    bash tests/run_tests.sh ./morans_i_mkl
fi
