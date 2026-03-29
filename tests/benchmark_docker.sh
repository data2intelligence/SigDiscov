#!/bin/bash
#SBATCH --job-name=sigdiscov_bench
#SBATCH --partition=norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=lscratch:50
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# Configurable via environment variables
PROJECT_DIR="${SIGDISCOV_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$PROJECT_DIR"

INPUT="${SIGDISCOV_INPUT:-/data/parks34/projects/0sigdiscov/archive/moran_i/datasets/visium/vst/1_vst.tsv}"
EXPECTED="${SIGDISCOV_EXPECTED:-/data/parks34/projects/0sigdiscov/archive/moran_i/datasets/1_spatial_sig_vst_inhouse_s0_r3.tsv}"
OUTPUT_DIR="${PROJECT_DIR}/tests/output"
mkdir -p "$OUTPUT_DIR"

# Use local scratch for Singularity cache (avoids home quota issues)
export SINGULARITY_CACHEDIR="/lscratch/${SLURM_JOB_ID}"
export SINGULARITY_TMPDIR="/lscratch/${SLURM_JOB_ID}"

echo "============================================"
echo "SigDiscov Benchmark: Native MKL vs Docker"
echo "Dataset: 1_vst.tsv (19729 genes x 3813 spots)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: 8"
echo "Singularity cache: $SINGULARITY_CACHEDIR"
echo "============================================"
echo ""

# -----------------------------------------------
# 1. Native MKL build
# -----------------------------------------------
echo "=== Test 1: Native Intel MKL build ==="
module load intel/2024.0.1.46 || { echo "ERROR: Intel module not available"; exit 1; }
make clean && make -j8
echo ""

echo "Running native MKL..."
time ./morans_i_mkl \
    -i "$INPUT" \
    -o "${OUTPUT_DIR}/bench_native" \
    -p 0 -r 3 -s 0 -b 1 -g 1 -t 8

NATIVE_OUTPUT="${OUTPUT_DIR}/bench_native_all_pairs_moran_i_raw.tsv"
echo ""

# -----------------------------------------------
# 2. Docker/Singularity OpenBLAS image
# -----------------------------------------------
echo "=== Test 2: Docker (OpenBLAS) via Singularity ==="
module load singularity/4.3.7 || { echo "ERROR: Singularity module not available"; exit 1; }

SIF_OPENBLAS="/lscratch/${SLURM_JOB_ID}/sigdiscov_openblas.sif"
echo "Pulling OpenBLAS image to lscratch..."
singularity pull "$SIF_OPENBLAS" docker://psychemistz/sigdiscov:latest

echo "Running OpenBLAS container..."
time singularity exec --bind /data,/vf "$SIF_OPENBLAS" \
    morans_i_mkl \
    -i "$INPUT" \
    -o "${OUTPUT_DIR}/bench_openblas" \
    -p 0 -r 3 -s 0 -b 1 -g 1 -t 8

OPENBLAS_OUTPUT="${OUTPUT_DIR}/bench_openblas_all_pairs_moran_i_raw.tsv"
echo ""

# -----------------------------------------------
# 3. Docker/Singularity MKL image
# -----------------------------------------------
echo "=== Test 3: Docker (MKL) via Singularity ==="

SIF_MKL="/lscratch/${SLURM_JOB_ID}/sigdiscov_mkl.sif"
echo "Pulling MKL image to lscratch..."
singularity pull "$SIF_MKL" docker://psychemistz/sigdiscov:latest-mkl

echo "Running MKL container..."
time singularity exec --bind /data,/vf "$SIF_MKL" \
    morans_i_mkl \
    -i "$INPUT" \
    -o "${OUTPUT_DIR}/bench_mkl_docker" \
    -p 0 -r 3 -s 0 -b 1 -g 1 -t 8

MKL_DOCKER_OUTPUT="${OUTPUT_DIR}/bench_mkl_docker_all_pairs_moran_i_raw.tsv"
echo ""

# -----------------------------------------------
# 4. Compare all outputs against expected
# -----------------------------------------------
echo "=== Output Comparison ==="

for label_file in "native:${NATIVE_OUTPUT}" "openblas:${OPENBLAS_OUTPUT}" "mkl_docker:${MKL_DOCKER_OUTPUT}"; do
    label="${label_file%%:*}"
    file="${label_file#*:}"

    if [ ! -f "$file" ]; then
        echo "  $label: OUTPUT MISSING ($file)"
        continue
    fi

    python3 -c "
import numpy as np, sys
tol = 1e-6
mx = 0.0; total = 0; bad = 0
with open('$EXPECTED') as ef, open('$file') as af:
    for el, al in zip(ef, af):
        e = np.array(el.strip().split('\t'), dtype=np.float64)
        a = np.array(al.strip().split('\t'), dtype=np.float64)
        d = np.abs(e - a)
        md = float(d.max())
        if md > mx: mx = md
        total += len(e)
        bad += int((d > tol).sum())
s = 'PASS' if bad == 0 else 'FAIL'
print(f'  $label: {s} ({total} values, max_diff={mx:.2e})')
"
done

echo ""
echo "=== Benchmark Complete ==="
