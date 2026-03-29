#!/bin/bash
#SBATCH --job-name=moran_I_perm_1000
#SBATCH --partition=norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00

set -euo pipefail

module load intel/2024.0.1.46

INPUT="${SIGDISCOV_INPUT:-/data/parks34/projects/0sigdiscov/moran_i/datasets/1_vst.tsv}"
CELLTYPE="${SIGDISCOV_CELLTYPE:-/data/parks34/projects/0sigdiscov/moran_i/datasets/1_SpaCET_res_T.csv}"

# nperm = 10
./build/morans_i_mkl -i "$INPUT" -o test -r 3 -s 0 \
  --analysis-mode residual --celltype-file "$CELLTYPE" --celltype-format deconv \
  --run-perm --num-perm 10 --perm-out-z --perm-out-p --perm-seed 42

# nperm = 100
./build/morans_i_mkl -i "$INPUT" -o test -r 3 -s 0 \
  --analysis-mode residual --celltype-file "$CELLTYPE" --celltype-format deconv \
  --run-perm --num-perm 100 --perm-out-z --perm-out-p --perm-seed 42

# nperm = 1000
./build/morans_i_mkl -i "$INPUT" -o test -r 3 -s 0 \
  --analysis-mode residual --celltype-file "$CELLTYPE" --celltype-format deconv \
  --run-perm --num-perm 1000 --perm-out-z --perm-out-p --perm-seed 42
