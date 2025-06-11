#!/bin/bash
#SBATCH --job-name=moran_I_perm_1000
#SBATCH --partition=norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00

module load intel/2024.0.1.46

# nperm = 10
#./morans_i_mkl -i ./dataset/visium/1_vst.tsv -o ./1_vst_perm10 -r 3 -s 0 --run-perm --n-perm 10 --perm-seed 42 --perm-output-zscores --perm-output-pvalues

./morans_i_mkl -i /data/parks34/projects/0sigdiscov/moran_i/datasets/1_vst.tsv -o test -r 3 -s 0 --analysis-mode residual --celltype-file /data/parks34/projects/0sigdiscov/moran_i/datasets/1_SpaCET_res_T.csv --celltype-format deconv --run-perm --num-perm 10 --perm-out-z --perm-out-p --perm-seed 42

# nperm = 100
#./morans_i_mkl -i ./dataset/visium/1_vst.tsv -o ./1_vst_perm100 -r 3 -s 0 --run-perm --n-perm 100 --perm-seed 42 --perm-output-zscores --perm-output-pvalues

./morans_i_mkl -i /data/parks34/projects/0sigdiscov/moran_i/datasets/1_vst.tsv -o test -r 3 -s 0 --analysis-mode residual --celltype-file /data/parks34/projects/0sigdiscov/moran_i/datasets/1_SpaCET_res_T.csv --celltype-format deconv --run-perm --num-perm 100 --perm-out-z --perm-out-p --perm-seed 42

# nperm = 1000
#./morans_i_mkl -i ./dataset/visium/1_vst.tsv -o ./1_vst_perm1000 -r 3 -s 0 --run-perm --n-perm 1000 --perm-seed 42 --perm-output-zscores --perm-output-pvalues
./morans_i_mkl -i /data/parks34/projects/0sigdiscov/moran_i/datasets/1_vst.tsv -o test -r 3 -s 0 --analysis-mode residual --celltype-file /data/parks34/projects/0sigdiscov/moran_i/datasets/1_SpaCET_res_T.csv --celltype-format deconv --run-perm --num-perm 100 --perm-out-z --perm-out-p --perm-seed 42

