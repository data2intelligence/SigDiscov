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
./morans_i_mkl -i ../sig_discovery/dataset/visium/1_vst.tsv -o ./1_vst_perm10 -r 3 -s 0 --run-perms --n-perms 10 --perm-seed 42 --perm-output-zscores --perm-output-pvalues

# nperm = 100
./morans_i_mkl -i ../sig_discovery/dataset/visium/1_vst.tsv -o ./1_vst_perm100 -r 3 -s 0 --run-perms --n-perms 100 --perm-seed 42 --perm-output-zscores --perm-output-pvalues

# nperm = 1000
./morans_i_mkl -i ../sig_discovery/dataset/visium/1_vst.tsv -o ./1_vst_perm1000 -r 3 -s 0 --run-perms --n-perms 1000 --perm-seed 42 --perm-output-zscores --perm-output-pvalues


