# Usage Reference

## Basic Usage

```bash
./morans_i_mkl -i <input.tsv> -o <output_prefix> [OPTIONS]
```

## Required Arguments

| Flag | Description |
|------|-------------|
| `-i <file>` | Input data matrix (Genes x Spots/Cells) |
| `-o <prefix>` | Output file prefix |

## General Options

| Flag | Description | Default |
|------|-------------|---------|
| `-r <int>` | Maximum grid radius for neighbor search | 5 |
| `-p <int>` | Platform: 0=Visium, 1=ST, 2=Single Cell, 3=Custom Weights | 0 |
| `-b <0\|1>` | 0=Single-gene, 1=Pairwise | 1 |
| `-g <0\|1>` | 0=First gene vs all, 1=All gene pairs (if `-b 1`) | 1 |
| `-s <0\|1>` | Include self-comparison (w_ii) | 0 |
| `--row-normalize <0\|1>` | Row-normalize weight matrix | 0 |
| `-t <int>` | OpenMP threads | 4 |
| `-m <int>` | MKL threads | value of `-t` |
| `--sigma <float>` | Custom RBF kernel sigma | platform default |

## Custom Weight Matrix Options

| Flag | Description | Default |
|------|-------------|---------|
| `-w <file>` | Custom weight matrix file (sets platform to 3) | -- |
| `--weight-format <fmt>` | auto, dense, sparse_coo, sparse_tsv | auto |
| `--normalize-weights` | Divide weights by sum (S0) | off |

## Single-Cell Options

| Flag | Description | Default |
|------|-------------|---------|
| `-c <file>` | Coordinates file (required for `-p 2`) | -- |
| `--id-col <name>` | Cell ID column | cell_ID |
| `--x-col <name>` | X coordinate column | sdimx |
| `--y-col <name>` | Y coordinate column | sdimy |
| `--scale <float>` | Coordinate scaling factor | 100.0 |

## Residual Moran's I Options

| Flag | Description | Default |
|------|-------------|---------|
| `--analysis-mode <mode>` | standard or residual | standard |
| `--celltype-file <file>` | Cell type data file | -- |
| `--celltype-format <fmt>` | deconv or sc | sc |
| `--celltype-id-col <name>` | Cell ID column | cell_ID |
| `--celltype-type-col <name>` | Cell type column | cellType |
| `--celltype-x-col <name>` | X coordinate column | sdimx |
| `--celltype-y-col <name>` | Y coordinate column | sdimy |
| `--spot-id-col <name>` | Spot ID column (deconv format) | spot_id |
| `--include-intercept <0\|1>` | Include intercept in regression | 1 |
| `--regularization <float>` | Ridge regularization lambda | 0.0 |
| `--normalize-residuals <0\|1>` | Normalize residuals | 1 |

## Permutation Test Options

| Flag | Description | Default |
|------|-------------|---------|
| `--run-perm` | Enable permutation testing | off |
| `--num-perm <int>` | Number of permutations (implies `--run-perm`) | 1000 |
| `--perm-seed <int>` | RNG seed | system time |
| `--perm-out-z <0\|1>` | Output Z-scores | 1 |
| `--perm-out-p <0\|1>` | Output p-values | 1 |

## Other

| Flag | Description |
|------|-------------|
| `--run-toy-example` | Run built-in 5x5 grid test (requires `-o`) |
| `-h`, `--help` | Show help |

## Output Files

| Mode | File | Contents |
|------|------|----------|
| Single-gene (`-b 0`) | `<prefix>_single_gene_moran_i.tsv` | Gene, MoranI |
| First vs all (`-b 1 -g 0`) | `<prefix>_first_vs_all_moran_i.tsv` | Gene, MoranI_vs_FirstGene |
| All pairs (`-b 1 -g 1`) | `<prefix>_all_pairs_moran_i_raw.tsv` | Lower triangular Moran's I |
| Permutation Z-scores | `<prefix>_zscores_lower_tri.tsv` | Z-scores |
| Permutation p-values | `<prefix>_pvalues_lower_tri.tsv` | P-values |
| Residual Moran's I | `<prefix>_residual_morans_i_raw.tsv` | Residual Moran's I matrix |
| Regression coefficients | `<prefix>_regression_coefficients.tsv` | Cell type coefficients |
| Residual Z-scores | `<prefix>_residual_zscores_lower_tri.tsv` | Residual Z-scores |
| Residual p-values | `<prefix>_residual_pvalues_lower_tri.tsv` | Residual p-values |

## Examples

```bash
# Single-gene Moran's I for Visium
./morans_i_mkl -i visium.tsv -o out -b 0 -p 0 -r 4

# Pairwise, first gene vs all
./morans_i_mkl -i visium.tsv -o out -b 1 -g 0

# Single-cell with coordinates
./morans_i_mkl -i sc_expr.tsv -o out -p 2 -c coords.tsv

# Residual with deconvolution
./morans_i_mkl -i expr.tsv -o out --analysis-mode residual \
  --celltype-file deconv.csv --celltype-format deconv

# Residual with regularization and permutation
./morans_i_mkl -i expr.tsv -o out --analysis-mode residual \
  --celltype-file props.tsv --regularization 0.1 \
  --run-perm --num-perm 1000

# Custom weight matrix
python3 tools/make_custom_w.py -i expr.tsv -o weights -r 5 -p 0
./morans_i_mkl -i expr.tsv -o out -w weights_dense.tsv --weight-format dense

# Toy example with permutations
./morans_i_mkl --run-toy-example -o toy --num-perm 100 --perm-seed 42
```

## Error Codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | `MORANS_I_SUCCESS` | Success |
| -1 | `MORANS_I_ERROR_MEMORY` | Memory allocation failure |
| -2 | `MORANS_I_ERROR_FILE` | File access or parsing error |
| -3 | `MORANS_I_ERROR_PARAMETER` | Invalid parameter |
| -4 | `MORANS_I_ERROR_COMPUTATION` | Computation error |

## Performance Tips

- Increase `-r` cautiously -- it increases computation quadratically
- Use `-b 0` or `-b 1 -g 0` for large datasets before all-vs-all
- Reduce `--num-perm` for initial exploratory analyses
- Match `-t` to available CPU cores
- Use `--regularization` for numerical stability with many cell types
