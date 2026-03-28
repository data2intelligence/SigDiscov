# Moran's I Calculator for Spatial Transcriptomics

**Version 1.3.0**

Optimized Moran's I spatial autocorrelation for spatial transcriptomics data. Uses Intel MKL (or OpenBLAS) for high-performance sparse matrix operations with OpenMP parallelization.

## Features

- **Standard Moran's I**: Single-gene, pairwise, and all-vs-all modes
- **Residual Moran's I**: Cell type-corrected spatial autocorrelation via ridge regression
- **Permutation testing**: Z-scores and p-values for statistical significance
- **Platform support**: Visium, ST, single-cell, and custom weight matrices
- **Performance**: MKL sparse CSR operations, OpenMP threading, VML vectorization

## Quick Start

```bash
# Build (Intel MKL)
source /opt/intel/oneapi/setvars.sh
make

# Build (GCC + OpenBLAS)
make CC=gcc USE_OPENBLAS=1

# Run
./morans_i_mkl -i expression.tsv -o results -r 3 -p 0 -b 1 -g 1

# With permutation testing
./morans_i_mkl -i expression.tsv -o results --run-perm --num-perm 1000

# Residual analysis (cell type corrected)
./morans_i_mkl -i expression.tsv -o results \
  --analysis-mode residual \
  --celltype-file celltypes.csv \
  --celltype-format deconv

# Built-in test
./morans_i_mkl --run-toy-example -o toy_test
```

## Docker (easiest)

```bash
# Run directly (works on Linux, macOS, Windows)
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/results -r 3 -p 0 -b 1 -g 1

# Intel MKL version (x86_64 only, best performance)
docker run --rm -v $(pwd):/data psychemistz/sigdiscov:latest-mkl \
  -i /data/expression.tsv -o /data/results -r 3 -p 0 -b 1 -g 1
```

## Requirements (building from source)

| Build | Requirements |
|-------|-------------|
| Docker (recommended) | Docker Desktop or Docker Engine |
| Intel MKL (HPC) | Intel oneAPI Base Toolkit + MKL |
| GCC + OpenBLAS (portable) | `libopenblas-dev`, `liblapacke-dev` |
| Python tool (optional) | `pip install -r requirements.txt` |

## Input Format

TSV or CSV (auto-detected). Genes as rows, spots/cells as columns:

```
Gene    1x1   1x2   2x1   2x2
Gene1   0.5   1.2   0.3   1.5
Gene2   2.1   1.8   1.4   0.6
```

For single-cell data, provide a coordinate file with `-c`:

```
cell_ID,sdimx,sdimy
cell_1,10.5,20.3
cell_2,15.8,18.7
```

## Documentation

See [`docs/`](docs/) for detailed documentation:

- [**Usage Reference**](docs/usage.md) -- all command-line options, calculation modes, and output formats
- [**Input Formats**](docs/input-formats.md) -- expression data, cell type data, custom weight matrices
- [**Mathematical Details**](docs/math.md) -- formulations for univariate, bivariate, and residual Moran's I
- [**Weight Matrix Generator**](docs/weight-generator.md) -- Python tool for custom weight matrices
- [**Architecture**](docs/architecture.md) -- source code modules and build system

## Testing

```bash
# SLURM-managed HPC
sbatch tests/run_tests_slurm.sh

# Direct
./tests/run_tests.sh ./morans_i_mkl
```

## License

MIT License -- see [LICENSE](LICENSE) for details.

## Citation

Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, Kenneth Aldape, Lalage Wakefield, Peng Jiang. Inference of secreted protein activities in intercellular communication. [[Link](https://github.com/data2intelligence/SecAct)]
