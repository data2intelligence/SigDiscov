# Moran's I Calculator for Spatial Transcriptomics

**Version 1.3.0**

Optimized Moran's I spatial autocorrelation for spatial transcriptomics data. Uses Intel MKL (or OpenBLAS) for high-performance sparse matrix operations with OpenMP parallelization.

## Features

- **Standard Moran's I**: Single-gene, pairwise, and all-vs-all modes
- **Residual Moran's I**: Cell type-corrected spatial autocorrelation via ridge regression
- **Permutation testing**: Z-scores and p-values for statistical significance
- **Platform support**: Visium, ST, single-cell, and custom weight matrices
- **Performance**: MKL sparse CSR operations, OpenMP threading, VML vectorization
- **Cross-platform**: Docker images for Linux, macOS, and Windows (+ Singularity/HPC)

## Quick Start (Docker)

No installation needed. Works on Linux, macOS (Intel + Apple Silicon), and Windows:

```bash
# OpenBLAS version (all platforms)
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/results -r 3 -p 0 -b 1 -g 1

# Intel MKL version (x86_64 only, best performance)
docker run --rm -v $(pwd):/data psychemistz/sigdiscov:latest-mkl \
  -i /data/expression.tsv -o /data/results -r 3 -p 0 -b 1 -g 1

# Residual analysis
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/results \
  --analysis-mode residual --celltype-file /data/celltypes.csv --celltype-format deconv

# With permutation testing
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/results --run-perm --num-perm 1000

# Built-in test
docker run --rm psychemistz/sigdiscov --run-toy-example -o /tmp/test
```

### Singularity (HPC clusters)

For SLURM-managed clusters like Biowulf that use Singularity:

```bash
# Pull image (once)
singularity pull sigdiscov.sif docker://psychemistz/sigdiscov:latest

# Run
singularity exec --bind /data sigdiscov.sif \
  morans_i_mkl -i /data/expression.tsv -o /data/results -r 3 -p 0 -b 1 -g 1
```

### Available Images

| Image | Arch | Best for |
|-------|------|----------|
| `psychemistz/sigdiscov:latest` | amd64, arm64 | All platforms (macOS Apple Silicon, Linux, Windows) |
| `psychemistz/sigdiscov:latest-mkl` | amd64 only | HPC / Intel systems (best performance) |

## Quick Start (Build from Source)

```bash
# Intel MKL (recommended for HPC)
source /opt/intel/oneapi/setvars.sh
make

# GCC + OpenBLAS (portable)
make CC=gcc USE_OPENBLAS=1

# Run (binary is in build/)
./build/morans_i_mkl -i expression.tsv -o results -r 3 -p 0 -b 1 -g 1
```

## Performance

Benchmark on Biowulf (19,729 genes x 3,813 Visium spots, 8 threads):

| Method | Time | vs Native |
|--------|------|-----------|
| Native MKL (icx compiler) | 64s | baseline |
| Docker MKL (Singularity) | 64s | +0% |
| Docker OpenBLAS (Singularity) | 73s | +14% |

All methods produce identical results (194M values, max difference 5e-7).

## Requirements (building from source)

| Build | Requirements |
|-------|-------------|
| Docker (easiest) | Docker Desktop or Docker Engine |
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
# Docker
docker run --rm psychemistz/sigdiscov --run-toy-example -o /tmp/test

# SLURM-managed HPC
sbatch tests/run_tests_slurm.sh

# Direct
./tests/run_tests.sh ./build/morans_i_mkl
```

## License

MIT License -- see [LICENSE](LICENSE) for details.

## Citation

Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, George Zaki, Kenneth Aldape, Lalage Wakefield, Peng Jiang. Inference of secreted protein activities in intercellular communication. *Nature Methods*, 2026. [[Full Text](https://github.com/data2intelligence/SecAct)]
