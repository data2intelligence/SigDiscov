# SigDiscov: Moran's I for Spatial Transcriptomics

**Version 1.3.0**

Optimized Moran's I spatial autocorrelation for spatial transcriptomics data. Uses Intel MKL (or OpenBLAS) for high-performance sparse matrix operations with OpenMP parallelization.

**Features**: Standard Moran's I (single-gene, pairwise, all-vs-all) | Residual Moran's I (cell type-corrected via ridge regression) | Permutation testing (Z-scores and p-values) | Visium, ST, single-cell, and custom weight matrices | Cross-platform via Docker

---

## Table of Contents

- [Getting Started (Docker -- Recommended)](#getting-started-docker----recommended)
- [Getting Started (Build from Source)](#getting-started-build-from-source)
- [Running on Your Own Data](#running-on-your-own-data)
- [Singularity / HPC Clusters](#singularity--hpc-clusters)
- [Input Format Reference](#input-format-reference)
- [Full Documentation](#full-documentation)
- [Testing](#testing)
- [Performance](#performance)
- [Citation](#citation)

---

## Getting Started (Docker -- Recommended)

Docker is the easiest way to run SigDiscov. No compiler, no MKL install, no dependency management. Works on Linux, macOS (Intel + Apple Silicon), and Windows.

### Step 1: Install Docker

If you don't have Docker installed:

| Platform | Install |
|----------|---------|
| **macOS** | [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/) |
| **Windows** | [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) (requires WSL2) |
| **Linux** | [Docker Engine](https://docs.docker.com/engine/install/) (e.g., `sudo apt-get install docker.io` on Ubuntu) |

After installation, verify Docker is running:

```bash
docker --version
# Docker version 24.x or later
```

### Step 2: Pull the SigDiscov image

```bash
# OpenBLAS version (works on all platforms including Apple Silicon)
docker pull psychemistz/sigdiscov:latest

# Intel MKL version (x86_64 only -- best performance on Intel/AMD CPUs)
docker pull psychemistz/sigdiscov:latest-mkl
```

| Image | Architecture | Best for |
|-------|-------------|----------|
| `psychemistz/sigdiscov:latest` | amd64, arm64 | All platforms (macOS Apple Silicon, Linux, Windows) |
| `psychemistz/sigdiscov:latest-mkl` | amd64 only | HPC / Intel/AMD systems (best performance) |

### Step 3: Quick smoke test (built-in toy example)

SigDiscov includes a built-in test dataset (5x5 grid, 5 synthetic genes) that requires no input files:

```bash
docker run --rm psychemistz/sigdiscov --run-toy-example -o /tmp/test
```

If this runs without errors, Docker and SigDiscov are working.

### Step 4: Verify on real Visium data

The repository includes a real Visium dataset subset (50 genes x 3,813 spots) in `examples/` along with the expected output extracted from a full computation. Run SigDiscov on it and verify the result matches:

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/data2intelligence/SigDiscov.git
cd SigDiscov

# Run SigDiscov on the example data
docker run --rm -v $(pwd)/examples:/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/my_output -r 3 -p 0 -b 1 -g 1 -s 0

# Compare your output against the expected result
diff examples/expected_output.tsv examples/my_output_all_pairs_moran_i_raw.tsv
```

If `diff` produces no output, your installation reproduces the expected result exactly.

Or use the verification script (also checks numerical tolerance for cross-platform differences):

```bash
bash examples/run_example.sh docker
```

### Step 5: Run on your own data

The `-v` flag mounts a directory from your computer into the Docker container. Here is how it works:

```bash
docker run --rm -v /path/on/your/computer:/data psychemistz/sigdiscov \
  -i /data/your_expression_file.tsv -o /data/results
```

**What `-v /path/on/your/computer:/data` means:**
- **Left side** (`/path/on/your/computer`): a directory on your laptop/server that contains your input files
- **Right side** (`/data`): where that directory appears *inside* the Docker container
- So if your file is at `/Users/jane/project/expr.tsv`, the command becomes:
  ```bash
  docker run --rm -v /Users/jane/project:/data psychemistz/sigdiscov \
    -i /data/expr.tsv -o /data/results
  ```
- Output files will be written back to `/Users/jane/project/` (prefixed with `results`)

**Using `$(pwd)` shortcut** (mounts the current directory):

```bash
cd /Users/jane/project        # directory containing your expression file
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expr.tsv -o /data/results -r 3 -p 0
```

See [Running on Your Own Data](#running-on-your-own-data) for full examples.

---

## Getting Started (Build from Source)

Building from source gives you a native binary without Docker overhead. There are two options:

### Option A: GCC + OpenBLAS (easiest, works on most Linux/macOS)

**Install dependencies:**

```bash
# Ubuntu / Debian
sudo apt-get install gcc make libopenblas-dev liblapacke-dev

# macOS (with Homebrew)
brew install gcc openblas lapack
```

**Build:**

```bash
git clone https://github.com/data2intelligence/SigDiscov.git
cd SigDiscov
make CC=gcc USE_OPENBLAS=1
```

The binary will be at `./build/morans_i_mkl`.

**Verify:**

```bash
# Smoke test
./build/morans_i_mkl --run-toy-example -o /tmp/test

# Full verification against real Visium data
bash examples/run_example.sh native ./build/morans_i_mkl
```

### Option B: Intel icx + MKL (best performance, recommended for HPC)

This requires Intel oneAPI Base Toolkit, which includes the `icx` compiler and MKL libraries.

**Install Intel oneAPI (free):**

1. Download from [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
2. Follow the installer instructions for your platform
3. The Docker images use oneAPI version **2024.0.1** -- any 2024.x release should work

**Build:**

```bash
# Activate the Intel oneAPI environment (REQUIRED before every build/run)
source /opt/intel/oneapi/setvars.sh

git clone https://github.com/data2intelligence/SigDiscov.git
cd SigDiscov
make
```

The binary will be at `./build/morans_i_mkl`.

**Verify:**

```bash
# Smoke test
./build/morans_i_mkl --run-toy-example -o /tmp/test

# Full verification against real Visium data
bash examples/run_example.sh native ./build/morans_i_mkl
```

**Troubleshooting:**

| Error | Fix |
|-------|-----|
| `icx: command not found` | Run `source /opt/intel/oneapi/setvars.sh` first |
| `MKLROOT not found` | Same fix -- `setvars.sh` sets this variable |
| `mkl.h: No such file` | Ensure MKL component was selected during oneAPI install |

---

## Running on Your Own Data

### Preparing your input

SigDiscov expects a **gene-by-spot expression matrix** in TSV or CSV format:

- Rows = genes, columns = spots (or cells)
- First row = header with spot/cell identifiers
- First column = gene names

**Visium/ST data**: column headers should encode grid coordinates as `ROWxCOL`:

```
Gene    1x1     1x2     1x3     2x1     2x2
TP53    0.51    1.23    0.82    0.30    1.52
BRCA1   2.10    1.80    0.93    1.41    0.64
```

**Single-cell data**: column headers are cell IDs (also provide a coordinate file with `-c`):

```
Gene    cell_1  cell_2  cell_3
TP53    0.51    1.23    0.82
BRCA1   2.10    1.80    0.93
```

For full input format details, see [docs/input-formats.md](docs/input-formats.md).

### Example commands

**Basic Moran's I on Visium data** (pairwise, all gene pairs):

```bash
# Docker
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/results -r 3 -p 0 -b 1 -g 1

# Native binary
./build/morans_i_mkl -i expression.tsv -o results -r 3 -p 0 -b 1 -g 1
```

Key flags: `-r 3` = neighbor radius 3, `-p 0` = Visium platform, `-b 1 -g 1` = all pairwise comparisons.

**Single-gene Moran's I** (faster, one value per gene):

```bash
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/results -b 0 -p 0 -r 4
```

**Residual Moran's I** (cell type-corrected):

```bash
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/results \
  --analysis-mode residual --celltype-file /data/celltypes.csv --celltype-format deconv
```

**With permutation testing** (adds Z-scores and p-values):

```bash
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/expression.tsv -o /data/results \
  --run-perm --num-perm 1000 --perm-seed 42
```

**Single-cell data with coordinates**:

```bash
docker run --rm -v $(pwd):/data psychemistz/sigdiscov \
  -i /data/sc_expression.tsv -o /data/results -p 2 -c /data/coordinates.csv
```

For the full list of options, see [docs/usage.md](docs/usage.md).

---

## Singularity / HPC Clusters

Most HPC clusters (e.g., Biowulf, SLURM-managed systems) use Singularity instead of Docker. Singularity can pull Docker images directly.

```bash
# Pull the image (once)
singularity pull sigdiscov.sif docker://psychemistz/sigdiscov:latest

# Or the MKL version for Intel/AMD nodes
singularity pull sigdiscov-mkl.sif docker://psychemistz/sigdiscov:latest-mkl

# Run (--bind mounts your data directory)
singularity exec --bind /your/data/dir sigdiscov.sif \
  morans_i_mkl -i /your/data/dir/expression.tsv -o /your/data/dir/results -r 3 -p 0

# SLURM example (submit as a job)
sbatch --cpus-per-task=8 --mem=32G --wrap="singularity exec --bind /your/data/dir sigdiscov.sif \
  morans_i_mkl -i /your/data/dir/expression.tsv -o /your/data/dir/results -r 3 -t 8"
```

---

## Input Format Reference

| Input | Format | Required? |
|-------|--------|-----------|
| Expression matrix | TSV or CSV (genes x spots) | Yes (`-i`) |
| Coordinate file | CSV with cell_ID, sdimx, sdimy | Only for single-cell (`-p 2 -c`) |
| Cell type data | CSV, deconv or single-cell format | Only for residual mode |
| Custom weights | Dense TSV, sparse COO, or sparse TSV | Only for custom mode (`-w`) |

See [docs/input-formats.md](docs/input-formats.md) for detailed examples of each format.

---

## Full Documentation

| Document | Description |
|----------|-------------|
| [Usage Reference](docs/usage.md) | All command-line options, calculation modes, and output formats |
| [Input Formats](docs/input-formats.md) | Expression data, cell type data, custom weight matrices |
| [Mathematical Details](docs/math.md) | Formulations for univariate, bivariate, and residual Moran's I |
| [Weight Matrix Generator](docs/weight-generator.md) | Python tool for custom weight matrices |
| [Architecture](docs/architecture.md) | Source code modules and build system |

---

## Testing

```bash
# Quickest test: built-in toy example (no input files needed)
docker run --rm psychemistz/sigdiscov --run-toy-example -o /tmp/test

# Verify on real Visium data (50 genes x 3,813 spots)
bash examples/run_example.sh docker            # Docker
bash examples/run_example.sh native ./build/morans_i_mkl  # native build

# Full regression suite (native build)
./tests/run_tests.sh ./build/morans_i_mkl

# Full regression suite (SLURM cluster)
sbatch tests/run_tests_slurm.sh
```

---

## Example Data

The `examples/` directory contains real Visium spatial transcriptomics data for testing:

| File | Description | Size |
|------|-------------|------|
| `expression.tsv` | 50 genes x 3,813 spots (subset of full dataset) | 1.3 MB |
| `expected_output.tsv` | Pairwise Moran's I for those 50 genes (lower triangle) | 14 KB |
| `run_example.sh` | Verification script (Docker, Singularity, or native) | 1 KB |

The expected output was extracted from a full computation on 19,729 genes. The full dataset (481 MB) is available from the [GitHub Releases](https://github.com/data2intelligence/SigDiscov/releases) page.

---

## Performance

Benchmark on Biowulf (19,729 genes x 3,813 Visium spots, 8 threads):

| Method | Time | vs Native |
|--------|------|-----------|
| Native MKL (icx compiler) | 64s | baseline |
| Docker MKL (Singularity) | 64s | +0% |
| Docker OpenBLAS (Singularity) | 73s | +14% |

All methods produce identical results (194M values, max difference 5e-7).

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

## Citation

Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, George Zaki, Kenneth Aldape, Lalage Wakefield, Peng Jiang. Inference of secreted protein activities in intercellular communication. *Nature Methods*, 2026. [[Full Text](https://github.com/data2intelligence/SecAct)]
