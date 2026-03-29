# Architecture

## Project Layout

```
.
├── src/                        # C source and headers
│   ├── main.c                  # CLI parsing, orchestration
│   ├── toy_example.c           # Built-in 5x5 grid toy example
│   ├── morans_i_utils.c        # Hash table, config, timing, validation
│   ├── morans_i_io_expression.c # Expression and coordinate I/O
│   ├── morans_i_io_celltype.c  # Cell type data reading and mapping
│   ├── morans_i_io_weights.c   # Weight matrix reading and validation
│   ├── morans_i_io_results.c   # All save_* output functions
│   ├── morans_i_core.c         # Z-normalization, Moran's I calculation
│   ├── morans_i_spatial.c      # RBF decay, distance, weight matrix
│   ├── morans_i_residual.c     # Regression, residual Moran's I
│   ├── morans_i_perm.c         # Standard permutation testing
│   ├── morans_i_memory.c       # All free_* functions
│   ├── morans_i_mkl.h          # Public API header
│   ├── morans_i_internal.h     # Internal shared helpers
│   └── openblas_compat.h       # MKL-to-OpenBLAS compatibility
├── tools/
│   └── make_custom_w.py    # Weight matrix generator
├── docker/
│   ├── Dockerfile          # Intel MKL image (x86_64)
│   └── Dockerfile.openblas # OpenBLAS image (multi-arch)
├── tests/                  # Test and benchmark scripts
├── docs/                   # Documentation
├── Makefile
├── docker-compose.yml
├── .dockerignore
├── README.md
└── requirements.txt
```

## Headers

| File | Purpose |
|------|---------|
| `src/morans_i_mkl.h` | Public API: types, function prototypes, constants |
| `src/morans_i_internal.h` | Internal: shared helpers, hash table API, sparse handle, RNG |
| `src/openblas_compat.h` | MKL-to-OpenBLAS compatibility layer (conditional) |

## Build System

```
make                        # Intel icx + MKL (default)
make CC=gcc USE_OPENBLAS=1  # GCC + OpenBLAS (portable)
make debug                  # Debug build (-O0 -g3)
make test                   # Submit SLURM test job
make clean                  # Remove artifacts
```

The Makefile uses `VPATH = src` so source files live in `src/` while object files and the binary are built in `build/` (override with `BUILDDIR=<dir>`). `USE_OPENBLAS` switches compiler, headers, and link libraries.

## Data Flow

```
Input TSV/CSV
    |
    v
read_vst_file() --> DenseMatrix (genes x spots)
    |
    v
z_normalize() --> DenseMatrix (z-scored genes x spots)
    |
    +-- extract_coordinates() or read_coordinates_file()
    |       |
    |       v
    |   build_spatial_weight_matrix() --> SparseMatrix W (CSR)
    |
    +-- read_custom_weight_matrix() --> SparseMatrix W (CSR)
    |
    v
Transpose to X_calc (spots x genes)
    |
    +-- calculate_morans_i(X, W) --> standard results
    |
    +-- calculate_residual_morans_i(X, Z, W) --> residual results
    |
    +-- run_permutation_test(X, W) --> z-scores, p-values
    |
    v
save_results() --> Output TSV files
```

## Docker Images

| Image | Base | Arch | Size |
|-------|------|------|------|
| `psychemistz/sigdiscov:latest` | Ubuntu 22.04 + OpenBLAS | amd64, arm64 | ~200MB |
| `psychemistz/sigdiscov:latest-mkl` | Intel oneAPI 2024.0.1 | amd64 | ~15GB |

Built automatically on every push to main via `.github/workflows/docker.yml`.

For HPC clusters using Singularity:
```
singularity pull sigdiscov.sif docker://psychemistz/sigdiscov:latest
singularity exec --bind /data sigdiscov.sif morans_i_mkl [args]
```

### Performance (Biowulf, 19729 genes x 3813 spots, 8 threads)

| Method | Time | Overhead |
|--------|------|----------|
| Native MKL | 64s | baseline |
| Docker MKL (Singularity) | 64s | +0% |
| Docker OpenBLAS (Singularity) | 73s | +14% |

## CI/CD

GitHub Actions:
- **CI** (`.github/workflows/ci.yml`): cppcheck, flake8, GCC+OpenBLAS build + smoke test
- **Docker** (`.github/workflows/docker.yml`): Build and push multi-arch images to Docker Hub

SLURM regression tests (`tests/`):
- Toy example with fixed seed for reproducibility
- Real-data validation against precomputed reference output
- Docker/Singularity benchmark against native build
