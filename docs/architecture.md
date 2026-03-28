# Architecture

## Source Modules

| File | Lines | Responsibility |
|------|-------|----------------|
| `main.c` | 2185 | CLI parsing, orchestration, toy example |
| `morans_i_utils.c` | ~540 | Hash table, config init, timing, input validation, permutation helpers |
| `morans_i_io.c` | ~2960 | All file I/O: VST, weight matrix, cell type, coordinates, results |
| `morans_i_core.c` | ~800 | Z-normalization, Moran's I calculation (all modes) |
| `morans_i_spatial.c` | ~700 | RBF decay, distance matrix, weight matrix construction |
| `morans_i_residual.c` | ~1200 | Regression, projection, residual Moran's I, residual permutations |
| `morans_i_perm.c` | ~360 | Standard permutation testing |
| `morans_i_memory.c` | ~130 | All `free_*` functions |

## Headers

| File | Purpose |
|------|---------|
| `morans_i_mkl.h` | Public API: types, function prototypes, constants |
| `morans_i_internal.h` | Internal: shared helpers, hash table API, sparse handle, RNG |
| `openblas_compat.h` | MKL-to-OpenBLAS compatibility layer (conditional) |

## Build System

```
make                        # Intel icx + MKL (default)
make CC=gcc USE_OPENBLAS=1  # GCC + OpenBLAS (portable)
make debug                  # Debug build (-O0 -g3)
make test                   # Submit SLURM test job
make clean                  # Remove artifacts
```

The Makefile detects build mode via `USE_OPENBLAS`. When defined:
- Compiler switches to GCC with `-fopenmp`
- `morans_i_mkl.h` includes `openblas_compat.h` instead of MKL headers
- Links `-lopenblas -llapacke` instead of MKL libraries

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

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
- **static-analysis**: cppcheck on all C sources
- **python-lint**: flake8 on `make_custom_w.py`
- **build**: GCC + OpenBLAS compilation + smoke test (toy example)

SLURM regression tests (`tests/`):
- Toy example with fixed seed for reproducibility
- Value sanity checks (positive autocorrelation for gradients, negative for checkerboard)
