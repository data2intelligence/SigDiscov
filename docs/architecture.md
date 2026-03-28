# Architecture

## Project Layout

```
.
├── src/                    # C source and headers
│   ├── main.c              # CLI parsing, orchestration
│   ├── toy_example.c       # Built-in 5x5 grid toy example
│   ├── morans_i_utils.c    # Hash table, config, timing, validation
│   ├── morans_i_io.c       # All file I/O
│   ├── morans_i_core.c     # Z-normalization, Moran's I calculation
│   ├── morans_i_spatial.c  # RBF decay, distance, weight matrix
│   ├── morans_i_residual.c # Regression, residual Moran's I
│   ├── morans_i_perm.c     # Standard permutation testing
│   ├── morans_i_memory.c   # All free_* functions
│   ├── morans_i_mkl.h      # Public API header
│   ├── morans_i_internal.h # Internal shared helpers
│   └── openblas_compat.h   # MKL-to-OpenBLAS compatibility
├── tools/
│   └── make_custom_w.py    # Weight matrix generator
├── tests/                  # Test scripts
├── docs/                   # Documentation
├── Makefile
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

The Makefile uses `VPATH = src` so source files live in `src/` while object files and the binary are built in the project root. `USE_OPENBLAS` switches compiler, headers, and link libraries.

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
- **static-analysis**: cppcheck on all C sources in `src/`
- **python-lint**: flake8 on `tools/make_custom_w.py`
- **build**: GCC + OpenBLAS compilation + smoke test (toy example)

SLURM regression tests (`tests/`):
- Toy example with fixed seed for reproducibility
- Real-data validation against precomputed reference output
