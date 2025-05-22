# Moran's I Calculator for Spatial Transcriptomics

**Version 1.1.0**

## Overview

This package provides an optimized implementation of Moran's I spatial autocorrelation statistic specifically designed for spatial transcriptomics data. The implementation utilizes Intel's Math Kernel Library (MKL) for high-performance matrix operations and OpenMP for parallelization, making it suitable for large-scale spatial transcriptomics datasets.

## Features

- Fast calculation of Moran's I using optimized matrix operations with MKL
- Support for different spatial transcriptomics platforms:
  - 10x Genomics Visium (hexagonal grid)
  - Older Spatial Transcriptomics (ST) platforms
  - Single-cell data with spatial coordinates
- Multiple calculation modes:
  - Single-gene Moran's I (spatial autocorrelation for each gene)
  - Pairwise Moran's I (correlation between the first gene and all others)
  - All-vs-all pairwise Moran's I (correlation between all gene pairs)
- Z-normalization of input gene expression data
- RBF kernel for spatial weights with platform-specific parameters
- Configurable maximum neighbor search radius
- Permutation testing to assess statistical significance 
- Optimized sparse matrix operations
- Multi-threaded implementation with OpenMP and MKL
- Robust error handling and memory management
- Unified configuration structure for parameter management
- Support for high-resolution timing and performance analysis

## Requirements

- Intel oneAPI Base Toolkit (for the Intel LLVM-based compiler)
- Intel oneAPI Math Kernel Library (MKL)
- C99-compatible compiler with OpenMP support

## Installation

### Prerequisites

1. Install Intel oneAPI Base Toolkit and MKL from [Intel's website](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)
2. Source the Intel oneAPI environment:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

### Building from Source

1. Clone the repository:
   ```bash
   git clone [repository URL]
   cd morans_i_mkl
   ```

2. Build with the provided Makefile:
   ```bash
   make
   ```

3. Optionally install system-wide:
   ```bash
   make install PREFIX=/usr/local
   ```

### Makefile Options

The provided Makefile supports several targets:

- `make all` - Build the program (default)
- `make clean` - Remove build artifacts
- `make install` - Install to $(PREFIX)/bin (default: /usr/local/bin)
- `make uninstall` - Remove from $(PREFIX)/bin
- `make debug` - Build with debug flags and symbols
- `make help` - Show Makefile help message

## Input Format

The program accepts tab-separated files (TSV) with the following structure:

- **Header row**: Contains spot/cell identifiers. For Visium/ST data, these should be coordinates in the format `ROWxCOL` (e.g., `12x34`). For single-cell data, these should be cell IDs that match those in the coordinate file.
- **Data rows**: Each row starts with a gene name followed by expression values for each spot/cell.
- The first cell in the header row can be empty or contain a label for the gene column.

Example:
```
Gene  1x1   1x2   1x3   2x1   2x2   2x3
Gene1 0.5   1.2   0.8   0.3   1.5   0.7
Gene2 2.1   1.8   0.9   1.4   0.6   1.2
```

For single-cell data, a coordinate file in TSV format is required with columns for cell IDs and their spatial coordinates.

Example coordinate file:
```
cell_ID sdimx  sdimy
cell_1  10.5   20.3
cell_2  15.8   18.7
```

## Usage

### Basic Usage

```bash
./morans_i_mkl -i <input.tsv> -o <output_prefix> [OPTIONS]
```

### Standard Run Options

```
Required Arguments:
  -i <file>        Input data matrix file (Genes x Spots/Cells).
  -o <prefix>      Output file prefix for results.

General Options:
  -r <int>         Maximum grid radius for neighbor search. Default: 5.
  -p <int>         Platform type (0: Visium, 1: Older ST, 2: Single Cell). Default: 0.
  -b <0|1>         Calculation mode: 0 = Single-gene, 1 = Pairwise. Default: 1.
  -g <0|1>         Gene selection (only applies if -b 1): 0 = Compute between first gene and all others,
                   1 = Compute for all gene pairs. Default: 1.
  -s <0|1>         Include self-comparison (spot i vs spot i)? 0 = No, 1 = Yes. Default: 0.
  -t <int>         Set number of OpenMP threads. Default: 4 (or OMP_NUM_THREADS).
  -m <int>         Set number of MKL threads. Default: Value of -t.
  --sigma <float>  Custom sigma parameter for RBF kernel.

Single-cell specific options:
  -c <file>        Coordinates/metadata file with cell locations (TSV format).
                   Required for single-cell data.
  --id-col <name>  Column name for cell IDs in metadata file. Default: 'cell_ID'.
  --x-col <name>   Column name for X coordinates in metadata file. Default: 'sdimx'.
  --y-col <name>   Column name for Y coordinates in metadata file. Default: 'sdimy'.
  --scale <float>  Scaling factor for coordinates to convert to integer grid. Default: 100.0.
```

### Permutation Test Options

```
  --run-perms             Enable permutation testing (with default settings).
  --n-perms <int>         Number of permutations. Default: 1000. Implies --run-perms.
  --perm-seed <int>       Seed for RNG in permutations. Default: Based on system time.
  --perm-output-zscores   Output Z-scores from permutation test.
  --perm-output-pvalues   Output p-values from permutation test.
```

### Toy Example Mode

```
  --run-toy-example       Runs a small, built-in 2D grid (5x5) example to test functionality.
                          Requires -o <prefix>. Permutation options can be used.
```

### Output Files

The format and file names depend on the calculation mode:

- **Single-gene mode (-b 0)**:
  - `<prefix>_single_gene_moran_i.tsv`: Two columns (Gene, MoranI)

- **Pairwise First Gene mode (-b 1, -g 0)**:
  - `<prefix>_first_vs_all_moran_i.tsv`: Two columns (Gene, MoranI_vs_FirstGene)

- **Pairwise All mode (-b 1, -g 1)**:
  - `<prefix>_all_pairs_moran_i_raw.tsv`: Raw lower triangular matrix of Moran's I values
  
- **Permutation Test outputs (if enabled for pairwise all / toy example)**:
  - `<prefix>_zscores_lower_tri.tsv`: Z-scores (if --perm-output-zscores)
  - `<prefix>_pvalues_lower_tri.tsv`: P-values (if --perm-output-pvalues)

## Examples

1. Calculate single-gene Moran's I for Visium data:
   ```bash
   ./morans_i_mkl -i visium_data.tsv -o results_single -b 0 -p 0 -r 4
   ```

2. Calculate pairwise Moran's I between the first gene and all others:
   ```bash
   ./morans_i_mkl -i visium_data.tsv -o results_firstvsall -b 1 -g 0
   ```

3. Process single-cell data with spatial coordinates:
   ```bash
   ./morans_i_mkl -i sc_expr.tsv -o results_sc -p 2 -c sc_coords.tsv --id-col cell_id --x-col x_coord --y-col y_coord
   ```

4. Run with permutation testing for statistical significance:
   ```bash
   ./morans_i_mkl -i visium_data.tsv -o results_perm -b 1 -g 1 --run-perms --n-perms 1000 --perm-output-zscores --perm-output-pvalues
   ```

5. Run the built-in toy example for testing:
   ```bash
   ./morans_i_mkl --run-toy-example -o toy_example --n-perms 100 --perm-seed 42
   ```

## Implementation Details

### Mathematical Formulations

#### Univariate Moran's I (Single-gene)

The univariate Moran's I is calculated using the matrix formula:
```
I_univariate = (Z^T W Z) / S0
```
where:
- Z is the standardized (z-score) vector of gene expression values
- W is the spatial weight matrix
- S0 is the sum of all weights in W
- ^T denotes the transpose operation

#### Bivariate Moran's I (Pairwise Analysis)

The bivariate Moran's I between two genes is calculated as:
```
I_bivariate = (Z_1^T W Z_2) / S0
```
where:
- Z_1 ∈ ℝⁿ is the standardized expression vector of the first gene across n spots
- Z_2 ∈ ℝⁿ is the standardized expression vector of the second gene across the same n spots
- W is the spatial weight matrix
- S0 is the sum of all weights in W

### Spatial Weight Matrix

The spatial weight matrix is constructed using a Radial Basis Function (RBF) kernel:
```
W_ij = exp(-(d_ij^2) / (2 * sigma^2))
```
where:
- d_ij is the Euclidean distance between spot i and j
- sigma is the scale parameter:
  - 100μm for Visium data (distance between adjacent spots)
  - 200μm for older ST platforms
  - Automatically inferred from data for single-cell (or custom value with --sigma)
- W_ij = 0 when the distance exceeds a maximum radius threshold
- W_ii = 0 to exclude self-comparisons (if include_same_spot = 0)

### Permutation Testing

Permutation testing is used to assess the statistical significance of observed Moran's I values. The process involves:

1. Calculate observed Moran's I value(s)
2. Randomly permute the gene expression values across spots (maintaining gene-wise Z-normalization)
3. Recalculate Moran's I for each permutation
4. Compute statistical measures:
   - Z-scores: (observed_I - mean(permuted_I)) / std(permuted_I)
   - P-values: proportion of permuted values with absolute value >= absolute observed value

### Efficient Implementation

Our implementation uses Intel MKL to optimize Moran's I calculation:

- Sparse CSR format for the spatial weight matrix to reduce memory usage
- Optimized sparse matrix-vector operations with mkl_sparse_d_mm()
- Dense matrix operations with cblas_dgemm()
- OpenMP parallelization for multi-threading
- Optimized vectorized math operations with Intel VML functions

This approach enables efficient processing of large spatial transcriptomics datasets containing thousands of spots and genes.

## Performance Considerations

- Increasing the maximum radius (`-r`) increases computational complexity
- For large datasets, adjust thread counts (`-t` and `-m`) to match your hardware
- All-vs-all pairwise mode (`-b 1 -g 1`) is significantly more memory-intensive than other modes
- Permutation testing (`--run-perms`) substantially increases computation time, especially with higher permutation counts
- For very large datasets, consider:
  - Using single-gene mode (`-b 0`) or first-gene-vs-all mode (`-b 1 -g 0`)
  - Reducing permutation count for initial analyses
  - Running on a high-memory system for all-vs-all analyses

## Error Codes

The program provides specific error codes for troubleshooting:

- `MORANS_I_SUCCESS` (0): Operation completed successfully
- `MORANS_I_ERROR_MEMORY` (-1): Memory allocation failure
- `MORANS_I_ERROR_FILE` (-2): File access or parsing error
- `MORANS_I_ERROR_PARAMETER` (-3): Invalid input parameter
- `MORANS_I_ERROR_COMPUTATION` (-4): Computation error

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, Kenneth Aldape, Lalage Wakefield, Peng Jiang. Inference of secreted protein activities in intercellular communication. [[Link](https://github.com/data2intelligence/SecAct)]