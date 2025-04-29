# Moran's I Calculator for Spatial Transcriptomics

## Overview

This package provides an optimized implementation of Moran's I spatial autocorrelation statistic specifically designed for spatial transcriptomics data. The implementation utilizes Intel's Math Kernel Library (MKL) for high-performance matrix operations and OpenMP for parallelization, making it suitable for large-scale spatial transcriptomics datasets.

## Features

- Fast calculation of Moran's I using optimized matrix operations
- Support for different spatial transcriptomics platforms:
  - 10x Genomics Visium
  - Older Spatial Transcriptomics (ST) platforms
  - Single-cell data with spatial coordinates
- Multiple calculation modes:
  - Single-gene Moran's I (spatial autocorrelation for each gene)
  - Pairwise Moran's I (correlation between the first gene and all others)
  - All-vs-all pairwise Moran's I (correlation between all gene pairs)
- Z-normalization of input gene expression data
- Gaussian distance decay for spatial weights
- Configurable maximum neighbor search radius
- Optimized sparse matrix operations
- Multi-threaded implementation with OpenMP and MKL

## Requirements

- Intel oneAPI Base Toolkit (for the Intel LLVM-based compiler)
- Intel oneAPI Math Kernel Library (MKL)
- C99-compatible compiler with OpenMP support

## Installation

1. Make sure you have Intel oneAPI installed and properly set up.
2. Source the Intel oneAPI environment:
   ```
   source /opt/intel/oneapi/setvars.sh
   ```
3. Clone the repository and build the program:
   ```
   git clone [repository URL]
   cd morans_i_mkl
   make
   ```

## Usage

### Basic Usage

```
./morans_i_mkl -i <input.tsv> -o <output.tsv> [OPTIONS]
```

### Command-line Options

```
Required Arguments:
  -i <file>  Input data matrix file (Genes x Spots/Cells).
  -o <file>  Output file.

Options:
  -r <int>   Maximum grid radius for neighbor search. Default: 5.
  -p <int>   Platform type (0: Visium, 1: Older ST, 2: Single Cell). Default: 0.
  -b <0|1>   Calculation mode: 0 = Single-gene Moran's I, 1 = Pairwise Moran's I. Default: 1.
  -g <0|1>   Gene selection (only applies if -b 1): 0 = Compute Moran's I only between the *first* gene and all others,
             1 = Compute for *all* gene pairs. Default: 1.
  -s <0|1>   Include self-comparison (spot i vs spot i)? 0 = No, 1 = Yes. Default: 0.
  -t <int>   Set number of OpenMP threads. Default: Use OMP_NUM_THREADS environment variable.
  -m <int>   Set number of MKL threads. Default: Use MKL_NUM_THREADS environment variable.

Single-cell specific options:
  -c <file>         Coordinates/metadata file with cell locations (CSV format). Required for single-cell data.
  --id-col <name>   Column name for cell IDs in metadata file. Default: 'cell_ID'.
  --x-col <name>    Column name for X coordinates in metadata file. Default: 'sdimx'.
  --y-col <name>    Column name for Y coordinates in metadata file. Default: 'sdimy'.
  --scale <float>   Scaling factor for coordinates to convert to integer grid. Default: 100.0.
```

## Input Format

Tab-separated file (TSV) with:
- First row: Header with spot coordinates (e.g., '12x34') or cell IDs. First cell can be empty/gene ID.
- Subsequent rows: Gene name followed by expression values for each spot/cell.

Example:
```
Gene	1x1	1x2	1x3	2x1	2x2	2x3
Gene1	0.5	1.2	0.8	0.3	1.5	0.7
Gene2	2.1	1.8	0.9	1.4	0.6	1.2
...
```

For single-cell data, a coordinate file in CSV format is required with columns for cell IDs and their spatial coordinates.

## Output Format

- **Single-gene mode (-b 0)**: TSV file with two columns: 'Gene', 'MoranI'
- **Pairwise First Gene mode (-b 1, -g 0)**: TSV file with two columns: 'Gene', 'MoranI_vs_Gene0'
- **Pairwise All mode (-b 1, -g 1)**: TSV file representing a symmetric matrix where rows and columns are gene names and cell (i, j) is Moran's I between gene_i and gene_j

## Examples

1. Calculate single-gene Moran's I for Visium data:
   ```
   ./morans_i_mkl -i visium_data.tsv -o moransI_results.tsv -b 0 -p 0 -r 4
   ```

2. Calculate pairwise Moran's I between the first gene and all others:
   ```
   ./morans_i_mkl -i visium_data.tsv -o moransI_pairwise.tsv -b 1 -g 0
   ```

3. Process single-cell data with spatial coordinates:
   ```
   ./morans_i_mkl -i sc_expr.tsv -o moransI_sc.tsv -p 2 -c sc_coords.csv --id-col cell_id --x-col x_coord --y-col y_coord
   ```

## Performance Considerations

- Increasing the maximum radius (`-r`) increases computational complexity
- For large datasets, consider adjusting thread counts (`-t` and `-m`)
- All-vs-all pairwise mode (`-b 1 -g 1`) is significantly more memory-intensive than other modes
- Self-comparisons can be excluded to focus on neighborhood relationships (`-s 0`)

## Implementation Details

This implementation uses several optimizations:
- Sparse CSR matrix format for the spatial weights matrix
- BLAS operations for dense matrix multiplications
- Sparse BLAS for sparse-dense matrix operations
- Vectorized division operations using Intel MKL VML
- OpenMP parallelization for multi-threading
- Efficient coordinate extraction and validation

## License

[Specify license information here]

## Citation

[Citation information here]
