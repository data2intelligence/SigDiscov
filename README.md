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
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```
3. Clone the repository and build the program:
   ```bash
   git clone [repository URL]
   cd morans_i_mkl
   make
   ```

## Input Format

The program accepts tab-separated files (TSV) with the following structure:

- **Header row**: Contains spot/cell identifiers. For Visium/ST data, these should be coordinates in the format `ROWxCOL` (e.g., `12x34`). For single-cell data, these should be cell IDs that match those in the coordinate file.
- **Data rows**: Each row starts with a gene name followed by expression values for each spot/cell.
- The header row can start with or without a tab before the first column name or have index name.

Example:
```
Gene  1x1 1x2 1x3 2x1 2x2 2x3
Gene1 0.5 1.2 0.8 0.3 1.5 0.7
Gene2 2.1 1.8 0.9 1.4 0.6 1.2
...
```

For single-cell data, a coordinate file in TSV format is required with columns for cell IDs and their spatial coordinates.

## Usage

### Basic Usage

```bash
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
  -c <file>         Coordinates/metadata file with cell locations (TSV format). Required for single-cell data.
  --id-col <name>   Column name for cell IDs in metadata file. Default: 'cell_ID'.
  --x-col <name>    Column name for X coordinates in metadata file. Default: 'sdimx'.
  --y-col <name>    Column name for Y coordinates in metadata file. Default: 'sdimy'.
  --scale <float>   Scaling factor for coordinates to convert to integer grid. Default: 100.0.
```

### Output Format

The format of the output file depends on the calculation mode:

- **Single-gene mode (-b 0)**: TSV file with two columns: 'Gene', 'MoranI'
- **Pairwise First Gene mode (-b 1, -g 0)**: TSV file with two columns: 'Gene', 'MoranI_vs_Gene0'
- **Pairwise All mode (-b 1, -g 1)**: TSV file representing a symmetric matrix where rows and columns are gene names and cell (i, j) is Moran's I between gene_i and gene_j

## Examples

1. Calculate single-gene Moran's I for Visium data:
   ```bash
   ./morans_i_mkl -i visium_data.tsv -o moransI_results.tsv -b 0 -p 0 -r 4
   ```

2. Calculate pairwise Moran's I between the first gene and all others:
   ```bash
   ./morans_i_mkl -i visium_data.tsv -o moransI_pairwise.tsv -b 1 -g 0
   ```

3. Process single-cell data with spatial coordinates:
   ```bash
   ./morans_i_mkl -i sc_expr.tsv -o moransI_sc.tsv -p 2 -c sc_coords.tsv --id-col cell_id --x-col x_coord --y-col y_coord
   ```

## Implementation Details

This implementation uses several optimizations:

- Sparse CSR matrix format for the spatial weights matrix
- BLAS operations for dense matrix multiplications
- Sparse BLAS for sparse-dense matrix operations
- Vectorized division operations using Intel MKL VML
- OpenMP parallelization for multi-threading
- Efficient coordinate extraction and validation
- Flexible handling of TSV files with or without leading tabs in the header row

### Moran's I Calculation

The implementation calculates Moran's I using the matrix formula:
```
I = (X' * W * X) / S0
```
where:
- X is the Z-normalized gene expression matrix
- W is the spatial weight matrix
- S0 is the sum of all weights in W
- X' is the transpose of X

### Spatial Weight Matrix

The spatial weight matrix is constructed using a Radial Basis Function (RBF) kernel as specified in the paper:
```
w_ij = exp(-(d_ij^2) / (2 * sigma^2))
```
where:
- dij is the Euclidean distance between spot i and j
- sigma is the scale parameter, set to 100μm for Visium data (the distance between a spot and its 1st layer neighbor)
- Wii = 0 to mask coexpression effects in the same spot
- Wij = 0 when the distance exceeds `2*sigma` (e.g., 200μm for Visium, covering two layers of neighbor spots)

For single-cell data, the sigma parameter is automatically inferred from the data by calculating the average nearest neighbor distance, or can be manually specified with the `--sigma` parameter.

## Performance Considerations

- Increasing the maximum radius (`-r`) increases computational complexity
- For large datasets, consider adjusting thread counts (`-t` and `-m`)
- All-vs-all pairwise mode (`-b 1 -g 1`) is significantly more memory-intensive than other modes
- Self-comparisons can be excluded to focus on neighborhood relationships (`-s 0`)
- For very large datasets, consider using single-gene mode (`-b 0`) or first-gene-vs-all mode (`-b 1 -g 0`)

## License

[Specify license information here]

## Citation

[Citation information here]