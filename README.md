# Moran's I Calculator for Spatial Transcriptomics

**Version 1.3.0**

## Overview

This package provides an optimized implementation of Moran's I spatial autocorrelation statistic specifically designed for spatial transcriptomics data. The implementation utilizes Intel's Math Kernel Library (MKL) for high-performance matrix operations and OpenMP for parallelization, making it suitable for large-scale spatial transcriptomics datasets.

## Features

- Fast calculation of Moran's I using optimized matrix operations with MKL
- Support for different spatial transcriptomics platforms:
  - 10x Genomics Visium (hexagonal grid)
  - Older Spatial Transcriptomics (ST) platforms
  - Single-cell data with spatial coordinates
  - Custom weight matrices (any spatial relationship)
- Multiple calculation modes:
  - Single-gene Moran's I (spatial autocorrelation for each gene)
  - Pairwise Moran's I (correlation between the first gene and all others)
  - All-vs-all pairwise Moran's I (correlation between all gene pairs)
- **NEW: Residual Moran's I analysis (cell type corrected spatial autocorrelation)**
- Z-normalization of input gene expression data
- RBF kernel for spatial weights with platform-specific parameters
- Support for user-defined spatial weight matrices
- Python weight matrix generator tool
- Configurable maximum neighbor search radius
- Permutation testing to assess statistical significance (standard and residual)
- Optimized sparse matrix operations
- Multi-threaded implementation with OpenMP and MKL
- Robust error handling and memory management
- Unified configuration structure for parameter management
- Support for high-resolution timing and performance analysis

## What's New in Version 1.3.0

### Residual Moran's I Analysis

A major new feature that enables cell type-corrected spatial autocorrelation analysis:

- **Cell type correction**: Remove cell type composition effects from gene expression before calculating spatial autocorrelation
- **Multiple data formats**: Support for both deconvolution proportions and single-cell annotations
- **Ridge regression**: Optional regularization for stable coefficient estimation
- **Residual permutation testing**: Statistical significance testing for residual spatial patterns
- **Comprehensive outputs**: Regression coefficients, residuals, and residual Moran's I results

### Cell Type Data Support

Two formats supported for cell type information:

1. **Deconvolution format**: Pre-computed cell type proportions per spot
2. **Single-cell format**: Individual cell annotations with spatial coordinates

### Enhanced Analysis Pipeline

- **Analysis mode selection**: Choose between standard and residual analysis
- **Flexible configuration**: Extensive options for residual analysis parameters
- **Integrated permutation testing**: Both standard and residual permutation tests
- **Improved output organization**: Clear separation of standard and residual results

## What's New in Version 1.2.0

### Custom Weight Matrix Support

You can now provide your own spatial weight matrix instead of relying on built-in platform-specific weight calculations. This enables:

- **Custom spatial relationships**: Define any spatial connectivity pattern
- **Non-standard platforms**: Support for any spatial transcriptomics technology
- **Alternative distance functions**: Use different distance metrics or kernel functions
- **Experimental designs**: Incorporate prior knowledge about spatial relationships
- **Flexible file formats**: Support for dense TSV, sparse COO, and sparse TSV formats

### Weight Matrix Generator Tool

A Python script that generates custom weight matrices using the exact same algorithm as the internal Moran's I calculation:

- **Identical parameters**: Uses same command-line options as main program (-r, -s, -p, etc.)
- **Direct VST integration**: Reads coordinates directly from VST file headers
- **Validation workflow**: Ensures custom matrices produce identical results
- **Multiple formats**: Outputs both dense and sparse matrix formats

### Supported Weight Matrix Formats

1. **Dense TSV Format**: Full matrix with spot names as headers and row labels
2. **Sparse COO Format**: Coordinate format with row coordinates, column coordinates, and weights
3. **Sparse TSV Format**: Three-column format with spot1, spot2, and weight
4. **Auto-detection**: Automatically detects format based on file content

## Requirements

- Intel oneAPI Base Toolkit (for the Intel LLVM-based compiler)
- Intel oneAPI Math Kernel Library (MKL)
- C99-compatible compiler with OpenMP support
- Python 3.6+ with numpy, pandas, scipy (for weight matrix generator)

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

### Python Dependencies

For the weight matrix generator tool:
```bash
pip install numpy pandas scipy
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

The program accepts tab-separated (TSV) or comma-separated (CSV) files with automatic format detection based on file content. Both formats follow the same structure:

- **Header row**: Contains spot/cell identifiers. For Visium/ST data, these should be coordinates in the format `ROWxCOL` (e.g., `12x34`, `50x102`). For single-cell data, these should be cell IDs that match those in the coordinate file.
- **Data rows**: Each row starts with a gene name followed by expression values for each spot/cell.
- The first cell in the header row can be empty or contain a label for the gene column.

Example (TSV format):
```
Gene  1x1   1x2   1x3   2x1   2x2   2x3
Gene1 0.5   1.2   0.8   0.3   1.5   0.7
Gene2 2.1   1.8   0.9   1.4   0.6   1.2
```

Example (CSV format):
```
Gene,1x1,1x2,1x3,2x1,2x2,2x3
Gene1,0.5,1.2,0.8,0.3,1.5,0.7
Gene2,2.1,1.8,0.9,1.4,0.6,1.2
```

For single-cell data, a coordinate file in TSV or CSV format is required with columns for cell IDs and their spatial coordinates.

Example coordinate file (CSV format):
```
cell_ID,sdimx,sdimy
cell_1,10.5,20.3
cell_2,15.8,18.7
```

## Cell Type Data Formats

Both TSV and CSV formats are supported with automatic format detection. The program handles two types of cell type data:

### Deconvolution Format

Cell type proportions per spot in TSV or CSV format. Two orientations are supported:

**Option 1: Spots as rows, cell types as columns (spots × cell_types)**
```
spot_id,Tcell,Bcell,Macrophage,Epithelial
spot_1,0.3,0.1,0.2,0.4
spot_2,0.4,0.0,0.3,0.3
spot_3,0.2,0.2,0.1,0.5
```

**Option 2: Cell types as rows, spots as columns (cell_types × spots)**
```
cell_type,1x1,1x2,2x1,2x2,3x1
Tcell,0.3,0.4,0.2,0.1,0.3
Bcell,0.1,0.0,0.2,0.3,0.1
Macrophage,0.2,0.3,0.1,0.4,0.2
Epithelial,0.4,0.3,0.5,0.2,0.4
```

For Visium datasets, spot names typically follow the format `ROWxCOL` (e.g., `50x102`, `59x19`).

### Single-cell Format

Individual cell annotations with coordinates in TSV or CSV format:

```
cell_ID,cellType,sdimx,sdimy
cell_1,Tcell,10.5,20.3
cell_2,Macrophage,15.8,18.7
cell_3,Epithelial,12.1,25.4
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
  -r <int>         Maximum grid radius for neighbor search. Default: 5. (Ignored if using custom weights)
  -p <int>         Platform type (0: Visium, 1: Older ST, 2: Single Cell, 3: Custom Weights). Default: 0.
  -b <0|1>         Calculation mode: 0 = Single-gene, 1 = Pairwise. Default: 1.
  -g <0|1>         Gene selection (only applies if -b 1): 0 = Compute between first gene and all others,
                   1 = Compute for all gene pairs. Default: 1.
  -s <0|1>         Include self-comparison (spot i vs spot i)? 0 = No, 1 = Yes. Default: 0.
  --row-normalize <0|1>  Normalize each row of weight matrix to sum to 1? 0 = No, 1 = Yes. Default: 0.
  -t <int>         Set number of OpenMP threads. Default: 4 (or OMP_NUM_THREADS).
  -m <int>         Set number of MKL threads. Default: Value of -t.
  --sigma <float>  Custom sigma parameter for RBF kernel.

Custom Weight Matrix Options:
  -w <file>                Custom weight matrix file. Sets platform mode to CUSTOM_WEIGHTS (3).
  --weight-format <format> Weight matrix format: auto (default), dense, sparse_coo, sparse_tsv.
  --normalize-weights      Normalize custom weights by dividing by sum (S0). Default: No.

Single-cell specific options:
  -c <file>        Coordinates/metadata file with cell locations (TSV format).
                   Required for single-cell data.
  --id-col <name>  Column name for cell IDs in metadata file. Default: 'cell_ID'.
  --x-col <name>   Column name for X coordinates in metadata file. Default: 'sdimx'.
  --y-col <name>   Column name for Y coordinates in metadata file. Default: 'sdimy'.
  --scale <float>  Scaling factor for coordinates to convert to integer grid. Default: 100.0.
```

### Residual Moran's I Options

```
  --analysis-mode <standard|residual>  Analysis mode. Default: standard.
  --celltype-file <file>               Cell type composition/annotation file.
  --celltype-format <deconv|sc>        Format: deconvolution or single_cell. Default: single_cell.
  --celltype-id-col <name>             Cell ID column name. Default: cell_ID.
  --celltype-type-col <name>           Cell type column name. Default: cellType.
  --celltype-x-col <name>              X coordinate column name. Default: sdimx.
  --celltype-y-col <name>              Y coordinate column name. Default: sdimy.
  --spot-id-col <name>                 Spot ID column for deconvolution format. Default: spot_id.
  --include-intercept <0|1>            Include intercept in regression. Default: 1.
  --regularization <float>             Ridge regularization parameter. Default: 0.0.
  --normalize-residuals <0|1>          Normalize residuals. Default: 1.
```

### Permutation Test Options

```
  --run-perm                Enable permutation testing.
  --num-perm <int>         Number of permutations. Default: 1000. Implies --run-perm.
  --perm-seed <int>        Seed for RNG in permutations. Default: Based on system time.
  --perm-out-z <0|1>       Output Z-scores from permutation test. Default: 1.
  --perm-out-p <0|1>       Output p-values from permutation test. Default: 1.
```

### Toy Example Mode

```
  --run-toy-example       Runs a small, built-in 2D grid (5x5) example to test functionality.
                          Requires -o <prefix>. Permutation options can be used.
```

## Weight Matrix Generator Tool

### Usage

The included Python script generates custom weight matrices using the exact same algorithm as the internal Moran's I calculation:

```bash
python3 make_custom_w.py -i <input.tsv> -o <output_prefix> [OPTIONS]
```

### Weight Generator Options

```
Required Arguments:
  -i, --input <file>       Input VST file
  -o, --output <prefix>    Output file prefix

Options:
  -r, --max-radius <int>   Maximum radius for neighbors (default: 5)
  -s, --include-same-spot <0|1>  Include self-connections: 0=No, 1=Yes (default: 0)
  -p, --platform <0|1|2>   Platform: 0=Visium, 1=Old ST, 2=Single Cell (default: 0)
  --sigma <float>          RBF kernel sigma (default: 100.0)
```

### Output Files

The format and file names depend on the calculation mode:

- **Single-gene mode (-b 0)**:
  - `<prefix>_single_gene_moran_i.tsv`: Two columns (Gene, MoranI)

- **Pairwise First Gene mode (-b 1, -g 0)**:
  - `<prefix>_first_vs_all_moran_i.tsv`: Two columns (Gene, MoranI_vs_FirstGene)

- **Pairwise All mode (-b 1, -g 1)**:
  - `<prefix>_all_pairs_moran_i_raw.tsv`: Raw lower triangular matrix of Moran's I values
  
- **Standard Permutation Test outputs (if enabled)**:
  - `<prefix>_zscores_lower_tri.tsv`: Z-scores (if --perm-out-z)
  - `<prefix>_pvalues_lower_tri.tsv`: P-values (if --perm-out-p)

- **Residual Analysis outputs**:
  - `<prefix>_residual_morans_i_raw.tsv`: Residual Moran's I matrix
  - `<prefix>_regression_coefficients.tsv`: Cell type regression coefficients
  
- **Residual Permutation outputs (if enabled)**:
  - `<prefix>_residual_zscores_lower_tri.tsv`: Residual Z-scores
  - `<prefix>_residual_pvalues_lower_tri.tsv`: Residual P-values

## Examples

### Standard Moran's I Analysis

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

### Residual Moran's I Analysis

4. Basic residual analysis with deconvolution data (CSV format):
   ```bash
   ./morans_i_mkl -i expression.tsv -o results_residual \
     --analysis-mode residual \
     --celltype-file deconv_proportions.csv \
     --celltype-format deconv
   ```

5. Residual analysis with Visium cell type data (cell types as rows, spots as columns):
   ```bash
   ./morans_i_mkl -i visium_expression.csv -o results_visium_residual \
     --analysis-mode residual \
     --celltype-file visium_celltypes.csv \
     --celltype-format deconv
   ```

6. Residual analysis with single-cell annotations:
   ```bash
   ./morans_i_mkl -i expression.tsv -o results_residual \
     --analysis-mode residual \
     --celltype-file single_cell_annotations.tsv \
     --celltype-format sc \
     --celltype-id-col cell_ID \
     --celltype-type-col cellType
   ```

7. Residual analysis with ridge regularization:
   ```bash
   ./morans_i_mkl -i expression.tsv -o results_residual \
     --analysis-mode residual \
     --celltype-file proportions.tsv \
     --regularization 0.1 \
     --include-intercept 1
   ```

8. Residual analysis with permutation testing:
   ```bash
   ./morans_i_mkl -i expression.tsv -o results_residual \
     --analysis-mode residual \
     --celltype-file proportions.tsv \
     --run-perm --num-perm 1000 \
     --perm-out-z --perm-out-p
   ```

### Custom Weight Matrix Analysis

9. Generate and use custom weight matrices:
   ```bash
   # Generate custom weights
   python3 make_custom_w.py -i expression_data.tsv -o custom_weights -r 5 -s 0 -p 0
   
   # Use dense format
   ./morans_i_mkl -i expression_data.tsv -o results_custom -w custom_weights_dense.tsv --weight-format dense
   
   # Use sparse format
   ./morans_i_mkl -i expression_data.tsv -o results_custom -w custom_weights_sparse.tsv --weight-format sparse_tsv
   ```

10. Use custom weights with normalization:
    ```bash
    ./morans_i_mkl -i expression_data.tsv -o results_custom -w weights.tsv --normalize-weights
    ```

### Advanced Examples

11. Run with permutation testing for statistical significance:
    ```bash
    ./morans_i_mkl -i visium_data.tsv -o results_perm -b 1 -g 1 --run-perm --num-perm 1000 --perm-out-z --perm-out-p
    ```

12. Run the built-in toy example for testing:
    ```bash
    ./morans_i_mkl --run-toy-example -o toy_example --num-perm 100 --perm-seed 42
    ```

13. Combined residual analysis with custom weights:
    ```bash
    ./morans_i_mkl -i expression.tsv -o results_combined \
      --analysis-mode residual \
      --celltype-file celltypes.tsv \
      -w custom_weights.tsv \
      --run-perm --num-perm 500
    ```

## Custom Weight Matrix File Formats

### Dense TSV Format
Full matrix with spot names as headers and row labels:
```
spot_id 1x1 1x2 2x1 2x2
1x1 0.0 0.8 0.6 0.3
1x2 0.8 0.0 0.3 0.6
2x1 0.6 0.3 0.0 0.8
2x2 0.3 0.6 0.8 0.0
```

### Sparse COO Format
Coordinate format with row coordinates, column coordinates, and weights:
```
row_spot  col_spot  weight
1x1 1x2 0.8
1x1 2x1 0.6
1x2 2x2 0.6
2x1 2x2 0.8
```

### Sparse TSV Format
Three-column format with spot names and weights:
```
spot1 spot2 weight
spot_A  spot_B  0.8
spot_A  spot_C  0.6
spot_B  spot_D  0.6
spot_C  spot_D  0.8
```

## Creating Custom Weight Matrices

### Using the Weight Matrix Generator (Recommended)

The provided Python script generates weight matrices using the exact same algorithm as the Moran's I program:

```bash
# Generate custom weights with specific parameters
python3 make_custom_w.py -i your_data.tsv -o my_weights -r 3 -s 0 -p 0 --sigma 150

# Use the generated weights
./morans_i_mkl -i your_data.tsv -o results -w my_weights_dense.tsv --weight-format dense
```

### Manual Python Implementation

You can also create custom weight matrices manually:

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Load spot coordinates
spots = pd.read_csv('spot_coordinates.tsv', sep='\t')  # columns: spot_id, x, y
coordinates = spots[['x', 'y']].values

# Generate RBF weights
distances = cdist(coordinates, coordinates, metric='euclidean')
sigma = 100.0  # Adjust based on your data scale
weights = np.exp(-(distances**2) / (2 * sigma**2))

# Remove self-connections
np.fill_diagonal(weights, 0.0)

# Remove very small weights
weights[weights < 1e-12] = 0.0

# Save as dense matrix
df = pd.DataFrame(weights, index=spots['spot_id'], columns=spots['spot_id'])
df.index.name = 'spot_id'
df.to_csv('custom_weights_dense.tsv', sep='\t')

# Or save as sparse matrix
from scipy.sparse import coo_matrix
sparse_weights = coo_matrix(weights)

with open('custom_weights_sparse.tsv', 'w') as f:
    f.write("spot1\tspot2\tweight\n")
    for i, j, w in zip(sparse_weights.row, sparse_weights.col, sparse_weights.data):
        f.write(f"{spots.iloc[i]['spot_id']}\t{spots.iloc[j]['spot_id']}\t{w:.8f}\n")
```

### Alternative Distance Functions

You can implement various spatial relationships:

```python
# Inverse distance weights
weights = 1.0 / (1.0 + distances)

# K-nearest neighbors (binary weights)
k = 6  # number of neighbors
weights = np.zeros_like(distances)
for i in range(len(distances)):
    nearest_k = np.argsort(distances[i])[1:k+1]  # exclude self
    weights[i, nearest_k] = 1.0

# Distance threshold (binary)
threshold = 150.0  # distance threshold
weights = (distances <= threshold).astype(float)

# Power law decay
alpha = 2.0
weights = 1.0 / (1.0 + distances**alpha)
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

#### Residual Moran's I

For residual analysis, the calculation proceeds as follows:

1. **Cell type regression**: Fit linear model for each gene
   ```
   X = Z * β + ε
   ```
   where Z is the cell type matrix and β are coefficients

2. **Residual computation**: Calculate residuals after removing cell type effects
   ```
   R = X - Z * β̂
   ```

3. **Residual Moran's I**: Apply standard Moran's I to residuals
   ```
   I_residual = (R_norm^T W R_norm) / S0
   ```
   where R_norm are the normalized residuals

### Spatial Weight Matrix

#### Built-in Platform Support

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

#### Custom Weight Matrices

When using custom weight matrices (-w option):
- The program reads your pre-computed weight matrix from file
- No distance calculations or RBF kernels are applied
- The matrix must be symmetric for meaningful spatial autocorrelation analysis
- Weights are typically non-negative, but negative weights are allowed
- Self-weights (diagonal elements) are typically zero but can be non-zero
- The sum of weights (S0) is calculated from your provided matrix

### Permutation Testing

Permutation testing is used to assess the statistical significance of observed Moran's I values. The process involves:

1. Calculate observed Moran's I value(s)
2. For standard analysis: Randomly permute the gene expression values across spots
3. For residual analysis: Randomly permute the residuals across spots
4. Recalculate Moran's I for each permutation
5. Compute statistical measures:
   - Z-scores: (observed_I - mean(permuted_I)) / std(permuted_I)
   - P-values: proportion of permuted values with absolute value >= absolute observed value

### Efficient Implementation

Our implementation uses Intel MKL to optimize Moran's I calculation:

- Sparse CSR format for the spatial weight matrix to reduce memory usage
- Optimized sparse matrix-vector operations with mkl_sparse_d_mm()
- Dense matrix operations with cblas_dgemm()
- OpenMP parallelization for multi-threading
- Optimized vectorized math operations with Intel VML functions
- Ridge regression using efficient matrix decompositions for residual analysis

This approach enables efficient processing of large spatial transcriptomics datasets containing thousands of spots and genes.

## Performance Considerations

- Increasing the maximum radius (`-r`) increases computational complexity (not applicable for custom weights)
- For large datasets, adjust thread counts (`-t` and `-m`) to match your hardware
- All-vs-all pairwise mode (`-b 1 -g 1`) is significantly more memory-intensive than other modes
- Permutation testing (`--run-perm`) substantially increases computation time, especially with higher permutation counts
- Residual analysis adds computational overhead for regression and residual calculation
- Custom weight matrices with high density (many non-zero elements) will be slower than sparse matrices
- For very large datasets, consider:
  - Using single-gene mode (`-b 0`) or first-gene-vs-all mode (`-b 1 -g 0`)
  - Reducing permutation count for initial analyses
  - Running on a high-memory system for all-vs-all analyses
  - Pre-filtering spots or genes to reduce matrix dimensions
  - Using regularization in residual analysis for numerical stability

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