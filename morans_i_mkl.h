/* morans_i_mkl.h - Header for optimized MKL-based Moran's I implementation
 *
 * This library provides efficient calculation of Moran's I spatial autocorrelation
 * statistics for gene expression data, optimized using Intel MKL.
 *
 * Version: 1.2.2 (Optimized custom weight matrix reading)
 */

#ifndef MORANS_I_MKL_H
#define MORANS_I_MKL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h> 
#include <time.h>  /* For timing functions, time() */
#include <string.h>  /* For string manipulation functions */
#include <omp.h>     /* For OpenMP parallelization */
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_vml.h"  /* For vdDiv and other VML functions */

/* Library version */
#define MORANS_I_MKL_VERSION "1.2.2"

/* Constants */
#define VISIUM 0                      /* Visium platform with hexagonal grid */
#define OLD 1                         /* Legacy platform mode */
#define SINGLE_CELL 2                 /* Single-cell data with irregular positions */
#define CUSTOM_WEIGHTS 3              /* Custom weight matrix provided */
#define ZERO_STD_THRESHOLD 1e-8       /* Threshold for considering standard deviation zero */
#define WEIGHT_THRESHOLD 1e-12        /* Threshold for considering spatial weight zero */
#define BUFFER_SIZE 1024              /* Buffer size for reading files */
#define DEFAULT_MAX_RADIUS 5          /* Default maximum radius for neighbors */
#define DEFAULT_PLATFORM_MODE VISIUM  /* Default spatial platform */
#define DEFAULT_CALC_PAIRWISE 1       /* Default: calculate pairwise Moran's I */
#define DEFAULT_CALC_ALL_VS_ALL 1     /* Default: calculate all genes vs all genes */
#define DEFAULT_INCLUDE_SAME_SPOT 0   /* Default: exclude self-connections */
#define DEFAULT_COORD_SCALE_FACTOR 100.0  /* Default coordinate scaling factor for single-cell grid conversion */
#define DEFAULT_NUM_THREADS 4         /* Default number of OpenMP threads if not set by OMP_NUM_THREADS or -t */
#define DEFAULT_MKL_NUM_THREADS 0     /* Default MKL threads (0: use OpenMP setting from config.n_threads or MKL internal default) */
#define DEFAULT_NUM_PERMUTATIONS 1000 /* Default number of permutations if enabled */
#define DEFAULT_ROW_NORMALIZE_WEIGHTS 0 /* Default: do not normalize rows to sum to 1 */

/* Weight matrix file format types */
#define WEIGHT_FORMAT_AUTO 0          /* Auto-detect format */
#define WEIGHT_FORMAT_DENSE 1         /* Dense TSV format with spot names */
#define WEIGHT_FORMAT_SPARSE_COO 2    /* Sparse COO format (row, col, value) */
#define WEIGHT_FORMAT_SPARSE_TSV 3    /* Sparse TSV format (spot1, spot2, weight) */

/* Error codes */
#define MORANS_I_SUCCESS 0
#define MORANS_I_ERROR_MEMORY -1
#define MORANS_I_ERROR_FILE -2
#define MORANS_I_ERROR_PARAMETER -3
#define MORANS_I_ERROR_COMPUTATION -4

/* Forward declaration for SpotNameHashTable to break potential circular dependency if needed,
   though direct inclusion of struct definition is usually fine here.
   If SpotNameHashTable itself used DenseMatrix/SparseMatrix, then forward declare those too.
   For now, assuming direct definition is okay.
*/
// struct SpotNameHashTable_st; // Not strictly needed if defined before use


/* Hash Table for Spot Name Lookup (Moved from .c to .h) */
typedef struct SpotNameHashNode_st {
    char* name;
    MKL_INT index;
    struct SpotNameHashNode_st* next;
} SpotNameHashNode;

typedef struct SpotNameHashTable_st {
    SpotNameHashNode** buckets;
    size_t num_buckets;
    size_t count;
} SpotNameHashTable;


/**
 * Dense matrix structure
 */
typedef struct {
    double* values;     /* Use mkl_malloc for alignment */
    MKL_INT nrows;      /* Number of rows */
    MKL_INT ncols;      /* Number of columns */
    char** rownames;    /* Row names (e.g., gene IDs, spot IDs) */
    char** colnames;    /* Column names (e.g., spot IDs, gene IDs) */
} DenseMatrix;

/**
 * Sparse matrix structure (CSR format)
 */
typedef struct {
    MKL_INT nrows;      /* Number of rows */
    MKL_INT ncols;      /* Number of columns */
    MKL_INT nnz;        /* Number of non-zero elements */
    MKL_INT* row_ptr;   /* CSR format row pointers (size nrows + 1) */
    MKL_INT* col_ind;   /* CSR format column indices (size nnz) */
    double* values;     /* CSR format values (size nnz) */
    char** rownames;    /* Row names (usually NULL for W matrix) */
    char** colnames;    /* Column names (usually NULL for W matrix) */
} SparseMatrix;

/**
 * Spot coordinates structure
 */
typedef struct {
    MKL_INT* spot_row;  /* Row coordinates (integer grid) */
    MKL_INT* spot_col;  /* Column coordinates (integer grid) */
    char** spot_names;  /* Spot identifiers (original names) */
    int* valid_mask;    /* Mask for valid spots (1=valid, 0=invalid/filtered) */
    MKL_INT total_spots; /* Total number of spots read initially */
    MKL_INT valid_spots; /* Number of spots deemed valid for analysis */
} SpotCoordinates;

/**
 * Permutation test parameters structure
 * These are passed to the run_permutation_test function.
 */
typedef struct {
    int n_permutations;     /* Number of permutations to run */
    unsigned int seed;      /* Random seed for reproducibility */
    int z_score_output;     /* Flag: whether to calculate and store Z-scores */
    int p_value_output;     /* Flag: whether to calculate and store p-values */
    /* n_threads for permutations will be derived from global OpenMP settings */
} PermutationParams;

/**
 * Permutation results structure
 * Holds the matrices generated by permutation testing.
 */
typedef struct {
    DenseMatrix* z_scores;  /* Z-scores for each gene pair (Genes x Genes) */
    DenseMatrix* p_values;  /* P-values for each gene pair (Genes x Genes) */
    DenseMatrix* mean_perm; /* Mean of permuted Moran's I values (Genes x Genes) */
    DenseMatrix* var_perm;  /* Variance of permuted Moran's I values (Genes x Genes) */
} PermutationResults;

/**
 * Configuration parameters for the Moran's I calculation process.
 * Populated from command-line arguments or defaults.
 */
typedef struct {
    int platform_mode;      /* Platform mode (VISIUM, OLD, SINGLE_CELL, CUSTOM_WEIGHTS) */
    int max_radius;         /* Maximum radius for neighboring spots (grid units) */
    int calc_pairwise;      /* Boolean: 1 if pairwise Moran's I, 0 if single-gene */
    int calc_all_vs_all;    /* Boolean (if pairwise): 1 if all genes vs all, 0 if first gene vs all */
    int include_same_spot;  /* Boolean: 1 if w_ii can be non-zero, 0 if w_ii is forced to 0 */
    double coord_scale;     /* Coordinate scaling factor for single-cell to integer grid */
    int n_threads;          /* Number of OpenMP threads to use */
    int mkl_n_threads;      /* Number of MKL threads (0: let MKL decide based on n_threads or its own default) */
    
    // Custom weight matrix configuration
    char* custom_weights_file;    /* Path to custom weight matrix file */
    int weight_format;            /* Format of weight matrix file */
    int normalize_weights;        /* Boolean: normalize custom weights (divide by sum) */
    int row_normalize_weights;    /* Boolean: normalize each row to sum to 1 */
    
    // Permutation-specific configuration
    int run_permutations;      /* Boolean: 1 to run permutation tests, 0 otherwise */
    int num_permutations;      /* Number of permutations if run_permutations is 1 */
    unsigned int perm_seed;    /* Seed for RNG in permutations */
    int perm_output_zscores;   /* Boolean: 1 to output Z-scores from permutations */
    int perm_output_pvalues;   /* Boolean: 1 to output p-values from permutations */
    char* output_prefix;       /* Prefix for output files */    
} MoransIConfig;

/* --- Function Prototypes --- */

/* Initialization and Configuration */
MoransIConfig initialize_default_config(void);
int initialize_morans_i(const MoransIConfig* config);
const char* morans_i_mkl_version(void);
void print_help(const char* program_name); // For library-level help, main.c has its own

/* Utility Functions */
int load_positive_value(const char* value_str, const char* param, unsigned int min, unsigned int max);
double load_double_value(const char* value_str, const char* param);
void print_mkl_status(sparse_status_t status, const char* function_name);
double get_time(void); // Prototype for get_time

/* Custom Weight Matrix Functions */
// The second argument of these three functions is now SpotNameHashTable*
SparseMatrix* read_custom_weight_matrix(const char* filename, int format, char** spot_names_from_expr, MKL_INT n_spots); // This one still needs char** to build the map initially
int detect_weight_matrix_format(const char* filename);
SparseMatrix* read_dense_weight_matrix(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots);
SparseMatrix* read_sparse_weight_matrix_coo(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots);
SparseMatrix* read_sparse_weight_matrix_tsv(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots);
int validate_weight_matrix(const SparseMatrix* W, char** spot_names, MKL_INT n_spots);

/* Spatial Data Processing */
double decay(double d, double sigma);
double infer_sigma_from_data(const SpotCoordinates* coords, double coord_scale);
DenseMatrix* create_distance_matrix(MKL_INT max_radius_grid_units, int platform_mode, double custom_sigma, double coord_scale);
SpotCoordinates* extract_coordinates(char** column_names, MKL_INT n_columns);
SpotCoordinates* read_coordinates_file(const char* filename, const char* id_column,
                                       const char* x_column, const char* y_column,
                                       double coord_scale);
int map_expression_to_coordinates(const DenseMatrix* expr_matrix, const SpotCoordinates* coords,
                                  MKL_INT** mapping_out, MKL_INT* num_mapped_spots_out);

/* Gene Expression Data Processing */
DenseMatrix* read_vst_file(const char* filename);
DenseMatrix* z_normalize(const DenseMatrix* data_matrix);

/* Moran's I Core Calculation */
SparseMatrix* build_spatial_weight_matrix(const MKL_INT* spot_row_valid, const MKL_INT* spot_col_valid,
                                          MKL_INT n_spots_valid, const DenseMatrix* distance_matrix,
                                          MKL_INT max_radius, int row_normalize);
double calculate_weight_sum(const SparseMatrix* W);
DenseMatrix* calculate_morans_i(const DenseMatrix* X_spots_x_genes, const SparseMatrix* W_spots_x_spots, int row_normalized);
double calculate_single_gene_moran_i(const double* gene_data_vector, const SparseMatrix* W_spots_x_spots, MKL_INT n_spots, int row_normalized);
double* calculate_first_gene_vs_all(const DenseMatrix* X_spots_x_genes, const SparseMatrix* W_spots_x_spots, double S0, int row_normalized);
double* calculate_morans_i_batch(const double* X_data_spots_x_genes, long long n_genes, long long n_spots,
                                 const double* W_values, const long long* W_row_ptr, const long long* W_col_ind,
                                 long long W_nnz, int paired_genes);

/* Permutation Testing */
PermutationResults* run_permutation_test(const DenseMatrix* X_spots_x_genes, const SparseMatrix* W_spots_x_spots,
                                         const PermutationParams* params, int row_normalized);

/* Results Saving */
int save_results(const DenseMatrix* result_matrix, const char* output_file); 
int save_single_gene_results(const DenseMatrix* X_calc_spots_x_genes, const SparseMatrix* W_spots_x_spots, double S0_unused, const char* output_file, int row_normalized); 
int save_first_gene_vs_all_results(const double* morans_values_array, const char** gene_names_array, MKL_INT n_genes, const char* output_file);
int save_lower_triangular_matrix_raw(const DenseMatrix* square_matrix, const char* output_file);
int save_permutation_results(const PermutationResults* perm_test_results,
                             const char* output_file_prefix);     

/* Memory Management */
void free_dense_matrix(DenseMatrix* matrix);
void free_sparse_matrix(SparseMatrix* matrix);
void free_spot_coordinates(SpotCoordinates* coords);
void free_permutation_results(PermutationResults* results);

#endif /* MORANS_I_MKL_H */