/* morans_i_mkl.c - Optimized MKL-based Moran's I implementation
 *
 * Version: 1.2.1 (Updated for improved safety and performance)
 *
 * This implements efficient calculation of Moran's I spatial autocorrelation
 * statistics for spatial transcriptomics data using Intel MKL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <ctype.h>
#include <omp.h>
#include <regex.h>
#include <errno.h>
#include <unistd.h>  /* For access() */
#include <limits.h>
#include <stdint.h>
#include "morans_i_mkl.h"

/* Define M_PI if not defined by math.h */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Enhanced debugging support */
#ifdef DEBUG_BUILD
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "[DEBUG %s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
    #define DEBUG_MATRIX_INFO(mat, name) \
        fprintf(stderr, "[DEBUG] Matrix %s: %lld x %lld, values=%p\n", \
                name, (long long)(mat)->nrows, (long long)(mat)->ncols, (void*)(mat)->values)
#else
    #define DEBUG_PRINT(fmt, ...)
    #define DEBUG_MATRIX_INFO(mat, name)
#endif

/* Maximum reasonable dimensions to prevent memory issues */
#define MAX_REASONABLE_DIM 10000000  /* 10M spots/genes */

/* Resource Management Pattern */
typedef struct {
    void** ptrs;
    void (**free_funcs)(void*);
    size_t count;
    size_t capacity;
} cleanup_list_t;

/* Forward declarations for helper functions */
static void cleanup_list_init(cleanup_list_t* list);
//static int cleanup_list_add(cleanup_list_t* list, void* ptr, void (*free_func)(void*));
static void cleanup_list_free_all(cleanup_list_t* list);
static void cleanup_list_destroy(cleanup_list_t* list);

static int safe_multiply_size_t(size_t a, size_t b, size_t *result);
static int validate_matrix_dimensions(MKL_INT nrows, MKL_INT ncols, const char* context);
static char* trim_whitespace_inplace(char* str);

/* VST file parsing helpers */
static int parse_vst_header(FILE* fp, char** line, size_t* line_buf_size,
                           MKL_INT* n_spots_out, char*** colnames_out);
static int count_vst_genes(FILE* fp, char** line, size_t* line_buf_size, 
                          MKL_INT* n_genes_out);
static int read_vst_data_rows(FILE* fp, char** line, size_t* line_buf_size,
                             DenseMatrix* matrix, MKL_INT n_genes_expected, MKL_INT n_spots_expected);

/* Permutation testing helpers */
static int permutation_worker(const DenseMatrix* X_original,
                             const SparseMatrix* W,
                             const PermutationParams* params,
                             int thread_id,
                             int start_perm,
                             int end_perm,
                             double scaling_factor,
                             double* local_mean_sum,
                             double* local_var_sum_sq,
                             double* local_p_counts,
                             const DenseMatrix* observed_results);

/* Library version information */
const char* morans_i_mkl_version(void) {
    return "1.2.1"; 
}

/* ===============================
 * UTILITY AND HELPER FUNCTIONS
 * =============================== */

/* Resource management implementation */
static void cleanup_list_init(cleanup_list_t* list) {
    if (!list) return;
    list->ptrs = NULL;
    list->free_funcs = NULL;
    list->count = 0;
    list->capacity = 0;
}

/*
static int cleanup_list_add(cleanup_list_t* list, void* ptr, void (*free_func)(void*)) {
    if (!list || !ptr) return -1;
    
    if (list->count >= list->capacity) {
        size_t new_capacity = list->capacity == 0 ? 8 : list->capacity * 2;
        void** new_ptrs = realloc(list->ptrs, new_capacity * sizeof(void*));
        void (**new_funcs)(void*) = realloc(list->free_funcs, new_capacity * sizeof(void(*)(void*)));
        
        if (!new_ptrs || !new_funcs) {
            free(new_ptrs);
            free(new_funcs);
            return -1;
        }
        
        list->ptrs = new_ptrs;
        list->free_funcs = new_funcs;
        list->capacity = new_capacity;
    }
    
    list->ptrs[list->count] = ptr;
    list->free_funcs[list->count] = free_func ? free_func : free;
    list->count++;
    return 0;
}
*/

static void cleanup_list_free_all(cleanup_list_t* list) {
    if (!list) return;
    
    for (size_t i = 0; i < list->count; i++) {
        if (list->ptrs[i] && list->free_funcs[i]) {
            list->free_funcs[i](list->ptrs[i]);
        }
    }
    list->count = 0;
}

static void cleanup_list_destroy(cleanup_list_t* list) {
    if (!list) return;
    cleanup_list_free_all(list);
    free(list->ptrs);
    free(list->free_funcs);
    list->ptrs = NULL;
    list->free_funcs = NULL;
    list->capacity = 0;
}

/* Safe multiplication to prevent overflow */
static int safe_multiply_size_t(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > SIZE_MAX / a) {
        return -1; // Overflow
    }
    *result = a * b;
    return 0;
}

/* Input validation helper */
static int validate_matrix_dimensions(MKL_INT nrows, MKL_INT ncols, const char* context) {
    if (nrows < 0 || ncols < 0) {
        fprintf(stderr, "Error: Negative dimensions in %s: %lld x %lld\n", 
                context, (long long)nrows, (long long)ncols);
        return MORANS_I_ERROR_PARAMETER;
    }
    
    if (nrows > MAX_REASONABLE_DIM || ncols > MAX_REASONABLE_DIM) {
        fprintf(stderr, "Warning: Very large dimensions in %s: %lld x %lld\n", 
                context, (long long)nrows, (long long)ncols);
    }
    
    return MORANS_I_SUCCESS;
}

/* Efficient in-place string trimming */
static char* trim_whitespace_inplace(char* str) {
    char* end;
    
    if (!str) return NULL;
    
    // Trim leading space
    while(isspace((unsigned char)*str)) str++;
    
    if(*str == 0) return str; // All spaces
    
    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    
    // Write new null terminator
    end[1] = '\0';
    
    return str;
}

/* Get current time in seconds with microsecond precision */
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

/* Initialize default configuration */
MoransIConfig initialize_default_config(void) {
    MoransIConfig config;
    config.platform_mode = DEFAULT_PLATFORM_MODE;
    config.max_radius = DEFAULT_MAX_RADIUS;
    config.calc_pairwise = DEFAULT_CALC_PAIRWISE;
    config.calc_all_vs_all = DEFAULT_CALC_ALL_VS_ALL;
    config.include_same_spot = DEFAULT_INCLUDE_SAME_SPOT;
    config.coord_scale = DEFAULT_COORD_SCALE_FACTOR;
    config.n_threads = DEFAULT_NUM_THREADS;
    config.mkl_n_threads = DEFAULT_MKL_NUM_THREADS;
    config.output_prefix = NULL;
    
    // Custom weight matrix defaults
    config.custom_weights_file = NULL;
    config.weight_format = WEIGHT_FORMAT_AUTO;
    config.normalize_weights = 0;
    config.row_normalize_weights = DEFAULT_ROW_NORMALIZE_WEIGHTS;
    
    // Permutation defaults
    config.run_permutations = 0;
    config.num_permutations = DEFAULT_NUM_PERMUTATIONS;
    config.perm_seed = (unsigned int)time(NULL);
    config.perm_output_zscores = 1;
    config.perm_output_pvalues = 1;
    return config;
}

/* Initialize the MKL environment based on configuration */
int initialize_morans_i(const MoransIConfig* config) {
    if (!config) {
        fprintf(stderr, "Error: Null configuration provided to initialize_morans_i\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (config->n_threads > 0) {
        omp_set_num_threads(config->n_threads);
    }
    printf("OpenMP num_threads (requested %d, max effective %d)\n",
           config->n_threads > 0 ? config->n_threads : omp_get_max_threads(), omp_get_max_threads());

    int mkl_threads_to_set;
    if (config->mkl_n_threads > 0) {
        mkl_threads_to_set = config->mkl_n_threads;
    } else if (config->n_threads > 0) {
        mkl_threads_to_set = config->n_threads;
    } else {
        mkl_threads_to_set = omp_get_max_threads();
    }
    mkl_set_num_threads(mkl_threads_to_set);
    printf("MKL num_threads set to: %d (requested %d)\n", mkl_get_max_threads(), mkl_threads_to_set);
    mkl_set_dynamic(0);

    printf("Moran's I MKL Library Initialization complete.\n");
    return MORANS_I_SUCCESS;
}

/* Calculate sum of weights in the weight matrix */
double calculate_weight_sum(const SparseMatrix* W) {
    if (!W || !W->values) {
        fprintf(stderr, "Error: Invalid sparse matrix in calculate_weight_sum\n");
        return 0.0;
    }
    double S0 = 0.0;
    #pragma omp parallel for reduction(+:S0)
    for (MKL_INT i = 0; i < W->nnz; i++) {
        S0 += W->values[i];
    }
    return S0;
}

/* Print help message */
void print_help(const char* program_name) {
    printf("\nCompute Pairwise or Single-Gene Moran's I for Spatial Transcriptomics Data\n\n");
    printf("Usage: %s -i <input.tsv> -o <output_prefix> [OPTIONS]\n\n", program_name);
    printf("Input Format:\n");
    printf("  Tab-separated file (TSV).\n");
    printf("  First row: Header with spot coordinates (e.g., '12x34') or cell IDs. First cell can be empty/gene ID header.\n");
    printf("  Subsequent rows: Gene name followed by expression values for each spot/cell.\n");
    printf("\nRequired Arguments:\n");
    printf("  -i <file>\tInput data matrix file (Genes x Spots/Cells).\n");
    printf("  -o <prefix>\tOutput file prefix for results (e.g., 'my_analysis_results').\n");
    printf("\nOptions:\n");
    printf("  -r <int>\tMaximum grid radius for neighbor search. Default: %d.\n", DEFAULT_MAX_RADIUS);
    printf("  -p <int>\tPlatform type (%d: Visium, %d: Older ST, %d: Single Cell). Affects distance calculation. Default: %d.\n",
           VISIUM, OLD, SINGLE_CELL, DEFAULT_PLATFORM_MODE);
    printf("  -b <0|1>\tCalculation mode: 0 = Single-gene Moran's I, 1 = Pairwise Moran's I. Default: %d.\n",
           DEFAULT_CALC_PAIRWISE);
    printf("  -g <0|1>\tGene selection (only applies if -b 1): 0 = Compute Moran's I only between the *first* gene and all others, "
           "1 = Compute for *all* gene pairs. Default: %d.\n", DEFAULT_CALC_ALL_VS_ALL);
    printf("  -s <0|1>\tInclude self-comparison (spot i vs spot i)? 0 = No, 1 = Yes. Affects the w_ii term. Default: %d.\n",
           DEFAULT_INCLUDE_SAME_SPOT);
    printf("  --row-normalize <0|1>\tNormalize each row of weight matrix to sum to 1? 0 = No, 1 = Yes. Default: %d.\n",
           DEFAULT_ROW_NORMALIZE_WEIGHTS);
    printf("  -t <int>\tSet number of OpenMP threads. Default: %d (or OMP_NUM_THREADS environment variable).\n",
           DEFAULT_NUM_THREADS);
    printf("  -m <int>\tSet number of MKL threads. Default: Value of -t, or OpenMP default if -t is not set.\n");
    printf("\nPermutation Test Options (apply if -b 1 and -g 1):\n");
    printf("  --run-perm <0|1>\tRun permutation test? 0 = No, 1 = Yes. Default: 0.\n");
    printf("  --num-perm <int>\tNumber of permutations. Default: %d.\n", DEFAULT_NUM_PERMUTATIONS);
    printf("  --perm-seed <int>\tRandom seed for permutations. Default: time-based.\n");
    printf("  --perm-out-z <0|1>\tOutput Z-scores from permutations? Default: 1.\n");
    printf("  --perm-out-p <0|1>\tOutput P-values from permutations? Default: 1.\n");

    printf("\nSingle-cell specific options:\n");
    printf("  -c <file>\tCoordinates/metadata file with cell locations (TSV format). Required for single-cell data.\n");
    printf("  --id-col <name>\tColumn name for cell IDs in metadata file. Default: 'cell_ID'.\n");
    printf("  --x-col <name>\tColumn name for X coordinates in metadata file. Default: 'sdimx'.\n");
    printf("  --y-col <name>\tColumn name for Y coordinates in metadata file. Default: 'sdimy'.\n");
    printf("  --scale <float>\tScaling factor for coordinates to convert to integer grid. Default: %.1f.\n",
           DEFAULT_COORD_SCALE_FACTOR);
    printf("  --sigma <float>\tCustom sigma parameter for RBF kernel (physical units). If not provided or <=0, inferred from data for single-cell or platform default used.\n");
    printf("\nOutput Format (files named based on <output_prefix>):\n");
    printf("  If -b 0 (Single-gene): <output_prefix>_single_gene_moran_i.tsv (Gene, MoranI).\n");
    printf("  If -b 1 and -g 1 (Pairwise All): <output_prefix>_all_pairs_moran_i.tsv (Symmetric matrix, or _raw.tsv for lower-tri).\n");
    printf("    Permutation outputs (if --run-perm 1): <prefix>_zscores_lower_tri.tsv, <prefix>_pvalues_lower_tri.tsv\n");
    printf("  If -b 1 and -g 0 (Pairwise First Gene): <output_prefix>_first_vs_all_moran_i.tsv (Gene, MoranI_vs_Gene0).\n");
    printf("\nExample:\n");
    printf("  %s -i expression.tsv -o morans_i_run1 -r 3 -p 0 -b 1 -g 1 -t 8 --run-perm 1 --num-perm 500\n\n", program_name);
    printf("Version: %s\n", morans_i_mkl_version());
}

/* ===============================
 * Z-NORMALIZATION FUNCTION
 * =============================== */

/* Z-Normalize function (Gene-wise: input is Genes x Spots, output is Genes x Spots) */
DenseMatrix* z_normalize(const DenseMatrix* data_matrix) {
    if (!data_matrix || !data_matrix->values) {
        fprintf(stderr, "Error: Invalid data matrix provided to z_normalize\n");
        return NULL;
    }

    MKL_INT n_genes = data_matrix->nrows;
    MKL_INT n_spots = data_matrix->ncols;

    if (validate_matrix_dimensions(n_genes, n_spots, "z_normalize") != MORANS_I_SUCCESS) {
        return NULL;
    }

    printf("Performing Z-normalization on %lld genes across %lld spots...\n",
           (long long)n_genes, (long long)n_spots);

    DenseMatrix* normalized = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!normalized) {
        perror("Failed to allocate DenseMatrix structure for normalized data");
        return NULL;
    }

    // Initialize structure
    normalized->nrows = n_genes;
    normalized->ncols = n_spots;
    normalized->rownames = NULL;
    normalized->colnames = NULL;
    normalized->values = NULL;

    // Allocate arrays
    size_t values_size;
    if (safe_multiply_size_t(n_genes, n_spots, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Matrix dimensions too large for z_normalize\n");
        free(normalized);
        return NULL;
    }

    normalized->values = (double*)mkl_malloc(values_size, 64);
    normalized->rownames = (char**)calloc(n_genes, sizeof(char*));
    normalized->colnames = (char**)calloc(n_spots, sizeof(char*));

    if (!normalized->values || !normalized->rownames || !normalized->colnames) {
        perror("Failed to allocate memory for normalized matrix data");
        if (normalized->values) mkl_free(normalized->values);
        free(normalized->rownames);
        free(normalized->colnames);
        free(normalized);
        return NULL;
    }

    // Copy names safely
    for (MKL_INT i = 0; i < n_genes; i++) {
        if (data_matrix->rownames && data_matrix->rownames[i]) {
            normalized->rownames[i] = strdup(data_matrix->rownames[i]);
            if (!normalized->rownames[i]) {
                perror("Failed to duplicate row name (gene)");
                free_dense_matrix(normalized);
                return NULL;
            }
        }
    }

    for (MKL_INT j = 0; j < n_spots; j++) {
        if (data_matrix->colnames && data_matrix->colnames[j]) {
            normalized->colnames[j] = strdup(data_matrix->colnames[j]);
            if (!normalized->colnames[j]) {
                perror("Failed to duplicate column name (spot)");
                free_dense_matrix(normalized);
                return NULL;
            }
        }
    }

    // Perform normalization with better error handling
    int global_alloc_error = 0;

    #pragma omp parallel
    {
        double* centered_values_tl = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
        double* std_dev_vector_tl = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);

        if (!centered_values_tl || !std_dev_vector_tl) {
            #pragma omp critical
            {
                fprintf(stderr, "Error: Thread %d failed to allocate memory for normalization buffers.\n", omp_get_thread_num());
                global_alloc_error = 1;
            }
        }

        if (!global_alloc_error) {
            #pragma omp for schedule(dynamic)
            for (MKL_INT i = 0; i < n_genes; i++) {
                if (global_alloc_error) continue;

                const double* current_gene_row_input = &(data_matrix->values[i * n_spots]);
                double* current_gene_row_output = &(normalized->values[i * n_spots]);

                double sum = 0.0;
                MKL_INT n_finite = 0;
                for (MKL_INT j = 0; j < n_spots; j++) {
                    if (isfinite(current_gene_row_input[j])) {
                        sum += current_gene_row_input[j];
                        n_finite++;
                    }
                }

                double mean = 0.0;
                double std_dev = 0.0;

                if (n_finite > 1) {
                    mean = sum / n_finite;
                    double sum_sq_diff = 0.0;
                    for (MKL_INT j = 0; j < n_spots; j++) {
                        if (isfinite(current_gene_row_input[j])) {
                            double diff = current_gene_row_input[j] - mean;
                            sum_sq_diff += diff * diff;
                        }
                    }
                    double variance = (n_finite > 0) ? sum_sq_diff / n_finite : 0.0;
                    if (variance < 0.0) variance = 0.0;
                    std_dev = sqrt(variance);
                }

                if (n_finite <= 1 || std_dev < ZERO_STD_THRESHOLD) {
                    for (MKL_INT j = 0; j < n_spots; j++) {
                        current_gene_row_output[j] = 0.0;
                    }
                } else {
                    for (MKL_INT j = 0; j < n_spots; j++) {
                        if (isfinite(current_gene_row_input[j])) {
                            centered_values_tl[j] = current_gene_row_input[j] - mean;
                        } else {
                            centered_values_tl[j] = 0.0;
                        }
                        std_dev_vector_tl[j] = std_dev;
                    }
                    vdDiv(n_spots, centered_values_tl, std_dev_vector_tl, current_gene_row_output);
                }
            }
        }

        if(centered_values_tl) mkl_free(centered_values_tl);
        if(std_dev_vector_tl) mkl_free(std_dev_vector_tl);
    }

    if (global_alloc_error) {
        fprintf(stderr, "Critical error during Z-normalization due to memory allocation failure in threads.\n");
        free_dense_matrix(normalized);
        return NULL;
    }

    printf("Z-normalization complete.\n");
    DEBUG_MATRIX_INFO(normalized, "z_normalized");
    return normalized;
}

/* ===============================
 * SPATIAL WEIGHT MATRIX FUNCTIONS
 * =============================== */

/* Build spatial weight matrix W (Sparse CSR) with optional row normalization */
SparseMatrix* build_spatial_weight_matrix(const MKL_INT* spot_row_valid, const MKL_INT* spot_col_valid,
                                         MKL_INT n_spots_valid, const DenseMatrix* distance_matrix,
                                         MKL_INT max_radius, int row_normalize) {

    if (!spot_row_valid || !spot_col_valid || !distance_matrix || !distance_matrix->values) {
        fprintf(stderr, "Error: Invalid parameters provided to build_spatial_weight_matrix\n");
        return NULL;
    }

    if (validate_matrix_dimensions(n_spots_valid, n_spots_valid, "weight matrix") != MORANS_I_SUCCESS) {
        return NULL;
    }

    if (n_spots_valid == 0) {
        printf("Warning: n_spots_valid is 0 in build_spatial_weight_matrix. Returning empty W.\n");
        SparseMatrix* W_empty = (SparseMatrix*)calloc(1, sizeof(SparseMatrix));
        if(!W_empty) { 
            perror("calloc for empty W"); 
            return NULL; 
        }
        W_empty->nrows = 0; 
        W_empty->ncols = 0; 
        W_empty->nnz = 0;
        W_empty->row_ptr = (MKL_INT*)mkl_calloc(1, sizeof(MKL_INT), 64);
        W_empty->col_ind = NULL;
        W_empty->values = NULL;
        if (!W_empty->row_ptr) { 
            perror("mkl_calloc for empty W->row_ptr"); 
            free(W_empty); 
            return NULL; 
        }
        return W_empty;
    }

    printf("Building sparse spatial weight matrix W (%lld x %lld)%s...\n",
           (long long)n_spots_valid, (long long)n_spots_valid,
           row_normalize ? " with row normalization" : "");

    MKL_INT estimated_neighbors_per_spot = (MKL_INT)(M_PI * max_radius * max_radius * 1.5);
    if (estimated_neighbors_per_spot <= 0) estimated_neighbors_per_spot = 27;
    
    size_t initial_capacity_size;
    if (safe_multiply_size_t(n_spots_valid, estimated_neighbors_per_spot, &initial_capacity_size) != 0) {
        initial_capacity_size = n_spots_valid; // Fallback to minimum
    }
    
    MKL_INT initial_capacity = (MKL_INT)initial_capacity_size;
    size_t max_dense_size;
    if (safe_multiply_size_t(n_spots_valid, n_spots_valid, &max_dense_size) == 0 && 
        initial_capacity > (MKL_INT)max_dense_size) {
        initial_capacity = (MKL_INT)max_dense_size;
    }

    if (initial_capacity <= 0) initial_capacity = (n_spots_valid > 0) ? n_spots_valid : 1;

    printf("  Initial estimated NNZ capacity: %lld\n", (long long)initial_capacity);

    MKL_INT current_capacity = initial_capacity;
    MKL_INT* temp_I = (MKL_INT*)malloc((size_t)current_capacity * sizeof(MKL_INT));
    MKL_INT* temp_J = (MKL_INT*)malloc((size_t)current_capacity * sizeof(MKL_INT));
    double* temp_V = (double*)malloc((size_t)current_capacity * sizeof(double));
    MKL_INT nnz_count = 0;

    if (!temp_I || !temp_J || !temp_V) {
        perror("Failed to allocate initial COO arrays");
        free(temp_I); free(temp_J); free(temp_V);
        return NULL;
    }

    volatile int critical_error_flag = 0;

    #pragma omp parallel
    {
        MKL_INT local_initial_cap = (estimated_neighbors_per_spot > 0) ? estimated_neighbors_per_spot * 2 : 256;
        MKL_INT local_capacity = local_initial_cap;
        MKL_INT* local_I_tl = (MKL_INT*)malloc((size_t)local_capacity * sizeof(MKL_INT));
        MKL_INT* local_J_tl = (MKL_INT*)malloc((size_t)local_capacity * sizeof(MKL_INT));
        double* local_V_tl = (double*)malloc((size_t)local_capacity * sizeof(double));
        MKL_INT local_nnz_tl = 0;
        int thread_alloc_error = 0;

        if (!local_I_tl || !local_J_tl || !local_V_tl) {
            #pragma omp critical
            {
                fprintf(stderr, "Error: Thread %d failed to alloc thread-local COO buffers.\n", omp_get_thread_num());
                critical_error_flag = 1;
            }
            thread_alloc_error = 1;
        }

        if (!thread_alloc_error && !critical_error_flag) {
            #pragma omp for schedule(dynamic, 128)
            for (MKL_INT i = 0; i < n_spots_valid; i++) {
                if (critical_error_flag || thread_alloc_error) {
                    continue; // Changed from break to continue since break is not allowed in OpenMP for loop
                }

                for (MKL_INT j = 0; j < n_spots_valid; j++) {
                    MKL_INT row_shift_abs = llabs(spot_row_valid[i] - spot_row_valid[j]);
                    MKL_INT col_shift_abs = llabs(spot_col_valid[i] - spot_col_valid[j]);

                    if (row_shift_abs < distance_matrix->nrows && col_shift_abs < distance_matrix->ncols) {
                        double weight = distance_matrix->values[row_shift_abs * distance_matrix->ncols + col_shift_abs];

                        if (fabs(weight) > WEIGHT_THRESHOLD) {
                            if (local_nnz_tl >= local_capacity) {
                                MKL_INT new_capacity = (MKL_INT)(local_capacity * 1.5) + 1;
                                MKL_INT* temp_li_new = (MKL_INT*)realloc(local_I_tl, (size_t)new_capacity * sizeof(MKL_INT));
                                MKL_INT* temp_lj_new = (MKL_INT*)realloc(local_J_tl, (size_t)new_capacity * sizeof(MKL_INT));
                                double* temp_lv_new = (double*)realloc(local_V_tl, (size_t)new_capacity * sizeof(double));
                                
                                if (!temp_li_new || !temp_lj_new || !temp_lv_new) {
                                    #pragma omp critical
                                    {
                                        fprintf(stderr, "Error: Thread %d failed to realloc thread-local COO buffers.\n", omp_get_thread_num());
                                        critical_error_flag = 1;
                                    }
                                    local_I_tl = temp_li_new ? temp_li_new : local_I_tl;
                                    local_J_tl = temp_lj_new ? temp_lj_new : local_J_tl;
                                    local_V_tl = temp_lv_new ? temp_lv_new : local_V_tl;
                                    thread_alloc_error = 1;
                                    continue; // Changed from break to continue
                                }
                                local_I_tl = temp_li_new;
                                local_J_tl = temp_lj_new;
                                local_V_tl = temp_lv_new;
                                local_capacity = new_capacity;
                            }
                            local_I_tl[local_nnz_tl] = i;
                            local_J_tl[local_nnz_tl] = j;
                            local_V_tl[local_nnz_tl] = weight;
                            local_nnz_tl++;
                        }
                    }
                }
                if (thread_alloc_error) {
                    continue; // Changed from break to continue
                }
            }
        }

        if (!critical_error_flag && !thread_alloc_error && local_nnz_tl > 0) {
            #pragma omp critical
            {
                if (!critical_error_flag) {
                    if (nnz_count + local_nnz_tl > current_capacity) {
                        MKL_INT needed_capacity = nnz_count + local_nnz_tl;
                        MKL_INT new_global_capacity = current_capacity;
                        while(new_global_capacity < needed_capacity) {
                            new_global_capacity = (MKL_INT)(new_global_capacity * 1.5) + 1;
                        }
                        
                        size_t max_dense;
                        if (safe_multiply_size_t(n_spots_valid, n_spots_valid, &max_dense) == 0 && 
                            new_global_capacity > (MKL_INT)max_dense) {
                            new_global_capacity = (MKL_INT)max_dense;
                        }
                        
                        if (needed_capacity > new_global_capacity && n_spots_valid > 0) {
                            fprintf(stderr, "Error: Cannot resize global COO buffer large enough (%lld needed, max %lld).\n", 
                                   (long long)needed_capacity, (long long)new_global_capacity);
                            critical_error_flag = 1;
                        } else if (n_spots_valid > 0) {
                            printf("  Resizing global COO buffer from %lld to %lld\n", 
                                   (long long)current_capacity, (long long)new_global_capacity);
                            MKL_INT* temp_gi_new = (MKL_INT*)realloc(temp_I, (size_t)new_global_capacity * sizeof(MKL_INT));
                            MKL_INT* temp_gj_new = (MKL_INT*)realloc(temp_J, (size_t)new_global_capacity * sizeof(MKL_INT));
                            double*  temp_gv_new = (double*)realloc(temp_V, (size_t)new_global_capacity * sizeof(double));

                            if (!temp_gi_new || !temp_gj_new || !temp_gv_new) {
                                fprintf(stderr, "Error: Failed to realloc global COO buffers.\n");
                                critical_error_flag = 1;
                            } else {
                                temp_I = temp_gi_new; 
                                temp_J = temp_gj_new; 
                                temp_V = temp_gv_new;
                                current_capacity = new_global_capacity;
                            }
                        }
                    }

                    if (!critical_error_flag && (nnz_count + local_nnz_tl <= current_capacity)) {
                        memcpy(temp_I + nnz_count, local_I_tl, (size_t)local_nnz_tl * sizeof(MKL_INT));
                        memcpy(temp_J + nnz_count, local_J_tl, (size_t)local_nnz_tl * sizeof(MKL_INT));
                        memcpy(temp_V + nnz_count, local_V_tl, (size_t)local_nnz_tl * sizeof(double));
                        nnz_count += local_nnz_tl;
                    } else if (!critical_error_flag) {
                        fprintf(stderr, "Warning: Could not merge thread %d results due to insufficient space.\n", omp_get_thread_num());
                        critical_error_flag = 1;
                    }
                }
            }
        }

        if(local_I_tl) free(local_I_tl);
        if(local_J_tl) free(local_J_tl);
        if(local_V_tl) free(local_V_tl);
    }

    if (critical_error_flag) {
        fprintf(stderr, "Error: A critical error occurred during parallel COO matrix construction.\n");
        free(temp_I); free(temp_J); free(temp_V);
        return NULL;
    }

    printf("  Generated %lld non-zero entries (COO format).\n", (long long)nnz_count);
    if (nnz_count == 0 && n_spots_valid > 0) {
        printf("Warning: No non-zero weights found. Moran's I will likely be zero or undefined.\n");
    }

    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!W) {
        perror("Failed to allocate SparseMatrix structure for W");
        free(temp_I); free(temp_J); free(temp_V);
        return NULL;
    }
    W->nrows = n_spots_valid;
    W->ncols = n_spots_valid;
    W->nnz = nnz_count;
    W->rownames = NULL;
    W->colnames = NULL;

    size_t nnz_alloc_size = (nnz_count > 0) ? (size_t)nnz_count : 1;
    W->row_ptr = (MKL_INT*)mkl_malloc(((size_t)n_spots_valid + 1) * sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc(nnz_alloc_size * sizeof(MKL_INT), 64);
        W->values  = (double*)mkl_malloc(nnz_alloc_size * sizeof(double), 64);
    } else {
        W->col_ind = NULL;
        W->values = NULL;
    }

    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) {
        perror("Failed to allocate CSR arrays for W");
        mkl_free(W->row_ptr); mkl_free(W->col_ind); mkl_free(W->values); free(W);
        free(temp_I); free(temp_J); free(temp_V);
        return NULL;
    }

    // Convert COO to CSR
    if (nnz_count > 0) {
        for (MKL_INT i = 0; i <= n_spots_valid; ++i) W->row_ptr[i] = 0;
        for (MKL_INT k = 0; k < nnz_count; ++k) W->row_ptr[temp_I[k] + 1]++;
        for (MKL_INT i = 0; i < n_spots_valid; ++i) W->row_ptr[i + 1] += W->row_ptr[i];

        MKL_INT* current_insertion_pos = (MKL_INT*)malloc(((size_t)n_spots_valid + 1) * sizeof(MKL_INT));
        if (!current_insertion_pos) {
            perror("Failed to allocate current_insertion_pos array");
            free_sparse_matrix(W);
            free(temp_I); free(temp_J); free(temp_V);
            return NULL;
        }
        memcpy(current_insertion_pos, W->row_ptr, ((size_t)n_spots_valid) * sizeof(MKL_INT));

        for (MKL_INT k = 0; k < nnz_count; ++k) {
            MKL_INT row = temp_I[k];
            MKL_INT index_in_csr = current_insertion_pos[row];
            W->col_ind[index_in_csr] = temp_J[k];
            W->values[index_in_csr] = temp_V[k];
            current_insertion_pos[row]++;
        }
        free(current_insertion_pos);
    } else {
        for (MKL_INT i = 0; i <= n_spots_valid; ++i) W->row_ptr[i] = 0;
    }

    free(temp_I); free(temp_J); free(temp_V);

    // Row normalization if requested
    if (row_normalize && W->nnz > 0) {
        printf("  Performing row normalization...\n");
        MKL_INT normalized_rows = 0;
        
        #pragma omp parallel for reduction(+:normalized_rows)
        for (MKL_INT i = 0; i < n_spots_valid; i++) {
            MKL_INT row_start = W->row_ptr[i];
            MKL_INT row_end = W->row_ptr[i + 1];
            
            if (row_end > row_start) {
                // Calculate row sum
                double row_sum = 0.0;
                for (MKL_INT k = row_start; k < row_end; k++) {
                    row_sum += W->values[k];
                }
                
                // Normalize row if sum is positive
                if (row_sum > ZERO_STD_THRESHOLD) {
                    for (MKL_INT k = row_start; k < row_end; k++) {
                        W->values[k] /= row_sum;
                    }
                    normalized_rows++;
                }
            }
        }
        
        printf("  Row normalization complete: %lld rows normalized.\n", (long long)normalized_rows);
    }

    printf("Sparse weight matrix W built successfully (CSR format, %lld NNZ).\n", (long long)W->nnz);

    // Sort column indices within each row
    if (W->nnz > 0) {
        sparse_matrix_t W_mkl_tmp_handle;
        sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl_tmp_handle, SPARSE_INDEX_BASE_ZERO,
                                                        W->nrows, W->ncols, W->row_ptr,
                                                        W->row_ptr + 1, W->col_ind, W->values);
        if (status == SPARSE_STATUS_SUCCESS) {
            status = mkl_sparse_order(W_mkl_tmp_handle);
            if (status != SPARSE_STATUS_SUCCESS) {
                print_mkl_status(status, "mkl_sparse_order (W)");
            }
            mkl_sparse_destroy(W_mkl_tmp_handle);
            printf("  Column indices within rows ordered.\n");
        } else {
            print_mkl_status(status, "mkl_sparse_d_create_csr (for ordering W)");
        }
    }
    
    DEBUG_MATRIX_INFO(W, "spatial_weight_matrix");
    return W;
}

/* ===============================
 * CUSTOM WEIGHT MATRIX FUNCTIONS
 * =============================== */

/* Detect weight matrix file format automatically */
int detect_weight_matrix_format(const char* filename) {
    if (!filename) {
        fprintf(stderr, "Error: NULL filename in detect_weight_matrix_format\n");
        return WEIGHT_FORMAT_AUTO;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open weight matrix file '%s' for format detection: %s\n", 
                filename, strerror(errno));
        return WEIGHT_FORMAT_AUTO;
    }
    
    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len = getline(&line, &line_buf_size, fp);
    
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty weight matrix file '%s'\n", filename);
        if (line) free(line);
        fclose(fp);
        return WEIGHT_FORMAT_AUTO;
    }
    
    // Trim whitespace
    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
        line[--line_len] = '\0';
    }
    
    // Count tab-separated fields in first line
    char* line_copy = strdup(line);
    if (!line_copy) {
        perror("strdup in detect_weight_matrix_format");
        if (line) free(line);
        fclose(fp);
        return WEIGHT_FORMAT_AUTO;
    }
    
    int field_count = 0;
    char* token = strtok(line_copy, "\t");
    while (token != NULL) {
        field_count++;
        token = strtok(NULL, "\t");
    }
    free(line_copy);
    
    int detected_format;
    if (field_count == 3) {
        // Check if first field looks like a coordinate or spot name
        char* line_copy2 = strdup(line);
        if (line_copy2) {
            char* first_field = strtok(line_copy2, "\t");
            if (first_field) {
                if (strstr(first_field, "x") && strpbrk(first_field, "0123456789")) {
                    detected_format = WEIGHT_FORMAT_SPARSE_COO;
                } else {
                    detected_format = WEIGHT_FORMAT_SPARSE_TSV;
                }
            } else {
                detected_format = WEIGHT_FORMAT_SPARSE_TSV;
            }
            free(line_copy2);
        } else {
            detected_format = WEIGHT_FORMAT_SPARSE_TSV;
        }
    } else if (field_count > 3) {
        detected_format = WEIGHT_FORMAT_DENSE;
    } else {
        detected_format = WEIGHT_FORMAT_SPARSE_TSV;
    }
    
    if (line) free(line);
    fclose(fp);
    
    const char* format_names[] = {"AUTO", "DENSE", "SPARSE_COO", "SPARSE_TSV"};
    printf("Detected weight matrix format: %s\n", format_names[detected_format]);
    
    return detected_format;
}

/* Validate that weight matrix is compatible with expression data */
int validate_weight_matrix(const SparseMatrix* W, char** spot_names, MKL_INT n_spots) {
    if (!W || !spot_names) {
        fprintf(stderr, "Error: NULL parameters in validate_weight_matrix\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    if (W->nrows != n_spots || W->ncols != n_spots) {
        fprintf(stderr, "Error: Weight matrix dimensions (%lldx%lld) don't match number of spots (%lld)\n",
                (long long)W->nrows, (long long)W->ncols, (long long)n_spots);
        return MORANS_I_ERROR_PARAMETER;
    }
    
    // Check for negative weights
    MKL_INT negative_weights = 0;
    for (MKL_INT i = 0; i < W->nnz; i++) {
        if (W->values[i] < 0.0) {
            negative_weights++;
        }
    }
    
    if (negative_weights > 0) {
        fprintf(stderr, "Warning: Found %lld negative weights in custom weight matrix\n", 
                (long long)negative_weights);
    }
    
    double weight_sum = calculate_weight_sum(W);
    printf("Custom weight matrix validation: %lldx%lld matrix, %lld non-zeros, sum = %.6f\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz, weight_sum);
    
    if (fabs(weight_sum) < ZERO_STD_THRESHOLD) {
        fprintf(stderr, "Warning: Sum of custom weights is near zero (%.6e). Moran's I may be undefined.\n", 
                weight_sum);
    }
    
    return MORANS_I_SUCCESS;
}

/* ===============================
 * VST FILE PARSING (REFACTORED)
 * =============================== */

/* Parse VST file header and extract column names */
static int parse_vst_header(FILE* fp, char** line, size_t* line_buf_size,
                           MKL_INT* n_spots_out, char*** colnames_out) {
    if (!fp || !line || !line_buf_size || !n_spots_out || !colnames_out) {
        return MORANS_I_ERROR_PARAMETER;
    }
    
    *n_spots_out = 0;
    *colnames_out = NULL;
    
    ssize_t line_len = getline(line, line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty or unreadable header in VST file.\n");
        return MORANS_I_ERROR_FILE;
    }
    
    // Trim line
    while (line_len > 0 && ((*line)[line_len - 1] == '\n' || (*line)[line_len - 1] == '\r')) {
        (*line)[--line_len] = '\0';
    }
    
    // Count fields
    char* header_copy = strdup(*line);
    if (!header_copy) {
        perror("strdup for header counting");
        return MORANS_I_ERROR_MEMORY;
    }
    
    MKL_INT field_count = 0;
    char* token = strtok(header_copy, "\t");
    while (token != NULL) {
        field_count++;
        token = strtok(NULL, "\t");
    }
    free(header_copy);
    
    if (field_count < 1) {
        fprintf(stderr, "Error: Header has %lld fields. Expected at least 1.\n", (long long)field_count);
        return MORANS_I_ERROR_FILE;
    }
    
    *n_spots_out = field_count; // All header fields are spot IDs
    
    // Allocate colnames array
    *colnames_out = (char**)malloc((size_t)field_count * sizeof(char*));
    if (!*colnames_out) {
        perror("malloc for colnames");
        return MORANS_I_ERROR_MEMORY;
    }
    
    // Initialize all to NULL for safe cleanup
    for (MKL_INT i = 0; i < field_count; i++) {
        (*colnames_out)[i] = NULL;
    }
    
    // Parse column names
    char* header_copy2 = strdup(*line);
    if (!header_copy2) {
        perror("strdup for header parsing");
        // Cleanup
        for (MKL_INT i = 0; i < field_count; i++) {
            free((*colnames_out)[i]);
        }
        free(*colnames_out);
        *colnames_out = NULL;
        return MORANS_I_ERROR_MEMORY;
    }
    
    token = strtok(header_copy2, "\t");
    MKL_INT col_idx = 0;
    while (token && col_idx < field_count) {
        char* trimmed = trim_whitespace_inplace(token);
        if (strlen(trimmed) > 0) {
            (*colnames_out)[col_idx] = strdup(trimmed);
            if (!(*colnames_out)[col_idx]) {
                perror("strdup for column name");
                free(header_copy2);
                // Cleanup already allocated names
                for (MKL_INT i = 0; i < col_idx; i++) {
                    free((*colnames_out)[i]);
                }
                free(*colnames_out);
                *colnames_out = NULL;
                return MORANS_I_ERROR_MEMORY;
            }
        } else {
            char default_name[32];
            snprintf(default_name, sizeof(default_name), "Spot_%lld", (long long)col_idx + 1);
            (*colnames_out)[col_idx] = strdup(default_name);
            if (!(*colnames_out)[col_idx]) {
                perror("strdup for default column name");
                free(header_copy2);
                for (MKL_INT i = 0; i < col_idx; i++) {
                    free((*colnames_out)[i]);
                }
                free(*colnames_out);
                *colnames_out = NULL;
                return MORANS_I_ERROR_MEMORY;
            }
        }
        col_idx++;
        token = strtok(NULL, "\t");
    }
    
    free(header_copy2);
    DEBUG_PRINT("Parsed VST header: %lld columns", (long long)*n_spots_out);
    return MORANS_I_SUCCESS;
}

/* Count genes in VST file */
static int count_vst_genes(FILE* fp, char** line, size_t* line_buf_size, MKL_INT* n_genes_out) {
    if (!fp || !line || !line_buf_size || !n_genes_out) {
        return MORANS_I_ERROR_PARAMETER;
    }
    
    *n_genes_out = 0;
    long current_pos = ftell(fp);
    
    ssize_t line_len;
    while ((line_len = getline(line, line_buf_size, fp)) > 0) {
        char* p = *line;
        while(isspace((unsigned char)*p)) p++;
        if(*p != '\0') (*n_genes_out)++;
    }
    
    fseek(fp, current_pos, SEEK_SET);
    DEBUG_PRINT("Counted VST genes: %lld", (long long)*n_genes_out);
    return MORANS_I_SUCCESS;
}

/* Read VST data rows */
static int read_vst_data_rows(FILE* fp, char** line, size_t* line_buf_size,
                             DenseMatrix* matrix, MKL_INT n_genes_expected, MKL_INT n_spots_expected) {
    if (!fp || !line || !line_buf_size || !matrix || !matrix->values || !matrix->rownames) {
        return MORANS_I_ERROR_PARAMETER;
    }
    
    MKL_INT gene_idx = 0;
    int file_lineno = 1; // Header was line 1
    ssize_t line_len;
    
    while ((line_len = getline(line, line_buf_size, fp)) > 0 && gene_idx < n_genes_expected) {
        file_lineno++;
        
        // Skip blank lines
        char* p = *line;
        while(isspace((unsigned char)*p)) p++;
        if(*p == '\0') continue;
        
        if (gene_idx >= n_genes_expected) {
            fprintf(stderr, "Warning: More data rows than estimated. Stopping at gene %lld (file line %d).\n", 
                   (long long)n_genes_expected, file_lineno);
            break;
        }
        
        // Trim EOL
        while (line_len > 0 && ((*line)[line_len - 1] == '\n' || (*line)[line_len - 1] == '\r')) {
            (*line)[--line_len] = '\0';
        }
        if(line_len == 0) continue;
        
        char* data_row_copy = strdup(*line);
        if (!data_row_copy) {
            perror("strdup data line");
            return MORANS_I_ERROR_MEMORY;
        }
        
        char* token = strtok(data_row_copy, "\t");
        if (!token) {
            fprintf(stderr, "Error: No gene name found on line %d\n", file_lineno);
            free(data_row_copy);
            return MORANS_I_ERROR_FILE;
        }
        
        // Store gene name
        matrix->rownames[gene_idx] = strdup(token);
        if (!matrix->rownames[gene_idx]) {
            perror("strdup gene name");
            free(data_row_copy);
            return MORANS_I_ERROR_MEMORY;
        }
        
        // Read expression values
        for (MKL_INT spot_idx = 0; spot_idx < n_spots_expected; ++spot_idx) {
            token = strtok(NULL, "\t");
            if (!token) {
                fprintf(stderr, "Error: File line %d, gene '%s': Expected %lld expression values, found only %lld.\n",
                        file_lineno, matrix->rownames[gene_idx], (long long)n_spots_expected, (long long)spot_idx);
                free(data_row_copy);
                return MORANS_I_ERROR_FILE;
            }
            
            char* endptr;
            errno = 0;
            double val = strtod(token, &endptr);
            if (errno == ERANGE || (*endptr != '\0' && !isspace((unsigned char)*endptr)) || endptr == token) {
                fprintf(stderr, "Error: Invalid number '%s' at file line %d, gene '%s', spot column %lld.\n",
                        token, file_lineno, matrix->rownames[gene_idx], (long long)spot_idx + 1);
                free(data_row_copy);
                return MORANS_I_ERROR_FILE;
            }
            matrix->values[gene_idx * n_spots_expected + spot_idx] = val;
        }
        
        // Check for extra columns
        if (strtok(NULL, "\t") != NULL) {
            fprintf(stderr, "Warning: Line %d has more columns than expected (%lld spots). Extra data ignored.\n",
                   file_lineno, (long long)n_spots_expected);
        }
        
        free(data_row_copy);
        gene_idx++;
    }
    
    if (gene_idx != n_genes_expected) {
        fprintf(stderr, "Warning: Read %lld genes, but estimated %lld. Adjusting matrix dimensions.\n",
                (long long)gene_idx, (long long)n_genes_expected);
        matrix->nrows = gene_idx;
    }
    
    DEBUG_PRINT("Read VST data: %lld genes processed", (long long)gene_idx);
    return MORANS_I_SUCCESS;
}

/* Read data from VST file (refactored) */
DenseMatrix* read_vst_file(const char* filename) {
    if (!filename) {
        fprintf(stderr, "Error: Null filename provided to read_vst_file\n");
        return NULL;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open '%s': %s\n", filename, strerror(errno));
        return NULL;
    }
    
    char *line = NULL;
    size_t line_buf_size = 0;
    DenseMatrix* matrix = NULL;
    char** colnames = NULL;
    cleanup_list_t cleanup;
    cleanup_list_init(&cleanup);
    
    printf("Reading VST file '%s'...\n", filename);
    
    // Parse header
    MKL_INT n_spots, n_genes;
    if (parse_vst_header(fp, &line, &line_buf_size, &n_spots, &colnames) != MORANS_I_SUCCESS) {
        goto cleanup_and_exit;
    }
    
    // Count genes
    if (count_vst_genes(fp, &line, &line_buf_size, &n_genes) != MORANS_I_SUCCESS) {
        goto cleanup_and_exit;
    }
    
    if (n_genes == 0 || n_spots == 0) {
        fprintf(stderr, "Error: No data found in VST file (genes=%lld, spots=%lld).\n", 
                (long long)n_genes, (long long)n_spots);
        goto cleanup_and_exit;
    }
    
    if (validate_matrix_dimensions(n_genes, n_spots, "VST file") != MORANS_I_SUCCESS) {
        goto cleanup_and_exit;
    }
    
    // Allocate matrix
    matrix = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!matrix) {
        perror("malloc DenseMatrix");
        goto cleanup_and_exit;
    }
    
    matrix->nrows = n_genes;
    matrix->ncols = n_spots;
    matrix->rownames = NULL;
    matrix->colnames = NULL;
    matrix->values = NULL;
    
    // Allocate components with overflow protection
    size_t values_size;
    if (safe_multiply_size_t(n_genes, n_spots, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Matrix dimensions too large for VST file\n");
        free(matrix);
        matrix = NULL;
        goto cleanup_and_exit;
    }
    
    matrix->values = (double*)mkl_malloc(values_size, 64);
    matrix->rownames = (char**)calloc(n_genes, sizeof(char*));
    matrix->colnames = colnames; // Transfer ownership
    colnames = NULL; // Prevent double-free
    
    if (!matrix->values || !matrix->rownames || !matrix->colnames) {
        perror("Failed to allocate matrix components");
        if (matrix) {
            if (matrix->values) mkl_free(matrix->values);
            free(matrix->rownames);
            // Don't free colnames here as it was already transferred
            free(matrix);
        }
        matrix = NULL;
        goto cleanup_and_exit;
    }
    
    // Read data rows
    rewind(fp);
    getline(&line, &line_buf_size, fp); // Skip header
    if (read_vst_data_rows(fp, &line, &line_buf_size, matrix, n_genes, n_spots) != MORANS_I_SUCCESS) {
        free_dense_matrix(matrix);
        matrix = NULL;
        goto cleanup_and_exit;
    }
    
    printf("Successfully loaded VST data: %lld genes x %lld spots from '%s'.\n",
           (long long)matrix->nrows, (long long)matrix->ncols, filename);

cleanup_and_exit:
    if (line) free(line);
    if (colnames) {
        for (MKL_INT i = 0; i < n_spots; i++) {
            free(colnames[i]);
        }
        free(colnames);
    }
    fclose(fp);
    cleanup_list_destroy(&cleanup);
    
    DEBUG_MATRIX_INFO(matrix, "loaded_vst");
    return matrix;
}

/* ===============================
 * MORAN'S I CALCULATION FUNCTIONS
 * =============================== */

/* Calculate pairwise Moran's I matrix: Result = (X_transpose * W * X) / S0 with row normalization support */
DenseMatrix* calculate_morans_i(const DenseMatrix* X, const SparseMatrix* W, int row_normalized) {
    if (!X || !W || !X->values) {
        fprintf(stderr, "Error: Invalid parameters provided to calculate_morans_i\n");
        return NULL;
    }
    if (W->nnz > 0 && !W->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in calculate_morans_i\n");
        return NULL;
    }

    MKL_INT n_spots = X->nrows;
    MKL_INT n_genes = X->ncols;

    if (n_spots != W->nrows || n_spots != W->ncols) {
        fprintf(stderr, "Error: Dimension mismatch between X (%lld spots x %lld genes) and W (%lldx%lld)\n",
                (long long)n_spots, (long long)n_genes, (long long)W->nrows, (long long)W->ncols);
        return NULL;
    }
    
    if (validate_matrix_dimensions(n_genes, n_genes, "Moran's I result") != MORANS_I_SUCCESS) {
        return NULL;
    }

    if (n_genes == 0) {
        fprintf(stderr, "Warning: n_genes is 0 in calculate_morans_i. Returning empty result matrix.\n");
        DenseMatrix* res_empty = (DenseMatrix*)calloc(1, sizeof(DenseMatrix));
        if(!res_empty) {
            perror("calloc for empty moran's I result"); 
            return NULL;
        }
        if (X->colnames) {
            res_empty->rownames = calloc(0, sizeof(char*)); 
            res_empty->colnames = calloc(0, sizeof(char*));
        }
        return res_empty;
    }

    printf("Calculating Moran's I for %lld genes using %lld spots (Matrix approach: X_T * W * X)%s...\n",
           (long long)n_genes, (long long)n_spots, 
           row_normalized ? " with row-normalized weights" : "");

    DenseMatrix* result = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!result) {
        perror("Failed alloc result struct for Moran's I");
        return NULL;
    }
    
    result->nrows = n_genes;
    result->ncols = n_genes;
    result->rownames = NULL;
    result->colnames = NULL;
    result->values = NULL;
    
    // Allocate with overflow protection
    size_t result_size;
    if (safe_multiply_size_t(n_genes, n_genes, &result_size) != 0 ||
        safe_multiply_size_t(result_size, sizeof(double), &result_size) != 0) {
        fprintf(stderr, "Error: Result matrix dimensions too large\n");
        free(result);
        return NULL;
    }
    
    result->values = (double*)mkl_malloc(result_size, 64);
    result->rownames = (char**)calloc(n_genes, sizeof(char*));
    result->colnames = (char**)calloc(n_genes, sizeof(char*));

    if (!result->values || !result->rownames || !result->colnames) {
        perror("Failed alloc result data for Moran's I");
        free_dense_matrix(result);
        return NULL;
    }

    // Copy gene names
    for (MKL_INT i = 0; i < n_genes; i++) {
        if (X->colnames && X->colnames[i]) {
            result->rownames[i] = strdup(X->colnames[i]);
            result->colnames[i] = strdup(X->colnames[i]);
            if (!result->rownames[i] || !result->colnames[i]) {
                perror("Failed to duplicate gene names for Moran's I result");
                free_dense_matrix(result);
                return NULL;
            }
        } else {
            char default_name_buf[32];
            snprintf(default_name_buf, sizeof(default_name_buf), "Gene%lld", (long long)i);
            result->rownames[i] = strdup(default_name_buf);
            result->colnames[i] = strdup(default_name_buf);
            if (!result->rownames[i] || !result->colnames[i]) {
                perror("Failed to duplicate default gene names");
                free_dense_matrix(result);
                return NULL;
            }
        }
    }

    /* Calculate scaling factor based on row normalization */
    double scaling_factor;
    if (row_normalized) {
        scaling_factor = 1.0;
        printf("  Using row-normalized weights (scaling factor = 1.0)\n");
    } else {
        double S0 = calculate_weight_sum(W);
        printf("  Sum of weights S0: %.6f\n", S0);

        if (fabs(S0) < DBL_EPSILON) {
            fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Moran's I results will be NaN/Inf or 0.\n", S0);
            if (S0 == 0.0) {
                for(size_t i=0; i < (size_t)n_genes * n_genes; ++i) result->values[i] = NAN;
                return result;
            }
        }
        
        scaling_factor = 1.0 / S0;
        printf("  Using 1/S0 = %.6e as scaling factor\n", scaling_factor);
    }

    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(
        &W_mkl, SPARSE_INDEX_BASE_ZERO, W->nrows, W->ncols,
        W->row_ptr, W->row_ptr + 1, W->col_ind, W->values);

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W)");
        free_dense_matrix(result);
        return NULL;
    }

    if (W->nnz > 0) {
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_optimize (W)");
        }
    }

    printf("  Step 1: Calculating Temp_WX = W * X ...\n");
    
    size_t temp_size;
    if (safe_multiply_size_t(n_spots, n_genes, &temp_size) != 0 ||
        safe_multiply_size_t(temp_size, sizeof(double), &temp_size) != 0) {
        fprintf(stderr, "Error: Temporary matrix dimensions too large\n");
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }
    
    double* Temp_WX_values = (double*)mkl_malloc(temp_size, 64);
    if (!Temp_WX_values) {
        perror("Failed alloc Temp_WX_values");
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }

    struct matrix_descr descrW;
    descrW.type = SPARSE_MATRIX_TYPE_GENERAL;
    
    double alpha_mm = 1.0, beta_mm = 0.0;

    status = mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha_mm,
        W_mkl,
        descrW,
        SPARSE_LAYOUT_ROW_MAJOR,
        X->values,
        n_genes,
        n_genes,
        beta_mm,
        Temp_WX_values,
        n_genes
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_mm (W * X)");
        mkl_free(Temp_WX_values);
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }

    printf("  Step 2: Calculating Result = (X_T * Temp_WX) * scaling_factor ...\n");
    cblas_dgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        n_genes,
        n_genes,
        n_spots,
        scaling_factor,
        X->values,
        n_genes,
        Temp_WX_values,
        n_genes,
        beta_mm,
        result->values,
        n_genes
    );

    mkl_free(Temp_WX_values);
    mkl_sparse_destroy(W_mkl);

    printf("Moran's I matrix calculation complete%s.\n", 
           row_normalized ? " (row-normalized)" : " and scaled");
    DEBUG_MATRIX_INFO(result, "morans_i_result");
    return result;
}

/* Calculate Moran's I for a single gene with row normalization support */
double calculate_single_gene_moran_i(const double* gene_data, const SparseMatrix* W, MKL_INT n_spots, int row_normalized) {
    if (!gene_data || !W) {
        fprintf(stderr, "Error: Invalid parameters provided to calculate_single_gene_moran_i\n");
        return NAN;
    }
    if (W->nnz > 0 && !W->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in calculate_single_gene_moran_i\n");
        return NAN;
    }

    if (n_spots != W->nrows || n_spots != W->ncols) {
        fprintf(stderr, "Error: n_spots mismatch with W dimensions in calculate_single_gene_moran_i\n");
        return NAN;
    }
    if (n_spots == 0) {
        fprintf(stderr, "Warning: n_spots is 0 in calculate_single_gene_moran_i. Returning NAN.\n");
        return NAN;
    }

    /* Calculate scaling factor based on row normalization */
    double scaling_factor;
    if (row_normalized) {
        scaling_factor = 1.0;
    } else {
        double S0 = calculate_weight_sum(W);
        if (fabs(S0) < DBL_EPSILON) {
            fprintf(stderr, "Warning: S0 is near zero in calculate_single_gene_moran_i. Result is NaN/Inf or 0.\n");
            return (S0 == 0.0) ? NAN : 0.0;
        }
        scaling_factor = 1.0 / S0;
    }

    double* Wz = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!Wz) {
        perror("Failed to allocate Wz");
        return NAN;
    }

    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl, SPARSE_INDEX_BASE_ZERO, 
                                                     W->nrows, W->ncols, W->row_ptr, 
                                                     W->row_ptr + 1, W->col_ind, W->values);
    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W for single_gene_moran_i)");
        mkl_free(Wz); 
        return NAN;
    }
    
    if (W->nnz > 0) {
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_optimize (W for single_gene_moran_i)");
        }
    }

    struct matrix_descr descrW; 
    descrW.type = SPARSE_MATRIX_TYPE_GENERAL;
    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descrW, gene_data, 0.0, Wz);
    mkl_sparse_destroy(W_mkl);

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_mv (W*z for single_gene_moran_i)");
        mkl_free(Wz); 
        return NAN;
    }
    
    double z_T_Wz = cblas_ddot(n_spots, gene_data, 1, Wz, 1);
    mkl_free(Wz);
    return z_T_Wz * scaling_factor;
}

/* Calculate Moran's I between the first gene and all others with row normalization support */
double* calculate_first_gene_vs_all(const DenseMatrix* X, const SparseMatrix* W, double S0_param, int row_normalized) {
    if (!X || !W || !X->values || X->ncols == 0) {
        fprintf(stderr, "Error: Invalid parameters or no genes in X for calculate_first_gene_vs_all\n");
        return NULL;
    }
    if (W->nnz > 0 && !W->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in calculate_first_gene_vs_all\n");
        return NULL;
    }

    MKL_INT n_spots = X->nrows; 
    MKL_INT n_genes = X->ncols;

    /* Calculate scaling factor based on row normalization */
    double scaling_factor;
    if (row_normalized) {
        scaling_factor = 1.0;
    } else {
        double S0 = S0_param;
        if (fabs(S0_param) < DBL_EPSILON) {
            fprintf(stderr, "Warning: S0 passed to calculate_first_gene_vs_all is near-zero (%.4e). Recalculating S0 from W.\n", S0_param);
            S0 = calculate_weight_sum(W);
            if (fabs(S0) < DBL_EPSILON) {
                fprintf(stderr, "Error: Recalculated S0 is also near-zero (%.4e). Results will be NaN/Inf or 0.\n", S0);
            }
        }

        if (S0 == 0.0) {
            scaling_factor = NAN;
        } else {
            scaling_factor = 1.0 / S0;
        }
    }

    double* moran_I_results = (double*)mkl_malloc((size_t)n_genes * sizeof(double), 64);
    if (!moran_I_results) {
        perror("Failed to allocate memory for first_gene_vs_all results"); 
        return NULL;
    }
    
    if (isnan(scaling_factor)) {
        for(MKL_INT g=0; g < n_genes; ++g) moran_I_results[g] = NAN;
        return moran_I_results;
    }

    double* z0_data = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!z0_data) {
        perror("Failed to allocate memory for first gene data (z0)");
        mkl_free(moran_I_results); 
        return NULL;
    }
    
    // Extract first gene (column 0)
    for (MKL_INT i = 0; i < n_spots; i++) {
        z0_data[i] = X->values[i * n_genes + 0];
    }

    double* W_z0 = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!W_z0) {
        perror("Failed to allocate memory for W_z0");
        mkl_free(z0_data); 
        mkl_free(moran_I_results); 
        return NULL;
    }

    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl, SPARSE_INDEX_BASE_ZERO, 
                                                     W->nrows, W->ncols, W->row_ptr, 
                                                     W->row_ptr + 1, W->col_ind, W->values);
    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W for first_gene_vs_all)");
        mkl_free(W_z0); mkl_free(z0_data); mkl_free(moran_I_results); 
        return NULL;
    }
    
    if (W->nnz > 0) {
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_optimize (W for first_gene_vs_all)");
        }
    }

    struct matrix_descr descrW; 
    descrW.type = SPARSE_MATRIX_TYPE_GENERAL;
    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descrW, z0_data, 0.0, W_z0);
    mkl_sparse_destroy(W_mkl);

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_mv (W*z0 for first_gene_vs_all)");
        mkl_free(W_z0); mkl_free(z0_data); mkl_free(moran_I_results); 
        return NULL;
    }

    // Calculate dot products
    #pragma omp parallel for
    for (MKL_INT g = 0; g < n_genes; g++) {
        double dot_product = 0;
        for (MKL_INT spot_idx = 0; spot_idx < n_spots; spot_idx++) {
            dot_product += X->values[spot_idx * n_genes + g] * W_z0[spot_idx];
        }
        moran_I_results[g] = dot_product * scaling_factor;
    }
    
    mkl_free(z0_data); 
    mkl_free(W_z0);
    return moran_I_results;
}

/* ===============================
 * PERMUTATION TESTING (REFACTORED)
 * =============================== */

/* Permutation worker function */
static int permutation_worker(const DenseMatrix* X_original,
                             const SparseMatrix* W,
                             const PermutationParams* params,
                             int thread_id,
                             int start_perm,
                             int end_perm,
                             double scaling_factor,
                             double* local_mean_sum,
                             double* local_var_sum_sq,
                             double* local_p_counts,
                             const DenseMatrix* observed_results) {
    
    MKL_INT n_spots = X_original->nrows;
    MKL_INT n_genes = X_original->ncols;
    size_t matrix_elements = (size_t)n_genes * n_genes;
    
    // Thread-local allocations
    DenseMatrix X_perm;
    X_perm.nrows = n_spots;
    X_perm.ncols = n_genes;
    X_perm.rownames = NULL;
    X_perm.colnames = NULL;
    X_perm.values = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
    
    double* gene_buffer = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    MKL_INT* indices_buffer = (MKL_INT*)mkl_malloc((size_t)n_spots * sizeof(MKL_INT), 64);
    double* temp_WX = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
    double* I_perm_values = (double*)mkl_malloc(matrix_elements * sizeof(double), 64);
    
    if (!X_perm.values || !gene_buffer || !indices_buffer || !temp_WX || !I_perm_values) {
        DEBUG_PRINT("Thread %d: Memory allocation failed", thread_id);
        // Cleanup
        if (X_perm.values) mkl_free(X_perm.values);
        if (gene_buffer) mkl_free(gene_buffer);
        if (indices_buffer) mkl_free(indices_buffer);
        if (temp_WX) mkl_free(temp_WX);
        if (I_perm_values) mkl_free(I_perm_values);
        return -1;
    }
    
    // Create sparse matrix handle
    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl, SPARSE_INDEX_BASE_ZERO,
                                                     W->nrows, W->ncols, W->row_ptr,
                                                     W->row_ptr + 1, W->col_ind, W->values);
    if (status != SPARSE_STATUS_SUCCESS) {
        DEBUG_PRINT("Thread %d: Failed to create sparse matrix", thread_id);
        mkl_free(X_perm.values); mkl_free(gene_buffer); mkl_free(indices_buffer);
        mkl_free(temp_WX); mkl_free(I_perm_values);
        return -1;
    }
    
    if (W->nnz > 0) {
        mkl_sparse_optimize(W_mkl);
    }
    
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    
    unsigned int local_seed = params->seed + thread_id + 1;
    
    // Perform permutations
    for (int perm = start_perm; perm < end_perm; perm++) {
        // Permute each gene independently
        for (MKL_INT j = 0; j < n_genes; j++) {
            // Copy original values
            for (MKL_INT i = 0; i < n_spots; i++) {
                gene_buffer[i] = X_original->values[i * n_genes + j];
                indices_buffer[i] = i;
            }
            
            // Fisher-Yates shuffle
            if (n_spots > 1) {
                for (MKL_INT i = n_spots - 1; i > 0; i--) {
                    MKL_INT k = rand_r(&local_seed) % (i + 1);
                    MKL_INT temp_idx = indices_buffer[i];
                    indices_buffer[i] = indices_buffer[k];
                    indices_buffer[k] = temp_idx;
                }
            }
            
            // Apply permutation
            for (MKL_INT i = 0; i < n_spots; i++) {
                X_perm.values[i * n_genes + j] = gene_buffer[indices_buffer[i]];
            }
        }
        
        // Calculate Moran's I for permuted data
        status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descr,
                                SPARSE_LAYOUT_ROW_MAJOR, X_perm.values, n_genes, n_genes,
                                0.0, temp_WX, n_genes);
        
        if (status != SPARSE_STATUS_SUCCESS) {
            DEBUG_PRINT("Thread %d: Sparse matrix multiplication failed", thread_id);
            continue; // Skip this permutation
        }
        
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n_genes, n_genes, n_spots, scaling_factor,
                    X_perm.values, n_genes, temp_WX, n_genes,
                    0.0, I_perm_values, n_genes);
        
        // Accumulate statistics
        for (size_t idx = 0; idx < matrix_elements; idx++) {
            double perm_val = I_perm_values[idx];
            if (!isfinite(perm_val)) perm_val = 0.0;
            
            local_mean_sum[idx] += perm_val;
            local_var_sum_sq[idx] += perm_val * perm_val;
            
            if (params->p_value_output && local_p_counts && observed_results) {
                if (fabs(perm_val) >= fabs(observed_results->values[idx])) {
                    local_p_counts[idx]++;
                }
            }
        }
    }
    
    // Cleanup
    mkl_sparse_destroy(W_mkl);
    mkl_free(X_perm.values);
    mkl_free(gene_buffer);
    mkl_free(indices_buffer);
    mkl_free(temp_WX);
    mkl_free(I_perm_values);
    
    return 0;
}

/* Run the full permutation test with row normalization support */
PermutationResults* run_permutation_test(const DenseMatrix* X_observed_spots_x_genes,
                                       const SparseMatrix* W_spots_x_spots,
                                       const PermutationParams* params,
                                       int row_normalized) {
    
    if (!X_observed_spots_x_genes || !W_spots_x_spots || !params ||
        !X_observed_spots_x_genes->values || !X_observed_spots_x_genes->colnames) {
        fprintf(stderr, "Error: Invalid parameters provided to run_permutation_test\n");
        return NULL;
    }
    if (W_spots_x_spots->nnz > 0 && !W_spots_x_spots->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in run_permutation_test\n");
        return NULL;
    }

    MKL_INT n_spots = X_observed_spots_x_genes->nrows;
    MKL_INT n_genes = X_observed_spots_x_genes->ncols;
    int n_perm = params->n_permutations;

    if (validate_matrix_dimensions(n_spots, n_genes, "permutation test input") != MORANS_I_SUCCESS) {
        return NULL;
    }

    if (n_genes == 0 || n_spots == 0) {
        fprintf(stderr, "Error: Expression matrix has zero dimensions in run_permutation_test\n");
        return NULL;
    }
    if (n_perm <= 0) {
        fprintf(stderr, "Error: Number of permutations (%d) must be positive\n", n_perm);
        return NULL;
    }

    printf("Running permutation test with %d permutations for %lld genes%s...\n", 
           n_perm, (long long)n_genes, row_normalized ? " (row-normalized weights)" : "");

    /* Calculate scaling factor based on row normalization */
    double scaling_factor;
    if (row_normalized) {
        scaling_factor = 1.0;
        printf("  Permutation Test: Using row-normalized weights (scaling factor = 1.0)\n");
    } else {
        double S0 = calculate_weight_sum(W_spots_x_spots);
        if (fabs(S0) < DBL_EPSILON) {
            fprintf(stderr, "Error: Sum of weights S0 is near-zero (%.4e) in permutation test\n", S0);
            return NULL;
        }
        scaling_factor = 1.0 / S0;
        printf("  Permutation Test: Using S0 = %.6f, scaling factor = %.6e\n", S0, scaling_factor);
    }

    // Calculate observed Moran's I for comparison
    DenseMatrix* observed_results = calculate_morans_i(X_observed_spots_x_genes, W_spots_x_spots, row_normalized);
    if (!observed_results) {
        fprintf(stderr, "Error: Failed to calculate observed Moran's I for permutation test\n");
        return NULL;
    }

    // Allocate results structure
    PermutationResults* results = (PermutationResults*)calloc(1, sizeof(PermutationResults));
    if (!results) {
        perror("Failed to allocate PermutationResults structure");
        free_dense_matrix(observed_results);
        return NULL;
    }

    size_t matrix_elements = (size_t)n_genes * n_genes;
    size_t matrix_bytes;
    if (safe_multiply_size_t(matrix_elements, sizeof(double), &matrix_bytes) != 0) {
        fprintf(stderr, "Error: Matrix size too large for permutation results\n");
        free(results);
        free_dense_matrix(observed_results);
        return NULL;
    }

    // Allocate result matrices
    results->mean_perm = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    results->var_perm = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (params->z_score_output) {
        results->z_scores = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    }
    if (params->p_value_output) {
        results->p_values = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    }

    if (!results->mean_perm || !results->var_perm ||
        (params->z_score_output && !results->z_scores) ||
        (params->p_value_output && !results->p_values)) {
        perror("Failed to allocate result matrix structures");
        free_permutation_results(results);
        free_dense_matrix(observed_results);
        return NULL;
    }

    // Initialize result matrices
    DenseMatrix* matrices[] = {results->mean_perm, results->var_perm, results->z_scores, results->p_values};
    int num_matrices = 2 + (params->z_score_output ? 1 : 0) + (params->p_value_output ? 1 : 0);

    for (int m = 0; m < num_matrices; m++) {
        if (!matrices[m]) continue;
        
        matrices[m]->nrows = n_genes;
        matrices[m]->ncols = n_genes;
        matrices[m]->values = (double*)mkl_calloc(matrix_elements, sizeof(double), 64);
        matrices[m]->rownames = (char**)calloc(n_genes, sizeof(char*));
        matrices[m]->colnames = (char**)calloc(n_genes, sizeof(char*));
        
        if (!matrices[m]->values || !matrices[m]->rownames || !matrices[m]->colnames) {
            perror("Failed to allocate result matrix components");
            free_permutation_results(results);
            free_dense_matrix(observed_results);
            return NULL;
        }
        
        // Copy gene names
        for (MKL_INT i = 0; i < n_genes; i++) {
            const char* gene_name = (X_observed_spots_x_genes->colnames[i]) ? 
                                   X_observed_spots_x_genes->colnames[i] : "UNKNOWN_GENE";
            matrices[m]->rownames[i] = strdup(gene_name);
            matrices[m]->colnames[i] = strdup(gene_name);
            if (!matrices[m]->rownames[i] || !matrices[m]->colnames[i]) {
                perror("Failed to duplicate gene names for permutation results");
                free_permutation_results(results);
                free_dense_matrix(observed_results);
                return NULL;
            }
        }
    }

    // Run permutations using multiple threads
    int num_threads = omp_get_max_threads();
    int perms_per_thread = n_perm / num_threads;
    int remaining_perms = n_perm % num_threads;
    
    printf("Starting permutation loop (%d permutations) using %d OpenMP threads...\n", 
           n_perm, num_threads);
    
    volatile int error_occurred = 0;
    double loop_start_time = get_time();

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int start_perm = thread_id * perms_per_thread;
        int end_perm = start_perm + perms_per_thread;
        if (thread_id == num_threads - 1) {
            end_perm += remaining_perms;
        }
        
        // Thread-local accumulators
        double* local_mean_sum = (double*)mkl_calloc(matrix_elements, sizeof(double), 64);
        double* local_var_sum_sq = (double*)mkl_calloc(matrix_elements, sizeof(double), 64);
        double* local_p_counts = NULL;
        
        if (params->p_value_output) {
            local_p_counts = (double*)mkl_calloc(matrix_elements, sizeof(double), 64);
        }
        
        if (!local_mean_sum || !local_var_sum_sq || 
            (params->p_value_output && !local_p_counts)) {
            #pragma omp critical
            {
                fprintf(stderr, "Thread %d: Failed to allocate local buffers\n", thread_id);
                error_occurred = 1;
            }
        } else if (!error_occurred) {
            // Run permutations for this thread
            int worker_result = permutation_worker(X_observed_spots_x_genes, W_spots_x_spots,
                                                  params, thread_id, start_perm, end_perm,
                                                  scaling_factor, local_mean_sum, local_var_sum_sq,
                                                  local_p_counts, observed_results);
            
            if (worker_result != 0) {
                #pragma omp critical
                {
                    fprintf(stderr, "Thread %d: Permutation worker failed\n", thread_id);
                    error_occurred = 1;
                }
            } else {
                // Merge results
                #pragma omp critical
                {
                    if (!error_occurred) {
                        for (size_t k = 0; k < matrix_elements; k++) {
                            results->mean_perm->values[k] += local_mean_sum[k];
                            results->var_perm->values[k] += local_var_sum_sq[k];
                            if (params->p_value_output && local_p_counts) {
                                results->p_values->values[k] += local_p_counts[k];
                            }
                        }
                    }
                }
            }
        }
        
        // Cleanup thread-local buffers
        if (local_mean_sum) mkl_free(local_mean_sum);
        if (local_var_sum_sq) mkl_free(local_var_sum_sq);
        if (local_p_counts) mkl_free(local_p_counts);
    }

    double loop_end_time = get_time();
    printf("Permutation loop completed in %.2f seconds\n", loop_end_time - loop_start_time);

    if (error_occurred) {
        fprintf(stderr, "Error occurred during permutation execution\n");
        free_permutation_results(results);
        free_dense_matrix(observed_results);
        return NULL;
    }

    // Finalize statistics
    double inv_n_perm = 1.0 / (double)n_perm;
    for (MKL_INT r = 0; r < n_genes; r++) {
        for (MKL_INT c = 0; c < n_genes; c++) {
            MKL_INT idx = r * n_genes + c;
            
            double sum_val = results->mean_perm->values[idx];
            double sum_sq_val = results->var_perm->values[idx];
            
            double mean_perm = sum_val * inv_n_perm;
            double var_perm = (sum_sq_val * inv_n_perm) - (mean_perm * mean_perm);
            
            if (var_perm < 0.0 && var_perm > -ZERO_STD_THRESHOLD) {
                var_perm = 0.0;
            } else if (var_perm < 0.0) {
                fprintf(stderr, "Warning: Negative variance (%.4e) for gene pair (%lld,%lld)\n",
                        var_perm, (long long)r, (long long)c);
                var_perm = 0.0;
            }
            
            results->mean_perm->values[idx] = mean_perm;
            results->var_perm->values[idx] = var_perm;
            
            if (params->p_value_output && results->p_values) {
                results->p_values->values[idx] = (results->p_values->values[idx] + 1.0) / (double)(n_perm + 1);
            }
            
            if (params->z_score_output && results->z_scores) {
                double std_dev = sqrt(var_perm);
                double observed_val = observed_results->values[idx];
                
                if (std_dev < ZERO_STD_THRESHOLD) {
                    if (fabs(observed_val - mean_perm) < ZERO_STD_THRESHOLD) {
                        results->z_scores->values[idx] = 0.0;
                    } else {
                        results->z_scores->values[idx] = (observed_val > mean_perm) ? INFINITY : -INFINITY;
                    }
                } else {
                    results->z_scores->values[idx] = (observed_val - mean_perm) / std_dev;
                }
            }
        }
    }

    free_dense_matrix(observed_results);
    printf("Permutation test complete.\n");
    return results;
}

/* ===============================
 * SPATIAL UTILITY FUNCTIONS
 * =============================== */

/* Gaussian distance decay function */
double decay(double d_physical, double sigma) {
    if (d_physical < 0.0) d_physical = 0.0;

    if (sigma <= ZERO_STD_THRESHOLD) {
        return (fabs(d_physical) < ZERO_STD_THRESHOLD) ? 1.0 : 0.0;
    }
    if (d_physical > 3.0 * sigma) {
        return 0.0;
    }
    return exp(-(d_physical * d_physical) / (2.0 * sigma * sigma));
}

/* Infer sigma from data for single-cell datasets */
double infer_sigma_from_data(const SpotCoordinates* coords, double coord_scale) {
    if (!coords || !coords->spot_row || !coords->spot_col || !coords->valid_mask) {
        fprintf(stderr, "Error: Null or invalid coordinates in infer_sigma_from_data\n");
        return 100.0;
    }
    if (coords->valid_spots < 2) {
        fprintf(stderr, "Warning: Not enough valid spots (%lld) to infer sigma, using default 100.0\n", 
                (long long)coords->valid_spots);
        return 100.0;
    }

    double sum_min_dist_sq = 0.0;
    MKL_INT count_valid_nn = 0;
    int max_samples_for_sigma = 1000;
    MKL_INT sample_step = (coords->valid_spots > max_samples_for_sigma) ? 
                         (coords->valid_spots / max_samples_for_sigma) : 1;
    if(sample_step == 0) sample_step = 1;

    printf("Inferring sigma: sampling up to %d spots (step %lld) from %lld valid spots...\n", 
           max_samples_for_sigma, (long long)sample_step, (long long)coords->valid_spots);

    // Create lists of valid coordinates
    double* valid_x_list = (double*)malloc((size_t)coords->valid_spots * sizeof(double));
    double* valid_y_list = (double*)malloc((size_t)coords->valid_spots * sizeof(double));
    
    if (!valid_x_list || !valid_y_list) {
        perror("Failed to allocate lists for sigma inference");
        free(valid_x_list);
        free(valid_y_list);
        return 100.0;
    }
    
    MKL_INT current_valid_idx = 0;
    for (MKL_INT i = 0; i < coords->total_spots; ++i) {
        if (coords->valid_mask[i] && current_valid_idx < coords->valid_spots) {
            valid_x_list[current_valid_idx] = (double)coords->spot_col[i] / coord_scale;
            valid_y_list[current_valid_idx] = (double)coords->spot_row[i] / coord_scale;
            current_valid_idx++;
        }
    }
    
    MKL_INT actual_valid = current_valid_idx;
    if(actual_valid < 2) {
        fprintf(stderr, "Warning: Less than 2 actual valid spots for sigma inference\n");
        free(valid_x_list);
        free(valid_y_list);
        return 100.0;
    }

    #pragma omp parallel for reduction(+:sum_min_dist_sq) reduction(+:count_valid_nn) schedule(dynamic)
    for (MKL_INT i = 0; i < actual_valid; i += sample_step) {
        double x_i = valid_x_list[i];
        double y_i = valid_y_list[i];
        double min_dist_sq = DBL_MAX;

        for (MKL_INT j = 0; j < actual_valid; j++) {
            if (i == j) continue;
            
            double dx = x_i - valid_x_list[j];
            double dy = y_i - valid_y_list[j];
            double dist_sq = dx * dx + dy * dy;
            
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
        
        if (min_dist_sq < DBL_MAX && min_dist_sq > 0) {
            sum_min_dist_sq += min_dist_sq;
            count_valid_nn++;
        }
    }
    
    free(valid_x_list);
    free(valid_y_list);

    if (count_valid_nn == 0) {
        fprintf(stderr, "Warning: Could not calculate nearest neighbor distances for sigma inference\n");
        return 100.0;
    }
    
    double avg_min_dist_sq = sum_min_dist_sq / count_valid_nn;
    double inferred_sigma = sqrt(avg_min_dist_sq);
    
    printf("Inferred sigma = %.4f based on average nearest neighbor distance from %lld samples\n", 
           inferred_sigma, (long long)count_valid_nn);
    return (inferred_sigma > ZERO_STD_THRESHOLD) ? inferred_sigma : 100.0;
}

/* Create spatial distance matrix (decay lookup table) */
DenseMatrix* create_distance_matrix(MKL_INT max_radius_grid_units, 
                                   int platform_mode,
                                   double custom_sigma_physical, 
                                   double coord_scale_for_sc) {
    if (max_radius_grid_units <= 0) {
        fprintf(stderr, "Error: max_radius_grid_units must be positive in create_distance_matrix\n");
        return NULL;
    }

    if (validate_matrix_dimensions(max_radius_grid_units, 2 * max_radius_grid_units, 
                                  "distance matrix") != MORANS_I_SUCCESS) {
        return NULL;
    }

    DenseMatrix* matrix = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!matrix) {
        perror("Failed to allocate DenseMatrix structure for decay matrix");
        return NULL;
    }
    
    matrix->nrows = max_radius_grid_units;      
    matrix->ncols = 2 * max_radius_grid_units;
    matrix->rownames = NULL;
    matrix->colnames = NULL;
    
    size_t values_size;
    if (safe_multiply_size_t(matrix->nrows, matrix->ncols, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Distance matrix dimensions too large\n");
        free(matrix);
        return NULL;
    }
    
    matrix->values = (double*)mkl_malloc(values_size, 64);
    if (!matrix->values) {
        perror("Failed to allocate values for decay matrix");
        free(matrix);
        return NULL;
    }

    double sigma_eff; 
    if (custom_sigma_physical > 0.0) {
        sigma_eff = custom_sigma_physical;
    } else {
        switch (platform_mode) {
            case VISIUM:       sigma_eff = 100.0; break;
            case OLD:          sigma_eff = 200.0; break;
            case SINGLE_CELL:  sigma_eff = 100.0; break;
            default:           sigma_eff = 100.0; break;
        }
    }

    printf("Distance Matrix: %lldx%lld, Platform: %d, Sigma: %.4f\n",
           (long long)matrix->nrows, (long long)matrix->ncols, platform_mode, sigma_eff);

    if (platform_mode == VISIUM || platform_mode == OLD) {
        double shift_factor_y = (platform_mode == VISIUM) ? (0.5 * sqrt(3.0)) : 0.5;
        double dist_unit = (platform_mode == VISIUM) ? 100.0 : 200.0;

        #pragma omp parallel for collapse(2) schedule(static)
        for (MKL_INT r = 0; r < matrix->nrows; r++) {
            for (MKL_INT c = 0; c < matrix->ncols; c++) {
                double x_dist = 0.5 * (double)c * dist_unit;
                double y_dist = (double)r * shift_factor_y * dist_unit;
                double d_total = sqrt(x_dist * x_dist + y_dist * y_dist);
                matrix->values[r * matrix->ncols + c] = decay(d_total, sigma_eff);
            }
        }
    } else if (platform_mode == SINGLE_CELL) {
        if (coord_scale_for_sc <= ZERO_STD_THRESHOLD) {
            fprintf(stderr, "Error: Invalid coord_scale_for_sc (%.4f) for SINGLE_CELL\n", coord_scale_for_sc);
            #pragma omp parallel for collapse(2) schedule(static)
            for (MKL_INT r = 0; r < matrix->nrows; r++) {
                for (MKL_INT c = 0; c < matrix->ncols; c++) {
                    matrix->values[r * matrix->ncols + c] = (r==0 && c==0) ? 1.0: 0.0;
                }
            }
        } else {
            printf("  SC physical distances from grid shifts / coord_scale: %.4f\n", coord_scale_for_sc);
            #pragma omp parallel for collapse(2) schedule(static)
            for (MKL_INT r = 0; r < matrix->nrows; r++) {
                for (MKL_INT c = 0; c < matrix->ncols; c++) {
                    double r_phys = (double)r / coord_scale_for_sc;
                    double c_phys = (double)c / coord_scale_for_sc;
                    double d_total = sqrt(c_phys * c_phys + r_phys * r_phys);
                    matrix->values[r * matrix->ncols + c] = decay(d_total, sigma_eff);
                }
            }
        }
    } else {
        fprintf(stderr, "Error: Unknown platform_mode %d in create_distance_matrix\n", platform_mode);
        free_dense_matrix(matrix);
        return NULL;
    }

    printf("Distance matrix (decay lookup table) created.\n");
    DEBUG_MATRIX_INFO(matrix, "distance_matrix");
    return matrix;
}

/* ===============================
 * COORDINATE PROCESSING FUNCTIONS
 * =============================== */

/* Extract coordinates from column names (spot names like "RxC") */
SpotCoordinates* extract_coordinates(char** column_names, MKL_INT n_columns) {
    if (!column_names && n_columns > 0) {
        fprintf(stderr, "Error: Null column_names with n_columns > 0 for extract_coordinates\n");
        return NULL;
    }
    if (n_columns < 0) {
        fprintf(stderr, "Error: Negative n_columns for extract_coordinates\n");
        return NULL;
    }

    SpotCoordinates* coords = (SpotCoordinates*)malloc(sizeof(SpotCoordinates));
    if (!coords) { 
        perror("malloc SpotCoordinates"); 
        return NULL; 
    }

    coords->total_spots = n_columns;
    coords->spot_row = NULL; 
    coords->spot_col = NULL; 
    coords->valid_mask = NULL; 
    coords->spot_names = NULL;
    coords->valid_spots = 0;

    if (n_columns > 0) {
        coords->spot_row = (MKL_INT*)malloc((size_t)n_columns * sizeof(MKL_INT));
        coords->spot_col = (MKL_INT*)malloc((size_t)n_columns * sizeof(MKL_INT));
        coords->valid_mask = (int*)calloc(n_columns, sizeof(int));
        coords->spot_names = (char**)malloc((size_t)n_columns * sizeof(char*));
        
        if (!coords->spot_row || !coords->spot_col || !coords->valid_mask || !coords->spot_names) {
            perror("Failed to allocate memory for coordinate arrays");
            free_spot_coordinates(coords); 
            return NULL;
        }
    } else {
        return coords;
    }

    regex_t regex;
    int reti = regcomp(&regex, "^([0-9]+)x([0-9]+)$", REG_EXTENDED);
    if (reti) {
        char errbuf[100]; 
        regerror(reti, &regex, errbuf, sizeof(errbuf));
        fprintf(stderr, "Error: Could not compile regex for coordinate extraction: %s\n", errbuf);
        free_spot_coordinates(coords); 
        return NULL;
    }

    printf("Extracting coordinates from %lld column names using regex...\n", (long long)n_columns);
    MKL_INT current_valid_count = 0;
    
    for (MKL_INT i = 0; i < n_columns; i++) {
        coords->spot_names[i] = NULL;
        coords->spot_row[i] = -1; 
        coords->spot_col[i] = -1;

        if (!column_names[i]) {
            fprintf(stderr, "Warning: Spot name at index %lld is NULL\n", (long long)i);
            coords->valid_mask[i] = 0;
            continue;
        }
        
        coords->spot_names[i] = strdup(column_names[i]);
        if (!coords->spot_names[i]) { 
            perror("strdup spot_name in extract_coordinates"); 
            regfree(&regex); 
            free_spot_coordinates(coords); 
            return NULL;
        }

        regmatch_t matches[3];
        reti = regexec(&regex, column_names[i], 3, matches, 0);
        if (reti == 0) {
            char val_str_buffer[64];
            
            // Extract row
            regoff_t len_row = matches[1].rm_eo - matches[1].rm_so;
            if (len_row > 0 && len_row < (regoff_t)sizeof(val_str_buffer)) {
                strncpy(val_str_buffer, column_names[i] + matches[1].rm_so, len_row);
                val_str_buffer[len_row] = '\0';
                coords->spot_row[i] = strtoll(val_str_buffer, NULL, 10);
            } else { 
                coords->spot_row[i] = -1; 
            }

            // Extract col
            regoff_t len_col = matches[2].rm_eo - matches[2].rm_so;
            if (len_col > 0 && len_col < (regoff_t)sizeof(val_str_buffer)) {
                strncpy(val_str_buffer, column_names[i] + matches[2].rm_so, len_col);
                val_str_buffer[len_col] = '\0';
                coords->spot_col[i] = strtoll(val_str_buffer, NULL, 10);
            } else { 
                coords->spot_col[i] = -1; 
            }

            if (coords->spot_row[i] >= 0 && coords->spot_col[i] >= 0) {
                coords->valid_mask[i] = 1; 
                current_valid_count++;
            } else { 
                coords->valid_mask[i] = 0;
                coords->spot_row[i] = -1; 
                coords->spot_col[i] = -1;
            }
        } else if (reti == REG_NOMATCH) {
            coords->valid_mask[i] = 0;
        } else {
            char errbuf[100]; 
            regerror(reti, &regex, errbuf, sizeof(errbuf));
            fprintf(stderr, "Warning: Regex match failed for '%s': %s\n", column_names[i], errbuf);
            coords->valid_mask[i] = 0;
        }
    }
    
    regfree(&regex); 
    coords->valid_spots = current_valid_count;
    printf("Coordinate extraction complete: %lld valid coordinates out of %lld total spots\n", 
           (long long)coords->valid_spots, (long long)n_columns);
    return coords;
}

/* ===============================
 * FILE I/O AND SAVING FUNCTIONS
 * =============================== */

/* Save results to file */
int save_results(const DenseMatrix* result_matrix, const char* output_file) {
    if (!result_matrix || !output_file) {
        fprintf(stderr, "Error: Cannot save NULL result matrix or empty filename\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n_rows = result_matrix->nrows;
    MKL_INT n_cols = result_matrix->ncols;

    if ((n_rows == 0 || n_cols == 0) && result_matrix->values != NULL) {
        fprintf(stderr, "Warning: Result matrix has zero dimension but non-NULL values\n");
    }
    if (n_rows > 0 && n_cols > 0 && result_matrix->values == NULL) {
        fprintf(stderr, "Error: Result matrix has non-zero dimensions but NULL values\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    
    if (n_rows == 0 || n_cols == 0) {
        printf("Warning: Result matrix is empty, saving empty file to %s\n", output_file);
        if (result_matrix->colnames != NULL) {
            fprintf(fp, " "); 
            for (MKL_INT j = 0; j < n_cols; j++) {
                fprintf(fp, "\t%s", result_matrix->colnames[j] ? result_matrix->colnames[j] : "UNKNOWN_COL");
            }
        }
        fprintf(fp, "\n");
        fclose(fp);
        return MORANS_I_SUCCESS;
    }

    printf("Saving results to %s...\n", output_file);
    
    // Write header
    if (n_cols > 0 && result_matrix->colnames != NULL) {
        fprintf(fp, " "); 
        for (MKL_INT j = 0; j < n_cols; j++) {
            fprintf(fp, "\t%s", result_matrix->colnames[j] ? result_matrix->colnames[j] : "UNKNOWN_COL");
        }
        fprintf(fp, "\n");
    } else if (n_cols > 0) { 
        fprintf(fp, " "); 
        for (MKL_INT j = 0; j < n_cols; j++) {
            fprintf(fp, "\tCol%lld", (long long)j + 1);
        }
        fprintf(fp, "\n");
    } else { 
        fprintf(fp, "\n"); 
    }

    // Write data rows
    for (MKL_INT i = 0; i < n_rows; i++) {
        fprintf(fp, "%s", (result_matrix->rownames && result_matrix->rownames[i]) ? 
                result_matrix->rownames[i] : "UNKNOWN_ROW");
        
        for (MKL_INT j = 0; j < n_cols; j++) {
            double value = result_matrix->values[i * n_cols + j];
            fprintf(fp, "\t"); 

            if (isnan(value)) {
                fprintf(fp, "NaN");
            } else if (isinf(value)) {
                fprintf(fp, "%sInf", (value > 0 ? "" : "-"));
            } else {
                if ((fabs(value) > 0 && fabs(value) < 1e-4) || fabs(value) > 1e6) {
                    fprintf(fp, "%.6e", value);
                } else {
                    fprintf(fp, "%.8f", value);
                }
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Results saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}

/* Save single gene Moran's I results with row normalization support */
int save_single_gene_results(const DenseMatrix* X_calc, const SparseMatrix* W, 
                            double S0_unused, const char* output_file, int row_normalized) {
    (void)S0_unused; // Mark as intentionally unused
    
    if (!X_calc || !W || !output_file) {
        fprintf(stderr, "Error: Invalid parameters provided to save_single_gene_results\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (X_calc->nrows > 0 && X_calc->ncols > 0 && !X_calc->values) {
        fprintf(stderr, "Error: X_calc has non-zero dims but NULL values\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (W->nnz > 0 && !W->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n_spots = X_calc->nrows; 
    MKL_INT n_genes = X_calc->ncols;
    
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    fprintf(fp, "Gene\tMoranI\n"); 

    if (n_genes == 0) {
        printf("Warning: No genes in X_calc, saving header-only file to %s\n", output_file);
        fclose(fp);
        return MORANS_I_SUCCESS;
    }
    if (n_spots == 0) {
        printf("Warning: No spots in X_calc. Moran's I is undefined. Saving NaNs.\n");
        for (MKL_INT g = 0; g < n_genes; g++) {
            fprintf(fp, "%s\tNaN\n", (X_calc->colnames && X_calc->colnames[g]) ? 
                    X_calc->colnames[g] : "UNKNOWN_GENE");
        }
        fclose(fp);
        return MORANS_I_SUCCESS; 
    }

    double* gene_data_col = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!gene_data_col) {
        perror("Failed to allocate gene_data_col");
        fclose(fp); 
        return MORANS_I_ERROR_MEMORY;
    }

    for (MKL_INT g = 0; g < n_genes; g++) {
        for (MKL_INT spot_idx = 0; spot_idx < n_spots; spot_idx++) {
            gene_data_col[spot_idx] = X_calc->values[spot_idx * n_genes + g];
        }
        double moran_val = calculate_single_gene_moran_i(gene_data_col, W, n_spots, row_normalized);

        fprintf(fp, "%s\t", (X_calc->colnames && X_calc->colnames[g]) ? 
                X_calc->colnames[g] : "UNKNOWN_GENE");
        
        if (isnan(moran_val)) {
            fprintf(fp, "NaN\n");
        } else if (isinf(moran_val)) {
            fprintf(fp, "%sInf\n", (moran_val > 0 ? "" : "-"));
        } else if (fabs(moran_val) > 0 && fabs(moran_val) < 1e-4) {
            fprintf(fp, "%.6e\n", moran_val);
        } else {
            fprintf(fp, "%.8f\n", moran_val);
        }
    }
    
    mkl_free(gene_data_col); 
    fclose(fp);
    printf("Single-gene Moran's I results saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}

/* Save first gene vs all results */
int save_first_gene_vs_all_results(const double* morans_values, const char** gene_names,
                                  MKL_INT n_genes, const char* output_file) {
    if (!output_file) {
        fprintf(stderr, "Error: Null output_file provided\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (n_genes > 0 && (!morans_values || !gene_names)) {
        fprintf(stderr, "Error: Non-zero n_genes but NULL morans_values or gene_names\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    
    const char* first_gene_name = (n_genes > 0 && gene_names && gene_names[0]) ? 
                                  gene_names[0] : "FirstGene";
    fprintf(fp, "Gene\tMoranI_vs_%s\n", first_gene_name);

    if (n_genes == 0) {
        printf("Warning: No genes to save, saving header-only file to %s\n", output_file);
        fclose(fp);
        return MORANS_I_SUCCESS;
    }
    
    printf("Saving first gene vs all results to %s...\n", output_file);
    for (MKL_INT g = 0; g < n_genes; g++) {
        fprintf(fp, "%s\t", (gene_names[g]) ? gene_names[g] : "UNKNOWN_GENE");
        double value = morans_values[g];
        
        if (isnan(value)) {
            fprintf(fp, "NaN\n");
        } else if (isinf(value)) {
            fprintf(fp, "%sInf\n", (value > 0 ? "" : "-"));
        } else if (fabs(value) > 0 && fabs(value) < 1e-4) {
            fprintf(fp, "%.6e\n", value);
        } else {
            fprintf(fp, "%.8f\n", value);
        }
    }
    
    fclose(fp);
    printf("First gene vs all results saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}

/* Save the lower triangular part of a symmetric matrix */
int save_lower_triangular_matrix_raw(const DenseMatrix* matrix, const char* output_file) {
    if (!matrix || !output_file) {
        fprintf(stderr, "Error: Cannot save NULL matrix or empty filename\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (matrix->nrows > 0 && matrix->ncols > 0 && !matrix->values) {
        fprintf(stderr, "Error: Matrix has non-zero dims but NULL values\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (matrix->nrows != matrix->ncols) {
        fprintf(stderr, "Error: Matrix must be square (dims %lldx%lld)\n",
                (long long)matrix->nrows, (long long)matrix->ncols);
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n = matrix->nrows; 

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    
    if (n == 0) {
        printf("Warning: Matrix is empty, saving empty file to %s\n", output_file);
        fclose(fp);
        return MORANS_I_SUCCESS;
    }

    printf("Saving lower triangular matrix to %s...\n", output_file);

    for (MKL_INT i = 0; i < n; i++) { 
        for (MKL_INT j = 0; j <= i; j++) { 
            double value = matrix->values[i * n + j]; 

            if (j > 0) { 
                fprintf(fp, "\t");
            }

            if (isnan(value)) {
                fprintf(fp, "NaN");
            } else if (isinf(value)) {
                fprintf(fp, "%sInf", (value > 0 ? "" : "-"));
            } else {
                if (fabs(value) != 0.0 && (fabs(value) < 1e-4 || fabs(value) > 1e6)) {
                    fprintf(fp, "%.6e", value);
                } else {
                    fprintf(fp, "%.8f", value);
                }
            }
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return MORANS_I_SUCCESS;
}

/* Save permutation test results */
int save_permutation_results(const PermutationResults* results, const char* output_file_prefix) {
    if (!results || !output_file_prefix) {
        fprintf(stderr, "Error: Cannot save NULL results or empty output prefix\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    char filename_buffer[BUFFER_SIZE];
    int status = MORANS_I_SUCCESS;

    printf("--- Saving Permutation Test Results (Lower Triangular Raw Format) ---\n");

    if (results->z_scores && results->z_scores->values) { 
        snprintf(filename_buffer, BUFFER_SIZE, "%s_zscores_lower_tri.tsv", output_file_prefix);
        printf("Saving Z-scores (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->z_scores, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving Z-scores\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    } else {
        printf("Z-scores not calculated, skipping save\n");
    }

    if (results->p_values && results->p_values->values) { 
        snprintf(filename_buffer, BUFFER_SIZE, "%s_pvalues_lower_tri.tsv", output_file_prefix);
        printf("Saving P-values (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->p_values, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving P-values\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    } else {
        printf("P-values not calculated, skipping save\n");
    }

    if (status == MORANS_I_SUCCESS) {
        printf("Permutation test results saved with prefix: %s\n", output_file_prefix);
    }
    return status;
}

/* ===============================
 * MEMORY MANAGEMENT FUNCTIONS
 * =============================== */

/* Free dense matrix */
void free_dense_matrix(DenseMatrix* matrix) {
    if (!matrix) return;
    
    if (matrix->values) mkl_free(matrix->values);
    
    if (matrix->rownames) {
        for (MKL_INT i = 0; i < matrix->nrows; i++) {
            if(matrix->rownames[i]) free(matrix->rownames[i]);
        }
        free(matrix->rownames);
    }
    
    if (matrix->colnames) {
        for (MKL_INT i = 0; i < matrix->ncols; i++) {
            if(matrix->colnames[i]) free(matrix->colnames[i]);
        }
        free(matrix->colnames);
    }
    
    free(matrix);
}

/* Free sparse matrix */
void free_sparse_matrix(SparseMatrix* matrix) {
    if (!matrix) return;
    
    if (matrix->row_ptr) mkl_free(matrix->row_ptr);
    if (matrix->col_ind) mkl_free(matrix->col_ind);
    if (matrix->values) mkl_free(matrix->values);
    
    if (matrix->rownames) { 
        for (MKL_INT i = 0; i < matrix->nrows; i++) {
            if(matrix->rownames[i]) free(matrix->rownames[i]);
        }
        free(matrix->rownames);
    }
    
    if (matrix->colnames) { 
        for (MKL_INT i = 0; i < matrix->ncols; i++) {
            if(matrix->colnames[i]) free(matrix->colnames[i]);
        }
        free(matrix->colnames);
    }
    
    free(matrix);
}

/* Free spot coordinates */
void free_spot_coordinates(SpotCoordinates* coords) {
    if (!coords) return;
    
    if(coords->spot_row) free(coords->spot_row);
    if(coords->spot_col) free(coords->spot_col);
    if(coords->valid_mask) free(coords->valid_mask);
    
    if (coords->spot_names) {
        for (MKL_INT i = 0; i < coords->total_spots; i++) {
            if(coords->spot_names[i]) free(coords->spot_names[i]);
        }
        free(coords->spot_names);
    }
    
    free(coords);
}

/* Free permutation results structure */
void free_permutation_results(PermutationResults* results) {
    if (!results) return;
    
    if (results->z_scores) free_dense_matrix(results->z_scores);
    if (results->p_values) free_dense_matrix(results->p_values);
    if (results->mean_perm) free_dense_matrix(results->mean_perm);
    if (results->var_perm) free_dense_matrix(results->var_perm);
    
    free(results);
}

/* ===============================
 * UTILITY AND ERROR HANDLING
 * =============================== */

/* Helper to print MKL sparse status */
void print_mkl_status(sparse_status_t status, const char* function_name) {
    if (status == SPARSE_STATUS_SUCCESS) return;
    
    fprintf(stderr, "MKL Sparse BLAS Error: Function '%s' failed with status: ", function_name);
    switch(status) {
        case SPARSE_STATUS_NOT_INITIALIZED: 
            fprintf(stderr, "SPARSE_STATUS_NOT_INITIALIZED\n"); 
            break;
        case SPARSE_STATUS_ALLOC_FAILED:    
            fprintf(stderr, "SPARSE_STATUS_ALLOC_FAILED\n"); 
            break;
        case SPARSE_STATUS_INVALID_VALUE:   
            fprintf(stderr, "SPARSE_STATUS_INVALID_VALUE\n"); 
            break;
        case SPARSE_STATUS_EXECUTION_FAILED:
            fprintf(stderr, "SPARSE_STATUS_EXECUTION_FAILED\n"); 
            break;
        case SPARSE_STATUS_INTERNAL_ERROR:  
            fprintf(stderr, "SPARSE_STATUS_INTERNAL_ERROR\n"); 
            break;
        case SPARSE_STATUS_NOT_SUPPORTED:   
            fprintf(stderr, "SPARSE_STATUS_NOT_SUPPORTED\n"); 
            break;
        default: 
            fprintf(stderr, "Unknown MKL Sparse Status Code (%d)\n", status); 
            break;
    }
}

/* Parse numeric parameters safely */
int load_positive_value(const char* value_str, const char* param, unsigned int min_val, unsigned int max_val) {
    if (!value_str || !param) { 
        fprintf(stderr, "Error: Null string or param name in load_positive_value\n"); 
        return -1; 
    }
    
    char* endptr; 
    errno = 0;
    long value = strtol(value_str, &endptr, 10);
    
    if (errno == ERANGE || *endptr != '\0' || value_str == endptr) { 
        fprintf(stderr, "Error: Parameter %s ('%s') is not a valid integer\n", param, value_str); 
        return -1;
    }
    if (value < (long)min_val || value > (long)max_val) { 
        fprintf(stderr, "Error: Parameter %s value %ld out of range [%u, %u]\n", param, value, min_val, max_val); 
        return -1;
    }
    
    return (int)value;
}

/* Parse double parameters safely */
double load_double_value(const char* value_str, const char* param) {
    if (!value_str || !param) { 
        fprintf(stderr, "Error: Null string or param name in load_double_value\n"); 
        return NAN;
    }
    
    char* endptr; 
    errno = 0; 
    double value = strtod(value_str, &endptr);
    
    if (errno == ERANGE || *endptr != '\0' || value_str == endptr) { 
        fprintf(stderr, "Error: Parameter %s ('%s') is not a valid double\n", param, value_str); 
        return NAN;
    }
    if (!isfinite(value)) { 
        fprintf(stderr, "Error: Parameter %s ('%s') resulted in non-finite value\n", param, value_str); 
        return NAN;
    }
    
    return value;
}

/* ===============================
 * CUSTOM WEIGHT MATRIX FUNCTIONS (COMPLETE IMPLEMENTATIONS)
 * =============================== */

/* Helper function to find spot index by name */
static MKL_INT find_spot_index(const char* spot_name, char** spot_names, MKL_INT n_spots) {
    if (!spot_name || !spot_names) return -1;
    
    for (MKL_INT i = 0; i < n_spots; i++) {
        if (spot_names[i] && strcmp(spot_name, spot_names[i]) == 0) {
            return i;
        }
    }
    return -1;
}

/* Read dense weight matrix from TSV file */
SparseMatrix* read_dense_weight_matrix(const char* filename, char** spot_names, MKL_INT n_spots) {
    if (!filename || !spot_names || n_spots <= 0) {
        fprintf(stderr, "Error: Invalid parameters in read_dense_weight_matrix\n");
        return NULL;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open dense weight matrix file '%s': %s\n", 
                filename, strerror(errno));
        return NULL;
    }
    
    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    
    printf("Reading dense weight matrix from '%s'...\n", filename);
    
    // Read header to get column spot names
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty header in dense weight matrix file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Trim line
    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
        line[--line_len] = '\0';
    }
    
    // Parse header - first field is row label, rest are column spot names
    char* header_copy = strdup(line);
    if (!header_copy) {
        perror("strdup header in read_dense_weight_matrix");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Count columns and map to spot indices
    MKL_INT* col_spot_indices = (MKL_INT*)malloc(n_spots * sizeof(MKL_INT));
    if (!col_spot_indices) {
        perror("malloc col_spot_indices");
        free(header_copy);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    char* token = strtok(header_copy, "\t");
    if (!token) {
        fprintf(stderr, "Error: No header fields found in dense weight matrix\n");
        free(col_spot_indices);
        free(header_copy);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Skip first column (row labels)
    token = strtok(NULL, "\t");
    MKL_INT n_cols_found = 0;
    
    while (token && n_cols_found < n_spots) {
        char* trimmed = trim_whitespace_inplace(token);
        MKL_INT spot_idx = find_spot_index(trimmed, spot_names, n_spots);
        if (spot_idx >= 0) {
            col_spot_indices[n_cols_found] = spot_idx;
            n_cols_found++;
        } else {
            fprintf(stderr, "Warning: Column spot '%s' not found in expression data\n", trimmed);
        }
        token = strtok(NULL, "\t");
    }
    
    free(header_copy);
    
    if (n_cols_found == 0) {
        fprintf(stderr, "Error: No matching column spots found in dense weight matrix\n");
        free(col_spot_indices);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Temporary storage for non-zero entries
    typedef struct {
        MKL_INT row, col;
        double value;
    } triplet_t;
    
    triplet_t* triplets = (triplet_t*)malloc(n_spots * n_spots * sizeof(triplet_t));
    if (!triplets) {
        perror("malloc triplets for dense matrix");
        free(col_spot_indices);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    MKL_INT nnz_count = 0;
    
    // Read data rows
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        // Trim line
        while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
            line[--line_len] = '\0';
        }
        if (line_len == 0) continue;
        
        char* row_copy = strdup(line);
        if (!row_copy) {
            perror("strdup row in read_dense_weight_matrix");
            break;
        }
        
        // Get row spot name
        token = strtok(row_copy, "\t");
        if (!token) {
            free(row_copy);
            continue;
        }
        
        char* row_spot_name = trim_whitespace_inplace(token);
        MKL_INT row_idx = find_spot_index(row_spot_name, spot_names, n_spots);
        
        if (row_idx >= 0) {
            // Read weight values
            for (MKL_INT col = 0; col < n_cols_found; col++) {
                token = strtok(NULL, "\t");
                if (!token) break;
                
                char* endptr;
                double weight = strtod(token, &endptr);
                if (endptr != token && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                    if (nnz_count < n_spots * n_spots) {
                        triplets[nnz_count].row = row_idx;
                        triplets[nnz_count].col = col_spot_indices[col];
                        triplets[nnz_count].value = weight;
                        nnz_count++;
                    }
                }
            }
        }
        
        free(row_copy);
    }
    
    fclose(fp);
    if (line) free(line);
    free(col_spot_indices);
    
    printf("Dense weight matrix read: %lld non-zero entries found\n", (long long)nnz_count);
    
    // Convert to sparse CSR format
    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!W) {
        perror("malloc SparseMatrix for dense weight matrix");
        free(triplets);
        return NULL;
    }
    
    W->nrows = n_spots;
    W->ncols = n_spots;
    W->nnz = nnz_count;
    W->rownames = NULL;
    W->colnames = NULL;
    
    // Allocate CSR arrays
    W->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc(nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc(nnz_count * sizeof(double), 64);
    } else {
        W->col_ind = NULL;
        W->values = NULL;
    }
    
    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) {
        perror("Failed to allocate CSR arrays for dense weight matrix");
        free_sparse_matrix(W);
        free(triplets);
        return NULL;
    }
    
    if (nnz_count > 0) {
        // Count entries per row
        for (MKL_INT k = 0; k < nnz_count; k++) {
            W->row_ptr[triplets[k].row + 1]++;
        }
        
        // Cumulative sum
        for (MKL_INT i = 0; i < n_spots; i++) {
            W->row_ptr[i + 1] += W->row_ptr[i];
        }
        
        // Fill CSR arrays
        MKL_INT* row_counters = (MKL_INT*)calloc(n_spots, sizeof(MKL_INT));
        if (!row_counters) {
            perror("calloc row_counters");
            free_sparse_matrix(W);
            free(triplets);
            return NULL;
        }
        
        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT row = triplets[k].row;
            MKL_INT pos = W->row_ptr[row] + row_counters[row];
            W->col_ind[pos] = triplets[k].col;
            W->values[pos] = triplets[k].value;
            row_counters[row]++;
        }
        
        free(row_counters);
    }
    
    free(triplets);
    
    printf("Dense weight matrix converted to sparse format: %lldx%lld with %lld NNZ\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz);
    
    return W;
}

/* Read sparse weight matrix in COO format */
SparseMatrix* read_sparse_weight_matrix_coo(const char* filename, char** spot_names, MKL_INT n_spots) {
    if (!filename || !spot_names || n_spots <= 0) {
        fprintf(stderr, "Error: Invalid parameters in read_sparse_weight_matrix_coo\n");
        return NULL;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open sparse COO weight matrix file '%s': %s\n", 
                filename, strerror(errno));
        return NULL;
    }
    
    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    
    printf("Reading sparse COO weight matrix from '%s'...\n", filename);
    
    // Skip header if present
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty sparse COO weight matrix file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Check if first line looks like header
    char* first_line_copy = strdup(line);
    if (!first_line_copy) {
        perror("strdup first line in read_sparse_weight_matrix_coo");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    char* token = strtok(first_line_copy, "\t");
    int looks_like_header = 0;
    if (token && (strstr(token, "row") || strstr(token, "col") || strstr(token, "coord"))) {
        looks_like_header = 1;
    }
    free(first_line_copy);
    
    if (!looks_like_header) {
        // Rewind to process first line as data
        rewind(fp);
    }
    
    // Temporary storage for triplets
    typedef struct {
        MKL_INT row, col;
        double value;
    } triplet_t;
    
    size_t triplets_capacity = 10000;
    triplet_t* triplets = (triplet_t*)malloc(triplets_capacity * sizeof(triplet_t));
    if (!triplets) {
        perror("malloc triplets for sparse COO");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    MKL_INT nnz_count = 0;
    regex_t coord_regex;
    int regex_compiled = 0;
    
    if (regcomp(&coord_regex, "^([0-9]+)x([0-9]+)$", REG_EXTENDED) == 0) {
        regex_compiled = 1;
    }
    
    // Read data lines
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        // Trim line
        while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
            line[--line_len] = '\0';
        }
        if (line_len == 0) continue;
        
        char* row_copy = strdup(line);
        if (!row_copy) {
            perror("strdup data line in read_sparse_weight_matrix_coo");
            break;
        }
        
        // Parse: row_coord, col_coord, weight
        char* row_coord_str = strtok(row_copy, "\t");
        char* col_coord_str = strtok(NULL, "\t");
        char* weight_str = strtok(NULL, "\t");
        
        if (!row_coord_str || !col_coord_str || !weight_str) {
            free(row_copy);
            continue;
        }
        
        // Parse coordinates to find spot indices
        MKL_INT row_idx = -1, col_idx = -1;
        
        if (regex_compiled) {
            // Try coordinate format first
            regmatch_t matches[3];
            if (regexec(&coord_regex, row_coord_str, 3, matches, 0) == 0) {
                // Extract row coordinate as spot name and find index
                row_idx = find_spot_index(row_coord_str, spot_names, n_spots);
            }
            if (regexec(&coord_regex, col_coord_str, 3, matches, 0) == 0) {
                col_idx = find_spot_index(col_coord_str, spot_names, n_spots);
            }
        }
        
        // If coordinate parsing failed, try direct spot name lookup
        if (row_idx < 0) {
            row_idx = find_spot_index(row_coord_str, spot_names, n_spots);
        }
        if (col_idx < 0) {
            col_idx = find_spot_index(col_coord_str, spot_names, n_spots);
        }
        
        if (row_idx >= 0 && col_idx >= 0) {
            char* endptr;
            double weight = strtod(weight_str, &endptr);
            if (endptr != weight_str && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                // Expand array if needed
                if (nnz_count >= (MKL_INT)triplets_capacity) {
                    triplets_capacity *= 2;
                    triplet_t* new_triplets = (triplet_t*)realloc(triplets, triplets_capacity * sizeof(triplet_t));
                    if (!new_triplets) {
                        perror("realloc triplets for sparse COO");
                        free(row_copy);
                        goto cleanup_coo;
                    }
                    triplets = new_triplets;
                }
                
                triplets[nnz_count].row = row_idx;
                triplets[nnz_count].col = col_idx;
                triplets[nnz_count].value = weight;
                nnz_count++;
            }
        }
        
        free(row_copy);
    }
    
cleanup_coo:
    if (regex_compiled) {
        regfree(&coord_regex);
    }
    fclose(fp);
    if (line) free(line);
    
    printf("Sparse COO weight matrix read: %lld non-zero entries found\n", (long long)nnz_count);
    
    // Convert to CSR format (same logic as dense version)
    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!W) {
        perror("malloc SparseMatrix for COO weight matrix");
        free(triplets);
        return NULL;
    }
    
    W->nrows = n_spots;
    W->ncols = n_spots;
    W->nnz = nnz_count;
    W->rownames = NULL;
    W->colnames = NULL;
    
    W->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc(nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc(nnz_count * sizeof(double), 64);
    } else {
        W->col_ind = NULL;
        W->values = NULL;
    }
    
    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) {
        perror("Failed to allocate CSR arrays for COO weight matrix");
        free_sparse_matrix(W);
        free(triplets);
        return NULL;
    }
    
    if (nnz_count > 0) {
        // Convert to CSR
        for (MKL_INT k = 0; k < nnz_count; k++) {
            W->row_ptr[triplets[k].row + 1]++;
        }
        
        for (MKL_INT i = 0; i < n_spots; i++) {
            W->row_ptr[i + 1] += W->row_ptr[i];
        }
        
        MKL_INT* row_counters = (MKL_INT*)calloc(n_spots, sizeof(MKL_INT));
        if (!row_counters) {
            perror("calloc row_counters for COO");
            free_sparse_matrix(W);
            free(triplets);
            return NULL;
        }
        
        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT row = triplets[k].row;
            MKL_INT pos = W->row_ptr[row] + row_counters[row];
            W->col_ind[pos] = triplets[k].col;
            W->values[pos] = triplets[k].value;
            row_counters[row]++;
        }
        
        free(row_counters);
    }
    
    free(triplets);
    
    printf("Sparse COO weight matrix converted to CSR format: %lldx%lld with %lld NNZ\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz);
    
    return W;
}

/* Read sparse weight matrix in TSV format */
SparseMatrix* read_sparse_weight_matrix_tsv(const char* filename, char** spot_names, MKL_INT n_spots) {
    if (!filename || !spot_names || n_spots <= 0) {
        fprintf(stderr, "Error: Invalid parameters in read_sparse_weight_matrix_tsv\n");
        return NULL;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open sparse TSV weight matrix file '%s': %s\n", 
                filename, strerror(errno));
        return NULL;
    }
    
    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    
    printf("Reading sparse TSV weight matrix from '%s'...\n", filename);
    
    // Skip header if present
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty sparse TSV weight matrix file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Check if first line looks like header
    char* first_line_copy = strdup(line);
    if (!first_line_copy) {
        perror("strdup first line in read_sparse_weight_matrix_tsv");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    char* token = strtok(first_line_copy, "\t");
    int looks_like_header = 0;
    if (token && (strstr(token, "spot") || strstr(token, "weight") || strstr(token, "from") || strstr(token, "to"))) {
        looks_like_header = 1;
    }
    free(first_line_copy);
    
    if (!looks_like_header) {
        rewind(fp);
    }
    
    // Temporary storage for triplets
    typedef struct {
        MKL_INT row, col;
        double value;
    } triplet_t;
    
    size_t triplets_capacity = 10000;
    triplet_t* triplets = (triplet_t*)malloc(triplets_capacity * sizeof(triplet_t));
    if (!triplets) {
        perror("malloc triplets for sparse TSV");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    MKL_INT nnz_count = 0;
    
    // Read data lines
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        // Trim line
        while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
            line[--line_len] = '\0';
        }
        if (line_len == 0) continue;
        
        char* row_copy = strdup(line);
        if (!row_copy) {
            perror("strdup data line in read_sparse_weight_matrix_tsv");
            break;
        }
        
        // Parse: spot1, spot2, weight
        char* spot1_str = strtok(row_copy, "\t");
        char* spot2_str = strtok(NULL, "\t");
        char* weight_str = strtok(NULL, "\t");
        
        if (!spot1_str || !spot2_str || !weight_str) {
            free(row_copy);
            continue;
        }
        
        // Find spot indices
        char* spot1_trimmed = trim_whitespace_inplace(spot1_str);
        char* spot2_trimmed = trim_whitespace_inplace(spot2_str);
        
        MKL_INT row_idx = find_spot_index(spot1_trimmed, spot_names, n_spots);
        MKL_INT col_idx = find_spot_index(spot2_trimmed, spot_names, n_spots);
        
        if (row_idx >= 0 && col_idx >= 0) {
            char* endptr;
            double weight = strtod(weight_str, &endptr);
            if (endptr != weight_str && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                // Expand array if needed
                if (nnz_count >= (MKL_INT)triplets_capacity) {
                    triplets_capacity *= 2;
                    triplet_t* new_triplets = (triplet_t*)realloc(triplets, triplets_capacity * sizeof(triplet_t));
                    if (!new_triplets) {
                        perror("realloc triplets for sparse TSV");
                        free(row_copy);
                        goto cleanup_tsv;
                    }
                    triplets = new_triplets;
                }
                
                triplets[nnz_count].row = row_idx;
                triplets[nnz_count].col = col_idx;
                triplets[nnz_count].value = weight;
                nnz_count++;
            }
        }
        
        free(row_copy);
    }
    
cleanup_tsv:
    fclose(fp);
    if (line) free(line);
    
    printf("Sparse TSV weight matrix read: %lld non-zero entries found\n", (long long)nnz_count);
    
    // Convert to CSR format (same logic as previous functions)
    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!W) {
        perror("malloc SparseMatrix for TSV weight matrix");
        free(triplets);
        return NULL;
    }
    
    W->nrows = n_spots;
    W->ncols = n_spots;
    W->nnz = nnz_count;
    W->rownames = NULL;
    W->colnames = NULL;
    
    W->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc(nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc(nnz_count * sizeof(double), 64);
    } else {
        W->col_ind = NULL;
        W->values = NULL;
    }
    
    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) {
        perror("Failed to allocate CSR arrays for TSV weight matrix");
        free_sparse_matrix(W);
        free(triplets);
        return NULL;
    }
    
    if (nnz_count > 0) {
        // Convert to CSR
        for (MKL_INT k = 0; k < nnz_count; k++) {
            W->row_ptr[triplets[k].row + 1]++;
        }
        
        for (MKL_INT i = 0; i < n_spots; i++) {
            W->row_ptr[i + 1] += W->row_ptr[i];
        }
        
        MKL_INT* row_counters = (MKL_INT*)calloc(n_spots, sizeof(MKL_INT));
        if (!row_counters) {
            perror("calloc row_counters for TSV");
            free_sparse_matrix(W);
            free(triplets);
            return NULL;
        }
        
        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT row = triplets[k].row;
            MKL_INT pos = W->row_ptr[row] + row_counters[row];
            W->col_ind[pos] = triplets[k].col;
            W->values[pos] = triplets[k].value;
            row_counters[row]++;
        }
        
        free(row_counters);
    }
    
    free(triplets);
    
    printf("Sparse TSV weight matrix converted to CSR format: %lldx%lld with %lld NNZ\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz);
    
    return W;
}

/* Main custom weight matrix reading function */
SparseMatrix* read_custom_weight_matrix(const char* filename, int format, 
                                       char** spot_names, MKL_INT n_spots) {
    if (!filename || !spot_names || n_spots <= 0) {
        fprintf(stderr, "Error: Invalid parameters in read_custom_weight_matrix\n");
        return NULL;
    }
    
    // Auto-detect format if requested
    int actual_format = format;
    if (format == WEIGHT_FORMAT_AUTO) {
        actual_format = detect_weight_matrix_format(filename);
        if (actual_format == WEIGHT_FORMAT_AUTO) {
            fprintf(stderr, "Error: Could not auto-detect weight matrix format for '%s'\n", filename);
            return NULL;
        }
    }
    
    SparseMatrix* W = NULL;
    
    switch (actual_format) {
        case WEIGHT_FORMAT_DENSE:
            printf("Reading custom weight matrix in DENSE format...\n");
            W = read_dense_weight_matrix(filename, spot_names, n_spots);
            break;
            
        case WEIGHT_FORMAT_SPARSE_COO:
            printf("Reading custom weight matrix in SPARSE_COO format...\n");
            W = read_sparse_weight_matrix_coo(filename, spot_names, n_spots);
            break;
            
        case WEIGHT_FORMAT_SPARSE_TSV:
            printf("Reading custom weight matrix in SPARSE_TSV format...\n");
            W = read_sparse_weight_matrix_tsv(filename, spot_names, n_spots);
            break;
            
        default:
            fprintf(stderr, "Error: Unsupported weight matrix format: %d\n", actual_format);
            return NULL;
    }
    
    if (W) {
        // Validate the weight matrix
        int validation_result = validate_weight_matrix(W, spot_names, n_spots);
        if (validation_result != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error: Weight matrix validation failed\n");
            free_sparse_matrix(W);
            return NULL;
        }
        
        printf("Custom weight matrix successfully loaded and validated.\n");
    }
    
    return W;
}

/* ===============================
 * BATCH CALCULATION FUNCTION
 * =============================== */

/* Calculate Moran's I using batch interface with raw arrays */
double* calculate_morans_i_batch(const double* X_data, long long n_genes_ll, long long n_spots_ll,
                                const double* W_values, const long long* W_row_ptr_ll, 
                                const long long* W_col_ind_ll, long long W_nnz_ll, int paired_genes) {
    
    if (!X_data || !W_values || !W_row_ptr_ll || !W_col_ind_ll) {
        fprintf(stderr, "Error: NULL parameters in calculate_morans_i_batch\n");
        return NULL;
    }
    
    // Convert to MKL_INT for internal use
    MKL_INT n_genes = (MKL_INT)n_genes_ll;
    MKL_INT n_spots = (MKL_INT)n_spots_ll;
    MKL_INT W_nnz = (MKL_INT)W_nnz_ll;
    
    if (n_genes <= 0 || n_spots <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in calculate_morans_i_batch: %lld genes, %lld spots\n",
                n_genes_ll, n_spots_ll);
        return NULL;
    }
    
    printf("Batch Moran's I calculation: %lld genes x %lld spots, %s\n",
           n_genes_ll, n_spots_ll, paired_genes ? "pairwise" : "single-gene");
    
    // Convert W arrays to MKL_INT format for internal use
    MKL_INT* W_row_ptr_mkl = (MKL_INT*)mkl_malloc((n_spots + 1) * sizeof(MKL_INT), 64);
    MKL_INT* W_col_ind_mkl = NULL;
    if (W_nnz > 0) {
        W_col_ind_mkl = (MKL_INT*)mkl_malloc(W_nnz * sizeof(MKL_INT), 64);
    }
    
    if (!W_row_ptr_mkl || (W_nnz > 0 && !W_col_ind_mkl)) {
        perror("Failed to allocate converted W arrays for batch calculation");
        if (W_row_ptr_mkl) mkl_free(W_row_ptr_mkl);
        if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
        return NULL;
    }
    
    // Convert indices
    for (MKL_INT i = 0; i <= n_spots; i++) {
        W_row_ptr_mkl[i] = (MKL_INT)W_row_ptr_ll[i];
    }
    for (MKL_INT i = 0; i < W_nnz; i++) {
        W_col_ind_mkl[i] = (MKL_INT)W_col_ind_ll[i];
    }
    
    // Calculate S0 for scaling
    double S0 = 0.0;
    #pragma omp parallel for reduction(+:S0)
    for (MKL_INT i = 0; i < W_nnz; i++) {
        S0 += W_values[i];
    }
    
    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero in batch calculation\n");
        mkl_free(W_row_ptr_mkl);
        if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
        return NULL;
    }
    
    double inv_S0 = 1.0 / S0;
    
    // Create MKL sparse matrix handle
    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl, SPARSE_INDEX_BASE_ZERO,
                                                     n_spots, n_spots, W_row_ptr_mkl,
                                                     W_row_ptr_mkl + 1, W_col_ind_mkl, 
                                                     (double*)W_values);
    
    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (batch W)");
        mkl_free(W_row_ptr_mkl);
        if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
        return NULL;
    }
    
    if (W_nnz > 0) {
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_optimize (batch W)");
        }
    }
    
    double* results = NULL;
    
    if (paired_genes) {
        // Pairwise calculation: result is n_genes x n_genes matrix
        size_t result_size = (size_t)n_genes * n_genes;
        results = (double*)mkl_malloc(result_size * sizeof(double), 64);
        if (!results) {
            perror("Failed to allocate results for batch pairwise calculation");
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl);
            if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        
        // Calculate W * X
        double* temp_WX = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
        if (!temp_WX) {
            perror("Failed to allocate temp_WX for batch calculation");
            mkl_free(results);
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl);
            if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descr,
                                SPARSE_LAYOUT_ROW_MAJOR, X_data, n_genes, n_genes,
                                0.0, temp_WX, n_genes);
        
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_d_mm (batch W * X)");
            mkl_free(temp_WX);
            mkl_free(results);
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl);
            if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        
        // Calculate X^T * (W * X) / S0
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n_genes, n_genes, n_spots, inv_S0,
                    X_data, n_genes, temp_WX, n_genes,
                    0.0, results, n_genes);
        
        mkl_free(temp_WX);
        
    } else {
        // Single-gene calculation: result is n_genes vector
        results = (double*)mkl_malloc((size_t)n_genes * sizeof(double), 64);
        if (!results) {
            perror("Failed to allocate results for batch single-gene calculation");
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl);
            if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        
        double* gene_buffer = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
        double* Wz_buffer = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
        if (!gene_buffer || !Wz_buffer) {
            perror("Failed to allocate buffers for batch single-gene calculation");
            mkl_free(results);
            if (gene_buffer) mkl_free(gene_buffer);
            if (Wz_buffer) mkl_free(Wz_buffer);
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl);
            if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        // Calculate Moran's I for each gene
        for (MKL_INT g = 0; g < n_genes; g++) {
            // Extract gene g data
            for (MKL_INT s = 0; s < n_spots; s++) {
                gene_buffer[s] = X_data[s * n_genes + g];
            }
            
            // Calculate W * gene_data
            status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descr,
                                    gene_buffer, 0.0, Wz_buffer);
            
            if (status != SPARSE_STATUS_SUCCESS) {
                print_mkl_status(status, "mkl_sparse_d_mv (batch single-gene)");
                results[g] = NAN;
                continue;
            }
            
            // Calculate gene_data^T * (W * gene_data) / S0
            double dot_product = cblas_ddot(n_spots, gene_buffer, 1, Wz_buffer, 1);
            results[g] = dot_product * inv_S0;
        }
        
        mkl_free(gene_buffer);
        mkl_free(Wz_buffer);
    }
    
    mkl_sparse_destroy(W_mkl);
    mkl_free(W_row_ptr_mkl);
    if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
    
    printf("Batch Moran's I calculation complete.\n");
    return results;
}

/* ===============================
 * COORDINATE MAPPING FUNCTION
 * =============================== */

/* Map expression matrix spots to coordinate data */
int map_expression_to_coordinates(const DenseMatrix* expr_matrix, const SpotCoordinates* coords,
                                 MKL_INT** mapping_out, MKL_INT* num_mapped_spots_out) {
    if (!expr_matrix || !coords || !mapping_out || !num_mapped_spots_out) {
        fprintf(stderr, "Error: NULL parameters in map_expression_to_coordinates\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    if (!expr_matrix->colnames) {
        fprintf(stderr, "Error: Expression matrix has no column names for mapping\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    if (!coords->spot_names || !coords->valid_mask) {
        fprintf(stderr, "Error: Coordinate data is missing spot names or validity mask\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    MKL_INT n_expr_spots = expr_matrix->ncols;
    *mapping_out = NULL;
    *num_mapped_spots_out = 0;
    
    if (n_expr_spots == 0) {
        printf("Warning: No spots in expression matrix to map\n");
        return MORANS_I_SUCCESS;
    }
    
    if (coords->total_spots == 0) {
        printf("Warning: No coordinate data available for mapping\n");
        return MORANS_I_SUCCESS;
    }
    
    printf("Mapping %lld expression spots to %lld coordinate entries (%lld valid)...\n",
           (long long)n_expr_spots, (long long)coords->total_spots, (long long)coords->valid_spots);
    
    // Allocate mapping array
    MKL_INT* mapping = (MKL_INT*)malloc((size_t)n_expr_spots * sizeof(MKL_INT));
    if (!mapping) {
        perror("Failed to allocate mapping array");
        return MORANS_I_ERROR_MEMORY;
    }
    
    // Initialize mapping array to -1 (not found)
    for (MKL_INT i = 0; i < n_expr_spots; i++) {
        mapping[i] = -1;
    }
    
    MKL_INT mapped_count = 0;
    MKL_INT valid_mapped_count = 0;
    
    // Create a hash table for faster coordinate lookup (simple linear search for now)
    // For each expression spot, find corresponding coordinate
    for (MKL_INT expr_idx = 0; expr_idx < n_expr_spots; expr_idx++) {
        const char* expr_spot_name = expr_matrix->colnames[expr_idx];
        
        if (!expr_spot_name || strlen(expr_spot_name) == 0) {
            continue; // Skip empty spot names
        }
        
        // Search for matching spot name in coordinates
        MKL_INT coord_idx = -1;
        for (MKL_INT c = 0; c < coords->total_spots; c++) {
            if (coords->spot_names[c] && strcmp(expr_spot_name, coords->spot_names[c]) == 0) {
                coord_idx = c;
                break;
            }
        }
        
        if (coord_idx >= 0) {
            mapping[expr_idx] = coord_idx;
            mapped_count++;
            
            // Check if this coordinate is valid
            if (coords->valid_mask[coord_idx]) {
                valid_mapped_count++;
            }
        }
    }
    
    printf("Mapping complete: %lld/%lld expression spots mapped to coordinates\n",
           (long long)mapped_count, (long long)n_expr_spots);
    printf("  %lld mapped spots have valid coordinates\n", (long long)valid_mapped_count);
    
    if (mapped_count == 0) {
        fprintf(stderr, "Warning: No expression spots could be mapped to coordinates\n");
        free(mapping);
        return MORANS_I_SUCCESS; // Not an error, just no matches
    }
    
    if (valid_mapped_count == 0) {
        fprintf(stderr, "Warning: No mapped spots have valid coordinates\n");
        free(mapping);
        return MORANS_I_SUCCESS; // Not an error, just no valid coordinates
    }
    
    // Provide some statistics about unmapped spots
    if (mapped_count < n_expr_spots) {
        MKL_INT unmapped_count = n_expr_spots - mapped_count;
        printf("Warning: %lld expression spots could not be mapped to coordinates\n", 
               (long long)unmapped_count);
        
        // Show a few examples of unmapped spots for debugging
        MKL_INT examples_shown = 0;
        const MKL_INT max_examples = 5;
        
        for (MKL_INT i = 0; i < n_expr_spots && examples_shown < max_examples; i++) {
            if (mapping[i] == -1 && expr_matrix->colnames[i]) {
                if (examples_shown == 0) {
                    printf("  Examples of unmapped spots: ");
                }
                printf("%s%s", expr_matrix->colnames[i], 
                       (examples_shown < max_examples - 1 && examples_shown < unmapped_count - 1) ? ", " : "");
                examples_shown++;
            }
        }
        if (examples_shown > 0) {
            printf("%s\n", (unmapped_count > max_examples) ? "..." : "");
        }
    }
    
    *mapping_out = mapping;
    *num_mapped_spots_out = mapped_count;
    
    return MORANS_I_SUCCESS;
}

/* Read coordinates from TSV file for single-cell data */
SpotCoordinates* read_coordinates_file(const char* filename, const char* id_column_name,
                                     const char* x_column_name, const char* y_column_name,
                                     double coord_scale) {
    if (!filename || !id_column_name || !x_column_name || !y_column_name) {
        fprintf(stderr, "Error: Null parameter(s) provided to read_coordinates_file\n");
        return NULL;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) { 
        fprintf(stderr, "Error: Failed to open coordinates file '%s': %s\n", filename, strerror(errno)); 
        return NULL;
    }

    char *line = NULL; 
    size_t line_buf_size = 0; 
    ssize_t line_len;
    int id_col_idx = -1, x_col_idx = -1, y_col_idx = -1;
    int header_field_count = 0;

    // Parse Header
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) { 
        fprintf(stderr, "Error: Empty header in coordinates file '%s'\n", filename); 
        fclose(fp); 
        if(line) free(line); 
        return NULL;
    }
    
    // Trim EOL
    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
        line[--line_len] = '\0';
    }

    char* header_copy = strdup(line);
    if(!header_copy) {
        perror("strdup header for coord file"); 
        fclose(fp); 
        free(line); 
        return NULL;
    }
    
    char* token_h = strtok(header_copy, "\t");
    int current_col_idx = 0;
    while (token_h) {
        char* trimmed_token = trim_whitespace_inplace(token_h);
        
        if (strcmp(trimmed_token, id_column_name) == 0) id_col_idx = current_col_idx;
        if (strcmp(trimmed_token, x_column_name) == 0) x_col_idx = current_col_idx;
        if (strcmp(trimmed_token, y_column_name) == 0) y_col_idx = current_col_idx;
        
        token_h = strtok(NULL, "\t"); 
        current_col_idx++;
        header_field_count++;
    }
    free(header_copy);
    
    if (id_col_idx == -1 || x_col_idx == -1 || y_col_idx == -1) {
        fprintf(stderr, "Error: Required columns not found in coordinates file '%s'\n", filename);
        fprintf(stderr, "  ID ('%s'): %s, X ('%s'): %s, Y ('%s'): %s\n", 
                id_column_name, (id_col_idx!=-1?"Found":"Missing"), 
                x_column_name, (x_col_idx!=-1?"Found":"Missing"), 
                y_column_name, (y_col_idx!=-1?"Found":"Missing"));
        fclose(fp); 
        if(line) free(line); 
        return NULL;
    }

    // Count data lines
    long current_pos = ftell(fp);
    MKL_INT num_data_lines = 0;
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) { 
        char* p = line; 
        while(isspace((unsigned char)*p)) p++; 
        if(*p != '\0') num_data_lines++;
    }
    fseek(fp, current_pos, SEEK_SET);

    if (num_data_lines == 0) { 
        fprintf(stderr, "Error: No data lines found in coordinates file '%s'\n", filename); 
        fclose(fp); 
        if(line) free(line); 
        return NULL;
    }

    SpotCoordinates* coords = (SpotCoordinates*)malloc(sizeof(SpotCoordinates));
    if (!coords) { 
        perror("malloc SpotCoordinates for file read"); 
        fclose(fp); 
        if(line) free(line); 
        return NULL; 
    }
    
    coords->total_spots = num_data_lines;
    coords->spot_row = (MKL_INT*)malloc((size_t)num_data_lines * sizeof(MKL_INT));
    coords->spot_col = (MKL_INT*)malloc((size_t)num_data_lines * sizeof(MKL_INT));
    coords->valid_mask = (int*)calloc(num_data_lines, sizeof(int));
    coords->spot_names = (char**)malloc((size_t)num_data_lines * sizeof(char*));
    coords->valid_spots = 0;
    
    if (!coords->spot_row || !coords->spot_col || !coords->valid_mask || !coords->spot_names) { 
        perror("Failed to allocate arrays for coordinates read from file"); 
        free_spot_coordinates(coords); 
        fclose(fp); 
        if(line) free(line); 
        return NULL;
    }

    // Initialize spot_names to NULL for safe cleanup
    for (MKL_INT i = 0; i < num_data_lines; i++) {
        coords->spot_names[i] = NULL;
    }

    // Read data lines
    MKL_INT spot_fill_idx = 0; 
    // Removed unused variable data_file_lineno
    
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && spot_fill_idx < num_data_lines) {
        char* p_chk = line; 
        while(isspace((unsigned char)*p_chk)) p_chk++; 
        if(*p_chk == '\0') continue;

        // Trim EOL
        while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
            line[--line_len] = '\0';
        }
        if (line_len == 0) continue;

        char* data_line_copy = strdup(line);
        if(!data_line_copy) { 
            perror("strdup data line for coord file"); 
            break;
        }
        
        char* field_tokens[header_field_count > 0 ? header_field_count : 1];
        int actual_tokens = 0;
        char* current_ptr = data_line_copy;
        
        for(int k = 0; k < header_field_count; ++k) {
            field_tokens[k] = strsep(&current_ptr, "\t");
            if(field_tokens[k] == NULL) break;
            actual_tokens++;
        }

        char* id_str = NULL; 
        double x_val = NAN, y_val = NAN;
        
        if (id_col_idx < actual_tokens && field_tokens[id_col_idx] != NULL) {
            id_str = field_tokens[id_col_idx];
        }
        if (x_col_idx < actual_tokens && field_tokens[x_col_idx] != NULL) { 
            char* end_x; 
            x_val = strtod(field_tokens[x_col_idx], &end_x); 
            if (end_x == field_tokens[x_col_idx]) x_val = NAN;
        }
        if (y_col_idx < actual_tokens && field_tokens[y_col_idx] != NULL) { 
            char* end_y; 
            y_val = strtod(field_tokens[y_col_idx], &end_y); 
            if (end_y == field_tokens[y_col_idx]) y_val = NAN;
        }

        if (id_str && strlen(id_str) > 0 && isfinite(x_val) && isfinite(y_val)) {
            coords->spot_names[spot_fill_idx] = strdup(id_str);
            if (!coords->spot_names[spot_fill_idx]) { 
                perror("strdup spot_id in read_coord_file"); 
                free(data_line_copy); 
                break;
            }
            coords->spot_row[spot_fill_idx] = (MKL_INT)round(y_val * coord_scale);
            coords->spot_col[spot_fill_idx] = (MKL_INT)round(x_val * coord_scale);
            coords->valid_mask[spot_fill_idx] = 1; 
            coords->valid_spots++;
        } else {
            coords->spot_names[spot_fill_idx] = strdup("INVALID_COORD_ENTRY");
            if(!coords->spot_names[spot_fill_idx]) {
                perror("strdup INVALID_COORD_ENTRY"); 
                free(data_line_copy); 
                break;
            }
            coords->spot_row[spot_fill_idx] = -1; 
            coords->spot_col[spot_fill_idx] = -1;
            coords->valid_mask[spot_fill_idx] = 0;
        }
        
        free(data_line_copy);
        spot_fill_idx++;
    }
    
    if (spot_fill_idx != coords->total_spots) {
        fprintf(stderr, "Warning: Estimated %lld data lines but processed %lld\n",
                (long long)coords->total_spots, (long long)spot_fill_idx);
        coords->total_spots = spot_fill_idx;
    }

    if(line) free(line); 
    fclose(fp);
    
    printf("Read %lld coordinate entries, %lld are valid after processing\n", 
           (long long)coords->total_spots, (long long)coords->valid_spots);
    
    if (coords->valid_spots == 0 && coords->total_spots > 0) {
        fprintf(stderr, "Critical Warning: No valid coordinates parsed. Check file format and column names.\n");
    }
    
    return coords;
}