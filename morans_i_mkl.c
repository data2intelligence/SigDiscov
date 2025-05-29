/* morans_i_mkl.c - Optimized MKL-based Moran's I implementation
 *
 * Version: 1.3.0 (Added Residual Moran's I functionality)
 *
 * This implements efficient calculation of Moran's I spatial autocorrelation
 * statistics for spatial transcriptomics data using Intel MKL, including
 * residual Moran's I for cell type correction.
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
#include "morans_i_mkl.h" // This now includes SpotNameHashTable definitions

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

/* Cell type file parsing helpers */
static int parse_celltype_header(const char* header_line, char delimiter,
                                const char* expected_id_col,
                                const char* expected_type_col,
                                const char* expected_x_col,
                                const char* expected_y_col,
                                int* id_col_idx,
                                int* type_col_idx,
                                int* x_col_idx,
                                int* y_col_idx);

static char** collect_unique_celltypes(FILE* fp, char delimiter, int type_col_idx, 
                                      MKL_INT* n_celltypes_out, MKL_INT* n_cells_out);

/* Hash Table Helper Function DEFINITIONS (these stay in .c and are static) */
static unsigned long spot_name_hash(const char* str) {
    unsigned long hash = 5381;
    int c;
    while ((c = (unsigned char)*str++)) // cast to unsigned char
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    return hash;
}

static int is_prime(size_t n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (size_t i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return 0;
    return 1;
}

/* Helper to find next prime (simple version for hash table sizing) */
static size_t get_next_prime(size_t n) {
    if (n <= 1) return 2;
    size_t prime = n;
    int found = 0;
    while (!found) {
        if (is_prime(prime))
            found = 1;
        else
            prime++;
    }
    return prime;
}

static SpotNameHashTable* spot_name_ht_create(size_t num_spots_hint) {
    SpotNameHashTable* ht = (SpotNameHashTable*)malloc(sizeof(SpotNameHashTable));
    if (!ht) {
        perror("spot_name_ht_create: malloc ht failed");
        return NULL;
    }

    // Aim for load factor around 1.0, choose prime number of buckets.
    // Ensure num_buckets is at least a small prime if hint is 0 or very small.
    size_t buckets_n = (num_spots_hint > 16) ? num_spots_hint : 17; // Use 17 as a small prime default
    ht->num_buckets = get_next_prime(buckets_n);

    ht->buckets = (SpotNameHashNode**)calloc(ht->num_buckets, sizeof(SpotNameHashNode*));
    if (!ht->buckets) {
        perror("spot_name_ht_create: calloc buckets failed");
        free(ht);
        return NULL;
    }
    ht->count = 0;
    DEBUG_PRINT("Created spot name hash table with %zu buckets for hint %zu.", ht->num_buckets, num_spots_hint);
    return ht;
}

static int spot_name_ht_insert(SpotNameHashTable* ht, const char* name, MKL_INT index) {
    if (!ht || !name) return -1; 
    unsigned long hash_val = spot_name_hash(name);
    size_t bucket_idx = hash_val % ht->num_buckets;

    // Check for duplicates (optional, but good for integrity if names should be unique)
    #ifdef DEBUG_BUILD
    SpotNameHashNode* existing = ht->buckets[bucket_idx];
    while(existing) {
        if (strcmp(existing->name, name) == 0) {
            DEBUG_PRINT("Hash table insert warning: Duplicate name '%s' (old_idx: %lld, new_idx: %lld). Overwriting is not implemented; new entry will shadow old in find.", name, (long long)existing->index, (long long)index);
            // Depending on policy, you might return an error or update the existing node.
            // For this application, distinct spot names from expression data are expected.
        }
        existing = existing->next;
    }
    #endif

    SpotNameHashNode* newNode = (SpotNameHashNode*)malloc(sizeof(SpotNameHashNode));
    if (!newNode) {
        perror("spot_name_ht_insert: malloc newNode failed");
        return -1; 
    }
    newNode->name = strdup(name);
    if (!newNode->name) {
        perror("spot_name_ht_insert: strdup name failed");
        free(newNode);
        return -1; 
    }
    newNode->index = index;
    newNode->next = ht->buckets[bucket_idx]; // Insert at the head of the list
    ht->buckets[bucket_idx] = newNode;
    ht->count++;
    return 0; // Success
}

static MKL_INT spot_name_ht_find(const SpotNameHashTable* ht, const char* name) {
    if (!ht || !name) return -1; 
    unsigned long hash_val = spot_name_hash(name);
    size_t bucket_idx = hash_val % ht->num_buckets;
    SpotNameHashNode* current = ht->buckets[bucket_idx];
    while (current) {
        if (strcmp(current->name, name) == 0) {
            return current->index;
        }
        current = current->next;
    }
    return -1; // Not found
}

static void spot_name_ht_free(SpotNameHashTable* ht) {
    if (!ht) return;
    for (size_t i = 0; i < ht->num_buckets; i++) {
        SpotNameHashNode* current = ht->buckets[i];
        while (current) {
            SpotNameHashNode* next_node = current->next;
            free(current->name);
            free(current);
            current = next_node;
        }
    }
    free(ht->buckets);
    free(ht);
}

/* Library version information */
const char* morans_i_mkl_version(void) {
    return "1.3.0"; 
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
    
    // Residual analysis defaults
    config.residual_config.analysis_mode = DEFAULT_ANALYSIS_MODE;
    config.residual_config.celltype_file = NULL;
    config.residual_config.celltype_format = DEFAULT_CELLTYPE_FORMAT;
    config.residual_config.celltype_id_col = strdup("cell_ID");
    config.residual_config.celltype_type_col = strdup("cellType");
    config.residual_config.celltype_x_col = strdup("sdimx");
    config.residual_config.celltype_y_col = strdup("sdimy");
    config.residual_config.spot_id_col = strdup("spot_id");
    config.residual_config.include_intercept = DEFAULT_INCLUDE_INTERCEPT;
    config.residual_config.regularization_lambda = DEFAULT_REGULARIZATION_LAMBDA;
    config.residual_config.normalize_residuals = DEFAULT_NORMALIZE_RESIDUALS;
    
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
    printf("\nResidual Moran's I Options:\n");
    printf("  --analysis-mode <standard|residual>\tAnalysis mode. Default: standard.\n");
    printf("  --celltype-file <file>\t\tCell type composition/annotation file.\n");
    printf("  --celltype-format <deconv|sc>\t\tFormat: deconvolution or single_cell. Default: single_cell.\n");
    printf("  --celltype-id-col <name>\t\tCell ID column name. Default: cell_ID.\n");
    printf("  --celltype-type-col <name>\t\tCell type column name. Default: cellType.\n");
    printf("  --include-intercept <0|1>\t\tInclude intercept in regression. Default: 1.\n");
    printf("  --regularization <float>\t\tRidge regularization parameter. Default: 0.0.\n");
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
    printf("  Residual Analysis outputs: <prefix>_residual_morans_i_raw.tsv, <prefix>_regression_coefficients.tsv\n");
    printf("\nExample:\n");
    printf("  %s -i expression.tsv -o morans_i_run1 -r 3 -p 0 -b 1 -g 1 -t 8 --run-perm 1 --num-perm 500\n\n", program_name);
    printf("Version: %s\n", morans_i_mkl_version());
}

/* ===============================
 * CELL TYPE DATA PROCESSING
 * =============================== */

/* Detect file delimiter (CSV vs TSV) */
int detect_file_delimiter(const char* filename) {
    if (!filename) {
        fprintf(stderr, "Error: NULL filename in detect_file_delimiter\n");
        return '\t'; // Default to tab
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Warning: Cannot open file '%s' for delimiter detection: %s\n", 
                filename, strerror(errno));
        return '\t'; // Default to tab
    }
    
    char* line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len = getline(&line, &line_buf_size, fp);
    fclose(fp);
    
    if (line_len <= 0) {
        if (line) free(line);
        return '\t'; // Default to tab
    }
    
    int comma_count = 0, tab_count = 0;
    for (ssize_t i = 0; i < line_len; i++) {
        if (line[i] == ',') comma_count++;
        else if (line[i] == '\t') tab_count++;
    }
    
    free(line);
    DEBUG_PRINT("Delimiter detection: commas=%d, tabs=%d", comma_count, tab_count);
    return (comma_count > tab_count) ? ',' : '\t';
}

/* Parse cell type header to find column indices */
static int parse_celltype_header(const char* header_line, char delimiter,
                                const char* expected_id_col,
                                const char* expected_type_col,
                                const char* expected_x_col,
                                const char* expected_y_col,
                                int* id_col_idx,
                                int* type_col_idx,
                                int* x_col_idx,
                                int* y_col_idx) {
    if (!header_line || !expected_id_col || !expected_type_col) {
        return MORANS_I_ERROR_PARAMETER;
    }
    
    *id_col_idx = *type_col_idx = *x_col_idx = *y_col_idx = -1;
    
    char* header_copy = strdup(header_line);
    if (!header_copy) {
        perror("strdup header in parse_celltype_header");
        return MORANS_I_ERROR_MEMORY;
    }
    
    char delim_str[2] = {delimiter, '\0'};
    char* token = strtok(header_copy, delim_str);
    int col_idx = 0;
    
    while (token) {
        char* trimmed = trim_whitespace_inplace(token);
        
        // Skip first column if it's an index column (empty or numeric-looking)
        if (col_idx == 0 && (strlen(trimmed) == 0 || isdigit((unsigned char)trimmed[0]))) {
            token = strtok(NULL, delim_str);
            col_idx++;
            continue;
        }
        
        if (strcmp(trimmed, expected_id_col) == 0) {
            *id_col_idx = col_idx;
        } else if (strcmp(trimmed, expected_type_col) == 0) {
            *type_col_idx = col_idx;
        } else if (expected_x_col && strcmp(trimmed, expected_x_col) == 0) {
            *x_col_idx = col_idx;
        } else if (expected_y_col && strcmp(trimmed, expected_y_col) == 0) {
            *y_col_idx = col_idx;
        }
        
        token = strtok(NULL, delim_str);
        col_idx++;
    }
    
    free(header_copy);
    
    DEBUG_PRINT("Column indices: ID=%d, Type=%d, X=%d, Y=%d", 
                *id_col_idx, *type_col_idx, *x_col_idx, *y_col_idx);
    
    return MORANS_I_SUCCESS;
}

/* Collect unique cell types from file */
static char** collect_unique_celltypes(FILE* fp, char delimiter, int type_col_idx, 
                                      MKL_INT* n_celltypes_out, MKL_INT* n_cells_out) {
    if (!fp || type_col_idx < 0 || !n_celltypes_out || !n_cells_out) {
        return NULL;
    }
    
    long current_pos = ftell(fp);
    char* line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    
    // First pass: collect all cell types
    char** temp_celltypes = NULL;
    MKL_INT temp_capacity = 100;
    MKL_INT temp_count = 0;
    MKL_INT cell_count = 0;
    
    temp_celltypes = (char**)malloc(temp_capacity * sizeof(char*));
    if (!temp_celltypes) {
        perror("malloc temp_celltypes");
        return NULL;
    }
    
    char delim_str[2] = {delimiter, '\0'};
    
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p == '\0') continue; // Skip empty lines
        
        char* line_copy = strdup(line);
        if (!line_copy) continue;
        
        char* token = strtok(line_copy, delim_str);
        int col_idx = 0;
        
        while (token && col_idx <= type_col_idx) {
            if (col_idx == type_col_idx) {
                char* celltype = trim_whitespace_inplace(token);
                if (strlen(celltype) > 0) {
                    // Check if this cell type is already in our list
                    int found = 0;
                    for (MKL_INT i = 0; i < temp_count; i++) {
                        if (strcmp(temp_celltypes[i], celltype) == 0) {
                            found = 1;
                            break;
                        }
                    }
                    
                    if (!found) {
                        if (temp_count >= temp_capacity) {
                            temp_capacity *= 2;
                            char** new_temp = (char**)realloc(temp_celltypes, temp_capacity * sizeof(char*));
                            if (!new_temp) {
                                for (MKL_INT i = 0; i < temp_count; i++) {
                                    free(temp_celltypes[i]);
                                }
                                free(temp_celltypes);
                                free(line_copy);
                                if (line) free(line);
                                return NULL;
                            }
                            temp_celltypes = new_temp;
                        }
                        temp_celltypes[temp_count] = strdup(celltype);
                        if (!temp_celltypes[temp_count]) {
                            perror("strdup celltype");
                            for (MKL_INT i = 0; i < temp_count; i++) {
                                free(temp_celltypes[i]);
                            }
                            free(temp_celltypes);
                            free(line_copy);
                            if (line) free(line);
                            return NULL;
                        }
                        temp_count++;
                    }
                    cell_count++;
                }
                break;
            }
            token = strtok(NULL, delim_str);
            col_idx++;
        }
        free(line_copy);
    }
    
    if (line) free(line);
    fseek(fp, current_pos, SEEK_SET);
    
    *n_celltypes_out = temp_count;
    *n_cells_out = cell_count;
    
    printf("Found %lld unique cell types from %lld cells\n", 
           (long long)temp_count, (long long)cell_count);
    
    return temp_celltypes;
}

/* Read cell type data in single-cell format */
CellTypeMatrix* read_celltype_singlecell_file(const char* filename, 
                                             const char* cell_id_col,
                                             const char* celltype_col,
                                             const char* x_col,
                                             const char* y_col) {
    if (!filename || !cell_id_col || !celltype_col) {
        fprintf(stderr, "Error: NULL parameters in read_celltype_singlecell_file\n");
        return NULL;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open cell type file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }
    
    char delimiter = detect_file_delimiter(filename);
    printf("Reading single-cell annotations from '%s' with delimiter '%c'\n", filename, delimiter);
    
    char* line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    
    // Read and parse header
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty header in cell type file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    int id_col_idx, type_col_idx, x_col_idx, y_col_idx;
    if (parse_celltype_header(line, delimiter, cell_id_col, celltype_col, x_col, y_col,
                             &id_col_idx, &type_col_idx, &x_col_idx, &y_col_idx) != MORANS_I_SUCCESS) {
        fprintf(stderr, "Error: Failed to parse cell type file header\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    if (id_col_idx < 0 || type_col_idx < 0) {
        fprintf(stderr, "Error: Required columns not found: %s=%s, %s=%s\n",
                cell_id_col, (id_col_idx >= 0 ? "Found" : "Missing"),
                celltype_col, (type_col_idx >= 0 ? "Found" : "Missing"));
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Collect unique cell types
    MKL_INT n_celltypes, n_cells;
    char** unique_celltypes = collect_unique_celltypes(fp, delimiter, type_col_idx, &n_celltypes, &n_cells);
    if (!unique_celltypes || n_celltypes == 0) {
        fprintf(stderr, "Error: No cell types found in file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Create cell type matrix
    CellTypeMatrix* celltype_matrix = (CellTypeMatrix*)malloc(sizeof(CellTypeMatrix));
    if (!celltype_matrix) {
        perror("malloc CellTypeMatrix");
        for (MKL_INT i = 0; i < n_celltypes; i++) {
            free(unique_celltypes[i]);
        }
        free(unique_celltypes);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    celltype_matrix->nrows = n_cells;
    celltype_matrix->ncols = n_celltypes;
    celltype_matrix->is_binary = 1;
    celltype_matrix->format_type = CELLTYPE_FORMAT_SINGLE_CELL;
    
    size_t values_size;
    if (safe_multiply_size_t(n_cells, n_celltypes, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Cell type matrix dimensions too large\n");
        free(celltype_matrix);
        for (MKL_INT i = 0; i < n_celltypes; i++) {
            free(unique_celltypes[i]);
        }
        free(unique_celltypes);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    celltype_matrix->values = (double*)mkl_calloc(values_size / sizeof(double), sizeof(double), 64);
    celltype_matrix->rownames = (char**)calloc(n_cells, sizeof(char*));
    celltype_matrix->colnames = (char**)calloc(n_celltypes, sizeof(char*));
    
    if (!celltype_matrix->values || !celltype_matrix->rownames || !celltype_matrix->colnames) {
        perror("Failed to allocate cell type matrix components");
        free_celltype_matrix(celltype_matrix);
        for (MKL_INT i = 0; i < n_celltypes; i++) {
            free(unique_celltypes[i]);
        }
        free(unique_celltypes);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Copy cell type names
    for (MKL_INT i = 0; i < n_celltypes; i++) {
        celltype_matrix->colnames[i] = strdup(unique_celltypes[i]);
        if (!celltype_matrix->colnames[i]) {
            perror("strdup cell type name");
            free_celltype_matrix(celltype_matrix);
            for (MKL_INT j = 0; j < n_celltypes; j++) {
                free(unique_celltypes[j]);
            }
            free(unique_celltypes);
            fclose(fp);
            if (line) free(line);
            return NULL;
        }
    }
    
    // Read data and populate matrix
    MKL_INT cell_idx = 0;
    char delim_str[2] = {delimiter, '\0'};
    
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && cell_idx < n_cells) {
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p == '\0') continue;
        
        char* line_copy = strdup(line);
        if (!line_copy) continue;
        
        char* token = strtok(line_copy, delim_str);
        int col_idx = 0;
        char* cell_id = NULL;
        char* cell_type = NULL;
        
        while (token) {
            if (col_idx == id_col_idx) {
                cell_id = trim_whitespace_inplace(token);
            } else if (col_idx == type_col_idx) {
                cell_type = trim_whitespace_inplace(token);
            }
            token = strtok(NULL, delim_str);
            col_idx++;
        }
        
        if (cell_id && cell_type && strlen(cell_id) > 0 && strlen(cell_type) > 0) {
            // Store cell ID
            celltype_matrix->rownames[cell_idx] = strdup(cell_id);
            if (!celltype_matrix->rownames[cell_idx]) {
                perror("strdup cell_id");
                free(line_copy);
                break;
            }
            
            // Find cell type index and set binary indicator
            for (MKL_INT ct_idx = 0; ct_idx < n_celltypes; ct_idx++) {
                if (strcmp(cell_type, unique_celltypes[ct_idx]) == 0) {
                    celltype_matrix->values[cell_idx * n_celltypes + ct_idx] = 1.0;
                    break;
                }
            }
            cell_idx++;
        }
        free(line_copy);
    }
    
    // Cleanup
    for (MKL_INT i = 0; i < n_celltypes; i++) {
        free(unique_celltypes[i]);
    }
    free(unique_celltypes);
    fclose(fp);
    if (line) free(line);
    
    if (cell_idx != n_cells) {
        fprintf(stderr, "Warning: Expected %lld cells but processed %lld\n", 
                (long long)n_cells, (long long)cell_idx);
        celltype_matrix->nrows = cell_idx;
    }
    
    printf("Successfully loaded single-cell annotations: %lld cells x %lld cell types\n",
           (long long)celltype_matrix->nrows, (long long)celltype_matrix->ncols);
    
    return celltype_matrix;
}


/* Read cell type data in deconvolution format */
CellTypeMatrix* read_celltype_deconvolution_file(const char* filename, const char* spot_id_col) {
    if (!filename || !spot_id_col) {
        fprintf(stderr, "Error: NULL parameters in read_celltype_deconvolution_file\n");
        return NULL;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open deconvolution file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }
    
    char delimiter = detect_file_delimiter(filename);
    printf("Reading deconvolution data from '%s' with delimiter '%c'\n", filename, delimiter);
    
    char* line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    
    // Read and parse header
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty header in deconvolution file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Trim line
    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
        line[--line_len] = '\0';
    }
    
    char* header_copy = strdup(line);
    if (!header_copy) {
        perror("strdup header");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Count fields and find spot ID column
    char delim_str[2] = {delimiter, '\0'};
    char* token = strtok(header_copy, delim_str);
    int field_count = 0;
    int spot_id_col_idx = -1;
    
    while (token) {
        char* trimmed = trim_whitespace_inplace(token);
        if (strcmp(trimmed, spot_id_col) == 0) {
            spot_id_col_idx = field_count;
        }
        field_count++;
        token = strtok(NULL, delim_str);
    }
    free(header_copy);
    
    if (spot_id_col_idx < 0) {
        fprintf(stderr, "Error: Spot ID column '%s' not found in header\n", spot_id_col);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    MKL_INT n_celltypes = field_count - 1; // Subtract spot ID column
    if (n_celltypes <= 0) {
        fprintf(stderr, "Error: No cell type columns found\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Count data rows
    long current_pos = ftell(fp);
    MKL_INT n_spots = 0;
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p != '\0') n_spots++;
    }
    fseek(fp, current_pos, SEEK_SET);
    
    if (n_spots == 0) {
        fprintf(stderr, "Error: No data rows found\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Create cell type matrix
    CellTypeMatrix* celltype_matrix = (CellTypeMatrix*)malloc(sizeof(CellTypeMatrix));
    if (!celltype_matrix) {
        perror("malloc CellTypeMatrix");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    celltype_matrix->nrows = n_spots;
    celltype_matrix->ncols = n_celltypes;
    celltype_matrix->is_binary = 0;
    celltype_matrix->format_type = CELLTYPE_FORMAT_DECONVOLUTION;
    
    size_t values_size;
    if (safe_multiply_size_t(n_spots, n_celltypes, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Deconvolution matrix dimensions too large\n");
        free(celltype_matrix);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    celltype_matrix->values = (double*)mkl_malloc(values_size, 64);
    celltype_matrix->rownames = (char**)calloc(n_spots, sizeof(char*));
    celltype_matrix->colnames = (char**)calloc(n_celltypes, sizeof(char*));
    
    if (!celltype_matrix->values || !celltype_matrix->rownames || !celltype_matrix->colnames) {
        perror("Failed to allocate deconvolution matrix components");
        free_celltype_matrix(celltype_matrix);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Read header again to get column names
    fseek(fp, 0, SEEK_SET);
    getline(&line, &line_buf_size, fp); // Re-read header
    header_copy = strdup(line);
    token = strtok(header_copy, delim_str);
    int col_idx = 0;
    int celltype_col_idx = 0;
    
    while (token) {
        char* trimmed = trim_whitespace_inplace(token);
        if (col_idx != spot_id_col_idx) {
            celltype_matrix->colnames[celltype_col_idx] = strdup(trimmed);
            if (!celltype_matrix->colnames[celltype_col_idx]) {
                perror("strdup cell type column name");
                free(header_copy);
                free_celltype_matrix(celltype_matrix);
                fclose(fp);
                if (line) free(line);
                return NULL;
            }
            celltype_col_idx++;
        }
        col_idx++;
        token = strtok(NULL, delim_str);
    }
    free(header_copy);
    
    // Read data rows
    MKL_INT spot_idx = 0;
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && spot_idx < n_spots) {
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p == '\0') continue;
        
        char* line_copy = strdup(line);
        if (!line_copy) continue;
        
        token = strtok(line_copy, delim_str);
        col_idx = 0;
        celltype_col_idx = 0;
        char* spot_id = NULL;
        
        while (token) {
            char* trimmed = trim_whitespace_inplace(token);
            if (col_idx == spot_id_col_idx) {
                spot_id = trimmed;
            } else {
                char* endptr;
                double value = strtod(trimmed, &endptr);
                if (endptr != trimmed && isfinite(value)) {
                    celltype_matrix->values[spot_idx * n_celltypes + celltype_col_idx] = value;
                } else {
                    celltype_matrix->values[spot_idx * n_celltypes + celltype_col_idx] = 0.0;
                }
                celltype_col_idx++;
            }
            col_idx++;
            token = strtok(NULL, delim_str);
        }
        
        if (spot_id && strlen(spot_id) > 0) {
            celltype_matrix->rownames[spot_idx] = strdup(spot_id);
            if (!celltype_matrix->rownames[spot_idx]) {
                perror("strdup spot_id");
                free(line_copy);
                break;
            }
            spot_idx++;
        }
        free(line_copy);
    }
    
    fclose(fp);
    if (line) free(line);
    
    if (spot_idx != n_spots) {
        fprintf(stderr, "Warning: Expected %lld spots but processed %lld\n", 
                (long long)n_spots, (long long)spot_idx);
        celltype_matrix->nrows = spot_idx;
    }
    
    printf("Successfully loaded deconvolution data: %lld spots x %lld cell types\n",
           (long long)celltype_matrix->nrows, (long long)celltype_matrix->ncols);
    
    return celltype_matrix;
}

/* Validate cell type matrix against expression matrix */
int validate_celltype_matrix(const CellTypeMatrix* Z, const DenseMatrix* X) {
    if (!Z || !X) {
        fprintf(stderr, "Error: NULL parameters in validate_celltype_matrix\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    if (Z->nrows == 0 || Z->ncols == 0) {
        fprintf(stderr, "Error: Cell type matrix has zero dimensions\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    printf("Validating cell type matrix (%lld x %lld) against expression matrix (%lld x %lld)\n",
           (long long)Z->nrows, (long long)Z->ncols, (long long)X->nrows, (long long)X->ncols);
    
    return MORANS_I_SUCCESS;
}

/* Map cell type matrix to expression matrix spots */
int map_celltype_to_expression(const CellTypeMatrix* celltype_matrix, const DenseMatrix* expr_matrix,
                               CellTypeMatrix** mapped_celltype_out) {
    if (!celltype_matrix || !expr_matrix || !mapped_celltype_out) {
        fprintf(stderr, "Error: NULL parameters in map_celltype_to_expression\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    *mapped_celltype_out = NULL;
    
    if (celltype_matrix->format_type == CELLTYPE_FORMAT_SINGLE_CELL) {
        // For single-cell data, need to map cell IDs to expression spots
        printf("Mapping single-cell annotations to expression matrix spots...\n");
        
        // Create hash table for fast cell ID lookup
        SpotNameHashTable* cell_map = spot_name_ht_create(celltype_matrix->nrows);
        if (!cell_map) {
            fprintf(stderr, "Error: Failed to create cell ID hash table\n");
            return MORANS_I_ERROR_MEMORY;
        }
        
        for (MKL_INT i = 0; i < celltype_matrix->nrows; i++) {
            if (celltype_matrix->rownames[i]) {
                if (spot_name_ht_insert(cell_map, celltype_matrix->rownames[i], i) != 0) {
                    fprintf(stderr, "Error: Failed to insert cell ID into hash table\n");
                    spot_name_ht_free(cell_map);
                    return MORANS_I_ERROR_MEMORY;
                }
            }
        }
        
        // Create new mapped cell type matrix
        CellTypeMatrix* mapped = (CellTypeMatrix*)malloc(sizeof(CellTypeMatrix));
        if (!mapped) {
            perror("malloc mapped CellTypeMatrix");
            spot_name_ht_free(cell_map);
            return MORANS_I_ERROR_MEMORY;
        }
        
        mapped->nrows = expr_matrix->nrows;  // Use expression matrix spot count
        mapped->ncols = celltype_matrix->ncols;
        mapped->is_binary = celltype_matrix->is_binary;
        mapped->format_type = celltype_matrix->format_type;
        
        size_t mapped_values_size;
        if (safe_multiply_size_t(mapped->nrows, mapped->ncols, &mapped_values_size) != 0 ||
            safe_multiply_size_t(mapped_values_size, sizeof(double), &mapped_values_size) != 0) {
            fprintf(stderr, "Error: Mapped matrix dimensions too large\n");
            free(mapped);
            spot_name_ht_free(cell_map);
            return MORANS_I_ERROR_MEMORY;
        }
        
        mapped->values = (double*)mkl_calloc(mapped_values_size / sizeof(double), sizeof(double), 64);
        mapped->rownames = (char**)calloc(mapped->nrows, sizeof(char*));
        mapped->colnames = (char**)calloc(mapped->ncols, sizeof(char*));
        
        if (!mapped->values || !mapped->rownames || !mapped->colnames) {
            perror("Failed to allocate mapped matrix components");
            free_celltype_matrix(mapped);
            spot_name_ht_free(cell_map);
            return MORANS_I_ERROR_MEMORY;
        }
        
        // Copy cell type names
        for (MKL_INT j = 0; j < mapped->ncols; j++) {
            mapped->colnames[j] = strdup(celltype_matrix->colnames[j]);
            if (!mapped->colnames[j]) {
                perror("strdup cell type name");
                free_celltype_matrix(mapped);
                spot_name_ht_free(cell_map);
                return MORANS_I_ERROR_MEMORY;
            }
        }
        
        // Map cell type data to expression spots
        MKL_INT mapped_count = 0;
        for (MKL_INT i = 0; i < expr_matrix->nrows; i++) {
            const char* spot_name = expr_matrix->rownames[i];
            if (spot_name) {
                mapped->rownames[i] = strdup(spot_name);
                if (!mapped->rownames[i]) {
                    perror("strdup spot name");
                    free_celltype_matrix(mapped);
                    spot_name_ht_free(cell_map);
                    return MORANS_I_ERROR_MEMORY;
                }
                
                MKL_INT cell_idx = spot_name_ht_find(cell_map, spot_name);
                if (cell_idx >= 0) {
                    // Copy cell type data
                    for (MKL_INT j = 0; j < mapped->ncols; j++) {
                        mapped->values[i * mapped->ncols + j] = 
                            celltype_matrix->values[cell_idx * celltype_matrix->ncols + j];
                    }
                    mapped_count++;
                }
                // If not found, values remain 0 (from calloc)
            }
        }
        
        spot_name_ht_free(cell_map);
        
        printf("Mapped %lld/%lld expression spots to cell type annotations\n",
               (long long)mapped_count, (long long)expr_matrix->nrows);
        
        if (mapped_count == 0) {
            fprintf(stderr, "Warning: No expression spots could be mapped to cell type data\n");
        }
        
        *mapped_celltype_out = mapped;
        
    } else {
        // For deconvolution data, assume spot names match directly
        printf("Using deconvolution data directly (assuming spot names match)\n");
        
        // For now, just return a reference to the original matrix
        // In a full implementation, you might want to reorder spots to match expression matrix
        *mapped_celltype_out = (CellTypeMatrix*)celltype_matrix; // Cast away const - be careful!
    }
    
    return MORANS_I_SUCCESS;
}

/* ===============================
 * MATRIX OPERATIONS FOR RESIDUAL ANALYSIS
 * =============================== */

/* Create centering matrix H_n = I - (1/n) * 1 * 1^T */
DenseMatrix* create_centering_matrix(MKL_INT n) {
    if (n <= 0) {
        fprintf(stderr, "Error: Invalid dimension for centering matrix: %lld\n", (long long)n);
        return NULL;
    }
    
    DenseMatrix* H = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!H) {
        perror("malloc centering matrix");
        return NULL;
    }
    
    H->nrows = n;
    H->ncols = n;
    H->rownames = NULL;
    H->colnames = NULL;
    
    size_t values_size;
    if (safe_multiply_size_t(n, n, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Centering matrix dimensions too large\n");
        free(H);
        return NULL;
    }
    
    H->values = (double*)mkl_malloc(values_size, 64);
    if (!H->values) {
        perror("mkl_malloc centering matrix values");
        free(H);
        return NULL;
    }
    
    double inv_n = 1.0 / (double)n;
    
    // H = I - (1/n) * J where J is matrix of all ones
    #pragma omp parallel for
    for (MKL_INT i = 0; i < n; i++) {
        for (MKL_INT j = 0; j < n; j++) {
            if (i == j) {
                H->values[i * n + j] = 1.0 - inv_n;
            } else {
                H->values[i * n + j] = -inv_n;
            }
        }
    }
    
    DEBUG_PRINT("Created centering matrix: %lld x %lld", (long long)n, (long long)n);
    return H;
}

/* Compute regression coefficients B̂ = (Z^T Z + λI)^(-1) Z^T X^T */
DenseMatrix* compute_regression_coefficients(const CellTypeMatrix* Z, const DenseMatrix* X, double lambda) {
    if (!Z || !X || !Z->values || !X->values) {
        fprintf(stderr, "Error: NULL parameters in compute_regression_coefficients\n");
        return NULL;
    }
    
    if (Z->nrows != X->nrows) {
        fprintf(stderr, "Error: Dimension mismatch in regression: Z has %lld rows, X has %lld rows\n",
                (long long)Z->nrows, (long long)X->nrows);
        return NULL;
    }
    
    MKL_INT n_spots = Z->nrows;
    MKL_INT n_celltypes = Z->ncols;
    MKL_INT n_genes = X->ncols;
    
    printf("Computing regression coefficients: %lld cell types x %lld genes (lambda=%.6f)\n",
           (long long)n_celltypes, (long long)n_genes, lambda);
    
    // Compute Z^T Z
    size_t ztZ_size;
    if (safe_multiply_size_t(n_celltypes, n_celltypes, &ztZ_size) != 0 ||
        safe_multiply_size_t(ztZ_size, sizeof(double), &ztZ_size) != 0) {
        fprintf(stderr, "Error: Z^T Z matrix too large\n");
        return NULL;
    }
    
    double* ZtZ = (double*)mkl_malloc(ztZ_size, 64);
    if (!ZtZ) {
        perror("mkl_malloc ZtZ");
        return NULL;
    }
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n_celltypes, n_celltypes, n_spots,
                1.0, Z->values, n_celltypes,
                Z->values, n_celltypes,
                0.0, ZtZ, n_celltypes);
    
    // Add regularization: Z^T Z + λI
    if (lambda > 0.0) {
        for (MKL_INT i = 0; i < n_celltypes; i++) {
            ZtZ[i * n_celltypes + i] += lambda;
        }
    }
    
    // Compute (Z^T Z + λI)^(-1) using Cholesky decomposition
    MKL_INT* ipiv = (MKL_INT*)mkl_malloc(n_celltypes * sizeof(MKL_INT), 64);
    if (!ipiv) {
        perror("mkl_malloc ipiv");
        mkl_free(ZtZ);
        return NULL;
    }
    
    MKL_INT info;
    // LU decomposition
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n_celltypes, n_celltypes, ZtZ, n_celltypes, ipiv);
    if (info != 0) {
        fprintf(stderr, "Error: LU decomposition failed with info=%lld\n", (long long)info);
        mkl_free(ipiv);
        mkl_free(ZtZ);
        return NULL;
    }
    
    // Matrix inversion
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n_celltypes, ZtZ, n_celltypes, ipiv);
    mkl_free(ipiv);
    if (info != 0) {
        fprintf(stderr, "Error: Matrix inversion failed with info=%lld\n", (long long)info);
        mkl_free(ZtZ);
        return NULL;
    }
    
    // Now ZtZ contains (Z^T Z + λI)^(-1)
    
    // Compute Z^T X
    size_t ZtX_size;
    if (safe_multiply_size_t(n_celltypes, n_genes, &ZtX_size) != 0 ||
        safe_multiply_size_t(ZtX_size, sizeof(double), &ZtX_size) != 0) {
        fprintf(stderr, "Error: Z^T X matrix too large\n");
        mkl_free(ZtZ);
        return NULL;
    }
    
    double* ZtX = (double*)mkl_malloc(ZtX_size, 64);
    if (!ZtX) {
        perror("mkl_malloc ZtX");
        mkl_free(ZtZ);
        return NULL;
    }
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n_celltypes, n_genes, n_spots,
                1.0, Z->values, n_celltypes,
                X->values, n_genes,
                0.0, ZtX, n_genes);
    
    // Create result matrix for B̂
    DenseMatrix* B = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!B) {
        perror("malloc regression coefficients matrix");
        mkl_free(ZtZ);
        mkl_free(ZtX);
        return NULL;
    }
    
    B->nrows = n_celltypes;
    B->ncols = n_genes;
    B->rownames = (char**)calloc(n_celltypes, sizeof(char*));
    B->colnames = (char**)calloc(n_genes, sizeof(char*));
    B->values = (double*)mkl_malloc(ZtX_size, 64);
    
    if (!B->values || !B->rownames || !B->colnames) {
        perror("Failed to allocate regression coefficients components");
        mkl_free(ZtZ);
        mkl_free(ZtX);
        free_dense_matrix(B);
        return NULL;
    }
    
    // Copy names
    for (MKL_INT i = 0; i < n_celltypes; i++) {
        if (Z->colnames && Z->colnames[i]) {
            B->rownames[i] = strdup(Z->colnames[i]);
        } else {
            char temp[32];
            snprintf(temp, sizeof(temp), "CellType_%lld", (long long)i);
            B->rownames[i] = strdup(temp);
        }
        if (!B->rownames[i]) {
            perror("strdup cell type name for coefficients");
            mkl_free(ZtZ);
            mkl_free(ZtX);
            free_dense_matrix(B);
            return NULL;
        }
    }
    
    for (MKL_INT j = 0; j < n_genes; j++) {
        if (X->colnames && X->colnames[j]) {
            B->colnames[j] = strdup(X->colnames[j]);
        } else {
            char temp[32];
            snprintf(temp, sizeof(temp), "Gene_%lld", (long long)j);
            B->colnames[j] = strdup(temp);
        }
        if (!B->colnames[j]) {
            perror("strdup gene name for coefficients");
            mkl_free(ZtZ);
            mkl_free(ZtX);
            free_dense_matrix(B);
            return NULL;
        }
    }
    
    // Compute final result: B̂ = (Z^T Z + λI)^(-1) Z^T X
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_celltypes, n_genes, n_celltypes,
                1.0, ZtZ, n_celltypes,
                ZtX, n_genes,
                0.0, B->values, n_genes);
    
    mkl_free(ZtZ);
    mkl_free(ZtX);
    
    printf("Regression coefficients computed successfully\n");
    return B;
}

/* Compute residual projection matrix M_res = I - Z(Z^T Z + λI)^(-1) Z^T */
DenseMatrix* compute_residual_projection_matrix(const CellTypeMatrix* Z, double lambda) {
    if (!Z || !Z->values) {
        fprintf(stderr, "Error: NULL parameters in compute_residual_projection_matrix\n");
        return NULL;
    }
    
    MKL_INT n_spots = Z->nrows;
    MKL_INT n_celltypes = Z->ncols;
    
    printf("Computing residual projection matrix: %lld x %lld (lambda=%.6f)\n",
           (long long)n_spots, (long long)n_spots, lambda);
    
    // Compute Z^T Z + λI and its inverse (reuse code from regression coefficients)
    size_t ztZ_size;
    if (safe_multiply_size_t(n_celltypes, n_celltypes, &ztZ_size) != 0 ||
        safe_multiply_size_t(ztZ_size, sizeof(double), &ztZ_size) != 0) {
        fprintf(stderr, "Error: Z^T Z matrix too large for projection\n");
        return NULL;
    }
    
    double* ZtZ_inv = (double*)mkl_malloc(ztZ_size, 64);
    if (!ZtZ_inv) {
        perror("mkl_malloc ZtZ_inv");
        return NULL;
    }
    
    // Compute Z^T Z
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n_celltypes, n_celltypes, n_spots,
                1.0, Z->values, n_celltypes,
                Z->values, n_celltypes,
                0.0, ZtZ_inv, n_celltypes);
    
    // Add regularization
    if (lambda > 0.0) {
        for (MKL_INT i = 0; i < n_celltypes; i++) {
            ZtZ_inv[i * n_celltypes + i] += lambda;
        }
    }
    
    // Invert matrix
    MKL_INT* ipiv = (MKL_INT*)mkl_malloc(n_celltypes * sizeof(MKL_INT), 64);
    if (!ipiv) {
        perror("mkl_malloc ipiv");
        mkl_free(ZtZ_inv);
        return NULL;
    }
    
    MKL_INT info;
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n_celltypes, n_celltypes, ZtZ_inv, n_celltypes, ipiv);
    if (info != 0) {
        fprintf(stderr, "Error: LU decomposition failed in projection matrix\n");
        mkl_free(ipiv);
        mkl_free(ZtZ_inv);
        return NULL;
    }
    
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n_celltypes, ZtZ_inv, n_celltypes, ipiv);
    mkl_free(ipiv);
    if (info != 0) {
        fprintf(stderr, "Error: Matrix inversion failed in projection matrix\n");
        mkl_free(ZtZ_inv);
        return NULL;
    }
    
    // Compute Z * (Z^T Z + λI)^(-1) * Z^T
    size_t temp_size;
    if (safe_multiply_size_t(n_spots, n_celltypes, &temp_size) != 0 ||
        safe_multiply_size_t(temp_size, sizeof(double), &temp_size) != 0) {
        fprintf(stderr, "Error: Temporary matrix too large for projection\n");
        mkl_free(ZtZ_inv);
        return NULL;
    }
    
    double* temp = (double*)mkl_malloc(temp_size, 64);
    if (!temp) {
        perror("mkl_malloc temp for projection");
        mkl_free(ZtZ_inv);
        return NULL;
    }
    
    // temp = Z * (Z^T Z + λI)^(-1)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_spots, n_celltypes, n_celltypes,
                1.0, Z->values, n_celltypes,
                ZtZ_inv, n_celltypes,
                0.0, temp, n_celltypes);
    
    mkl_free(ZtZ_inv);
    
    // Create projection matrix
    DenseMatrix* M_res = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!M_res) {
        perror("malloc projection matrix");
        mkl_free(temp);
        return NULL;
    }
    
    M_res->nrows = n_spots;
    M_res->ncols = n_spots;
    M_res->rownames = NULL;
    M_res->colnames = NULL;
    
    size_t proj_size;
    if (safe_multiply_size_t(n_spots, n_spots, &proj_size) != 0 ||
        safe_multiply_size_t(proj_size, sizeof(double), &proj_size) != 0) {
        fprintf(stderr, "Error: Projection matrix too large\n");
        free(M_res);
        mkl_free(temp);
        return NULL;
    }
    
    M_res->values = (double*)mkl_malloc(proj_size, 64);
    if (!M_res->values) {
        perror("mkl_malloc projection matrix values");
        free(M_res);
        mkl_free(temp);
        return NULL;
    }
    
    // Compute Z * (Z^T Z + λI)^(-1) * Z^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_spots, n_spots, n_celltypes,
                1.0, temp, n_celltypes,
                Z->values, n_celltypes,
                0.0, M_res->values, n_spots);
    
    mkl_free(temp);
    
    // Compute M_res = I - Z(Z^T Z + λI)^(-1) Z^T
    #pragma omp parallel for
    for (MKL_INT i = 0; i < n_spots; i++) {
        for (MKL_INT j = 0; j < n_spots; j++) {
            if (i == j) {
                M_res->values[i * n_spots + j] = 1.0 - M_res->values[i * n_spots + j];
            } else {
                M_res->values[i * n_spots + j] = -M_res->values[i * n_spots + j];
            }
        }
    }
    
    printf("Residual projection matrix computed successfully\n");
    return M_res;
}

/* Apply residual projection: R = X * M_res */
DenseMatrix* apply_residual_projection(const DenseMatrix* X, const DenseMatrix* M_res) {
    if (!X || !M_res || !X->values || !M_res->values) {
        fprintf(stderr, "Error: NULL parameters in apply_residual_projection\n");
        return NULL;
    }
    
    if (X->nrows != M_res->nrows || X->nrows != M_res->ncols) {
        fprintf(stderr, "Error: Dimension mismatch in residual projection\n");
        return NULL;
    }
    
    MKL_INT n_spots = X->nrows;
    MKL_INT n_genes = X->ncols;
    
    printf("Applying residual projection: %lld spots x %lld genes\n",
           (long long)n_spots, (long long)n_genes);
    
    DenseMatrix* R = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!R) {
        perror("malloc residual matrix");
        return NULL;
    }
    
    R->nrows = n_spots;
    R->ncols = n_genes;
    R->rownames = (char**)calloc(n_spots, sizeof(char*));
    R->colnames = (char**)calloc(n_genes, sizeof(char*));
    
    size_t residual_size;
    if (safe_multiply_size_t(n_spots, n_genes, &residual_size) != 0 ||
        safe_multiply_size_t(residual_size, sizeof(double), &residual_size) != 0) {
        fprintf(stderr, "Error: Residual matrix too large\n");
        free(R);
        return NULL;
    }
    
    R->values = (double*)mkl_malloc(residual_size, 64);
    if (!R->values || !R->rownames || !R->colnames) {
        perror("Failed to allocate residual matrix components");
        free_dense_matrix(R);
        return NULL;
    }
    
    // Copy names from X
    for (MKL_INT i = 0; i < n_spots; i++) {
        if (X->rownames && X->rownames[i]) {
            R->rownames[i] = strdup(X->rownames[i]);
        } else {
            char temp[32];
            snprintf(temp, sizeof(temp), "Spot_%lld", (long long)i);
            R->rownames[i] = strdup(temp);
        }
        if (!R->rownames[i]) {
            perror("strdup spot name for residuals");
            free_dense_matrix(R);
            return NULL;
        }
    }
    
    for (MKL_INT j = 0; j < n_genes; j++) {
        if (X->colnames && X->colnames[j]) {
            R->colnames[j] = strdup(X->colnames[j]);
        } else {
            char temp[32];
            snprintf(temp, sizeof(temp), "Gene_%lld", (long long)j);
            R->colnames[j] = strdup(temp);
        }
        if (!R->colnames[j]) {
            perror("strdup gene name for residuals");
            free_dense_matrix(R);
            return NULL;
        }
    }
    
    // Compute R = M_res * X
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_spots, n_genes, n_spots,
                1.0, M_res->values, n_spots,
                X->values, n_genes,
                0.0, R->values, n_genes);
    
    printf("Residual projection applied successfully\n");
    return R;
}

/* Center matrix columns: R_rc = R * H_n */
DenseMatrix* center_matrix_columns(const DenseMatrix* matrix) {
    if (!matrix || !matrix->values) {
        fprintf(stderr, "Error: NULL parameters in center_matrix_columns\n");
        return NULL;
    }
    
    MKL_INT n_spots = matrix->nrows;
    MKL_INT n_genes = matrix->ncols;
    
    printf("Centering matrix columns: %lld spots x %lld genes\n",
           (long long)n_spots, (long long)n_genes);
    
    DenseMatrix* centered = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!centered) {
        perror("malloc centered matrix");
        return NULL;
    }
    
    centered->nrows = n_spots;
    centered->ncols = n_genes;
    centered->rownames = (char**)calloc(n_spots, sizeof(char*));
    centered->colnames = (char**)calloc(n_genes, sizeof(char*));
    
    size_t matrix_size;
    if (safe_multiply_size_t(n_spots, n_genes, &matrix_size) != 0 ||
        safe_multiply_size_t(matrix_size, sizeof(double), &matrix_size) != 0) {
        fprintf(stderr, "Error: Centered matrix too large\n");
        free(centered);
        return NULL;
    }
    
    centered->values = (double*)mkl_malloc(matrix_size, 64);
    if (!centered->values || !centered->rownames || !centered->colnames) {
        perror("Failed to allocate centered matrix components");
        free_dense_matrix(centered);
        return NULL;
    }
    
    // Copy names
    for (MKL_INT i = 0; i < n_spots; i++) {
        if (matrix->rownames && matrix->rownames[i]) {
            centered->rownames[i] = strdup(matrix->rownames[i]);
        } else {
            char temp[32];
            snprintf(temp, sizeof(temp), "Spot_%lld", (long long)i);
            centered->rownames[i] = strdup(temp);
        }
        if (!centered->rownames[i]) {
            perror("strdup spot name for centered matrix");
            free_dense_matrix(centered);
            return NULL;
        }
    }
    
    for (MKL_INT j = 0; j < n_genes; j++) {
        if (matrix->colnames && matrix->colnames[j]) {
            centered->colnames[j] = strdup(matrix->colnames[j]);
        } else {
            char temp[32];
            snprintf(temp, sizeof(temp), "Gene_%lld", (long long)j);
            centered->colnames[j] = strdup(temp);
        }
        if (!centered->colnames[j]) {
            perror("strdup gene name for centered matrix");
            free_dense_matrix(centered);
            return NULL;
        }
    }
    
    // Center each column (gene) by subtracting mean
    #pragma omp parallel for
    for (MKL_INT j = 0; j < n_genes; j++) {
        double column_sum = 0.0;
        for (MKL_INT i = 0; i < n_spots; i++) {
            column_sum += matrix->values[i * n_genes + j];
        }
        double column_mean = column_sum / (double)n_spots;
        
        for (MKL_INT i = 0; i < n_spots; i++) {
            centered->values[i * n_genes + j] = matrix->values[i * n_genes + j] - column_mean;
        }
    }
    
    printf("Matrix columns centered successfully\n");
    return centered;
}

/* Normalize matrix rows: R_normalized = D * R_rc */
DenseMatrix* normalize_matrix_rows(const DenseMatrix* matrix) {
    if (!matrix || !matrix->values) {
        fprintf(stderr, "Error: NULL parameters in normalize_matrix_rows\n");
        return NULL;
    }
    
    MKL_INT n_spots = matrix->nrows;
    MKL_INT n_genes = matrix->ncols;
    
    printf("Normalizing matrix rows: %lld spots x %lld genes\n",
           (long long)n_spots, (long long)n_genes);
    
    DenseMatrix* normalized = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!normalized) {
        perror("malloc normalized matrix");
        return NULL;
    }
    
    normalized->nrows = n_spots;
    normalized->ncols = n_genes;
    normalized->rownames = (char**)calloc(n_spots, sizeof(char*));
    normalized->colnames = (char**)calloc(n_genes, sizeof(char*));
    
    size_t matrix_size;
    if (safe_multiply_size_t(n_spots, n_genes, &matrix_size) != 0 ||
        safe_multiply_size_t(matrix_size, sizeof(double), &matrix_size) != 0) {
        fprintf(stderr, "Error: Normalized matrix too large\n");
        free(normalized);
        return NULL;
    }
    
    normalized->values = (double*)mkl_malloc(matrix_size, 64);
    if (!normalized->values || !normalized->rownames || !normalized->colnames) {
        perror("Failed to allocate normalized matrix components");
        free_dense_matrix(normalized);
        return NULL;
    }
    
    // Copy names
    for (MKL_INT i = 0; i < n_spots; i++) {
        if (matrix->rownames && matrix->rownames[i]) {
            normalized->rownames[i] = strdup(matrix->rownames[i]);
        } else {
            char temp[32];
            snprintf(temp, sizeof(temp), "Spot_%lld", (long long)i);
            normalized->rownames[i] = strdup(temp);
        }
        if (!normalized->rownames[i]) {
            perror("strdup spot name for normalized matrix");
            free_dense_matrix(normalized);
            return NULL;
        }
    }
    
    for (MKL_INT j = 0; j < n_genes; j++) {
        if (matrix->colnames && matrix->colnames[j]) {
            normalized->colnames[j] = strdup(matrix->colnames[j]);
        } else {
            char temp[32];
            snprintf(temp, sizeof(temp), "Gene_%lld", (long long)j);
            normalized->colnames[j] = strdup(temp);
        }
        if (!normalized->colnames[j]) {
            perror("strdup gene name for normalized matrix");
            free_dense_matrix(normalized);
            return NULL;
        }
    }
    
    // Normalize each row by its L2 norm
    #pragma omp parallel for
    for (MKL_INT i = 0; i < n_spots; i++) {
        double row_sum_sq = 0.0;
        for (MKL_INT j = 0; j < n_genes; j++) {
            double val = matrix->values[i * n_genes + j];
            row_sum_sq += val * val;
        }
        
        double row_norm = sqrt(row_sum_sq / (double)n_genes);
        
        if (row_norm < ZERO_STD_THRESHOLD) {
            // Row is essentially zero, set to zero
            for (MKL_INT j = 0; j < n_genes; j++) {
                normalized->values[i * n_genes + j] = 0.0;
            }
        } else {
            // Normalize by row norm
            for (MKL_INT j = 0; j < n_genes; j++) {
                normalized->values[i * n_genes + j] = matrix->values[i * n_genes + j] / row_norm;
            }
        }
    }
    
    printf("Matrix rows normalized successfully\n");
    return normalized;
}


/* ===============================
 * RESIDUAL MORAN'S I CORE CALCULATION
 * =============================== */

/* Calculate residual Moran's I matrix: I_R = (1/S0) R_normalized W R_normalized^T */
DenseMatrix* calculate_residual_morans_i_matrix(const DenseMatrix* R_normalized, const SparseMatrix* W) {
    if (!R_normalized || !W || !R_normalized->values) {
        fprintf(stderr, "Error: Invalid parameters provided to calculate_residual_morans_i_matrix\n");
        return NULL;
    }
    if (W->nnz > 0 && !W->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in calculate_residual_morans_i_matrix\n");
        return NULL;
    }

    MKL_INT n_spots = R_normalized->nrows;
    MKL_INT n_genes = R_normalized->ncols;

    if (n_spots != W->nrows || n_spots != W->ncols) {
        fprintf(stderr, "Error: Dimension mismatch between R_normalized (%lld spots x %lld genes) and W (%lldx%lld)\n",
                (long long)n_spots, (long long)n_genes, (long long)W->nrows, (long long)W->ncols);
        return NULL;
    }
    
    if (validate_matrix_dimensions(n_genes, n_genes, "Residual Moran's I result") != MORANS_I_SUCCESS) {
        return NULL;
    }

    printf("Calculating Residual Moran's I for %lld genes using %lld spots...\n",
           (long long)n_genes, (long long)n_spots);

    DenseMatrix* result = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!result) {
        perror("Failed alloc result struct for Residual Moran's I");
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
        perror("Failed alloc result data for Residual Moran's I");
        free_dense_matrix(result);
        return NULL;
    }

    // Copy gene names
    for (MKL_INT i = 0; i < n_genes; i++) {
        if (R_normalized->colnames && R_normalized->colnames[i]) {
            result->rownames[i] = strdup(R_normalized->colnames[i]);
            result->colnames[i] = strdup(R_normalized->colnames[i]);
            if (!result->rownames[i] || !result->colnames[i]) {
                perror("Failed to duplicate gene names for Residual Moran's I result");
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

    /* Calculate scaling factor */
    double S0 = calculate_weight_sum(W);
    printf("  Sum of weights S0: %.6f\n", S0);

    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Residual Moran's I results will be NaN/Inf or 0.\n", S0);
        if (S0 == 0.0) {
            for(size_t i=0; i < (size_t)n_genes * n_genes; ++i) result->values[i] = NAN;
            return result;
        }
    }
    
    double scaling_factor = 1.0 / S0;
    printf("  Using 1/S0 = %.6e as scaling factor\n", scaling_factor);

    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(
        &W_mkl, SPARSE_INDEX_BASE_ZERO, W->nrows, W->ncols,
        W->row_ptr, W->row_ptr + 1, W->col_ind, W->values);

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W for residual)");
        free_dense_matrix(result);
        return NULL;
    }

    if (W->nnz > 0) {
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_optimize (W for residual)");
        }
    }

    printf("  Step 1: Calculating Temp_WR = W * R_normalized ...\n");
    
    size_t temp_size;
    if (safe_multiply_size_t(n_spots, n_genes, &temp_size) != 0 ||
        safe_multiply_size_t(temp_size, sizeof(double), &temp_size) != 0) {
        fprintf(stderr, "Error: Temporary matrix dimensions too large\n");
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }
    
    double* Temp_WR_values = (double*)mkl_malloc(temp_size, 64);
    if (!Temp_WR_values) {
        perror("Failed alloc Temp_WR_values");
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
        R_normalized->values,
        n_genes,
        n_genes,
        beta_mm,
        Temp_WR_values,
        n_genes
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_mm (W * R_normalized)");
        mkl_free(Temp_WR_values);
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }

    printf("  Step 2: Calculating Result = (R_normalized_T * Temp_WR) * scaling_factor ...\n");
    cblas_dgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        n_genes,
        n_genes,
        n_spots,
        scaling_factor,
        R_normalized->values,
        n_genes,
        Temp_WR_values,
        n_genes,
        beta_mm,
        result->values,
        n_genes
    );

    mkl_free(Temp_WR_values);
    mkl_sparse_destroy(W_mkl);

    printf("Residual Moran's I matrix calculation complete and scaled.\n");
    DEBUG_MATRIX_INFO(result, "residual_morans_i_result");
    return result;
}

/* Main residual Moran's I calculation function */
ResidualResults* calculate_residual_morans_i(const DenseMatrix* X, const CellTypeMatrix* Z, 
                                           const SparseMatrix* W, const ResidualConfig* config) {
    if (!X || !Z || !W || !config) {
        fprintf(stderr, "Error: NULL parameters in calculate_residual_morans_i\n");
        return NULL;
    }
    
    printf("Starting residual Moran's I analysis...\n");
    
    ResidualResults* results = (ResidualResults*)calloc(1, sizeof(ResidualResults));
    if (!results) {
        perror("Failed to allocate ResidualResults");
        return NULL;
    }
    
    // Step 1: Compute regression coefficients B̂ = (Z^T Z + λI)^(-1) Z^T X^T
    printf("Step 1: Computing regression coefficients...\n");
    results->regression_coefficients = compute_regression_coefficients(Z, X, config->regularization_lambda);
    if (!results->regression_coefficients) {
        fprintf(stderr, "Error: Failed to compute regression coefficients\n");
        free_residual_results(results);
        return NULL;
    }
    
    // Step 2: Compute residual projection matrix M_res = I - Z(Z^T Z + λI)^(-1) Z^T
    printf("Step 2: Computing residual projection matrix...\n");
    DenseMatrix* M_res = compute_residual_projection_matrix(Z, config->regularization_lambda);
    if (!M_res) {
        fprintf(stderr, "Error: Failed to compute residual projection matrix\n");
        free_residual_results(results);
        return NULL;
    }
    
    // Step 3: Apply residual projection R = X * M_res
    printf("Step 3: Applying residual projection...\n");
    DenseMatrix* R = apply_residual_projection(X, M_res);
    free_dense_matrix(M_res); // Free intermediate matrix
    if (!R) {
        fprintf(stderr, "Error: Failed to apply residual projection\n");
        free_residual_results(results);
        return NULL;
    }
    
    // Step 4: Center residuals R_rc = R * H_n
    printf("Step 4: Centering residuals...\n");
    DenseMatrix* R_centered = center_matrix_columns(R);
    free_dense_matrix(R); // Free intermediate matrix
    if (!R_centered) {
        fprintf(stderr, "Error: Failed to center residuals\n");
        free_residual_results(results);
        return NULL;
    }
    
    // Store residuals (before normalization)
    results->residuals = R_centered;
    
    // Step 5: Normalize residuals R_normalized = D * R_rc (if requested)
    DenseMatrix* R_normalized;
    if (config->normalize_residuals) {
        printf("Step 5: Normalizing residuals...\n");
        R_normalized = normalize_matrix_rows(R_centered);
        if (!R_normalized) {
            fprintf(stderr, "Error: Failed to normalize residuals\n");
            free_residual_results(results);
            return NULL;
        }
    } else {
        printf("Step 5: Skipping residual normalization (as requested)\n");
        // Use centered residuals directly
        R_normalized = (DenseMatrix*)malloc(sizeof(DenseMatrix));
        if (!R_normalized) {
            perror("malloc R_normalized");
            free_residual_results(results);
            return NULL;
        }
        *R_normalized = *R_centered; // Shallow copy
        // Don't free R_centered since results->residuals points to it
    }
    
    // Step 6: Compute residual Moran's I I_R = (1/S0) R_normalized W R_normalized^T
    printf("Step 6: Computing residual Moran's I...\n");
    results->residual_morans_i = calculate_residual_morans_i_matrix(R_normalized, W);
    
    if (config->normalize_residuals) {
        free_dense_matrix(R_normalized); // Free if we created a separate normalized matrix
    } else {
        free(R_normalized); // Just free the structure, not the data
    }
    
    if (!results->residual_morans_i) {
        fprintf(stderr, "Error: Failed to compute residual Moran's I matrix\n");
        free_residual_results(results);
        return NULL;
    }
    
    printf("Residual Moran's I analysis completed successfully.\n");
    return results;
}

/* ===============================
 * RESIDUAL PERMUTATION TESTING
 * =============================== */

/* Residual permutation worker function */
static int residual_permutation_worker(const DenseMatrix* X_original,
                                      const CellTypeMatrix* Z,
                                      const SparseMatrix* W,
                                      const ResidualConfig* config,
                                      const PermutationParams* params,
                                      int thread_id,
                                      int start_perm,
                                      int end_perm,
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
    
    if (!X_perm.values || !gene_buffer || !indices_buffer) {
        DEBUG_PRINT("Thread %d: Memory allocation failed for residual permutation", thread_id);
        // Cleanup
        if (X_perm.values) mkl_free(X_perm.values);
        if (gene_buffer) mkl_free(gene_buffer);
        if (indices_buffer) mkl_free(indices_buffer);
        return -1;
    }
    
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
        
        // Calculate residual Moran's I for permuted data
        // This involves the full residual analysis pipeline
        ResidualResults* perm_results = calculate_residual_morans_i(&X_perm, Z, W, config);
        if (!perm_results || !perm_results->residual_morans_i) {
            DEBUG_PRINT("Thread %d: Permutation %d failed", thread_id, perm);
            if (perm_results) free_residual_results(perm_results);
            continue; // Skip this permutation
        }
        
        // Accumulate statistics
        for (size_t idx = 0; idx < matrix_elements; idx++) {
            double perm_val = perm_results->residual_morans_i->values[idx];
            if (!isfinite(perm_val)) perm_val = 0.0;
            
            local_mean_sum[idx] += perm_val;
            local_var_sum_sq[idx] += perm_val * perm_val;
            
            if (params->p_value_output && local_p_counts && observed_results) {
                if (fabs(perm_val) >= fabs(observed_results->values[idx])) {
                    local_p_counts[idx]++;
                }
            }
        }
        
        free_residual_results(perm_results);
    }
    
    // Cleanup
    mkl_free(X_perm.values);
    mkl_free(gene_buffer);
    mkl_free(indices_buffer);
    
    return 0;
}

/* Run residual permutation test */
PermutationResults* run_residual_permutation_test(const DenseMatrix* X, const CellTypeMatrix* Z,
                                                 const SparseMatrix* W, const PermutationParams* params,
                                                 const ResidualConfig* config) {
    
    if (!X || !Z || !W || !params || !config || !X->values || !X->colnames) {
        fprintf(stderr, "Error: Invalid parameters provided to run_residual_permutation_test\n");
        return NULL;
    }
    if (W->nnz > 0 && !W->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in run_residual_permutation_test\n");
        return NULL;
    }

    MKL_INT n_spots = X->nrows;
    MKL_INT n_genes = X->ncols;
    int n_perm = params->n_permutations;

    if (validate_matrix_dimensions(n_spots, n_genes, "residual permutation test input") != MORANS_I_SUCCESS) {
        return NULL;
    }

    if (n_genes == 0 || n_spots == 0) {
        fprintf(stderr, "Error: Expression matrix has zero dimensions in run_residual_permutation_test\n");
        return NULL;
    }
    if (n_perm <= 0) {
        fprintf(stderr, "Error: Number of permutations (%d) must be positive\n", n_perm);
        return NULL;
    }

    printf("Running residual permutation test with %d permutations for %lld genes...\n", 
           n_perm, (long long)n_genes);

    // Calculate observed residual Moran's I for comparison
    ResidualResults* observed_residual_results = calculate_residual_morans_i(X, Z, W, config);
    if (!observed_residual_results || !observed_residual_results->residual_morans_i) {
        fprintf(stderr, "Error: Failed to calculate observed residual Moran's I for permutation test\n");
        if (observed_residual_results) free_residual_results(observed_residual_results);
        return NULL;
    }

    DenseMatrix* observed_results = observed_residual_results->residual_morans_i;

    // Allocate results structure
    PermutationResults* results = (PermutationResults*)calloc(1, sizeof(PermutationResults));
    if (!results) {
        perror("Failed to allocate PermutationResults structure for residual test");
        free_residual_results(observed_residual_results);
        return NULL;
    }

    size_t matrix_elements = (size_t)n_genes * n_genes;
    size_t matrix_bytes;
    if (safe_multiply_size_t(matrix_elements, sizeof(double), &matrix_bytes) != 0) {
        fprintf(stderr, "Error: Matrix size too large for residual permutation results\n");
        free(results);
        free_residual_results(observed_residual_results);
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
        perror("Failed to allocate result matrix structures for residual test");
        free_permutation_results(results);
        free_residual_results(observed_residual_results);
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
            perror("Failed to allocate result matrix components for residual test");
            free_permutation_results(results);
            free_residual_results(observed_residual_results);
            return NULL;
        }
        
        // Copy gene names
        for (MKL_INT i = 0; i < n_genes; i++) {
            const char* gene_name = (X->colnames[i]) ? X->colnames[i] : "UNKNOWN_GENE";
            matrices[m]->rownames[i] = strdup(gene_name);
            matrices[m]->colnames[i] = strdup(gene_name);
            if (!matrices[m]->rownames[i] || !matrices[m]->colnames[i]) {
                perror("Failed to duplicate gene names for residual permutation results");
                free_permutation_results(results);
                free_residual_results(observed_residual_results);
                return NULL;
            }
        }
    }

    // Run permutations using multiple threads
    int num_threads = omp_get_max_threads();
    int perms_per_thread = n_perm / num_threads;
    int remaining_perms = n_perm % num_threads;
    
    printf("Starting residual permutation loop (%d permutations) using %d OpenMP threads...\n", 
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
                fprintf(stderr, "Thread %d: Failed to allocate local buffers for residual permutation\n", thread_id);
                error_occurred = 1;
            }
        } else if (!error_occurred) {
            // Run permutations for this thread
            int worker_result = residual_permutation_worker(X, Z, W, config, params, thread_id, 
                                                          start_perm, end_perm,
                                                          local_mean_sum, local_var_sum_sq,
                                                          local_p_counts, observed_results);
            
            if (worker_result != 0) {
                #pragma omp critical
                {
                    fprintf(stderr, "Thread %d: Residual permutation worker failed\n", thread_id);
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
    printf("Residual permutation loop completed in %.2f seconds\n", loop_end_time - loop_start_time);

    if (error_occurred) {
        fprintf(stderr, "Error occurred during residual permutation execution\n");
        free_permutation_results(results);
        free_residual_results(observed_residual_results);
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
                fprintf(stderr, "Warning: Negative variance (%.4e) for gene pair (%lld,%lld) in residual test\n",
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

    free_residual_results(observed_residual_results);
    printf("Residual permutation test complete.\n");
    return results;
}

/* ===============================
 * UPDATED FILE I/O FUNCTIONS
 * =============================== */

/* Save regression coefficients */
int save_regression_coefficients(const DenseMatrix* coefficients, const char* output_file) {
    if (!coefficients || !output_file) {
        fprintf(stderr, "Error: Cannot save NULL regression coefficients or empty filename\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    
    MKL_INT n_celltypes = coefficients->nrows;
    MKL_INT n_genes = coefficients->ncols;
    
    printf("Saving regression coefficients to %s...\n", output_file);
    
    // Write header
    fprintf(fp, "CellType");
    for (MKL_INT j = 0; j < n_genes; j++) {
        fprintf(fp, "\t%s", coefficients->colnames && coefficients->colnames[j] ? 
                coefficients->colnames[j] : "UNKNOWN_GENE");
    }
    fprintf(fp, "\n");
    
    // Write data rows
    for (MKL_INT i = 0; i < n_celltypes; i++) {
        fprintf(fp, "%s", coefficients->rownames && coefficients->rownames[i] ? 
                coefficients->rownames[i] : "UNKNOWN_CELLTYPE");
        
        for (MKL_INT j = 0; j < n_genes; j++) {
            double value = coefficients->values[i * n_genes + j];
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
    printf("Regression coefficients saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}

/* Save residual analysis results */
int save_residual_results(const ResidualResults* results, const char* output_prefix) {
    if (!results || !output_prefix) {
        fprintf(stderr, "Error: Cannot save NULL residual results or empty prefix\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    char filename_buffer[BUFFER_SIZE];
    int status = MORANS_I_SUCCESS;
    
    printf("--- Saving Residual Analysis Results ---\n");
    
    // Save residual Moran's I matrix (lower triangular)
    if (results->residual_morans_i && results->residual_morans_i->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_residual_morans_i_raw.tsv", output_prefix);
        printf("Saving residual Moran's I (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->residual_morans_i, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving residual Moran's I matrix\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }
    
    // Save regression coefficients
    if (results->regression_coefficients && results->regression_coefficients->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_regression_coefficients.tsv", output_prefix);
        printf("Saving regression coefficients to: %s\n", filename_buffer);
        int temp_status = save_regression_coefficients(results->regression_coefficients, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving regression coefficients\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }
    
    // Save residual Z-scores (if available)
    if (results->residual_zscores && results->residual_zscores->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_residual_zscores_lower_tri.tsv", output_prefix);
        printf("Saving residual Z-scores (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->residual_zscores, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving residual Z-scores\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }
    
    // Save residual P-values (if available)
    if (results->residual_pvalues && results->residual_pvalues->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_residual_pvalues_lower_tri.tsv", output_prefix);
        printf("Saving residual P-values (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->residual_pvalues, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving residual P-values\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }
    
    // Save residuals matrix (full matrix)
    if (results->residuals && results->residuals->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_residuals_matrix.tsv", output_prefix);
        printf("Saving residuals matrix to: %s\n", filename_buffer);
        int temp_status = save_results(results->residuals, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving residuals matrix\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }
    
    if (status == MORANS_I_SUCCESS) {
        printf("Residual analysis results saved with prefix: %s\n", output_prefix);
    }
    return status;
}

/* ===============================
 * MEMORY MANAGEMENT FUNCTIONS (UPDATED)
 * =============================== */

/* Free cell type matrix */
void free_celltype_matrix(CellTypeMatrix* matrix) {
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

/* Free residual results */
void free_residual_results(ResidualResults* results) {
    if (!results) return;
    
    if (results->regression_coefficients) free_dense_matrix(results->regression_coefficients);
    if (results->residuals) free_dense_matrix(results->residuals);
    if (results->residual_morans_i) free_dense_matrix(results->residual_morans_i);
    if (results->residual_mean_perm) free_dense_matrix(results->residual_mean_perm);
    if (results->residual_var_perm) free_dense_matrix(results->residual_var_perm);
    if (results->residual_zscores) free_dense_matrix(results->residual_zscores);
    if (results->residual_pvalues) free_dense_matrix(results->residual_pvalues);
    
    free(results);
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
    size_t values_size_bytes; // Renamed to avoid confusion with element count
    size_t num_elements;
    if (safe_multiply_size_t(n_genes, n_spots, &num_elements) != 0 ||
        safe_multiply_size_t(num_elements, sizeof(double), &values_size_bytes) != 0) {
        fprintf(stderr, "Error: Matrix dimensions too large for z_normalize\n");
        free(normalized);
        return NULL;
    }
    
    normalized->values = (double*)mkl_malloc(values_size_bytes, 64);
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
                    double variance = (n_finite > 0) ? sum_sq_diff / n_finite : 0.0; // Use n_finite, not n_spots for variance with NaNs
                    if (variance < 0.0) variance = 0.0; // Should not happen with sum of squares
                    std_dev = sqrt(variance);
                } else if (n_finite == 1) { // Single finite value, mean is the value, std_dev is 0
                    mean = sum; // sum is just the single value
                    std_dev = 0.0;
                }
                // if n_finite == 0, mean and std_dev remain 0.0

                if (n_finite <= 1 || std_dev < ZERO_STD_THRESHOLD) {
                    for (MKL_INT j = 0; j < n_spots; j++) {
                        current_gene_row_output[j] = 0.0;
                    }
                } else {
                    for (MKL_INT j = 0; j < n_spots; j++) {
                        if (isfinite(current_gene_row_input[j])) {
                            centered_values_tl[j] = current_gene_row_input[j] - mean;
                        } else {
                            centered_values_tl[j] = 0.0; // Non-finite values become 0 after normalization
                        }
                        std_dev_vector_tl[j] = std_dev; // Broadcast std_dev
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
        W_empty->row_ptr = (MKL_INT*)mkl_calloc(1, sizeof(MKL_INT), 64); // n_spots_valid + 1 = 1
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
    if (estimated_neighbors_per_spot <= 0) estimated_neighbors_per_spot = 27; // Default from STUtility (max_radius=5 -> ~117, this is lower)
    
    size_t initial_capacity_size_t;
    if (safe_multiply_size_t(n_spots_valid, estimated_neighbors_per_spot, &initial_capacity_size_t) != 0) {
        initial_capacity_size_t = n_spots_valid; // Fallback to minimum if overflow
    }
    
    MKL_INT initial_capacity = (MKL_INT)initial_capacity_size_t;
    size_t max_dense_elements; // max possible NNZ if dense
    if (safe_multiply_size_t(n_spots_valid, n_spots_valid, &max_dense_elements) == 0 && 
        initial_capacity > (MKL_INT)max_dense_elements) { // Ensure initial_capacity is not larger than dense
        initial_capacity = (MKL_INT)max_dense_elements;
    }

    if (initial_capacity <= 0) initial_capacity = (n_spots_valid > 0) ? n_spots_valid : 1; // Ensure positive

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

    volatile int critical_error_flag = 0; // Flag for critical errors like memory allocation

    #pragma omp parallel
    {
        // Thread-local storage for COO triplets
        MKL_INT local_initial_cap = (estimated_neighbors_per_spot > 0) ? estimated_neighbors_per_spot * 2 : 256; // Heuristic
        MKL_INT local_capacity = local_initial_cap;
        MKL_INT* local_I_tl = (MKL_INT*)malloc((size_t)local_capacity * sizeof(MKL_INT));
        MKL_INT* local_J_tl = (MKL_INT*)malloc((size_t)local_capacity * sizeof(MKL_INT));
        double* local_V_tl = (double*)malloc((size_t)local_capacity * sizeof(double));
        MKL_INT local_nnz_tl = 0;
        int thread_alloc_error = 0; // Thread-local error flag

        if (!local_I_tl || !local_J_tl || !local_V_tl) {
            #pragma omp critical
            {
                fprintf(stderr, "Error: Thread %d failed to alloc thread-local COO buffers.\n", omp_get_thread_num());
                critical_error_flag = 1; // Signal global error
            }
            thread_alloc_error = 1; // Mark this thread as having an error
        }

        if (!thread_alloc_error && !critical_error_flag) { // Proceed only if no errors so far
            #pragma omp for schedule(dynamic, 128) // Dynamic schedule for load balancing
            for (MKL_INT i = 0; i < n_spots_valid; i++) {
                if (critical_error_flag || thread_alloc_error) { // Check flags before/during loop
                     continue; // Skip iteration if error occurred. Cannot 'break' from OpenMP for loop.
                }

                for (MKL_INT j = 0; j < n_spots_valid; j++) {
                    MKL_INT row_shift_abs = llabs(spot_row_valid[i] - spot_row_valid[j]);
                    MKL_INT col_shift_abs = llabs(spot_col_valid[i] - spot_col_valid[j]);

                    // Check bounds for distance_matrix access
                    if (row_shift_abs < distance_matrix->nrows && col_shift_abs < distance_matrix->ncols) {
                        double weight = distance_matrix->values[row_shift_abs * distance_matrix->ncols + col_shift_abs];

                        if (fabs(weight) > WEIGHT_THRESHOLD) { // Weight is significant
                            if (local_nnz_tl >= local_capacity) { // Resize local buffer if needed
                                MKL_INT new_capacity = (MKL_INT)(local_capacity * 1.5) + 1; // Growth factor
                                MKL_INT* temp_li_new = (MKL_INT*)realloc(local_I_tl, (size_t)new_capacity * sizeof(MKL_INT));
                                MKL_INT* temp_lj_new = (MKL_INT*)realloc(local_J_tl, (size_t)new_capacity * sizeof(MKL_INT));
                                double* temp_lv_new = (double*)realloc(local_V_tl, (size_t)new_capacity * sizeof(double));
                                
                                if (!temp_li_new || !temp_lj_new || !temp_lv_new) {
                                    #pragma omp critical
                                    {
                                        fprintf(stderr, "Error: Thread %d failed to realloc thread-local COO buffers.\n", omp_get_thread_num());
                                        critical_error_flag = 1;
                                    }
                                    // Keep old pointers if realloc failed for some but not all
                                    local_I_tl = temp_li_new ? temp_li_new : local_I_tl;
                                    local_J_tl = temp_lj_new ? temp_lj_new : local_J_tl;
                                    local_V_tl = temp_lv_new ? temp_lv_new : local_V_tl;
                                    thread_alloc_error = 1; // Mark error for this thread
                                    continue; // Skip to next i due to realloc failure
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
                if (thread_alloc_error) { // If error occurred within inner loop (e.g. realloc)
                     continue; // Skip to next i
                }
            } // End of omp for loop
        } // End of if (!thread_alloc_error && !critical_error_flag)

        // Merge thread-local results into global COO arrays (critical section)
        if (!critical_error_flag && !thread_alloc_error && local_nnz_tl > 0) {
            #pragma omp critical
            {
                if (!critical_error_flag) { // Re-check global error flag inside critical section
                    if (nnz_count + local_nnz_tl > current_capacity) { // Resize global buffer if needed
                        MKL_INT needed_capacity = nnz_count + local_nnz_tl;
                        MKL_INT new_global_capacity = current_capacity;
                        while(new_global_capacity < needed_capacity && new_global_capacity > 0) { // Prevent overflow with new_global_capacity > 0
                            new_global_capacity = (MKL_INT)(new_global_capacity * 1.5) + 1;
                            if (new_global_capacity <= current_capacity) { // Overflow or no increase
                                new_global_capacity = needed_capacity > current_capacity ? needed_capacity : current_capacity + 1; // Try to reach at least needed
                                if (new_global_capacity <= current_capacity) { // Still stuck, indicates potential overflow
                                    critical_error_flag = 1; break;
                                }
                            }
                        }
                        if(critical_error_flag) {
                             fprintf(stderr, "Error: Global COO buffer resize failed due to capacity calculation issue.\n");
                        }

                        size_t max_dense_nnz;
                        if (safe_multiply_size_t(n_spots_valid, n_spots_valid, &max_dense_nnz) == 0 && 
                            new_global_capacity > (MKL_INT)max_dense_nnz) {
                            new_global_capacity = (MKL_INT)max_dense_nnz;
                        }
                        
                        if (needed_capacity > new_global_capacity && n_spots_valid > 0) { // Check if still insufficient
                            fprintf(stderr, "Error: Cannot resize global COO buffer large enough (%lld needed, max %lld).\n", 
                                   (long long)needed_capacity, (long long)new_global_capacity);
                            critical_error_flag = 1;
                        } else if (n_spots_valid > 0 && !critical_error_flag) {
                            printf("  Resizing global COO buffer from %lld to %lld\n", 
                                   (long long)current_capacity, (long long)new_global_capacity);
                            MKL_INT* temp_gi_new = (MKL_INT*)realloc(temp_I, (size_t)new_global_capacity * sizeof(MKL_INT));
                            MKL_INT* temp_gj_new = (MKL_INT*)realloc(temp_J, (size_t)new_global_capacity * sizeof(MKL_INT));
                            double*  temp_gv_new = (double*)realloc(temp_V, (size_t)new_global_capacity * sizeof(double));

                            if (!temp_gi_new || !temp_gj_new || !temp_gv_new) {
                                fprintf(stderr, "Error: Failed to realloc global COO buffers.\n");
                                critical_error_flag = 1;
                                // Keep old pointers if realloc failed, to free them later
                                temp_I = temp_gi_new ? temp_gi_new : temp_I;
                                temp_J = temp_gj_new ? temp_gj_new : temp_J;
                                temp_V = temp_gv_new ? temp_gv_new : temp_V;
                            } else {
                                temp_I = temp_gi_new; 
                                temp_J = temp_gj_new; 
                                temp_V = temp_gv_new;
                                current_capacity = new_global_capacity;
                            }
                        }
                    }

                    // Copy local data to global arrays if no critical error and space is sufficient
                    if (!critical_error_flag && (nnz_count + local_nnz_tl <= current_capacity)) {
                        memcpy(temp_I + nnz_count, local_I_tl, (size_t)local_nnz_tl * sizeof(MKL_INT));
                        memcpy(temp_J + nnz_count, local_J_tl, (size_t)local_nnz_tl * sizeof(MKL_INT));
                        memcpy(temp_V + nnz_count, local_V_tl, (size_t)local_nnz_tl * sizeof(double));
                        nnz_count += local_nnz_tl;
                    } else if (!critical_error_flag) { // Should not happen if resize logic is correct
                        fprintf(stderr, "Warning: Could not merge thread %d results due to insufficient space after resize attempt.\n", omp_get_thread_num());
                        critical_error_flag = 1; // Treat as critical if merge fails
                    }
                } // End of if (!critical_error_flag) inside critical section
            } // End of omp critical
        } // End of if (!critical_error_flag && !thread_alloc_error && local_nnz_tl > 0)

        // Free thread-local buffers
        if(local_I_tl) free(local_I_tl);
        if(local_J_tl) free(local_J_tl);
        if(local_V_tl) free(local_V_tl);
    } // End of omp parallel

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

    size_t nnz_alloc_size = (nnz_count > 0) ? (size_t)nnz_count : 1; // MKL requires size >= 1 for empty arrays in some create calls
    W->row_ptr = (MKL_INT*)mkl_malloc(((size_t)n_spots_valid + 1) * sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc(nnz_alloc_size * sizeof(MKL_INT), 64);
        W->values  = (double*)mkl_malloc(nnz_alloc_size * sizeof(double), 64);
    } else { // Handle nnz_count == 0
        W->col_ind = NULL; // Or mkl_malloc(1 * sizeof(MKL_INT), 64) if MKL needs non-NULL
        W->values = NULL;  // Or mkl_malloc(1 * sizeof(double), 64)
    }

    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) {
        perror("Failed to allocate CSR arrays for W");
        mkl_free(W->row_ptr); mkl_free(W->col_ind); mkl_free(W->values); free(W);
        free(temp_I); free(temp_J); free(temp_V);
        return NULL;
    }

    // Convert COO to CSR
    if (nnz_count > 0) {
        // Initialize row_ptr counts part
        for (MKL_INT i = 0; i <= n_spots_valid; ++i) W->row_ptr[i] = 0;
        // Count elements in each row
        for (MKL_INT k = 0; k < nnz_count; ++k) W->row_ptr[temp_I[k] + 1]++;
        // Cumulative sum to get row_ptr
        for (MKL_INT i = 0; i < n_spots_valid; ++i) W->row_ptr[i + 1] += W->row_ptr[i];

        // Temporary array to keep track of current insertion positions for each row
        // MKL's mkl_sparse_coo_to_csr can also be used if available and convenient
        MKL_INT* current_insertion_pos = (MKL_INT*)malloc(((size_t)n_spots_valid) * sizeof(MKL_INT)); // Only n_spots_valid needed
        if (!current_insertion_pos) {
            perror("Failed to allocate current_insertion_pos array");
            free_sparse_matrix(W);
            free(temp_I); free(temp_J); free(temp_V);
            return NULL;
        }
        // Initialize current_insertion_pos with starting positions of each row from row_ptr
        memcpy(current_insertion_pos, W->row_ptr, ((size_t)n_spots_valid) * sizeof(MKL_INT));

        // Place elements into col_ind and values
        for (MKL_INT k = 0; k < nnz_count; ++k) {
            MKL_INT row = temp_I[k];
            MKL_INT index_in_csr = current_insertion_pos[row];
            W->col_ind[index_in_csr] = temp_J[k];
            W->values[index_in_csr] = temp_V[k];
            current_insertion_pos[row]++;
        }
        free(current_insertion_pos);
    } else { // nnz_count == 0
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
            
            if (row_end > row_start) { // If row is not empty
                double row_sum = 0.0;
                for (MKL_INT k = row_start; k < row_end; k++) {
                    row_sum += W->values[k];
                }
                
                if (row_sum > ZERO_STD_THRESHOLD) { // Avoid division by zero or tiny numbers
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

    // Sort column indices within each row (MKL requirement for some routines, good practice)
    if (W->nnz > 0) {
        sparse_matrix_t W_mkl_tmp_handle;
        // MKL expects row_end (ja) to be W->row_ptr + 1.
        sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl_tmp_handle, SPARSE_INDEX_BASE_ZERO,
                                                        W->nrows, W->ncols, W->row_ptr,
                                                        W->row_ptr + 1, W->col_ind, W->values);
        if (status == SPARSE_STATUS_SUCCESS) {
            status = mkl_sparse_order(W_mkl_tmp_handle); // Sorts column indices and permutes values accordingly
            if (status != SPARSE_STATUS_SUCCESS) {
                print_mkl_status(status, "mkl_sparse_order (W)");
            }
            // Note: mkl_sparse_order modifies the arrays W->col_ind and W->values in place.
            mkl_sparse_destroy(W_mkl_tmp_handle); // Destroy temporary handle, not the data arrays.
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
        return WEIGHT_FORMAT_AUTO; // Return AUTO to signify error or let caller handle
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        // Don't print strerror directly in library function if it's meant to be more generic,
        // but for this tool it's fine.
        fprintf(stderr, "Error: Cannot open weight matrix file '%s' for format detection: %s\n", 
                filename, strerror(errno));
        return WEIGHT_FORMAT_AUTO; // Indicate failure to detect
    }
    
    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    int detected_format = WEIGHT_FORMAT_AUTO; // Default to AUTO/error

    line_len = getline(&line, &line_buf_size, fp);
    
    if (line_len > 0) {
        // Trim whitespace (especially newline characters)
        char* current_line_ptr = line;
        while (line_len > 0 && (current_line_ptr[line_len - 1] == '\n' || current_line_ptr[line_len - 1] == '\r')) {
            current_line_ptr[--line_len] = '\0';
        }
        
        if (line_len > 0) { // Ensure line is not empty after trimming
            char* line_copy = strdup(current_line_ptr); // Work on a copy for strtok
            if (!line_copy) {
                perror("strdup in detect_weight_matrix_format");
                // Fallthrough to cleanup and return WEIGHT_FORMAT_AUTO
            } else {
                int field_count = 0;
                char* token = strtok(line_copy, "\t");
                while (token != NULL) {
                    field_count++;
                    token = strtok(NULL, "\t");
                }
                free(line_copy); // Free the copy

                if (field_count == 3) {
                    // Further heuristics to distinguish COO from TSV if needed
                    // For example, check if the first two fields look like coordinates (contain 'x')
                    // or if they are more general strings.
                    // This simple version might misclassify if spot names are "1x1", "1x2" in TSV.
                    // A more robust check might try to parse the third field as a number.
                    // For now, let's use a simplified heuristic based on typical use.
                    // If the original code had a more complex regex for this, reinstate it.
                    // The provided code originally had a regex in a different context.
                    // A simple heuristic: If fields are short and potentially numeric/coord-like.
                    char* line_copy2 = strdup(current_line_ptr); // New copy for checking content
                    if (line_copy2) {
                        char* first_field = strtok(line_copy2, "\t");
                        if (first_field) {
                            // If first field looks like a coordinate (e.g., "12x34")
                            // This is a basic check; a regex would be more robust.
                            // A simple check: contains 'x' and digits.
                            int has_x = 0, has_digit = 0;
                            char* p = first_field;
                            while(*p) {
                                if (*p == 'x' || *p == 'X') has_x = 1;
                                if (isdigit((unsigned char)*p)) has_digit = 1;
                                p++;
                            }
                            if (has_x && has_digit) {
                                detected_format = WEIGHT_FORMAT_SPARSE_COO;
                            } else {
                                detected_format = WEIGHT_FORMAT_SPARSE_TSV;
                            }
                        } else { // Should not happen if field_count was 3
                             detected_format = WEIGHT_FORMAT_SPARSE_TSV;
                        }
                        free(line_copy2);
                    } else {
                        perror("strdup line_copy2 in detect_weight_matrix_format");
                        detected_format = WEIGHT_FORMAT_SPARSE_TSV; // Fallback
                    }

                } else if (field_count > 3) {
                    detected_format = WEIGHT_FORMAT_DENSE;
                } else if (field_count < 3 && field_count > 0) {
                    // Could be an error or a very sparse format not well-defined
                    fprintf(stderr, "Warning: File '%s' has only %d fields in the first line. Format detection might be inaccurate.\n", filename, field_count);
                    detected_format = WEIGHT_FORMAT_SPARSE_TSV; // Default for few fields
                } else { // field_count == 0 (empty line was already handled) or other unexpected
                    fprintf(stderr, "Warning: Could not determine format from first line of '%s' (field_count=%d).\n", filename, field_count);
                }
            }
        } else { // Line became empty after trimming EOL
            fprintf(stderr, "Warning: First line of '%s' is effectively empty after EOL trimming.\n", filename);
        }
    } else { // getline failed or empty file
        if (ferror(fp)) {
            fprintf(stderr, "Error reading from '%s' for format detection: %s\n", filename, strerror(errno));
        } else {
            fprintf(stderr, "Warning: File '%s' is empty or unreadable for format detection.\n", filename);
        }
    }
    
    if (line) free(line);
    fclose(fp);
    
    const char* format_names[] = {"AUTO/ERROR", "DENSE", "SPARSE_COO", "SPARSE_TSV"};
    if (detected_format > 0 && detected_format < 4) { // Valid detected format
        printf("Detected weight matrix format: %s\n", format_names[detected_format]);
    } else {
        printf("Could not reliably detect weight matrix format for '%s'. Defaulting or error.\n", filename);
    }
    
    return detected_format;
}

/* Validate that weight matrix is compatible with expression data */
int validate_weight_matrix(const SparseMatrix* W, char** spot_names, MKL_INT n_spots) {
    // spot_names parameter is from the expression data (X_calc->rownames or equivalent)
    // n_spots is the number of rows/cols W *should* have, matching expression data.
    if (!W || (n_spots > 0 && !spot_names)) { // Allow n_spots = 0 if W is also empty
        fprintf(stderr, "Error: NULL parameters in validate_weight_matrix (W=%p, spot_names=%p for n_spots=%lld)\n",
                (void*)W, (void*)spot_names, (long long)n_spots);
        return MORANS_I_ERROR_PARAMETER;
    }
    
    if (W->nrows != n_spots || W->ncols != n_spots) {
        fprintf(stderr, "Error: Custom Weight matrix dimensions (%lldx%lld) do not match the number of spots/cells (%lld) from expression data.\n",
                (long long)W->nrows, (long long)W->ncols, (long long)n_spots);
        return MORANS_I_ERROR_PARAMETER;
    }
    
    // Check for negative weights
    MKL_INT negative_weights = 0;
    if (W->values) { // W->values can be NULL if W->nnz is 0
        for (MKL_INT i = 0; i < W->nnz; i++) {
            if (W->values[i] < 0.0) {
                negative_weights++;
            }
        }
    }
    
    if (negative_weights > 0) {
        fprintf(stderr, "Warning: Found %lld negative weights in custom weight matrix. This is unusual for standard spatial weights but might be intended.\n", 
               (long long)negative_weights);
    }
    
    double weight_sum = calculate_weight_sum(W); // Uses W->values and W->nnz
    printf("Custom weight matrix validation: %lldx%lld matrix, %lld non-zeros, sum of weights (S0) = %.6f\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz, weight_sum);
    
    if (W->nnz > 0 && fabs(weight_sum) < ZERO_STD_THRESHOLD) { // Check sum only if matrix is not empty
        fprintf(stderr, "Warning: Sum of custom weights is near zero (%.6e) for a non-empty matrix. Moran's I may be undefined or unstable.\n", 
                weight_sum);
    }
    if (W->nnz == 0 && n_spots > 0) {
        fprintf(stderr, "Warning: Custom weight matrix has 0 non-zero entries for %lld spots. Moran's I will be 0 or undefined.\n", (long long)n_spots);
    }
    
    // Optional: Check if row/column indices are within bounds [0, n_spots-1]
    // This should be guaranteed if W was constructed correctly, but for an arbitrary custom W:
    if (W->col_ind && W->nnz > 0) {
        for (MKL_INT i = 0; i < W->nnz; ++i) {
            if (W->col_ind[i] < 0 || W->col_ind[i] >= W->ncols) {
                fprintf(stderr, "Error: Invalid column index %lld (at NNZ index %lld) in custom weight matrix. Expected range [0, %lld].\n",
                        (long long)W->col_ind[i], (long long)i, (long long)W->ncols -1);
                return MORANS_I_ERROR_PARAMETER;
            }
        }
    }
    if (W->row_ptr) {
        for (MKL_INT i = 0; i < W->nrows; ++i) {
            if (W->row_ptr[i] > W->row_ptr[i+1] || W->row_ptr[i] < 0 || W->row_ptr[i+1] > W->nnz) {
                 fprintf(stderr, "Error: Invalid row_ptr entry for row %lld in custom weight matrix.\n", (long long)i);
                 return MORANS_I_ERROR_PARAMETER;
            }
        }
         if (W->row_ptr[0] != 0 || W->row_ptr[W->nrows] != W->nnz) {
             fprintf(stderr, "Error: Invalid CSR row_ptr structure (start/end mismatch) in custom weight matrix.\n");
             return MORANS_I_ERROR_PARAMETER;
         }
    }


    return MORANS_I_SUCCESS;
}

/* Helper function to determine data start offset by checking for a header */
static long get_data_start_offset(FILE* fp, char** line_buffer, size_t* line_buf_size,
                                  const char* filename_for_errmsg) {
    long original_pos = ftell(fp);
    if (original_pos == -1) {
        perror("get_data_start_offset: ftell failed at start");
        return -1L;
    }
    rewind(fp); // Go to the beginning to check the first line

    ssize_t line_len = getline(line_buffer, line_buf_size, fp);
    if (line_len <= 0) { // Empty or unreadable file
        if (ferror(fp)) {
             fprintf(stderr, "Error reading first line of '%s': %s\n", filename_for_errmsg, strerror(errno));
        } else {
             fprintf(stderr, "Warning: File '%s' is empty or first line unreadable.\n", filename_for_errmsg);
        }
        fseek(fp, original_pos, SEEK_SET); // Restore original position
        return 0L; // Assume no header, start from beginning (though it's empty)
    }

    // Trim EOL chars from the line before copying
    char* current_line_ptr = *line_buffer;
    while (line_len > 0 && (current_line_ptr[line_len - 1] == '\n' || current_line_ptr[line_len - 1] == '\r')) {
        current_line_ptr[--line_len] = '\0';
    }
    
    char* first_line_copy = strdup(current_line_ptr);
    if (!first_line_copy) {
        perror("get_data_start_offset: strdup failed");
        fseek(fp, original_pos, SEEK_SET);
        return -1L;
    }

    char* token = strtok(first_line_copy, "\t");
    int looks_like_header = 0;
    if (token) {
        // Heuristic: check for common header keywords or non-numeric/non-coord-like first token
        // This heuristic is specific to expected formats.
        // For sparse_tsv/coo: first token might be a spot name or coordinate.
        // If it contains alpha characters not part of "x" (for "RxC"), it's likely a header.
        int has_alpha = 0;
        char *p = token;
        while(*p) {
            if (isalpha((unsigned char)*p) && *p != 'x' && *p != 'X') { // Allow 'x' for RxC format
                has_alpha = 1;
                break;
            }
            p++;
        }
        // A more robust check specific to the format (e.g. _tsv checks for "spot", "weight")
        // could be done if format is known here. For now, a general alpha check.
        if (has_alpha || strstr(token, "spot") || strstr(token, "weight") || strstr(token, "coord") ||
            strstr(token, "row") || strstr(token, "col") || strstr(token, "from") || strstr(token, "to")) {
            looks_like_header = 1;
        }
    }
    free(first_line_copy);

    long data_offset;
    if (looks_like_header) {
        data_offset = ftell(fp); // Position after reading the header line
        if (data_offset == -1L) {
            perror("get_data_start_offset: ftell failed after header read");
            fseek(fp, original_pos, SEEK_SET);
            return -1L;
        }
    } else {
        data_offset = 0; // No header, data starts at the beginning
    }
    
    // Restore original file pointer position before returning, or seek to data_offset
    // Let's keep it simple: the caller will fseek to the returned offset.
    // So, ensure fp is reset to its original state if we don't want this function to change it.
    // OR, this function's purpose is to position fp correctly for the next read.
    // The current callers (read_sparse_matrix_tsv) handle fseek themselves.
    // So, just return offset. We already rewound fp.
    return data_offset;
}

/* Read dense weight matrix from TSV file */
SparseMatrix* read_dense_weight_matrix(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots) {
    if (!filename || !spot_map || n_spots <= 0) {
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
    
    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
        line[--line_len] = '\0';
    }
    
    char* header_copy = strdup(line);
    if (!header_copy) {
        perror("strdup header in read_dense_weight_matrix");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    // Parse header to get column spot indices
    MKL_INT* col_spot_indices = (MKL_INT*)malloc((size_t)n_spots * sizeof(MKL_INT));
    if (!col_spot_indices) {
        perror("malloc col_spot_indices");
        free(header_copy);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    char* token = strtok(header_copy, "\t"); // Skip first field (row label placeholder)
    if (!token) {
        fprintf(stderr, "Error: Empty header line in dense weight matrix\n");
        free(header_copy);
        free(col_spot_indices);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    MKL_INT n_cols_found = 0;
    token = strtok(NULL, "\t");
    while (token && n_cols_found < n_spots) {
        char* trimmed = trim_whitespace_inplace(token);
        MKL_INT spot_idx = spot_name_ht_find(spot_map, trimmed);
        if (spot_idx >= 0) {
            col_spot_indices[n_cols_found++] = spot_idx;
        } else {
            fprintf(stderr, "Warning (Dense Read): Column spot '%s' in weight matrix header not found in expression data spot list. Skipping column.\n", trimmed);
        }
        token = strtok(NULL, "\t");
    }
    free(header_copy);
    
    if (n_cols_found == 0 && n_spots > 0) {
        fprintf(stderr, "Error: No matching column spots found in dense weight matrix header and expression data.\n");
        free(col_spot_indices);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    printf("  Found %lld matching column spots in header\n", (long long)n_cols_found);
    
    // Count potential NNZ and read data
    typedef struct { MKL_INT row, col; double value; } triplet_t;
    
    // Estimate NNZ capacity
    MKL_INT estimated_max_nnz = n_spots * n_cols_found; // Upper bound for dense
    if (estimated_max_nnz <= 0) estimated_max_nnz = 1000000; // Fallback
    
    triplet_t* triplets = (triplet_t*)malloc((size_t)estimated_max_nnz * sizeof(triplet_t));
    if (!triplets) {
        perror("Failed to allocate triplets for dense weight matrix");
        free(col_spot_indices);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }
    
    MKL_INT nnz_count = 0;
    int data_rows_processed = 0;

    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        // Trim line
        while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
            line[--line_len] = '\0';
        }
        if (line_len == 0) continue; // Skip empty lines
        
        char* line_copy_data = strdup(line);
        if (!line_copy_data) {
            perror("strdup data line in dense weight matrix");
            break;
        }
        
        char* row_label_token = strtok(line_copy_data, "\t");
        if (!row_label_token) {
            free(line_copy_data);
            continue;
        }

        char* row_spot_name = trim_whitespace_inplace(row_label_token);
        MKL_INT row_idx = spot_name_ht_find(spot_map, row_spot_name);

        if (row_idx >= 0) {
            // Read values for this row
            for (MKL_INT col_file_idx = 0; col_file_idx < n_cols_found; ++col_file_idx) {
                char* val_token = strtok(NULL, "\t");
                if (!val_token) {
                    fprintf(stderr, "Warning: Row '%s' has fewer columns than expected\n", row_spot_name);
                    break;
                }
                
                char* endptr;
                double weight = strtod(val_token, &endptr);
                if (endptr != val_token && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                    if (nnz_count < estimated_max_nnz) {
                        triplets[nnz_count].row = row_idx;
                        triplets[nnz_count].col = col_spot_indices[col_file_idx];
                        triplets[nnz_count].value = weight;
                        nnz_count++;
                    } else {
                        fprintf(stderr, "Warning: Exceeded estimated_max_nnz in dense read. Some data might be lost.\n");
                        break;
                    }
                }
            }
            data_rows_processed++;
        } else {
            if (data_rows_processed < 10) { // Limit warnings
                fprintf(stderr, "Warning: Row spot '%s' not found in expression data\n", row_spot_name);
            }
        }
        free(line_copy_data);
    }
    
    fclose(fp);
    if (line) free(line);
    free(col_spot_indices);
    
    printf("Dense weight matrix read: %lld non-zero entries from %d data rows\n", 
           (long long)nnz_count, data_rows_processed);
    
    // Convert to sparse CSR format
    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!W) {
        perror("Failed to allocate SparseMatrix for dense weight matrix");
        free(triplets);
        return NULL;
    }
    
    W->nrows = n_spots; 
    W->ncols = n_spots; 
    W->nnz = nnz_count;
    W->rownames = NULL;
    W->colnames = NULL;
    
    W->row_ptr = (MKL_INT*)mkl_calloc((size_t)n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc((size_t)nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc((size_t)nnz_count * sizeof(double), 64);
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
        printf("  Converting dense matrix to CSR format...\n");
        
        // Count entries per row
        for (MKL_INT k = 0; k < nnz_count; k++) {
            W->row_ptr[triplets[k].row + 1]++;
        }
        
        // Convert counts to cumulative sums
        for (MKL_INT i = 0; i < n_spots; i++) {
            W->row_ptr[i + 1] += W->row_ptr[i];
        }
        
        // Fill CSR arrays
        MKL_INT* csr_fill_counters = (MKL_INT*)calloc((size_t)n_spots, sizeof(MKL_INT));
        if (!csr_fill_counters) {
            perror("Failed to allocate CSR fill counters for dense weight matrix");
            free_sparse_matrix(W);
            free(triplets);
            return NULL;
        }
        
        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT r = triplets[k].row;
            MKL_INT insert_at = W->row_ptr[r] + csr_fill_counters[r];
            if (insert_at < nnz_count) { // Bounds check
                W->col_ind[insert_at] = triplets[k].col;
                W->values[insert_at] = triplets[k].value;
                csr_fill_counters[r]++;
            } else {
                fprintf(stderr, "Error: CSR insertion index out of bounds in dense matrix conversion\n");
                break;
            }
        }
        free(csr_fill_counters);
        printf("  CSR conversion complete\n");
    }
    
    free(triplets);
    
    printf("Dense weight matrix converted to CSR format: %lldx%lld with %lld NNZ\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz);
    
    return W;
}


/* Read sparse weight matrix in COO format */
SparseMatrix* read_sparse_weight_matrix_coo(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots) {
    // This function will be very similar to read_sparse_weight_matrix_tsv,
    // using two passes and the spot_map for lookups.
    // The main difference is parsing "RxC" style coordinates if used,
    // OR direct spot name lookup using spot_map.
    // For simplicity and given the user's current file is TSV, this is sketched:
    if (!filename || !spot_map || n_spots <= 0) return NULL;
    
    FILE* fp = fopen(filename, "r");
    if (!fp) return NULL;

    char *line = NULL; size_t line_buf_size = 0; ssize_t line_len;
    printf("Reading sparse COO weight matrix from '%s'...\n", filename);

    long data_start_offset = get_data_start_offset(fp, &line, &line_buf_size, filename);
    if (data_start_offset < 0) { fclose(fp); if(line) free(line); return NULL; }

    fseek(fp, data_start_offset, SEEK_SET);
    MKL_INT true_nnz = 0;
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        char* p = line; while(isspace((unsigned char)*p)) p++;
        if(*p != '\0') true_nnz++;
    }

    typedef struct { MKL_INT row, col; double value; } triplet_t;
    triplet_t* triplets = NULL;
    if (true_nnz > 0) {
        triplets = (triplet_t*)malloc((size_t)true_nnz * sizeof(triplet_t));
        if (!triplets) { /* error */ fclose(fp); if(line) free(line); return NULL; }
    } else {
        // Handle case of no data lines after header, or empty file
        fclose(fp); if(line) free(line);
        SparseMatrix* W_empty = (SparseMatrix*)calloc(1, sizeof(SparseMatrix));
        W_empty->nrows = n_spots; W_empty->ncols = n_spots; W_empty->nnz = 0;
        W_empty->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
        return W_empty; // Return empty matrix
    }

    fseek(fp, data_start_offset, SEEK_SET);
    MKL_INT nnz_count = 0;
    regex_t coord_regex;
    int regex_compiled = (regcomp(&coord_regex, "^([0-9]+)[xX]([0-9]+)$", REG_EXTENDED) == 0);


    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && nnz_count < true_nnz) {
        // ... (current line trimming) ...
        char* row_copy = strdup(line);
        if (!row_copy) { /* error */ break; }

        char* row_coord_str = strtok(row_copy, "\t");
        char* col_coord_str = strtok(NULL, "\t");
        char* weight_str = strtok(NULL, "\t");

        if (row_coord_str && col_coord_str && weight_str) {
            MKL_INT row_idx = spot_name_ht_find(spot_map, trim_whitespace_inplace(row_coord_str));
            MKL_INT col_idx = spot_name_ht_find(spot_map, trim_whitespace_inplace(col_coord_str));
            // The original regex logic for RxC was to parse RxC strings.
            // If spot_map keys are RxC, then direct lookup is fine.
            // If spot_map keys are other IDs but file uses RxC, then a mapping RxC -> canonical ID -> index is needed,
            // which is more complex than current scope. Assuming file uses names compatible with spot_map.

            if (row_idx >= 0 && col_idx >= 0) {
                char* endptr;
                double weight = strtod(weight_str, &endptr);
                if (endptr != weight_str && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                    triplets[nnz_count].row = row_idx;
                    triplets[nnz_count].col = col_idx;
                    triplets[nnz_count].value = weight;
                    nnz_count++;
                }
            }
        }
        free(row_copy);
    }

    if (regex_compiled) regfree(&coord_regex);
    fclose(fp);
    if (line) free(line);

    // ... (CSR conversion based on nnz_count, then free triplets) ...
    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    // ... (Initialize W, W->nnz = nnz_count, allocate CSR arrays) ...
    // (Same CSR conversion logic as in read_sparse_weight_matrix_tsv)
    // Free triplets at the end.
    if (!W) { /* ... */ if (triplets) free(triplets); return NULL; }
    W->nrows = n_spots; W->ncols = n_spots; W->nnz = nnz_count;
    W->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc((size_t)nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc((size_t)nnz_count * sizeof(double), 64);
    } else {
        W->col_ind = NULL; W->values = NULL;
    }
    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) { /* ... */ if (triplets) free(triplets); free_sparse_matrix(W); return NULL;}

    if (nnz_count > 0) {
        for (MKL_INT k = 0; k < nnz_count; k++) W->row_ptr[triplets[k].row + 1]++;
        for (MKL_INT i = 0; i < n_spots; i++) W->row_ptr[i+1] += W->row_ptr[i];
        
        MKL_INT* csr_fill_counters = (MKL_INT*)calloc(n_spots, sizeof(MKL_INT));
        if(!csr_fill_counters) { /* ... */ if (triplets) free(triplets); free_sparse_matrix(W); return NULL;}
        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT r = triplets[k].row;
            MKL_INT insert_at = W->row_ptr[r] + csr_fill_counters[r];
            W->col_ind[insert_at] = triplets[k].col;
            W->values[insert_at] = triplets[k].value;
            csr_fill_counters[r]++;
        }
        free(csr_fill_counters);
    }
    if (triplets) free(triplets);
    return W;
}

/* Read sparse weight matrix in TSV format */
SparseMatrix* read_sparse_weight_matrix_tsv(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots) {
    if (!filename || !spot_map || n_spots < 0) { // n_spots can be 0 for empty expression matrix
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

    // Determine data start offset by checking for a header
    long data_start_offset = get_data_start_offset(fp, &line, &line_buf_size, filename);
    if (data_start_offset < 0) { // Error in get_data_start_offset
        fclose(fp); if(line) free(line);
        return NULL;
    }
    
    // Single-pass reading with dynamic allocation
    typedef struct { MKL_INT row, col; double value; } triplet_t;
    triplet_t* triplets = NULL;
    MKL_INT triplets_capacity = 0;
    MKL_INT nnz_count = 0; // Actual valid triplets found
    
    // Initial capacity estimate (be generous to avoid frequent reallocations)
    MKL_INT initial_capacity = 1000000; // Start with 1M entries
    if (n_spots > 0) {
        // Estimate ~20-50 connections per spot for typical spatial transcriptomics data
        size_t estimated_nnz;
        if (safe_multiply_size_t(n_spots, 35, &estimated_nnz) == 0 && estimated_nnz < 50000000) {
            initial_capacity = (MKL_INT)estimated_nnz;
        } else {
            initial_capacity = 50000000; // Cap at 50M to avoid excessive memory
        }
    }
    
    triplets_capacity = initial_capacity;
    triplets = (triplet_t*)malloc((size_t)triplets_capacity * sizeof(triplet_t));
    if (!triplets) {
        perror("read_sparse_weight_matrix_tsv: malloc initial triplets failed");
        fclose(fp); if(line) free(line); return NULL;
    }
    
    printf("  Starting with triplets capacity: %lld\n", (long long)triplets_capacity);
    
    // Single pass: Read and parse data lines
    if (fseek(fp, data_start_offset, SEEK_SET) != 0) {
        perror("read_sparse_weight_matrix_tsv: fseek for data reading failed");
        free(triplets); fclose(fp); if(line) free(line); return NULL;
    }
    
    int line_num = 0; // For error messages
    int resize_count = 0; // Track number of resizes

    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        line_num++;
        char* p_check_empty = line; // Check if line is not just whitespace
        while(isspace((unsigned char)*p_check_empty)) p_check_empty++;
        if(*p_check_empty == '\0') continue; // Skip empty or whitespace-only lines

        // Grow triplets array if needed
        if (nnz_count >= triplets_capacity) {
            MKL_INT new_capacity = triplets_capacity * 2;
            if (new_capacity < triplets_capacity) { // Overflow check
                fprintf(stderr, "Error: Triplets array size overflow at line %d\n", line_num);
                break;
            }
            
            triplet_t* new_triplets = (triplet_t*)realloc(triplets, (size_t)new_capacity * sizeof(triplet_t));
            if (!new_triplets) {
                perror("read_sparse_weight_matrix_tsv: realloc triplets failed");
                break;
            }
            triplets = new_triplets;
            triplets_capacity = new_capacity;
            resize_count++;
            
            // Print resize notification, but not too frequently
            if (resize_count <= 10 || resize_count % 5 == 0) {
                printf("  Resized triplets array to %lld entries (resize #%d)\n", 
                       (long long)new_capacity, resize_count);
            }
        }
        
        char* row_copy = strdup(line); // strdup the original line content from buffer
        if (!row_copy) {
            perror("read_sparse_weight_matrix_tsv: strdup data line failed");
            break; // Critical memory error
        }
        
        char* spot1_str = strtok(row_copy, "\t");
        char* spot2_str = strtok(NULL, "\t");
        char* weight_str = strtok(NULL, "\t");
        
        if (!spot1_str || !spot2_str || !weight_str) {
            if (line_num <= 10 || line_num % 100000 == 0) { // Limit warning frequency
                fprintf(stderr, "Warning: Line %d in '%s' has too few fields. Skipping.\n", line_num, filename);
            }
            free(row_copy);
            continue;
        }
        
        char* spot1_trimmed = trim_whitespace_inplace(spot1_str);
        char* spot2_trimmed = trim_whitespace_inplace(spot2_str);
        char* weight_trimmed = trim_whitespace_inplace(weight_str);  // CRITICAL FIX: Trim weight string
        
        MKL_INT row_idx = spot_name_ht_find(spot_map, spot1_trimmed);
        MKL_INT col_idx = spot_name_ht_find(spot_map, spot2_trimmed);
        
        if (row_idx >= 0 && col_idx >= 0) {
            char* endptr;
            errno = 0;  // Clear errno before parsing
            double weight = strtod(weight_trimmed, &endptr);
            
            // ENHANCED VALIDATION: More lenient about trailing whitespace
            if (errno == 0 && endptr != weight_trimmed && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                // Additional check: ensure no significant trailing characters
                while (*endptr && isspace((unsigned char)*endptr)) endptr++;
                
                if (*endptr == '\0') {  // Now check if we've consumed all significant characters
                    triplets[nnz_count].row = row_idx;
                    triplets[nnz_count].col = col_idx;
                    triplets[nnz_count].value = weight;
                    nnz_count++;
                } else {
                    if (line_num <= 10 || line_num % 100000 == 0) {
                        fprintf(stderr, "Warning: Line %d in '%s': Invalid weight '%s' (trailing chars). Skipping.\n", 
                                line_num, filename, weight_trimmed);
                    }
                }
            } else {
                if (line_num <= 10 || line_num % 100000 == 0) { // Limit warning frequency
                    fprintf(stderr, "Warning: Line %d in '%s': Invalid weight '%s' or value too small (%.2e <= %.2e). Skipping.\n", 
                            line_num, filename, weight_trimmed, fabs(weight), WEIGHT_THRESHOLD);
                }
            }
        } else {
            // Only show spot lookup warnings for first few lines to avoid spam
            if (line_num <= 10) {
                if (row_idx < 0) fprintf(stderr, "Warning: Line %d in '%s': Spot1 '%s' not found in expression data. Skipping entry.\n", 
                                        line_num, filename, spot1_trimmed);
                if (col_idx < 0) fprintf(stderr, "Warning: Line %d in '%s': Spot2 '%s' not found in expression data. Skipping entry.\n", 
                                        line_num, filename, spot2_trimmed);
            }
        }
        free(row_copy);
        
        // Progress reporting for large files
        if (line_num % 1000000 == 0) {
            printf("  Processed %d lines, found %lld valid entries\n", line_num, (long long)nnz_count);
        }
    }
    
    fclose(fp);
    if (line) free(line);
    
    printf("Sparse TSV weight matrix read: %lld valid non-zero entries found from %d data lines.\n", 
           (long long)nnz_count, line_num);
    
    if (resize_count > 0) {
        printf("  Array was resized %d times during reading\n", resize_count);
    }
    
    // Trim triplets array to actual size to save memory
    if (nnz_count < triplets_capacity && nnz_count > 0) {
        triplet_t* trimmed_triplets = (triplet_t*)realloc(triplets, (size_t)nnz_count * sizeof(triplet_t));
        if (trimmed_triplets) {
            triplets = trimmed_triplets;
            triplets_capacity = nnz_count;
            printf("  Trimmed triplets array to final size: %lld entries\n", (long long)nnz_count);
        }
        // If realloc fails, just continue with the larger array - not a critical error
    }
    
    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!W) {
        perror("read_sparse_weight_matrix_tsv: malloc SparseMatrix failed");
        if (triplets) free(triplets);
        return NULL;
    }
    
    W->nrows = n_spots;
    W->ncols = n_spots;
    W->nnz = nnz_count; // Use actual valid count
    W->rownames = NULL; W->colnames = NULL;
    
    W->row_ptr = (MKL_INT*)mkl_calloc((size_t)n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc((size_t)nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc((size_t)nnz_count * sizeof(double), 64);
    } else {
        W->col_ind = NULL; W->values = NULL;
    }
    
    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) {
        perror("read_sparse_weight_matrix_tsv: Failed to allocate CSR arrays");
        free_sparse_matrix(W); // Will free W->row_ptr if it was allocated
        if (triplets) free(triplets);
        return NULL;
    }
    
    if (nnz_count > 0) {
        printf("  Converting to CSR format...\n");
        
        // Count entries per row
        for (MKL_INT k = 0; k < nnz_count; k++) {
            W->row_ptr[triplets[k].row + 1]++;
        }
        
        // Convert counts to cumulative sums (CSR row_ptr format)
        for (MKL_INT i = 0; i < n_spots; i++) {
            W->row_ptr[i + 1] += W->row_ptr[i];
        }
        
        // Fill CSR arrays
        MKL_INT* csr_fill_counters = (MKL_INT*)calloc(n_spots, sizeof(MKL_INT));
        if (!csr_fill_counters) {
            perror("read_sparse_weight_matrix_tsv: calloc csr_fill_counters failed");
            free_sparse_matrix(W); if (triplets) free(triplets); return NULL;
        }
        
        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT r = triplets[k].row;
            MKL_INT insert_at = W->row_ptr[r] + csr_fill_counters[r];
            if (insert_at < nnz_count) { // Bounds check
                 W->col_ind[insert_at] = triplets[k].col;
                 W->values[insert_at] = triplets[k].value;
                 csr_fill_counters[r]++;
            } else {
                fprintf(stderr, "Error: CSR insertion index out of bounds. This should not happen.\n");
                // This indicates a logic error in CSR construction
            }
        }
        free(csr_fill_counters);
        printf("  CSR conversion complete\n");
    }
    
    if (triplets) free(triplets);
    
    printf("Sparse TSV weight matrix converted to CSR format: %lldx%lld with %lld NNZ\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz);
    
    return W;
}

/* Main custom weight matrix reading function */
SparseMatrix* read_custom_weight_matrix(const char* filename, int format, 
                                       char** spot_names_from_expr, MKL_INT n_spots) {
    if (!filename || !spot_names_from_expr || n_spots < 0) {
        fprintf(stderr, "Error: Invalid parameters in read_custom_weight_matrix\n");
        return NULL;
    }
    
    int actual_format = format;
    if (format == WEIGHT_FORMAT_AUTO) {
        actual_format = detect_weight_matrix_format(filename);
        if (actual_format == WEIGHT_FORMAT_AUTO) {
            fprintf(stderr, "Error: Could not auto-detect weight matrix format for '%s'\n", filename);
            return NULL;
        }
    }
    
    SpotNameHashTable* spot_map = spot_name_ht_create(n_spots > 0 ? (size_t)n_spots : 16);
    if (!spot_map) {
        fprintf(stderr, "Error: Failed to create spot name hash table.\n");
        return NULL;
    }

    int inserted_count = 0;
    for (MKL_INT i = 0; i < n_spots; i++) {
        if (spot_names_from_expr[i] && strlen(spot_names_from_expr[i]) > 0) {
            if (spot_name_ht_insert(spot_map, spot_names_from_expr[i], i) == 0) {
                inserted_count++;
            } else {
                fprintf(stderr, "Error: Failed to insert spot '%s' into hash table.\n", spot_names_from_expr[i]);
                spot_name_ht_free(spot_map);
                return NULL;
            }
        }
    }
    if (inserted_count == 0 && n_spots > 0) {
        fprintf(stderr, "Warning: No spot names from expression data were added to hash map for custom weights.\n");
        // Continue if n_spots is 0, otherwise this might be an issue.
    }
     if (n_spots > 0 && spot_map->count != (size_t)inserted_count) {
         fprintf(stderr, "Warning: Hash map count (%zu) differs from inserted count (%d).\n", spot_map->count, inserted_count);
     }


    SparseMatrix* W = NULL;
    switch (actual_format) {
        case WEIGHT_FORMAT_DENSE:
            printf("Reading custom weight matrix in DENSE format...\n");
            W = read_dense_weight_matrix(filename, spot_map, n_spots);
            break;
        case WEIGHT_FORMAT_SPARSE_COO:
            printf("Reading custom weight matrix in SPARSE_COO format...\n");
            W = read_sparse_weight_matrix_coo(filename, spot_map, n_spots);
            break;
        case WEIGHT_FORMAT_SPARSE_TSV:
            printf("Reading custom weight matrix in SPARSE_TSV format...\n");
            W = read_sparse_weight_matrix_tsv(filename, spot_map, n_spots);
            break;
        default:
            fprintf(stderr, "Error: Unsupported weight matrix format: %d\n", actual_format);
            spot_name_ht_free(spot_map);
            return NULL;
    }
    
    spot_name_ht_free(spot_map); // Free the hash table after use
    
    if (W) {
        int validation_result = validate_weight_matrix(W, spot_names_from_expr, n_spots);
        if (validation_result != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error: Weight matrix validation failed\n");
            free_sparse_matrix(W);
            return NULL;
        }
        printf("Custom weight matrix successfully loaded and validated.\n");
    } else {
         fprintf(stderr, "Failed to load custom weight matrix in format %d.\n", actual_format);
    }
    
    return W;
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
 * BATCH CALCULATION FUNCTION
 * =============================== */

/* Calculate Moran's I using batch interface with raw arrays */
double* calculate_morans_i_batch(const double* X_data, long long n_genes_ll, long long n_spots_ll,
                                const double* W_values, const long long* W_row_ptr_ll, 
                                const long long* W_col_ind_ll, long long W_nnz_ll, int paired_genes) {
    
    if (!X_data || (W_nnz_ll > 0 && (!W_values || !W_row_ptr_ll || !W_col_ind_ll)) || (W_nnz_ll == 0 && !W_row_ptr_ll) ) {
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
    MKL_INT* W_row_ptr_mkl = (MKL_INT*)mkl_malloc(((size_t)n_spots + 1) * sizeof(MKL_INT), 64);
    MKL_INT* W_col_ind_mkl = NULL;
    if (W_nnz > 0) {
        W_col_ind_mkl = (MKL_INT*)mkl_malloc((size_t)W_nnz * sizeof(MKL_INT), 64);
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
    if (W_nnz > 0) { // Only access W_col_ind_ll if W_nnz > 0
        for (MKL_INT i = 0; i < W_nnz; i++) {
            W_col_ind_mkl[i] = (MKL_INT)W_col_ind_ll[i];
        }
    }
    
    // Calculate S0 for scaling
    double S0 = 0.0;
    if (W_nnz > 0) { // Only access W_values if W_nnz > 0
        #pragma omp parallel for reduction(+:S0)
        for (MKL_INT i = 0; i < W_nnz; i++) {
            S0 += W_values[i];
        }
    }
    
    if (fabs(S0) < DBL_EPSILON && W_nnz > 0) { // Check S0 only if W is not empty
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero in batch calculation\n");
        // Depending on policy, might return NULL or proceed with NaN/Inf results
    }
    if (W_nnz == 0) {
        printf("Warning: Weight matrix W is empty (NNZ=0) in batch calculation.\n");
        // S0 will be 0. Results will be 0 or NaN.
    }

    
    double inv_S0 = (fabs(S0) > DBL_EPSILON) ? (1.0 / S0) : 0.0; // Handle S0=0 to avoid Inf
    if (inv_S0 == 0.0 && fabs(S0) > DBL_EPSILON) { // S0 is very large, inv_S0 underflowed to 0
        // This case is unlikely to be an issue unless S0 is astronomically large
    } else if (inv_S0 == 0.0 && fabs(S0) < DBL_EPSILON) {
        printf("Note: S0 is zero or near-zero. Results will be scaled by 0 or be NaN/Inf.\n");
    }


    // Create MKL sparse matrix handle
    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl, SPARSE_INDEX_BASE_ZERO,
                                                     n_spots, n_spots, W_row_ptr_mkl,
                                                     W_row_ptr_mkl + 1, W_col_ind_mkl, 
                                                     (double*)W_values); // W_values might be NULL if W_nnz=0
    
    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (batch W)");
        mkl_free(W_row_ptr_mkl);
        if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
        return NULL;
    }
    
    if (W_nnz > 0) { // Optimize only if there are non-zeros
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_optimize (batch W)");
            // Non-fatal, can proceed without optimization
        }
    }
    
    double* results = NULL;
    
    if (paired_genes) {
        // Pairwise calculation: result is n_genes x n_genes matrix
        size_t result_size_elements;
        size_t result_size_bytes;

        if (safe_multiply_size_t(n_genes, n_genes, &result_size_elements) != 0 ||
            safe_multiply_size_t(result_size_elements, sizeof(double), &result_size_bytes) != 0) {
            fprintf(stderr, "Error: Result matrix dimensions too large for batch pairwise calc.\n");
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl); if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        results = (double*)mkl_malloc(result_size_bytes, 64);

        if (!results) {
            perror("Failed to allocate results for batch pairwise calculation");
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl); if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        
        size_t temp_wx_elements;
        size_t temp_wx_bytes;
        if (safe_multiply_size_t(n_spots, n_genes, &temp_wx_elements) != 0 ||
            safe_multiply_size_t(temp_wx_elements, sizeof(double), &temp_wx_bytes) != 0) {
            fprintf(stderr, "Error: Temp_WX matrix dimensions too large for batch pairwise calc.\n");
            mkl_free(results);
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl); if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        double* temp_WX = (double*)mkl_malloc(temp_wx_bytes, 64);

        if (!temp_WX) {
            perror("Failed to allocate temp_WX for batch calculation");
            mkl_free(results);
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl); if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descr,
                                SPARSE_LAYOUT_ROW_MAJOR, (double*)X_data, n_genes, n_genes, // X_data is const, MKL takes non-const. This is typical.
                                0.0, temp_WX, n_genes);
        
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_d_mm (batch W * X)");
            mkl_free(temp_WX);
            mkl_free(results);
            mkl_sparse_destroy(W_mkl);
            mkl_free(W_row_ptr_mkl); if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
            return NULL;
        }
        
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n_genes, n_genes, n_spots, inv_S0,
                    X_data, n_genes, temp_WX, n_genes,
                    0.0, results, n_genes);
        
        mkl_free(temp_WX);
        
    } else { // Single-gene calculation
        size_t result_size_bytes;
        if (safe_multiply_size_t(n_genes, sizeof(double), &result_size_bytes) != 0) {
             fprintf(stderr, "Error: Result vector too large for batch single-gene calc.\n");
             mkl_sparse_destroy(W_mkl);
             mkl_free(W_row_ptr_mkl); if (W_col_ind_mkl) mkl_free(W_col_ind_mkl);
             return NULL;
        }
        results = (double*)mkl_malloc(result_size_bytes, 64);
        if (!results) { /* ... */ }

        size_t buffer_bytes;
        if (safe_multiply_size_t(n_spots, sizeof(double), &buffer_bytes) != 0) {
            /* ... */
        }
        
        double* gene_buffer = (double*)mkl_malloc(buffer_bytes, 64);
        double* Wz_buffer = (double*)mkl_malloc(buffer_bytes, 64);
        if (!gene_buffer || !Wz_buffer) { /* ... */ }
        
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        for (MKL_INT g = 0; g < n_genes; g++) {
            for (MKL_INT s = 0; s < n_spots; s++) {
                gene_buffer[s] = X_data[s * n_genes + g];
            }
            
            status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descr,
                                    gene_buffer, 0.0, Wz_buffer);
            
            if (status != SPARSE_STATUS_SUCCESS) {
                print_mkl_status(status, "mkl_sparse_d_mv (batch single-gene)");
                results[g] = NAN;
                continue;
            }
            
            double dot_product = cblas_ddot(n_spots, gene_buffer, 1, Wz_buffer, 1);
            if (fabs(S0) < DBL_EPSILON) { // Explicit check for S0 for single gene case if inv_S0 might be 0.0
                 results[g] = (dot_product == 0.0) ? 0.0 : (dot_product > 0 ? INFINITY : -INFINITY) ; // Or NAN
                 if(dot_product == 0.0 && S0 == 0.0) results[g] = NAN; // 0/0 is NaN
            } else {
                 results[g] = dot_product * inv_S0;
            }
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