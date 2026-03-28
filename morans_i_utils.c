/* morans_i_utils.c - Utility and helper functions for Moran's I implementation
 *
 * Version: 1.3.0 (Added Residual Moran's I functionality)
 *
 * This module contains hash table implementations, input validation helpers,
 * configuration initialization, and other utility functions used across
 * the Moran's I spatial autocorrelation library.
 */

#include <unistd.h>
#include "morans_i_internal.h"

/* Library version information */
const char* morans_i_mkl_version(void) {
    return "1.3.0";
}

/* ===============================
 * HASH TABLE IMPLEMENTATIONS
 * =============================== */

/* Hash Table Helper Function DEFINITIONS */
unsigned long spot_name_hash(const char* str) {
    unsigned long hash = 5381;
    int c;
    while ((c = (unsigned char)*str++)) // cast to unsigned char
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    return hash;
}

int is_prime(size_t n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (size_t i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return 0;
    return 1;
}

/* Helper to find next prime (simple version for hash table sizing) */
size_t get_next_prime(size_t n) {
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

SpotNameHashTable* spot_name_ht_create(size_t num_spots_hint) {
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

int spot_name_ht_insert(SpotNameHashTable* ht, const char* name, MKL_INT index) {
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

MKL_INT spot_name_ht_find(const SpotNameHashTable* ht, const char* name) {
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

void spot_name_ht_free(SpotNameHashTable* ht) {
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

/* ===============================
 * UTILITY AND HELPER FUNCTIONS
 * =============================== */

/* Safe multiplication to prevent overflow */
int safe_multiply_size_t(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > SIZE_MAX / a) {
        return -1; // Overflow
    }
    *result = a * b;
    return 0;
}

/* Input validation helper */
int validate_matrix_dimensions(MKL_INT nrows, MKL_INT ncols, const char* context) {
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
char* trim_whitespace_inplace(char* str) {
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
 * PERMUTATION RESULT HELPERS
 * =============================== */

PermutationResults* alloc_permutation_results(MKL_INT n_genes, const char** gene_names,
                                              const PermutationParams* params) {
    PermutationResults* results = (PermutationResults*)calloc(1, sizeof(PermutationResults));
    if (!results) {
        perror("Failed to allocate PermutationResults structure");
        return NULL;
    }

    size_t matrix_elements = (size_t)n_genes * n_genes;
    size_t matrix_bytes;
    if (safe_multiply_size_t(matrix_elements, sizeof(double), &matrix_bytes) != 0) {
        fprintf(stderr, "Error: Matrix size too large for permutation results\n");
        free(results);
        return NULL;
    }

    results->mean_perm = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    results->var_perm = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (params->z_score_output)
        results->z_scores = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (params->p_value_output)
        results->p_values = (DenseMatrix*)malloc(sizeof(DenseMatrix));

    if (!results->mean_perm || !results->var_perm ||
        (params->z_score_output && !results->z_scores) ||
        (params->p_value_output && !results->p_values)) {
        perror("Failed to allocate result matrix structures");
        free_permutation_results(results);
        return NULL;
    }

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
            return NULL;
        }

        for (MKL_INT i = 0; i < n_genes; i++) {
            const char* name = (gene_names && gene_names[i]) ? gene_names[i] : "UNKNOWN_GENE";
            matrices[m]->rownames[i] = strdup(name);
            matrices[m]->colnames[i] = strdup(name);
            if (!matrices[m]->rownames[i] || !matrices[m]->colnames[i]) {
                perror("Failed to duplicate gene names for permutation results");
                free_permutation_results(results);
                return NULL;
            }
        }
    }

    return results;
}

void finalize_permutation_statistics(PermutationResults* results,
                                     const DenseMatrix* observed_results,
                                     const PermutationParams* params,
                                     MKL_INT n_genes, int n_perm) {
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
}
