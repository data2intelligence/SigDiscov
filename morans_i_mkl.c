/* morans_i_mkl.c - Optimized MKL-based Moran's I implementation
 *
 * Version: 1.1.1 (Updated for permutation performance)
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
#include "morans_i_mkl.h"

/* Define M_PI if not defined by math.h (e.g., on some systems or with strict C standards) */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Library version information */
const char* morans_i_mkl_version(void) {
    return "1.1.1"; 
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
    // Permutation defaults
    config.run_permutations = 0; // Default to off unless specified
    config.num_permutations = DEFAULT_NUM_PERMUTATIONS;
    config.perm_seed = (unsigned int)time(NULL); // Default seed
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


/* Z-Normalize function (Gene-wise: input is Genes x Spots, output is Genes x Spots) */
DenseMatrix* z_normalize(const DenseMatrix* data_matrix) { // Expects Genes x Spots
    if (!data_matrix || !data_matrix->values) {
        fprintf(stderr, "Error: Invalid data matrix provided to z_normalize\n");
        return NULL;
    }

    MKL_INT n_genes = data_matrix->nrows; // Number of genes
    MKL_INT n_spots = data_matrix->ncols; // Number of spots

    printf("Performing Z-normalization on %lld genes across %lld spots...\n",
           (long long)n_genes, (long long)n_spots);

    DenseMatrix* normalized = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!normalized) {
        perror("Failed to allocate DenseMatrix structure for normalized data");
        return NULL;
    }
    normalized->nrows = n_genes;
    normalized->ncols = n_spots;
    normalized->rownames = (char**)malloc((size_t)n_genes * sizeof(char*)); // Gene names
    normalized->colnames = (char**)malloc((size_t)n_spots * sizeof(char*)); // Spot names
    normalized->values = (double*)mkl_malloc((size_t)n_genes * n_spots * sizeof(double), 64);


    if (!normalized->rownames || !normalized->colnames || !normalized->values) {
        perror("Failed to allocate memory for normalized matrix data");
        if (normalized->values) mkl_free(normalized->values);
        if (normalized->rownames) { // Free individual names if allocated
            for(MKL_INT i=0; i<n_genes && normalized->rownames[i] != NULL; ++i) free(normalized->rownames[i]);
            free(normalized->rownames);
        }
        if (normalized->colnames) { // Free individual names if allocated
             for(MKL_INT i=0; i<n_spots && normalized->colnames[i] != NULL; ++i) free(normalized->colnames[i]);
            free(normalized->colnames);
        }
        free(normalized);
        return NULL;
    }

    /* Copy row (gene) and column (spot) names */
    for (MKL_INT i = 0; i < n_genes; i++) {
        if (data_matrix->rownames && data_matrix->rownames[i]) {
            normalized->rownames[i] = strdup(data_matrix->rownames[i]);
            if (!normalized->rownames[i]) {
                perror("Failed to duplicate row name (gene)");
                free_dense_matrix(normalized); 
                return NULL;
            }
        } else {
            normalized->rownames[i] = NULL; // Or a default name
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
        } else {
            normalized->colnames[j] = NULL; // Or a default name
        }
    }

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
        #pragma omp flush(global_alloc_error)


        if (!global_alloc_error) {
            #pragma omp for schedule(dynamic)
            for (MKL_INT i = 0; i < n_genes; i++) { 
                #pragma omp flush(global_alloc_error)
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
        fprintf(stderr, "Critical error during Z-normalization due to memory allocation failure in threads. Result may be incomplete.\n");
        free_dense_matrix(normalized);
        return NULL;
    }

    printf("Z-normalization complete.\n");
    return normalized;
}


/* Build spatial weight matrix W (Sparse CSR) */
SparseMatrix* build_spatial_weight_matrix(const MKL_INT* spot_row_valid, const MKL_INT* spot_col_valid,
                                         MKL_INT n_spots_valid, const DenseMatrix* distance_matrix,
                                         MKL_INT max_radius) {

    if (!spot_row_valid || !spot_col_valid || !distance_matrix || !distance_matrix->values) {
        fprintf(stderr, "Error: Invalid parameters provided to build_spatial_weight_matrix\n");
        return NULL;
    }
    if (n_spots_valid == 0) {
        printf("Warning: n_spots_valid is 0 in build_spatial_weight_matrix. Returning empty W.\n");
        SparseMatrix* W_empty = (SparseMatrix*)calloc(1, sizeof(SparseMatrix));
        if(!W_empty) { perror("calloc for empty W"); return NULL; }
        W_empty->nrows = 0; W_empty->ncols = 0; W_empty->nnz = 0;
        W_empty->row_ptr = (MKL_INT*)mkl_calloc(1, sizeof(MKL_INT), 64); // MKL CSR needs row_ptr[0]=0
        W_empty->col_ind = NULL; // No non-zeros
        W_empty->values = NULL;  // No non-zeros
        if (!W_empty->row_ptr) { perror("mkl_calloc for empty W->row_ptr"); free(W_empty); return NULL; }
        return W_empty;
    }


    printf("Building sparse spatial weight matrix W (%lld x %lld)...\n",
           (long long)n_spots_valid, (long long)n_spots_valid);

    MKL_INT estimated_neighbors_per_spot = (MKL_INT)(M_PI * max_radius * max_radius * 1.5);
    if (estimated_neighbors_per_spot <= 0) estimated_neighbors_per_spot = 27; // A reasonable minimum guess
    MKL_INT initial_capacity = n_spots_valid * estimated_neighbors_per_spot;
    if (initial_capacity > n_spots_valid * n_spots_valid) { // Cap at dense size
        initial_capacity = n_spots_valid * n_spots_valid;
    }
    if (initial_capacity <= 0) initial_capacity = (n_spots_valid > 0) ? n_spots_valid : 1; // Ensure positive if n_spots_valid > 0


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

    int critical_error_flag = 0;

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
                critical_error_flag = 1; // Signal global error
            }
            thread_alloc_error = 1; // Signal error for this thread
        }
        #pragma omp flush(critical_error_flag)


        if (!thread_alloc_error && !critical_error_flag) {
            #pragma omp for schedule(dynamic, 128)
            for (MKL_INT i = 0; i < n_spots_valid; i++) {
                #pragma omp flush(critical_error_flag) // Check before starting expensive work
                if (critical_error_flag || thread_alloc_error) continue;

                for (MKL_INT j = 0; j < n_spots_valid; j++) {
                    MKL_INT row_shift_abs = llabs(spot_row_valid[i] - spot_row_valid[j]);
                    MKL_INT col_shift_abs = llabs(spot_col_valid[i] - spot_col_valid[j]);

                    if (row_shift_abs < distance_matrix->nrows && col_shift_abs < distance_matrix->ncols) {
                        double weight = distance_matrix->values[row_shift_abs * distance_matrix->ncols + col_shift_abs];

                        if (fabs(weight) > WEIGHT_THRESHOLD) {
                            if (local_nnz_tl >= local_capacity) {
                                local_capacity = (MKL_INT)(local_capacity * 1.5) + 1; // Grow by 1.5x + 1
                                MKL_INT* temp_li_new = (MKL_INT*)realloc(local_I_tl, (size_t)local_capacity * sizeof(MKL_INT));
                                MKL_INT* temp_lj_new = (MKL_INT*)realloc(local_J_tl, (size_t)local_capacity * sizeof(MKL_INT));
                                double* temp_lv_new = (double*)realloc(local_V_tl, (size_t)local_capacity * sizeof(double));
                                if (!temp_li_new || !temp_lj_new || !temp_lv_new) {
                                    #pragma omp critical
                                    {
                                     fprintf(stderr, "Error: Thread %d failed to realloc thread-local COO buffers.\n", omp_get_thread_num());
                                     critical_error_flag = 1;
                                    }
                                    // Preserve old buffers if realloc failed for one, to allow freeing later
                                    local_I_tl = temp_li_new ? temp_li_new : local_I_tl;
                                    local_J_tl = temp_lj_new ? temp_lj_new : local_J_tl;
                                    local_V_tl = temp_lv_new ? temp_lv_new : local_V_tl;
                                    thread_alloc_error = 1; // Mark error for this thread
                                    goto end_inner_loop_build_w; 
                                }
                                local_I_tl = temp_li_new; local_J_tl = temp_lj_new; local_V_tl = temp_lv_new;
                            }
                            local_I_tl[local_nnz_tl] = i;
                            local_J_tl[local_nnz_tl] = j;
                            local_V_tl[local_nnz_tl] = weight;
                            local_nnz_tl++;
                        }
                    }
                }
                end_inner_loop_build_w:; 
                if (thread_alloc_error) continue; 
            }
        }


        #pragma omp flush(critical_error_flag)
        if (!critical_error_flag && !thread_alloc_error && local_nnz_tl > 0) {
            #pragma omp critical
            {
                if (!critical_error_flag) { // Double check inside critical section
                    if (nnz_count + local_nnz_tl > current_capacity) {
                        MKL_INT needed_capacity = nnz_count + local_nnz_tl;
                        MKL_INT new_global_capacity = current_capacity;
                        while(new_global_capacity < needed_capacity) {
                            new_global_capacity = (MKL_INT)(new_global_capacity * 1.5) + 1;
                        }
                         if (new_global_capacity > n_spots_valid * n_spots_valid) { // Cap at dense
                            new_global_capacity = n_spots_valid * n_spots_valid;
                        }
                        if (needed_capacity > new_global_capacity && n_spots_valid > 0) { // Check if still not enough
                             fprintf(stderr, "Error: Cannot resize global COO buffer large enough (%lld needed, max %lld).\n", (long long)needed_capacity, (long long)(n_spots_valid * n_spots_valid));
                             critical_error_flag = 1;
                        } else if (n_spots_valid > 0) { // Only print and realloc if spots > 0
                            printf("  Resizing global COO buffer from %lld to %lld\n", (long long)current_capacity, (long long)new_global_capacity);
                            MKL_INT* temp_gi_new = (MKL_INT*)realloc(temp_I, (size_t)new_global_capacity * sizeof(MKL_INT));
                            MKL_INT* temp_gj_new = (MKL_INT*)realloc(temp_J, (size_t)new_global_capacity * sizeof(MKL_INT));
                            double*  temp_gv_new = (double*)realloc(temp_V, (size_t)new_global_capacity * sizeof(double));

                            if (!temp_gi_new || !temp_gj_new || !temp_gv_new) {
                                fprintf(stderr, "Error: Failed to realloc global COO buffers. Data for this thread may be lost.\n");
                                critical_error_flag = 1;
                            } else {
                                temp_I = temp_gi_new; temp_J = temp_gj_new; temp_V = temp_gv_new;
                                current_capacity = new_global_capacity;
                            }
                        } else if (needed_capacity > new_global_capacity && n_spots_valid == 0) {
                            // If n_spots_valid is 0, needed_capacity should also be 0. This case is unlikely.
                             critical_error_flag = 1;
                        }
                    }

                    if (!critical_error_flag && (nnz_count + local_nnz_tl <= current_capacity)) {
                        memcpy(temp_I + nnz_count, local_I_tl, (size_t)local_nnz_tl * sizeof(MKL_INT));
                        memcpy(temp_J + nnz_count, local_J_tl, (size_t)local_nnz_tl * sizeof(MKL_INT));
                        memcpy(temp_V + nnz_count, local_V_tl, (size_t)local_nnz_tl * sizeof(double));
                        nnz_count += local_nnz_tl;
                    } else if (!critical_error_flag) { // If still not enough space after trying to realloc
                        fprintf(stderr, "Warning: Could not merge thread %d results due to insufficient space in global COO buffer. Data lost.\n", omp_get_thread_num());
                         critical_error_flag = 1; // Mark as error
                    }
                }
            }
        }

        if(local_I_tl) free(local_I_tl);
        if(local_J_tl) free(local_J_tl);
        if(local_V_tl) free(local_V_tl);
    } // End omp parallel

    if (critical_error_flag) {
        fprintf(stderr, "Error: A critical error occurred during parallel COO matrix construction. Aborting W matrix build.\n");
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

    // Allocate CSR arrays. If nnz is 0, allocate minimally for MKL compatibility (row_ptr needs n_rows+1)
    size_t nnz_alloc_size = (nnz_count > 0) ? (size_t)nnz_count : 1; // Min alloc 1 for val/col_ind if nnz=0
                                                                  // but mkl_sparse_d_create_csr allows NULL if nnz=0.
                                                                  // Let's be safe and alloc 1.
    W->row_ptr = (MKL_INT*)mkl_malloc(((size_t)n_spots_valid + 1) * sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc(nnz_alloc_size * sizeof(MKL_INT), 64);
        W->values  = (double*)mkl_malloc(nnz_alloc_size * sizeof(double), 64);
    } else { // nnz == 0
        W->col_ind = NULL; // MKL allows NULL for col_ind and values if nnz=0
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
        // Compute row_ptr
        for (MKL_INT i = 0; i <= n_spots_valid; ++i) W->row_ptr[i] = 0; // Initialize row_ptr
        for (MKL_INT k = 0; k < nnz_count; ++k) W->row_ptr[temp_I[k] + 1]++; // Count elements in each row
        for (MKL_INT i = 0; i < n_spots_valid; ++i) W->row_ptr[i + 1] += W->row_ptr[i]; // Cumulative sum

        // Fill col_ind and values
        // Need a temporary array to keep track of current insertion positions for each row
        MKL_INT* current_insertion_pos = (MKL_INT*)malloc(((size_t)n_spots_valid + 1) * sizeof(MKL_INT));
        if (!current_insertion_pos) {
             perror("Failed to allocate current_insertion_pos array");
             free_sparse_matrix(W); // Safe free
             free(temp_I); free(temp_J); free(temp_V);
             return NULL;
        }
        memcpy(current_insertion_pos, W->row_ptr, ((size_t)n_spots_valid) * sizeof(MKL_INT)); // Copy start indices (not n_spots_valid+1)
                                                                                            // row_ptr[0] to row_ptr[n_spots_valid-1] are start indices

        for (MKL_INT k = 0; k < nnz_count; ++k) {
            MKL_INT row = temp_I[k];
            MKL_INT index_in_csr = current_insertion_pos[row];
            W->col_ind[index_in_csr] = temp_J[k];
            W->values[index_in_csr] = temp_V[k];
            current_insertion_pos[row]++;
        }
        free(current_insertion_pos);
    } else { // nnz_count == 0
        for (MKL_INT i = 0; i <= n_spots_valid; ++i) W->row_ptr[i] = 0; // All row pointers are 0
    }


    free(temp_I); free(temp_J); free(temp_V);

    printf("Sparse weight matrix W built successfully (CSR format, %lld NNZ).\n", (long long)W->nnz);

    // Optional: Sort column indices within each row (MKL often requires this or does it internally)
    if (W->nnz > 0) {
        sparse_matrix_t W_mkl_tmp_handle;
        sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl_tmp_handle, SPARSE_INDEX_BASE_ZERO,
                                                        W->nrows, W->ncols, W->row_ptr,
                                                        W->row_ptr + 1, W->col_ind, W->values);
        if (status == SPARSE_STATUS_SUCCESS) {
            status = mkl_sparse_order(W_mkl_tmp_handle); // Sorts column indices within each row
            if (status != SPARSE_STATUS_SUCCESS) {
                print_mkl_status(status, "mkl_sparse_order (W)");
                // This is not necessarily fatal, but good to note.
            }
            mkl_sparse_destroy(W_mkl_tmp_handle);
            printf("  Column indices within rows ordered (if necessary).\n");
        } else {
            print_mkl_status(status, "mkl_sparse_d_create_csr (for ordering W)");
        }
    }
    return W;
}

/* Calculate pairwise Moran's I matrix: Result = (X_transpose * W * X) / S0 */
DenseMatrix* calculate_morans_i(const DenseMatrix* X, const SparseMatrix* W) {
    if (!X || !W || !X->values /* W->values can be NULL if W->nnz is 0 */) {
        fprintf(stderr, "Error: Invalid parameters provided to calculate_morans_i (X or W is NULL, or X->values is NULL).\n");
        return NULL;
    }
    if (W->nnz > 0 && !W->values) {
         fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in calculate_morans_i.\n");
        return NULL;
    }


    MKL_INT n_spots = X->nrows;
    MKL_INT n_genes = X->ncols;

    if (n_spots != W->nrows || n_spots != W->ncols) {
        fprintf(stderr, "Error: Dimension mismatch between X (%lld spots x %lld genes) and W (%lldx%lld)\n",
                (long long)n_spots, (long long)n_genes, (long long)W->nrows, (long long)W->ncols);
        return NULL;
    }
    if (n_genes == 0) {
        fprintf(stderr, "Warning: n_genes is 0 in calculate_morans_i. Returning empty result matrix.\n");
        DenseMatrix* res_empty = (DenseMatrix*)calloc(1, sizeof(DenseMatrix));
        if(!res_empty) {perror("calloc for empty moran's I result"); return NULL;}
        // Rownames/colnames can be allocated if X->colnames exists, even if n_genes=0 (0-length arrays)
        if (X->colnames) {
             res_empty->rownames = calloc(0, sizeof(char*)); 
             res_empty->colnames = calloc(0, sizeof(char*));
        }
        return res_empty;
    }


    printf("Calculating Moran's I for %lld genes using %lld spots (Matrix approach: X_T * W * X / S0)...\n",
           (long long)n_genes, (long long)n_spots);

    DenseMatrix* result = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!result) {
        perror("Failed alloc result struct for Moran's I");
        return NULL;
    }
    result->nrows = n_genes;
    result->ncols = n_genes;
    result->values = (double*)mkl_malloc((size_t)n_genes * n_genes * sizeof(double), 64);
    result->rownames = (char**)calloc(n_genes, sizeof(char*)); // calloc for safer freeing
    result->colnames = (char**)calloc(n_genes, sizeof(char*)); // calloc for safer freeing

    if (!result->values || !result->rownames || !result->colnames) {
        perror("Failed alloc result data for Moran's I");
        free_dense_matrix(result);
        return NULL;
    }

    for (MKL_INT i = 0; i < n_genes; i++) {
        if (X->colnames && X->colnames[i]) {
            result->rownames[i] = strdup(X->colnames[i]);
            result->colnames[i] = strdup(X->colnames[i]);
            if (!result->rownames[i] || !result->colnames[i]) {
                perror("Failed to duplicate gene names for Moran's I result");
                free_dense_matrix(result);
                return NULL;
            }
        } else { // Should not happen if n_genes > 0 and X is well-formed
            char default_name_buf[32];
            snprintf(default_name_buf, sizeof(default_name_buf), "Gene%lld", (long long)i);
            result->rownames[i] = strdup(default_name_buf);
            result->colnames[i] = strdup(default_name_buf);
        }
    }

    double S0 = calculate_weight_sum(W);
    printf("  Sum of weights S0: %.6f\n", S0);

    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Moran's I results will be NaN/Inf or 0.\n", S0);
        if (S0 == 0.0) { // Or very close, to avoid division by zero
             for(size_t i=0; i < (size_t)n_genes * n_genes; ++i) result->values[i] = NAN;
             return result; // Return matrix of NaNs
        }
    }
    double inv_S0 = 1.0 / S0; // Could be Inf if S0 is exactly 0
    printf("  Using 1/S0 = %.6e as scaling factor\n", inv_S0);


    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(
        &W_mkl, SPARSE_INDEX_BASE_ZERO, W->nrows, W->ncols,
        W->row_ptr, W->row_ptr + 1, W->col_ind, W->values); // W->values can be NULL if W->nnz=0

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W)");
        free_dense_matrix(result);
        return NULL;
    }

    if (W->nnz > 0) { // Optimize only if there are non-zeros
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_optimize (W)");
            // Non-fatal, can proceed
        }
    }


    printf("  Step 1: Calculating Temp_WX = W * X ...\n");
    double* Temp_WX_values = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
    if (!Temp_WX_values) {
        perror("Failed alloc Temp_WX_values");
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }

    struct matrix_descr descrW;
    descrW.type = SPARSE_MATRIX_TYPE_GENERAL; // Or specific if known (e.g. symmetric)
    // descrW.diag = SPARSE_DIAG_NON_UNIT; // if applicable
    
    double alpha_mm = 1.0, beta_mm = 0.0;

    status = mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha_mm,  // alpha
        W_mkl,     // Handle to sparse matrix A (W)
        descrW,    // Descriptor of matrix A
        SPARSE_LAYOUT_ROW_MAJOR, // Layout of dense matrix B (X)
        X->values, // Pointer to dense matrix B (X)
        n_genes,   // Number of columns in B (X), and columns in C (Temp_WX)
        n_genes,   // Leading dimension of B (X) (ldb) -> X is n_spots-by-n_genes, so ldb is n_genes
        beta_mm,   // beta
        Temp_WX_values, // Pointer to dense matrix C (Temp_WX)
        n_genes    // Leading dimension of C (Temp_WX) (ldc)
    );


    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_mm (W * X)");
        mkl_free(Temp_WX_values);
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }

    printf("  Step 2: Calculating Result = (X_T * Temp_WX) / S0 ...\n");
    // Result = inv_S0 * (X_transpose * Temp_WX) + 0.0 * Result
    // X is n_spots x n_genes (stored row-major)
    // X_transpose is n_genes x n_spots
    // Temp_WX is n_spots x n_genes (stored row-major)
    // Result is n_genes x n_genes
    cblas_dgemm(
        CblasRowMajor,    // Layout
        CblasTrans,       // opA: Transpose X
        CblasNoTrans,     // opB: No transpose for Temp_WX
        n_genes,          // M: rows of opA (X_T), rows of Result
        n_genes,          // N: cols of opB (Temp_WX), cols of Result
        n_spots,          // K: cols of opA (X_T), rows of opB (Temp_WX)
        inv_S0,           // alpha: scalar multiplier for (opA * opB)
        X->values,        // A: matrix X
        n_genes,          // lda: leading dimension of X (which is n_genes as it's N_spots x N_genes)
        Temp_WX_values,   // B: matrix Temp_WX
        n_genes,          // ldb: leading dimension of Temp_WX (n_genes)
        beta_mm,          // beta: scalar for C
        result->values,   // C: matrix Result
        n_genes           // ldc: leading dimension of Result (n_genes)
    );

    mkl_free(Temp_WX_values);
    mkl_sparse_destroy(W_mkl);

    printf("Moran's I matrix calculation complete and scaled by 1/S0.\n");
    return result;
}

/* Implementation of the batch calculation function for Cython integration */
double* calculate_morans_i_batch(const double* X_data, long long n_genes_ll, long long n_spots_ll,
                               const double* W_values, const long long* W_row_ptr_ll, const long long* W_col_ind_ll,
                               long long W_nnz_ll, int paired_genes) {
    if (X_data == NULL || W_row_ptr_ll == NULL ) { // W_values/W_col_ind can be NULL if W_nnz_ll is 0
        fprintf(stderr, "Error: NULL X_data or W_row_ptr_ll input to calculate_morans_i_batch\n");
        return NULL;
    }
    if (W_nnz_ll > 0 && (W_values == NULL || W_col_ind_ll == NULL)) {
        fprintf(stderr, "Error: W_nnz > 0 but W_values or W_col_ind_ll is NULL in calculate_morans_i_batch\n");
        return NULL;
    }


    MKL_INT n_genes = (MKL_INT)n_genes_ll;
    MKL_INT n_spots = (MKL_INT)n_spots_ll;
    MKL_INT W_nnz = (MKL_INT)W_nnz_ll;

    if (n_genes == 0 || n_spots == 0) {
        fprintf(stderr, "Warning: n_genes (%lld) or n_spots (%lld) is 0 in batch mode. Returning NULL or empty array.\n", (long long)n_genes, (long long)n_spots);
        size_t result_size = paired_genes ? (size_t)n_genes * n_genes : (size_t)n_genes;
        if (result_size == 0) result_size = 1; // Avoid calloc(0,...) issues on some systems for return.
        double* empty_res = (double*)calloc(result_size, sizeof(double)); // So Python gets an array
        return empty_res; 
    }


    double S0 = 0.0;
    for (MKL_INT i = 0; i < W_nnz; i++) S0 += W_values[i];

    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e) in calculate_morans_i_batch. Results will be 0 or NaN/Inf.\n", S0);
        double* error_result;
        size_t result_size = paired_genes ? (size_t)n_genes * n_genes : (size_t)n_genes;
        error_result = (double*)calloc(result_size, sizeof(double)); // Ensure memory is allocated
        if (!error_result && result_size > 0) {
            perror("calloc for error_result in batch"); return NULL; // Critical alloc fail
        }
        if (S0 == 0.0) { // Fill with NaN if S0 is exactly zero
            for(size_t i = 0; i < result_size; ++i) error_result[i] = NAN;
        } // Otherwise, it's zero-filled by calloc, which is fine for near-zero S0 (results will be scaled to large numbers or zero)
        return error_result;
    }
    double inv_S0 = 1.0 / S0;

    double* result_values;
    size_t result_size_bytes = paired_genes ? (size_t)n_genes * n_genes * sizeof(double) : (size_t)n_genes * sizeof(double);
    if (result_size_bytes == 0 && n_genes > 0) { /* e.g. n_genes=0, paired_genes=false. Should be caught by n_genes=0 earlier */ }
    else if (result_size_bytes == 0) result_size_bytes = sizeof(double); // Min allocation if all dims are zero

    result_values = (double*)mkl_malloc(result_size_bytes, 64);

    if (result_values == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for result_values in calculate_morans_i_batch\n");
        return NULL;
    }
    if (!paired_genes) memset(result_values, 0, result_size_bytes); // For single gene, init to 0

    sparse_matrix_t W_mkl;
    MKL_INT* W_row_ptr = (MKL_INT*)mkl_malloc(((size_t)n_spots + 1) * sizeof(MKL_INT), 64);
    MKL_INT* W_col_ind = NULL;
    if (W_nnz > 0) {
        W_col_ind = (MKL_INT*)mkl_malloc((size_t)W_nnz * sizeof(MKL_INT), 64);
    }


    if (!W_row_ptr || (W_nnz > 0 && !W_col_ind) ) {
        fprintf(stderr, "Error: Memory allocation failed for MKL index arrays in batch\n");
        mkl_free(result_values);
        if (W_row_ptr) mkl_free(W_row_ptr);
        if (W_col_ind) mkl_free(W_col_ind);
        return NULL;
    }

    // Copy long long indices to MKL_INT indices
    for (MKL_INT i = 0; i <= n_spots; i++) W_row_ptr[i] = (MKL_INT)W_row_ptr_ll[i];
    if (W_nnz > 0) {
        for (MKL_INT i = 0; i < W_nnz; i++) W_col_ind[i] = (MKL_INT)W_col_ind_ll[i];
    }

    sparse_status_t status = mkl_sparse_d_create_csr(
        &W_mkl, SPARSE_INDEX_BASE_ZERO, n_spots, n_spots,
        W_row_ptr, W_row_ptr + 1, W_col_ind, (double*)W_values); // W_values cast to non-const, MKL doesn't modify if used read-only

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W_batch)");
        mkl_free(result_values); mkl_free(W_row_ptr); if(W_col_ind) mkl_free(W_col_ind);
        return NULL;
    }
    if (W_nnz > 0) {
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) print_mkl_status(status, "mkl_sparse_optimize (W_batch)");
    }


    struct matrix_descr descrW; descrW.type = SPARSE_MATRIX_TYPE_GENERAL;
    double alpha = 1.0, beta = 0.0;

    if (paired_genes) {
        double* Temp_WX = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
        if (Temp_WX == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for Temp_WX in batch\n");
            mkl_sparse_destroy(W_mkl); mkl_free(W_row_ptr); if(W_col_ind) mkl_free(W_col_ind); mkl_free(result_values);
            return NULL;
        }
        // W (n_spots x n_spots) * X_data (n_spots x n_genes) -> Temp_WX (n_spots x n_genes)
        // X_data is treated as Spot major (N_spots rows, N_genes cols). ldx = N_genes.
        status = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, alpha, W_mkl, descrW, 
                                  SPARSE_LAYOUT_ROW_MAJOR, X_data, n_genes, n_genes, 
                                  beta, Temp_WX, n_genes);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_d_mm (W*X batch)");
            mkl_free(Temp_WX); mkl_sparse_destroy(W_mkl); mkl_free(W_row_ptr); if(W_col_ind) mkl_free(W_col_ind); mkl_free(result_values);
            return NULL;
        }
        // Result (n_genes x n_genes) = inv_S0 * X_data_T (n_genes x n_spots) * Temp_WX (n_spots x n_genes)
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                    n_genes, n_genes, n_spots, 
                    inv_S0, X_data, n_genes,      // X_data, ldx=n_genes
                    Temp_WX, n_genes,             // Temp_WX, ldc=n_genes
                    beta, result_values, n_genes); // Result, ldc=n_genes
        mkl_free(Temp_WX);
    } else { // Single-gene Moran's I (diagonal of the paired matrix)
        double* Temp_WX = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
        if (!Temp_WX) {
            fprintf(stderr, "Error: Memory allocation failed for Temp_WX in batch (single gene mode)\n");
            mkl_sparse_destroy(W_mkl); mkl_free(W_row_ptr); if(W_col_ind) mkl_free(W_col_ind); mkl_free(result_values);
            return NULL;
        }
        status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, W_mkl, descrW, 
                                 SPARSE_LAYOUT_ROW_MAJOR, X_data, n_genes, n_genes, 
                                 beta, Temp_WX, n_genes);
        if (status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status, "mkl_sparse_d_mm (W*X for single gene batch)");
            mkl_free(Temp_WX); mkl_sparse_destroy(W_mkl); mkl_free(W_row_ptr); if(W_col_ind) mkl_free(W_col_ind); mkl_free(result_values);
            return NULL;
        }

        // For each gene g: result_values[g] = inv_S0 * (X_data[:,g]_T * Temp_WX[:,g])
        #pragma omp parallel for
        for (MKL_INT g = 0; g < n_genes; g++) {
            double dot_product = 0.0;
            // X_data is spot-major: X_data[spot_idx * n_genes + gene_idx]
            // Temp_WX is also spot-major
            for (MKL_INT spot_idx = 0; spot_idx < n_spots; spot_idx++) {
                dot_product += X_data[spot_idx * n_genes + g] * Temp_WX[spot_idx * n_genes + g];
            }
            result_values[g] = dot_product * inv_S0;
        }
        mkl_free(Temp_WX);
    }
    mkl_sparse_destroy(W_mkl); mkl_free(W_row_ptr); if(W_col_ind) mkl_free(W_col_ind);
    return result_values;
}

/* Calculate Moran's I for a single gene */
double calculate_single_gene_moran_i(const double* gene_data, const SparseMatrix* W, MKL_INT n_spots) {
    if (!gene_data || !W) {
        fprintf(stderr, "Error: Invalid parameters provided to calculate_single_gene_moran_i\n");
        return NAN;
    }
     if (W->nnz > 0 && !W->values) {
         fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in calculate_single_gene_moran_i.\n");
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


    double S0 = calculate_weight_sum(W);
    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: S0 is near zero in calculate_single_gene_moran_i. Result is NaN/Inf or 0.\n");
        return (S0 == 0.0) ? NAN : 0.0; // Or specific handling if S0 is non-zero but tiny
    }

    double* Wz = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!Wz) {
        perror("Failed to allocate Wz in calculate_single_gene_moran_i");
        return NAN;
    }

    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl, SPARSE_INDEX_BASE_ZERO, 
                                                     W->nrows, W->ncols, W->row_ptr, 
                                                     W->row_ptr + 1, W->col_ind, W->values);
    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W for single_gene_moran_i)");
        mkl_free(Wz); return NAN;
    }
    if (W->nnz > 0) { // Optimize only if W is not empty
        status = mkl_sparse_optimize(W_mkl);
        // Allow to proceed even if optimization fails, it might still work
        if (status != SPARSE_STATUS_SUCCESS) print_mkl_status(status, "mkl_sparse_optimize (W for single_gene_moran_i)");
    }


    struct matrix_descr descrW; descrW.type = SPARSE_MATRIX_TYPE_GENERAL;
    // Wz = 1.0 * W * gene_data + 0.0 * Wz
    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descrW, gene_data, 0.0, Wz);
    mkl_sparse_destroy(W_mkl);

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_mv (W*z for single_gene_moran_i)");
        mkl_free(Wz); return NAN;
    }
    // z^T * Wz
    double z_T_Wz = cblas_ddot(n_spots, gene_data, 1, Wz, 1);
    mkl_free(Wz);
    return z_T_Wz / S0;
}


/* Calculate Moran's I between the first gene and all others */
double* calculate_first_gene_vs_all(const DenseMatrix* X, const SparseMatrix* W, double S0_param) {
    if (!X || !W || !X->values || X->ncols == 0) {
        fprintf(stderr, "Error: Invalid parameters or no genes in X for calculate_first_gene_vs_all\n");
        return NULL;
    }
     if (W->nnz > 0 && !W->values) {
         fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in calculate_first_gene_vs_all.\n");
        return NULL;
    }


    MKL_INT n_spots = X->nrows; MKL_INT n_genes = X->ncols;

    double S0 = S0_param;
    if (fabs(S0_param) < DBL_EPSILON) { // If S0 passed is zero, try to recalculate
        fprintf(stderr, "Warning: S0 passed to calculate_first_gene_vs_all is near-zero (%.4e). Recalculating S0 from W.\n", S0_param);
        S0 = calculate_weight_sum(W);
        if (fabs(S0) < DBL_EPSILON) {
            fprintf(stderr, "Error: Recalculated S0 is also near-zero (%.4e). Results will be NaN/Inf or 0.\n", S0);
        }
    }

    double inv_S0;
    if (S0 == 0.0) inv_S0 = NAN; // To propagate NANs
    else inv_S0 = 1.0 / S0;


    double* moran_I_results = (double*)mkl_malloc((size_t)n_genes * sizeof(double), 64);
    if (!moran_I_results) {
        perror("Failed to allocate memory for first_gene_vs_all results"); return NULL;
    }
    // If S0 is 0 or inv_S0 is NAN, fill results with NAN and return
    if (isnan(inv_S0)) {
        for(MKL_INT g=0; g < n_genes; ++g) moran_I_results[g] = NAN;
        return moran_I_results;
    }


    double* z0_data = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!z0_data) {
        perror("Failed to allocate memory for first gene data (z0)");
        mkl_free(moran_I_results); return NULL;
    }
    // X is Spots x Genes, extract first gene (column 0)
    for (MKL_INT i = 0; i < n_spots; i++) z0_data[i] = X->values[i * n_genes + 0];

    double* W_z0 = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!W_z0) {
        perror("Failed to allocate memory for W_z0");
        mkl_free(z0_data); mkl_free(moran_I_results); return NULL;
    }

    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl, SPARSE_INDEX_BASE_ZERO, W->nrows, W->ncols, W->row_ptr, W->row_ptr + 1, W->col_ind, W->values);
    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W for first_gene_vs_all)");
        mkl_free(W_z0); mkl_free(z0_data); mkl_free(moran_I_results); return NULL;
    }
    if (W->nnz > 0) {
        status = mkl_sparse_optimize(W_mkl);
        if (status != SPARSE_STATUS_SUCCESS) print_mkl_status(status, "mkl_sparse_optimize (W for first_gene_vs_all)");
    }

    struct matrix_descr descrW; descrW.type = SPARSE_MATRIX_TYPE_GENERAL;
    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, W_mkl, descrW, z0_data, 0.0, W_z0);
    mkl_sparse_destroy(W_mkl);

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_mv (W*z0 for first_gene_vs_all)");
        mkl_free(W_z0); mkl_free(z0_data); mkl_free(moran_I_results); return NULL;
    }

    // For each gene g: moran_I_results[g] = inv_S0 * (X[:,g]_T * W_z0)
    #pragma omp parallel for
    for (MKL_INT g = 0; g < n_genes; g++) {
        double dot_product = 0;
        // X is Spots x Genes (row-major)
        // X->values[spot_idx * n_genes + g] is element (spot_idx, g)
        for (MKL_INT spot_idx = 0; spot_idx < n_spots; spot_idx++) {
            dot_product += X->values[spot_idx * n_genes + g] * W_z0[spot_idx];
        }
        moran_I_results[g] = dot_product * inv_S0;
    }
    mkl_free(z0_data); mkl_free(W_z0);
    return moran_I_results;
}


/**
 * @brief Permutes expression data for each gene and fills a target matrix.
 *
 * This function takes an original expression matrix (X_original, Spots x Genes)
 * and permutes the expression values for each gene independently across spots.
 * The permuted values for each gene column are written into the corresponding
 * column of the pre-allocated X_target_perm matrix.
 *
 * Temporary buffers (temp_gene_column_buffer, temp_indices_buffer) must be
 * pre-allocated by the caller to avoid frequent allocations within loops.
 *
 * @param X_target_perm Pointer to the pre-allocated DenseMatrix where permuted data will be stored.
 *                      Its `values` array must be allocated and have dimensions matching X_original.
 * @param X_original Pointer to the constant DenseMatrix containing the original expression data.
 * @param temp_gene_column_buffer Pre-allocated buffer of size n_spots (double) for temporary storage of a gene's column.
 * @param temp_indices_buffer Pre-allocated buffer of size n_spots (MKL_INT) for shuffling indices.
 * @param p_iteration_seed Pointer to an unsigned int seed state for rand_r (will be updated).
 * @return int 1 on success, 0 on failure.
 */
static int permute_and_fill_target_matrix(
    DenseMatrix* X_target_perm,
    const DenseMatrix* X_original,
    double* temp_gene_column_buffer,
    MKL_INT* temp_indices_buffer,
    unsigned int* p_iteration_seed)
{
    if (!X_target_perm || !X_target_perm->values ||
        !X_original   || !X_original->values   ||
        !temp_gene_column_buffer || !temp_indices_buffer || !p_iteration_seed) {
        // Error already printed by caller in case of buffer allocation failure
        // This is an internal consistency check primarily.
        // fprintf(stderr, "Error (permute_and_fill_target_matrix): Invalid input pointer(s).\n");
        return 0; // Failure
    }

    MKL_INT n_spots = X_original->nrows;
    MKL_INT n_genes = X_original->ncols;

    if (X_target_perm->nrows != n_spots || X_target_perm->ncols != n_genes) {
        fprintf(stderr, "Error (permute_and_fill_target_matrix): Dimension mismatch between target (%lldx%lld) and original (%lldx%lld).\n",
                (long long)X_target_perm->nrows, (long long)X_target_perm->ncols, (long long)n_spots, (long long)n_genes);
        return 0; // Failure
    }

    for (MKL_INT j = 0; j < n_genes; j++) { // For each gene (column)
        // 1. Copy original values of gene j from X_original into temp_gene_column_buffer
        //    and initialize indices.
        for (MKL_INT i = 0; i < n_spots; i++) {
            temp_gene_column_buffer[i] = X_original->values[i * n_genes + j]; // X_original is Spots x Genes
            temp_indices_buffer[i] = i;
        }

        // 2. Fisher-Yates shuffle for the current gene's indices (in temp_indices_buffer)
        if (n_spots > 1) {
            for (MKL_INT i = n_spots - 1; i > 0; i--) {
                MKL_INT k = rand_r(p_iteration_seed) % (i + 1);
                MKL_INT temp_idx_val = temp_indices_buffer[i];
                temp_indices_buffer[i] = temp_indices_buffer[k];
                temp_indices_buffer[k] = temp_idx_val;
            }
        }

        // 3. Apply permuted indices to fill the j-th column in X_target_perm
        //    X_target_perm[spot_original_idx, gene_j] = temp_gene_column_buffer[permuted_indices[spot_original_idx]]
        //    This means: X_target_perm->values[i * n_genes + j] = temp_gene_column_buffer[temp_indices_buffer[i]];
        for (MKL_INT i = 0; i < n_spots; i++) {
            X_target_perm->values[i * n_genes + j] = temp_gene_column_buffer[temp_indices_buffer[i]];
        }
    }
    return 1; // Success
}


/* Run the full permutation test */
PermutationResults* run_permutation_test(const DenseMatrix* X_observed_spots_x_genes,
                                       const SparseMatrix* W_spots_x_spots,
                                       const PermutationParams* params) {

    // Initialize pointers that will be used in cleanup section to NULL
    sparse_matrix_t W_mkl_handle = NULL;
    PermutationResults* results = NULL;
    DenseMatrix* observed_morans_i_for_perm = NULL;
    double* Temp_WX_obs_perm = NULL;
    sparse_status_t status_mkl_perm;

    // Parameter validation
    if (!X_observed_spots_x_genes || !W_spots_x_spots || !params ||
        !X_observed_spots_x_genes->values || !X_observed_spots_x_genes->colnames) {
        fprintf(stderr, "Error: Invalid parameters provided to run_permutation_test (NULL pointers or missing data).\n");
        return NULL;
    }
    if (W_spots_x_spots->nnz > 0 && !W_spots_x_spots->values) {
         fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in run_permutation_test.\n");
        return NULL;
    }

    MKL_INT n_spots = X_observed_spots_x_genes->nrows;
    MKL_INT n_genes = X_observed_spots_x_genes->ncols;
    int n_perm = params->n_permutations;

    if (n_genes == 0 || n_spots == 0) {
        fprintf(stderr, "Error: Expression matrix (X) has zero dimensions (spots=%lld, genes=%lld) in run_permutation_test.\n", (long long)n_spots, (long long)n_genes);
        return NULL;
    }
    if (n_perm <= 0) {
        fprintf(stderr, "Error: Number of permutations (%d) must be positive.\n", n_perm);
        return NULL;
    }

    printf("Running permutation test with %d permutations for %lld genes...\n", n_perm, (long long)n_genes);

    double S0 = calculate_weight_sum(W_spots_x_spots);
    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Error: Sum of weights S0 is near-zero (%.4e) in permutation test. Cannot proceed reliably.\n", S0);
        return NULL; // W_mkl_handle, results, etc. are still NULL
    }
    double inv_S0 = 1.0 / S0;
    printf("  Permutation Test: Using S0 = %.6f, inv_S0 = %.6e\n", S0, inv_S0);

    // Create W_mkl_handle (W_mkl_handle is already NULL initialized)
    status_mkl_perm = mkl_sparse_d_create_csr(
        &W_mkl_handle, SPARSE_INDEX_BASE_ZERO, W_spots_x_spots->nrows, W_spots_x_spots->ncols,
        W_spots_x_spots->row_ptr, W_spots_x_spots->row_ptr + 1, W_spots_x_spots->col_ind, W_spots_x_spots->values);

    if (status_mkl_perm != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status_mkl_perm, "mkl_sparse_d_create_csr (W_perm_test)");
        // W_mkl_handle might be in an indeterminate state if create_csr failed partway.
        // However, mkl_sparse_destroy should handle potentially invalid handles if called.
        // Best practice might be to not call destroy if create failed, but MKL is usually robust.
        // For safety, let's assume W_mkl_handle is not valid to destroy if create fails.
        // The cleanup label will try to destroy it if it's not NULL, which it won't be if create fails and returns.
        return NULL;
    }
    if (W_spots_x_spots->nnz > 0) {
        status_mkl_perm = mkl_sparse_optimize(W_mkl_handle);
        if (status_mkl_perm != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(status_mkl_perm, "mkl_sparse_optimize (W_perm_test)");
        }
    }
    struct matrix_descr descrW_perm;
    descrW_perm.type = SPARSE_MATRIX_TYPE_GENERAL;
    double alpha_mm_perm = 1.0, beta_mm_perm = 0.0;

    // Allocate PermutationResults struct itself
    results = (PermutationResults*)calloc(1, sizeof(PermutationResults));
    if (!results) {
        perror("Failed to allocate PermutationResults structure");
        goto perm_error_cleanup_results; // W_mkl_handle is valid here
    }
    results->mean_perm = NULL; results->var_perm = NULL;
    results->z_scores = NULL; results->p_values = NULL;


    size_t moran_matrix_size_elems = (size_t)n_genes * n_genes;
    results->mean_perm = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!results->mean_perm) { perror("alloc mean_perm struct"); goto perm_error_cleanup_results; }
    results->mean_perm->nrows = n_genes; results->mean_perm->ncols = n_genes;
    results->mean_perm->values = (double*)mkl_calloc(moran_matrix_size_elems, sizeof(double), 64);
    results->mean_perm->rownames = (char**)calloc(n_genes, sizeof(char*));
    results->mean_perm->colnames = (char**)calloc(n_genes, sizeof(char*));
    if (!results->mean_perm->values || !results->mean_perm->rownames || !results->mean_perm->colnames) {
        perror("alloc mean_perm components"); goto perm_error_cleanup_results;
    }

    results->var_perm = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!results->var_perm) { perror("alloc var_perm struct"); goto perm_error_cleanup_results; }
    results->var_perm->nrows = n_genes; results->var_perm->ncols = n_genes;
    results->var_perm->values = (double*)mkl_calloc(moran_matrix_size_elems, sizeof(double), 64);
    results->var_perm->rownames = (char**)calloc(n_genes, sizeof(char*));
    results->var_perm->colnames = (char**)calloc(n_genes, sizeof(char*));
    if (!results->var_perm->values || !results->var_perm->rownames || !results->var_perm->colnames) {
        perror("alloc var_perm components"); goto perm_error_cleanup_results;
    }

    if (params->z_score_output) {
        results->z_scores = (DenseMatrix*)malloc(sizeof(DenseMatrix));
        if(!results->z_scores) { perror("alloc z_scores struct"); goto perm_error_cleanup_results; }
        results->z_scores->nrows = n_genes; results->z_scores->ncols = n_genes;
        results->z_scores->values = (double*)mkl_calloc(moran_matrix_size_elems, sizeof(double), 64);
        results->z_scores->rownames = (char**)calloc(n_genes, sizeof(char*));
        results->z_scores->colnames = (char**)calloc(n_genes, sizeof(char*));
        if(!results->z_scores->values || !results->z_scores->rownames || !results->z_scores->colnames){
            perror("alloc z_scores components"); goto perm_error_cleanup_results;
        }
    }
    if (params->p_value_output) {
        results->p_values = (DenseMatrix*)malloc(sizeof(DenseMatrix));
        if(!results->p_values) { perror("alloc p_values struct"); goto perm_error_cleanup_results; }
        results->p_values->nrows = n_genes; results->p_values->ncols = n_genes;
        results->p_values->values = (double*)mkl_calloc(moran_matrix_size_elems, sizeof(double), 64);
        results->p_values->rownames = (char**)calloc(n_genes, sizeof(char*));
        results->p_values->colnames = (char**)calloc(n_genes, sizeof(char*));
        if(!results->p_values->values || !results->p_values->rownames || !results->p_values->colnames){
             perror("alloc p_values components"); goto perm_error_cleanup_results;
        }
    }

    // All results->... struct members and their basic arrays allocated. Now names.
    for(MKL_INT i=0; i < n_genes; ++i) {
        const char* gene_name_src = (X_observed_spots_x_genes->colnames[i]) ? X_observed_spots_x_genes->colnames[i] : "UNKNOWN_GENE";
        if (results->mean_perm) { // Should always be true if we reached here
            results->mean_perm->rownames[i] = strdup(gene_name_src);
            results->mean_perm->colnames[i] = strdup(gene_name_src);
            if(!results->mean_perm->rownames[i] || !results->mean_perm->colnames[i]) {
                 perror("strdup failed for mean_perm gene names"); goto perm_error_cleanup_results;
            }
        }
        if (results->var_perm) { // Should always be true
            results->var_perm->rownames[i] = strdup(gene_name_src);
            results->var_perm->colnames[i] = strdup(gene_name_src);
             if(!results->var_perm->rownames[i] || !results->var_perm->colnames[i]) {
                 perror("strdup failed for var_perm gene names"); goto perm_error_cleanup_results;
            }
        }
        if (results->z_scores) { // Only if allocated
            results->z_scores->rownames[i] = strdup(gene_name_src);
            results->z_scores->colnames[i] = strdup(gene_name_src);
            if (!results->z_scores->rownames[i] || !results->z_scores->colnames[i]) {
                perror("strdup failed for z_scores gene names"); goto perm_error_cleanup_results;
            }
        }
        if (results->p_values) { // Only if allocated
            results->p_values->rownames[i] = strdup(gene_name_src);
            results->p_values->colnames[i] = strdup(gene_name_src);
            if (!results->p_values->rownames[i] || !results->p_values->colnames[i]) {
                perror("strdup failed for p_values gene names"); goto perm_error_cleanup_results;
            }
        }
    }

    // Now allocate observed_morans_i_for_perm and Temp_WX_obs_perm (they are currently NULL)
    printf("  Permutation Test: Calculating observed Moran's I for reference...\n");
    observed_morans_i_for_perm = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!observed_morans_i_for_perm) { perror("alloc observed_morans_i_for_perm struct"); goto perm_error_cleanup_results;}
    observed_morans_i_for_perm->nrows = n_genes; observed_morans_i_for_perm->ncols = n_genes;
    observed_morans_i_for_perm->rownames = NULL; observed_morans_i_for_perm->colnames = NULL;
    observed_morans_i_for_perm->values = (double*)mkl_malloc(moran_matrix_size_elems * sizeof(double), 64);
    if (!observed_morans_i_for_perm->values) {
        perror("alloc observed_morans_i_for_perm values");
        goto perm_error_cleanup_results; // observed_morans_i_for_perm is not NULL, but its ->values is. free_dense_matrix will handle it.
    }

    Temp_WX_obs_perm = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
    if (!Temp_WX_obs_perm) {
        perror("alloc Temp_WX_obs_perm");
        goto perm_error_cleanup_results;
    }
    status_mkl_perm = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha_mm_perm, W_mkl_handle, descrW_perm,
                             SPARSE_LAYOUT_ROW_MAJOR, X_observed_spots_x_genes->values, n_genes, n_genes,
                             beta_mm_perm, Temp_WX_obs_perm, n_genes);
    if (status_mkl_perm != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status_mkl_perm, "mkl_sparse_d_mm (W*X_obs for perm)");
        goto perm_error_cleanup_results;
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_genes, n_genes, n_spots, inv_S0,
                X_observed_spots_x_genes->values, n_genes, Temp_WX_obs_perm, n_genes, beta_mm_perm,
                observed_morans_i_for_perm->values, n_genes);
    mkl_free(Temp_WX_obs_perm); Temp_WX_obs_perm = NULL; // Successfully used and now freed
    printf("  Permutation Test: Observed Moran's I calculated for comparison.\n");


    // --- Start of parallel region ---
    volatile int completed_perms = 0;
    int progress_interval_percent = 5;
    volatile int next_progress_milestone_count = (n_perm * progress_interval_percent) / 100;
    if (next_progress_milestone_count == 0 && n_perm > 0) next_progress_milestone_count = 1;
    int min_perms_for_etr = (n_perm > 20) ? 10 : 1;
    if (n_perm < 20) progress_interval_percent = 0;

    printf("Starting permutation loop (%d permutations) using up to %d OpenMP threads...\n", n_perm, omp_get_max_threads());
    volatile int perm_loop_error_flag = 0;
    double loop_start_time = get_time();

    #pragma omp parallel
    {
        DenseMatrix X_p_tl_struct;
        X_p_tl_struct.nrows = n_spots; X_p_tl_struct.ncols = n_genes;
        X_p_tl_struct.rownames = NULL; X_p_tl_struct.colnames = NULL;
        X_p_tl_struct.values = NULL;

        double* gene_col_buffer_tl = NULL;
        MKL_INT* perm_indices_buffer_tl = NULL;
        double* Temp_WX_p_local_tl = NULL;
        double* I_p_values_current_perm_tl = NULL;
        double* local_mean_sum_tl = NULL;
        double* local_var_sum_sq_tl = NULL;
        double* local_p_counts_tl = NULL;

        X_p_tl_struct.values = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
        gene_col_buffer_tl = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
        perm_indices_buffer_tl = (MKL_INT*)mkl_malloc((size_t)n_spots * sizeof(MKL_INT), 64);
        Temp_WX_p_local_tl = (double*)mkl_malloc((size_t)n_spots * n_genes * sizeof(double), 64);
        I_p_values_current_perm_tl = (double*)mkl_malloc(moran_matrix_size_elems * sizeof(double), 64);
        local_mean_sum_tl = (double*)mkl_calloc(moran_matrix_size_elems, sizeof(double), 64);
        local_var_sum_sq_tl = (double*)mkl_calloc(moran_matrix_size_elems, sizeof(double), 64);
        if (params->p_value_output) {
            local_p_counts_tl = (double*)mkl_calloc(moran_matrix_size_elems, sizeof(double), 64);
        }
        unsigned int thread_local_seed_state = params->seed + omp_get_thread_num() + 1;

        int thread_alloc_ok = (X_p_tl_struct.values && gene_col_buffer_tl && perm_indices_buffer_tl &&
                               Temp_WX_p_local_tl && I_p_values_current_perm_tl &&
                               local_mean_sum_tl && local_var_sum_sq_tl &&
                               (!params->p_value_output || local_p_counts_tl) );

        if (!thread_alloc_ok) {
            #pragma omp critical (PermErrorCrit)
            {
                if (!perm_loop_error_flag) {
                    fprintf(stderr, "Permutation Thread Error: Failed to allocate thread-local buffers. Thread %d.\n", omp_get_thread_num());
                }
                perm_loop_error_flag = 1;
            }
        }

        #pragma omp flush(perm_loop_error_flag)

        if (thread_alloc_ok && !perm_loop_error_flag) {
            #pragma omp for schedule(dynamic)
            for (int p_idx = 0; p_idx < n_perm; p_idx++) {
                if(perm_loop_error_flag) {
                    continue;
                }

                if (!permute_and_fill_target_matrix(&X_p_tl_struct, X_observed_spots_x_genes,
                                                   gene_col_buffer_tl, perm_indices_buffer_tl,
                                                   &thread_local_seed_state)) {
                    #pragma omp critical (PermErrorCrit)
                    {
                        if(!perm_loop_error_flag) fprintf(stderr, "Permutation %d (Thread %d): Failed during permute_and_fill_target_matrix.\n", p_idx + 1, omp_get_thread_num());
                        perm_loop_error_flag = 1;
                    }
                    continue;
                }

                sparse_status_t stat_mm_loop_local = mkl_sparse_d_mm(
                    SPARSE_OPERATION_NON_TRANSPOSE, alpha_mm_perm, W_mkl_handle, descrW_perm,
                    SPARSE_LAYOUT_ROW_MAJOR, X_p_tl_struct.values, n_genes, n_genes,
                    beta_mm_perm, Temp_WX_p_local_tl, n_genes);

                if (stat_mm_loop_local != SPARSE_STATUS_SUCCESS) {
                    #pragma omp critical (PermErrorCrit)
                    {
                        if(!perm_loop_error_flag) {print_mkl_status(stat_mm_loop_local, "mkl_sparse_d_mm (W*X_p in perm loop)");}
                        perm_loop_error_flag = 1;
                    }
                    continue;
                }

                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            n_genes, n_genes, n_spots,
                            inv_S0, X_p_tl_struct.values, n_genes,
                            Temp_WX_p_local_tl, n_genes,
                            beta_mm_perm, I_p_values_current_perm_tl, n_genes);

                for (MKL_INT r_idx = 0; r_idx < n_genes; r_idx++) {
                    for (MKL_INT c_idx = 0; c_idx < n_genes; c_idx++) {
                        MKL_INT flat_idx = r_idx * n_genes + c_idx;
                        double perm_val = I_p_values_current_perm_tl[flat_idx];
                        if (!isfinite(perm_val)) perm_val = 0.0;
                        local_mean_sum_tl[flat_idx] += perm_val;
                        local_var_sum_sq_tl[flat_idx] += perm_val * perm_val;
                        if (params->p_value_output && local_p_counts_tl) {
                            if (fabs(perm_val) >= fabs(observed_morans_i_for_perm->values[flat_idx])) {
                                local_p_counts_tl[flat_idx]++;
                            }
                        }
                    }
                }

                #pragma omp critical (ProgressUpdateCrit)
                {
                    if (!perm_loop_error_flag) {
                        int current_completed = ++completed_perms;
                        int print_now = 0;
                        if (progress_interval_percent > 0) {
                             if (current_completed >= next_progress_milestone_count || current_completed == n_perm) {
                                print_now = 1;
                                if (current_completed < n_perm) {
                                    int new_milestone = next_progress_milestone_count + (n_perm * progress_interval_percent) / 100;
                                    if (new_milestone <= current_completed) new_milestone = current_completed + 1;
                                    next_progress_milestone_count = (new_milestone > n_perm) ? n_perm : new_milestone;
                                }
                            }
                        } else if (n_perm <= 20) { print_now = 1; }

                        if (print_now) {
                            double time_now = get_time();
                            double time_elapsed_s = time_now - loop_start_time;
                            double percent_done_val = (100.0 * current_completed) / n_perm;
                            int bar_width = 20;
                            int filled_len = (int)(bar_width * current_completed / n_perm);
                            char bar_str[bar_width + 3];
                            bar_str[0] = '[';
                            for (int k_bar = 0; k_bar < bar_width; ++k_bar) bar_str[k_bar + 1] = (k_bar < filled_len) ? '=' : ' ';
                            bar_str[bar_width + 1] = ']'; bar_str[bar_width + 2] = '\0';
                            printf("\rPermutations: %s %d/%d (%.1f%%)", bar_str, current_completed, n_perm, percent_done_val);
                            if (current_completed >= min_perms_for_etr && time_elapsed_s > 0.5 && current_completed < n_perm) {
                                double time_per_perm_s = time_elapsed_s / current_completed;
                                double time_remaining_s = time_per_perm_s * (n_perm - current_completed);
                                if (time_remaining_s < 60) printf(" ETR: %.0fs", time_remaining_s);
                                else if (time_remaining_s < 3600) printf(" ETR: %.1fm", time_remaining_s / 60.0);
                                else printf(" ETR: %.1fh", time_remaining_s / 3600.0);
                            }
                            fflush(stdout);
                        }
                    }
                }
            }
        }

        if (thread_alloc_ok && !perm_loop_error_flag) {
            #pragma omp critical (MeanPermReduce)
            {
                if (!perm_loop_error_flag && results && results->mean_perm && results->mean_perm->values) {
                    for(size_t k=0; k < moran_matrix_size_elems; ++k) results->mean_perm->values[k] += local_mean_sum_tl[k];
                } else if(results && results->mean_perm && results->mean_perm->values) {results->mean_perm->values[0] = NAN;} // Mark as tainted
            }
            #pragma omp critical (VarPermReduce)
            {
                if (!perm_loop_error_flag && results && results->var_perm && results->var_perm->values) {
                     for(size_t k=0; k < moran_matrix_size_elems; ++k) results->var_perm->values[k] += local_var_sum_sq_tl[k];
                } else if(results && results->var_perm && results->var_perm->values) {results->var_perm->values[0] = NAN;}
            }
            if (params->p_value_output && results && results->p_values && results->p_values->values && local_p_counts_tl) {
                #pragma omp critical (PvalPermReduce)
                {
                    if (!perm_loop_error_flag) {
                        for(size_t k=0; k < moran_matrix_size_elems; ++k) results->p_values->values[k] += local_p_counts_tl[k];
                    } else if(results->p_values && results->p_values->values) {results->p_values->values[0] = NAN;}
                }
            }
        }

        if(X_p_tl_struct.values) mkl_free(X_p_tl_struct.values);
        if(gene_col_buffer_tl) mkl_free(gene_col_buffer_tl);
        if(perm_indices_buffer_tl) mkl_free(perm_indices_buffer_tl);
        if(Temp_WX_p_local_tl) mkl_free(Temp_WX_p_local_tl);
        if(I_p_values_current_perm_tl) mkl_free(I_p_values_current_perm_tl);
        if(local_mean_sum_tl) mkl_free(local_mean_sum_tl);
        if(local_var_sum_sq_tl) mkl_free(local_var_sum_sq_tl);
        if(local_p_counts_tl) mkl_free(local_p_counts_tl);
    }
    // --- End of parallel region ---

    if (progress_interval_percent > 0 || n_perm <=20) { printf("\n"); }

    // W_mkl_handle already checked and destroyed if not NULL at perm_error_cleanup_results or here
    // No, it's destroyed here if loop completes normally.
    if (W_mkl_handle) mkl_sparse_destroy(W_mkl_handle); W_mkl_handle = NULL;


    int final_check_nan = 0;
    if (results && results->mean_perm && results->mean_perm->values && moran_matrix_size_elems > 0 && isnan(results->mean_perm->values[0])) {
        final_check_nan = 1;
    }


    if(perm_loop_error_flag || final_check_nan){
        fprintf(stderr, "Error occurred during permutation loop execution. Results may be incomplete or unreliable.\n");
        // Temp_WX_obs_perm should be NULL here if it was successfully freed after use, or never allocated, or freed in cleanup
        // observed_morans_i_for_perm should be NULL if successfully freed after use, or never allocated, or freed in cleanup
        goto perm_error_cleanup_results; // Jumps to the cleanup section that handles W_mkl_handle and results
    }

    // Finalize calculations for mean, variance, p-values, and Z-scores
    double inv_n_perm_double_final = 1.0 / (double)n_perm;
    for (MKL_INT r_fin = 0; r_fin < n_genes; r_fin++) {
        for (MKL_INT c_fin = 0; c_fin < n_genes; c_fin++) {
            MKL_INT flat_idx_fin = r_fin * n_genes + c_fin;

            double sum_val_fin = results->mean_perm->values[flat_idx_fin];
            double sum_sq_val_fin = results->var_perm->values[flat_idx_fin];

            double mean_p_fin = sum_val_fin * inv_n_perm_double_final;
            double var_p_fin = (sum_sq_val_fin * inv_n_perm_double_final) - (mean_p_fin * mean_p_fin);

            if (var_p_fin < 0.0 && var_p_fin > -ZERO_STD_THRESHOLD) var_p_fin = 0.0;
            else if (var_p_fin < 0.0) {
                 fprintf(stderr, "Warning: Negative variance (%.4e) calculated for gene pair index (%lld,%lld). Clamping to 0.\n",
                         var_p_fin, (long long)r_fin, (long long)c_fin);
                 var_p_fin = 0.0;
            }
            results->mean_perm->values[flat_idx_fin] = mean_p_fin;
            results->var_perm->values[flat_idx_fin] = var_p_fin;

            if (params->p_value_output && results->p_values) {
                results->p_values->values[flat_idx_fin] = (results->p_values->values[flat_idx_fin] + 1.0) / (double)(n_perm + 1.0);
            }
            if (params->z_score_output && results->z_scores) {
                double std_dev_p_fin = sqrt(var_p_fin);
                // observed_morans_i_for_perm must be valid here
                if (!observed_morans_i_for_perm || !observed_morans_i_for_perm->values) {
                    // This case should ideally not be reached if setup was correct
                    fprintf(stderr, "Error: observed_morans_i_for_perm->values is NULL during Z-score calculation.\n");
                    results->z_scores->values[flat_idx_fin] = NAN;
                } else {
                    double observed_val_fin = observed_morans_i_for_perm->values[flat_idx_fin];
                    if (std_dev_p_fin < ZERO_STD_THRESHOLD) {
                        if (fabs(observed_val_fin - mean_p_fin) < ZERO_STD_THRESHOLD) results->z_scores->values[flat_idx_fin] = 0.0;
                        else results->z_scores->values[flat_idx_fin] = (observed_val_fin > mean_p_fin) ? INFINITY : -INFINITY;
                    } else {
                        results->z_scores->values[flat_idx_fin] = (observed_val_fin - mean_p_fin) / std_dev_p_fin;
                    }
                }
            }
        }
    }

    if (observed_morans_i_for_perm) free_dense_matrix(observed_morans_i_for_perm);
    // Temp_WX_obs_perm is already NULL here
    // W_mkl_handle is already NULL here

    printf("Permutation test complete.\n");
    return results;

perm_error_cleanup_results:
    if (W_mkl_handle) mkl_sparse_destroy(W_mkl_handle);
    if (Temp_WX_obs_perm) mkl_free(Temp_WX_obs_perm);
    if (observed_morans_i_for_perm) free_dense_matrix(observed_morans_i_for_perm);
    if (results) free_permutation_results(results);
    return NULL;
}



/* Save results to file */
int save_results(const DenseMatrix* result_matrix, const char* output_file) {
    if (!result_matrix || /*!result_matrix->values ||*/ !output_file) { // values can be NULL if nrows/ncols is 0
        fprintf(stderr, "Error: Cannot save NULL result matrix or empty filename.\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n_rows = result_matrix->nrows;
    MKL_INT n_cols = result_matrix->ncols;

    if ((n_rows == 0 || n_cols == 0) && result_matrix->values != NULL) {
         fprintf(stderr, "Warning: Result matrix has a zero dimension but values pointer is not NULL. This is unusual.\n");
    }
    if (n_rows > 0 && n_cols > 0 && result_matrix->values == NULL) {
        fprintf(stderr, "Error: Result matrix has non-zero dimensions (%lldx%lld) but values pointer is NULL.\n", (long long)n_rows, (long long)n_cols);
        return MORANS_I_ERROR_PARAMETER;
    }


    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s' for writing: %s\n",
                output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    
    if (n_rows == 0 || n_cols == 0) {
        printf("Warning: Result matrix is empty (dims %lldx%lld), saving (nearly) empty file to %s.\n", (long long)n_rows, (long long)n_cols, output_file);
        // Write header even if empty, for consistency, if colnames are available
        if (result_matrix->colnames != NULL) { // Might have colnames even if n_cols=0 (e.g. from an empty input)
            fprintf(fp, " "); 
            for (MKL_INT j = 0; j < n_cols; j++) { // Loop won't run if n_cols=0
                 fprintf(fp, "\t%s", result_matrix->colnames[j] ? result_matrix->colnames[j] : "UNKNOWN_COL");
            }
        }
        fprintf(fp, "\n"); // Always print at least a newline
        fclose(fp);
        return MORANS_I_SUCCESS;
    }

    printf("Saving results to %s...\n", output_file);
    
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

    for (MKL_INT i = 0; i < n_rows; i++) {
        fprintf(fp, "%s", (result_matrix->rownames && result_matrix->rownames[i]) ? result_matrix->rownames[i] : "UNKNOWN_ROW");
        for (MKL_INT j = 0; j < n_cols; j++) {
            double value = result_matrix->values[i * n_cols + j];
            fprintf(fp, "\t"); 

            if (isnan(value)) fprintf(fp, "NaN");
            else if (isinf(value)) fprintf(fp, "%sInf", (value > 0 ? "" : "-"));
            else {
                if ((fabs(value) > 0 && fabs(value) < 1e-4) || fabs(value) > 1e6) fprintf(fp, "%.6e", value);
                else fprintf(fp, "%.8f", value);
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Results saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}

/* Save single gene Moran's I results */
// Mark S0 as S0_unused if not directly used, or adjust logic.
int save_single_gene_results(const DenseMatrix* X_calc, const SparseMatrix* W, double S0_unused, const char* output_file) {
    (void)S0_unused; // Explicitly mark as unused to silence compiler warning if it's truly not used.
    if (!X_calc || !W || !output_file) {
        fprintf(stderr, "Error: Invalid parameters provided to save_single_gene_results\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (X_calc->nrows > 0 && X_calc->ncols > 0 && !X_calc->values) {
         fprintf(stderr, "Error: X_calc has non-zero dims but NULL values in save_single_gene_results\n");
        return MORANS_I_ERROR_PARAMETER;
    }
     if (W->nnz > 0 && !W->values) {
         fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in save_single_gene_results\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n_spots = X_calc->nrows; MKL_INT n_genes = X_calc->ncols;
    
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s' for writing: %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    fprintf(fp, "Gene\tMoranI\n"); 

    if (n_genes == 0) {
        printf("Warning: No genes in X_calc for single_gene_results, saving header-only file to %s.\n", output_file);
        fclose(fp);
        return MORANS_I_SUCCESS;
    }
    if (n_spots == 0) {
        printf("Warning: No spots in X_calc for single_gene_results. Moran's I is undefined. Saving NaNs.\n");
         for (MKL_INT g = 0; g < n_genes; g++) {
             fprintf(fp, "%s\tNaN\n", (X_calc->colnames && X_calc->colnames[g]) ? X_calc->colnames[g] : "UNKNOWN_GENE");
        }
        fclose(fp);
        return MORANS_I_SUCCESS; 
    }

    double* gene_data_col = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!gene_data_col) {
        perror("Failed to allocate gene_data_col in save_single_gene_results");
        fclose(fp); return MORANS_I_ERROR_MEMORY;
    }

    for (MKL_INT g = 0; g < n_genes; g++) {
        for (MKL_INT spot_idx = 0; spot_idx < n_spots; spot_idx++) {
            gene_data_col[spot_idx] = X_calc->values[spot_idx * n_genes + g];
        }
        double moran_val = calculate_single_gene_moran_i(gene_data_col, W, n_spots);

        fprintf(fp, "%s\t", (X_calc->colnames && X_calc->colnames[g]) ? X_calc->colnames[g] : "UNKNOWN_GENE");
        if (isnan(moran_val)) fprintf(fp, "NaN\n");
        else if (isinf(moran_val)) fprintf(fp, "%sInf\n", (moran_val > 0 ? "" : "-"));
        else if (fabs(moran_val) > 0 && fabs(moran_val) < 1e-4) fprintf(fp, "%.6e\n", moran_val);
        else fprintf(fp, "%.8f\n", moran_val);
    }
    mkl_free(gene_data_col); fclose(fp);
    printf("Single-gene Moran's I results saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}


/* Save first gene vs all results */
int save_first_gene_vs_all_results(const double* morans_values, const char** gene_names,
                                  MKL_INT n_genes, const char* output_file) {
    if (!output_file) {
        fprintf(stderr, "Error: Null output_file provided to save_first_gene_vs_all_results\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    // morans_values and gene_names can be NULL if n_genes is 0.
     if (n_genes > 0 && (!morans_values || !gene_names)) {
        fprintf(stderr, "Error: Non-zero n_genes but NULL morans_values or gene_names in save_first_gene_vs_all_results\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s' for writing: %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    
    const char* first_gene_name_header = (n_genes > 0 && gene_names && gene_names[0]) ? gene_names[0] : "FirstGene";
    fprintf(fp, "Gene\tMoranI_vs_%s\n", first_gene_name_header);

    if (n_genes == 0) {
        printf("Warning: No genes to save for first_gene_vs_all, saving header-only file to %s\n", output_file);
        fclose(fp);
        return MORANS_I_SUCCESS;
    }
    
    printf("Saving first gene vs all Moran's I results to %s...\n", output_file);
    for (MKL_INT g = 0; g < n_genes; g++) {
        fprintf(fp, "%s\t", (gene_names[g]) ? gene_names[g] : "UNKNOWN_GENE");
        double value = morans_values[g];
        if (isnan(value)) fprintf(fp, "NaN\n");
        else if (isinf(value)) fprintf(fp, "%sInf\n", (value > 0 ? "" : "-"));
        else if (fabs(value) > 0 && fabs(value) < 1e-4) fprintf(fp, "%.6e\n", value);
        else fprintf(fp, "%.8f\n", value);
    }
    fclose(fp);
    printf("First gene vs all Moran's I results saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}


/* Save permutation test results to file */
int save_permutation_results(const PermutationResults* results,
                             const char* output_file_prefix) {

    if (!results || !output_file_prefix) {
        fprintf(stderr, "Error (save_permutation_results): Cannot save NULL results or empty output prefix.\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    char filename_buffer[BUFFER_SIZE];
    int status = MORANS_I_SUCCESS;
    int temp_status;

    printf("--- Saving Permutation Test Results (Lower Triangular Raw Format) ---\n");

    if (results->z_scores && results->z_scores->values) { 
        snprintf(filename_buffer, BUFFER_SIZE, "%s_zscores_lower_tri.tsv", output_file_prefix);
        printf("Saving Z-scores (lower triangular) to: %s\n", filename_buffer);
        temp_status = save_lower_triangular_matrix_raw(results->z_scores, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving Z-scores (lower tri).\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    } else {
        printf("Z-scores not calculated or not requested by config, skipping save.\n");
    }

    if (results->p_values && results->p_values->values) { 
        snprintf(filename_buffer, BUFFER_SIZE, "%s_pvalues_lower_tri.tsv", output_file_prefix);
        printf("Saving P-values (lower triangular) to: %s\n", filename_buffer);
        temp_status = save_lower_triangular_matrix_raw(results->p_values, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving P-values (lower tri).\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    } else {
        printf("P-values not calculated or not requested by config, skipping save.\n");
    }

    if (status == MORANS_I_SUCCESS) {
        printf("Permutation test Z-scores/P-values (if requested) saved with prefix: %s\n", output_file_prefix);
    }
    return status;
}


/* Save the lower triangular part of a symmetric matrix to a file in raw format. */
int save_lower_triangular_matrix_raw(const DenseMatrix* matrix, const char* output_file) {
    if (!matrix || !output_file) {
        fprintf(stderr, "Error (save_lower_triangular_matrix_raw): Cannot save NULL matrix or empty filename.\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    // matrix->values can be NULL if matrix is 0x0
    if (matrix->nrows > 0 && matrix->ncols > 0 && !matrix->values) {
         fprintf(stderr, "Error (save_lower_triangular_matrix_raw): Matrix has non-zero dims but NULL values.\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (matrix->nrows != matrix->ncols) {
        fprintf(stderr, "Error (save_lower_triangular_matrix_raw): Matrix must be square (dims %lldx%lld).\n",
                (long long)matrix->nrows, (long long)matrix->ncols);
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n = matrix->nrows; 

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error (save_lower_triangular_matrix_raw): Failed to open output file '%s': %s\n",
                output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    
    if (n == 0) {
        printf("Warning (save_lower_triangular_matrix_raw): Matrix is empty, saving empty file to %s.\n", output_file);
        // fprintf(fp, "\n"); // An empty file, or just close it.
        fclose(fp);
        return MORANS_I_SUCCESS;
    }


    printf("Saving lower triangular matrix (raw) to %s...\n", output_file);

    for (MKL_INT i = 0; i < n; i++) { 
        for (MKL_INT j = 0; j <= i; j++) { 
            double value = matrix->values[i * n + j]; 

            if (j > 0) { 
                fprintf(fp, "\t");
            }

            if (isnan(value)) fprintf(fp, "NaN");
            else if (isinf(value)) fprintf(fp, "%sInf", (value > 0 ? "" : "-"));
            else {
                if (fabs(value) != 0.0 && (fabs(value) < 1e-4 || fabs(value) > 1e6) ) fprintf(fp, "%.6e", value);
                else fprintf(fp, "%.8f", value);
            }
        }
        fprintf(fp, "\n"); 
    }

    fclose(fp);
    return MORANS_I_SUCCESS;
}

/* Read data from VST (variance-stabilized transformation) file */
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

    char *line = NULL; size_t line_buf_size = 0; ssize_t line_size_getline;
    MKL_INT n_genes_est = 0;
    MKL_INT n_header_fields = 0; // Total fields in header (these will be the spot IDs)
    MKL_INT n_spots_est = 0;     // Actual number of spot columns
    DenseMatrix* matrix = NULL;

    printf("Reading VST file '%s': First pass to determine dimensions...\n", filename);

    // --- First Pass: Count rows (genes) & columns (spots) ---
    line_size_getline = getline(&line, &line_buf_size, fp);
    if (line_size_getline <= 0) {
        fprintf(stderr, "Error: Empty or unreadable header in '%s'.\n", filename);
        fclose(fp); if(line) free(line);
        return NULL;
    }

    char* header_line_copy_for_counting = strdup(line);
    if (!header_line_copy_for_counting) {
        perror("strdup for header_line_copy_for_counting failed");
        fclose(fp); if(line) free(line); return NULL;
    }
    ssize_t stripped_header_len = strlen(header_line_copy_for_counting);
    while (stripped_header_len > 0 && (header_line_copy_for_counting[stripped_header_len - 1] == '\n' || header_line_copy_for_counting[stripped_header_len - 1] == '\r')) {
        header_line_copy_for_counting[--stripped_header_len] = '\0';
    }

    char* temp_ptr_header = header_line_copy_for_counting;
    char* token_count_header;
    n_header_fields = 0;
    if (stripped_header_len > 0) {
        while ((token_count_header = strsep(&temp_ptr_header, "\t")) != NULL) {
            n_header_fields++; // Counts total fields in the header
        }
    }
    free(header_line_copy_for_counting); header_line_copy_for_counting = NULL;

    // For THIS SPECIFIC FILE STRUCTURE:
    // The header fields ARE the spot IDs.
    // Data rows have one EXTRA field at the beginning (gene name).
    if (n_header_fields < 1) {
        fprintf(stderr, "Error: Header in '%s' has %lld fields. Expected at least 1 spot ID. Header: \"%s\"\n",
                filename, (long long)n_header_fields, line);
        fclose(fp); if(line) free(line); return NULL;
    }
    n_spots_est = n_header_fields; // All header fields are spot IDs


    while ((line_size_getline = getline(&line, &line_buf_size, fp)) > 0) {
        char* p_line = line; while(isspace((unsigned char)*p_line)) p_line++;
        if(*p_line != '\0') n_genes_est++;
    }
    if (n_genes_est == 0) { fprintf(stderr, "Error: No data rows (genes) found in '%s'.\n", filename); fclose(fp); free(line); return NULL;}
    printf("  First pass complete: Estimated %lld genes x %lld spots (header has %lld spot ID fields).\n",
           (long long)n_genes_est, (long long)n_spots_est, (long long)n_header_fields);

    matrix = malloc(sizeof(DenseMatrix));
    if (!matrix) { perror("malloc DenseMatrix"); fclose(fp); free(line); return NULL; }
    matrix->nrows = n_genes_est; // Genes
    matrix->ncols = n_spots_est; // Spots (e.g., 3813)
    
    // ... (allocation of matrix members as before) ...
    if (n_genes_est == 0 || n_spots_est == 0) {
        matrix->values = NULL;
        matrix->rownames = (n_genes_est > 0) ? calloc(n_genes_est, sizeof(char*)) : NULL;
        matrix->colnames = (n_spots_est > 0) ? calloc(n_spots_est, sizeof(char*)) : NULL;
        if ((n_genes_est > 0 && !matrix->rownames ) || (n_spots_est > 0 && !matrix->colnames ) ) {
             perror("Allocation for empty matrix names failed"); free_dense_matrix(matrix); fclose(fp); free(line); return NULL;
        }
    } else {
        matrix->values = mkl_malloc((size_t)n_genes_est * n_spots_est * sizeof(double), 64);
        matrix->rownames = calloc(n_genes_est, sizeof(char*));
        matrix->colnames = calloc(n_spots_est, sizeof(char*));
        if (!matrix->values || !matrix->rownames || !matrix->colnames) { perror("Allocation for matrix components"); free_dense_matrix(matrix); fclose(fp); free(line); return NULL;}
    }


    rewind(fp);
    if (line) { free(line); line = NULL; line_buf_size = 0; }
    
    line_size_getline = getline(&line, &line_buf_size, fp); // Read header line
    if (line_size_getline <= 0 && n_spots_est > 0) { /* error */ }
    ssize_t current_line_len_for_header = line_size_getline;
    while (current_line_len_for_header > 0 && (line[current_line_len_for_header - 1] == '\n' || line[current_line_len_for_header - 1] == '\r')) {
        line[--current_line_len_for_header] = '\0';
    }

    if (n_spots_est > 0 && matrix->colnames) {
        char* header_for_colnames_copy = strdup(line);
        if (!header_for_colnames_copy) { /* error */ }
        char* current_pos_header = header_for_colnames_copy;
        char* token_h;
        MKL_INT spot_idx_fill = 0;

        // DO NOT SKIP the first token from the header, as it's the first spot ID
        while ((token_h = strsep(&current_pos_header, "\t")) != NULL && spot_idx_fill < n_spots_est) {
            if (strlen(token_h) > 0) {
                matrix->colnames[spot_idx_fill] = strdup(token_h);
                if (!matrix->colnames[spot_idx_fill]) { /* error */ }
            } else {
                // Handle empty spot ID in header if necessary
                char default_spot_name[32];
                snprintf(default_spot_name, sizeof(default_spot_name), "SpotH%lld_Empty", (long long)spot_idx_fill + 1);
                fprintf(stderr, "Warning: Empty spot ID in header at column %lld. Using '%s'.\n", (long long)spot_idx_fill, default_spot_name);
                matrix->colnames[spot_idx_fill] = strdup(default_spot_name);
                 if (!matrix->colnames[spot_idx_fill]) { /* error */ }
            }
            spot_idx_fill++;
        }
        free(header_for_colnames_copy);
        if (spot_idx_fill != n_spots_est) { // This should now match if header has n_spots_est fields
            fprintf(stderr, "Error: Spot column name count mismatch. Expected %lld spot IDs from header, actually populated %lld. Header: \"%s\"\n",
                    (long long)n_spots_est, (long long)spot_idx_fill, line);
            free_dense_matrix(matrix); fclose(fp); if(line) free(line); return NULL;
        }
    }

    MKL_INT gene_idx_fill = 0; int file_lineno = 1;
    while ((line_size_getline = getline(&line, &line_buf_size, fp)) > 0) { // Start reading data rows
        file_lineno++;
        // ... (skip blank lines, trim EOL as before) ...
        char* p_line_check = line;
        while(isspace((unsigned char)*p_line_check)) p_line_check++;
        if(*p_line_check == '\0') continue;

        if (gene_idx_fill >= n_genes_est) {
            fprintf(stderr, "Warning: More data rows than estimated. Stopping at gene %lld (file line %d).\n", (long long)n_genes_est, file_lineno);
            break;
        }
        ssize_t current_data_line_len = line_size_getline;
        while (current_data_line_len > 0 && (line[current_data_line_len - 1] == '\n' || line[current_data_line_len - 1] == '\r')) {
            line[--current_data_line_len] = '\0';
        }
        if(current_data_line_len == 0) continue;


        char* data_row_copy = strdup(line);
        if (!data_row_copy) { /* error */ }
        char* current_pos_data = data_row_copy;
        char* token_d;

        token_d = strsep(&current_pos_data, "\t"); // This is the Gene name
        if (token_d == NULL) { /* error */ }
        if (n_genes_est > 0 && matrix->rownames) {
             matrix->rownames[gene_idx_fill] = strdup(token_d);
             if (!matrix->rownames[gene_idx_fill]) { /* error */ }
        }

        // Now read n_spots_est (e.g., 3813) expression values
        if (n_spots_est > 0 && matrix->values) {
            for (MKL_INT s_idx = 0; s_idx < n_spots_est; ++s_idx) {
                token_d = strsep(&current_pos_data, "\t");
                if (token_d == NULL) {
                    fprintf(stderr, "Error: File line %d, gene '%s': Expected %lld expression values, found only %lld. Row content: \"%s\"\n",
                            file_lineno, (matrix->rownames && matrix->rownames[gene_idx_fill])?matrix->rownames[gene_idx_fill]:"N/A",
                            (long long)n_spots_est, (long long)s_idx, line);
                    free(data_row_copy); free_dense_matrix(matrix); fclose(fp); if(line) free(line); return NULL;
                }
                // ... (strtod and store value as before) ...
                char* endptr; errno = 0; double val = strtod(token_d, &endptr);
                if (errno == ERANGE || (*endptr != '\0' && !isspace((unsigned char)*endptr)) || endptr == token_d) {
                    fprintf(stderr, "Error: Invalid number '%s' at file line %d, gene '%s', spot column %d.\n",
                            token_d, file_lineno, (matrix->rownames && matrix->rownames[gene_idx_fill])?matrix->rownames[gene_idx_fill]:"N/A", (int)s_idx + 1);
                    free(data_row_copy); free_dense_matrix(matrix); fclose(fp); if(line) free(line); return NULL;
                }
                matrix->values[gene_idx_fill * n_spots_est + s_idx] = val;
            }
            // Check for extra data columns (should NOT happen now if n_spots_est is correct for data content)
            if (strsep(&current_pos_data, "\t") != NULL) {
                 fprintf(stderr, "Warning: Line %d, gene '%s': Data row has more columns than expected (%lld spots). THIS SHOULD NOT HAPPEN WITH NEW LOGIC. Extra data ignored.\n",
                         file_lineno, (matrix->rownames && matrix->rownames[gene_idx_fill])?matrix->rownames[gene_idx_fill]:"N/A",
                         (long long)n_spots_est);
            }
        }
        free(data_row_copy);
        gene_idx_fill++;
    }

    // ... (gene count adjustment and final print as before, using matrix->nrows which is n_genes_est) ...
    if (gene_idx_fill != n_genes_est) {
        fprintf(stderr, "Warning: Read %lld genes, but estimated %lld from first pass. Adjusting matrix dimensions.\n",
                (long long)gene_idx_fill, (long long)n_genes_est);
        if (gene_idx_fill < n_genes_est) {
            fprintf(stderr, "Adjusting number of genes read from %lld to %lld.\n", (long long)n_genes_est, (long long)gene_idx_fill);
            matrix->nrows = gene_idx_fill; 
            n_genes_est = gene_idx_fill; 
            if (n_genes_est == 0) {
                 fprintf(stderr, "Error: No valid gene data rows were actually processed.\n");
                 free_dense_matrix(matrix); fclose(fp); if(line) free(line); return NULL;
            }
        } else { 
             fprintf(stderr, "Error: Internal logic error, gene_idx_fill > n_genes_est.\n");
             free_dense_matrix(matrix); fclose(fp); if(line) free(line); return NULL;
        }
    }

    if(line) free(line);
    fclose(fp);
    printf("Successfully loaded VST data: %lld genes x %lld spots from '%s'. (Stored internally as Genes x Spots)\n",
           (long long)matrix->nrows, (long long)matrix->ncols, filename);
    return matrix;
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

/* Free dense matrix */
void free_dense_matrix(DenseMatrix* matrix) {
    if (!matrix) return;
    if (matrix->values) mkl_free(matrix->values);
    if (matrix->rownames) {
        for (MKL_INT i = 0; i < matrix->nrows; i++) if(matrix->rownames[i]) free(matrix->rownames[i]);
        free(matrix->rownames);
    }
    if (matrix->colnames) {
        for (MKL_INT i = 0; i < matrix->ncols; i++) if(matrix->colnames[i]) free(matrix->colnames[i]);
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
        for (MKL_INT i = 0; i < matrix->nrows; i++) if(matrix->rownames[i]) free(matrix->rownames[i]);
        free(matrix->rownames);
    }
    if (matrix->colnames) { 
        for (MKL_INT i = 0; i < matrix->ncols; i++) if(matrix->colnames[i]) free(matrix->colnames[i]);
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
        for (MKL_INT i = 0; i < coords->total_spots; i++) if(coords->spot_names[i]) free(coords->spot_names[i]);
        free(coords->spot_names);
    }
    free(coords);
}

/* Helper to print MKL sparse status */
void print_mkl_status(sparse_status_t status, const char* function_name) {
    if (status == SPARSE_STATUS_SUCCESS) return;
    fprintf(stderr, "MKL Sparse BLAS Error: Function '%s' failed with status: ", function_name);
    switch(status) {
        case SPARSE_STATUS_NOT_INITIALIZED: fprintf(stderr, "SPARSE_STATUS_NOT_INITIALIZED\n"); break;
        case SPARSE_STATUS_ALLOC_FAILED:    fprintf(stderr, "SPARSE_STATUS_ALLOC_FAILED\n"); break;
        case SPARSE_STATUS_INVALID_VALUE:   fprintf(stderr, "SPARSE_STATUS_INVALID_VALUE\n"); break;
        case SPARSE_STATUS_EXECUTION_FAILED:fprintf(stderr, "SPARSE_STATUS_EXECUTION_FAILED\n"); break;
        case SPARSE_STATUS_INTERNAL_ERROR:  fprintf(stderr, "SPARSE_STATUS_INTERNAL_ERROR\n"); break;
        case SPARSE_STATUS_NOT_SUPPORTED:   fprintf(stderr, "SPARSE_STATUS_NOT_SUPPORTED\n"); break;
        default: fprintf(stderr, "Unknown MKL Sparse Status Code (%d)\n", status); break;
    }
}

/* Parse numeric parameters safely */
int load_positive_value(const char* value_str, const char* param, unsigned int min_val, unsigned int max_val) {
    if (!value_str || !param) { fprintf(stderr, "Error: Null string or param name in load_positive_value\n"); return -1; }
    char* endptr; errno = 0;
    long value = strtol(value_str, &endptr, 10);
    if (errno == ERANGE || *endptr != '\0' || value_str == endptr) { fprintf(stderr, "Error: Parameter %s ('%s') is not a valid integer.\n", param, value_str); return -1;}
    if (value < (long)min_val || value > (long)max_val) { fprintf(stderr, "Error: Parameter %s value %ld out of range [%u, %u].\n", param, value, min_val, max_val); return -1;}
    return (int)value;
}

/* Parse double parameters safely */
double load_double_value(const char* value_str, const char* param) {
    if (!value_str || !param) { fprintf(stderr, "Error: Null string or param name in load_double_value\n"); return NAN;}
    char* endptr; errno = 0; double value = strtod(value_str, &endptr);
    if (errno == ERANGE || *endptr != '\0' || value_str == endptr) { fprintf(stderr, "Error: Parameter %s ('%s') is not a valid double.\n", param, value_str); return NAN;}
    // Do not check for !isfinite here, as Inf might be a valid input for some parameters (though unlikely for user params)
    // However, for most numeric inputs from users, non-finite is an error.
    // Let's assume for now that if strtod succeeds, it's "valid" unless downstream logic checks isfinite.
    // For sigma, coord_scale etc., non-finite is bad. Added check back.
     if (!isfinite(value)) { fprintf(stderr, "Error: Parameter %s ('%s') resulted in non-finite double value.\n", param, value_str); return NAN;}
    return value;
}

/* Infer sigma from data for single-cell datasets */
double infer_sigma_from_data(const SpotCoordinates* coords, double coord_scale) {
    if (!coords || !coords->spot_row || !coords->spot_col || !coords->valid_mask) {
        fprintf(stderr, "Error: Null or invalid coordinates in infer_sigma_from_data\n"); return 100.0; // Default sigma
    }
    if (coords->valid_spots < 2) { fprintf(stderr, "Warning: Not enough valid spots (%lld) to infer sigma, using default 100.0.\n", (long long)coords->valid_spots); return 100.0; }

    double sum_min_dist_sq = 0.0;
    MKL_INT count_valid_nn = 0; // Using MKL_INT for counter that might be large
    int max_samples_for_sigma = 1000; // Sample up to this many spots to estimate NN distances
    MKL_INT sample_step = (coords->valid_spots > max_samples_for_sigma) ? (coords->valid_spots / max_samples_for_sigma) : 1;
    if(sample_step == 0) sample_step = 1; // Ensure step is at least 1

    printf("Inferring sigma: sampling up to %d spots (step %lld) from %lld valid spots...\n", max_samples_for_sigma, (long long)sample_step, (long long)coords->valid_spots);

    // Create temporary lists of valid coordinates in physical units
    // No, this is inefficient. Iterate through valid_mask and use original coords with scale.
    // But for parallel processing, having separate lists of just the valid ones is better.
    double* valid_x_physical_list = (double*)malloc((size_t)coords->valid_spots * sizeof(double));
    double* valid_y_physical_list = (double*)malloc((size_t)coords->valid_spots * sizeof(double));
    if (!valid_x_physical_list || !valid_y_physical_list) {
        perror("Failed to allocate lists for sigma inference");
        if(valid_x_physical_list) free(valid_x_physical_list); 
        if(valid_y_physical_list) free(valid_y_physical_list);
        fprintf(stderr, "Warning: Memory error during sigma inference setup, using default 100.0.\n"); return 100.0;
    }
    MKL_INT current_valid_idx = 0;
    for (MKL_INT i = 0; i < coords->total_spots; ++i) {
        if (coords->valid_mask[i]) {
            if (current_valid_idx < coords->valid_spots) { // Boundary check
                valid_x_physical_list[current_valid_idx] = (double)coords->spot_col[i] / coord_scale;
                valid_y_physical_list[current_valid_idx] = (double)coords->spot_row[i] / coord_scale;
                current_valid_idx++;
            } else { // Should not happen if coords->valid_spots is correct
                fprintf(stderr, "Warning: Exceeded valid_spots count (%lld) during sigma inference list population. current_valid_idx=%lld\n", (long long)coords->valid_spots, (long long)current_valid_idx);
                break; 
            }
        }
    }
    // If current_valid_idx is less than coords->valid_spots, it means valid_spots was overestimated.
    // Use current_valid_idx as the true number of valid points populated.
    MKL_INT actual_valid_for_inference = current_valid_idx; 
    if(actual_valid_for_inference < 2) {
        fprintf(stderr, "Warning: Less than 2 actual valid spots for sigma inference after filtering/populating list. Using default 100.0.\n");
        free(valid_x_physical_list); free(valid_y_physical_list);
        return 100.0;
    }


    #pragma omp parallel for reduction(+:sum_min_dist_sq) reduction(+:count_valid_nn) schedule(dynamic)
    for (MKL_INT i_sampled_idx = 0; i_sampled_idx < actual_valid_for_inference; i_sampled_idx += sample_step) {
        double x_i = valid_x_physical_list[i_sampled_idx]; 
        double y_i = valid_y_physical_list[i_sampled_idx];
        double min_dist_sq_local = DBL_MAX;

        for (MKL_INT j_nn_search_idx = 0; j_nn_search_idx < actual_valid_for_inference; j_nn_search_idx++) {
            if (i_sampled_idx == j_nn_search_idx) continue; // Don't compare to self
            
            double x_j = valid_x_physical_list[j_nn_search_idx]; 
            double y_j = valid_y_physical_list[j_nn_search_idx];
            double dx = x_i - x_j; 
            double dy = y_i - y_j; 
            double dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq_local) {
                min_dist_sq_local = dist_sq;
            }
        }
        if (min_dist_sq_local < DBL_MAX && min_dist_sq_local > 0) { // Ensure a valid NN was found (and not zero distance to self if logic was off)
            sum_min_dist_sq += min_dist_sq_local; 
            count_valid_nn++;
        }
    }
    free(valid_x_physical_list); free(valid_y_physical_list);

    if (count_valid_nn == 0) { fprintf(stderr, "Warning: Could not calculate any nearest neighbor distances for sigma inference (count_valid_nn=0), using default 100.0.\n"); return 100.0;}
    
    double avg_min_dist_sq = sum_min_dist_sq / count_valid_nn;
    double inferred_sigma = sqrt(avg_min_dist_sq); 
    
    printf("Inferred sigma = %.4f based on average nearest neighbor distance from %lld samples.\n", inferred_sigma, (long long)count_valid_nn);
    return (inferred_sigma > ZERO_STD_THRESHOLD) ? inferred_sigma : 100.0; // Ensure sigma is positive, else default
}

/* Gaussian distance decay function */
double decay(double d_physical, double sigma) {
    if (d_physical < 0.0) d_physical = 0.0; 

    if (sigma <= ZERO_STD_THRESHOLD) { 
        return (fabs(d_physical) < ZERO_STD_THRESHOLD) ? 1.0 : 0.0;
    }
    if (d_physical > 3.0 * sigma) { // Cutoff beyond 3 sigma
        return 0.0;
    }
    return exp(-(d_physical * d_physical) / (2.0 * sigma * sigma));
}


/* Create spatial distance matrix (decay lookup table) */
DenseMatrix* create_distance_matrix(MKL_INT max_radius_grid_units, 
                                   int platform_mode,
                                   double custom_sigma_physical, 
                                   double coord_scale_for_sc) { 
    if (max_radius_grid_units <= 0) {
        fprintf(stderr, "Error: max_radius_grid_units must be positive in create_distance_matrix. Got %lld.\n", (long long)max_radius_grid_units);
        return NULL;
    }

    DenseMatrix* matrix = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!matrix) {
        perror("Failed to allocate DenseMatrix structure for decay matrix");
        return NULL;
    }
    matrix->nrows = max_radius_grid_units;      
    matrix->ncols = 2 * max_radius_grid_units;  // Allows for dx up to max_radius_grid_units, dy up to 2*max_radius_grid_units-1 (or vice-versa)
                                                // The original "correct" code had ncols = 2 * max_radius.
                                                // This seems to imply that col_shift_abs can go up to 2*max_radius-1.
                                                // Let's match the original structure where max_radius implies a square search window.
                                                // The distance_matrix stores weights for (row_shift_abs, col_shift_abs)
                                                // Max row_shift_abs = max_radius_grid_units - 1
                                                // Max col_shift_abs = max_radius_grid_units - 1
                                                // So distance_matrix needs to be max_radius_grid_units x max_radius_grid_units.
                                                // The `build_spatial_weight_matrix` uses:
                                                // if (row_shift_abs < distance_matrix->nrows && col_shift_abs < distance_matrix->ncols)
                                                // Let's stick to the provided dimensions from the original code.
                                                //nrows = max_radius_param; ncols = 2 * max_radius_param;

    matrix->rownames = NULL;
    matrix->colnames = NULL;
    matrix->values = (double*)mkl_malloc((size_t)matrix->nrows * matrix->ncols * sizeof(double), 64);
    if (!matrix->values) {
        perror("Failed to allocate values for decay matrix");
        free(matrix);
        return NULL;
    }

    double sigma_eff; 
    if (custom_sigma_physical > 0.0) {
        sigma_eff = custom_sigma_physical;
    } else { // Default sigmas if not provided or inferred
        if (platform_mode == VISIUM)           sigma_eff = 100.0; // physical units (e.g. um)
        else if (platform_mode == OLD)         sigma_eff = 200.0; // physical units
        else if (platform_mode == SINGLE_CELL) sigma_eff = 100.0; // Default if inference fails or not run
        else                                   sigma_eff = 100.0; // Fallback
    }

    printf("Distance Matrix Lookup Table: Dimensions %lldx%lld. Platform: %d. Effective sigma (physical units) = %.4f.\n",
           (long long)matrix->nrows, (long long)matrix->ncols, platform_mode, sigma_eff);

    if (platform_mode == VISIUM || platform_mode == OLD) {
        double shift_factor_y_grid = (platform_mode == VISIUM) ? (0.5 * sqrt(3.0)) : 0.5; // Conversion for y grid shift to physical distance component
        double dist_unit_physical  = (platform_mode == VISIUM) ? 100.0 : 200.0; // Physical distance per unit grid col shift

        #pragma omp parallel for collapse(2) schedule(static)
        for (MKL_INT r_shift_grid = 0; r_shift_grid < matrix->nrows; r_shift_grid++) { // row_shift_abs in grid units
            for (MKL_INT c_shift_grid = 0; c_shift_grid < matrix->ncols; c_shift_grid++) { // col_shift_abs in grid units
                // Convert grid shifts to physical distance components
                double x_dist_physical = 0.5 * (double)c_shift_grid * dist_unit_physical; 
                double y_dist_physical = (double)r_shift_grid * shift_factor_y_grid * dist_unit_physical;
                
                double d_physical_total = sqrt(x_dist_physical * x_dist_physical + y_dist_physical * y_dist_physical);
                matrix->values[r_shift_grid * matrix->ncols + c_shift_grid] = decay(d_physical_total, sigma_eff);
            }
        }
    } else if (platform_mode == SINGLE_CELL) {
        if (coord_scale_for_sc <= ZERO_STD_THRESHOLD) {
            fprintf(stderr, "Error: Invalid coord_scale_for_sc (%.4f) for SINGLE_CELL in create_distance_matrix.\n", coord_scale_for_sc);
             // Fill with 0, except 0,0?
            #pragma omp parallel for collapse(2) schedule(static)
            for (MKL_INT r_shift_grid = 0; r_shift_grid < matrix->nrows; r_shift_grid++) {
                for (MKL_INT c_shift_grid = 0; c_shift_grid < matrix->ncols; c_shift_grid++) {
                    matrix->values[r_shift_grid * matrix->ncols + c_shift_grid] = (r_shift_grid==0 && c_shift_grid==0) ? 1.0: 0.0;
                }
            }
        } else {
            printf("  (For SC, physical distances for lookup table are from grid shifts / coord_scale: %.4f)\n", coord_scale_for_sc);
            #pragma omp parallel for collapse(2) schedule(static)
            for (MKL_INT r_shift_grid = 0; r_shift_grid < matrix->nrows; r_shift_grid++) { 
                for (MKL_INT c_shift_grid = 0; c_shift_grid < matrix->ncols; c_shift_grid++) { 
                    // These are grid shifts. Convert to physical shifts.
                    double r_physical_component = (double)r_shift_grid / coord_scale_for_sc;
                    double c_physical_component = (double)c_shift_grid / coord_scale_for_sc;
                    double d_physical_total = sqrt(c_physical_component * c_physical_component + r_physical_component * r_physical_component);
                    matrix->values[r_shift_grid * matrix->ncols + c_shift_grid] = decay(d_physical_total, sigma_eff);
                }
            }
        }
    } else {
        fprintf(stderr, "Error: Unknown platform_mode %d in create_distance_matrix. Cannot create decay lookup table.\n", platform_mode);
        free_dense_matrix(matrix);
        return NULL;
    }

    printf("Distance matrix (decay lookup table) created.\n");
    return matrix;
}


/* Extract coordinates from column names (spot names like "RxC") */
SpotCoordinates* extract_coordinates(char** column_names, MKL_INT n_columns) {
    if (!column_names && n_columns > 0) { // If n_columns is 0, column_names can be NULL
        fprintf(stderr, "Error: Null column_names with n_columns > 0 for extract_coordinates\n");
        return NULL;
    }
     if (n_columns < 0) {
        fprintf(stderr, "Error: Negative n_columns for extract_coordinates\n");
        return NULL;
    }


    SpotCoordinates* coords = (SpotCoordinates*)malloc(sizeof(SpotCoordinates));
    if (!coords) { perror("malloc SpotCoordinates"); return NULL; }

    coords->total_spots = n_columns;
    coords->spot_row = NULL; coords->spot_col = NULL; 
    coords->valid_mask = NULL; coords->spot_names = NULL;
    coords->valid_spots = 0;

    if (n_columns > 0) {
        coords->spot_row = (MKL_INT*)malloc((size_t)n_columns * sizeof(MKL_INT));
        coords->spot_col = (MKL_INT*)malloc((size_t)n_columns * sizeof(MKL_INT));
        coords->valid_mask = (int*)calloc(n_columns, sizeof(int)); // Initialize to 0 (invalid)
        coords->spot_names = (char**)malloc((size_t)n_columns * sizeof(char*));
        if (!coords->spot_row || !coords->spot_col || !coords->valid_mask || !coords->spot_names) {
            perror("Failed to allocate memory for coordinate arrays in extract_coordinates");
            free_spot_coordinates(coords); return NULL;
        }
    } else { // n_columns == 0
        return coords; // Return empty but valid SpotCoordinates struct
    }


    regex_t regex;
    int reti = regcomp(&regex, "^([0-9]+)x([0-9]+)$", REG_EXTENDED);
    if (reti) {
        char errbuf[100]; regerror(reti, &regex, errbuf, sizeof(errbuf));
        fprintf(stderr, "Error: Could not compile regex for coordinate extraction: %s\n", errbuf);
        free_spot_coordinates(coords); return NULL;
    }

    printf("Extracting coordinates from %lld column names using regex ^([0-9]+)x([0-9]+)$ ...\n", (long long)n_columns);
    MKL_INT current_valid_count = 0;
    for (MKL_INT i = 0; i < n_columns; i++) {
        coords->spot_names[i] = NULL; // Initialize to NULL
        coords->spot_row[i] = -1; coords->spot_col[i] = -1; // Default to invalid

        if (!column_names[i]) { // Handle NULL spot name in input
            fprintf(stderr, "Warning: Spot name at index %lld is NULL. Treating as invalid.\n", (long long)i);
            coords->valid_mask[i] = 0;
            continue;
        }
        coords->spot_names[i] = strdup(column_names[i]); // strdup after checking column_names[i]
        if (!coords->spot_names[i]) { perror("strdup spot_name in extract_coordinates"); regfree(&regex); free_spot_coordinates(coords); return NULL;}
        

        regmatch_t matches[3]; // 0: whole match, 1: first group, 2: second group
        reti = regexec(&regex, column_names[i], 3, matches, 0);
        if (reti == 0) { // Match found
            char val_str_buffer[64]; // Buffer for string part of coordinate
            
            // Extract row (first group)
            regoff_t len_row = matches[1].rm_eo - matches[1].rm_so;
            if (len_row > 0 && len_row < (regoff_t)sizeof(val_str_buffer)) {
                strncpy(val_str_buffer, column_names[i] + matches[1].rm_so, len_row);
                val_str_buffer[len_row] = '\0';
                coords->spot_row[i] = strtoll(val_str_buffer, NULL, 10); // Use strtoll for MKL_INT
            } else { coords->spot_row[i] = -1; } // Mark as invalid if parse fails or too long

            // Extract col (second group)
            regoff_t len_col = matches[2].rm_eo - matches[2].rm_so;
             if (len_col > 0 && len_col < (regoff_t)sizeof(val_str_buffer)) {
                strncpy(val_str_buffer, column_names[i] + matches[2].rm_so, len_col);
                val_str_buffer[len_col] = '\0';
                coords->spot_col[i] = strtoll(val_str_buffer, NULL, 10);
            } else { coords->spot_col[i] = -1; }

            if (coords->spot_row[i] >= 0 && coords->spot_col[i] >= 0) { // Assuming 0-based or positive coords are valid
                coords->valid_mask[i] = 1; 
                current_valid_count++;
            } else { 
                coords->valid_mask[i] = 0; // Explicitly mark as invalid
                coords->spot_row[i] = -1; coords->spot_col[i] = -1; // Reset if one part was bad
            }
        } else if (reti == REG_NOMATCH) {
            coords->valid_mask[i] = 0; // No match, invalid coordinate format
            // fprintf(stderr, "Warning: Coordinate format mismatch for '%s'. Treating as invalid.\n", column_names[i]);
        } else { // Other regex error
            char errbuf[100]; regerror(reti, &regex, errbuf, sizeof(errbuf));
            fprintf(stderr, "Warning: Regex match failed for '%s': %s. Treating as invalid.\n", column_names[i], errbuf);
            coords->valid_mask[i] = 0;
        }
    }
    regfree(&regex); 
    coords->valid_spots = current_valid_count;
    printf("Coordinate extraction complete. Found %lld valid coordinates (format 'RxC') out of %lld total spots.\n", (long long)coords->valid_spots, (long long)n_columns);
    return coords;
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
    if (!fp) { fprintf(stderr, "Error: Failed to open coordinates file '%s': %s\n", filename, strerror(errno)); return NULL;}

    char *line = NULL; size_t line_buf_size = 0; ssize_t line_len;
    MKL_INT num_data_lines_est = 0; 
    int id_col_idx = -1, x_col_idx = -1, y_col_idx = -1;
    int header_field_count = 0;

    // --- Parse Header ---
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) { fprintf(stderr, "Error: Empty or unreadable header in coordinates file '%s'.\n", filename); fclose(fp); if(line) free(line); return NULL;}
    // Trim EOL characters
    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) line[--line_len] = '\0';

    char* header_copy = strdup(line); // For strtok
    if(!header_copy) {perror("strdup header for coord file"); fclose(fp); free(line); return NULL;}
    
    char* token_h = strtok(header_copy, "\t"); // Use tab as primary delimiter
    int current_col_idx_parse = 0;
    while (token_h) {
        // Trim whitespace and quotes from token_h
        char* trimmed_token = token_h;
        while(isspace((unsigned char)*trimmed_token)) trimmed_token++;
        size_t tok_len = strlen(trimmed_token);
        while(tok_len > 0 && isspace((unsigned char)trimmed_token[tok_len-1])) trimmed_token[--tok_len] = '\0';
        if(tok_len > 1 && *trimmed_token == '"' && trimmed_token[tok_len-1] == '"') {
            trimmed_token[tok_len-1] = '\0'; 
            trimmed_token++;
        }
        
        if (strcmp(trimmed_token, id_column_name) == 0) id_col_idx = current_col_idx_parse;
        if (strcmp(trimmed_token, x_column_name) == 0) x_col_idx = current_col_idx_parse;
        if (strcmp(trimmed_token, y_column_name) == 0) y_col_idx = current_col_idx_parse;
        
        token_h = strtok(NULL, "\t"); 
        current_col_idx_parse++;
        header_field_count++;
    }
    free(header_copy);
    if (id_col_idx == -1 || x_col_idx == -1 || y_col_idx == -1) {
        fprintf(stderr, "Error: Required columns not found in coordinates file '%s'.\n  ID ('%s'): %s, X ('%s'): %s, Y ('%s'): %s\n", 
                filename, id_column_name, (id_col_idx!=-1?"Found":"Missing"), 
                x_column_name, (x_col_idx!=-1?"Found":"Missing"), 
                y_column_name, (y_col_idx!=-1?"Found":"Missing"));
        fclose(fp); if(line) free(line); return NULL;
    }

    // --- Count Data Lines (First Pass for Allocation Size) ---
    long current_pos = ftell(fp); // Save current position after header
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) { 
        char* p = line; while(isspace((unsigned char)*p)) p++; 
        if(*p != '\0') num_data_lines_est++; // Count non-empty lines
    }
    fseek(fp, current_pos, SEEK_SET); // Rewind to after header for data reading

    if (num_data_lines_est == 0) { fprintf(stderr, "Error: No data lines found in coordinates file '%s'.\n", filename); fclose(fp); if(line) free(line); return NULL;}

    SpotCoordinates* coords = (SpotCoordinates*)malloc(sizeof(SpotCoordinates));
    if (!coords) { perror("malloc SpotCoordinates for file read"); fclose(fp); if(line) free(line); return NULL; }
    coords->total_spots = num_data_lines_est; // Store estimated count
    coords->spot_row = (MKL_INT*)malloc((size_t)num_data_lines_est * sizeof(MKL_INT));
    coords->spot_col = (MKL_INT*)malloc((size_t)num_data_lines_est * sizeof(MKL_INT));
    coords->valid_mask = (int*)calloc(num_data_lines_est, sizeof(int)); // Init to 0
    coords->spot_names = (char**)malloc((size_t)num_data_lines_est * sizeof(char*));
    coords->valid_spots = 0; // Will be incremented
    if (!coords->spot_row || !coords->spot_col || !coords->valid_mask || !coords->spot_names) { 
        perror("Failed to allocate arrays for coordinates read from file"); 
        free_spot_coordinates(coords); fclose(fp); if(line) free(line); return NULL;
    }


    // --- Read Data Lines (Second Pass) ---
    MKL_INT spot_fill_idx = 0; 
    int data_file_lineno = 1; // Line number in file (header was 1)
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && spot_fill_idx < num_data_lines_est) {
        data_file_lineno++;
        char* p_chk = line; while(isspace((unsigned char)*p_chk)) p_chk++; 
        if(*p_chk == '\0') continue; // Skip truly empty lines

        // Trim EOL
        ssize_t current_data_len = line_len;
        while (current_data_len > 0 && (line[current_data_len - 1] == '\n' || line[current_data_len - 1] == '\r')) line[--current_data_len] = '\0';
        if (current_data_len == 0) continue; // Skip if line becomes empty after stripping EOL

        char* data_line_copy = strdup(line); // For strtok
        if(!data_line_copy) { perror("strdup data line for coord file"); break; } // Exit loop on critical alloc error
        
        char* current_field_ptr = data_line_copy;
        char* field_tokens[header_field_count > 0 ? header_field_count : 1]; // Max tokens = num header fields
        int actual_tokens_in_row = 0;
        for(int k=0; k < header_field_count; ++k) {
            field_tokens[k] = strsep(&current_field_ptr, "\t");
            if(field_tokens[k] == NULL) break; // No more tokens
            actual_tokens_in_row++;
        }


        char* id_str_val = NULL; double x_val = NAN, y_val = NAN;
        
        if (id_col_idx < actual_tokens_in_row && field_tokens[id_col_idx] != NULL) id_str_val = field_tokens[id_col_idx];
        if (x_col_idx < actual_tokens_in_row && field_tokens[x_col_idx] != NULL) { char* end_x; x_val = strtod(field_tokens[x_col_idx], &end_x); if (end_x == field_tokens[x_col_idx]) x_val = NAN; } // Check for conversion failure
        if (y_col_idx < actual_tokens_in_row && field_tokens[y_col_idx] != NULL) { char* end_y; y_val = strtod(field_tokens[y_col_idx], &end_y); if (end_y == field_tokens[y_col_idx]) y_val = NAN; }


        if (id_str_val && strlen(id_str_val) > 0 && isfinite(x_val) && isfinite(y_val)) {
            coords->spot_names[spot_fill_idx] = strdup(id_str_val);
            if (!coords->spot_names[spot_fill_idx]) { perror("strdup spot_id in read_coord_file"); free(data_line_copy); break;}
            coords->spot_row[spot_fill_idx] = (MKL_INT)round(y_val * coord_scale);
            coords->spot_col[spot_fill_idx] = (MKL_INT)round(x_val * coord_scale);
            coords->valid_mask[spot_fill_idx] = 1; 
            coords->valid_spots++;
        } else {
            // Store a placeholder for invalid entries if necessary, or just skip
            coords->spot_names[spot_fill_idx] = strdup("INVALID_COORD_ENTRY"); // So spot_names array is full
            if(!coords->spot_names[spot_fill_idx]) {perror("strdup INVALID_COORD_ENTRY"); free(data_line_copy); break;}
            coords->spot_row[spot_fill_idx] = -1; 
            coords->spot_col[spot_fill_idx] = -1;
            coords->valid_mask[spot_fill_idx] = 0;
            fprintf(stderr, "Warning: Invalid or incomplete data on line %d of coord file '%s'. ID:'%s' X:'%s' Y:'%s'. Parsed X:%.2f, Y:%.2f. Skipping entry.\n", 
                    data_file_lineno, filename,
                    (id_col_idx < actual_tokens_in_row && field_tokens[id_col_idx]) ? field_tokens[id_col_idx] : "MISSING_ID_FIELD",
                    (x_col_idx < actual_tokens_in_row && field_tokens[x_col_idx]) ? field_tokens[x_col_idx] : "MISSING_X_FIELD",
                    (y_col_idx < actual_tokens_in_row && field_tokens[y_col_idx]) ? field_tokens[y_col_idx] : "MISSING_Y_FIELD",
                    x_val, y_val);
        }
        free(data_line_copy);
        spot_fill_idx++;
    }
    
    // If spot_fill_idx < num_data_lines_est, it means some lines were skipped or loop broke.
    // Adjust total_spots to actual number processed if different.
    if (spot_fill_idx != coords->total_spots) {
        fprintf(stderr, "Warning: Estimated %lld data lines but processed %lld for coordinates. total_spots adjusted.\n",
                (long long)coords->total_spots, (long long)spot_fill_idx);
        coords->total_spots = spot_fill_idx; // Update total_spots to reflect actual entries processed
        // Potentially realloc arrays down to size spot_fill_idx if significant difference.
    }

    if(line) free(line); 
    fclose(fp);
    printf("Read %lld coordinate entries from file '%s', %lld are valid after processing and scaling.\n", (long long)coords->total_spots, filename, (long long)coords->valid_spots);
    if (coords->valid_spots == 0 && coords->total_spots > 0) {
        fprintf(stderr, "Critical Warning: No valid coordinates were parsed from the coordinate file. Please check file format and column names (--id-col, --x-col, --y-col).\n");
    }
    return coords;
}

/* Map expression matrix columns (spot IDs) to coordinate spots (by spot_names) */
int map_expression_to_coordinates(const DenseMatrix* expr_matrix, const SpotCoordinates* coords,
                                 MKL_INT** mapping_out, MKL_INT* num_mapped_spots_out) {
    // ... (implementation as provided in the previous good version) ...
    if (!expr_matrix || !coords || !mapping_out || !num_mapped_spots_out) {
        fprintf(stderr, "Error: Invalid parameters to map_expression_to_coordinates (NULL pointers).\n");
        if (num_mapped_spots_out) *num_mapped_spots_out = 0;
        if (mapping_out) *mapping_out = NULL;
        return MORANS_I_ERROR_PARAMETER;
    }
    if (expr_matrix->ncols > 0 && !expr_matrix->colnames) {
         fprintf(stderr, "Error: expr_matrix has columns but NULL colnames in map_expression_to_coordinates.\n");
        if (num_mapped_spots_out) *num_mapped_spots_out = 0; *mapping_out = NULL;
        return MORANS_I_ERROR_PARAMETER;
    }
    if (coords->total_spots > 0 && (!coords->spot_names || !coords->valid_mask) ) {
         fprintf(stderr, "Error: coords has spots but NULL spot_names or valid_mask in map_expression_to_coordinates.\n");
        if (num_mapped_spots_out) *num_mapped_spots_out = 0; *mapping_out = NULL;
        return MORANS_I_ERROR_PARAMETER;
    }

    *num_mapped_spots_out = 0;
    *mapping_out = NULL;

    if (coords->valid_spots == 0) {
        printf("Warning: No valid spots in SpotCoordinates data to map to expression matrix.\n");
        return MORANS_I_SUCCESS;
    }
    MKL_INT* mapping_array = (MKL_INT*)malloc((size_t)coords->valid_spots * sizeof(MKL_INT));
    if (!mapping_array) { 
        perror("Failed to allocate mapping array in map_expression_to_coordinates"); 
        return MORANS_I_ERROR_MEMORY;
    }
    for(MKL_INT i=0; i<coords->valid_spots; ++i) mapping_array[i] = -1; 

    MKL_INT mapped_count = 0; 
    MKL_INT current_valid_coord_processed_idx = 0; 

    for (MKL_INT i_coord = 0; i_coord < coords->total_spots; i_coord++) {
        if (!coords->valid_mask[i_coord]) continue; 

        if (!coords->spot_names[i_coord]) { 
            fprintf(stderr, "Warning: Valid coordinate spot at index %lld has NULL name. Skipping.\n", (long long)i_coord);
            current_valid_coord_processed_idx++;
            continue;
        }

        // int found_match_in_expr = 0; // REMOVE THIS LINE

        for (MKL_INT j_expr = 0; j_expr < expr_matrix->ncols; j_expr++) {
            if (!expr_matrix->colnames[j_expr]) { 
                 fprintf(stderr, "Warning: Expression matrix column %lld has NULL name. Cannot match.\n", (long long)j_expr);
                continue;
            }
            if (strcmp(coords->spot_names[i_coord], expr_matrix->colnames[j_expr]) == 0) {
                if (current_valid_coord_processed_idx < coords->valid_spots) { 
                    mapping_array[current_valid_coord_processed_idx] = j_expr; 
                    mapped_count++;
                } else {
                     fprintf(stderr, "Error: current_valid_coord_processed_idx overflow in mapping. This should not happen.\n");
                }
                break; 
            }
        }
        current_valid_coord_processed_idx++; 
    }
    
    *mapping_out = mapping_array;
    *num_mapped_spots_out = mapped_count;

    printf("Attempted to map %lld valid coordinate spots. Successfully mapped %lld to expression data columns.\n", 
           (long long)coords->valid_spots, (long long)mapped_count);
    
    if (mapped_count < coords->valid_spots) {
        fprintf(stderr, "Warning: %lld valid coordinate spots were NOT found in the expression matrix column names.\n", 
                (long long)(coords->valid_spots - mapped_count));
    }
    if (mapped_count == 0 && coords->valid_spots > 0) {
        fprintf(stderr, "Critical Error: No coordinate spot IDs could be matched to expression matrix spot IDs.\n"
                        "Please ensure spot/cell IDs are consistent between the expression matrix header and the coordinate file/spot name format.\n");
    }
    return MORANS_I_SUCCESS;
}

