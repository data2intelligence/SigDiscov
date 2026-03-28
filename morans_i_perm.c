/*
 * morans_i_perm.c
 *
 * Permutation testing module for Moran's I computation.
 * Contains standard permutation test implementation including the
 * parallel permutation worker and the main run_permutation_test function.
 *
 * Split from morans_i_mkl.c
 */

#include "morans_i_internal.h"

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
        scaling_factor = 1.0 / (double)n_spots;  // Correct scaling
        printf("  Permutation Test: Using row-normalized weights (scaling factor = 1.0 / n_spots)\n");
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
