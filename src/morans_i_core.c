/*
 * morans_i_core.c
 *
 * Core Moran's I calculation module.
 *
 * Contains z-normalization, pairwise Moran's I matrix computation,
 * single-gene Moran's I, first-gene-vs-all, and batch calculation functions.
 *
 * Split from morans_i_mkl.c for modularity.
 */

#include "morans_i_internal.h"

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
    if (data_matrix->rownames) {
        if (copy_string_array(normalized->rownames, (const char**)data_matrix->rownames, n_genes) != MORANS_I_SUCCESS) {
            free_dense_matrix(normalized);
            return NULL;
        }
    }

    if (data_matrix->colnames) {
        if (copy_string_array(normalized->colnames, (const char**)data_matrix->colnames, n_spots) != MORANS_I_SUCCESS) {
            free_dense_matrix(normalized);
            return NULL;
        }
    }

    // Perform normalization
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (MKL_INT i = 0; i < n_genes; i++) {
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
                double inv_std = 1.0 / std_dev;
                for (MKL_INT j = 0; j < n_spots; j++) {
                    if (isfinite(current_gene_row_input[j])) {
                        current_gene_row_output[j] = (current_gene_row_input[j] - mean) * inv_std;
                    } else {
                        current_gene_row_output[j] = 0.0;
                    }
                }
            }
        }
    }

    printf("Z-normalization complete.\n");
    DEBUG_MATRIX_INFO(normalized, "z_normalized");
    return normalized;
}

/* ===============================
 * MORAN'S I CALCULATION FUNCTIONS
 * =============================== */

/* Shared inner function: compute pairwise Moran's I matrix with a pre-computed scaling factor.
 * Result = (X^T * W * X) * scaling_factor
 * The 'label' parameter is used for printf messages (e.g., "standard" or "residual"). */
DenseMatrix* compute_pairwise_morans_i_scaled(const DenseMatrix* X, const SparseMatrix* W,
                                               double scaling_factor, const char* label) {
    if (!X || !W || !X->values) {
        fprintf(stderr, "Error: Invalid parameters provided to compute_pairwise_morans_i_scaled (%s)\n",
                label ? label : "unknown");
        return NULL;
    }
    if (W->nnz > 0 && !W->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL in compute_pairwise_morans_i_scaled (%s)\n",
                label ? label : "unknown");
        return NULL;
    }

    MKL_INT n_spots = X->nrows;
    MKL_INT n_genes = X->ncols;

    if (n_spots != W->nrows || n_spots != W->ncols) {
        fprintf(stderr, "Error: Dimension mismatch between X (%lld spots x %lld genes) and W (%lldx%lld) in %s\n",
                (long long)n_spots, (long long)n_genes, (long long)W->nrows, (long long)W->ncols,
                label ? label : "unknown");
        return NULL;
    }

    if (validate_matrix_dimensions(n_genes, n_genes, label ? label : "Moran's I result") != MORANS_I_SUCCESS) {
        return NULL;
    }

    if (n_genes == 0) {
        fprintf(stderr, "Warning: n_genes is 0 in compute_pairwise_morans_i_scaled (%s). Returning empty result matrix.\n",
                label ? label : "unknown");
        DenseMatrix* res_empty = (DenseMatrix*)calloc(1, sizeof(DenseMatrix));
        if(!res_empty) {
            perror("calloc for empty moran's I result");
            return NULL;
        }
        if (X->colnames) {
            res_empty->rownames = NULL;
            res_empty->colnames = NULL;
        }
        return res_empty;
    }

    printf("Calculating %s Moran's I for %lld genes using %lld spots (Matrix approach: X_T * W * X)...\n",
           label ? label : "", (long long)n_genes, (long long)n_spots);

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

    // Copy gene names (with fallback for NULL entries)
    if (copy_string_array_with_fallback(result->rownames, (const char**)X->colnames, n_genes, "Gene%lld") != MORANS_I_SUCCESS ||
        copy_string_array_with_fallback(result->colnames, (const char**)X->colnames, n_genes, "Gene%lld") != MORANS_I_SUCCESS) {
        free_dense_matrix(result);
        return NULL;
    }

    sparse_matrix_t W_mkl;
    if (create_sparse_handle(W, &W_mkl) != SPARSE_STATUS_SUCCESS) {
        free_dense_matrix(result);
        return NULL;
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

    sparse_status_t status = mkl_sparse_d_mm(
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

    printf("%s Moran's I matrix calculation complete.\n", label ? label : "");
    DEBUG_MATRIX_INFO(result, label ? label : "morans_i_result");
    return result;
}

/* Calculate pairwise Moran's I matrix: Result = (X_transpose * W * X) / S0 with row normalization support */
DenseMatrix* calculate_morans_i(const DenseMatrix* X, const SparseMatrix* W, int row_normalized) {
    MKL_INT n_spots = X->nrows;
    MKL_INT n_genes = X->ncols;

    /* Calculate scaling factor based on row normalization */
    double scaling_factor;
    if (row_normalized) {
        scaling_factor = 1.0 / (double)n_spots;  // Correct scaling
        printf("  Using row-normalized weights (scaling factor = 1.0 / n_spots)\n");
    } else {
        double S0 = calculate_weight_sum(W);
        printf("  Sum of weights S0: %.6f\n", S0);

        if (fabs(S0) < DBL_EPSILON) {
            fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Moran's I results will be NaN/Inf or 0.\n", S0);
            if (S0 == 0.0) {
                /* Must still allocate a valid result to return NaN values */
                DenseMatrix* nan_result = compute_pairwise_morans_i_scaled(X, W, 1.0, "standard");
                if (nan_result) {
                    for (size_t i = 0; i < (size_t)n_genes * n_genes; ++i)
                        nan_result->values[i] = NAN;
                }
                return nan_result;
            }
        }

        scaling_factor = 1.0 / S0;
        printf("  Using 1/S0 = %.6e as scaling factor\n", scaling_factor);
    }

    return compute_pairwise_morans_i_scaled(X, W, scaling_factor,
                                            row_normalized ? "standard (row-normalized)" : "standard");
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
        scaling_factor = 1.0 / (double)n_spots;  // Correct scaling
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
    if (create_sparse_handle(W, &W_mkl) != SPARSE_STATUS_SUCCESS) {
        mkl_free(Wz);
        return NAN;
    }

    struct matrix_descr descrW;
    sparse_status_t status;
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
        scaling_factor = 1.0 / (double)n_spots;  // Correct scaling
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
    if (create_sparse_handle(W, &W_mkl) != SPARSE_STATUS_SUCCESS) {
        mkl_free(W_z0); mkl_free(z0_data); mkl_free(moran_I_results);
        return NULL;
    }

    struct matrix_descr descrW;
    sparse_status_t status;
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
