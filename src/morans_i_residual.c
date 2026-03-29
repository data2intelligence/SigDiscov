/* morans_i_residual.c - Residual Moran's I analysis module
 *
 * This module contains all residual Moran's I analysis functionality,
 * including matrix operations for residual computation, core residual
 * Moran's I calculation, and residual permutation testing.
 *
 * Split from morans_i_mkl.c as part of modularization.
 */

#include "morans_i_internal.h"

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
    if (copy_string_array_with_fallback(B->rownames, (const char**)Z->colnames, n_celltypes, "CellType_%lld") != MORANS_I_SUCCESS) {
        mkl_free(ZtZ);
        mkl_free(ZtX);
        free_dense_matrix(B);
        return NULL;
    }

    if (copy_string_array_with_fallback(B->colnames, (const char**)X->colnames, n_genes, "Gene_%lld") != MORANS_I_SUCCESS) {
        mkl_free(ZtZ);
        mkl_free(ZtX);
        free_dense_matrix(B);
        return NULL;
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

    DenseMatrix* R = alloc_dense_matrix_like(X, "Spot_%lld", "Gene_%lld");
    if (!R) return NULL;

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

    DenseMatrix* centered = alloc_dense_matrix_like(matrix, "Spot_%lld", "Gene_%lld");
    if (!centered) return NULL;

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

    DenseMatrix* normalized = alloc_dense_matrix_like(matrix, "Spot_%lld", "Gene_%lld");
    if (!normalized) return NULL;

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
    if (!R_normalized || !W) {
        fprintf(stderr, "Error: NULL parameters in calculate_residual_morans_i_matrix\n");
        return NULL;
    }

    MKL_INT n_genes = R_normalized->ncols;

    /* Calculate scaling factor: always 1/S0 for residual */
    double S0 = calculate_weight_sum(W);
    printf("  Sum of weights S0: %.6f\n", S0);

    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Residual Moran's I results will be NaN/Inf or 0.\n", S0);
        if (S0 == 0.0) {
            /* Allocate result via the shared function then fill with NaN */
            DenseMatrix* nan_result = compute_pairwise_morans_i_scaled(R_normalized, W, 1.0, "residual");
            if (nan_result) {
                for (size_t i = 0; i < (size_t)n_genes * n_genes; ++i)
                    nan_result->values[i] = NAN;
            }
            return nan_result;
        }
    }

    double scaling_factor = 1.0 / S0;
    printf("  Using 1/S0 = %.6e as scaling factor\n", scaling_factor);

    return compute_pairwise_morans_i_scaled(R_normalized, W, scaling_factor, "residual");
}

/* Main residual Moran's I calculation function */
ResidualResults* calculate_residual_morans_i(const DenseMatrix* X, const CellTypeMatrix* Z,
                                           const SparseMatrix* W, const ResidualConfig* config,
                                           int verbose) { // Add verbose parameter
    if (!X || !Z || !W || !config) {
        fprintf(stderr, "Error: NULL parameters in calculate_residual_morans_i\n");
        return NULL;
    }

    // Wrap the initial print in the verbose check
    if (verbose) {
        printf("Starting residual Moran's I analysis...\n");
    }

    ResidualResults* results = (ResidualResults*)calloc(1, sizeof(ResidualResults));
    if (!results) {
        perror("Failed to allocate ResidualResults");
        return NULL;
    }

    // Step 1: Compute regression coefficients B̂
    if (verbose) printf("Step 1: Computing regression coefficients...\n");
    // We will need to thread this verbose flag down to sub-functions as well.
    // Let's assume for now sub-functions also get a verbose flag.
    results->regression_coefficients = compute_regression_coefficients(Z, X, config->regularization_lambda);
    if (!results->regression_coefficients) {
        fprintf(stderr, "Error: Failed to compute regression coefficients\n");
        free_residual_results(results);
        return NULL;
    }

    // Step 2: Compute residual projection matrix M_res
    if (verbose) printf("Step 2: Computing residual projection matrix...\n");
    DenseMatrix* M_res = compute_residual_projection_matrix(Z, config->regularization_lambda);
    if (!M_res) {
        fprintf(stderr, "Error: Failed to compute residual projection matrix\n");
        free_residual_results(results);
        return NULL;
    }

    // Step 3: Apply residual projection R = X * M_res
    if (verbose) printf("Step 3: Applying residual projection...\n");
    DenseMatrix* R = apply_residual_projection(X, M_res);
    free_dense_matrix(M_res); // Free intermediate matrix
    if (!R) {
        fprintf(stderr, "Error: Failed to apply residual projection\n");
        free_residual_results(results);
        return NULL;
    }

    // Step 4: Center residuals R_rc = R * H_n
    if (verbose) printf("Step 4: Centering residuals...\n");
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
        if (verbose) printf("Step 5: Normalizing residuals...\n");
        R_normalized = normalize_matrix_rows(R_centered);
        if (!R_normalized) {
            fprintf(stderr, "Error: Failed to normalize residuals\n");
            free_residual_results(results);
            return NULL;
        }
    } else {
        if (verbose) printf("Step 5: Skipping residual normalization (as requested)\n");
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
    if (verbose) printf("Step 6: Computing residual Moran's I...\n");
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

    if (verbose) {
        printf("Residual Moran's I analysis completed successfully.\n");
    }
    return results;
}

/* ===============================
 * RESIDUAL PERMUTATION TESTING
 * =============================== */

/* Residual permutation worker function */
static int residual_permutation_worker(const ResidualPermWorkerContext* ctx,
                                      int thread_id,
                                      int start_perm,
                                      int end_perm,
                                      double* local_mean_sum,
                                      double* local_var_sum_sq,
                                      double* local_p_counts) {

    const DenseMatrix* X_original = ctx->X_original;
    const SparseMatrix* W = ctx->W;
    const ResidualConfig* config = ctx->config;
    const PermutationParams* params = ctx->params;
    const DenseMatrix* observed_results = ctx->observed_results;

    MKL_INT n_spots = X_original->nrows;
    MKL_INT n_genes = X_original->ncols;
    size_t matrix_elements = (size_t)n_genes * n_genes;

    // Thread-local allocations with overflow protection
    size_t xperm_size;
    if (safe_multiply_size_t((size_t)n_spots, (size_t)n_genes, &xperm_size) != 0 ||
        safe_multiply_size_t(xperm_size, sizeof(double), &xperm_size) != 0) {
        DEBUG_PRINT("Thread %d: Overflow computing X_perm allocation size", thread_id);
        return -1;
    }

    double* xperm_values = (double*)mkl_malloc(xperm_size, 64);
    double* gene_buffer = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    MKL_INT* indices_buffer = (MKL_INT*)mkl_malloc((size_t)n_spots * sizeof(MKL_INT), 64);

    /* Pre-allocate reusable intermediates with values only (no name copying).
     * R_buf:      n_spots x n_genes  -- residual projection result
     * R_norm_buf: n_spots x n_genes  -- centered (and optionally row-normalized) result */
    DenseMatrix* R_buf = alloc_dense_matrix_values_only(n_spots, n_genes);
    DenseMatrix* R_norm_buf = alloc_dense_matrix_values_only(n_spots, n_genes);

    if (!xperm_values || !gene_buffer || !indices_buffer || !R_buf || !R_norm_buf) {
        DEBUG_PRINT("Thread %d: Memory allocation failed for residual permutation", thread_id);
        if (xperm_values) mkl_free(xperm_values);
        if (gene_buffer) mkl_free(gene_buffer);
        if (indices_buffer) mkl_free(indices_buffer);
        if (R_buf) free_dense_matrix(R_buf);
        if (R_norm_buf) free_dense_matrix(R_norm_buf);
        return -1;
    }

    /* Precompute weight sum and scaling factor once for all permutations */
    double S0 = calculate_weight_sum(W);
    double scaling_factor;
    int s0_is_zero = 0;
    if (fabs(S0) < DBL_EPSILON) {
        scaling_factor = 1.0;  /* will produce NaN results, same as non-perm path */
        s0_is_zero = (S0 == 0.0);
    } else {
        scaling_factor = 1.0 / S0;
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
                    MKL_INT k = rand_range_unbiased(&local_seed, i + 1);
                    MKL_INT temp_idx = indices_buffer[i];
                    indices_buffer[i] = indices_buffer[k];
                    indices_buffer[k] = temp_idx;
                }
            }

            // Apply permutation
            for (MKL_INT i = 0; i < n_spots; i++) {
                xperm_values[i * n_genes + j] = gene_buffer[indices_buffer[i]];
            }
        }

        /* --- Inline residual pipeline (avoids name allocation) --- */

        /* Step 1: R = M_res * X_perm  (same as apply_residual_projection) */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_spots, n_genes, n_spots,
                    1.0, ctx->M_res->values, n_spots,
                    xperm_values, n_genes,
                    0.0, R_buf->values, n_genes);

        /* Step 2: Center columns in-place into R_norm_buf (same as center_matrix_columns) */
        for (MKL_INT j = 0; j < n_genes; j++) {
            double column_sum = 0.0;
            for (MKL_INT i = 0; i < n_spots; i++) {
                column_sum += R_buf->values[i * n_genes + j];
            }
            double column_mean = column_sum / (double)n_spots;
            for (MKL_INT i = 0; i < n_spots; i++) {
                R_norm_buf->values[i * n_genes + j] = R_buf->values[i * n_genes + j] - column_mean;
            }
        }

        /* Step 3: Normalize rows if requested (same as normalize_matrix_rows).
         * Operates on R_norm_buf in-place when normalizing, reading from
         * an intermediate copy in R_buf. */
        if (config->normalize_residuals) {
            /* Copy centered values to R_buf so we can read from it while writing R_norm_buf */
            memcpy(R_buf->values, R_norm_buf->values, (size_t)n_spots * n_genes * sizeof(double));
            for (MKL_INT i = 0; i < n_spots; i++) {
                double row_sum_sq = 0.0;
                for (MKL_INT j = 0; j < n_genes; j++) {
                    double val = R_buf->values[i * n_genes + j];
                    row_sum_sq += val * val;
                }
                double row_norm = sqrt(row_sum_sq / (double)n_genes);
                if (row_norm < ZERO_STD_THRESHOLD) {
                    for (MKL_INT j = 0; j < n_genes; j++) {
                        R_norm_buf->values[i * n_genes + j] = 0.0;
                    }
                } else {
                    for (MKL_INT j = 0; j < n_genes; j++) {
                        R_norm_buf->values[i * n_genes + j] = R_buf->values[i * n_genes + j] / row_norm;
                    }
                }
            }
        }

        /* Step 4: Compute Moran's I = scaling_factor * R_norm^T * W * R_norm
         * (same as compute_pairwise_morans_i_scaled + calculate_residual_morans_i_matrix)
         * Re-use R_buf->values as scratch for W * R_norm. */
        if (s0_is_zero) {
            /* All results would be NaN, which the isfinite guard maps to 0.
             * Skip the expensive sparse-dense multiply entirely. */
            continue;
        }

        /* Create MKL sparse handle for W */
        sparse_matrix_t W_mkl;
        sparse_status_t sp_status = mkl_sparse_d_create_csr(
            &W_mkl, SPARSE_INDEX_BASE_ZERO, W->nrows, W->ncols,
            W->row_ptr, W->row_ptr + 1, W->col_ind, W->values);
        if (sp_status != SPARSE_STATUS_SUCCESS) {
            DEBUG_PRINT("Thread %d: Permutation %d MKL sparse handle creation failed", thread_id, perm);
            continue;
        }

        struct matrix_descr descrW;
        descrW.type = SPARSE_MATRIX_TYPE_GENERAL;

        /* Temp_WR = W * R_norm  (n_spots x n_genes, stored in R_buf->values) */
        sp_status = mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            1.0,
            W_mkl,
            descrW,
            SPARSE_LAYOUT_ROW_MAJOR,
            R_norm_buf->values,
            n_genes,
            n_genes,
            0.0,
            R_buf->values,
            n_genes
        );

        mkl_sparse_destroy(W_mkl);

        if (sp_status != SPARSE_STATUS_SUCCESS) {
            DEBUG_PRINT("Thread %d: Permutation %d sparse mm failed", thread_id, perm);
            continue;
        }

        /* perm_morans_values = scaling_factor * R_norm^T * Temp_WR  (n_genes x n_genes)
         * We need a separate buffer for the result since it's n_genes x n_genes,
         * not n_spots x n_genes. Reuse xperm_values if n_genes*n_genes <= n_spots*n_genes,
         * which is true when n_genes <= n_spots (always the case in practice).
         * Otherwise fall back to a temporary allocation. */
        double* perm_result_values;
        int perm_result_allocated = 0;
        if ((size_t)n_genes * n_genes <= (size_t)n_spots * n_genes) {
            perm_result_values = xperm_values;  /* safe to reuse: X_perm not needed until next iteration */
        } else {
            perm_result_values = (double*)mkl_malloc(matrix_elements * sizeof(double), 64);
            if (!perm_result_values) {
                DEBUG_PRINT("Thread %d: Permutation %d result alloc failed", thread_id, perm);
                continue;
            }
            perm_result_allocated = 1;
        }

        cblas_dgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            n_genes,
            n_genes,
            n_spots,
            scaling_factor,
            R_norm_buf->values,
            n_genes,
            R_buf->values,
            n_genes,
            0.0,
            perm_result_values,
            n_genes
        );

        // Accumulate statistics
        for (size_t idx = 0; idx < matrix_elements; idx++) {
            double perm_val = perm_result_values[idx];
            if (!isfinite(perm_val)) perm_val = 0.0;

            local_mean_sum[idx] += perm_val;
            local_var_sum_sq[idx] += perm_val * perm_val;

            if (params->p_value_output && local_p_counts && observed_results) {
                if (fabs(perm_val) >= fabs(observed_results->values[idx])) {
                    local_p_counts[idx]++;
                }
            }
        }

        if (perm_result_allocated) mkl_free(perm_result_values);
    }

    // Cleanup
    mkl_free(xperm_values);
    mkl_free(gene_buffer);
    mkl_free(indices_buffer);
    free_dense_matrix(R_buf);
    free_dense_matrix(R_norm_buf);

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
    ResidualResults* observed_residual_results = calculate_residual_morans_i(X, Z, W, config, 0);
    if (!observed_residual_results || !observed_residual_results->residual_morans_i) {
        fprintf(stderr, "Error: Failed to calculate observed residual Moran's I for permutation test\n");
        if (observed_residual_results) free_residual_results(observed_residual_results);
        return NULL;
    }

    DenseMatrix* observed_results = observed_residual_results->residual_morans_i;

    // Allocate and initialize results structure
    PermutationResults* results = alloc_permutation_results(
        n_genes, (const char**)X->colnames, params);
    if (!results) {
        free_residual_results(observed_residual_results);
        return NULL;
    }

    size_t matrix_elements = (size_t)n_genes * n_genes;

    // Run permutations using multiple threads
    int num_threads = omp_get_max_threads();
    int perms_per_thread = n_perm / num_threads;
    int remaining_perms = n_perm % num_threads;

    printf("Starting residual permutation loop (%d permutations) using %d OpenMP threads...\n",
           n_perm, num_threads);

    // Precompute M_res ONCE before the parallel loop to avoid O(k^3) inversion per permutation
    DenseMatrix* M_res = compute_residual_projection_matrix(Z, config->regularization_lambda);
    if (!M_res) {
        fprintf(stderr, "Error: Failed to precompute residual projection matrix for permutation test\n");
        free_permutation_results(results);
        free_residual_results(observed_residual_results);
        return NULL;
    }

    ResidualPermWorkerContext ctx;
    ctx.X_original = X;
    ctx.Z = Z;
    ctx.W = W;
    ctx.config = config;
    ctx.params = params;
    ctx.observed_results = observed_results;
    ctx.M_res = M_res;

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
            int worker_result = residual_permutation_worker(&ctx, thread_id,
                                                          start_perm, end_perm,
                                                          local_mean_sum, local_var_sum_sq,
                                                          local_p_counts);

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

    // Free precomputed projection matrix
    free_dense_matrix(M_res);

    if (error_occurred) {
        fprintf(stderr, "Error occurred during residual permutation execution\n");
        free_permutation_results(results);
        free_residual_results(observed_residual_results);
        return NULL;
    }

    // Finalize statistics
    finalize_permutation_statistics(results, observed_results, params, n_genes, n_perm);

    free_residual_results(observed_residual_results);
    printf("Residual permutation test complete.\n");
    return results;
}
