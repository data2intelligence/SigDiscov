/* morans_i_mkl.c - Optimized MKL-based Moran's I implementation */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <ctype.h>
#include <omp.h>
#include <regex.h>
#include <errno.h>
#include <unistd.h>  /* For access() */
#include "morans_i_mkl.h"

/* Print help message */
void print_help(const char* program_name) {
    printf("\nCompute Pairwise or Single-Gene Moran's I for Spatial Transcriptomics Data\n\n");
    printf("Usage: %s -i <input.tsv> -o <output.tsv> [OPTIONS]\n\n", program_name);
    printf("Input Format:\n");
    printf("  Tab-separated file (TSV).\n");
    printf("  First row: Header with spot coordinates (e.g., '12x34') or cell IDs. First cell can be empty/gene ID.\n");
    printf("  Subsequent rows: Gene name followed by expression values for each spot/cell.\n");
    printf("\nRequired Arguments:\n");
    printf("  -i <file>\tInput data matrix file (Genes x Spots/Cells).\n");
    printf("  -o <file>\tOutput file.\n");
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
    printf("  -t <int>\tSet number of OpenMP threads. Default: Use OMP_NUM_THREADS environment variable.\n");
    printf("  -m <int>\tSet number of MKL threads. Default: Use MKL_NUM_THREADS environment variable.\n");
    printf("\nSingle-cell specific options:\n");
    printf("  -c <file>\tCoordinates/metadata file with cell locations (CSV format). Required for single-cell data.\n");
    printf("  --id-col <name>\tColumn name for cell IDs in metadata file. Default: 'cell_ID'.\n");
    printf("  --x-col <name>\tColumn name for X coordinates in metadata file. Default: 'sdimx'.\n");
    printf("  --y-col <name>\tColumn name for Y coordinates in metadata file. Default: 'sdimy'.\n");
    printf("  --scale <float>\tScaling factor for coordinates to convert to integer grid. Default: %.1f.\n", 
           DEFAULT_COORD_SCALE_FACTOR);
    printf("  --sigma <float>\tCustom sigma parameter for RBF kernel. If not provided, inferred from data for single-cell or default platform values used.\n");
    printf("\nOutput Format:\n");
    printf("  If -b 0 (Single-gene): TSV file with two columns: 'Gene', 'MoranI'.\n");
    printf("  If -b 1 and -g 1 (Pairwise All): TSV file representing a symmetric matrix. Rows and columns are gene names. "
           "Cell (i, j) is Moran's I between gene_i and gene_j.\n");
    printf("  If -b 1 and -g 0 (Pairwise First Gene): TSV file with two columns: 'Gene', 'MoranI_vs_Gene0', "
           "where Gene0 is the first gene in the input file.\n");
}

/* Z-Normalize function (Gene-wise) */
DenseMatrix* z_normalize(DenseMatrix* data_matrix) {
    MKL_INT n_rows = data_matrix->nrows;
    MKL_INT n_cols = data_matrix->ncols;

    printf("Performing Z-normalization on %lld genes across %lld spots...\n",
           (long long)n_rows, (long long)n_cols);

    DenseMatrix* normalized = (DenseMatrix*)malloc(sizeof(DenseMatrix));
     if (!normalized) {
        perror("Failed to allocate DenseMatrix structure for normalized data");
        return NULL;
     }
    normalized->nrows = n_rows;
    normalized->ncols = n_cols;
    normalized->rownames = (char**)malloc(n_rows * sizeof(char*));
    normalized->colnames = (char**)malloc(n_cols * sizeof(char*));
    normalized->values = (double*)mkl_malloc(n_rows * n_cols * sizeof(double), 64);

    if (!normalized->rownames || !normalized->colnames || !normalized->values) {
        perror("Failed to allocate memory for normalized matrix data");
        if (normalized->values) mkl_free(normalized->values);
        free(normalized->rownames); free(normalized->colnames); free(normalized);
        return NULL;
    }

    /* Copy row and column names */
    for (MKL_INT i = 0; i < n_rows; i++) {
        normalized->rownames[i] = strdup(data_matrix->rownames[i]);
    }
    for (MKL_INT j = 0; j < n_cols; j++) {
        normalized->colnames[j] = strdup(data_matrix->colnames[j]);
    }

    /* Process each gene (row) in parallel */
    #pragma omp parallel for schedule(dynamic)
    for (MKL_INT i = 0; i < n_rows; i++) {
        double* gene_row = &(data_matrix->values[i * n_cols]);
        double* norm_row = &(normalized->values[i * n_cols]);

        /* Calculate mean (using only finite values) */
        double sum = 0.0;
        MKL_INT n_finite = 0;
        for (MKL_INT j = 0; j < n_cols; j++) {
            if (isfinite(gene_row[j])) {
                sum += gene_row[j];
                n_finite++;
            }
        }

        double mean = 0.0;
        double std_dev = 0.0;

        if (n_finite > 1) {
            mean = sum / n_finite;

            /* Calculate variance (using only finite values) */
            double sum_sq_diff = 0.0;
            for (MKL_INT j = 0; j < n_cols; j++) {
                 if (isfinite(gene_row[j])) {
                    double diff = gene_row[j] - mean;
                    sum_sq_diff += diff * diff;
                 }
            }
            double variance = sum_sq_diff / n_finite;
            if (variance < 0.0) variance = 0.0;
            std_dev = sqrt(variance);
        }

        /* Check if standard deviation is too small */
        if (n_finite <= 1 || std_dev < ZERO_STD_THRESHOLD) {
            for (MKL_INT j = 0; j < n_cols; j++) {
                norm_row[j] = 0.0;
            }
        } else {
            /* Prepare data for vectorized Z-score normalization */
            double* centered = (double*)mkl_malloc(n_cols * sizeof(double), 64);
            double* std_dev_vec = (double*)mkl_malloc(n_cols * sizeof(double), 64);
            
            if (!centered || !std_dev_vec) {
                if (centered) mkl_free(centered);
                if (std_dev_vec) mkl_free(std_dev_vec);
                
                /* Fall back to scalar division if allocation fails */
                double inv_std_dev = 1.0 / std_dev;
                for (MKL_INT j = 0; j < n_cols; j++) {
                    if (isfinite(gene_row[j])) {
                        norm_row[j] = (gene_row[j] - mean) * inv_std_dev;
                    } else {
                        norm_row[j] = 0.0;
                    }
                }
                continue;
            }
            
            /* First, subtract mean from all values */
            for (MKL_INT j = 0; j < n_cols; j++) {
                if (isfinite(gene_row[j])) {
                    centered[j] = gene_row[j] - mean;
                } else {
                    centered[j] = 0.0;
                }
                std_dev_vec[j] = std_dev; /* Fill vector with same standard deviation */
            }
            
            /* Use vdDiv for vectorized division */
            vdDiv(n_cols, centered, std_dev_vec, norm_row);
            
            /* Clean up temporary arrays */
            mkl_free(centered);
            mkl_free(std_dev_vec);
        }
    }

    printf("Z-normalization complete.\n");
    return normalized;
}

/* Build spatial weight matrix W (Sparse CSR) */
SparseMatrix* build_spatial_weight_matrix(MKL_INT* spot_row_valid, MKL_INT* spot_col_valid,
                                          MKL_INT n_spots_valid, DenseMatrix* distance_matrix,
                                          MKL_INT max_radius) {

    printf("Building sparse spatial weight matrix W (%lld x %lld)...\n",
           (long long)n_spots_valid, (long long)n_spots_valid);

    /* Estimate initial NNZ capacity (heuristic) */
    MKL_INT estimated_neighbors = (MKL_INT)(M_PI * max_radius * max_radius * 1.5);
    if (estimated_neighbors <= 0) estimated_neighbors = 1;
    MKL_INT initial_capacity = n_spots_valid * estimated_neighbors;
    if (initial_capacity > n_spots_valid * n_spots_valid) {
        initial_capacity = n_spots_valid * n_spots_valid;
    }
     if (initial_capacity <= 0) initial_capacity = n_spots_valid;
    printf("  Initial estimated NNZ capacity: %lld\n", (long long)initial_capacity);

    /* Using temporary COO storage (triplets) which will be converted to CSR */
    MKL_INT current_capacity = initial_capacity;
    MKL_INT* temp_I = (MKL_INT*)malloc(current_capacity * sizeof(MKL_INT));
    MKL_INT* temp_J = (MKL_INT*)malloc(current_capacity * sizeof(MKL_INT));
    double* temp_V = (double*)malloc(current_capacity * sizeof(double));
    MKL_INT nnz_count = 0;

    if (!temp_I || !temp_J || !temp_V) {
        perror("Failed to allocate initial COO arrays");
        free(temp_I); free(temp_J); free(temp_V);
        return NULL;
    }

    /* Parallel loop over spots to find neighbors and weights */
    #pragma omp parallel
    {
        /* Thread-local storage for triplets to reduce contention */
        MKL_INT local_capacity = 1024;
        MKL_INT* local_I = (MKL_INT*)malloc(local_capacity * sizeof(MKL_INT));
        MKL_INT* local_J = (MKL_INT*)malloc(local_capacity * sizeof(MKL_INT));
        double* local_V = (double*)malloc(local_capacity * sizeof(double));
        MKL_INT local_nnz = 0;

        if (!local_I || !local_J || !local_V) {
             fprintf(stderr, "Error: Failed to alloc thread-local COO buffers.\n");
             #pragma omp cancel parallel
        }

        #pragma omp for schedule(dynamic, 128)
        for (MKL_INT i = 0; i < n_spots_valid; i++) {
            for (MKL_INT j = 0; j < n_spots_valid; j++) {
                MKL_INT row_shift = labs(spot_row_valid[i] - spot_row_valid[j]);
                MKL_INT col_shift = labs(spot_col_valid[i] - spot_col_valid[j]);

                if (row_shift < max_radius && col_shift < (2 * max_radius)) {
                    double weight = distance_matrix->values[row_shift * distance_matrix->ncols + col_shift];

                    if (fabs(weight) > WEIGHT_THRESHOLD) {
                        if (local_nnz >= local_capacity) {
                            local_capacity *= 2;
                            MKL_INT* temp_li = (MKL_INT*)realloc(local_I, local_capacity * sizeof(MKL_INT));
                            MKL_INT* temp_lj = (MKL_INT*)realloc(local_J, local_capacity * sizeof(MKL_INT));
                            double* temp_lv = (double*)realloc(local_V, local_capacity * sizeof(double));
                            if (!temp_li || !temp_lj || !temp_lv) {
                                fprintf(stderr, "Error: Failed to realloc thread-local COO buffers.\n");
                                #pragma omp cancel for
                                break;
                            }
                            local_I = temp_li; local_J = temp_lj; local_V = temp_lv;
                        }
                        local_I[local_nnz] = i;
                        local_J[local_nnz] = j;
                        local_V[local_nnz] = weight;
                        local_nnz++;
                    }
                }
            }
             #pragma omp cancellation point for
        }

        #pragma omp critical
        {
            if (nnz_count + local_nnz > current_capacity) {
                current_capacity = (nnz_count + local_nnz) * 1.5;
                if (current_capacity > n_spots_valid * n_spots_valid) {
                     current_capacity = n_spots_valid * n_spots_valid;
                }
                if (nnz_count + local_nnz > current_capacity) {
                    fprintf(stderr, "Error: Cannot resize global COO buffer large enough (%lld needed, max %lld).\n", (long long)(nnz_count + local_nnz), (long long)(n_spots_valid * n_spots_valid));
                } else {
                    printf("  Resizing global COO buffer to %lld\n", (long long)current_capacity);
                    MKL_INT* temp_gi = (MKL_INT*)realloc(temp_I, current_capacity * sizeof(MKL_INT));
                    MKL_INT* temp_gj = (MKL_INT*)realloc(temp_J, current_capacity * sizeof(MKL_INT));
                    double* temp_gv = (double*)realloc(temp_V, current_capacity * sizeof(double));
                    if (!temp_gi || !temp_gj || !temp_gv) {
                        fprintf(stderr, "Error: Failed to realloc global COO buffers. Aborting merge.\n");
                    } else {
                         temp_I = temp_gi; temp_J = temp_gj; temp_V = temp_gv;
                    }
                }
            }

            if (temp_I && temp_J && temp_V && (nnz_count + local_nnz <= current_capacity)) {
                memcpy(temp_I + nnz_count, local_I, local_nnz * sizeof(MKL_INT));
                memcpy(temp_J + nnz_count, local_J, local_nnz * sizeof(MKL_INT));
                memcpy(temp_V + nnz_count, local_V, local_nnz * sizeof(double));
                nnz_count += local_nnz;
            } else {
                 fprintf(stderr, "Warning: Could not merge thread %d results due to memory error or insufficient space.\n", omp_get_thread_num());
            }
        }

        free(local_I);
        free(local_J);
        free(local_V);
    }

    printf("  Generated %lld non-zero entries (COO format).\n", (long long)nnz_count);

    /* Now, convert the COO triplets (temp_I, temp_J, temp_V) to CSR format */
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

    /* Allocate CSR arrays */
    W->row_ptr = (MKL_INT*)mkl_malloc((n_spots_valid + 1) * sizeof(MKL_INT), 64);
    W->col_ind = (MKL_INT*)mkl_malloc(nnz_count * sizeof(MKL_INT), 64);
    W->values = (double*)mkl_malloc(nnz_count * sizeof(double), 64);

    if (!W->row_ptr || !W->col_ind || !W->values) {
        perror("Failed to allocate CSR arrays for W");
        mkl_free(W->row_ptr); mkl_free(W->col_ind); mkl_free(W->values); free(W);
        free(temp_I); free(temp_J); free(temp_V);
        return NULL;
    }

    /* Manual conversion from COO to CSR */
    for (MKL_INT i = 0; i <= n_spots_valid; ++i) W->row_ptr[i] = 0;
    for (MKL_INT k = 0; k < nnz_count; ++k) {
        W->row_ptr[temp_I[k] + 1]++;
    }
    for (MKL_INT i = 0; i < n_spots_valid; ++i) {
        W->row_ptr[i + 1] += W->row_ptr[i];
    }
    MKL_INT* current_pos = (MKL_INT*)calloc(n_spots_valid, sizeof(MKL_INT));
    for (MKL_INT k = 0; k < nnz_count; ++k) {
        MKL_INT row = temp_I[k];
        MKL_INT index_in_csr = W->row_ptr[row] + current_pos[row];
        W->col_ind[index_in_csr] = temp_J[k];
        W->values[index_in_csr] = temp_V[k];
        current_pos[row]++;
    }
    free(current_pos);

    /* Free temporary COO arrays */
    free(temp_I);
    free(temp_J);
    free(temp_V);

    printf("Sparse weight matrix W built successfully (CSR format, %lld NNZ).\n", (long long)W->nnz);

    /* Optional: Sort column indices within each row */
    sparse_matrix_t W_mkl_tmp;
    sparse_status_t status = mkl_sparse_d_create_csr(&W_mkl_tmp, SPARSE_INDEX_BASE_ZERO, W->nrows, W->ncols, W->row_ptr, W->row_ptr + 1, W->col_ind, W->values);
     if (status == SPARSE_STATUS_SUCCESS) {
         status = mkl_sparse_order(W_mkl_tmp);
         if (status != SPARSE_STATUS_SUCCESS) {
             print_mkl_status(status, "mkl_sparse_order");
         }
         mkl_sparse_destroy(W_mkl_tmp);
         printf("  Column indices within rows ordered.\n");
     } else {
         print_mkl_status(status, "mkl_sparse_d_create_csr (for ordering)");
     }

    return W;
}

/* Calculate pairwise Moran's I matrix: Result = X' * W * X / S0 */
DenseMatrix* calculate_morans_i(DenseMatrix* X, SparseMatrix* W) {
    MKL_INT n_spots = X->nrows;
    MKL_INT n_genes = X->ncols;

    if (n_spots != W->nrows || n_spots != W->ncols) {
        fprintf(stderr, "Error: Dimension mismatch between X (%lldx%lld) and W (%lldx%lld)\n",
                (long long)n_spots, (long long)n_genes, (long long)W->nrows, (long long)W->ncols);
        return NULL;
    }

    printf("Calculating Moran's I for %lld genes using %lld spots (Matrix approach: X' * W * X / S0)...\n",
           (long long)n_genes, (long long)n_spots);

    /* Create result matrix (Genes x Genes) */
    DenseMatrix* result = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!result) { perror("Failed alloc result struct"); return NULL; }
    result->nrows = n_genes;
    result->ncols = n_genes;
    result->values = (double*)mkl_malloc(n_genes * n_genes * sizeof(double), 64);
    result->rownames = (char**)malloc(n_genes * sizeof(char*));
    result->colnames = (char**)malloc(n_genes * sizeof(char*));

    if (!result->values || !result->rownames || !result->colnames) {
        perror("Failed alloc result data");
        if(result->values) mkl_free(result->values);
        free(result->rownames); free(result->colnames); free(result);
        return NULL;
    }

    /* Copy gene names from X's colnames */
    for (MKL_INT i = 0; i < n_genes; i++) {
        result->rownames[i] = strdup(X->colnames[i]);
        result->colnames[i] = strdup(X->colnames[i]);
    }

    /* Calculate sum of weights S0 */
    double S0 = 0.0;
    #pragma omp parallel for reduction(+:S0)
    for (MKL_INT i = 0; i < W->nnz; i++) {
        S0 += W->values[i];
    }
    printf("  Sum of weights S0: %.6f\n", S0);

    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Setting to small positive value.\n", S0);
        S0 = DBL_EPSILON;
    }

    /* Prepare for division by S0 */
    double inv_S0 = 1.0 / S0;
    printf("  Using 1/S0 = %.6e as scaling factor\n", inv_S0);

    /* Create MKL sparse handle for W */
    sparse_matrix_t W_mkl;
    sparse_status_t status = mkl_sparse_d_create_csr(
        &W_mkl, SPARSE_INDEX_BASE_ZERO, W->nrows, W->ncols,
        W->row_ptr, W->row_ptr + 1, W->col_ind, W->values);

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr (W)");
        free_dense_matrix(result);
        return NULL;
    }

    /* Optimize the sparse matrix handle */
    status = mkl_sparse_optimize(W_mkl);
    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_optimize (W)");
    }

    /* Step 1: Calculate Temp = W * X (Sparse @ Dense) */
    printf("  Step 1: Calculating Temp = W * X ...\n");
    double* Temp_values = (double*)mkl_malloc(n_spots * n_genes * sizeof(double), 64);
    if (!Temp_values) {
        perror("Failed alloc Temp_values");
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }

    struct matrix_descr descrW;
    descrW.type = SPARSE_MATRIX_TYPE_GENERAL;

    double alpha = 1.0, beta = 0.0;

    status = mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        W_mkl,
        descrW,
        SPARSE_LAYOUT_ROW_MAJOR,
        X->values,
        n_genes,
        n_genes,
        beta,
        Temp_values,
        n_genes);

    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_mm (W * X)");
        mkl_free(Temp_values);
        mkl_sparse_destroy(W_mkl);
        free_dense_matrix(result);
        return NULL;
    }

    /* Step 2: Calculate Result = X' * Temp / S0 (Dense' @ Dense with scaling) */
    printf("  Step 2: Calculating Result = X' * Temp / S0 ...\n");
    cblas_dgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        n_genes,
        n_genes,
        n_spots,
        inv_S0,  /* Apply scaling factor 1/S0 directly in matrix multiplication */
        X->values,
        n_genes,
        Temp_values,
        n_genes,
        beta,
        result->values,
        n_genes);

    /* Clean up intermediate data and MKL handle */
    mkl_free(Temp_values);
    mkl_sparse_destroy(W_mkl);

    printf("Moran's I matrix calculation complete and scaled by 1/S0.\n");

    return result;
}

/* Implementation of the batch calculation function for Cython */
double* calculate_morans_i_batch(double* X_data, long long n_genes, long long n_spots,
                               double* W_values, long long* W_row_ptr, long long* W_col_ind,
                               long long W_nnz, int paired_genes) {
    /* Validate inputs */
    if (X_data == NULL || W_values == NULL || W_row_ptr == NULL || W_col_ind == NULL) {
        fprintf(stderr, "Error: NULL input to calculate_morans_i_batch\n");
        return NULL;
    }

    /* Calculate sum of weights (S0) */
    double S0 = 0.0;
    for (long long i = 0; i < W_nnz; i++) {
        S0 += W_values[i];
    }

    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero in calculate_morans_i_batch\n");
        /* Return zeros array instead of NULL to be consistent with Python behavior */
        double* zeros;
        if (paired_genes) {
            zeros = (double*)calloc(n_genes * n_genes, sizeof(double));
        } else {
            zeros = (double*)calloc(n_genes, sizeof(double));
        }
        return zeros;
    }

    /* Allocate result array */
    double* result;
    if (paired_genes) {
        result = (double*)calloc(n_genes * n_genes, sizeof(double));
    } else {
        result = (double*)calloc(n_genes, sizeof(double));
    }

    if (result == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for result in calculate_morans_i_batch\n");
        return NULL;
    }

    /* Create MKL sparse matrix format */
    sparse_matrix_t W_mkl;
    
    /* Create temporary arrays with MKL_INT type for compatibility */
    MKL_INT* tmp_row_ptr = (MKL_INT*)malloc((n_spots + 1) * sizeof(MKL_INT));
    MKL_INT* tmp_col_ind = (MKL_INT*)malloc(W_nnz * sizeof(MKL_INT));
    
    if (!tmp_row_ptr || !tmp_col_ind) {
        fprintf(stderr, "Error: Memory allocation failed for temporary MKL arrays\n");
        free(result);
        if (tmp_row_ptr) free(tmp_row_ptr);
        if (tmp_col_ind) free(tmp_col_ind);
        return NULL;
    }
    
    /* Copy the data with type conversion */
    for (long long i = 0; i <= n_spots; i++) {
        tmp_row_ptr[i] = (MKL_INT)W_row_ptr[i];
    }
    
    for (long long i = 0; i < W_nnz; i++) {
        tmp_col_ind[i] = (MKL_INT)W_col_ind[i];
    }
    
    sparse_status_t status = mkl_sparse_d_create_csr(
        &W_mkl, SPARSE_INDEX_BASE_ZERO, (MKL_INT)n_spots, (MKL_INT)n_spots,
        tmp_row_ptr, tmp_row_ptr + 1, tmp_col_ind, W_values);

    if (status != SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "Error: Failed to create MKL sparse matrix\n");
        free(result);
        free(tmp_row_ptr);
        free(tmp_col_ind);
        return NULL;
    }

    if (status != SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "Error: Failed to create MKL sparse matrix\n");
        free(result);
        free(tmp_row_ptr);
        free(tmp_col_ind);
        return NULL;
    }

    /* Optimize the sparse matrix handle */
    status = mkl_sparse_optimize(W_mkl);
    if (status != SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "Warning: Failed to optimize MKL sparse matrix\n");
        /* Continue anyway as this is just an optimization */
    }

    /* Shared parameters for MKL operations */
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    double alpha = 1.0, beta = 0.0;

    if (paired_genes) {
        /* Calculate X' * W * X using MKL BLAS operations for efficiency */

        /* Step 1: Calculate Temp = W * X (Sparse @ Dense) */
        double* Temp = (double*)calloc(n_spots * n_genes, sizeof(double));
        if (Temp == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for temporary buffer\n");
            mkl_sparse_destroy(W_mkl);
            free(result);
            free(tmp_row_ptr);
            free(tmp_col_ind);
            return NULL;
        }

        /* Perform sparse-dense matrix multiply: Temp = W * X */
        status = mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            W_mkl,
            descr,
            SPARSE_LAYOUT_ROW_MAJOR,
            X_data,              /* Input matrix: n_spots x n_genes */
            (MKL_INT)n_genes,    /* Number of columns in X */
            (MKL_INT)n_genes,    /* Leading dimension of X (column stride) */
            beta,
            Temp,                /* Output matrix: n_spots x n_genes */
            (MKL_INT)n_genes     /* Leading dimension of Temp (column stride) */
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "Error: Failed in sparse matrix multiplication (W * X)\n");
            free(Temp);
            mkl_sparse_destroy(W_mkl);
            free(result);
            free(tmp_row_ptr);
            free(tmp_col_ind);
            return NULL;
        }

        /* Step 2: Calculate Result = X' * Temp (Dense @ Dense) */
        cblas_dgemm(
            CblasRowMajor,       /* Matrix layout: row-major */
            CblasTrans,          /* Op(A): transpose X */
            CblasNoTrans,        /* Op(B): no transpose for Temp */
            (MKL_INT)n_genes,    /* M: rows of Op(A) = columns of X */
            (MKL_INT)n_genes,    /* N: columns of Op(B) = columns of Temp */
            (MKL_INT)n_spots,    /* K: columns of Op(A) = rows of X = rows of Temp */
            1.0 / S0,            /* Alpha: scale by 1/S0 */
            X_data,              /* A: X data */
            (MKL_INT)n_genes,    /* LDA: leading dimension of X */
            Temp,                /* B: Temp data */
            (MKL_INT)n_genes,    /* LDB: leading dimension of Temp */
            0.0,                 /* Beta: no initial value for C */
            result,              /* C: result matrix */
            (MKL_INT)n_genes     /* LDC: leading dimension of result */
        );

        free(Temp);
    } else {
        /* For autocorrelation (non-paired), calculate diagonal elements only */

        /* Use a more efficient approach for autocorrelation */
        double* WX = (double*)calloc(n_spots * n_genes, sizeof(double));
        if (WX == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for WX\n");
            mkl_sparse_destroy(W_mkl);
            free(result);
            free(tmp_row_ptr);
            free(tmp_col_ind);
            return NULL;
        }

        /* Calculate WX = W * X */
        status = mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            W_mkl,
            descr,
            SPARSE_LAYOUT_ROW_MAJOR,
            X_data,
            (MKL_INT)n_genes,
            (MKL_INT)n_genes,
            beta,
            WX,
            (MKL_INT)n_genes
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "Error: Failed in sparse matrix multiplication (W * X) for autocorrelation\n");
            free(WX);
            mkl_sparse_destroy(W_mkl);
            free(result);
            free(tmp_row_ptr);
            free(tmp_col_ind);
            return NULL;
        }

        /* Calculate result[g] = sum(X[g,i] * WX[i,g]) / S0 for each gene g */
        for (long long g = 0; g < n_genes; g++) {
            double sum = 0.0;
            for (long long i = 0; i < n_spots; i++) {
                sum += X_data[i * n_genes + g] * WX[i * n_genes + g];
            }
            result[g] = sum / S0;
        }

        free(WX);
    }

    /* Clean up MKL handle and temporary arrays */
    mkl_sparse_destroy(W_mkl);
    free(tmp_row_ptr);
    free(tmp_col_ind);

    return result;
}

/* Calculate Moran's I for a single gene */
double calculate_single_gene_moran_i(double* gene_data, SparseMatrix* W, MKL_INT n_spots) {
    double sum_w_ij_z_i_z_j = 0.0;
    
    /* For each row in W */
    for (MKL_INT i = 0; i < n_spots; i++) {
        double z_i = gene_data[i];
        
        /* For each non-zero entry in row i */
        for (MKL_INT k = W->row_ptr[i]; k < W->row_ptr[i+1]; k++) {
            MKL_INT j = W->col_ind[k];
            double w_ij = W->values[k];
            double z_j = gene_data[j];
            
            sum_w_ij_z_i_z_j += w_ij * z_i * z_j;
        }
    }
    
    /* Calculate sum of weights (S0) */
    double S0 = 0.0;
    for (MKL_INT k = 0; k < W->nnz; k++) {
        S0 += W->values[k];
    }
    
    /* Return Moran's I */
    if (fabs(S0) < DBL_EPSILON) {
        return 0.0; /* Avoid division by zero */
    }
    
    return sum_w_ij_z_i_z_j / S0;
}

/* Calculate Moran's I between the first gene and all others */
double* calculate_first_gene_vs_all(DenseMatrix* X, SparseMatrix* W, double S0) {
    MKL_INT n_spots = X->nrows;
    MKL_INT n_genes = X->ncols;
    
    /* Allocate array for results and intermediate storage */
    double* raw_results = (double*)mkl_malloc(n_genes * sizeof(double), 64);
    double* final_results = (double*)mkl_malloc(n_genes * sizeof(double), 64);
    double* s0_vector = (double*)mkl_malloc(n_genes * sizeof(double), 64);
    
    if (!raw_results || !final_results || !s0_vector) {
        perror("Failed to allocate memory for first gene vs all calculation");
        if (raw_results) mkl_free(raw_results);
        if (final_results) mkl_free(final_results);
        if (s0_vector) mkl_free(s0_vector);
        return NULL;
    }
    
    /* Get the first gene data */
    double* first_gene = (double*)mkl_malloc(n_spots * sizeof(double), 64);
    if (!first_gene) {
        perror("Failed to allocate memory for first gene data");
        mkl_free(raw_results);
        mkl_free(final_results);
        mkl_free(s0_vector);
        return NULL;
    }
    
    /* Copy first gene data */
    for (MKL_INT i = 0; i < n_spots; i++) {
        first_gene[i] = X->values[i * n_genes + 0]; /* First gene (index 0) */
    }
    
    /* Ensure S0 is not zero */
    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Setting to small positive value.\n", S0);
        S0 = DBL_EPSILON;
    }
    
    /* Fill S0 vector for vectorized division */
    for (MKL_INT g = 0; g < n_genes; g++) {
        s0_vector[g] = S0;
    }
    
    /* Calculate Moran's I for each gene vs the first gene */
    #pragma omp parallel for
    for (MKL_INT g = 0; g < n_genes; g++) {
        double sum_w_ij_z_i_z_j = 0.0;
        
        /* For each row in W */
        for (MKL_INT i = 0; i < n_spots; i++) {
            double z_i = first_gene[i]; /* First gene */
            
            /* For each non-zero entry in row i */
            for (MKL_INT k = W->row_ptr[i]; k < W->row_ptr[i+1]; k++) {
                MKL_INT j = W->col_ind[k];
                double w_ij = W->values[k];
                double z_j = X->values[j * n_genes + g]; /* Current gene (g) */
                
                sum_w_ij_z_i_z_j += w_ij * z_i * z_j;
            }
        }
        
        /* Store raw result */
        raw_results[g] = sum_w_ij_z_i_z_j;
    }
    
    /* Use vectorized division to scale by S0 */
    vdDiv(n_genes, raw_results, s0_vector, final_results);
    
    /* Clean up intermediate data */
    mkl_free(first_gene);
    mkl_free(raw_results);
    mkl_free(s0_vector);
    
    return final_results;
}

/* Save results in lower triangle format without headers */
void save_results(DenseMatrix* result_matrix, const char* output_file) {
    if (!result_matrix || !result_matrix->values) {
        fprintf(stderr, "Error: Cannot save NULL result matrix.\n");
        return;
    }
    MKL_INT n_genes = result_matrix->nrows;
    if (n_genes == 0) {
        printf("Warning: Result matrix is empty, saving empty file.\n");
    }

    /* Open output file */
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s' for writing: ", output_file);
        perror("");
        return;
    }

    printf("Saving Moran's I results...\n");

    /* Write lower triangle row by row (values are already scaled by 1/S0) */
    for (MKL_INT i = 0; i < n_genes; i++) {
        for (MKL_INT j = 0; j <= i; j++) {
            double value = result_matrix->values[i * n_genes + j];

            /* Format to avoid scientific notation for reasonable values */
            if (j > 0) {
                 fprintf(fp, "\t");
            }
            
            if (isnan(value)) {
                 fprintf(fp, "NaN");
            } else if (isinf(value)) {
                 fprintf(fp, "%sInf", (value > 0 ? "" : "-"));
            } else {
                 /* Use scientific notation for very small values */
                 if (fabs(value) < 0.0001 && value != 0.0) {
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
}

/* Save single gene Moran's I results (one value per line, no headers) */
void save_single_gene_results(DenseMatrix* znorm_data, SparseMatrix* W, double S0, const char* output_file) {
    MKL_INT n_spots = znorm_data->nrows;
    MKL_INT n_genes = znorm_data->ncols;
    
    /* Open output file */
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s' for writing: ", output_file);
        perror("");
        return;
    }
    
    /* Prepare for vectorized calculation */
    double* moran_values = (double*)mkl_malloc(n_genes * sizeof(double), 64);
    double* scaled_values = (double*)mkl_malloc(n_genes * sizeof(double), 64);
    double* s0_vector = (double*)mkl_malloc(n_genes * sizeof(double), 64);
    
    if (!moran_values || !scaled_values || !s0_vector) {
        fprintf(stderr, "Error: Memory allocation failed for vectorized calculation.\n");
        if (moran_values) mkl_free(moran_values);
        if (scaled_values) mkl_free(scaled_values);
        if (s0_vector) mkl_free(s0_vector);
        fclose(fp);
        return;
    }
    
    /* Ensure S0 is not zero */
    if (fabs(S0) < DBL_EPSILON) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Setting to small positive value.\n", S0);
        S0 = DBL_EPSILON;
    }
    
    /* Fill S0 vector for vectorized division */
    for (MKL_INT g = 0; g < n_genes; g++) {
        s0_vector[g] = S0;
    }
    
    /* Calculate Moran's I for each gene */
    #pragma omp parallel for
    for (MKL_INT g = 0; g < n_genes; g++) {
        /* Extract data for this gene */
        double* gene_data = (double*)malloc(n_spots * sizeof(double));
        if (!gene_data) {
            fprintf(stderr, "Error: Memory allocation failed for gene data in thread %d.\n", 
                    omp_get_thread_num());
            moran_values[g] = 0.0;
            continue;
        }
        
        for (MKL_INT i = 0; i < n_spots; i++) {
            gene_data[i] = znorm_data->values[i * n_genes + g];
        }
        
        /* Calculate Moran's I for this gene (unscaled) */
        moran_values[g] = calculate_single_gene_moran_i(gene_data, W, n_spots);
        
        free(gene_data);
    }
    
    /* Vectorized division by S0 */
    vdDiv(n_genes, moran_values, s0_vector, scaled_values);
    
    /* Write results to file */
    for (MKL_INT g = 0; g < n_genes; g++) {
        double value = scaled_values[g];
        
        /* Write to file (one value per line, no gene name) */
        if (isnan(value)) {
            fprintf(fp, "NaN\n");
        } else if (isinf(value)) {
            fprintf(fp, "%sInf\n", (value > 0 ? "" : "-"));
        } else if (fabs(value) < 0.0001 && value != 0.0) {
            fprintf(fp, "%.6e\n", value);
        } else {
            fprintf(fp, "%.8f\n", value);
        }
    }
    
    /* Clean up */
    mkl_free(moran_values);
    mkl_free(scaled_values);
    mkl_free(s0_vector);
    
    fclose(fp);
    printf("Single-gene Moran's I results saved to: %s\n", output_file);
}

/* Save first gene vs all results (one value per line, no headers) */
void save_first_gene_vs_all_results(double* morans_values, const char** gene_names, MKL_INT n_genes, const char* output_file) {
    /* Open output file */
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s' for writing: ", output_file);
        perror("");
        return;
    }
    
    printf("Saving pre-scaled first gene vs all Moran's I results...\n");
    
    /* Write each value (one per line, no headers) */
    for (MKL_INT g = 0; g < n_genes; g++) {
        if (isnan(morans_values[g])) {
            fprintf(fp, "NaN\n");
        } else if (isinf(morans_values[g])) {
            fprintf(fp, "%sInf\n", (morans_values[g] > 0 ? "" : "-"));
        } else if (fabs(morans_values[g]) < 0.0001 && morans_values[g] != 0.0) {
            fprintf(fp, "%.6e\n", morans_values[g]);
        } else {
            fprintf(fp, "%.8f\n", morans_values[g]);
        }
    }
    
    fclose(fp);
    printf("First gene vs all Moran's I results saved to: %s\n", output_file);
}


DenseMatrix* read_vst_file(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_size;
    MKL_INT n_rows = 0;
    MKL_INT n_cols = 0;
    DenseMatrix* matrix = NULL;
    int has_leading_tab = 0;  // Flag to track if header has leading tab

    // --- First Pass: count rows & columns ---
    printf("Starting first pass to determine matrix dimensions...\n");

    // 1) Read header line
    line_size = getline(&line, &line_buf_size, fp);
    if (line_size <= 0) {
        fprintf(stderr, "Error: Empty or unreadable header in '%s'.\n", filename);
        fclose(fp); free(line);
        return NULL;
    }
    // strip any trailing CR/LF
    while (line_size > 0 && (line[line_size - 1] == '\n' || line[line_size - 1] == '\r')) {
        line[--line_size] = '\0';
    }

    // 2) Count data columns = number of tabs in the header
    n_cols = 0;
    // Check if file starts with tab
    has_leading_tab = (line[0] == '\t');
    
    for (char* p = line; *p; ++p) {
        if (*p == '\t') n_cols++;
    }
    
    // Adjust column count based on leading tab presence
    if (has_leading_tab) {
        // If there's a leading tab, we have one less data column than tabs
        // (since the first tab is just before the first column name)
    } else {
        // If there's no leading tab, we have one more column than tabs
        // (the first column name has no tab before it)
        n_cols++;
    }
    
    if (n_cols <= 0) {
        fprintf(stderr, "Error: No data columns detected in '%s'.\n", filename);
        fclose(fp); free(line);
        return NULL;
    }

    // 3) Count data rows (non-empty lines after header)
    while ((line_size = getline(&line, &line_buf_size, fp)) > 0) {
        // skip blank lines
        int has_char = 0;
        for (ssize_t i = 0; i < line_size; ++i) {
            if (!isspace((unsigned char)line[i])) { has_char = 1; break; }
        }
        if (has_char) n_rows++;
    }
    if (n_rows == 0) {
        fprintf(stderr, "Error: No data rows found in '%s'.\n", filename);
        fclose(fp); free(line);
        return NULL;
    }
    printf("First pass complete: %lld rows × %lld columns\n",
           (long long)n_rows, (long long)n_cols);

    // --- Allocate matrix ---
    matrix = malloc(sizeof(DenseMatrix));
    if (!matrix) { perror("malloc"); fclose(fp); free(line); return NULL; }
    matrix->nrows = n_rows;
    matrix->ncols = n_cols;
    matrix->values  = mkl_malloc((size_t)n_rows * n_cols * sizeof(double), 64);
    matrix->rownames = calloc(n_rows, sizeof(char*));
    matrix->colnames = calloc(n_cols, sizeof(char*));
    if (!matrix->values || !matrix->rownames || !matrix->colnames) {
        perror("Allocation");
        free_dense_matrix(matrix);
        fclose(fp); free(line);
        return NULL;
    }

    // --- Second Pass: read data ---
    rewind(fp);
    free(line);
    line = NULL;
    line_buf_size = 0;

    // 1) Read header again for column names
    line_size = getline(&line, &line_buf_size, fp);
    while (line_size > 0 && (line[line_size - 1] == '\n' || line[line_size - 1] == '\r')) {
        line[--line_size] = '\0';
    }
    if (line_size <= 0) {
        fprintf(stderr, "Error: Could not re-read header in '%s'.\n", filename);
        free(line);
        free_dense_matrix(matrix);
        fclose(fp);
        return NULL;
    }

    // Check again if file starts with tab (just to be sure)
    has_leading_tab = (line[0] == '\t');
    
    // parse with strsep so empty fields are counted
    char* dup = strdup(line);
    if (!dup) { perror("strdup"); free(line); free_dense_matrix(matrix); fclose(fp); return NULL; }
    char* ptr = dup;
    char* tok;
    MKL_INT col_idx = 0;
    
    while ((tok = strsep(&ptr, "\t")) != NULL) {
        if (col_idx == 0 && has_leading_tab) {
            // Skip the first empty field if there's a leading tab
        } else if (col_idx < n_cols) {
            matrix->colnames[col_idx - (has_leading_tab ? 1 : 0)] = strdup(tok);
            if (!matrix->colnames[col_idx - (has_leading_tab ? 1 : 0)]) {
                perror("strdup");
                free(dup);
                free(line);
                free_dense_matrix(matrix);
                fclose(fp);
                return NULL;
            }
        }
        col_idx++;
    }
    free(dup);
    
    // Verify column count
    if ((has_leading_tab && col_idx != n_cols + 1) || 
        (!has_leading_tab && col_idx != n_cols)) {
        fprintf(stderr,
                "Error: Header in '%s' has %lld columns, expected %lld.\n",
                filename, 
                (long long)(col_idx - (has_leading_tab ? 1 : 0)), 
                (long long)n_cols);
        free(line);
        free_dense_matrix(matrix);
        fclose(fp);
        return NULL;
    }

    // 2) Read each data row
    MKL_INT row_idx = 0;
    MKL_INT lineno  = 1; // header
    while ((line_size = getline(&line, &line_buf_size, fp)) > 0) {
        lineno++;
        // strip newline
        while (line_size > 0 && (line[line_size - 1] == '\n' || line[line_size - 1] == '\r')) {
            line[--line_size] = '\0';
        }
        // skip blank
        int has_char = 0;
        for (ssize_t i = 0; i < line_size; ++i) {
            if (!isspace((unsigned char)line[i])) { has_char = 1; break; }
        }
        if (!has_char) continue;

        if (row_idx >= n_rows) {
            fprintf(stderr, "Warning: extra row at line %d, skipping.\n", lineno);
            break;
        }

        char* rowdup = strdup(line);
        if (!rowdup) { perror("strdup"); break; }
        char* rp = rowdup;
        char* cell = strsep(&rp, "\t");
        matrix->rownames[row_idx] = strdup(cell);
        if (!matrix->rownames[row_idx]) {
            perror("strdup");
            free(rowdup);
            break;
        }

        MKL_INT c = 0;
        while (c < n_cols && (cell = strsep(&rp, "\t")) != NULL) {
            char* endptr;
            errno = 0;
            double v = strtod(cell, &endptr);
            if (endptr == cell || (*endptr && !isspace((unsigned char)*endptr)) || errno == ERANGE) {
                fprintf(stderr,
                        "Error: bad number '%s' at line %d, col %d\n",
                        cell, lineno, c+1);
                free(rowdup);
                goto fail;
            }
            matrix->values[row_idx * n_cols + c] = v;
            c++;
        }
        free(rowdup);
        if (c != n_cols) {
            fprintf(stderr, "Error: expected %lld values at line %d, got %d\n",
                    (long long)n_cols, lineno, c);
            goto fail;
        }
        row_idx++;
    }

    if (row_idx != n_rows) {
        fprintf(stderr,
                "Error: read %d of %lld expected rows in '%s'\n",
                row_idx, (long long)n_rows, filename);
        goto fail;
    }

    free(line);
    fclose(fp);
    printf("Successfully loaded %lld×%lld from '%s'.\n",
           (long long)n_rows, (long long)n_cols, filename);
    return matrix;

fail:
    free(line);
    free_dense_matrix(matrix);
    fclose(fp);
    return NULL;
}

/* Free dense matrix */
void free_dense_matrix(DenseMatrix* matrix) {
    if (!matrix) return;

    if (matrix->values) {
        mkl_free(matrix->values);
        matrix->values = NULL;
    }

    if (matrix->rownames) {
        for (MKL_INT i = 0; i < matrix->nrows; i++) {
            free(matrix->rownames[i]);
        }
        free(matrix->rownames);
        matrix->rownames = NULL;
    }

    if (matrix->colnames) {
        for (MKL_INT i = 0; i < matrix->ncols; i++) {
            free(matrix->colnames[i]);
        }
        free(matrix->colnames);
        matrix->colnames = NULL;
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
            free(matrix->rownames[i]);
        }
        free(matrix->rownames);
    }

    if (matrix->colnames) {
        for (MKL_INT i = 0; i < matrix->ncols; i++) {
            free(matrix->colnames[i]);
        }
        free(matrix->colnames);
    }

    free(matrix);
}

/* Free spot coordinates */
void free_spot_coordinates(SpotCoordinates* coords) {
    if (!coords) return;

    free(coords->spot_row);
    free(coords->spot_col);
    free(coords->valid_mask);

    if (coords->spot_names) {
        for (MKL_INT i = 0; i < coords->total_spots; i++) {
            free(coords->spot_names[i]);
        }
        free(coords->spot_names);
    }

    free(coords);
}

/* Helper to print MKL sparse status */
void print_mkl_status(sparse_status_t status, const char* function_name) {
     if (status == SPARSE_STATUS_SUCCESS) return;

     fprintf(stderr, "Error: MKL Sparse BLAS function '%s' failed with status: ", function_name);
     switch(status) {
         case SPARSE_STATUS_NOT_INITIALIZED: fprintf(stderr, "SPARSE_STATUS_NOT_INITIALIZED\n"); break;
         case SPARSE_STATUS_ALLOC_FAILED:    fprintf(stderr, "SPARSE_STATUS_ALLOC_FAILED\n"); break;
         case SPARSE_STATUS_INVALID_VALUE:   fprintf(stderr, "SPARSE_STATUS_INVALID_VALUE\n"); break;
         case SPARSE_STATUS_EXECUTION_FAILED:fprintf(stderr, "SPARSE_STATUS_EXECUTION_FAILED\n"); break;
         case SPARSE_STATUS_INTERNAL_ERROR:  fprintf(stderr, "SPARSE_STATUS_INTERNAL_ERROR\n"); break;
         case SPARSE_STATUS_NOT_SUPPORTED:   fprintf(stderr, "SPARSE_STATUS_NOT_SUPPORTED\n"); break;
         default:                            fprintf(stderr, "Unknown MKL Sparse Status (%d)\n", status); break;
     }
}

/* Parse numeric parameters safely */
int load_positive_value(const char* value_str, const char* param, unsigned int min, unsigned int max) {
    char* endptr;
    long value = strtol(value_str, &endptr, 10);
    
    if (*endptr != '\0' || value < min || value > max) {
        fprintf(stderr, "Error: Parameter %s must be between %u and %u.\n", param, min, max);
        exit(1);
    }
    
    return (int)value;
}

/* Infer sigma from data for single-cell datasets */
double infer_sigma_from_data(SpotCoordinates* coords, double coord_scale) {
    if (coords->valid_spots < 2) {
        fprintf(stderr, "Warning: Not enough valid spots to infer sigma, using default.\n");
        return 100.0; /* Default fallback */
    }
    
    double sum_min_dist = 0.0;
    int count = 0;
    
    /* Sample a subset of cells if there are too many (for performance) */
    int max_samples = 1000;
    int sample_step = coords->valid_spots > max_samples ? coords->valid_spots / max_samples : 1;
    
    printf("Inferring sigma from data: sampling %d spots...\n", 
           coords->valid_spots / sample_step);
    
    /* For each valid spot, find its nearest neighbor */
    for (MKL_INT i = 0; i < coords->total_spots; i += sample_step) {
        if (!coords->valid_mask[i]) continue;
        
        double min_dist = DBL_MAX;
        
        for (MKL_INT j = 0; j < coords->total_spots; j++) {
            if (i == j || !coords->valid_mask[j]) continue;
            
            /* Get real coordinates (divide by scale to get original units) */
            double x_i = coords->spot_col[i] / coord_scale;
            double y_i = coords->spot_row[i] / coord_scale;
            double x_j = coords->spot_col[j] / coord_scale;
            double y_j = coords->spot_row[j] / coord_scale;
            
            /* Calculate Euclidean distance */
            double dx = x_i - x_j;
            double dy = y_i - y_j;
            double dist = sqrt(dx*dx + dy*dy);
            
            if (dist < min_dist) min_dist = dist;
        }
        
        if (min_dist < DBL_MAX) {
            sum_min_dist += min_dist;
            count++;
        }
    }
    
    if (count == 0) {
        fprintf(stderr, "Warning: Could not calculate nearest neighbor distances, using default sigma.\n");
        return 100.0;
    }
    
    /* Calculate average nearest neighbor distance */
    double avg_nn_dist = sum_min_dist / count;
    
    /* Set sigma to approximately the average nearest neighbor distance */
    double sigma = avg_nn_dist;
    
    printf("Inferred sigma = %.2f based on average nearest neighbor distance\n", sigma);
    
    return sigma;
}

/* Gaussian distance decay function */
double decay(double d, double sigma) {
    if (d < 0.0) d = 0.0;
    
    /* Cut off weights beyond 2*sigma distance */
    if (d > 2.0 * sigma) return 0.0;
    
    /* RBF kernel as described in the paper: exp(-d^2/(2*sigma^2)) */
    return exp(-(d * d) / (2.0 * sigma * sigma));
}

/* Create spatial distance matrix with platform-specific parameters */
DenseMatrix* create_distance_matrix(MKL_INT max_radius, int platform_mode, double custom_sigma) {
    DenseMatrix* matrix = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!matrix) {
        perror("Failed to allocate DenseMatrix structure for decay matrix");
        return NULL;
    }
    matrix->nrows = max_radius;
    matrix->ncols = 2 * max_radius;
    matrix->rownames = NULL;
    matrix->colnames = NULL;
    matrix->values = (double*)mkl_malloc(matrix->nrows * matrix->ncols * sizeof(double), 64);
    if (!matrix->values) {
        perror("Failed to allocate values for decay matrix");
        free(matrix);
        return NULL;
    }

    /* Set platform-specific parameters */
    double shift_factor_y = (platform_mode == VISIUM) ? (0.5 * sqrt(3.0)) : 0.5;
    double dist_unit = (platform_mode == VISIUM) ? 100.0 : 200.0;
    
    /* For single-cell data, use custom sigma if provided */
    double sigma = custom_sigma;
    if (sigma <= 0.0) {
        /* Default values based on platform */
        sigma = (platform_mode == VISIUM) ? 100.0 : 
               ((platform_mode == OLD) ? 200.0 : 100.0); /* Default for single-cell */
    }
    
    printf("Creating distance matrix with dimensions %lldx%lld (max_row_shift x max_col_shift)...\n",
           (long long)matrix->nrows, (long long)matrix->ncols);
    printf("Using sigma = %.2f for RBF kernel\n", sigma);

    #pragma omp parallel for collapse(2)
    for (MKL_INT i = 0; i < matrix->nrows; i++) {
        for (MKL_INT j = 0; j < matrix->ncols; j++) {
            double x_dist = 0.5 * j * dist_unit;
            double y_dist = i * shift_factor_y * dist_unit;
            double d = sqrt(x_dist * x_dist + y_dist * y_dist);
            matrix->values[i * matrix->ncols + j] = decay(d, sigma);
        }
    }
    printf("Distance decay matrix created.\n");
    return matrix;
}

/* Extract coordinates from column names */
SpotCoordinates* extract_coordinates(char** column_names, MKL_INT n_columns) {
    SpotCoordinates* coords = (SpotCoordinates*)malloc(sizeof(SpotCoordinates));
     if (!coords) {
        perror("Failed to allocate SpotCoordinates structure");
        return NULL;
    }
    coords->total_spots = n_columns;
    coords->spot_row = (MKL_INT*)malloc(n_columns * sizeof(MKL_INT));
    coords->spot_col = (MKL_INT*)malloc(n_columns * sizeof(MKL_INT));
    coords->valid_mask = (int*)calloc(n_columns, sizeof(int));
    coords->spot_names = (char**)malloc(n_columns * sizeof(char*));
    coords->valid_spots = 0;

    if (!coords->spot_row || !coords->spot_col || !coords->valid_mask || !coords->spot_names) {
        perror("Failed to allocate memory for coordinate arrays");
        free(coords->spot_row); free(coords->spot_col); free(coords->valid_mask); free(coords->spot_names);
        free(coords);
        return NULL;
    }

    regex_t regex;
    regmatch_t matches[3];

    int reti = regcomp(&regex, "^([0-9]+)x([0-9]+)$", REG_EXTENDED);
    if (reti) {
        char errbuf[100];
        regerror(reti, &regex, errbuf, sizeof(errbuf));
        fprintf(stderr, "Error: Could not compile regex: %s\n", errbuf);
        free(coords->spot_row); free(coords->spot_col); free(coords->valid_mask); free(coords->spot_names);
        free(coords);
        return NULL;
    }

    printf("Extracting coordinates using regex ^([0-9]+)x([0-9]+)$ ...\n");
    MKL_INT valid_count = 0;
    for (MKL_INT i = 0; i < n_columns; i++) {
        coords->spot_names[i] = strdup(column_names[i]);
        coords->spot_row[i] = -1;
        coords->spot_col[i] = -1;
        coords->valid_mask[i] = 0;

        reti = regexec(&regex, column_names[i], 3, matches, 0);
        if (reti == 0) {
            char row_str[32];
            size_t row_len = matches[1].rm_eo - matches[1].rm_so;
            if (row_len < sizeof(row_str)) {
                strncpy(row_str, column_names[i] + matches[1].rm_so, row_len);
                row_str[row_len] = '\0';
                coords->spot_row[i] = atoll(row_str);
            } else { continue; }

            char col_str[32];
            size_t col_len = matches[2].rm_eo - matches[2].rm_so;
            if (col_len < sizeof(col_str)) {
                 strncpy(col_str, column_names[i] + matches[2].rm_so, col_len);
                 col_str[col_len] = '\0';
                 coords->spot_col[i] = atoll(col_str);
            } else { continue; }

            if (coords->spot_row[i] >= 0 && coords->spot_col[i] >= 0) {
                 coords->valid_mask[i] = 1;
                 valid_count++;
            } else {
                coords->spot_row[i] = -1;
                coords->spot_col[i] = -1;
            }
        } else if (reti != REG_NOMATCH) {
            char errbuf[100];
            regerror(reti, &regex, errbuf, sizeof(errbuf));
            fprintf(stderr, "Warning: Regex match failed for '%s': %s\n", column_names[i], errbuf);
        }
    }

    regfree(&regex);
    coords->valid_spots = valid_count;
    printf("Coordinate extraction complete.\n");
    return coords;
}

/* Process double parameter safely */
double load_double_value(const char* value_str, const char* param) {
    char* endptr;
    double value = strtod(value_str, &endptr);
    
    if (*endptr != '\0' || !isfinite(value)) {
        fprintf(stderr, "Error: Parameter %s must be a valid floating-point number.\n", param);
        exit(1);
    }
    
    return value;
}

/* Read coordinates from TSV file for single-cell data */
SpotCoordinates* read_coordinates_file(const char* filename, const char* id_column, 
                                     const char* x_column, const char* y_column,
                                     double coord_scale) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open coordinates file '%s': %s\n", 
                filename, strerror(errno));
        return NULL;
    }
    
    /* First pass: Count lines and find column indices */
    char *line = NULL;
    size_t line_buf_size = 0;
    
    /* Read header to find column positions */
    ssize_t line_size = getline(&line, &line_buf_size, fp);
    if (line_size <= 0) {
        fprintf(stderr, "Error: Empty or unreadable header in '%s'.\n", filename);
        fclose(fp); free(line);
        return NULL;
    }
    
    /* Parse header to find column indices */
    int id_col_idx = -1, x_col_idx = -1, y_col_idx = -1;
    char *header_copy = strdup(line);
    char *token = strtok(header_copy, "\t");  /* Changed delimiter to tab */
    int col_idx = 0;
    
    while (token != NULL) {
        /* Strip whitespace and quotes */
        while (*token && isspace((unsigned char)*token)) token++;
        if (*token == '"') token++;
        
        char *end = token + strlen(token) - 1;
        while (end > token && isspace((unsigned char)*end)) end--;
        if (*end == '"' && end > token) end--;
        *(end + 1) = '\0';
        
        if (strcmp(token, id_column) == 0) id_col_idx = col_idx;
        else if (strcmp(token, x_column) == 0) x_col_idx = col_idx;
        else if (strcmp(token, y_column) == 0) y_col_idx = col_idx;
        
        token = strtok(NULL, "\t");  /* Changed delimiter to tab */
        col_idx++;
    }
    free(header_copy);
    
    if (id_col_idx < 0 || x_col_idx < 0 || y_col_idx < 0) {
        fprintf(stderr, "Error: Could not find required columns in coordinates file.\n");
        fprintf(stderr, "  ID column '%s': %s\n", id_column, id_col_idx < 0 ? "Not found" : "Found");
        fprintf(stderr, "  X column '%s': %s\n", x_column, x_col_idx < 0 ? "Not found" : "Found");
        fprintf(stderr, "  Y column '%s': %s\n", y_column, y_col_idx < 0 ? "Not found" : "Found");
        fclose(fp); free(line);
        return NULL;
    }
    
    /* Count number of data rows */
    MKL_INT n_spots = 0;
    while ((line_size = getline(&line, &line_buf_size, fp)) > 0) {
        n_spots++;
    }
    
    /* Allocate structure */
    SpotCoordinates* coords = (SpotCoordinates*)malloc(sizeof(SpotCoordinates));
    if (!coords) {
        perror("Failed to allocate SpotCoordinates structure");
        fclose(fp); free(line);
        return NULL;
    }
    
    coords->total_spots = n_spots;
    coords->spot_row = (MKL_INT*)malloc(n_spots * sizeof(MKL_INT));
    coords->spot_col = (MKL_INT*)malloc(n_spots * sizeof(MKL_INT));
    coords->valid_mask = (int*)calloc(n_spots, sizeof(int));
    coords->spot_names = (char**)malloc(n_spots * sizeof(char*));
    coords->valid_spots = 0;
    
    if (!coords->spot_row || !coords->spot_col || !coords->valid_mask || !coords->spot_names) {
        perror("Failed to allocate memory for coordinate arrays");
        free(coords->spot_row); free(coords->spot_col); 
        free(coords->valid_mask); free(coords->spot_names);
        free(coords); fclose(fp); free(line);
        return NULL;
    }
    
    /* Second pass: read coordinates */
    rewind(fp);
    getline(&line, &line_buf_size, fp); /* Skip header */
    
    MKL_INT spot_idx = 0;
    while ((line_size = getline(&line, &line_buf_size, fp)) > 0 && spot_idx < n_spots) {
        /* Parse TSV line */
        char *line_copy = strdup(line);
        char *token = strtok(line_copy, "\t");  /* Changed delimiter to tab */
        int col_idx = 0;
        
        char *id_value = NULL;
        double x_value = 0.0, y_value = 0.0;
        int has_valid_coords = 0;
        
        while (token != NULL) {
            /* Strip whitespace and quotes */
            while (*token && isspace((unsigned char)*token)) token++;
            if (*token == '"') token++;
            
            char *end = token + strlen(token) - 1;
            while (end > token && isspace((unsigned char)*end)) end--;
            if (*end == '"' && end > token) end--;
            *(end + 1) = '\0';
            
            if (col_idx == id_col_idx) {
                id_value = strdup(token);
            } else if (col_idx == x_col_idx) {
                char *endptr;
                x_value = strtod(token, &endptr);
                if (endptr == token || *endptr != '\0') has_valid_coords = 0;
                else has_valid_coords = 1;
            } else if (col_idx == y_col_idx) {
                char *endptr;
                y_value = strtod(token, &endptr);
                if (endptr == token || *endptr != '\0') has_valid_coords = 0;
                else if (has_valid_coords) has_valid_coords = 1; /* Both X and Y are valid */
            }
            
            token = strtok(NULL, "\t");  /* Changed delimiter to tab */
            col_idx++;
        }
        free(line_copy);
        
        /* Store values */
        if (id_value && has_valid_coords) {
            coords->spot_names[spot_idx] = id_value;
            
            /* Convert floating-point coordinates to integer grid */
            /* Scale and round them to fit our grid system */
            coords->spot_row[spot_idx] = (MKL_INT)round(y_value * coord_scale);
            coords->spot_col[spot_idx] = (MKL_INT)round(x_value * coord_scale);
            coords->valid_mask[spot_idx] = 1;
            coords->valid_spots++;
        } else {
            if (id_value) free(id_value);
            coords->spot_names[spot_idx] = strdup("unknown");
            coords->spot_row[spot_idx] = -1;
            coords->spot_col[spot_idx] = -1;
            coords->valid_mask[spot_idx] = 0;
        }
        
        spot_idx++;
    }
    
    free(line);
    fclose(fp);
    printf("Read %lld coordinates from file, %lld are valid.\n", 
           (long long)coords->total_spots, (long long)coords->valid_spots);
    return coords;
}

/* Map expression matrix columns to coordinate spots */
int map_expression_to_coordinates(DenseMatrix* expr_matrix, SpotCoordinates* coords, 
                                 MKL_INT** mapping) {
    /* Create a mapping between expression matrix columns and coordinate spots */
    *mapping = (MKL_INT*)malloc(coords->valid_spots * sizeof(MKL_INT));
    if (!(*mapping)) {
        perror("Failed to allocate mapping array");
        return 0;
    }
    
    /* Initialize all mappings to -1 (invalid) */
    for (MKL_INT i = 0; i < coords->valid_spots; i++) {
        (*mapping)[i] = -1;
    }
    
    /* Count valid spots with expression data */
    MKL_INT valid_count = 0;
    MKL_INT valid_index = 0;
    
    /* For each valid coordinate spot, find matching expression column */
    for (MKL_INT i = 0; i < coords->total_spots; i++) {
        if (!coords->valid_mask[i]) continue;
        
        /* For each expression column, check if the name matches */
        for (MKL_INT j = 0; j < expr_matrix->ncols; j++) {
            if (strcmp(coords->spot_names[i], expr_matrix->colnames[j]) == 0) {
                (*mapping)[valid_index] = j;
                valid_count++;
                break;
            }
        }
        valid_index++;
    }
    
    printf("Found expression data for %lld of %lld valid coordinates.\n",
           (long long)valid_count, (long long)coords->valid_spots);
    
    return (valid_count > 0);
}