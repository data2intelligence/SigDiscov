/*
 * morans_i_spatial.c
 *
 * Spatial module for Moran's I computation.
 * Contains spatial weight matrix construction, distance/decay functions,
 * and coordinate extraction utilities.
 *
 * Split from morans_i_mkl.c
 */

#include "morans_i_internal.h"

/* ===============================
 * SPATIAL WEIGHT MATRIX FUNCTIONS
 * =============================== */

/* Forward declaration for merge_thread_coo (used inside parallel_compute_coo) */
static void merge_thread_coo(const MKL_INT* local_I, const MKL_INT* local_J, const double* local_V,
                             MKL_INT local_nnz,
                             MKL_INT** global_I, MKL_INT** global_J, double** global_V,
                             MKL_INT* global_nnz, MKL_INT* current_capacity,
                             MKL_INT n_spots_valid, volatile int* critical_error_flag);

/*
 * parallel_compute_coo() -- OpenMP parallel section that computes COO triplets
 * from spot coordinates and a distance matrix.
 *
 * On success, *out_I, *out_J, *out_V hold the COO arrays and *out_nnz their count.
 * Caller must free(*out_I), free(*out_J), free(*out_V) on success.
 * Returns 0 on success, -1 on error (arrays freed internally on error).
 */
static int parallel_compute_coo(const MKL_INT* spot_row_valid, const MKL_INT* spot_col_valid,
                                MKL_INT n_spots_valid, const DenseMatrix* distance_matrix,
                                MKL_INT estimated_neighbors_per_spot,
                                MKL_INT** out_I, MKL_INT** out_J, double** out_V,
                                MKL_INT* out_nnz) {

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
        return -1;
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
            merge_thread_coo(local_I_tl, local_J_tl, local_V_tl, local_nnz_tl,
                             &temp_I, &temp_J, &temp_V, &nnz_count, &current_capacity,
                             n_spots_valid, &critical_error_flag);
        }

        // Free thread-local buffers
        if(local_I_tl) free(local_I_tl);
        if(local_J_tl) free(local_J_tl);
        if(local_V_tl) free(local_V_tl);
    } // End of omp parallel

    if (critical_error_flag) {
        fprintf(stderr, "Error: A critical error occurred during parallel COO matrix construction.\n");
        free(temp_I); free(temp_J); free(temp_V);
        return -1;
    }

    *out_I = temp_I;
    *out_J = temp_J;
    *out_V = temp_V;
    *out_nnz = nnz_count;
    return 0;
}

/*
 * merge_thread_coo() -- Merge thread-local COO triplets into the global arrays.
 * Contains its own OpenMP critical section for thread-safe merging.
 *
 * On error, sets *critical_error_flag to 1.
 */
static void merge_thread_coo(const MKL_INT* local_I, const MKL_INT* local_J, const double* local_V,
                             MKL_INT local_nnz,
                             MKL_INT** global_I, MKL_INT** global_J, double** global_V,
                             MKL_INT* global_nnz, MKL_INT* current_capacity,
                             MKL_INT n_spots_valid, volatile int* critical_error_flag) {
    #pragma omp critical
    {
        if (!(*critical_error_flag)) { // Re-check global error flag inside critical section
            if (*global_nnz + local_nnz > *current_capacity) { // Resize global buffer if needed
                MKL_INT needed_capacity = *global_nnz + local_nnz;
                MKL_INT new_global_capacity = *current_capacity;
                while(new_global_capacity < needed_capacity && new_global_capacity > 0) { // Prevent overflow with new_global_capacity > 0
                    new_global_capacity = (MKL_INT)(new_global_capacity * 1.5) + 1;
                    if (new_global_capacity <= *current_capacity) { // Overflow or no increase
                        new_global_capacity = needed_capacity > *current_capacity ? needed_capacity : *current_capacity + 1; // Try to reach at least needed
                        if (new_global_capacity <= *current_capacity) { // Still stuck, indicates potential overflow
                            *critical_error_flag = 1; break;
                        }
                    }
                }
                if(*critical_error_flag) {
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
                    *critical_error_flag = 1;
                } else if (n_spots_valid > 0 && !(*critical_error_flag)) {
                    printf("  Resizing global COO buffer from %lld to %lld\n",
                           (long long)*current_capacity, (long long)new_global_capacity);
                    MKL_INT* temp_gi_new = (MKL_INT*)realloc(*global_I, (size_t)new_global_capacity * sizeof(MKL_INT));
                    MKL_INT* temp_gj_new = (MKL_INT*)realloc(*global_J, (size_t)new_global_capacity * sizeof(MKL_INT));
                    double*  temp_gv_new = (double*)realloc(*global_V, (size_t)new_global_capacity * sizeof(double));

                    if (!temp_gi_new || !temp_gj_new || !temp_gv_new) {
                        fprintf(stderr, "Error: Failed to realloc global COO buffers.\n");
                        *critical_error_flag = 1;
                        // Keep old pointers if realloc failed, to free them later
                        *global_I = temp_gi_new ? temp_gi_new : *global_I;
                        *global_J = temp_gj_new ? temp_gj_new : *global_J;
                        *global_V = temp_gv_new ? temp_gv_new : *global_V;
                    } else {
                        *global_I = temp_gi_new;
                        *global_J = temp_gj_new;
                        *global_V = temp_gv_new;
                        *current_capacity = new_global_capacity;
                    }
                }
            }

            // Copy local data to global arrays if no critical error and space is sufficient
            if (!(*critical_error_flag) && (*global_nnz + local_nnz <= *current_capacity)) {
                memcpy(*global_I + *global_nnz, local_I, (size_t)local_nnz * sizeof(MKL_INT));
                memcpy(*global_J + *global_nnz, local_J, (size_t)local_nnz * sizeof(MKL_INT));
                memcpy(*global_V + *global_nnz, local_V, (size_t)local_nnz * sizeof(double));
                *global_nnz += local_nnz;
            } else if (!(*critical_error_flag)) { // Should not happen if resize logic is correct
                fprintf(stderr, "Warning: Could not merge thread %d results due to insufficient space after resize attempt.\n", omp_get_thread_num());
                *critical_error_flag = 1; // Treat as critical if merge fails
            }
        } // End of if (!*critical_error_flag) inside critical section
    } // End of omp critical
}

/*
 * coo_to_csr() -- Convert COO triplets to a SparseMatrix in CSR format.
 *
 * On success, returns a newly allocated SparseMatrix. The COO arrays
 * (temp_I, temp_J, temp_V) are freed by this function regardless of outcome.
 * Returns NULL on allocation failure.
 */
static SparseMatrix* coo_to_csr(MKL_INT* temp_I, MKL_INT* temp_J, double* temp_V,
                                MKL_INT nnz_count, MKL_INT n_spots_valid) {

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
    return W;
}

/*
 * apply_row_normalization() -- Normalize each row of a CSR sparse matrix
 * so that the row sums to 1.
 */
static void apply_row_normalization(SparseMatrix* W) {
    if (!W || W->nnz <= 0) return;

    MKL_INT n_spots_valid = W->nrows;

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

    /* Step 1: Parallel COO computation */
    MKL_INT* temp_I = NULL;
    MKL_INT* temp_J = NULL;
    double* temp_V = NULL;
    MKL_INT nnz_count = 0;

    if (parallel_compute_coo(spot_row_valid, spot_col_valid, n_spots_valid, distance_matrix,
                             estimated_neighbors_per_spot,
                             &temp_I, &temp_J, &temp_V, &nnz_count) != 0) {
        return NULL;
    }

    printf("  Generated %lld non-zero entries (COO format).\n", (long long)nnz_count);
    if (nnz_count == 0 && n_spots_valid > 0) {
        printf("Warning: No non-zero weights found. Moran's I will likely be zero or undefined.\n");
    }

    /* Step 2: Convert COO to CSR (frees temp_I, temp_J, temp_V) */
    SparseMatrix* W = coo_to_csr(temp_I, temp_J, temp_V, nnz_count, n_spots_valid);
    if (!W) {
        return NULL;
    }

    /* Step 3: Optional row normalization */
    if (row_normalize) {
        apply_row_normalization(W);
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
