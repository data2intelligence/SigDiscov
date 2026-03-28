/* toy_example.c - Toy example functions for Moran's I verification
 *
 * Contains 2D grid toy example for testing and validating the
 * Moran's I calculation pipeline.
 */

#include "morans_i_internal.h"

/* ============================================================================
 * TOY EXAMPLE FUNCTIONS
 * ============================================================================ */

/* Helper function for toy examples */
MKL_INT grid_to_1d_idx(MKL_INT r, MKL_INT c, MKL_INT num_grid_cols) {
    return r * num_grid_cols + c;
}

DenseMatrix* create_theoretical_toy_moran_i_matrix_2d(MKL_INT n_genes, char** gene_names) {
    if (n_genes != 5) {
        fprintf(stderr, "Error (create_theoretical_toy_moran_i): This function is hardcoded for 5 genes.\n");
        return NULL;
    }

    DenseMatrix* theoretical_I = (DenseMatrix*)calloc(1, sizeof(DenseMatrix));
    if (!theoretical_I) {
        perror("calloc theoretical_I");
        return NULL;
    }

    theoretical_I->nrows = n_genes;
    theoretical_I->ncols = n_genes;
    theoretical_I->values = (double*)mkl_calloc((size_t)n_genes * n_genes, sizeof(double), 64);
    theoretical_I->rownames = (char**)calloc(n_genes, sizeof(char*));
    theoretical_I->colnames = (char**)calloc(n_genes, sizeof(char*));

    if (!theoretical_I->values || !theoretical_I->rownames || !theoretical_I->colnames) {
        perror("mkl_calloc theoretical_I components");
        free_dense_matrix(theoretical_I);
        return NULL;
    }

    for (MKL_INT i = 0; i < n_genes; ++i) {
        theoretical_I->rownames[i] = strdup(gene_names[i]); // Use names from X_calc
        theoretical_I->colnames[i] = strdup(gene_names[i]);
        if (!theoretical_I->rownames[i] || !theoretical_I->colnames[i]) {
            perror("strdup theoretical_I gene names");
            free_dense_matrix(theoretical_I);
            return NULL;
        }
    }

    // Values based on our derivations/expectations:
    // G0: Row gradient
    // G1: Row gradient (identical to G0)
    // G2: Column gradient
    // G3: Checkerboard
    // G4: Radial

    // I(G0,G0)
    theoretical_I->values[0 * n_genes + 0] = 0.75;
    // I(G0,G1) - G0 and G1 are identical
    theoretical_I->values[0 * n_genes + 1] = 0.75;
    theoretical_I->values[1 * n_genes + 0] = 0.75;
    // I(G0,G2) - Row vs Col gradient (orthogonal)
    theoretical_I->values[0 * n_genes + 2] = 0.0;
    theoretical_I->values[2 * n_genes + 0] = 0.0;
    // I(G0,G3) - Row grad vs Checkerboard (expected near 0)
    theoretical_I->values[0 * n_genes + 3] = 0.0; // Approx.
    theoretical_I->values[3 * n_genes + 0] = 0.0; // Approx.
    // I(G0,G4) - Row grad vs Radial (expected near 0)
    theoretical_I->values[0 * n_genes + 4] = 0.0; // Approx.
    theoretical_I->values[4 * n_genes + 0] = 0.0; // Approx.

    // I(G1,G1)
    theoretical_I->values[1 * n_genes + 1] = 0.75;
    // I(G1,G2) - Row vs Col gradient (orthogonal)
    theoretical_I->values[1 * n_genes + 2] = 0.0;
    theoretical_I->values[2 * n_genes + 1] = 0.0;
    // I(G1,G3) - Row grad vs Checkerboard
    theoretical_I->values[1 * n_genes + 3] = 0.0; // Approx.
    theoretical_I->values[3 * n_genes + 1] = 0.0; // Approx.
    // I(G1,G4) - Row grad vs Radial
    theoretical_I->values[1 * n_genes + 4] = 0.0; // Approx.
    theoretical_I->values[4 * n_genes + 1] = 0.0; // Approx.

    // I(G2,G2) - Col gradient
    theoretical_I->values[2 * n_genes + 2] = 0.75;
    // I(G2,G3) - Col grad vs Checkerboard
    theoretical_I->values[2 * n_genes + 3] = 0.0; // Approx.
    theoretical_I->values[3 * n_genes + 2] = 0.0; // Approx.
    // I(G2,G4) - Col grad vs Radial
    theoretical_I->values[2 * n_genes + 4] = 0.0; // Approx.
    theoretical_I->values[4 * n_genes + 2] = 0.0; // Approx.

    // I(G3,G3) - Checkerboard
    theoretical_I->values[3 * n_genes + 3] = -1.0;
    // I(G3,G4) - Checkerboard vs Radial
    theoretical_I->values[3 * n_genes + 4] = 0.0; // Approx.
    theoretical_I->values[4 * n_genes + 3] = 0.0; // Approx.

    // I(G4,G4) - Radial Autocorrelation
    theoretical_I->values[4 * n_genes + 4] = 0.4; // Placeholder: Expected positive

    printf("Theoretical Moran's I expectation matrix created.\n");
    return theoretical_I;
}

SparseMatrix* create_toy_W_matrix_2d(MKL_INT num_grid_rows, MKL_INT num_grid_cols) {
    MKL_INT n_spots = num_grid_rows * num_grid_cols;
    if (n_spots == 0) {
        fprintf(stderr, "Error (create_toy_W_matrix_2d): Cannot create W for 0 spots.\n");
        return NULL;
    }

    SparseMatrix* W = (SparseMatrix*)calloc(1, sizeof(SparseMatrix));
    if (!W) {
        perror("calloc toy W 2D");
        return NULL;
    }

    W->nrows = n_spots;
    W->ncols = n_spots;

    MKL_INT max_possible_nnz = n_spots * 4;
    MKL_INT* temp_I = NULL;
    MKL_INT* temp_J = NULL;
    double* temp_V = NULL;
    MKL_INT* current_pos = NULL;

    if (max_possible_nnz > 0) {
        temp_I = (MKL_INT*)malloc(max_possible_nnz * sizeof(MKL_INT));
        temp_J = (MKL_INT*)malloc(max_possible_nnz * sizeof(MKL_INT));
        temp_V = (double*)malloc(max_possible_nnz * sizeof(double));
        if (!temp_I || !temp_J || !temp_V) {
            perror("malloc temp COO for toy W 2D");
            goto cleanup_toy_w_coo_2d;
        }
    }

    MKL_INT current_nnz = 0;
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    for (MKL_INT r = 0; r < num_grid_rows; ++r) {
        for (MKL_INT c = 0; c < num_grid_cols; ++c) {
            MKL_INT spot_idx_i = grid_to_1d_idx(r, c, num_grid_cols);
            for (int move = 0; move < 4; ++move) {
                MKL_INT nr = r + dr[move];
                MKL_INT nc = c + dc[move];
                if (nr >= 0 && nr < num_grid_rows && nc >= 0 && nc < num_grid_cols) {
                    MKL_INT spot_idx_j = grid_to_1d_idx(nr, nc, num_grid_cols);
                    if (temp_I && current_nnz < max_possible_nnz) {
                        temp_I[current_nnz] = spot_idx_i;
                        temp_J[current_nnz] = spot_idx_j;
                        temp_V[current_nnz] = 1.0;
                        current_nnz++;
                    } else if (temp_I) {
                        fprintf(stderr, "Error: Exceeded temp COO capacity for toy W 2D.\n");
                        goto cleanup_toy_w_coo_2d;
                    }
                }
            }
        }
    }
    W->nnz = current_nnz;

    W->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
    if (!W->row_ptr) {
        perror("mkl_calloc W->row_ptr for toy W 2D");
        goto cleanup_toy_w_coo_2d;
    }

    if (W->nnz > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc(W->nnz * sizeof(MKL_INT), 64);
        W->values  = (double*)mkl_malloc(W->nnz * sizeof(double), 64);
        if (!W->col_ind || !W->values) {
            perror("mkl_malloc W->col_ind/values for toy W 2D");
            goto cleanup_toy_w_coo_2d;
        }

        for (MKL_INT k = 0; k < W->nnz; ++k) W->row_ptr[temp_I[k] + 1]++;
        for (MKL_INT i = 0; i < n_spots; ++i) W->row_ptr[i + 1] += W->row_ptr[i];

        current_pos = (MKL_INT*)mkl_malloc((size_t)(n_spots + 1) * sizeof(MKL_INT), 64);
        if (!current_pos) {
            perror("mkl_malloc current_pos for CSR conversion");
            goto cleanup_toy_w_coo_2d;
        }
        memcpy(current_pos, W->row_ptr, (size_t)(n_spots + 1) * sizeof(MKL_INT));

        for (MKL_INT k = 0; k < W->nnz; ++k) {
            MKL_INT r_csr = temp_I[k];
            MKL_INT insert_idx = current_pos[r_csr];
            W->col_ind[insert_idx] = temp_J[k];
            W->values[insert_idx] = temp_V[k];
            current_pos[r_csr]++;
        }
    } else {
        W->col_ind = NULL;
        W->values = NULL;
    }

cleanup_toy_w_coo_2d:
    free(temp_I);
    free(temp_J);
    free(temp_V);
    mkl_free(current_pos);
    if ((W->row_ptr && W->nnz > 0 && (!W->col_ind || !W->values)) || (!W->row_ptr && n_spots > 0) ) {
        free_sparse_matrix(W);
        return NULL;
    }
    if (W->nnz > 0 && (!W->col_ind || !W->values)) {
         free_sparse_matrix(W);
         return NULL;
    }

    printf("Toy W matrix (2D Grid, Rook, %lldx%lld spots) created with %lld NNZ.\n",
           (long long)num_grid_rows, (long long)num_grid_cols, (long long)W->nnz);
    return W;
}

DenseMatrix* create_toy_X_calc_matrix_2d(MKL_INT num_grid_rows, MKL_INT num_grid_cols, MKL_INT n_genes) {
    MKL_INT n_spots = num_grid_rows * num_grid_cols;
    if (n_spots == 0 || n_genes == 0) {
        fprintf(stderr, "Error (create_toy_X_calc_2d): n_spots or n_genes is zero.\n");
        return NULL;
    }
    if (n_genes < 5 && n_genes > 0) {
        fprintf(stderr, "Warning (create_toy_X_calc_2d): Requested %lld genes, but patterns defined for up to 5. Some patterns might be omitted.\n",
                (long long)n_genes);
    }

    DenseMatrix* X = (DenseMatrix*)calloc(1, sizeof(DenseMatrix));
    if (!X) {
        perror("calloc toy X 2D");
        return NULL;
    }

    X->nrows = n_spots;
    X->ncols = n_genes;
    X->values = (double*)mkl_calloc((size_t)n_spots * n_genes, sizeof(double), 64);
    X->rownames = (char**)calloc(n_spots, sizeof(char*));
    X->colnames = (char**)calloc(n_genes, sizeof(char*));

    if (!X->values || !X->rownames || !X->colnames) {
        perror("mkl_calloc toy X 2D components");
        free_dense_matrix(X);
        return NULL;
    }

    for (MKL_INT r = 0; r < num_grid_rows; ++r) {
        for (MKL_INT c = 0; c < num_grid_cols; ++c) {
            MKL_INT spot_idx = grid_to_1d_idx(r, c, num_grid_cols);
            char name_buf[32];
            snprintf(name_buf, 32, "S_r%d_c%d", (int)r, (int)c);
            X->rownames[spot_idx] = strdup(name_buf);
            if (!X->rownames[spot_idx]) {
                perror("strdup toy spot name 2D");
                free_dense_matrix(X);
                return NULL;
            }
        }
    }
    for (MKL_INT j = 0; j < n_genes; ++j) {
        char name_buf[32];
        snprintf(name_buf, 32, "Gene%lld", (long long)j);
        X->colnames[j] = strdup(name_buf);
        if (!X->colnames[j]) {
            perror("strdup toy gene name 2D");
            free_dense_matrix(X);
            return NULL;
        }
    }

    // Gene0: Gradient along rows (increases with row index r)
    if (n_genes >= 1) {
        for (MKL_INT r = 0; r < num_grid_rows; ++r) {
            for (MKL_INT c = 0; c < num_grid_cols; ++c) {
                X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 0] = (double)r;
            }
        }
    }

    // Gene1: Identical to Gene0 (also row gradient)
    if (n_genes >= 2) {
        for (MKL_INT r = 0; r < num_grid_rows; ++r) {
            for (MKL_INT c = 0; c < num_grid_cols; ++c) {
                X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 1] = (double)r;
            }
        }
    }

    // Gene2: Gradient along columns (increases with col index c)
    if (n_genes >= 3) {
        for (MKL_INT r = 0; r < num_grid_rows; ++r) {
            for (MKL_INT c = 0; c < num_grid_cols; ++c) {
                X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 2] = (double)c;
            }
        }
    }

    // Gene3: Checkerboard pattern ((r+c) % 2)
    if (n_genes >= 4) {
        for (MKL_INT r = 0; r < num_grid_rows; ++r) {
            for (MKL_INT c = 0; c < num_grid_cols; ++c) {
                X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 3] =
                    ((r + c) % 2 == 0) ? 10.0 : 5.0;
            }
        }
    }

    // Gene4: Radial pattern (distance from center, decreasing outwards)
    if (n_genes >= 5) {
        double center_r = (double)(num_grid_rows - 1) / 2.0;
        double center_c = (double)(num_grid_cols - 1) / 2.0;
        double max_dist_val = 0.0;

        for (MKL_INT r_corn = 0; r_corn < num_grid_rows; r_corn += (num_grid_rows-1 > 0 ? num_grid_rows-1 : 1)) {
             for (MKL_INT c_corn = 0; c_corn < num_grid_cols; c_corn += (num_grid_cols-1 > 0 ? num_grid_cols-1 : 1)) {
                double d = sqrt(pow(r_corn - center_r, 2) + pow(c_corn - center_c, 2));
                if (d > max_dist_val) max_dist_val = d;
             }
        }
        if (max_dist_val == 0 && (num_grid_rows > 1 || num_grid_cols > 1)) max_dist_val = 1.0;

        for (MKL_INT r = 0; r < num_grid_rows; ++r) {
            for (MKL_INT c = 0; c < num_grid_cols; ++c) {
                double dist_from_center = sqrt(pow(r - center_r, 2) + pow(c - center_c, 2));
                X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 4] =
                    (max_dist_val > 0) ? (max_dist_val - dist_from_center) : 0.0;
            }
        }
    }

    // --- Z-Normalize each gene column ---
    for (MKL_INT j = 0; j < n_genes; ++j) {
        double sum = 0.0;
        for (MKL_INT i = 0; i < n_spots; ++i) sum += X->values[i*n_genes + j];
        double mean = (n_spots > 0) ? sum / n_spots : 0.0;

        double sum_sq_diff = 0.0;
        for (MKL_INT i = 0; i < n_spots; ++i) {
            sum_sq_diff += pow(X->values[i*n_genes + j] - mean, 2);
        }
        double stddev = (n_spots > 0) ? sqrt(sum_sq_diff / n_spots) : 0.0;

        if (stddev < ZERO_STD_THRESHOLD) {
            for (MKL_INT i = 0; i < n_spots; ++i) X->values[i*n_genes + j] = 0.0;
        } else {
            for (MKL_INT i = 0; i < n_spots; ++i) {
                X->values[i*n_genes + j] = (X->values[i*n_genes + j] - mean) / stddev;
            }
        }
    }

    printf("Toy X_calc matrix (2D Grid, %lld spots x %lld genes) created and Z-normalized.\n",
           (long long)X->nrows, (long long)X->ncols);
    return X;
}

int run_toy_example_2d(const char* output_prefix_toy, MoransIConfig* config) {
    printf("\n--- Running 2D Grid Toy Example (5x5 spots, 5 genes) ---\n");
    MKL_INT grid_rows = 5;
    MKL_INT grid_cols = 5;
    MKL_INT toy_n_genes = 5;
    int status = MORANS_I_SUCCESS;

    DenseMatrix* toy_X_calc = create_toy_X_calc_matrix_2d(grid_rows, grid_cols, toy_n_genes);
    SparseMatrix* toy_W = create_toy_W_matrix_2d(grid_rows, grid_cols);

    DenseMatrix* toy_observed_I = NULL;
    DenseMatrix* toy_theoretical_I = NULL;
    PermutationResults* toy_perm_results = NULL;

    if (!toy_X_calc || !toy_W) {
        fprintf(stderr, "Failed to create 2D toy matrices for example.\n");
        status = MORANS_I_ERROR_COMPUTATION;
        goto toy_cleanup;
    }

    char file_buffer[BUFFER_SIZE];

    snprintf(file_buffer, BUFFER_SIZE, "%s_toy_2D_X_calc_Znorm.tsv", output_prefix_toy);
    if (save_results(toy_X_calc, file_buffer) == MORANS_I_SUCCESS) {
        printf("Saved Z-normalized 2D toy X_calc to %s for inspection.\n", file_buffer);
    }

    toy_theoretical_I = create_theoretical_toy_moran_i_matrix_2d(toy_n_genes, toy_X_calc->colnames);
    if (toy_theoretical_I) {
        snprintf(file_buffer, BUFFER_SIZE, "%s_toy_2D_theoretical_I_full.tsv", output_prefix_toy);
        if(save_results(toy_theoretical_I, file_buffer) == MORANS_I_SUCCESS) {
            printf("2D Toy 'hand-derived' theoretical Moran's I (full matrix) saved to %s\n", file_buffer);
        } else {
            fprintf(stderr, "Warning: Failed to save theoretical toy Moran's I matrix.\n");
        }
    } else {
        fprintf(stderr, "Warning: Failed to create theoretical toy Moran's I matrix.\n");
    }

    printf("Calculating observed Moran's I for 2D toy example...\n");
    toy_observed_I = calculate_morans_i(toy_X_calc, toy_W, config->row_normalize_weights);
    if (!toy_observed_I) {
        fprintf(stderr, "Failed to calculate observed Moran's I for 2D toy example.\n");
        status = MORANS_I_ERROR_COMPUTATION;
    } else {
        snprintf(file_buffer, BUFFER_SIZE, "%s_toy_2D_observed_I_full.tsv", output_prefix_toy);
        if (save_results(toy_observed_I, file_buffer) == MORANS_I_SUCCESS) {
            printf("2D Toy observed Moran's I (full matrix) saved to %s\n", file_buffer);
        } else {
             fprintf(stderr, "Warning: Failed to save observed toy Moran's I matrix.\n");
        }

        if (config->run_permutations) {
            PermutationParams toy_perm_params = config_to_perm_params(config);
            if (toy_perm_params.n_permutations <= 0 || toy_perm_params.n_permutations >= 5000)
                toy_perm_params.n_permutations = 100;

            printf("Running permutation test for 2D toy example (%d permutations)...\n", toy_perm_params.n_permutations);
            toy_perm_results = run_permutation_test(toy_X_calc, toy_W, &toy_perm_params, config->row_normalize_weights);

            if (toy_perm_results) {
                if (save_permutation_results(toy_perm_results, output_prefix_toy) != MORANS_I_SUCCESS) {
                     fprintf(stderr, "Warning: Failed to save some or all toy permutation results.\n");
                } else {
                    printf("2D Toy permutation results saved with prefix: %s\n", output_prefix_toy);
                }
            } else {
                fprintf(stderr, "Permutation test failed for 2D toy example.\n");
                if (status == MORANS_I_SUCCESS) status = MORANS_I_ERROR_COMPUTATION;
            }
        } else {
            printf("Permutation testing not enabled by CLI flags (--run-perm), skipping for 2D toy example.\n");
        }
    }

toy_cleanup:
    free_dense_matrix(toy_X_calc);
    free_sparse_matrix(toy_W);
    free_dense_matrix(toy_observed_I);
    free_dense_matrix(toy_theoretical_I);
    free_permutation_results(toy_perm_results);
    printf("--- 2D Grid Toy Example Finished ---\n\n");
    return status;
}
