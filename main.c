/* main.c - Main program for Moran's I calculation using Intel MKL
 *
 * Version: 1.1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h> 
#include <ctype.h>
#include <omp.h>
#include <errno.h>
#include <unistd.h>  
#include <time.h>    
#include <limits.h> 
#include "morans_i_mkl.h" 

/* For strdup support if not readily available */
#if defined(__STRICT_ANSI__) && !defined(strdup) && !defined(_GNU_SOURCE) && \
    (!defined(_XOPEN_SOURCE) || _XOPEN_SOURCE < 500) && \
    (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L)
static char* my_strdup(const char* s) {
    if (s == NULL) return NULL;
    size_t len = strlen(s) + 1;
    char* new_str = (char*)malloc(len);
    if (new_str == NULL) { perror("my_strdup: malloc failed"); return NULL; }
    return (char*)memcpy(new_str, s, len);
}
#define strdup my_strdup
#endif


void print_elapsed_time(double start_time, double end_time, const char* operation) {
    double elapsed = end_time - start_time;
    printf("Time for %s: %.6f seconds\n", operation, elapsed);
}

void print_main_help(const char* program_name) {
    printf("\nCompute Pairwise or Single-Gene Moran's I for Spatial Transcriptomics Data\n\n");
    printf("Usage: %s -i <input.tsv> -o <output_prefix> [OPTIONS]\n", program_name);
    printf("   or: %s --run-toy-example -o <toy_output_prefix> [PERMUTATION_OPTIONS]\n\n", program_name);
    printf("Input Format:\n");
    printf("  Tab-separated file (TSV).\n");
    printf("  First row: Header with spot coordinates (e.g., '12x34') or cell IDs. First cell can be empty/gene ID header.\n");
    printf("  Subsequent rows: Gene name followed by expression values for each spot/cell.\n");
    printf("\nRequired Arguments for Standard Run:\n");
    printf("  -i <file>\tInput data matrix file (Genes x Spots/Cells).\n");
    printf("  -o <prefix>\tOutput file prefix for results (e.g., 'my_analysis_results').\n");
    printf("\nGeneral Options:\n");
    printf("  -r <int>\tMaximum grid radius for neighbor search. Default: %d.\n", DEFAULT_MAX_RADIUS);
    printf("  -p <int>\tPlatform type (%d: Visium, %d: Older ST, %d: Single Cell). Default: %d.\n",
           VISIUM, OLD, SINGLE_CELL, DEFAULT_PLATFORM_MODE);
    printf("  -b <0|1>\tCalculation mode: 0 = Single-gene, 1 = Pairwise. Default: %d.\n",
           DEFAULT_CALC_PAIRWISE);
    printf("  -g <0|1>\tGene selection (if -b 1): 0 = First gene vs all, 1 = All gene pairs. Default: %d.\n",
           DEFAULT_CALC_ALL_VS_ALL);
    printf("  -s <0|1>\tInclude self-comparison (w_ii)? 0 = No, 1 = Yes. Default: %d.\n", DEFAULT_INCLUDE_SAME_SPOT);
    printf("  -t <int>\tSet number of OpenMP threads. Default: %d (or OMP_NUM_THREADS).\n", DEFAULT_NUM_THREADS);
    printf("  -m <int>\tSet number of MKL threads. Default: Value of -t or OpenMP default.\n");
    printf("\nSingle-cell Specific Options:\n");
    printf("  -c <file>\tCoordinates/metadata file (TSV). Required for single-cell data.\n");
    printf("  --id-col <name>\tColumn name for cell IDs in metadata. Default: 'cell_ID'.\n");
    printf("  --x-col <name>\tColumn name for X coordinates in metadata. Default: 'sdimx'.\n");
    printf("  --y-col <name>\tColumn name for Y coordinates in metadata. Default: 'sdimy'.\n");
    printf("  --scale <float>\tScaling factor for SC coordinates to integer grid. Default: %.1f.\n", DEFAULT_COORD_SCALE_FACTOR);
    printf("  --sigma <float>\tCustom sigma for RBF kernel (physical units). If <=0, inferred for SC or platform default.\n");
    printf("\nPermutation Test Options (apply if -b 1 -g 1 or for --run-toy-example):\n");
    printf("  --run-perms\tEnable permutation testing.\n");
    printf("  --n-perms <int>\tNumber of permutations. Default: %d. Implies --run-perms.\n", DEFAULT_NUM_PERMUTATIONS);
    printf("  --perm-seed <int>\tSeed for RNG. Default: Based on system time. Implies --run-perms.\n");
    printf("  --perm-output-zscores\tOutput Z-scores. Implies --run-perms.\n");
    printf("  --perm-output-pvalues\tOutput p-values. Implies --run-perms.\n");
    printf("\nToy Example Mode:\n");
    printf("  --run-toy-example\tRuns a small, built-in 2D grid (5x5) example to test functionality.\n"
           "                    \tRequires -o <prefix>. Permutation options can be used.\n");
    printf("\nOutput Format (files named based on <output_prefix>):\n");
    printf("  Single-gene (-b 0): <prefix>_single_gene_moran_i.tsv (Gene, MoranI).\n");
    printf("  Pairwise All (-b 1 -g 1): <prefix>_all_pairs_moran_i_raw.tsv (Observed Moran's I, Raw lower triangular).\n");
    printf("  Pairwise First (-b 1 -g 0): <prefix>_first_vs_all_moran_i.tsv (Gene, MoranI_vs_Gene0).\n");
    printf("  Permutation outputs (if enabled for pairwise all / toy example - saved as raw lower triangular):\n");
    printf("    <prefix>_zscores_lower_tri.tsv (if --perm-output-zscores)\n");
    printf("    <prefix>_pvalues_lower_tri.tsv (if --perm-output-pvalues)\n");
    printf("\nExample:\n");
    printf("  %s -i expr.tsv -o run1 -r 3 -p 0 -b 1 -g 1 -t 8 --run-perms --n-perms 1000\n", program_name);
    printf("  %s --run-toy-example -o toy_2d_run --n-perms 100 --perm-seed 42\n\n", program_name);
    printf("Version: %s\n", morans_i_mkl_version());
}

// --- Toy Example Implementation for a 2D Grid ---
// --- Helper: grid_to_1d_idx remains the same ---
static inline MKL_INT grid_to_1d_idx(MKL_INT r, MKL_INT c, MKL_INT num_grid_cols) {
    return r * num_grid_cols + c;
}

/*
 * Creates a DenseMatrix containing the theoretically expected Moran's I values
 * for the specific 5-gene, 5x5 grid toy example.
 * Note: Some values are exact based on derivation, others might be qualitative expectations.
 * The "true" theoretical values for the generated X and W are what calculate_morans_i produces.
 * This function is more for "expected pattern" verification.
 */
DenseMatrix* create_theoretical_toy_moran_i_matrix_2d(MKL_INT n_genes, char** gene_names) {
    if (n_genes != 5) {
        fprintf(stderr, "Error (create_theoretical_toy_moran_i): This function is hardcoded for 5 genes.\n");
        return NULL;
    }

    DenseMatrix* theoretical_I = (DenseMatrix*)calloc(1, sizeof(DenseMatrix));
    if (!theoretical_I) { perror("calloc theoretical_I"); return NULL; }

    theoretical_I->nrows = n_genes;
    theoretical_I->ncols = n_genes;
    theoretical_I->values = (double*)mkl_calloc((size_t)n_genes * n_genes, sizeof(double), 64);
    theoretical_I->rownames = (char**)calloc(n_genes, sizeof(char*));
    theoretical_I->colnames = (char**)calloc(n_genes, sizeof(char*));

    if (!theoretical_I->values || !theoretical_I->rownames || !theoretical_I->colnames) {
        perror("mkl_calloc theoretical_I components");
        free_dense_matrix(theoretical_I); return NULL;
    }

    for (MKL_INT i = 0; i < n_genes; ++i) {
        theoretical_I->rownames[i] = strdup(gene_names[i]); // Use names from X_calc
        theoretical_I->colnames[i] = strdup(gene_names[i]);
        if (!theoretical_I->rownames[i] || !theoretical_I->colnames[i]) {
            perror("strdup theoretical_I gene names");
            free_dense_matrix(theoretical_I); return NULL;
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
    // This is harder to pin down to an exact theoretical value without re-implementing the sum.
    // We can either:
    // 1. Put an estimated qualitative value (e.g., 0.5 as a placeholder for "positive")
    // 2. Leave it as 0.0 and note that the actual calculated value should be compared.
    // 3. Or, even better, this "theoretical" matrix could actually *use* the output
    //    of calculate_morans_i for this specific entry if the goal is to show what the
    //    code *should* produce if the hand-derivations are also correct.
    // Let's use a placeholder, as this matrix is for "hand-derived expectations".
    theoretical_I->values[4 * n_genes + 4] = 0.4; // Placeholder: Expected positive, actual value will depend on exact Z-scores and W sums.

    printf("Theoretical Moran's I expectation matrix created.\n");
    return theoretical_I;
}


// --- Helper: create_toy_W_matrix_2d remains the same ---
SparseMatrix* create_toy_W_matrix_2d(MKL_INT num_grid_rows, MKL_INT num_grid_cols) {
    MKL_INT n_spots = num_grid_rows * num_grid_cols;
    if (n_spots == 0) {
        fprintf(stderr, "Error (create_toy_W_matrix_2d): Cannot create W for 0 spots.\n");
        return NULL;
    }

    SparseMatrix* W = (SparseMatrix*)calloc(1, sizeof(SparseMatrix));
    if (!W) { perror("calloc toy W 2D"); return NULL; }

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
            goto cleanup_toy_w_coo_2d; // Updated label
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
                        goto cleanup_toy_w_coo_2d; // Updated label
                    }
                }
            }
        }
    }
    W->nnz = current_nnz;

    W->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
    if (!W->row_ptr) { perror("mkl_calloc W->row_ptr for toy W 2D"); goto cleanup_toy_w_coo_2d;} // Updated label

    if (W->nnz > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc(W->nnz * sizeof(MKL_INT), 64);
        W->values  = (double*)mkl_malloc(W->nnz * sizeof(double), 64);
        if (!W->col_ind || !W->values) {
            perror("mkl_malloc W->col_ind/values for toy W 2D");
            goto cleanup_toy_w_coo_2d; // Updated label
        }

        for (MKL_INT k = 0; k < W->nnz; ++k) W->row_ptr[temp_I[k] + 1]++;
        for (MKL_INT i = 0; i < n_spots; ++i) W->row_ptr[i + 1] += W->row_ptr[i];
        
        current_pos = (MKL_INT*)mkl_malloc((size_t)(n_spots + 1) * sizeof(MKL_INT), 64); // Corrected size
        if (!current_pos) { perror("mkl_malloc current_pos for CSR conversion"); goto cleanup_toy_w_coo_2d;} // Updated label
        memcpy(current_pos, W->row_ptr, (size_t)(n_spots + 1) * sizeof(MKL_INT)); // Corrected size

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

cleanup_toy_w_coo_2d: // Centralized cleanup label for this function
    free(temp_I); 
    free(temp_J); 
    free(temp_V);
    mkl_free(current_pos);
    // If any CSR allocation failed after COO temp arrays were used, W might be partially formed
    if ((W->row_ptr && W->nnz > 0 && (!W->col_ind || !W->values)) || (!W->row_ptr && n_spots > 0) ) { 
        free_sparse_matrix(W); // free_sparse_matrix handles partially allocated W
        return NULL;
    }
    if (W->nnz > 0 && (!W->col_ind || !W->values)) { // Final check if CSR data alloc failed
         free_sparse_matrix(W);
         return NULL;
    }


    printf("Toy W matrix (2D Grid, Rook, %lldx%lld spots) created with %lld NNZ.\n", (long long)num_grid_rows, (long long)num_grid_cols, (long long)W->nnz);
    return W;
}


/*
 * Creates a toy X_calc matrix ( (num_grid_rows * num_grid_cols) x N_genes )
 * with 2D spatial patterns, then Z-normalizes it. Now uses 5 genes.
 */
DenseMatrix* create_toy_X_calc_matrix_2d(MKL_INT num_grid_rows, MKL_INT num_grid_cols, MKL_INT n_genes) {
    MKL_INT n_spots = num_grid_rows * num_grid_cols;
    if (n_spots == 0 || n_genes == 0) {
        fprintf(stderr, "Error (create_toy_X_calc_2d): n_spots or n_genes is zero.\n");
        return NULL;
    }
    if (n_genes < 5 && n_genes > 0) { // Check if enough genes for all defined patterns
        fprintf(stderr, "Warning (create_toy_X_calc_2d): Requested %lld genes, but patterns defined for up to 5. Some patterns might be omitted.\n", (long long)n_genes);
    }


    DenseMatrix* X = (DenseMatrix*)calloc(1, sizeof(DenseMatrix));
    if (!X) { perror("calloc toy X 2D"); return NULL; }

    X->nrows = n_spots; X->ncols = n_genes;
    X->values = (double*)mkl_calloc((size_t)n_spots * n_genes, sizeof(double), 64);
    X->rownames = (char**)calloc(n_spots, sizeof(char*));
    X->colnames = (char**)calloc(n_genes, sizeof(char*));

    if (!X->values || !X->rownames || !X->colnames) {
        perror("mkl_calloc toy X 2D components");
        free_dense_matrix(X); return NULL;
    }

    for (MKL_INT r = 0; r < num_grid_rows; ++r) {
        for (MKL_INT c = 0; c < num_grid_cols; ++c) {
            MKL_INT spot_idx = grid_to_1d_idx(r, c, num_grid_cols);
            char name_buf[32]; snprintf(name_buf, 32, "S_r%d_c%d", (int)r, (int)c);
            X->rownames[spot_idx] = strdup(name_buf);
            if (!X->rownames[spot_idx]) { perror("strdup toy spot name 2D"); free_dense_matrix(X); return NULL; }
        }
    }
    for (MKL_INT j = 0; j < n_genes; ++j) {
        char name_buf[32]; snprintf(name_buf, 32, "Gene%lld", (long long)j);
        X->colnames[j] = strdup(name_buf);
        if (!X->colnames[j]) { perror("strdup toy gene name 2D"); free_dense_matrix(X); return NULL; }
    }

    // Gene0: Gradient along rows (increases with row index r)
    if (n_genes >= 1) for (MKL_INT r = 0; r < num_grid_rows; ++r) for (MKL_INT c=0; c<num_grid_cols; ++c) X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 0] = (double)r;
    
    // Gene1: Identical to Gene0 (also row gradient)
    if (n_genes >= 2) for (MKL_INT r = 0; r < num_grid_rows; ++r) for (MKL_INT c=0; c<num_grid_cols; ++c) X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 1] = (double)r;
    
    // Gene2: Gradient along columns (increases with col index c)
    if (n_genes >= 3) for (MKL_INT r = 0; r < num_grid_rows; ++r) for (MKL_INT c=0; c<num_grid_cols; ++c) X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 2] = (double)c;
    
    // Gene3: Checkerboard pattern ((r+c) % 2)
    if (n_genes >= 4) for (MKL_INT r = 0; r < num_grid_rows; ++r) for (MKL_INT c=0; c<num_grid_cols; ++c) X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 3] = ((r + c) % 2 == 0) ? 10.0 : 5.0;
    
    // Gene4: Radial pattern (distance from center, decreasing outwards)
    if (n_genes >= 5) {
        double center_r = (double)(num_grid_rows -1) / 2.0;
        double center_c = (double)(num_grid_cols -1) / 2.0;
        double max_dist_val = 0.0; // Find max distance to a corner to scale values later
        for (MKL_INT r_corn = 0; r_corn < num_grid_rows; r_corn += (num_grid_rows-1 > 0 ? num_grid_rows-1 : 1) ) {
             for (MKL_INT c_corn = 0; c_corn < num_grid_cols; c_corn += (num_grid_cols-1 > 0 ? num_grid_cols-1 : 1) ) {
                double d = sqrt(pow(r_corn - center_r, 2) + pow(c_corn - center_c, 2));
                if (d > max_dist_val) max_dist_val = d;
             }
        }
        if (max_dist_val == 0 && (num_grid_rows > 1 || num_grid_cols > 1) ) max_dist_val = 1.0; // Avoid div by zero for non-1x1 grid

        for (MKL_INT r = 0; r < num_grid_rows; ++r) for (MKL_INT c=0; c<num_grid_cols; ++c) {
            double dist_from_center = sqrt(pow(r - center_r, 2) + pow(c - center_c, 2));
            X->values[grid_to_1d_idx(r,c,num_grid_cols)*n_genes + 4] = (max_dist_val > 0) ? (max_dist_val - dist_from_center) : 0.0;
        }
    }

    // --- Z-Normalize each gene column ---
    for (MKL_INT j = 0; j < n_genes; ++j) { 
        double sum = 0.0;
        for (MKL_INT i = 0; i < n_spots; ++i) sum += X->values[i*n_genes + j];
        double mean = (n_spots > 0) ? sum / n_spots : 0.0;
        
        double sum_sq_diff = 0.0;
        for (MKL_INT i = 0; i < n_spots; ++i) sum_sq_diff += pow(X->values[i*n_genes + j] - mean, 2);
        double stddev = (n_spots > 0) ? sqrt(sum_sq_diff / n_spots) : 0.0;

        if (stddev < ZERO_STD_THRESHOLD) { 
            for (MKL_INT i = 0; i < n_spots; ++i) X->values[i*n_genes + j] = 0.0;
        } else {
            for (MKL_INT i = 0; i < n_spots; ++i) X->values[i*n_genes + j] = (X->values[i*n_genes + j] - mean) / stddev;
        }
    }
    printf("Toy X_calc matrix (2D Grid, %lld spots x %lld genes) created and Z-normalized.\n", (long long)X->nrows, (long long)X->ncols);
    return X;
}

/* Orchestrator for the 2D Grid Toy Example. */
int run_toy_example_2d(const char* output_prefix_toy, const MoransIConfig* cli_config) {
    printf("\n--- Running 2D Grid Toy Example (5x5 spots, 5 genes) ---\n");
    MKL_INT grid_rows = 5; 
    MKL_INT grid_cols = 5; 
    MKL_INT toy_n_genes = 5; 
    int status = MORANS_I_SUCCESS;

    DenseMatrix* toy_X_calc = create_toy_X_calc_matrix_2d(grid_rows, grid_cols, toy_n_genes);
    SparseMatrix* toy_W = create_toy_W_matrix_2d(grid_rows, grid_cols);

    DenseMatrix* toy_observed_I = NULL; // Initialize to NULL
    DenseMatrix* toy_theoretical_I = NULL; // Initialize to NULL
    PermutationResults* toy_perm_results = NULL; // Initialize to NULL


    if (!toy_X_calc || !toy_W) {
        fprintf(stderr, "Failed to create 2D toy matrices for example.\n");
        status = MORANS_I_ERROR_COMPUTATION;
        goto toy_cleanup; // Use goto for cleanup
    }

    char file_buffer[BUFFER_SIZE]; // Reusable buffer for filenames

    snprintf(file_buffer, BUFFER_SIZE, "%s_toy_2D_X_calc_Znorm.tsv", output_prefix_toy);
    if (save_results(toy_X_calc, file_buffer) == MORANS_I_SUCCESS) {
        printf("Saved Z-normalized 2D toy X_calc to %s for inspection.\n", file_buffer);
    }

    // Create and save the "hand-derived" theoretical matrix
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
    toy_observed_I = calculate_morans_i(toy_X_calc, toy_W);
    if (!toy_observed_I) {
        fprintf(stderr, "Failed to calculate observed Moran's I for 2D toy example.\n");
        status = MORANS_I_ERROR_COMPUTATION;
        // No goto here, will be caught by next if block or cleanup will handle toy_observed_I = NULL
    } else {
        snprintf(file_buffer, BUFFER_SIZE, "%s_toy_2D_observed_I_full.tsv", output_prefix_toy);
        if (save_results(toy_observed_I, file_buffer) == MORANS_I_SUCCESS) {
            printf("2D Toy observed Moran's I (full matrix) saved to %s\n", file_buffer);
        } else {
             fprintf(stderr, "Warning: Failed to save observed toy Moran's I matrix.\n");
        }


        if (cli_config->run_permutations) {
            PermutationParams toy_perm_params;
            toy_perm_params.n_permutations = (cli_config->num_permutations > 0 && cli_config->num_permutations < 5000) ? cli_config->num_permutations : 10000;
            toy_perm_params.seed = cli_config->perm_seed; 
            toy_perm_params.z_score_output = cli_config->perm_output_zscores;
            toy_perm_params.p_value_output = cli_config->perm_output_pvalues;

            printf("Running permutation test for 2D toy example (%d permutations)...\n", toy_perm_params.n_permutations);
            toy_perm_results = run_permutation_test(toy_X_calc, toy_W, &toy_perm_params);

            if (toy_perm_results) {
                // save_permutation_results now saves Z and P lower triangular files
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
            printf("Permutation testing not enabled by CLI flags (--run-perms), skipping for 2D toy example.\n");
        }
    }

toy_cleanup: // Label for cleaning up toy-specific allocations
    free_dense_matrix(toy_X_calc);
    free_sparse_matrix(toy_W);
    free_dense_matrix(toy_observed_I); // Safe if NULL
    free_dense_matrix(toy_theoretical_I); // Safe if NULL
    free_permutation_results(toy_perm_results); // Safe if NULL
    printf("--- 2D Grid Toy Example Finished ---\n\n");
    return status;
}


/* Main function */
int main(int argc, char* argv[]) {
    double main_op_start_time, main_op_end_time;
    double total_start_time = get_time();

    MoransIConfig config = initialize_default_config();
    char input_file[BUFFER_SIZE] = "";
    char output_file_prefix[BUFFER_SIZE] = "";
    char meta_file[BUFFER_SIZE] = "";
    char id_column[BUFFER_SIZE] = "cell_ID";
    char x_column[BUFFER_SIZE] = "sdimx";
    char y_column[BUFFER_SIZE] = "sdimy";
    double custom_sigma = 0.0;
    int use_metadata_file = 0;
    int run_the_toy_example_flag = 0; // 0 = no toy, 2 = 2D grid toy

    DenseMatrix* vst_matrix_ptr = NULL;
    DenseMatrix* znorm_genes_x_spots_ptr = NULL;
    SpotCoordinates* spot_coords_data_ptr = NULL;
    MKL_INT* valid_spot_original_indices_ptr = NULL;
    MKL_INT* valid_spot_rows_for_W_ptr = NULL;
    MKL_INT* valid_spot_cols_for_W_ptr = NULL;
    char**   valid_spot_names_list_ptr = NULL;
    DenseMatrix* decay_lookup_matrix_ptr = NULL;
    SparseMatrix* W_matrix_ptr = NULL;
    DenseMatrix* X_calc_ptr = NULL;
    DenseMatrix* observed_morans_i_matrix_ptr = NULL;
    PermutationResults* perm_results_ptr = NULL;
    int final_status = MORANS_I_SUCCESS;
    MKL_INT num_valid_spots_for_helpers = 0;

    if (argc == 1 || (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))) {
        print_main_help(argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -i\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            strncpy(input_file, argv[i], BUFFER_SIZE - 1); input_file[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "-o") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -o\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            strncpy(output_file_prefix, argv[i], BUFFER_SIZE - 1); output_file_prefix[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "--run-toy-example") == 0) {
            run_the_toy_example_flag = 2; 
        } else if (strcmp(argv[i], "-c") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -c\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            strncpy(meta_file, argv[i], BUFFER_SIZE - 1); meta_file[BUFFER_SIZE - 1] = '\0';
            use_metadata_file = 1;
        } else if (strcmp(argv[i], "--id-col") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for --id-col\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            strncpy(id_column, argv[i], BUFFER_SIZE - 1); id_column[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "--x-col") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for --x-col\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            strncpy(x_column, argv[i], BUFFER_SIZE - 1); x_column[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "--y-col") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for --y-col\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            strncpy(y_column, argv[i], BUFFER_SIZE - 1); y_column[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "--scale") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for --scale\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.coord_scale = load_double_value(argv[i], "--scale");
            if (isnan(config.coord_scale) || config.coord_scale <= 0) { fprintf(stderr, "Error: --scale must be a positive number.\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup;}
        } else if (strcmp(argv[i], "--sigma") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for --sigma\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            custom_sigma = load_double_value(argv[i], "--sigma");
            if (isnan(custom_sigma)) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
        } else if (strcmp(argv[i], "-r") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -r\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.max_radius = load_positive_value(argv[i], "-r", 1, 1000);
            if (config.max_radius < 0) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
        } else if (strcmp(argv[i], "-p") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -p\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.platform_mode = load_positive_value(argv[i], "-p", 0, 2);
            if (config.platform_mode < 0) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -b\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.calc_pairwise = load_positive_value(argv[i], "-b", 0, 1);
            if (config.calc_pairwise < 0) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
        } else if (strcmp(argv[i], "-g") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -g\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.calc_all_vs_all = load_positive_value(argv[i], "-g", 0, 1);
            if (config.calc_all_vs_all < 0) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
        } else if (strcmp(argv[i], "-s") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -s\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.include_same_spot = load_positive_value(argv[i], "-s", 0, 1);
            if (config.include_same_spot < 0) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
        } else if (strcmp(argv[i], "-t") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -t\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.n_threads = load_positive_value(argv[i], "-t", 1, 1024);
            if (config.n_threads < 0) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
        } else if (strcmp(argv[i], "-m") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for -m\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.mkl_n_threads = load_positive_value(argv[i], "-m", 1, 1024);
            if (config.mkl_n_threads < 0) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
        } else if (strcmp(argv[i], "--run-perms") == 0) { 
            config.run_permutations = 1;
        } else if (strcmp(argv[i], "--n-perms") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for --n-perms\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.num_permutations = load_positive_value(argv[i], "--n-perms", 1, 10000000);
            if (config.num_permutations < 0) { final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            config.run_permutations = 1;
        } else if (strcmp(argv[i], "--perm-seed") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: Missing value for --perm-seed\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup; }
            long seed_val = strtol(argv[i], NULL, 10); 
            if (errno == ERANGE || seed_val < 0 || (unsigned long)seed_val > UINT_MAX) { 
                fprintf(stderr, "Error: Invalid seed value for --perm-seed '%s'.\n", argv[i]); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup;
            }
            config.perm_seed = (unsigned int)seed_val;
            config.run_permutations = 1;
        } else if (strcmp(argv[i], "--perm-output-zscores") == 0) {
            config.perm_output_zscores = 1;
            config.run_permutations = 1;
        } else if (strcmp(argv[i], "--perm-output-pvalues") == 0) {
            config.perm_output_pvalues = 1;
            config.run_permutations = 1;
        } else {
            fprintf(stderr, "Error: Unknown parameter '%s'. Use -h for help.\n", argv[i]);
            final_status = MORANS_I_ERROR_PARAMETER; goto cleanup;
        }
    }

    if (initialize_morans_i(&config) != MORANS_I_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize Moran's I MKL/OpenMP environment.\n");
        final_status = MORANS_I_ERROR_COMPUTATION; goto cleanup;
    }
    if (strlen(output_file_prefix) == 0) { fprintf(stderr, "Error: Output file prefix (-o) must be specified.\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup;}

    if (run_the_toy_example_flag == 2) {
        final_status = run_toy_example_2d(output_file_prefix, &config);
        goto cleanup; 
    }

    // --- REGULAR EXECUTION PATH ---
    if (strlen(input_file) == 0) { fprintf(stderr, "Error: Input file (-i) must be specified for standard run.\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup;}
    if (access(input_file, R_OK) != 0) { fprintf(stderr, "Error: Cannot access input file '%s': %s\n", input_file, strerror(errno)); final_status = MORANS_I_ERROR_FILE; goto cleanup;}
    if (use_metadata_file) {
        if (strlen(meta_file) == 0) { fprintf(stderr, "Error: Metadata file path is empty despite -c being used.\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup;}
        if (access(meta_file, R_OK) != 0) { fprintf(stderr, "Error: Cannot access metadata file '%s': %s\n", meta_file, strerror(errno)); final_status = MORANS_I_ERROR_FILE; goto cleanup;}
        if (config.platform_mode != SINGLE_CELL) {
            printf("Warning: Metadata file (-c) provided. Forcing platform mode to SINGLE_CELL (%d).\n", SINGLE_CELL);
            config.platform_mode = SINGLE_CELL;
        }
    } else {
        if (config.platform_mode == SINGLE_CELL) { fprintf(stderr, "Error: Platform mode SINGLE_CELL selected, but no metadata file (-c) provided.\n"); final_status = MORANS_I_ERROR_PARAMETER; goto cleanup;}
    }

    printf("\nMoran's I calculation utility version %s\n\n", morans_i_mkl_version());
    printf("--- Parameters ---\n");
    printf("Input file: %s\n", input_file);
    printf("Output file prefix: %s\n", output_file_prefix);
    if (use_metadata_file) {
        printf("Metadata file: %s\n", meta_file);
        printf("  ID column: %s\n", id_column); printf("  X coordinate column: %s\n", x_column); printf("  Y coordinate column: %s\n", y_column);
        printf("  Coordinate scale factor: %.2f\n", config.coord_scale);
    }
    if (custom_sigma > 0) printf("Custom Sigma for RBF: %.4f\n", custom_sigma);
    else printf("Custom Sigma for RBF: Not set (will use platform default or infer for SC).\n");
    printf("Max radius (grid units): %d\n", config.max_radius);
    printf("Platform: %d (%s)\n", config.platform_mode, config.platform_mode == VISIUM ? "Visium" : (config.platform_mode == OLD ? "Old ST" : "Single Cell"));
    printf("Mode: %s\n", config.calc_pairwise ? "Pairwise Moran's I" : "Single-Gene Moran's I");
    if (config.calc_pairwise) printf("  Gene Pairs: %s\n", config.calc_all_vs_all ? "All vs All" : "First Gene vs All Others");
    printf("Include Self-Comparisons (w_ii): %s\n", config.include_same_spot ? "Yes" : "No");
    if(config.run_permutations) {
        printf("Permutation Testing: Enabled\n");
        printf("  Number of Permutations: %d\n", config.num_permutations);
        printf("  Permutation Seed: %u\n", config.perm_seed);
        printf("  Output Z-scores: %s\n", config.perm_output_zscores ? "Yes" : "No");
        printf("  Output P-values: %s\n", config.perm_output_pvalues ? "Yes" : "No");
    }
    printf("------------------\n");

    printf("Loading gene expression data from %s...\n", input_file);
    main_op_start_time = get_time(); vst_matrix_ptr = read_vst_file(input_file); main_op_end_time = get_time();
    print_elapsed_time(main_op_start_time, main_op_end_time, "gene expression data loading");
    if (!vst_matrix_ptr) { fprintf(stderr, "Error: Failed to load gene expression data.\n"); final_status = MORANS_I_ERROR_FILE; goto cleanup;}
    printf("Loaded data matrix: %lld genes x %lld spots/cells.\n", (long long)vst_matrix_ptr->nrows, (long long)vst_matrix_ptr->ncols);
    if (vst_matrix_ptr->nrows == 0 || vst_matrix_ptr->ncols == 0) { fprintf(stderr, "Error: Loaded expression matrix is empty.\n"); final_status = MORANS_I_ERROR_FILE; goto cleanup;}

    MKL_INT total_expr_elements_main = vst_matrix_ptr->nrows * vst_matrix_ptr->ncols; MKL_INT non_finite_count_main = 0; 
    #pragma omp parallel for reduction(+:non_finite_count_main)
    for (MKL_INT i_main = 0; i_main < total_expr_elements_main; i_main++) if (!isfinite(vst_matrix_ptr->values[i_main])) { vst_matrix_ptr->values[i_main] = 0.0; non_finite_count_main++;}
    if (non_finite_count_main > 0) printf("Warning: Found and replaced %lld non-finite values with 0.0 in expression data.\n", (long long)non_finite_count_main);

    printf("Performing Z-normalization (gene-wise)...\n");
    main_op_start_time = get_time(); znorm_genes_x_spots_ptr = z_normalize(vst_matrix_ptr); main_op_end_time = get_time();
    print_elapsed_time(main_op_start_time, main_op_end_time, "Z-normalization");
    free_dense_matrix(vst_matrix_ptr); vst_matrix_ptr = NULL; 
    if (!znorm_genes_x_spots_ptr) { fprintf(stderr, "Error: Z-normalization failed.\n"); final_status = MORANS_I_ERROR_COMPUTATION; goto cleanup;}

    printf("Preparing spatial coordinate data...\n");
    main_op_start_time = get_time();
    if (use_metadata_file) spot_coords_data_ptr = read_coordinates_file(meta_file, id_column, x_column, y_column, config.coord_scale);
    else spot_coords_data_ptr = extract_coordinates(znorm_genes_x_spots_ptr->colnames, znorm_genes_x_spots_ptr->ncols);
    main_op_end_time = get_time(); print_elapsed_time(main_op_start_time, main_op_end_time, "spatial coordinate processing");
    if (!spot_coords_data_ptr) { fprintf(stderr, "Error: Failed to obtain or process spot coordinates.\n"); final_status = MORANS_I_ERROR_COMPUTATION; goto cleanup;}
    printf("Processed %lld total spot coordinate entries, %lld are valid.\n", (long long)spot_coords_data_ptr->total_spots, (long long)spot_coords_data_ptr->valid_spots);
    if (spot_coords_data_ptr->valid_spots == 0) { fprintf(stderr, "Error: No valid spot coordinates found. Cannot proceed.\n"); final_status = MORANS_I_ERROR_COMPUTATION; goto cleanup;}

    double sigma_for_decay_main = custom_sigma; 
    if (config.platform_mode == SINGLE_CELL && sigma_for_decay_main <= 0.0) {
        printf("Inferring sigma for RBF kernel from single-cell coordinate data...\n");
        sigma_for_decay_main = infer_sigma_from_data(spot_coords_data_ptr, config.coord_scale);
        if (sigma_for_decay_main <= 0) { fprintf(stderr, "Warning: Failed to infer positive sigma (got %.2f), using default of 50.0 for SC.\n", sigma_for_decay_main); sigma_for_decay_main = 50.0;}
    }
    
    num_valid_spots_for_helpers = spot_coords_data_ptr->valid_spots;
    valid_spot_original_indices_ptr = (MKL_INT*)malloc((size_t)num_valid_spots_for_helpers * sizeof(MKL_INT));
    valid_spot_rows_for_W_ptr = (MKL_INT*)malloc((size_t)num_valid_spots_for_helpers * sizeof(MKL_INT));
    valid_spot_cols_for_W_ptr = (MKL_INT*)malloc((size_t)num_valid_spots_for_helpers * sizeof(MKL_INT));
    valid_spot_names_list_ptr = (char**)calloc(num_valid_spots_for_helpers, sizeof(char*));
    if (!valid_spot_original_indices_ptr || !valid_spot_rows_for_W_ptr || !valid_spot_cols_for_W_ptr || !valid_spot_names_list_ptr) {
        perror("Memory allocation failed for valid spot helper arrays"); final_status = MORANS_I_ERROR_MEMORY; goto cleanup;
    }
    MKL_INT v_idx_main = 0; 
    for (MKL_INT i_main_loop = 0; i_main_loop < spot_coords_data_ptr->total_spots; i_main_loop++) { 
        if (spot_coords_data_ptr->valid_mask[i_main_loop]) {
            if (v_idx_main >= num_valid_spots_for_helpers) { fprintf(stderr, "Error: Index v_idx_main out of bounds for helper arrays.\n"); final_status = MORANS_I_ERROR_COMPUTATION; goto cleanup;}
            valid_spot_rows_for_W_ptr[v_idx_main] = spot_coords_data_ptr->spot_row[i_main_loop];
            valid_spot_cols_for_W_ptr[v_idx_main] = spot_coords_data_ptr->spot_col[i_main_loop];
            valid_spot_names_list_ptr[v_idx_main] = strdup(spot_coords_data_ptr->spot_names[i_main_loop]);
            if(!valid_spot_names_list_ptr[v_idx_main]) { perror("strdup for valid_spot_names_list"); final_status = MORANS_I_ERROR_MEMORY; goto cleanup;}
            int expr_col_idx_main = -1; 
            for (MKL_INT j_main_loop = 0; j_main_loop < znorm_genes_x_spots_ptr->ncols; ++j_main_loop) if (strcmp(spot_coords_data_ptr->spot_names[i_main_loop], znorm_genes_x_spots_ptr->colnames[j_main_loop]) == 0) { expr_col_idx_main = j_main_loop; break;}
            valid_spot_original_indices_ptr[v_idx_main] = expr_col_idx_main;
            if(expr_col_idx_main == -1 && use_metadata_file) fprintf(stderr, "Warning: Valid coordinate spot '%s' from metadata not found in expression matrix colnames.\n", spot_coords_data_ptr->spot_names[i_main_loop]);
            v_idx_main++;
        }
    }
    if (v_idx_main != num_valid_spots_for_helpers) {
        fprintf(stderr, "Warning: Actual populated valid spots (%lld) differs from initial count (%lld). Adjusting.\n", (long long)v_idx_main, (long long)num_valid_spots_for_helpers);
        num_valid_spots_for_helpers = v_idx_main; 
        if (num_valid_spots_for_helpers == 0) { fprintf(stderr, "Error: No valid spots remain. Cannot proceed.\n"); final_status = MORANS_I_ERROR_COMPUTATION; goto cleanup; }
    }

    printf("Creating distance decay matrix (max_radius_grid_units=%d)...\n", config.max_radius);
    main_op_start_time = get_time();
    decay_lookup_matrix_ptr = create_distance_matrix(config.max_radius, config.platform_mode, sigma_for_decay_main, config.coord_scale);
    main_op_end_time = get_time(); print_elapsed_time(main_op_start_time, main_op_end_time, "distance decay matrix creation");
    if (!decay_lookup_matrix_ptr) { fprintf(stderr, "Error: Failed to create distance decay lookup matrix.\n"); final_status = MORANS_I_ERROR_COMPUTATION; goto cleanup;}
    if (!config.include_same_spot && decay_lookup_matrix_ptr->nrows > 0 && decay_lookup_matrix_ptr->ncols > 0) {
        printf("Excluding self-comparisons: setting decay_matrix[0,0] to 0.0.\n");
        decay_lookup_matrix_ptr->values[0] = 0.0;
    }

    printf("Building spatial weight matrix W for %lld valid spots...\n", (long long)num_valid_spots_for_helpers);
    main_op_start_time = get_time();
    W_matrix_ptr = build_spatial_weight_matrix(valid_spot_rows_for_W_ptr, valid_spot_cols_for_W_ptr, num_valid_spots_for_helpers, decay_lookup_matrix_ptr, config.max_radius);
    main_op_end_time = get_time(); print_elapsed_time(main_op_start_time, main_op_end_time, "spatial weight matrix W construction");
    free_dense_matrix(decay_lookup_matrix_ptr); decay_lookup_matrix_ptr = NULL;
    if (!W_matrix_ptr) { fprintf(stderr, "Error: Failed to build spatial weight matrix W.\n"); final_status = MORANS_I_ERROR_COMPUTATION; goto cleanup;}

    double S0_val_main = calculate_weight_sum(W_matrix_ptr); 
    printf("Sum of all weights S0 = %.6f (from %lld NNZ in W)\n", S0_val_main, (long long)W_matrix_ptr->nnz);
    if (fabs(S0_val_main) < DBL_EPSILON && W_matrix_ptr->nnz > 0) fprintf(stderr, "Warning: S0 is near-zero. Moran's I results will likely be 0, NaN, or Inf.\n");
    if (W_matrix_ptr->nnz == 0 && num_valid_spots_for_helpers > 0) fprintf(stderr, "Warning: Spatial Weight Matrix W has no non-zero elements.\n");

    printf("Preparing final calculation matrix X_calc (Valid_Spots x Genes)...\n");
    main_op_start_time = get_time(); X_calc_ptr = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!X_calc_ptr) { perror("malloc X_calc_ptr struct"); final_status = MORANS_I_ERROR_MEMORY; goto cleanup; }
    X_calc_ptr->nrows = num_valid_spots_for_helpers; X_calc_ptr->ncols = znorm_genes_x_spots_ptr->nrows;
    X_calc_ptr->values = NULL; X_calc_ptr->rownames = NULL; X_calc_ptr->colnames = NULL;
    X_calc_ptr->values = (double*)mkl_malloc((size_t)X_calc_ptr->nrows * X_calc_ptr->ncols * sizeof(double), 64);
    X_calc_ptr->rownames = (char**)calloc(X_calc_ptr->nrows, sizeof(char*)); 
    X_calc_ptr->colnames = (char**)calloc(X_calc_ptr->ncols, sizeof(char*)); 
    if (!X_calc_ptr->values || !X_calc_ptr->rownames || !X_calc_ptr->colnames) {
        perror("Memory allocation failed for X_calc_ptr components"); final_status = MORANS_I_ERROR_MEMORY; goto cleanup;
    }

    for (MKL_INT g_main = 0; g_main < X_calc_ptr->ncols; g_main++) { X_calc_ptr->colnames[g_main] = strdup(znorm_genes_x_spots_ptr->rownames[g_main]); if(!X_calc_ptr->colnames[g_main]) {perror("strdup X_calc_ptr gene name"); final_status = MORANS_I_ERROR_MEMORY; goto cleanup;}}
    for (MKL_INT i_main_xc = 0; i_main_xc < X_calc_ptr->nrows; i_main_xc++) { X_calc_ptr->rownames[i_main_xc] = strdup(valid_spot_names_list_ptr[i_main_xc]); if(!X_calc_ptr->rownames[i_main_xc]) {perror("strdup X_calc_ptr spot name"); final_status = MORANS_I_ERROR_MEMORY; goto cleanup;}}

    #pragma omp parallel for
    for (MKL_INT i_valid_spot_main = 0; i_valid_spot_main < X_calc_ptr->nrows; i_valid_spot_main++) { 
        MKL_INT original_expr_spot_col_idx_main = valid_spot_original_indices_ptr[i_valid_spot_main]; 
        for (MKL_INT j_gene_main = 0; j_gene_main < X_calc_ptr->ncols; j_gene_main++) { 
            if (original_expr_spot_col_idx_main != -1) X_calc_ptr->values[i_valid_spot_main * X_calc_ptr->ncols + j_gene_main] = znorm_genes_x_spots_ptr->values[j_gene_main * znorm_genes_x_spots_ptr->ncols + original_expr_spot_col_idx_main];
            else X_calc_ptr->values[i_valid_spot_main * X_calc_ptr->ncols + j_gene_main] = 0.0; 
        }
    }
    main_op_end_time = get_time(); print_elapsed_time(main_op_start_time, main_op_end_time, "final X_calc matrix preparation");

    free_dense_matrix(znorm_genes_x_spots_ptr); znorm_genes_x_spots_ptr = NULL;
    free_spot_coordinates(spot_coords_data_ptr); spot_coords_data_ptr = NULL;
    if (valid_spot_names_list_ptr) { for(MKL_INT k=0; k < num_valid_spots_for_helpers; ++k) {if (valid_spot_names_list_ptr[k]) free(valid_spot_names_list_ptr[k]);} free(valid_spot_names_list_ptr); valid_spot_names_list_ptr = NULL; }
    free(valid_spot_original_indices_ptr); valid_spot_original_indices_ptr = NULL;
    free(valid_spot_rows_for_W_ptr); valid_spot_rows_for_W_ptr = NULL;
    free(valid_spot_cols_for_W_ptr); valid_spot_cols_for_W_ptr = NULL;

    printf("Calculating Moran's I based on selected mode...\n");
    main_op_start_time = get_time();
    char result_output_filename_main[BUFFER_SIZE]; 

    if (!config.calc_pairwise) { 
        snprintf(result_output_filename_main, BUFFER_SIZE, "%s_single_gene_moran_i.tsv", output_file_prefix);
        printf("Mode: Single-Gene Moran's I. Output: %s\n", result_output_filename_main);
        final_status = save_single_gene_results(X_calc_ptr, W_matrix_ptr, S0_val_main, result_output_filename_main);
    } else { 
        if (!config.calc_all_vs_all) { 
            snprintf(result_output_filename_main, BUFFER_SIZE, "%s_first_vs_all_moran_i.tsv", output_file_prefix);
            printf("Mode: Pairwise Moran's I (First Gene vs All Others). Output: %s\n", result_output_filename_main);
            if (X_calc_ptr->ncols == 0) { fprintf(stderr, "Error: No genes in X_calc_ptr for first-vs-all Moran's I.\n"); final_status = MORANS_I_ERROR_PARAMETER;
            } else {
                double* first_vs_all_results_arr_main = calculate_first_gene_vs_all(X_calc_ptr, W_matrix_ptr, S0_val_main); 
                if (first_vs_all_results_arr_main) {
                    final_status = save_first_gene_vs_all_results(first_vs_all_results_arr_main, (const char**)X_calc_ptr->colnames, X_calc_ptr->ncols, result_output_filename_main);
                    mkl_free(first_vs_all_results_arr_main);
                } else { fprintf(stderr, "Error: Failed to calculate Moran's I for first gene vs all others.\n"); final_status = MORANS_I_ERROR_COMPUTATION;}
            }
        } else { 
            char raw_output_filename_main[BUFFER_SIZE]; 
            snprintf(raw_output_filename_main, BUFFER_SIZE, "%s_all_pairs_moran_i_raw.tsv", output_file_prefix);
            printf("Mode: Pairwise Moran's I (All Gene Pairs - Raw Lower Triangular). Output: %s\n", raw_output_filename_main);
            observed_morans_i_matrix_ptr = calculate_morans_i(X_calc_ptr, W_matrix_ptr);
            if (observed_morans_i_matrix_ptr) {
                final_status = save_lower_triangular_matrix_raw(observed_morans_i_matrix_ptr, raw_output_filename_main);
            } else {
                fprintf(stderr, "Error: Failed to calculate all-pairs Moran's I.\n");
                final_status = MORANS_I_ERROR_COMPUTATION;
            }
        }
    }
    main_op_end_time = get_time();
    print_elapsed_time(main_op_start_time, main_op_end_time, "Observed Moran's I calculation and saving");

    if (final_status == MORANS_I_SUCCESS && config.run_permutations) {
        if (observed_morans_i_matrix_ptr != NULL && config.calc_pairwise && config.calc_all_vs_all) {
            printf("--- Running Permutation Test ---\n");
            PermutationParams perm_params_main; 
            perm_params_main.n_permutations = config.num_permutations;
            perm_params_main.seed = config.perm_seed; 
            perm_params_main.z_score_output = config.perm_output_zscores;
            perm_params_main.p_value_output = config.perm_output_pvalues;

            main_op_start_time = get_time();
            perm_results_ptr = run_permutation_test(X_calc_ptr, W_matrix_ptr, &perm_params_main);
            main_op_end_time = get_time();
            print_elapsed_time(main_op_start_time, main_op_end_time, "Permutation Test computation");

            if (perm_results_ptr) {
                main_op_start_time = get_time();
                int save_perm_status_main = save_permutation_results(perm_results_ptr, output_file_prefix);
                if(save_perm_status_main != MORANS_I_SUCCESS) {
                    fprintf(stderr, "Error saving permutation results.\n");
                    if (final_status == MORANS_I_SUCCESS) final_status = save_perm_status_main;
                }
                main_op_end_time = get_time();
                print_elapsed_time(main_op_start_time, main_op_end_time, "Saving Permutation Test results");
            } else {
                fprintf(stderr, "Error: Permutation test failed to produce results.\n");
                if (final_status == MORANS_I_SUCCESS) final_status = MORANS_I_ERROR_COMPUTATION;
            }
        } else if (config.run_permutations) { 
             if (!config.calc_pairwise || !config.calc_all_vs_all) {
                printf("Warning: Permutation testing is primarily designed for 'all gene pairs' mode (-b 1 -g 1).\n");
                printf("         Current mode: -b %d -g %d. Skipping permutations.\n", config.calc_pairwise, config.calc_all_vs_all);
             } else if (observed_morans_i_matrix_ptr == NULL && (config.calc_pairwise && config.calc_all_vs_all) ) {
                printf("Warning: Observed Moran's I calculation failed for all-pairs. Skipping permutations.\n");
             } else {
                printf("Warning: Permutation testing requested but prerequisites not met. Skipping permutations.\n");
             }
        }
    }

cleanup:
    if (vst_matrix_ptr) free_dense_matrix(vst_matrix_ptr);
    if (znorm_genes_x_spots_ptr) free_dense_matrix(znorm_genes_x_spots_ptr);
    if (spot_coords_data_ptr) free_spot_coordinates(spot_coords_data_ptr);
    if (decay_lookup_matrix_ptr) free_dense_matrix(decay_lookup_matrix_ptr);
    if (W_matrix_ptr) free_sparse_matrix(W_matrix_ptr);
    if (X_calc_ptr) free_dense_matrix(X_calc_ptr);
    if (observed_morans_i_matrix_ptr) free_dense_matrix(observed_morans_i_matrix_ptr);
    if (perm_results_ptr) free_permutation_results(perm_results_ptr);

    if (valid_spot_names_list_ptr) {
        for(MKL_INT k=0; k < num_valid_spots_for_helpers; ++k) {
            if(valid_spot_names_list_ptr[k]) free(valid_spot_names_list_ptr[k]);
        }
        free(valid_spot_names_list_ptr);
    }
    if (valid_spot_original_indices_ptr) free(valid_spot_original_indices_ptr);
    if (valid_spot_rows_for_W_ptr) free(valid_spot_rows_for_W_ptr);
    if (valid_spot_cols_for_W_ptr) free(valid_spot_cols_for_W_ptr);

    double total_end_time = get_time();
    print_elapsed_time(total_start_time, total_end_time, "TOTAL EXECUTION");
    printf("--- Moran's I calculation utility finished with status %d ---\n", final_status);
    return final_status;
}