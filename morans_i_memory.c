/*
 * morans_i_memory.c
 *
 * Memory management module for Moran's I computation.
 * Contains all free_* functions for releasing allocated structures.
 *
 * Split from morans_i_mkl.c
 */

#include "morans_i_internal.h"

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
