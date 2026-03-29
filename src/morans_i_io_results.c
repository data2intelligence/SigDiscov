/* morans_i_io_results.c - Results saving module for Moran's I implementation
 *
 * Contains all functions for saving computation results to files: general
 * matrix output, single-gene results, first-gene-vs-all results, lower
 * triangular matrix output, permutation test results, residual analysis
 * results, and regression coefficients.
 *
 * Split from morans_i_io.c
 */

#include "morans_i_internal.h"

/* ===============================
 * FILE I/O AND SAVING FUNCTIONS
 * =============================== */

/* Save regression coefficients */
int save_regression_coefficients(const DenseMatrix* coefficients, const char* output_file) {
    if (!coefficients || !output_file) {
        fprintf(stderr, "Error: Cannot save NULL regression coefficients or empty filename\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }

    MKL_INT n_celltypes = coefficients->nrows;
    MKL_INT n_genes = coefficients->ncols;

    printf("Saving regression coefficients to %s...\n", output_file);

    // Write header
    fprintf(fp, "CellType");
    for (MKL_INT j = 0; j < n_genes; j++) {
        fprintf(fp, "\t%s", coefficients->colnames && coefficients->colnames[j] ?
                coefficients->colnames[j] : "UNKNOWN_GENE");
    }
    fprintf(fp, "\n");

    // Write data rows
    for (MKL_INT i = 0; i < n_celltypes; i++) {
        fprintf(fp, "%s", coefficients->rownames && coefficients->rownames[i] ?
                coefficients->rownames[i] : "UNKNOWN_CELLTYPE");

        for (MKL_INT j = 0; j < n_genes; j++) {
            double value = coefficients->values[i * n_genes + j];
            fprintf(fp, "\t");

            if (isnan(value)) {
                fprintf(fp, "NaN");
            } else if (isinf(value)) {
                fprintf(fp, "%sInf", (value > 0 ? "" : "-"));
            } else {
                if ((fabs(value) > 0 && fabs(value) < 1e-4) || fabs(value) > 1e6) {
                    fprintf(fp, "%.6e", value);
                } else {
                    fprintf(fp, "%.8f", value);
                }
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Regression coefficients saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}

/* Save residual analysis results */
int save_residual_results(const ResidualResults* results, const char* output_prefix) {
    if (!results || !output_prefix) {
        fprintf(stderr, "Error: Cannot save NULL residual results or empty prefix\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    char filename_buffer[BUFFER_SIZE];
    int status = MORANS_I_SUCCESS;

    printf("--- Saving Residual Analysis Results ---\n");

    // Save residual Moran's I matrix (lower triangular)
    if (results->residual_morans_i && results->residual_morans_i->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_residual_morans_i_raw.tsv", output_prefix);
        printf("Saving residual Moran's I (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->residual_morans_i, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving residual Moran's I matrix\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }

    // Save regression coefficients
    if (results->regression_coefficients && results->regression_coefficients->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_regression_coefficients.tsv", output_prefix);
        printf("Saving regression coefficients to: %s\n", filename_buffer);
        int temp_status = save_regression_coefficients(results->regression_coefficients, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving regression coefficients\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }

    // Save residual Z-scores (if available)
    if (results->residual_zscores && results->residual_zscores->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_residual_zscores_lower_tri.tsv", output_prefix);
        printf("Saving residual Z-scores (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->residual_zscores, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving residual Z-scores\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }

    // Save residual P-values (if available)
    if (results->residual_pvalues && results->residual_pvalues->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_residual_pvalues_lower_tri.tsv", output_prefix);
        printf("Saving residual P-values (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->residual_pvalues, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving residual P-values\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }

    // Save residuals matrix (full matrix)
    if (results->residuals && results->residuals->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_residuals_matrix.tsv", output_prefix);
        printf("Saving residuals matrix to: %s\n", filename_buffer);
        int temp_status = save_results(results->residuals, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving residuals matrix\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    }

    if (status == MORANS_I_SUCCESS) {
        printf("Residual analysis results saved with prefix: %s\n", output_prefix);
    }
    return status;
}

/* Save results to file */
int save_results(const DenseMatrix* result_matrix, const char* output_file) {
    if (!result_matrix || !output_file) {
        fprintf(stderr, "Error: Cannot save NULL result matrix or empty filename\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n_rows = result_matrix->nrows;
    MKL_INT n_cols = result_matrix->ncols;

    if ((n_rows == 0 || n_cols == 0) && result_matrix->values != NULL) {
        fprintf(stderr, "Warning: Result matrix has zero dimension but non-NULL values\n");
    }
    if (n_rows > 0 && n_cols > 0 && result_matrix->values == NULL) {
        fprintf(stderr, "Error: Result matrix has non-zero dimensions but NULL values\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }

    if (n_rows == 0 || n_cols == 0) {
        printf("Warning: Result matrix is empty, saving empty file to %s\n", output_file);
        if (result_matrix->colnames != NULL) {
            fprintf(fp, " ");
            for (MKL_INT j = 0; j < n_cols; j++) {
                fprintf(fp, "\t%s", result_matrix->colnames[j] ? result_matrix->colnames[j] : "UNKNOWN_COL");
            }
        }
        fprintf(fp, "\n");
        fclose(fp);
        return MORANS_I_SUCCESS;
    }

    printf("Saving results to %s...\n", output_file);

    // Write header
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

    // Write data rows
    for (MKL_INT i = 0; i < n_rows; i++) {
        fprintf(fp, "%s", (result_matrix->rownames && result_matrix->rownames[i]) ?
                result_matrix->rownames[i] : "UNKNOWN_ROW");

        for (MKL_INT j = 0; j < n_cols; j++) {
            double value = result_matrix->values[i * n_cols + j];
            fprintf(fp, "\t");

            if (isnan(value)) {
                fprintf(fp, "NaN");
            } else if (isinf(value)) {
                fprintf(fp, "%sInf", (value > 0 ? "" : "-"));
            } else {
                if ((fabs(value) > 0 && fabs(value) < 1e-4) || fabs(value) > 1e6) {
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
    return MORANS_I_SUCCESS;
}

/* Save single gene Moran's I results with row normalization support */
int save_single_gene_results(const DenseMatrix* X_calc, const SparseMatrix* W,
                            double S0_unused, const char* output_file, int row_normalized) {
    (void)S0_unused; // Mark as intentionally unused

    if (!X_calc || !W || !output_file) {
        fprintf(stderr, "Error: Invalid parameters provided to save_single_gene_results\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (X_calc->nrows > 0 && X_calc->ncols > 0 && !X_calc->values) {
        fprintf(stderr, "Error: X_calc has non-zero dims but NULL values\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (W->nnz > 0 && !W->values) {
        fprintf(stderr, "Error: W->nnz > 0 but W->values is NULL\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n_spots = X_calc->nrows;
    MKL_INT n_genes = X_calc->ncols;

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }
    fprintf(fp, "Gene\tMoranI\n");

    if (n_genes == 0) {
        printf("Warning: No genes in X_calc, saving header-only file to %s\n", output_file);
        fclose(fp);
        return MORANS_I_SUCCESS;
    }
    if (n_spots == 0) {
        printf("Warning: No spots in X_calc. Moran's I is undefined. Saving NaNs.\n");
        for (MKL_INT g = 0; g < n_genes; g++) {
            fprintf(fp, "%s\tNaN\n", (X_calc->colnames && X_calc->colnames[g]) ?
                    X_calc->colnames[g] : "UNKNOWN_GENE");
        }
        fclose(fp);
        return MORANS_I_SUCCESS;
    }

    double* gene_data_col = (double*)mkl_malloc((size_t)n_spots * sizeof(double), 64);
    if (!gene_data_col) {
        perror("Failed to allocate gene_data_col");
        fclose(fp);
        return MORANS_I_ERROR_MEMORY;
    }

    for (MKL_INT g = 0; g < n_genes; g++) {
        for (MKL_INT spot_idx = 0; spot_idx < n_spots; spot_idx++) {
            gene_data_col[spot_idx] = X_calc->values[spot_idx * n_genes + g];
        }
        double moran_val = calculate_single_gene_moran_i(gene_data_col, W, n_spots, row_normalized);

        fprintf(fp, "%s\t", (X_calc->colnames && X_calc->colnames[g]) ?
                X_calc->colnames[g] : "UNKNOWN_GENE");

        if (isnan(moran_val)) {
            fprintf(fp, "NaN\n");
        } else if (isinf(moran_val)) {
            fprintf(fp, "%sInf\n", (moran_val > 0 ? "" : "-"));
        } else if (fabs(moran_val) > 0 && fabs(moran_val) < 1e-4) {
            fprintf(fp, "%.6e\n", moran_val);
        } else {
            fprintf(fp, "%.8f\n", moran_val);
        }
    }

    mkl_free(gene_data_col);
    fclose(fp);
    printf("Single-gene Moran's I results saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}

/* Save first gene vs all results */
int save_first_gene_vs_all_results(const double* morans_values, const char** gene_names,
                                  MKL_INT n_genes, const char* output_file) {
    if (!output_file) {
        fprintf(stderr, "Error: Null output_file provided\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (n_genes > 0 && (!morans_values || !gene_names)) {
        fprintf(stderr, "Error: Non-zero n_genes but NULL morans_values or gene_names\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }

    const char* first_gene_name = (n_genes > 0 && gene_names && gene_names[0]) ?
                                  gene_names[0] : "FirstGene";
    fprintf(fp, "Gene\tMoranI_vs_%s\n", first_gene_name);

    if (n_genes == 0) {
        printf("Warning: No genes to save, saving header-only file to %s\n", output_file);
        fclose(fp);
        return MORANS_I_SUCCESS;
    }

    printf("Saving first gene vs all results to %s...\n", output_file);
    for (MKL_INT g = 0; g < n_genes; g++) {
        fprintf(fp, "%s\t", (gene_names[g]) ? gene_names[g] : "UNKNOWN_GENE");
        double value = morans_values[g];

        if (isnan(value)) {
            fprintf(fp, "NaN\n");
        } else if (isinf(value)) {
            fprintf(fp, "%sInf\n", (value > 0 ? "" : "-"));
        } else if (fabs(value) > 0 && fabs(value) < 1e-4) {
            fprintf(fp, "%.6e\n", value);
        } else {
            fprintf(fp, "%.8f\n", value);
        }
    }

    fclose(fp);
    printf("First gene vs all results saved to: %s\n", output_file);
    return MORANS_I_SUCCESS;
}

/* Save the lower triangular part of a symmetric matrix */
int save_lower_triangular_matrix_raw(const DenseMatrix* matrix, const char* output_file) {
    if (!matrix || !output_file) {
        fprintf(stderr, "Error: Cannot save NULL matrix or empty filename\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    if (matrix->nrows > 0 && matrix->ncols > 0 && !matrix->values) {
        fprintf(stderr, "Error: Matrix has non-zero dims but NULL values\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (matrix->nrows != matrix->ncols) {
        fprintf(stderr, "Error: Matrix must be square (dims %lldx%lld)\n",
                (long long)matrix->nrows, (long long)matrix->ncols);
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n = matrix->nrows;

    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file '%s': %s\n", output_file, strerror(errno));
        return MORANS_I_ERROR_FILE;
    }

    if (n == 0) {
        printf("Warning: Matrix is empty, saving empty file to %s\n", output_file);
        fclose(fp);
        return MORANS_I_SUCCESS;
    }

    printf("Saving lower triangular matrix to %s...\n", output_file);

    for (MKL_INT i = 0; i < n; i++) {
        for (MKL_INT j = 0; j <= i; j++) {
            double value = matrix->values[i * n + j];

            if (j > 0) {
                fprintf(fp, "\t");
            }

            if (isnan(value)) {
                fprintf(fp, "NaN");
            } else if (isinf(value)) {
                fprintf(fp, "%sInf", (value > 0 ? "" : "-"));
            } else {
                if (fabs(value) != 0.0 && (fabs(value) < 1e-4 || fabs(value) > 1e6)) {
                    fprintf(fp, "%.6e", value);
                } else {
                    fprintf(fp, "%.8f", value);
                }
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return MORANS_I_SUCCESS;
}

/* Save permutation test results */
int save_permutation_results(const PermutationResults* results, const char* output_file_prefix) {
    if (!results || !output_file_prefix) {
        fprintf(stderr, "Error: Cannot save NULL results or empty output prefix\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    char filename_buffer[BUFFER_SIZE];
    int status = MORANS_I_SUCCESS;

    printf("--- Saving Permutation Test Results (Lower Triangular Raw Format) ---\n");

    if (results->z_scores && results->z_scores->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_zscores_lower_tri.tsv", output_file_prefix);
        printf("Saving Z-scores (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->z_scores, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving Z-scores\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    } else {
        printf("Z-scores not calculated, skipping save\n");
    }

    if (results->p_values && results->p_values->values) {
        snprintf(filename_buffer, BUFFER_SIZE, "%s_pvalues_lower_tri.tsv", output_file_prefix);
        printf("Saving P-values (lower triangular) to: %s\n", filename_buffer);
        int temp_status = save_lower_triangular_matrix_raw(results->p_values, filename_buffer);
        if (temp_status != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error saving P-values\n");
            if (status == MORANS_I_SUCCESS) status = temp_status;
        }
    } else {
        printf("P-values not calculated, skipping save\n");
    }

    if (status == MORANS_I_SUCCESS) {
        printf("Permutation test results saved with prefix: %s\n", output_file_prefix);
    }
    return status;
}
