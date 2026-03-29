/* morans_i_io_weights.c - Weight matrix I/O module for Moran's I implementation
 *
 * Contains functions for reading custom spatial weight matrices from files
 * in dense, sparse COO, and sparse TSV formats, plus format auto-detection
 * and validation against expression data.
 *
 * Split from morans_i_io.c
 */

#include "morans_i_internal.h"

/* ===============================
 * CUSTOM WEIGHT MATRIX FUNCTIONS
 * =============================== */

/* Detect weight matrix file format automatically */
int detect_weight_matrix_format(const char* filename) {
    if (!filename) {
        fprintf(stderr, "Error: NULL filename in detect_weight_matrix_format\n");
        return WEIGHT_FORMAT_AUTO; // Return AUTO to signify error or let caller handle
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        // Don't print strerror directly in library function if it's meant to be more generic,
        // but for this tool it's fine.
        fprintf(stderr, "Error: Cannot open weight matrix file '%s' for format detection: %s\n",
                filename, strerror(errno));
        return WEIGHT_FORMAT_AUTO; // Indicate failure to detect
    }

    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    int detected_format = WEIGHT_FORMAT_AUTO; // Default to AUTO/error

    line_len = getline(&line, &line_buf_size, fp);

    if (line_len > 0) {
        // Trim whitespace (especially newline characters)
        char* current_line_ptr = line;
        while (line_len > 0 && (current_line_ptr[line_len - 1] == '\n' || current_line_ptr[line_len - 1] == '\r')) {
            current_line_ptr[--line_len] = '\0';
        }

        if (line_len > 0) { // Ensure line is not empty after trimming
            char* line_copy = strdup(current_line_ptr); // Work on a copy for strtok
            if (!line_copy) {
                perror("strdup in detect_weight_matrix_format");
                // Fallthrough to cleanup and return WEIGHT_FORMAT_AUTO
            } else {
                int field_count = 0;
                char* token = strtok(line_copy, "\t");
                while (token != NULL) {
                    field_count++;
                    token = strtok(NULL, "\t");
                }
                free(line_copy); // Free the copy

                if (field_count == 3) {
                    // Further heuristics to distinguish COO from TSV if needed
                    // For example, check if the first two fields look like coordinates (contain 'x')
                    // or if they are more general strings.
                    // This simple version might misclassify if spot names are "1x1", "1x2" in TSV.
                    // A more robust check might try to parse the third field as a number.
                    // For now, let's use a simplified heuristic based on typical use.
                    // If the original code had a more complex regex for this, reinstate it.
                    // The provided code originally had a regex in a different context.
                    // A simple heuristic: If fields are short and potentially numeric/coord-like.
                    char* line_copy2 = strdup(current_line_ptr); // New copy for checking content
                    if (line_copy2) {
                        char* first_field = strtok(line_copy2, "\t");
                        if (first_field) {
                            // If first field looks like a coordinate (e.g., "12x34")
                            // This is a basic check; a regex would be more robust.
                            // A simple check: contains 'x' and digits.
                            int has_x = 0, has_digit = 0;
                            char* p = first_field;
                            while(*p) {
                                if (*p == 'x' || *p == 'X') has_x = 1;
                                if (isdigit((unsigned char)*p)) has_digit = 1;
                                p++;
                            }
                            if (has_x && has_digit) {
                                detected_format = WEIGHT_FORMAT_SPARSE_COO;
                            } else {
                                detected_format = WEIGHT_FORMAT_SPARSE_TSV;
                            }
                        } else { // Should not happen if field_count was 3
                             detected_format = WEIGHT_FORMAT_SPARSE_TSV;
                        }
                        free(line_copy2);
                    } else {
                        perror("strdup line_copy2 in detect_weight_matrix_format");
                        detected_format = WEIGHT_FORMAT_SPARSE_TSV; // Fallback
                    }

                } else if (field_count > 3) {
                    detected_format = WEIGHT_FORMAT_DENSE;
                } else if (field_count < 3 && field_count > 0) {
                    // Could be an error or a very sparse format not well-defined
                    fprintf(stderr, "Warning: File '%s' has only %d fields in the first line. Format detection might be inaccurate.\n", filename, field_count);
                    detected_format = WEIGHT_FORMAT_SPARSE_TSV; // Default for few fields
                } else { // field_count == 0 (empty line was already handled) or other unexpected
                    fprintf(stderr, "Warning: Could not determine format from first line of '%s' (field_count=%d).\n", filename, field_count);
                }
            }
        } else { // Line became empty after trimming EOL
            fprintf(stderr, "Warning: First line of '%s' is effectively empty after EOL trimming.\n", filename);
        }
    } else { // getline failed or empty file
        if (ferror(fp)) {
            fprintf(stderr, "Error reading from '%s' for format detection: %s\n", filename, strerror(errno));
        } else {
            fprintf(stderr, "Warning: File '%s' is empty or unreadable for format detection.\n", filename);
        }
    }

    if (line) free(line);
    fclose(fp);

    const char* format_names[] = {"AUTO/ERROR", "DENSE", "SPARSE_COO", "SPARSE_TSV"};
    if (detected_format > 0 && detected_format < 4) { // Valid detected format
        printf("Detected weight matrix format: %s\n", format_names[detected_format]);
    } else {
        printf("Could not reliably detect weight matrix format for '%s'. Defaulting or error.\n", filename);
    }

    return detected_format;
}

/* Validate that weight matrix is compatible with expression data */
int validate_weight_matrix(const SparseMatrix* W, char** spot_names, MKL_INT n_spots) {
    // spot_names parameter is from the expression data (X_calc->rownames or equivalent)
    // n_spots is the number of rows/cols W *should* have, matching expression data.
    if (!W || (n_spots > 0 && !spot_names)) { // Allow n_spots = 0 if W is also empty
        fprintf(stderr, "Error: NULL parameters in validate_weight_matrix (W=%p, spot_names=%p for n_spots=%lld)\n",
                (void*)W, (void*)spot_names, (long long)n_spots);
        return MORANS_I_ERROR_PARAMETER;
    }

    if (W->nrows != n_spots || W->ncols != n_spots) {
        fprintf(stderr, "Error: Custom Weight matrix dimensions (%lldx%lld) do not match the number of spots/cells (%lld) from expression data.\n",
                (long long)W->nrows, (long long)W->ncols, (long long)n_spots);
        return MORANS_I_ERROR_PARAMETER;
    }

    // Check for negative weights
    MKL_INT negative_weights = 0;
    if (W->values) { // W->values can be NULL if W->nnz is 0
        for (MKL_INT i = 0; i < W->nnz; i++) {
            if (W->values[i] < 0.0) {
                negative_weights++;
            }
        }
    }

    if (negative_weights > 0) {
        fprintf(stderr, "Warning: Found %lld negative weights in custom weight matrix. This is unusual for standard spatial weights but might be intended.\n",
               (long long)negative_weights);
    }

    double weight_sum = calculate_weight_sum(W); // Uses W->values and W->nnz
    printf("Custom weight matrix validation: %lldx%lld matrix, %lld non-zeros, sum of weights (S0) = %.6f\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz, weight_sum);

    if (W->nnz > 0 && fabs(weight_sum) < ZERO_STD_THRESHOLD) { // Check sum only if matrix is not empty
        fprintf(stderr, "Warning: Sum of custom weights is near zero (%.6e) for a non-empty matrix. Moran's I may be undefined or unstable.\n",
                weight_sum);
    }
    if (W->nnz == 0 && n_spots > 0) {
        fprintf(stderr, "Warning: Custom weight matrix has 0 non-zero entries for %lld spots. Moran's I will be 0 or undefined.\n", (long long)n_spots);
    }

    // Optional: Check if row/column indices are within bounds [0, n_spots-1]
    // This should be guaranteed if W was constructed correctly, but for an arbitrary custom W:
    if (W->col_ind && W->nnz > 0) {
        for (MKL_INT i = 0; i < W->nnz; ++i) {
            if (W->col_ind[i] < 0 || W->col_ind[i] >= W->ncols) {
                fprintf(stderr, "Error: Invalid column index %lld (at NNZ index %lld) in custom weight matrix. Expected range [0, %lld].\n",
                        (long long)W->col_ind[i], (long long)i, (long long)W->ncols -1);
                return MORANS_I_ERROR_PARAMETER;
            }
        }
    }
    if (W->row_ptr) {
        for (MKL_INT i = 0; i < W->nrows; ++i) {
            if (W->row_ptr[i] > W->row_ptr[i+1] || W->row_ptr[i] < 0 || W->row_ptr[i+1] > W->nnz) {
                 fprintf(stderr, "Error: Invalid row_ptr entry for row %lld in custom weight matrix.\n", (long long)i);
                 return MORANS_I_ERROR_PARAMETER;
            }
        }
         if (W->row_ptr[0] != 0 || W->row_ptr[W->nrows] != W->nnz) {
             fprintf(stderr, "Error: Invalid CSR row_ptr structure (start/end mismatch) in custom weight matrix.\n");
             return MORANS_I_ERROR_PARAMETER;
         }
    }


    return MORANS_I_SUCCESS;
}

/* Helper function to determine data start offset by checking for a header */
static long get_data_start_offset(FILE* fp, char** line_buffer, size_t* line_buf_size,
                                  const char* filename_for_errmsg) {
    long original_pos = ftell(fp);
    if (original_pos == -1) {
        perror("get_data_start_offset: ftell failed at start");
        return -1L;
    }
    rewind(fp); // Go to the beginning to check the first line

    ssize_t line_len = getline(line_buffer, line_buf_size, fp);
    if (line_len <= 0) { // Empty or unreadable file
        if (ferror(fp)) {
             fprintf(stderr, "Error reading first line of '%s': %s\n", filename_for_errmsg, strerror(errno));
        } else {
             fprintf(stderr, "Warning: File '%s' is empty or first line unreadable.\n", filename_for_errmsg);
        }
        fseek(fp, original_pos, SEEK_SET); // Restore original position
        return 0L; // Assume no header, start from beginning (though it's empty)
    }

    // Trim EOL chars from the line before copying
    char* current_line_ptr = *line_buffer;
    while (line_len > 0 && (current_line_ptr[line_len - 1] == '\n' || current_line_ptr[line_len - 1] == '\r')) {
        current_line_ptr[--line_len] = '\0';
    }

    char* first_line_copy = strdup(current_line_ptr);
    if (!first_line_copy) {
        perror("get_data_start_offset: strdup failed");
        fseek(fp, original_pos, SEEK_SET);
        return -1L;
    }

    char* token = strtok(first_line_copy, "\t");
    int looks_like_header = 0;
    if (token) {
        // Heuristic: check for common header keywords or non-numeric/non-coord-like first token
        // This heuristic is specific to expected formats.
        // For sparse_tsv/coo: first token might be a spot name or coordinate.
        // If it contains alpha characters not part of "x" (for "RxC"), it's likely a header.
        int has_alpha = 0;
        char *p = token;
        while(*p) {
            if (isalpha((unsigned char)*p) && *p != 'x' && *p != 'X') { // Allow 'x' for RxC format
                has_alpha = 1;
                break;
            }
            p++;
        }
        // A more robust check specific to the format (e.g. _tsv checks for "spot", "weight")
        // could be done if format is known here. For now, a general alpha check.
        if (has_alpha || strstr(token, "spot") || strstr(token, "weight") || strstr(token, "coord") ||
            strstr(token, "row") || strstr(token, "col") || strstr(token, "from") || strstr(token, "to")) {
            looks_like_header = 1;
        }
    }
    free(first_line_copy);

    long data_offset;
    if (looks_like_header) {
        data_offset = ftell(fp); // Position after reading the header line
        if (data_offset == -1L) {
            perror("get_data_start_offset: ftell failed after header read");
            fseek(fp, original_pos, SEEK_SET);
            return -1L;
        }
    } else {
        data_offset = 0; // No header, data starts at the beginning
    }

    // Restore original file pointer position before returning, or seek to data_offset
    // Let's keep it simple: the caller will fseek to the returned offset.
    // So, ensure fp is reset to its original state if we don't want this function to change it.
    // OR, this function's purpose is to position fp correctly for the next read.
    // The current callers (read_sparse_matrix_tsv) handle fseek themselves.
    // So, just return offset. We already rewound fp.
    return data_offset;
}

/* Read dense weight matrix from TSV file */
SparseMatrix* read_dense_weight_matrix(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots) {
    if (!filename || !spot_map || n_spots <= 0) {
        fprintf(stderr, "Error: Invalid parameters in read_dense_weight_matrix\n");
        return NULL;
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open dense weight matrix file '%s': %s\n",
                filename, strerror(errno));
        return NULL;
    }

    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;

    printf("Reading dense weight matrix from '%s'...\n", filename);

    // Read header to get column spot names
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty header in dense weight matrix file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
        line[--line_len] = '\0';
    }

    char* header_copy = strdup(line);
    if (!header_copy) {
        perror("strdup header in read_dense_weight_matrix");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Parse header to get column spot indices
    MKL_INT* col_spot_indices = (MKL_INT*)malloc((size_t)n_spots * sizeof(MKL_INT));
    if (!col_spot_indices) {
        perror("malloc col_spot_indices");
        free(header_copy);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    char* token = strtok(header_copy, "\t"); // Skip first field (row label placeholder)
    if (!token) {
        fprintf(stderr, "Error: Empty header line in dense weight matrix\n");
        free(header_copy);
        free(col_spot_indices);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    MKL_INT n_cols_found = 0;
    token = strtok(NULL, "\t");
    while (token && n_cols_found < n_spots) {
        char* trimmed = trim_whitespace_inplace(token);
        MKL_INT spot_idx = spot_name_ht_find(spot_map, trimmed);
        if (spot_idx >= 0) {
            col_spot_indices[n_cols_found++] = spot_idx;
        } else {
            fprintf(stderr, "Warning (Dense Read): Column spot '%s' in weight matrix header not found in expression data spot list. Skipping column.\n", trimmed);
        }
        token = strtok(NULL, "\t");
    }
    free(header_copy);

    if (n_cols_found == 0 && n_spots > 0) {
        fprintf(stderr, "Error: No matching column spots found in dense weight matrix header and expression data.\n");
        free(col_spot_indices);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    printf("  Found %lld matching column spots in header\n", (long long)n_cols_found);

    // Count potential NNZ and read data
    typedef struct { MKL_INT row, col; double value; } triplet_t;

    // Estimate NNZ capacity
    MKL_INT estimated_max_nnz = n_spots * n_cols_found; // Upper bound for dense
    if (estimated_max_nnz <= 0) estimated_max_nnz = 1000000; // Fallback

    triplet_t* triplets = (triplet_t*)malloc((size_t)estimated_max_nnz * sizeof(triplet_t));
    if (!triplets) {
        perror("Failed to allocate triplets for dense weight matrix");
        free(col_spot_indices);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    MKL_INT nnz_count = 0;
    int data_rows_processed = 0;

    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        // Trim line
        while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
            line[--line_len] = '\0';
        }
        if (line_len == 0) continue; // Skip empty lines

        char* line_copy_data = strdup(line);
        if (!line_copy_data) {
            perror("strdup data line in dense weight matrix");
            break;
        }

        char* row_label_token = strtok(line_copy_data, "\t");
        if (!row_label_token) {
            free(line_copy_data);
            continue;
        }

        char* row_spot_name = trim_whitespace_inplace(row_label_token);
        MKL_INT row_idx = spot_name_ht_find(spot_map, row_spot_name);

        if (row_idx >= 0) {
            // Read values for this row
            for (MKL_INT col_file_idx = 0; col_file_idx < n_cols_found; ++col_file_idx) {
                char* val_token = strtok(NULL, "\t");
                if (!val_token) {
                    fprintf(stderr, "Warning: Row '%s' has fewer columns than expected\n", row_spot_name);
                    break;
                }

                char* endptr;
                double weight = strtod(val_token, &endptr);
                if (endptr != val_token && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                    if (nnz_count < estimated_max_nnz) {
                        triplets[nnz_count].row = row_idx;
                        triplets[nnz_count].col = col_spot_indices[col_file_idx];
                        triplets[nnz_count].value = weight;
                        nnz_count++;
                    } else {
                        fprintf(stderr, "Warning: Exceeded estimated_max_nnz in dense read. Some data might be lost.\n");
                        break;
                    }
                }
            }
            data_rows_processed++;
        } else {
            if (data_rows_processed < 10) { // Limit warnings
                fprintf(stderr, "Warning: Row spot '%s' not found in expression data\n", row_spot_name);
            }
        }
        free(line_copy_data);
    }

    fclose(fp);
    if (line) free(line);
    free(col_spot_indices);

    printf("Dense weight matrix read: %lld non-zero entries from %d data rows\n",
           (long long)nnz_count, data_rows_processed);

    // Convert to sparse CSR format
    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!W) {
        perror("Failed to allocate SparseMatrix for dense weight matrix");
        free(triplets);
        return NULL;
    }

    W->nrows = n_spots;
    W->ncols = n_spots;
    W->nnz = nnz_count;
    W->rownames = NULL;
    W->colnames = NULL;

    W->row_ptr = (MKL_INT*)mkl_calloc((size_t)n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc((size_t)nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc((size_t)nnz_count * sizeof(double), 64);
    } else {
        W->col_ind = NULL;
        W->values = NULL;
    }

    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) {
        perror("Failed to allocate CSR arrays for dense weight matrix");
        free_sparse_matrix(W);
        free(triplets);
        return NULL;
    }

    if (nnz_count > 0) {
        printf("  Converting dense matrix to CSR format...\n");

        // Count entries per row
        for (MKL_INT k = 0; k < nnz_count; k++) {
            W->row_ptr[triplets[k].row + 1]++;
        }

        // Convert counts to cumulative sums
        for (MKL_INT i = 0; i < n_spots; i++) {
            W->row_ptr[i + 1] += W->row_ptr[i];
        }

        // Fill CSR arrays
        MKL_INT* csr_fill_counters = (MKL_INT*)calloc((size_t)n_spots, sizeof(MKL_INT));
        if (!csr_fill_counters) {
            perror("Failed to allocate CSR fill counters for dense weight matrix");
            free_sparse_matrix(W);
            free(triplets);
            return NULL;
        }

        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT r = triplets[k].row;
            MKL_INT insert_at = W->row_ptr[r] + csr_fill_counters[r];
            if (insert_at < nnz_count) { // Bounds check
                W->col_ind[insert_at] = triplets[k].col;
                W->values[insert_at] = triplets[k].value;
                csr_fill_counters[r]++;
            } else {
                fprintf(stderr, "Error: CSR insertion index out of bounds in dense matrix conversion\n");
                break;
            }
        }
        free(csr_fill_counters);
        printf("  CSR conversion complete\n");
    }

    free(triplets);

    printf("Dense weight matrix converted to CSR format: %lldx%lld with %lld NNZ\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz);

    return W;
}


/* Read sparse weight matrix in COO format */
SparseMatrix* read_sparse_weight_matrix_coo(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots) {
    // This function will be very similar to read_sparse_weight_matrix_tsv,
    // using two passes and the spot_map for lookups.
    // The main difference is parsing "RxC" style coordinates if used,
    // OR direct spot name lookup using spot_map.
    // For simplicity and given the user's current file is TSV, this is sketched:
    if (!filename || !spot_map || n_spots <= 0) return NULL;

    FILE* fp = fopen(filename, "r");
    if (!fp) return NULL;

    char *line = NULL; size_t line_buf_size = 0; ssize_t line_len;
    printf("Reading sparse COO weight matrix from '%s'...\n", filename);

    long data_start_offset = get_data_start_offset(fp, &line, &line_buf_size, filename);
    if (data_start_offset < 0) { fclose(fp); if(line) free(line); return NULL; }

    fseek(fp, data_start_offset, SEEK_SET);
    MKL_INT true_nnz = 0;
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        char* p = line; while(isspace((unsigned char)*p)) p++;
        if(*p != '\0') true_nnz++;
    }

    typedef struct { MKL_INT row, col; double value; } triplet_t;
    triplet_t* triplets = NULL;
    if (true_nnz > 0) {
        triplets = (triplet_t*)malloc((size_t)true_nnz * sizeof(triplet_t));
        if (!triplets) { /* error */ fclose(fp); if(line) free(line); return NULL; }
    } else {
        // Handle case of no data lines after header, or empty file
        fclose(fp); if(line) free(line);
        SparseMatrix* W_empty = (SparseMatrix*)calloc(1, sizeof(SparseMatrix));
        W_empty->nrows = n_spots; W_empty->ncols = n_spots; W_empty->nnz = 0;
        W_empty->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
        return W_empty; // Return empty matrix
    }

    fseek(fp, data_start_offset, SEEK_SET);
    MKL_INT nnz_count = 0;
    regex_t coord_regex;
    int regex_compiled = (regcomp(&coord_regex, "^([0-9]+)[xX]([0-9]+)$", REG_EXTENDED) == 0);


    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && nnz_count < true_nnz) {
        // ... (current line trimming) ...
        char* row_copy = strdup(line);
        if (!row_copy) { /* error */ break; }

        char* row_coord_str = strtok(row_copy, "\t");
        char* col_coord_str = strtok(NULL, "\t");
        char* weight_str = strtok(NULL, "\t");

        if (row_coord_str && col_coord_str && weight_str) {
            MKL_INT row_idx = spot_name_ht_find(spot_map, trim_whitespace_inplace(row_coord_str));
            MKL_INT col_idx = spot_name_ht_find(spot_map, trim_whitespace_inplace(col_coord_str));
            // The original regex logic for RxC was to parse RxC strings.
            // If spot_map keys are RxC, then direct lookup is fine.
            // If spot_map keys are other IDs but file uses RxC, then a mapping RxC -> canonical ID -> index is needed,
            // which is more complex than current scope. Assuming file uses names compatible with spot_map.

            if (row_idx >= 0 && col_idx >= 0) {
                char* endptr;
                double weight = strtod(weight_str, &endptr);
                if (endptr != weight_str && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                    triplets[nnz_count].row = row_idx;
                    triplets[nnz_count].col = col_idx;
                    triplets[nnz_count].value = weight;
                    nnz_count++;
                }
            }
        }
        free(row_copy);
    }

    if (regex_compiled) regfree(&coord_regex);
    fclose(fp);
    if (line) free(line);

    // ... (CSR conversion based on nnz_count, then free triplets) ...
    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    // ... (Initialize W, W->nnz = nnz_count, allocate CSR arrays) ...
    // (Same CSR conversion logic as in read_sparse_weight_matrix_tsv)
    // Free triplets at the end.
    if (!W) { /* ... */ if (triplets) free(triplets); return NULL; }
    W->nrows = n_spots; W->ncols = n_spots; W->nnz = nnz_count;
    W->row_ptr = (MKL_INT*)mkl_calloc(n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc((size_t)nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc((size_t)nnz_count * sizeof(double), 64);
    } else {
        W->col_ind = NULL; W->values = NULL;
    }
    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) { /* ... */ if (triplets) free(triplets); free_sparse_matrix(W); return NULL;}

    if (nnz_count > 0) {
        for (MKL_INT k = 0; k < nnz_count; k++) W->row_ptr[triplets[k].row + 1]++;
        for (MKL_INT i = 0; i < n_spots; i++) W->row_ptr[i+1] += W->row_ptr[i];

        MKL_INT* csr_fill_counters = (MKL_INT*)calloc(n_spots, sizeof(MKL_INT));
        if(!csr_fill_counters) { /* ... */ if (triplets) free(triplets); free_sparse_matrix(W); return NULL;}
        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT r = triplets[k].row;
            MKL_INT insert_at = W->row_ptr[r] + csr_fill_counters[r];
            W->col_ind[insert_at] = triplets[k].col;
            W->values[insert_at] = triplets[k].value;
            csr_fill_counters[r]++;
        }
        free(csr_fill_counters);
    }
    if (triplets) free(triplets);
    return W;
}

/* Read sparse weight matrix in TSV format */
SparseMatrix* read_sparse_weight_matrix_tsv(const char* filename, SpotNameHashTable* spot_map, MKL_INT n_spots) {
    if (!filename || !spot_map || n_spots < 0) { // n_spots can be 0 for empty expression matrix
        fprintf(stderr, "Error: Invalid parameters in read_sparse_weight_matrix_tsv\n");
        return NULL;
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open sparse TSV weight matrix file '%s': %s\n",
                filename, strerror(errno));
        return NULL;
    }

    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;

    printf("Reading sparse TSV weight matrix from '%s'...\n", filename);

    // Determine data start offset by checking for a header
    long data_start_offset = get_data_start_offset(fp, &line, &line_buf_size, filename);
    if (data_start_offset < 0) { // Error in get_data_start_offset
        fclose(fp); if(line) free(line);
        return NULL;
    }

    // Single-pass reading with dynamic allocation
    typedef struct { MKL_INT row, col; double value; } triplet_t;
    triplet_t* triplets = NULL;
    MKL_INT triplets_capacity = 0;
    MKL_INT nnz_count = 0; // Actual valid triplets found

    // Initial capacity estimate (be generous to avoid frequent reallocations)
    MKL_INT initial_capacity = 1000000; // Start with 1M entries
    if (n_spots > 0) {
        // Estimate ~20-50 connections per spot for typical spatial transcriptomics data
        size_t estimated_nnz;
        if (safe_multiply_size_t(n_spots, 35, &estimated_nnz) == 0 && estimated_nnz < 50000000) {
            initial_capacity = (MKL_INT)estimated_nnz;
        } else {
            initial_capacity = 50000000; // Cap at 50M to avoid excessive memory
        }
    }

    triplets_capacity = initial_capacity;
    triplets = (triplet_t*)malloc((size_t)triplets_capacity * sizeof(triplet_t));
    if (!triplets) {
        perror("read_sparse_weight_matrix_tsv: malloc initial triplets failed");
        fclose(fp); if(line) free(line); return NULL;
    }

    printf("  Starting with triplets capacity: %lld\n", (long long)triplets_capacity);

    // Single pass: Read and parse data lines
    if (fseek(fp, data_start_offset, SEEK_SET) != 0) {
        perror("read_sparse_weight_matrix_tsv: fseek for data reading failed");
        free(triplets); fclose(fp); if(line) free(line); return NULL;
    }

    int line_num = 0; // For error messages
    int resize_count = 0; // Track number of resizes

    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        line_num++;
        char* p_check_empty = line; // Check if line is not just whitespace
        while(isspace((unsigned char)*p_check_empty)) p_check_empty++;
        if(*p_check_empty == '\0') continue; // Skip empty or whitespace-only lines

        // Grow triplets array if needed
        if (nnz_count >= triplets_capacity) {
            MKL_INT new_capacity = triplets_capacity * 2;
            if (new_capacity < triplets_capacity) { // Overflow check
                fprintf(stderr, "Error: Triplets array size overflow at line %d\n", line_num);
                break;
            }

            triplet_t* new_triplets = (triplet_t*)realloc(triplets, (size_t)new_capacity * sizeof(triplet_t));
            if (!new_triplets) {
                perror("read_sparse_weight_matrix_tsv: realloc triplets failed");
                break;
            }
            triplets = new_triplets;
            triplets_capacity = new_capacity;
            resize_count++;

            // Print resize notification, but not too frequently
            if (resize_count <= 10 || resize_count % 5 == 0) {
                printf("  Resized triplets array to %lld entries (resize #%d)\n",
                       (long long)new_capacity, resize_count);
            }
        }

        char* row_copy = strdup(line); // strdup the original line content from buffer
        if (!row_copy) {
            perror("read_sparse_weight_matrix_tsv: strdup data line failed");
            break; // Critical memory error
        }

        char* spot1_str = strtok(row_copy, "\t");
        char* spot2_str = strtok(NULL, "\t");
        char* weight_str = strtok(NULL, "\t");

        if (!spot1_str || !spot2_str || !weight_str) {
            if (line_num <= 10 || line_num % 100000 == 0) { // Limit warning frequency
                fprintf(stderr, "Warning: Line %d in '%s' has too few fields. Skipping.\n", line_num, filename);
            }
            free(row_copy);
            continue;
        }

        char* spot1_trimmed = trim_whitespace_inplace(spot1_str);
        char* spot2_trimmed = trim_whitespace_inplace(spot2_str);
        char* weight_trimmed = trim_whitespace_inplace(weight_str);  // CRITICAL FIX: Trim weight string

        MKL_INT row_idx = spot_name_ht_find(spot_map, spot1_trimmed);
        MKL_INT col_idx = spot_name_ht_find(spot_map, spot2_trimmed);

        if (row_idx >= 0 && col_idx >= 0) {
            char* endptr;
            errno = 0;  // Clear errno before parsing
            double weight = strtod(weight_trimmed, &endptr);

            // ENHANCED VALIDATION: More lenient about trailing whitespace
            if (errno == 0 && endptr != weight_trimmed && isfinite(weight) && fabs(weight) > WEIGHT_THRESHOLD) {
                // Additional check: ensure no significant trailing characters
                while (*endptr && isspace((unsigned char)*endptr)) endptr++;

                if (*endptr == '\0') {  // Now check if we've consumed all significant characters
                    triplets[nnz_count].row = row_idx;
                    triplets[nnz_count].col = col_idx;
                    triplets[nnz_count].value = weight;
                    nnz_count++;
                } else {
                    if (line_num <= 10 || line_num % 100000 == 0) {
                        fprintf(stderr, "Warning: Line %d in '%s': Invalid weight '%s' (trailing chars). Skipping.\n",
                                line_num, filename, weight_trimmed);
                    }
                }
            } else {
                if (line_num <= 10 || line_num % 100000 == 0) { // Limit warning frequency
                    fprintf(stderr, "Warning: Line %d in '%s': Invalid weight '%s' or value too small (%.2e <= %.2e). Skipping.\n",
                            line_num, filename, weight_trimmed, fabs(weight), WEIGHT_THRESHOLD);
                }
            }
        } else {
            // Only show spot lookup warnings for first few lines to avoid spam
            if (line_num <= 10) {
                if (row_idx < 0) fprintf(stderr, "Warning: Line %d in '%s': Spot1 '%s' not found in expression data. Skipping entry.\n",
                                        line_num, filename, spot1_trimmed);
                if (col_idx < 0) fprintf(stderr, "Warning: Line %d in '%s': Spot2 '%s' not found in expression data. Skipping entry.\n",
                                        line_num, filename, spot2_trimmed);
            }
        }
        free(row_copy);

        // Progress reporting for large files
        if (line_num % 1000000 == 0) {
            printf("  Processed %d lines, found %lld valid entries\n", line_num, (long long)nnz_count);
        }
    }

    fclose(fp);
    if (line) free(line);

    printf("Sparse TSV weight matrix read: %lld valid non-zero entries found from %d data lines.\n",
           (long long)nnz_count, line_num);

    if (resize_count > 0) {
        printf("  Array was resized %d times during reading\n", resize_count);
    }

    // Trim triplets array to actual size to save memory
    if (nnz_count < triplets_capacity && nnz_count > 0) {
        triplet_t* trimmed_triplets = (triplet_t*)realloc(triplets, (size_t)nnz_count * sizeof(triplet_t));
        if (trimmed_triplets) {
            triplets = trimmed_triplets;
            triplets_capacity = nnz_count;
            printf("  Trimmed triplets array to final size: %lld entries\n", (long long)nnz_count);
        }
        // If realloc fails, just continue with the larger array - not a critical error
    }

    SparseMatrix* W = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!W) {
        perror("read_sparse_weight_matrix_tsv: malloc SparseMatrix failed");
        if (triplets) free(triplets);
        return NULL;
    }

    W->nrows = n_spots;
    W->ncols = n_spots;
    W->nnz = nnz_count; // Use actual valid count
    W->rownames = NULL; W->colnames = NULL;

    W->row_ptr = (MKL_INT*)mkl_calloc((size_t)n_spots + 1, sizeof(MKL_INT), 64);
    if (nnz_count > 0) {
        W->col_ind = (MKL_INT*)mkl_malloc((size_t)nnz_count * sizeof(MKL_INT), 64);
        W->values = (double*)mkl_malloc((size_t)nnz_count * sizeof(double), 64);
    } else {
        W->col_ind = NULL; W->values = NULL;
    }

    if (!W->row_ptr || (nnz_count > 0 && (!W->col_ind || !W->values))) {
        perror("read_sparse_weight_matrix_tsv: Failed to allocate CSR arrays");
        free_sparse_matrix(W); // Will free W->row_ptr if it was allocated
        if (triplets) free(triplets);
        return NULL;
    }

    if (nnz_count > 0) {
        printf("  Converting to CSR format...\n");

        // Count entries per row
        for (MKL_INT k = 0; k < nnz_count; k++) {
            W->row_ptr[triplets[k].row + 1]++;
        }

        // Convert counts to cumulative sums (CSR row_ptr format)
        for (MKL_INT i = 0; i < n_spots; i++) {
            W->row_ptr[i + 1] += W->row_ptr[i];
        }

        // Fill CSR arrays
        MKL_INT* csr_fill_counters = (MKL_INT*)calloc(n_spots, sizeof(MKL_INT));
        if (!csr_fill_counters) {
            perror("read_sparse_weight_matrix_tsv: calloc csr_fill_counters failed");
            free_sparse_matrix(W); if (triplets) free(triplets); return NULL;
        }

        for (MKL_INT k = 0; k < nnz_count; k++) {
            MKL_INT r = triplets[k].row;
            MKL_INT insert_at = W->row_ptr[r] + csr_fill_counters[r];
            if (insert_at < nnz_count) { // Bounds check
                 W->col_ind[insert_at] = triplets[k].col;
                 W->values[insert_at] = triplets[k].value;
                 csr_fill_counters[r]++;
            } else {
                fprintf(stderr, "Error: CSR insertion index out of bounds. This should not happen.\n");
                // This indicates a logic error in CSR construction
            }
        }
        free(csr_fill_counters);
        printf("  CSR conversion complete\n");
    }

    if (triplets) free(triplets);

    printf("Sparse TSV weight matrix converted to CSR format: %lldx%lld with %lld NNZ\n",
           (long long)W->nrows, (long long)W->ncols, (long long)W->nnz);

    return W;
}

/* Main custom weight matrix reading function */
SparseMatrix* read_custom_weight_matrix(const char* filename, int format,
                                       char** spot_names_from_expr, MKL_INT n_spots) {
    if (!filename || !spot_names_from_expr || n_spots < 0) {
        fprintf(stderr, "Error: Invalid parameters in read_custom_weight_matrix\n");
        return NULL;
    }

    int actual_format = format;
    if (format == WEIGHT_FORMAT_AUTO) {
        actual_format = detect_weight_matrix_format(filename);
        if (actual_format == WEIGHT_FORMAT_AUTO) {
            fprintf(stderr, "Error: Could not auto-detect weight matrix format for '%s'\n", filename);
            return NULL;
        }
    }

    SpotNameHashTable* spot_map = spot_name_ht_create(n_spots > 0 ? (size_t)n_spots : 16);
    if (!spot_map) {
        fprintf(stderr, "Error: Failed to create spot name hash table.\n");
        return NULL;
    }

    int inserted_count = 0;
    for (MKL_INT i = 0; i < n_spots; i++) {
        if (spot_names_from_expr[i] && strlen(spot_names_from_expr[i]) > 0) {
            if (spot_name_ht_insert(spot_map, spot_names_from_expr[i], i) == 0) {
                inserted_count++;
            } else {
                fprintf(stderr, "Error: Failed to insert spot '%s' into hash table.\n", spot_names_from_expr[i]);
                spot_name_ht_free(spot_map);
                return NULL;
            }
        }
    }
    if (inserted_count == 0 && n_spots > 0) {
        fprintf(stderr, "Warning: No spot names from expression data were added to hash map for custom weights.\n");
        // Continue if n_spots is 0, otherwise this might be an issue.
    }
     if (n_spots > 0 && spot_map->count != (size_t)inserted_count) {
         fprintf(stderr, "Warning: Hash map count (%zu) differs from inserted count (%d).\n", spot_map->count, inserted_count);
     }


    SparseMatrix* W = NULL;
    switch (actual_format) {
        case WEIGHT_FORMAT_DENSE:
            printf("Reading custom weight matrix in DENSE format...\n");
            W = read_dense_weight_matrix(filename, spot_map, n_spots);
            break;
        case WEIGHT_FORMAT_SPARSE_COO:
            printf("Reading custom weight matrix in SPARSE_COO format...\n");
            W = read_sparse_weight_matrix_coo(filename, spot_map, n_spots);
            break;
        case WEIGHT_FORMAT_SPARSE_TSV:
            printf("Reading custom weight matrix in SPARSE_TSV format...\n");
            W = read_sparse_weight_matrix_tsv(filename, spot_map, n_spots);
            break;
        default:
            fprintf(stderr, "Error: Unsupported weight matrix format: %d\n", actual_format);
            spot_name_ht_free(spot_map);
            return NULL;
    }

    spot_name_ht_free(spot_map); // Free the hash table after use

    if (W) {
        int validation_result = validate_weight_matrix(W, spot_names_from_expr, n_spots);
        if (validation_result != MORANS_I_SUCCESS) {
            fprintf(stderr, "Error: Weight matrix validation failed\n");
            free_sparse_matrix(W);
            return NULL;
        }
        printf("Custom weight matrix successfully loaded and validated.\n");
    } else {
         fprintf(stderr, "Failed to load custom weight matrix in format %d.\n", actual_format);
    }

    return W;
}
