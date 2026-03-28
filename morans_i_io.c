/* morans_i_io.c - File I/O module for Moran's I implementation
 *
 * Contains all file reading/writing functions: VST parsing, weight matrix I/O,
 * cell type data processing, coordinate file reading, and result saving.
 *
 * Split from morans_i_mkl.c v1.3.0
 */

#include <unistd.h>
#include "morans_i_internal.h"

/* ===============================
 * CELL TYPE DATA PROCESSING
 * =============================== */

/* Detect file delimiter (CSV vs TSV) */
int detect_file_delimiter(const char* filename) {
    if (!filename) {
        fprintf(stderr, "Error: NULL filename in detect_file_delimiter\n");
        return '\t'; // Default to tab
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Warning: Cannot open file '%s' for delimiter detection: %s\n",
                filename, strerror(errno));
        return '\t'; // Default to tab
    }

    char* line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len = getline(&line, &line_buf_size, fp);
    fclose(fp);

    if (line_len <= 0) {
        if (line) free(line);
        return '\t'; // Default to tab
    }

    int comma_count = 0, tab_count = 0;
    for (ssize_t i = 0; i < line_len; i++) {
        if (line[i] == ',') comma_count++;
        else if (line[i] == '\t') tab_count++;
    }

    free(line);
    DEBUG_PRINT("Delimiter detection: commas=%d, tabs=%d", comma_count, tab_count);
    return (comma_count > tab_count) ? ',' : '\t';
}

/* Parse cell type header to find column indices */
static int parse_celltype_header(const char* header_line, char delimiter,
                                const char* expected_id_col,
                                const char* expected_type_col,
                                const char* expected_x_col,
                                const char* expected_y_col,
                                int* id_col_idx,
                                int* type_col_idx,
                                int* x_col_idx,
                                int* y_col_idx) {
    if (!header_line || !expected_id_col || !expected_type_col) {
        return MORANS_I_ERROR_PARAMETER;
    }

    *id_col_idx = *type_col_idx = *x_col_idx = *y_col_idx = -1;

    char* header_copy = strdup(header_line);
    if (!header_copy) {
        perror("strdup header in parse_celltype_header");
        return MORANS_I_ERROR_MEMORY;
    }

    char delim_str[2] = {delimiter, '\0'};
    char* token = strtok(header_copy, delim_str);
    int col_idx = 0;

    while (token) {
        char* trimmed = trim_whitespace_inplace(token);

        // Skip first column if it's an index column (empty or numeric-looking)
        if (col_idx == 0 && (strlen(trimmed) == 0 || isdigit((unsigned char)trimmed[0]))) {
            token = strtok(NULL, delim_str);
            col_idx++;
            continue;
        }

        if (strcmp(trimmed, expected_id_col) == 0) {
            *id_col_idx = col_idx;
        } else if (strcmp(trimmed, expected_type_col) == 0) {
            *type_col_idx = col_idx;
        } else if (expected_x_col && strcmp(trimmed, expected_x_col) == 0) {
            *x_col_idx = col_idx;
        } else if (expected_y_col && strcmp(trimmed, expected_y_col) == 0) {
            *y_col_idx = col_idx;
        }

        token = strtok(NULL, delim_str);
        col_idx++;
    }

    free(header_copy);

    DEBUG_PRINT("Column indices: ID=%d, Type=%d, X=%d, Y=%d",
                *id_col_idx, *type_col_idx, *x_col_idx, *y_col_idx);

    return MORANS_I_SUCCESS;
}

/* Collect unique cell types from file */
/*
 * FINAL CORRECTED VERSION of collect_unique_celltypes
 * This version is robust and does not modify the line buffer from getline.
 */
static char** collect_unique_celltypes(FILE* fp, char delimiter, int type_col_idx,
                                      MKL_INT* n_celltypes_out, MKL_INT* n_cells_out) {
    if (!fp || type_col_idx < 0 || !n_celltypes_out || !n_cells_out) {
        return NULL;
    }

    long original_pos = ftell(fp);
    if (fseek(fp, 0, SEEK_END) != 0) return NULL;
    //long file_size = ftell(fp);
    fseek(fp, original_pos, SEEK_SET); // Go back to where we were

    char* line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;

    char** temp_celltypes = NULL;
    MKL_INT temp_capacity = 100;
    MKL_INT temp_count = 0;
    MKL_INT cell_count = 0;

    temp_celltypes = (char**)malloc(temp_capacity * sizeof(char*));
    if (!temp_celltypes) {
        perror("malloc temp_celltypes");
        return NULL;
    }

    char field_buffer[BUFFER_SIZE]; // Temporary buffer for the field

    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        // Skip empty or whitespace-only lines
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p == '\0') continue;

        // Find the correct field without using strtok
        const char* start = line;
        int current_col = 0;

        // Find the start of the target column
        while (current_col < type_col_idx && *start) {
            if (*start == delimiter) {
                current_col++;
            }
            start++;
        }

        if (current_col == type_col_idx) {
            // Find the end of the field
            const char* end = start;
            while (*end && *end != delimiter && *end != '\n' && *end != '\r') {
                end++;
            }

            // Copy the field into our buffer
            size_t field_len = end - start;
            if (field_len < BUFFER_SIZE) {
                strncpy(field_buffer, start, field_len);
                field_buffer[field_len] = '\0';

                char* celltype = trim_whitespace_inplace(field_buffer);
                if (strlen(celltype) > 0) {
                    cell_count++;

                    // Check if this cell type is already in our list
                    int found = 0;
                    for (MKL_INT i = 0; i < temp_count; i++) {
                        if (strcmp(temp_celltypes[i], celltype) == 0) {
                            found = 1;
                            break;
                        }
                    }

                    if (!found) {
                        if (temp_count >= temp_capacity) {
                            temp_capacity *= 2;
                            char** new_temp = (char**)realloc(temp_celltypes, temp_capacity * sizeof(char*));
                            if (!new_temp) {
                                // Handle realloc failure
                                for(MKL_INT i=0; i<temp_count; ++i) free(temp_celltypes[i]);
                                free(temp_celltypes);
                                free(line);
                                fseek(fp, original_pos, SEEK_SET);
                                return NULL;
                            }
                            temp_celltypes = new_temp;
                        }
                        temp_celltypes[temp_count] = strdup(celltype);
                        if (!temp_celltypes[temp_count]) {
                            // Handle strdup failure
                             for(MKL_INT i=0; i<temp_count; ++i) free(temp_celltypes[i]);
                             free(temp_celltypes);
                             free(line);
                             fseek(fp, original_pos, SEEK_SET);
                             return NULL;
                        }
                        temp_count++;
                    }
                }
            }
        }
    }

    free(line);
    fseek(fp, original_pos, SEEK_SET);

    *n_celltypes_out = temp_count;
    *n_cells_out = cell_count;

    printf("Found %lld unique cell types from %lld cells\n", (long long)temp_count, (long long)cell_count);

    return temp_celltypes;
}

/* Read cell type data in single-cell format */
CellTypeMatrix* read_celltype_singlecell_file(const char* filename,
                                             const char* cell_id_col,
                                             const char* celltype_col,
                                             const char* x_col,
                                             const char* y_col) {
    if (!filename || !cell_id_col || !celltype_col) {
        fprintf(stderr, "Error: NULL parameters in read_celltype_singlecell_file\n");
        return NULL;
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open cell type file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    char delimiter = detect_file_delimiter(filename);
    printf("Reading single-cell annotations from '%s' with delimiter '%c'\n", filename, delimiter);

    char* line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;

    // Read and parse header
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty header in cell type file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    int id_col_idx, type_col_idx, x_col_idx, y_col_idx;
    if (parse_celltype_header(line, delimiter, cell_id_col, celltype_col, x_col, y_col,
                             &id_col_idx, &type_col_idx, &x_col_idx, &y_col_idx) != MORANS_I_SUCCESS) {
        fprintf(stderr, "Error: Failed to parse cell type file header\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    if (id_col_idx < 0 || type_col_idx < 0) {
        fprintf(stderr, "Error: Required columns not found: %s=%s, %s=%s\n",
                cell_id_col, (id_col_idx >= 0 ? "Found" : "Missing"),
                celltype_col, (type_col_idx >= 0 ? "Found" : "Missing"));
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Collect unique cell types
    MKL_INT n_celltypes, n_cells;
    char** unique_celltypes = collect_unique_celltypes(fp, delimiter, type_col_idx, &n_celltypes, &n_cells);
    if (!unique_celltypes || n_celltypes == 0) {
        fprintf(stderr, "Error: No cell types found in file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Create cell type matrix
    CellTypeMatrix* celltype_matrix = (CellTypeMatrix*)malloc(sizeof(CellTypeMatrix));
    if (!celltype_matrix) {
        perror("malloc CellTypeMatrix");
        for (MKL_INT i = 0; i < n_celltypes; i++) {
            free(unique_celltypes[i]);
        }
        free(unique_celltypes);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    celltype_matrix->nrows = n_cells;
    celltype_matrix->ncols = n_celltypes;
    celltype_matrix->is_binary = 1;
    celltype_matrix->format_type = CELLTYPE_FORMAT_SINGLE_CELL;

    size_t values_size;
    if (safe_multiply_size_t(n_cells, n_celltypes, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Cell type matrix dimensions too large\n");
        free(celltype_matrix);
        for (MKL_INT i = 0; i < n_celltypes; i++) {
            free(unique_celltypes[i]);
        }
        free(unique_celltypes);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    celltype_matrix->values = (double*)mkl_calloc(values_size / sizeof(double), sizeof(double), 64);
    celltype_matrix->rownames = (char**)calloc(n_cells, sizeof(char*));
    celltype_matrix->colnames = (char**)calloc(n_celltypes, sizeof(char*));

    if (!celltype_matrix->values || !celltype_matrix->rownames || !celltype_matrix->colnames) {
        perror("Failed to allocate cell type matrix components");
        free_celltype_matrix(celltype_matrix);
        for (MKL_INT i = 0; i < n_celltypes; i++) {
            free(unique_celltypes[i]);
        }
        free(unique_celltypes);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Copy cell type names
    for (MKL_INT i = 0; i < n_celltypes; i++) {
        celltype_matrix->colnames[i] = strdup(unique_celltypes[i]);
        if (!celltype_matrix->colnames[i]) {
            perror("strdup cell type name");
            free_celltype_matrix(celltype_matrix);
            for (MKL_INT j = 0; j < n_celltypes; j++) {
                free(unique_celltypes[j]);
            }
            free(unique_celltypes);
            fclose(fp);
            if (line) free(line);
            return NULL;
        }
    }

    // Read data and populate matrix
    MKL_INT cell_idx = 0;
    char delim_str[2] = {delimiter, '\0'};

    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && cell_idx < n_cells) {
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p == '\0') continue;

        char* line_copy = strdup(line);
        if (!line_copy) continue;

        char* token = strtok(line_copy, delim_str);
        int col_idx = 0;
        char* cell_id = NULL;
        char* cell_type = NULL;

        while (token) {
            if (col_idx == id_col_idx) {
                cell_id = trim_whitespace_inplace(token);
            } else if (col_idx == type_col_idx) {
                cell_type = trim_whitespace_inplace(token);
            }
            token = strtok(NULL, delim_str);
            col_idx++;
        }

        if (cell_id && cell_type && strlen(cell_id) > 0 && strlen(cell_type) > 0) {
            // Store cell ID
            celltype_matrix->rownames[cell_idx] = strdup(cell_id);
            if (!celltype_matrix->rownames[cell_idx]) {
                perror("strdup cell_id");
                free(line_copy);
                free_celltype_matrix(celltype_matrix);
                for (MKL_INT j = 0; j < n_celltypes; j++) {
                    free(unique_celltypes[j]);
                }
                free(unique_celltypes);
                fclose(fp);
                if (line) free(line);
                return NULL;
            }

            // Find cell type index and set binary indicator
            for (MKL_INT ct_idx = 0; ct_idx < n_celltypes; ct_idx++) {
                if (strcmp(cell_type, unique_celltypes[ct_idx]) == 0) {
                    celltype_matrix->values[cell_idx * n_celltypes + ct_idx] = 1.0;
                    break;
                }
            }
            cell_idx++;
        }
        free(line_copy);
    }

    // Cleanup
    for (MKL_INT i = 0; i < n_celltypes; i++) {
        free(unique_celltypes[i]);
    }
    free(unique_celltypes);
    fclose(fp);
    if (line) free(line);

    if (cell_idx != n_cells) {
        fprintf(stderr, "Warning: Expected %lld cells but processed %lld\n",
                (long long)n_cells, (long long)cell_idx);
        celltype_matrix->nrows = cell_idx;
    }

    printf("Successfully loaded single-cell annotations: %lld cells x %lld cell types\n",
           (long long)celltype_matrix->nrows, (long long)celltype_matrix->ncols);

    return celltype_matrix;
}


/* Read cell type data in deconvolution format */
CellTypeMatrix* read_celltype_deconvolution_file(const char* filename, const char* spot_id_col) {
    if (!filename || !spot_id_col) {
        fprintf(stderr, "Error: NULL parameters in read_celltype_deconvolution_file\n");
        return NULL;
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open deconvolution file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    char delimiter = detect_file_delimiter(filename);
    printf("Reading deconvolution data from '%s' with delimiter '%c'\n", filename, delimiter);

    char* line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;

    // Read and parse header
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty header in deconvolution file\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Trim line
    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
        line[--line_len] = '\0';
    }

    char* header_copy = strdup(line);
    if (!header_copy) {
        perror("strdup header");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Count fields and find spot ID column
    char delim_str[2] = {delimiter, '\0'};
    char* token = strtok(header_copy, delim_str);
    int field_count = 0;
    int spot_id_col_idx = -1;

    while (token) {
        char* trimmed = trim_whitespace_inplace(token);
        if (strcmp(trimmed, spot_id_col) == 0) {
            spot_id_col_idx = field_count;
        }
        field_count++;
        token = strtok(NULL, delim_str);
    }
    free(header_copy);

    if (spot_id_col_idx < 0) {
        fprintf(stderr, "Error: Spot ID column '%s' not found in header\n", spot_id_col);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    MKL_INT n_celltypes = field_count - 1; // Subtract spot ID column
    if (n_celltypes <= 0) {
        fprintf(stderr, "Error: No cell type columns found\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Count data rows
    long current_pos = ftell(fp);
    MKL_INT n_spots = 0;
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p != '\0') n_spots++;
    }
    fseek(fp, current_pos, SEEK_SET);

    if (n_spots == 0) {
        fprintf(stderr, "Error: No data rows found\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Create cell type matrix
    CellTypeMatrix* celltype_matrix = (CellTypeMatrix*)malloc(sizeof(CellTypeMatrix));
    if (!celltype_matrix) {
        perror("malloc CellTypeMatrix");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    celltype_matrix->nrows = n_spots;
    celltype_matrix->ncols = n_celltypes;
    celltype_matrix->is_binary = 0;
    celltype_matrix->format_type = CELLTYPE_FORMAT_DECONVOLUTION;

    size_t values_size;
    if (safe_multiply_size_t(n_spots, n_celltypes, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Deconvolution matrix dimensions too large\n");
        free(celltype_matrix);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    celltype_matrix->values = (double*)mkl_malloc(values_size, 64);
    celltype_matrix->rownames = (char**)calloc(n_spots, sizeof(char*));
    celltype_matrix->colnames = (char**)calloc(n_celltypes, sizeof(char*));

    if (!celltype_matrix->values || !celltype_matrix->rownames || !celltype_matrix->colnames) {
        perror("Failed to allocate deconvolution matrix components");
        free_celltype_matrix(celltype_matrix);
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    // Read header again to get column names
    fseek(fp, 0, SEEK_SET);
    getline(&line, &line_buf_size, fp); // Re-read header
    header_copy = strdup(line);
    token = strtok(header_copy, delim_str);
    int col_idx = 0;
    int celltype_col_idx = 0;

    while (token) {
        char* trimmed = trim_whitespace_inplace(token);
        if (col_idx != spot_id_col_idx) {
            celltype_matrix->colnames[celltype_col_idx] = strdup(trimmed);
            if (!celltype_matrix->colnames[celltype_col_idx]) {
                perror("strdup cell type column name");
                free(header_copy);
                free_celltype_matrix(celltype_matrix);
                fclose(fp);
                if (line) free(line);
                return NULL;
            }
            celltype_col_idx++;
        }
        col_idx++;
        token = strtok(NULL, delim_str);
    }
    free(header_copy);

    // Read data rows
    MKL_INT spot_idx = 0;
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && spot_idx < n_spots) {
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p == '\0') continue;

        char* line_copy = strdup(line);
        if (!line_copy) continue;

        token = strtok(line_copy, delim_str);
        col_idx = 0;
        celltype_col_idx = 0;
        char* spot_id = NULL;

        while (token) {
            char* trimmed = trim_whitespace_inplace(token);
            if (col_idx == spot_id_col_idx) {
                spot_id = trimmed;
            } else {
                char* endptr;
                double value = strtod(trimmed, &endptr);
                if (endptr != trimmed && isfinite(value)) {
                    celltype_matrix->values[spot_idx * n_celltypes + celltype_col_idx] = value;
                } else {
                    celltype_matrix->values[spot_idx * n_celltypes + celltype_col_idx] = 0.0;
                }
                celltype_col_idx++;
            }
            col_idx++;
            token = strtok(NULL, delim_str);
        }

        if (spot_id && strlen(spot_id) > 0) {
            celltype_matrix->rownames[spot_idx] = strdup(spot_id);
            if (!celltype_matrix->rownames[spot_idx]) {
                perror("strdup spot_id");
                free(line_copy);
                free_celltype_matrix(celltype_matrix);
                fclose(fp);
                if (line) free(line);
                return NULL;
            }
            spot_idx++;
        }
        free(line_copy);
    }

    fclose(fp);
    if (line) free(line);

    if (spot_idx != n_spots) {
        fprintf(stderr, "Warning: Expected %lld spots but processed %lld\n",
                (long long)n_spots, (long long)spot_idx);
        celltype_matrix->nrows = spot_idx;
    }

    printf("Successfully loaded deconvolution data: %lld spots x %lld cell types\n",
           (long long)celltype_matrix->nrows, (long long)celltype_matrix->ncols);

    return celltype_matrix;
}

/* Validate cell type matrix against expression matrix */
int validate_celltype_matrix(const CellTypeMatrix* Z, const DenseMatrix* X) {
    if (!Z || !X) {
        fprintf(stderr, "Error: NULL parameters in validate_celltype_matrix\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (Z->nrows == 0 || Z->ncols == 0) {
        fprintf(stderr, "Error: Cell type matrix has zero dimensions\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    printf("Validating cell type matrix (%lld x %lld) against expression matrix (%lld x %lld)\n",
           (long long)Z->nrows, (long long)Z->ncols, (long long)X->nrows, (long long)X->ncols);

    return MORANS_I_SUCCESS;
}

/* Map cell type matrix to expression matrix spots */
int map_celltype_to_expression(const CellTypeMatrix* celltype_matrix, const DenseMatrix* expr_matrix,
                               CellTypeMatrix** mapped_celltype_out) {
    if (!celltype_matrix || !expr_matrix || !mapped_celltype_out) {
        fprintf(stderr, "Error: NULL parameters in map_celltype_to_expression\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    *mapped_celltype_out = NULL;

    if (celltype_matrix->format_type == CELLTYPE_FORMAT_SINGLE_CELL) {
        // For single-cell data, need to map cell IDs to expression spots
        printf("Mapping single-cell annotations to expression matrix spots...\n");

        // Create hash table for fast cell ID lookup
        SpotNameHashTable* cell_map = spot_name_ht_create(celltype_matrix->nrows);
        if (!cell_map) {
            fprintf(stderr, "Error: Failed to create cell ID hash table\n");
            return MORANS_I_ERROR_MEMORY;
        }

        for (MKL_INT i = 0; i < celltype_matrix->nrows; i++) {
            if (celltype_matrix->rownames[i]) {
                if (spot_name_ht_insert(cell_map, celltype_matrix->rownames[i], i) != 0) {
                    fprintf(stderr, "Error: Failed to insert cell ID into hash table\n");
                    spot_name_ht_free(cell_map);
                    return MORANS_I_ERROR_MEMORY;
                }
            }
        }

        // Create new mapped cell type matrix
        CellTypeMatrix* mapped = (CellTypeMatrix*)malloc(sizeof(CellTypeMatrix));
        if (!mapped) {
            perror("malloc mapped CellTypeMatrix");
            spot_name_ht_free(cell_map);
            return MORANS_I_ERROR_MEMORY;
        }

        mapped->nrows = expr_matrix->nrows;  // Use expression matrix spot count
        mapped->ncols = celltype_matrix->ncols;
        mapped->is_binary = celltype_matrix->is_binary;
        mapped->format_type = celltype_matrix->format_type;

        size_t mapped_values_size;
        if (safe_multiply_size_t(mapped->nrows, mapped->ncols, &mapped_values_size) != 0 ||
            safe_multiply_size_t(mapped_values_size, sizeof(double), &mapped_values_size) != 0) {
            fprintf(stderr, "Error: Mapped matrix dimensions too large\n");
            free(mapped);
            spot_name_ht_free(cell_map);
            return MORANS_I_ERROR_MEMORY;
        }

        mapped->values = (double*)mkl_calloc(mapped_values_size / sizeof(double), sizeof(double), 64);
        mapped->rownames = (char**)calloc(mapped->nrows, sizeof(char*));
        mapped->colnames = (char**)calloc(mapped->ncols, sizeof(char*));

        if (!mapped->values || !mapped->rownames || !mapped->colnames) {
            perror("Failed to allocate mapped matrix components");
            free_celltype_matrix(mapped);
            spot_name_ht_free(cell_map);
            return MORANS_I_ERROR_MEMORY;
        }

        // Copy cell type names
        for (MKL_INT j = 0; j < mapped->ncols; j++) {
            mapped->colnames[j] = strdup(celltype_matrix->colnames[j]);
            if (!mapped->colnames[j]) {
                perror("strdup cell type name");
                free_celltype_matrix(mapped);
                spot_name_ht_free(cell_map);
                return MORANS_I_ERROR_MEMORY;
            }
        }

        // Map cell type data to expression spots
        MKL_INT mapped_count = 0;
        for (MKL_INT i = 0; i < expr_matrix->nrows; i++) {
            const char* spot_name = expr_matrix->rownames[i];
            if (spot_name) {
                mapped->rownames[i] = strdup(spot_name);
                if (!mapped->rownames[i]) {
                    perror("strdup spot name");
                    free_celltype_matrix(mapped);
                    spot_name_ht_free(cell_map);
                    return MORANS_I_ERROR_MEMORY;
                }

                MKL_INT cell_idx = spot_name_ht_find(cell_map, spot_name);
                if (cell_idx >= 0) {
                    // Copy cell type data
                    for (MKL_INT j = 0; j < mapped->ncols; j++) {
                        mapped->values[i * mapped->ncols + j] =
                            celltype_matrix->values[cell_idx * celltype_matrix->ncols + j];
                    }
                    mapped_count++;
                }
                // If not found, values remain 0 (from calloc)
            }
        }

        spot_name_ht_free(cell_map);

        printf("Mapped %lld/%lld expression spots to cell type annotations\n",
               (long long)mapped_count, (long long)expr_matrix->nrows);

        if (mapped_count == 0) {
            fprintf(stderr, "Warning: No expression spots could be mapped to cell type data\n");
        }

        *mapped_celltype_out = mapped;

    } else {
        // For deconvolution data, assume spot names match directly
        printf("Using deconvolution data directly (assuming spot names match)\n");

        // For now, just return a reference to the original matrix
        // In a full implementation, you might want to reorder spots to match expression matrix
        *mapped_celltype_out = (CellTypeMatrix*)celltype_matrix; // Cast away const - be careful!
    }

    return MORANS_I_SUCCESS;
}

/* ===============================
 * UPDATED FILE I/O FUNCTIONS
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


/* ===============================
 * VST FILE PARSING (REFACTORED)
 * =============================== */
/* ========================================
   PART 1: UPDATED FUNCTION DECLARATIONS
   Replace the existing declarations around lines 63-68
   ======================================== */

// Replace the old declarations with these:
static int detect_file_format(const char* filename);
static int parse_vst_header(FILE* fp, char** line, size_t* line_buf_size,
                           MKL_INT* n_spots_out, char*** colnames_out, int is_csv);
static int count_vst_genes(FILE* fp, char** line, size_t* line_buf_size, MKL_INT* n_genes_out);
static int read_vst_data_rows(FILE* fp, char** line, size_t* line_buf_size,
                             DenseMatrix* matrix, MKL_INT n_genes_expected,
                             MKL_INT n_spots_expected, int is_csv);

/* ========================================
   PART 2: COMPLETE FUNCTION IMPLEMENTATIONS
   Replace your existing implementations with these:
   ======================================== */

/* Helper function to detect file format */
static int detect_file_format(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return -1;

    char buffer[1024];
    if (!fgets(buffer, sizeof(buffer), fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);

    // Count commas and tabs in first line
    int comma_count = 0, tab_count = 0;
    for (char* p = buffer; *p; p++) {
        if (*p == ',') comma_count++;
        else if (*p == '\t') tab_count++;
    }

    // If more commas than tabs, assume CSV
    return (comma_count > tab_count) ? 0 : 1; // 0=CSV, 1=TSV
}

/* Parse VST file header and extract column names - UPDATED VERSION */
static int parse_vst_header(FILE* fp, char** line, size_t* line_buf_size,
                           MKL_INT* n_spots_out, char*** colnames_out, int is_csv) {
    if (!fp || !line || !line_buf_size || !n_spots_out || !colnames_out) {
        return MORANS_I_ERROR_PARAMETER;
    }

    *n_spots_out = 0;
    *colnames_out = NULL;

    ssize_t line_len = getline(line, line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty or unreadable header in VST file.\n");
        return MORANS_I_ERROR_FILE;
    }

    // Trim line
    while (line_len > 0 && ((*line)[line_len - 1] == '\n' || (*line)[line_len - 1] == '\r')) {
        (*line)[--line_len] = '\0';
    }

    // Use correct delimiter based on format
    char* delimiter = is_csv ? "," : "\t";

    // Count total fields first
    char* header_copy = strdup(*line);
    if (!header_copy) {
        perror("strdup for header counting");
        return MORANS_I_ERROR_MEMORY;
    }

    MKL_INT total_field_count = 0;
    char* token = strtok(header_copy, delimiter);
    while (token != NULL) {
        total_field_count++;
        token = strtok(NULL, delimiter);
    }
    free(header_copy);

    if (total_field_count < 1) {
        fprintf(stderr, "Error: Header has %lld fields. Expected at least 1.\n", (long long)total_field_count);
        return MORANS_I_ERROR_FILE;
    }

    printf("Detected %lld columns in %s format\n", (long long)total_field_count,
           is_csv ? "CSV" : "TSV");

    // Allocate temporary array for all fields
    char** temp_colnames = (char**)malloc((size_t)total_field_count * sizeof(char*));
    if (!temp_colnames) {
        perror("malloc for temp colnames");
        return MORANS_I_ERROR_MEMORY;
    }

    // Initialize all to NULL for safe cleanup
    for (MKL_INT i = 0; i < total_field_count; i++) {
        temp_colnames[i] = NULL;
    }

    // Parse all column names, counting valid (non-empty) ones
    char* header_copy2 = strdup(*line);
    if (!header_copy2) {
        perror("strdup for header parsing");
        free(temp_colnames);
        return MORANS_I_ERROR_MEMORY;
    }

    token = strtok(header_copy2, delimiter);
    MKL_INT col_idx = 0;
    MKL_INT valid_cols = 0;

    while (token && col_idx < total_field_count) {
        char* trimmed = trim_whitespace_inplace(token);

        if (strlen(trimmed) > 0) {
            // Valid non-empty column name
            temp_colnames[valid_cols] = strdup(trimmed);
            if (!temp_colnames[valid_cols]) {
                perror("strdup for column name");
                free(header_copy2);
                // Cleanup already allocated names
                for (MKL_INT i = 0; i < valid_cols; i++) {
                    free(temp_colnames[i]);
                }
                free(temp_colnames);
                return MORANS_I_ERROR_MEMORY;
            }
            valid_cols++;
        }
        // Skip empty fields (like the first empty field in your CSV)

        col_idx++;
        token = strtok(NULL, delimiter);
    }

    free(header_copy2);

    if (valid_cols == 0) {
        fprintf(stderr, "Error: No valid column names found\n");
        free(temp_colnames);
        return MORANS_I_ERROR_FILE;
    }

    // Allocate final array with correct size
    *colnames_out = (char**)malloc((size_t)valid_cols * sizeof(char*));
    if (!*colnames_out) {
        perror("malloc for final colnames");
        for (MKL_INT i = 0; i < valid_cols; i++) {
            free(temp_colnames[i]);
        }
        free(temp_colnames);
        return MORANS_I_ERROR_MEMORY;
    }

    // Copy valid column names to final array
    for (MKL_INT i = 0; i < valid_cols; i++) {
        (*colnames_out)[i] = temp_colnames[i]; // Transfer ownership
    }
    free(temp_colnames);

    *n_spots_out = valid_cols;

    printf("Parsed %lld valid column names\n", (long long)*n_spots_out);
    DEBUG_PRINT("Parsed VST header: %lld columns", (long long)*n_spots_out);
    return MORANS_I_SUCCESS;
}

/* Count genes in VST file */
static int count_vst_genes(FILE* fp, char** line, size_t* line_buf_size, MKL_INT* n_genes_out) {
    if (!fp || !line || !line_buf_size || !n_genes_out) {
        return MORANS_I_ERROR_PARAMETER;
    }

    *n_genes_out = 0;
    long current_pos = ftell(fp);

    ssize_t line_len;
    while ((line_len = getline(line, line_buf_size, fp)) > 0) {
        char* p = *line;
        while(isspace((unsigned char)*p)) p++;
        if(*p != '\0') (*n_genes_out)++;
    }

    fseek(fp, current_pos, SEEK_SET);
    DEBUG_PRINT("Counted VST genes: %lld", (long long)*n_genes_out);
    return MORANS_I_SUCCESS;
}

/* Read VST data rows - UPDATED VERSION */
static int read_vst_data_rows(FILE* fp, char** line, size_t* line_buf_size,
                             DenseMatrix* matrix, MKL_INT n_genes_expected,
                             MKL_INT n_spots_expected, int is_csv) {
    if (!fp || !line || !line_buf_size || !matrix || !matrix->values || !matrix->rownames) {
        return MORANS_I_ERROR_PARAMETER;
    }

    // Use correct delimiter based on format
    char* delimiter = is_csv ? "," : "\t";

    MKL_INT gene_idx = 0;
    int file_lineno = 1; // Header was line 1
    ssize_t line_len;

    while ((line_len = getline(line, line_buf_size, fp)) > 0 && gene_idx < n_genes_expected) {
        file_lineno++;

        // Skip blank lines
        char* p = *line;
        while(isspace((unsigned char)*p)) p++;
        if(*p == '\0') continue;

        if (gene_idx >= n_genes_expected) {
            fprintf(stderr, "Warning: More data rows than estimated. Stopping at gene %lld (file line %d).\n",
                   (long long)n_genes_expected, file_lineno);
            break;
        }

        // Trim EOL
        while (line_len > 0 && ((*line)[line_len - 1] == '\n' || (*line)[line_len - 1] == '\r')) {
            (*line)[--line_len] = '\0';
        }
        if(line_len == 0) continue;

        char* data_row_copy = strdup(*line);
        if (!data_row_copy) {
            perror("strdup data line");
            return MORANS_I_ERROR_MEMORY;
        }

        char* token = strtok(data_row_copy, delimiter);
        if (!token) {
            fprintf(stderr, "Error: No gene name found on line %d\n", file_lineno);
            free(data_row_copy);
            return MORANS_I_ERROR_FILE;
        }

        // Store gene name (first token should be gene name)
        char* gene_name = trim_whitespace_inplace(token);
        matrix->rownames[gene_idx] = strdup(gene_name);
        if (!matrix->rownames[gene_idx]) {
            perror("strdup gene name");
            free(data_row_copy);
            return MORANS_I_ERROR_MEMORY;
        }

        // Read expression values, skipping empty fields
        MKL_INT values_read = 0;
        token = strtok(NULL, delimiter);

        while (token && values_read < n_spots_expected) {
            char* trimmed = trim_whitespace_inplace(token);

            // Skip empty values (like empty first field)
            if (strlen(trimmed) == 0) {
                token = strtok(NULL, delimiter);
                continue;
            }

            char* endptr;
            errno = 0;
            double val = strtod(trimmed, &endptr);
            if (errno == ERANGE || (*endptr != '\0' && !isspace((unsigned char)*endptr)) || endptr == trimmed) {
                fprintf(stderr, "Error: Invalid number '%s' at file line %d, gene '%s', value %lld.\n",
                        trimmed, file_lineno, matrix->rownames[gene_idx], (long long)values_read + 1);
                free(data_row_copy);
                return MORANS_I_ERROR_FILE;
            }

            matrix->values[gene_idx * n_spots_expected + values_read] = val;
            values_read++;
            token = strtok(NULL, delimiter);
        }

        if (values_read != n_spots_expected) {
            fprintf(stderr, "Error: File line %d, gene '%s': Expected %lld expression values, found %lld.\n",
                    file_lineno, matrix->rownames[gene_idx], (long long)n_spots_expected, (long long)values_read);
            free(data_row_copy);
            return MORANS_I_ERROR_FILE;
        }

        free(data_row_copy);
        gene_idx++;
    }

    if (gene_idx != n_genes_expected) {
        fprintf(stderr, "Warning: Read %lld genes, but estimated %lld. Adjusting matrix dimensions.\n",
                (long long)gene_idx, (long long)n_genes_expected);
        matrix->nrows = gene_idx;
    }

    DEBUG_PRINT("Read VST data: %lld genes processed", (long long)gene_idx);
    return MORANS_I_SUCCESS;
}

/* Enhanced VST file reader with CSV/TSV auto-detection - UPDATED VERSION */
DenseMatrix* read_vst_file(const char* filename) {
    if (!filename) {
        fprintf(stderr, "Error: Null filename provided to read_vst_file\n");
        return NULL;
    }

    // Detect file format
    int format = detect_file_format(filename);
    if (format < 0) {
        fprintf(stderr, "Error: Failed to detect file format for '%s'\n", filename);
        return NULL;
    }

    int is_csv = (format == 0);
    printf("Detected %s format for file '%s'\n", is_csv ? "CSV" : "TSV", filename);

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    char *line = NULL;
    size_t line_buf_size = 0;
    DenseMatrix* matrix = NULL;
    char** colnames = NULL;

    printf("Reading expression file '%s'...\n", filename);

    // Parse header with format detection
    MKL_INT n_spots;
    int header_result = parse_vst_header(fp, &line, &line_buf_size, &n_spots, &colnames, is_csv);
    if (header_result != MORANS_I_SUCCESS) {
        goto cleanup_and_exit;
    }

    // Count genes by reading through file
    long header_pos = ftell(fp);
    MKL_INT n_genes;
    if (count_vst_genes(fp, &line, &line_buf_size, &n_genes) != MORANS_I_SUCCESS) {
        goto cleanup_and_exit;
    }

    if (n_genes == 0 || n_spots == 0) {
        fprintf(stderr, "Error: No data found (genes=%lld, spots=%lld).\n",
                (long long)n_genes, (long long)n_spots);
        goto cleanup_and_exit;
    }

    printf("Matrix dimensions: %lld genes x %lld spots\n",
           (long long)n_genes, (long long)n_spots);

    // Allocate matrix
    matrix = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!matrix) {
        perror("malloc DenseMatrix");
        goto cleanup_and_exit;
    }

    matrix->nrows = n_genes;
    matrix->ncols = n_spots;
    matrix->rownames = NULL;
    matrix->colnames = NULL;
    matrix->values = NULL;

    // Allocate components
    size_t values_size = (size_t)n_genes * n_spots * sizeof(double);
    matrix->values = (double*)mkl_malloc(values_size, 64);
    matrix->rownames = (char**)calloc(n_genes, sizeof(char*));
    matrix->colnames = colnames; // Transfer ownership
    colnames = NULL; // Prevent double-free

    if (!matrix->values || !matrix->rownames || !matrix->colnames) {
        perror("Failed to allocate matrix components");
        if (matrix) {
            if (matrix->values) mkl_free(matrix->values);
            free(matrix->rownames);
            free(matrix);
        }
        matrix = NULL;
        goto cleanup_and_exit;
    }

    // Read data rows with format detection
    fseek(fp, header_pos, SEEK_SET);
    int data_result = read_vst_data_rows(fp, &line, &line_buf_size, matrix, n_genes, n_spots, is_csv);
    if (data_result != MORANS_I_SUCCESS) {
        free_dense_matrix(matrix);
        matrix = NULL;
        goto cleanup_and_exit;
    }

    printf("Successfully loaded expression data: %lld genes x %lld spots from '%s'.\n",
           (long long)matrix->nrows, (long long)matrix->ncols, filename);

cleanup_and_exit:
    if (line) free(line);
    if (colnames) {
        for (MKL_INT i = 0; i < n_spots; i++) {
            free(colnames[i]);
        }
        free(colnames);
    }
    fclose(fp);

    return matrix;
}

/* ===============================
 * FILE I/O AND SAVING FUNCTIONS
 * =============================== */

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

/* ===============================
 * COORDINATE MAPPING FUNCTION
 * =============================== */

/* Map expression matrix spots to coordinate data */
int map_expression_to_coordinates(const DenseMatrix* expr_matrix, const SpotCoordinates* coords,
                                 MKL_INT** mapping_out, MKL_INT* num_mapped_spots_out) {
    if (!expr_matrix || !coords || !mapping_out || !num_mapped_spots_out) {
        fprintf(stderr, "Error: NULL parameters in map_expression_to_coordinates\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (!expr_matrix->colnames) {
        fprintf(stderr, "Error: Expression matrix has no column names for mapping\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (!coords->spot_names || !coords->valid_mask) {
        fprintf(stderr, "Error: Coordinate data is missing spot names or validity mask\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    MKL_INT n_expr_spots = expr_matrix->ncols;
    *mapping_out = NULL;
    *num_mapped_spots_out = 0;

    if (n_expr_spots == 0) {
        printf("Warning: No spots in expression matrix to map\n");
        return MORANS_I_SUCCESS;
    }

    if (coords->total_spots == 0) {
        printf("Warning: No coordinate data available for mapping\n");
        return MORANS_I_SUCCESS;
    }

    printf("Mapping %lld expression spots to %lld coordinate entries (%lld valid)...\n",
           (long long)n_expr_spots, (long long)coords->total_spots, (long long)coords->valid_spots);

    // Allocate mapping array
    MKL_INT* mapping = (MKL_INT*)malloc((size_t)n_expr_spots * sizeof(MKL_INT));
    if (!mapping) {
        perror("Failed to allocate mapping array");
        return MORANS_I_ERROR_MEMORY;
    }

    // Initialize mapping array to -1 (not found)
    for (MKL_INT i = 0; i < n_expr_spots; i++) {
        mapping[i] = -1;
    }

    MKL_INT mapped_count = 0;
    MKL_INT valid_mapped_count = 0;

    // Create a hash table for faster coordinate lookup (simple linear search for now)
    // For each expression spot, find corresponding coordinate
    for (MKL_INT expr_idx = 0; expr_idx < n_expr_spots; expr_idx++) {
        const char* expr_spot_name = expr_matrix->colnames[expr_idx];

        if (!expr_spot_name || strlen(expr_spot_name) == 0) {
            continue; // Skip empty spot names
        }

        // Search for matching spot name in coordinates
        MKL_INT coord_idx = -1;
        for (MKL_INT c = 0; c < coords->total_spots; c++) {
            if (coords->spot_names[c] && strcmp(expr_spot_name, coords->spot_names[c]) == 0) {
                coord_idx = c;
                break;
            }
        }

        if (coord_idx >= 0) {
            mapping[expr_idx] = coord_idx;
            mapped_count++;

            // Check if this coordinate is valid
            if (coords->valid_mask[coord_idx]) {
                valid_mapped_count++;
            }
        }
    }

    printf("Mapping complete: %lld/%lld expression spots mapped to coordinates\n",
           (long long)mapped_count, (long long)n_expr_spots);
    printf("  %lld mapped spots have valid coordinates\n", (long long)valid_mapped_count);

    if (mapped_count == 0) {
        fprintf(stderr, "Warning: No expression spots could be mapped to coordinates\n");
        free(mapping);
        return MORANS_I_SUCCESS; // Not an error, just no matches
    }

    if (valid_mapped_count == 0) {
        fprintf(stderr, "Warning: No mapped spots have valid coordinates\n");
        free(mapping);
        return MORANS_I_SUCCESS; // Not an error, just no valid coordinates
    }

    // Provide some statistics about unmapped spots
    if (mapped_count < n_expr_spots) {
        MKL_INT unmapped_count = n_expr_spots - mapped_count;
        printf("Warning: %lld expression spots could not be mapped to coordinates\n",
               (long long)unmapped_count);

        // Show a few examples of unmapped spots for debugging
        MKL_INT examples_shown = 0;
        const MKL_INT max_examples = 5;

        for (MKL_INT i = 0; i < n_expr_spots && examples_shown < max_examples; i++) {
            if (mapping[i] == -1 && expr_matrix->colnames[i]) {
                if (examples_shown == 0) {
                    printf("  Examples of unmapped spots: ");
                }
                printf("%s%s", expr_matrix->colnames[i],
                       (examples_shown < max_examples - 1 && examples_shown < unmapped_count - 1) ? ", " : "");
                examples_shown++;
            }
        }
        if (examples_shown > 0) {
            printf("%s\n", (unmapped_count > max_examples) ? "..." : "");
        }
    }

    *mapping_out = mapping;
    *num_mapped_spots_out = mapped_count;

    return MORANS_I_SUCCESS;
}

/* Read coordinates from CSV/TSV file for single-cell data */
SpotCoordinates* read_coordinates_file(const char* filename, const char* id_column_name,
                                     const char* x_column_name, const char* y_column_name,
                                     double coord_scale) {
    if (!filename || !id_column_name || !x_column_name || !y_column_name) {
        fprintf(stderr, "Error: Null parameter(s) provided to read_coordinates_file\n");
        return NULL;
    }

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open coordinates file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    // DETECT DELIMITER
    char delimiter = detect_file_delimiter(filename);
    printf("Reading coordinates from '%s' with delimiter '%c'\n", filename, delimiter);

    char *line = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    int id_col_idx = -1, x_col_idx = -1, y_col_idx = -1;
    int header_field_count = 0;

    // Parse Header
    line_len = getline(&line, &line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty header in coordinates file '%s'\n", filename);
        fclose(fp);
        if(line) free(line);
        return NULL;
    }

    // Trim EOL
    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
        line[--line_len] = '\0';
    }

    char* header_copy = strdup(line);
    if(!header_copy) {
        perror("strdup header for coord file");
        fclose(fp);
        free(line);
        return NULL;
    }

    // Use strsep instead of strtok to preserve empty tokens
    char delim_str[2] = {delimiter, '\0'};
    char* header_ptr = header_copy;
    char* token_h;
    int current_col_idx = 0;

    while ((token_h = strsep(&header_ptr, delim_str)) != NULL) {
        char* trimmed_token = trim_whitespace_inplace(token_h);

        // Only match non-empty column names (skip leading comma columns)
        if (strlen(trimmed_token) > 0) {
            if (strcmp(trimmed_token, id_column_name) == 0) {
                id_col_idx = current_col_idx;
            }
            if (strcmp(trimmed_token, x_column_name) == 0) {
                x_col_idx = current_col_idx;
            }
            if (strcmp(trimmed_token, y_column_name) == 0) {
                y_col_idx = current_col_idx;
            }
        }

        current_col_idx++;
        header_field_count++;
    }
    free(header_copy);

    if (id_col_idx == -1 || x_col_idx == -1 || y_col_idx == -1) {
        fprintf(stderr, "Error: Required columns not found in coordinates file '%s'\n", filename);
        fprintf(stderr, "  ID ('%s'): %s, X ('%s'): %s, Y ('%s'): %s\n",
                id_column_name, (id_col_idx!=-1?"Found":"Missing"),
                x_column_name, (x_col_idx!=-1?"Found":"Missing"),
                y_column_name, (y_col_idx!=-1?"Found":"Missing"));
        fclose(fp);
        if(line) free(line);
        return NULL;
    }

    // Count data lines
    long current_pos = ftell(fp);
    MKL_INT num_data_lines = 0;
    while ((line_len = getline(&line, &line_buf_size, fp)) > 0) {
        char* p = line;
        while(isspace((unsigned char)*p)) p++;
        if(*p != '\0') num_data_lines++;
    }
    fseek(fp, current_pos, SEEK_SET);

    if (num_data_lines == 0) {
        fprintf(stderr, "Error: No data lines found in coordinates file '%s'\n", filename);
        fclose(fp);
        if(line) free(line);
        return NULL;
    }

    SpotCoordinates* coords = (SpotCoordinates*)malloc(sizeof(SpotCoordinates));
    if (!coords) {
        perror("malloc SpotCoordinates for file read");
        fclose(fp);
        if(line) free(line);
        return NULL;
    }

    coords->total_spots = num_data_lines;
    coords->spot_row = (MKL_INT*)malloc((size_t)num_data_lines * sizeof(MKL_INT));
    coords->spot_col = (MKL_INT*)malloc((size_t)num_data_lines * sizeof(MKL_INT));
    coords->valid_mask = (int*)calloc(num_data_lines, sizeof(int));
    coords->spot_names = (char**)malloc((size_t)num_data_lines * sizeof(char*));
    coords->valid_spots = 0;

    if (!coords->spot_row || !coords->spot_col || !coords->valid_mask || !coords->spot_names) {
        perror("Failed to allocate arrays for coordinates read from file");
        free_spot_coordinates(coords);
        fclose(fp);
        if(line) free(line);
        return NULL;
    }

    // Initialize spot_names to NULL for safe cleanup
    for (MKL_INT i = 0; i < num_data_lines; i++) {
        coords->spot_names[i] = NULL;
    }

    // Read data lines
    MKL_INT spot_fill_idx = 0;

    while ((line_len = getline(&line, &line_buf_size, fp)) > 0 && spot_fill_idx < num_data_lines) {
        char* p_chk = line;
        while(isspace((unsigned char)*p_chk)) p_chk++;
        if(*p_chk == '\0') continue;

        // Trim EOL
        while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) {
            line[--line_len] = '\0';
        }
        if (line_len == 0) continue;

        char* data_line_copy = strdup(line);
        if(!data_line_copy) {
            perror("strdup data line for coord file");
            break;
        }

        char* field_tokens[header_field_count > 0 ? header_field_count : 1];
        int actual_tokens = 0;
        char* current_ptr = data_line_copy;

        // Use strsep to preserve empty tokens (consistent with header parsing)
        for(int k = 0; k < header_field_count; ++k) {
            field_tokens[k] = strsep(&current_ptr, delim_str);
            if(field_tokens[k] == NULL) break;
            actual_tokens++;
        }

        char* id_str = NULL;
        double x_val = NAN, y_val = NAN;

        if (id_col_idx >= 0 && id_col_idx < actual_tokens && field_tokens[id_col_idx] != NULL) {
            id_str = trim_whitespace_inplace(field_tokens[id_col_idx]);
        }
        if (x_col_idx >= 0 && x_col_idx < actual_tokens && field_tokens[x_col_idx] != NULL) {
            char* x_token = trim_whitespace_inplace(field_tokens[x_col_idx]);
            char* end_x;
            x_val = strtod(x_token, &end_x);
            if (end_x == x_token || *end_x != '\0') x_val = NAN;
        }
        if (y_col_idx >= 0 && y_col_idx < actual_tokens && field_tokens[y_col_idx] != NULL) {
            char* y_token = trim_whitespace_inplace(field_tokens[y_col_idx]);
            char* end_y;
            y_val = strtod(y_token, &end_y);
            if (end_y == y_token || *end_y != '\0') y_val = NAN;
        }

        if (id_str && strlen(id_str) > 0 && isfinite(x_val) && isfinite(y_val)) {
            coords->spot_names[spot_fill_idx] = strdup(id_str);
            if (!coords->spot_names[spot_fill_idx]) {
                perror("strdup spot_id in read_coord_file");
                free(data_line_copy);
                break;
            }
            coords->spot_row[spot_fill_idx] = (MKL_INT)round(y_val * coord_scale);
            coords->spot_col[spot_fill_idx] = (MKL_INT)round(x_val * coord_scale);
            coords->valid_mask[spot_fill_idx] = 1;
            coords->valid_spots++;
        } else {
            coords->spot_names[spot_fill_idx] = strdup("INVALID_COORD_ENTRY");
            if(!coords->spot_names[spot_fill_idx]) {
                perror("strdup INVALID_COORD_ENTRY");
                free(data_line_copy);
                break;
            }
            coords->spot_row[spot_fill_idx] = -1;
            coords->spot_col[spot_fill_idx] = -1;
            coords->valid_mask[spot_fill_idx] = 0;
        }

        free(data_line_copy);
        spot_fill_idx++;
    }

    if (spot_fill_idx != coords->total_spots) {
        fprintf(stderr, "Warning: Estimated %lld data lines but processed %lld\n",
                (long long)coords->total_spots, (long long)spot_fill_idx);
        coords->total_spots = spot_fill_idx;
    }

    if(line) free(line);
    fclose(fp);

    printf("Read %lld coordinate entries, %lld are valid after processing\n",
           (long long)coords->total_spots, (long long)coords->valid_spots);

    if (coords->valid_spots == 0 && coords->total_spots > 0) {
        fprintf(stderr, "Critical Warning: No valid coordinates parsed. Check file format and column names.\n");
    }

    return coords;
}
