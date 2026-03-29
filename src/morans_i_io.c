/* morans_i_io.c - Expression and coordinate I/O module for Moran's I implementation
 *
 * Contains file format detection, VST expression file parsing, and coordinate
 * file reading/mapping functions.  Cell type I/O, weight matrix I/O, and
 * results saving have been split into separate modules:
 *   morans_i_io_celltype.c  - cell type reading and mapping
 *   morans_i_io_weights.c   - weight matrix reading and validation
 *   morans_i_io_results.c   - all save_* functions
 *
 * Split from the original monolithic morans_i_io.c
 */

#include "morans_i_internal.h"

/* ===============================
 * FILE FORMAT DETECTION
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

/* ===============================
 * VST FILE PARSING (REFACTORED)
 * =============================== */

/* Forward declarations for static VST helpers */
static int detect_file_format(const char* filename);
static int parse_vst_header(FILE* fp, char** line, size_t* line_buf_size,
                           MKL_INT* n_spots_out, char*** colnames_out, int is_csv);
static int count_vst_genes(FILE* fp, char** line, size_t* line_buf_size, MKL_INT* n_genes_out);
static int read_vst_data_rows(FILE* fp, char** line, size_t* line_buf_size,
                             DenseMatrix* matrix, MKL_INT n_genes_expected,
                             MKL_INT n_spots_expected, int is_csv);

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
 * COORDINATE I/O FUNCTIONS
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
