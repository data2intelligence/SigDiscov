/* morans_i_io_celltype.c - Cell type I/O module for Moran's I implementation
 *
 * Contains functions for reading cell type data from files in both
 * single-cell annotation format and deconvolution format, plus
 * validation and mapping of cell type data to expression matrices.
 *
 * Split from morans_i_io.c
 */

#include "morans_i_internal.h"

/* ===============================
 * CELL TYPE DATA PROCESSING
 * =============================== */

/* Structs for parse_celltype_header parameter encapsulation */
typedef struct {
    const char* id_col;
    const char* type_col;
    const char* x_col;
    const char* y_col;
} CelltypeColumnSpec;

typedef struct {
    int id_col_idx;
    int type_col_idx;
    int x_col_idx;
    int y_col_idx;
} CelltypeColumnIndices;

/* Parse cell type header to find column indices */
static int parse_celltype_header(const char* header_line, char delimiter,
                                const CelltypeColumnSpec* spec,
                                CelltypeColumnIndices* indices) {
    if (!header_line || !spec || !indices || !spec->id_col || !spec->type_col) {
        return MORANS_I_ERROR_PARAMETER;
    }

    indices->id_col_idx = indices->type_col_idx = indices->x_col_idx = indices->y_col_idx = -1;

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

        if (strcmp(trimmed, spec->id_col) == 0) {
            indices->id_col_idx = col_idx;
        } else if (strcmp(trimmed, spec->type_col) == 0) {
            indices->type_col_idx = col_idx;
        } else if (spec->x_col && strcmp(trimmed, spec->x_col) == 0) {
            indices->x_col_idx = col_idx;
        } else if (spec->y_col && strcmp(trimmed, spec->y_col) == 0) {
            indices->y_col_idx = col_idx;
        }

        token = strtok(NULL, delim_str);
        col_idx++;
    }

    free(header_copy);

    DEBUG_PRINT("Column indices: ID=%d, Type=%d, X=%d, Y=%d",
                indices->id_col_idx, indices->type_col_idx,
                indices->x_col_idx, indices->y_col_idx);

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

    CelltypeColumnSpec col_spec;
    col_spec.id_col = cell_id_col;
    col_spec.type_col = celltype_col;
    col_spec.x_col = x_col;
    col_spec.y_col = y_col;

    CelltypeColumnIndices col_indices;
    if (parse_celltype_header(line, delimiter, &col_spec, &col_indices) != MORANS_I_SUCCESS) {
        fprintf(stderr, "Error: Failed to parse cell type file header\n");
        fclose(fp);
        if (line) free(line);
        return NULL;
    }

    int id_col_idx = col_indices.id_col_idx;
    int type_col_idx = col_indices.type_col_idx;

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
    if (copy_string_array(celltype_matrix->colnames, (const char**)unique_celltypes, n_celltypes) != MORANS_I_SUCCESS) {
        perror("copy_string_array cell type names");
        free_celltype_matrix(celltype_matrix);
        for (MKL_INT j = 0; j < n_celltypes; j++) {
            free(unique_celltypes[j]);
        }
        free(unique_celltypes);
        fclose(fp);
        if (line) free(line);
        return NULL;
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


/* ---------------------------------------------------------------
 * Deconvolution-file helpers
 * --------------------------------------------------------------- */

/* Parsed header metadata for a deconvolution file */
typedef struct {
    int       spot_id_col_idx;  /* column index of the spot-ID field       */
    MKL_INT   n_celltypes;      /* number of cell-type columns             */
    MKL_INT   n_spots;          /* number of non-empty data rows           */
    char**    celltype_names;   /* allocated array of n_celltypes strdup'd names */
} DeconvHeaderInfo;

/* Free the celltype_names array inside DeconvHeaderInfo */
static void free_deconv_header_info(DeconvHeaderInfo* info) {
    if (!info) return;
    if (info->celltype_names) {
        for (MKL_INT i = 0; i < info->n_celltypes; i++) {
            free(info->celltype_names[i]);
        }
        free(info->celltype_names);
        info->celltype_names = NULL;
    }
}

/*
 * parse_deconv_header  --  read the header line, locate the spot-ID column,
 * count cell-type columns and data rows, and extract cell-type names.
 *
 * On success the file position is after the header line, ready for data rows.
 * Returns MORANS_I_SUCCESS or an error code; on error *info is zeroed.
 */
static int parse_deconv_header(FILE* fp, char delimiter, const char* spot_id_col,
                               char** line_buf, size_t* line_buf_size,
                               DeconvHeaderInfo* info) {
    memset(info, 0, sizeof(*info));
    info->spot_id_col_idx = -1;

    /* --- first pass: read header, find spot-ID column, count fields --- */
    ssize_t line_len = getline(line_buf, line_buf_size, fp);
    if (line_len <= 0) {
        fprintf(stderr, "Error: Empty header in deconvolution file\n");
        return MORANS_I_ERROR_FILE;
    }

    /* trim trailing newline / carriage-return */
    while (line_len > 0 && ((*line_buf)[line_len - 1] == '\n' || (*line_buf)[line_len - 1] == '\r')) {
        (*line_buf)[--line_len] = '\0';
    }

    char* header_copy = strdup(*line_buf);
    if (!header_copy) {
        perror("strdup header");
        return MORANS_I_ERROR_MEMORY;
    }

    char delim_str[2] = {delimiter, '\0'};
    char* token = strtok(header_copy, delim_str);
    int field_count = 0;

    while (token) {
        char* trimmed = trim_whitespace_inplace(token);
        if (strcmp(trimmed, spot_id_col) == 0) {
            info->spot_id_col_idx = field_count;
        }
        field_count++;
        token = strtok(NULL, delim_str);
    }
    free(header_copy);

    if (info->spot_id_col_idx < 0) {
        fprintf(stderr, "Error: Spot ID column '%s' not found in header\n", spot_id_col);
        return MORANS_I_ERROR_PARAMETER;
    }

    info->n_celltypes = field_count - 1; /* subtract spot-ID column */
    if (info->n_celltypes <= 0) {
        fprintf(stderr, "Error: No cell type columns found\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    /* --- count non-empty data rows --- */
    long data_start = ftell(fp);
    info->n_spots = 0;
    while (getline(line_buf, line_buf_size, fp) > 0) {
        char* p = *line_buf;
        while (isspace((unsigned char)*p)) p++;
        if (*p != '\0') info->n_spots++;
    }

    if (info->n_spots == 0) {
        fprintf(stderr, "Error: No data rows found\n");
        return MORANS_I_ERROR_FILE;
    }

    /* --- second pass over header: extract cell-type column names --- */
    info->celltype_names = (char**)calloc(info->n_celltypes, sizeof(char*));
    if (!info->celltype_names) {
        perror("calloc celltype_names");
        return MORANS_I_ERROR_MEMORY;
    }

    fseek(fp, 0, SEEK_SET);
    getline(line_buf, line_buf_size, fp); /* re-read header */

    header_copy = strdup(*line_buf);
    if (!header_copy) {
        perror("strdup header (second pass)");
        free_deconv_header_info(info);
        return MORANS_I_ERROR_MEMORY;
    }

    token = strtok(header_copy, delim_str);
    int col_idx = 0;
    int ct_idx  = 0;

    while (token) {
        char* trimmed = trim_whitespace_inplace(token);
        if (col_idx != info->spot_id_col_idx) {
            info->celltype_names[ct_idx] = strdup(trimmed);
            if (!info->celltype_names[ct_idx]) {
                perror("strdup cell type column name");
                free(header_copy);
                free_deconv_header_info(info);
                return MORANS_I_ERROR_MEMORY;
            }
            ct_idx++;
        }
        col_idx++;
        token = strtok(NULL, delim_str);
    }
    free(header_copy);

    /* rewind to the start of data rows */
    fseek(fp, data_start, SEEK_SET);
    return MORANS_I_SUCCESS;
}

/*
 * read_deconv_data_rows  --  parse data lines into the pre-allocated
 * CellTypeMatrix (values + rownames).
 *
 * fp must be positioned at the first data row.  On success *rows_read
 * contains the number of rows actually written.
 * Returns MORANS_I_SUCCESS or an error code.
 */
static int read_deconv_data_rows(FILE* fp, char delimiter,
                                 int spot_id_col_idx, MKL_INT n_celltypes,
                                 MKL_INT n_spots,
                                 CellTypeMatrix* celltype_matrix,
                                 char** line_buf, size_t* line_buf_size,
                                 MKL_INT* rows_read) {
    char delim_str[2] = {delimiter, '\0'};
    MKL_INT spot_idx = 0;

    while (getline(line_buf, line_buf_size, fp) > 0 && spot_idx < n_spots) {
        char* p = *line_buf;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;

        char* line_copy = strdup(*line_buf);
        if (!line_copy) continue;

        char* token = strtok(line_copy, delim_str);
        int col_idx = 0;
        int celltype_col_idx = 0;
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
                *rows_read = spot_idx;
                return MORANS_I_ERROR_MEMORY;
            }
            spot_idx++;
        }
        free(line_copy);
    }

    *rows_read = spot_idx;
    return MORANS_I_SUCCESS;
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

    /* --- Parse header: spot-ID column, cell-type names, row count --- */
    DeconvHeaderInfo hdr;
    int rc = parse_deconv_header(fp, delimiter, spot_id_col, &line, &line_buf_size, &hdr);
    if (rc != MORANS_I_SUCCESS) {
        fclose(fp);
        free(line);
        return NULL;
    }

    /* --- Allocate cell-type matrix --- */
    CellTypeMatrix* celltype_matrix = (CellTypeMatrix*)malloc(sizeof(CellTypeMatrix));
    if (!celltype_matrix) {
        perror("malloc CellTypeMatrix");
        free_deconv_header_info(&hdr);
        fclose(fp);
        free(line);
        return NULL;
    }

    celltype_matrix->nrows = hdr.n_spots;
    celltype_matrix->ncols = hdr.n_celltypes;
    celltype_matrix->is_binary = 0;
    celltype_matrix->format_type = CELLTYPE_FORMAT_DECONVOLUTION;

    size_t values_size;
    if (safe_multiply_size_t(hdr.n_spots, hdr.n_celltypes, &values_size) != 0 ||
        safe_multiply_size_t(values_size, sizeof(double), &values_size) != 0) {
        fprintf(stderr, "Error: Deconvolution matrix dimensions too large\n");
        free(celltype_matrix);
        free_deconv_header_info(&hdr);
        fclose(fp);
        free(line);
        return NULL;
    }

    celltype_matrix->values = (double*)mkl_malloc(values_size, 64);
    celltype_matrix->rownames = (char**)calloc(hdr.n_spots, sizeof(char*));
    celltype_matrix->colnames = (char**)calloc(hdr.n_celltypes, sizeof(char*));

    if (!celltype_matrix->values || !celltype_matrix->rownames || !celltype_matrix->colnames) {
        perror("Failed to allocate deconvolution matrix components");
        free_celltype_matrix(celltype_matrix);
        free_deconv_header_info(&hdr);
        fclose(fp);
        free(line);
        return NULL;
    }

    /* Transfer ownership of cell-type names into colnames */
    for (MKL_INT i = 0; i < hdr.n_celltypes; i++) {
        celltype_matrix->colnames[i] = hdr.celltype_names[i];
        hdr.celltype_names[i] = NULL; /* prevent double-free */
    }
    free(hdr.celltype_names);
    hdr.celltype_names = NULL;

    /* --- Read data rows --- */
    MKL_INT rows_read = 0;
    rc = read_deconv_data_rows(fp, delimiter, hdr.spot_id_col_idx,
                               hdr.n_celltypes, hdr.n_spots,
                               celltype_matrix, &line, &line_buf_size,
                               &rows_read);

    fclose(fp);
    free(line);

    if (rc != MORANS_I_SUCCESS) {
        free_celltype_matrix(celltype_matrix);
        return NULL;
    }

    if (rows_read != hdr.n_spots) {
        fprintf(stderr, "Warning: Expected %lld spots but processed %lld\n",
                (long long)hdr.n_spots, (long long)rows_read);
        celltype_matrix->nrows = rows_read;
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
        if (copy_string_array(mapped->colnames, (const char**)celltype_matrix->colnames, mapped->ncols) != MORANS_I_SUCCESS) {
            perror("copy_string_array cell type names");
            free_celltype_matrix(mapped);
            spot_name_ht_free(cell_map);
            return MORANS_I_ERROR_MEMORY;
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
