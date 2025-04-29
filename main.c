/* main.c - Main program for Moran's I calculation using Intel MKL */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <ctype.h>
#include <omp.h>
#include <errno.h>
#include <unistd.h>  /* For access() */
#include <time.h>    /* For timing */
#include <sys/time.h> /* For high-resolution timing */
#include "morans_i_mkl.h"

/* Get current time in seconds with microsecond precision */
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

/* Print elapsed time in seconds */
void print_elapsed_time(double start_time, double end_time, const char* operation) {
    double elapsed = end_time - start_time;
    printf("Time for %s: %.6f seconds\n", operation, elapsed);
}

/* Main function */
int main(int argc, char* argv[]) {
    /* For timing */
    double start_time, end_time;
    double total_start_time = get_time();
    
    /* Set default parameters */
    char input_file[BUFFER_SIZE] = "";
    char output_file[BUFFER_SIZE] = "";
    char meta_file[BUFFER_SIZE] = "";           /* Coordinates/metadata file */
    char id_column[BUFFER_SIZE] = "cell_ID";    /* Default column names */
    char x_column[BUFFER_SIZE] = "sdimx";
    char y_column[BUFFER_SIZE] = "sdimy";
    double coord_scale = DEFAULT_COORD_SCALE_FACTOR;
    double custom_sigma = 0.0; /* Default: 0 means use platform-specific default */
    MKL_INT max_radius = DEFAULT_MAX_RADIUS;
    int platform_mode = DEFAULT_PLATFORM_MODE;
    int calc_pairwise = DEFAULT_CALC_PAIRWISE;
    int calc_all_vs_all = DEFAULT_CALC_ALL_VS_ALL;
    int include_same_spot = DEFAULT_INCLUDE_SAME_SPOT;

    /* Check for help flag */
    if (argc == 1 || (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))) {
        print_help(argv[0]);
        return 0;
    }

    /* Parse command line arguments */
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            fprintf(stderr, "Error: Missing value for parameter %s.\n", argv[i]);
            return 1;
        }
        
        if (strcmp(argv[i], "-i") == 0) {
            strncpy(input_file, argv[i+1], BUFFER_SIZE - 1);
            input_file[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "-o") == 0) {
            strncpy(output_file, argv[i+1], BUFFER_SIZE - 1);
            output_file[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "-c") == 0) {
            strncpy(meta_file, argv[i+1], BUFFER_SIZE - 1);
            meta_file[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "--id-col") == 0) {
            strncpy(id_column, argv[i+1], BUFFER_SIZE - 1);
            id_column[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "--x-col") == 0) {
            strncpy(x_column, argv[i+1], BUFFER_SIZE - 1);
            x_column[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "--y-col") == 0) {
            strncpy(y_column, argv[i+1], BUFFER_SIZE - 1);
            y_column[BUFFER_SIZE - 1] = '\0';
        } else if (strcmp(argv[i], "--scale") == 0) {
            coord_scale = load_double_value(argv[i+1], "--scale");
        } else if (strcmp(argv[i], "--sigma") == 0) {
            custom_sigma = load_double_value(argv[i+1], "--sigma");
        } else if (strcmp(argv[i], "-r") == 0) {
            max_radius = load_positive_value(argv[i+1], "-r", 1, 100);
        } else if (strcmp(argv[i], "-p") == 0) {
            platform_mode = load_positive_value(argv[i+1], "-p", 0, 10);
        } else if (strcmp(argv[i], "-b") == 0) {
            calc_pairwise = load_positive_value(argv[i+1], "-b", 0, 1);
        } else if (strcmp(argv[i], "-g") == 0) {
            calc_all_vs_all = load_positive_value(argv[i+1], "-g", 0, 1);
        } else if (strcmp(argv[i], "-s") == 0) {
            include_same_spot = load_positive_value(argv[i+1], "-s", 0, 1);
        } else if (strcmp(argv[i], "-t") == 0) {
            int omp_threads = load_positive_value(argv[i+1], "-t", 1, 256);
            omp_set_num_threads(omp_threads);
            printf("Setting OpenMP threads to: %d\n", omp_threads);
        } else if (strcmp(argv[i], "-m") == 0) {
            int mkl_threads = load_positive_value(argv[i+1], "-m", 1, 256);
            mkl_set_num_threads(mkl_threads);
            printf("Setting MKL threads to: %d\n", mkl_threads);
        } else {
            fprintf(stderr, "Error: Cannot recognize parameter \"%s\". Use -h for help.\n", argv[i]);
            return 1;
        }
    }

    /* Validate parameters */
    if (strlen(input_file) == 0) {
        fprintf(stderr, "Error: Input file (-i) must be specified.\n");
        return 1;
    }
    if (strlen(output_file) == 0) {
        fprintf(stderr, "Error: Output file (-o) must be specified.\n");
        return 1;
    }
    if (access(input_file, R_OK) != 0) {
        fprintf(stderr, "Error: Cannot access input file: %s\n", input_file);
        return 1;
    }
    
    /* Check if metadata file is provided and accessible */
    int use_metadata_file = 0;
    if (strlen(meta_file) > 0) {
        if (access(meta_file, R_OK) != 0) {
            fprintf(stderr, "Error: Cannot access metadata file: %s\n", meta_file);
            return 1;
        }
        use_metadata_file = 1;
        platform_mode = SINGLE_CELL; /* Force single cell mode */
    }

    printf("--- Parameters ---\n");
    printf("Input file: %s\n", input_file);
    printf("Output file: %s\n", output_file);
    if (use_metadata_file) {
        printf("Metadata file: %s\n", meta_file);
        printf("ID column: %s\n", id_column);
        printf("X coordinate column: %s\n", x_column);
        printf("Y coordinate column: %s\n", y_column);
        printf("Coordinate scale factor: %.2f\n", coord_scale);
    }
    printf("Max radius: %lld\n", (long long)max_radius);
    printf("Platform: %d (%s)\n", platform_mode, 
           platform_mode == VISIUM ? "Visium" : 
           (platform_mode == OLD ? "Old ST" : "Single Cell"));
    printf("Mode: %s\n", calc_pairwise ? "Pairwise" : "Single Gene");
    if (calc_pairwise) printf("Gene Pairs: %s\n", calc_all_vs_all ? "All vs All" : "First Gene vs All");
    printf("Include Self-Pairs: %s\n", include_same_spot ? "Yes" : "No");
    printf("Implementation: Matrix-based (Intel MKL)\n");
    printf("------------------\n");

    /* MKL threading info */
    int omp_threads = omp_get_max_threads();
    int mkl_threads = mkl_get_max_threads();
    printf("Thread configuration:\n");
    printf("  OpenMP threads: %d\n", omp_threads);
    printf("  MKL threads: %d\n", mkl_threads);
    printf("------------------\n");

    /* For single-cell data, infer sigma if not provided and platform is single-cell */
    if (platform_mode == SINGLE_CELL && custom_sigma <= 0.0) {
        custom_sigma = infer_sigma_from_data(coords, coord_scale);
    }

    /* 1. Load data */
    printf("Loading data from %s...\n", input_file);
    start_time = get_time();
    DenseMatrix* vst_matrix = read_vst_file(input_file);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "data loading");
    
    if (vst_matrix == NULL) {
        fprintf(stderr, "Error: Failed to load data.\n");
        return 1;
    }
    printf("Loaded data matrix: %lld genes x %lld spots.\n",
           (long long)vst_matrix->nrows, (long long)vst_matrix->ncols);

    /* Check for non-finite values (replace with 0.0) */
    MKL_INT total_elements = vst_matrix->nrows * vst_matrix->ncols;
    #pragma omp parallel for
    for (MKL_INT i = 0; i < total_elements; i++) {
        if (!isfinite(vst_matrix->values[i])) {
            vst_matrix->values[i] = 0.0;
        }
    }

    /* 2. Z-Normalize data */
    printf("Performing Z-normalization...\n");
    start_time = get_time();
    DenseMatrix* znorm_matrix = z_normalize(vst_matrix);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "Z-normalization");
    
    /* Free original data only after znorm_matrix is successfully created */
    if (!znorm_matrix) {
        fprintf(stderr, "Error: Z-normalization failed.\n");
        free_dense_matrix(vst_matrix);
        return 1;
    }
    free_dense_matrix(vst_matrix); /* Free original data now */

    /* 3. Prepare Spatial Data */
    printf("Preparing spatial data...\n");
    start_time = get_time();
    SpotCoordinates* coords;
    
    if (use_metadata_file) {
        /* Use metadata file for coordinates */
        coords = read_coordinates_file(meta_file, id_column, x_column, y_column, coord_scale);
    } else {
        /* Extract coordinates from spot names */
        coords = extract_coordinates(znorm_matrix->colnames, znorm_matrix->ncols);
    }
    
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "coordinate extraction");
    
    if (!coords) {
         fprintf(stderr, "Error: Failed to extract coordinates.\n");
         free_dense_matrix(znorm_matrix);
         return 1;
    }
    printf("%lld / %lld spots have valid coordinates.\n",
           (long long)coords->valid_spots, (long long)coords->total_spots);

    if (coords->valid_spots == 0) {
        fprintf(stderr, "Error: No valid spot coordinates found. Cannot proceed.\n");
        free_dense_matrix(znorm_matrix);
        free_spot_coordinates(coords);
        return 1;
    }

    /* For metadata files, map expression columns to coordinate spots */
    MKL_INT* expr_to_coord_mapping = NULL;
    if (use_metadata_file) {
        printf("Mapping expression data to coordinates...\n");
        if (!map_expression_to_coordinates(znorm_matrix, coords, &expr_to_coord_mapping)) {
            fprintf(stderr, "Error: Failed to map expression data to coordinates.\n");
            free_dense_matrix(znorm_matrix);
            free_spot_coordinates(coords);
            return 1;
        }
    }

    /* Extract valid coordinates and original indices */
    MKL_INT* spot_row_valid = (MKL_INT*)malloc(coords->valid_spots * sizeof(MKL_INT));
    MKL_INT* spot_col_valid = (MKL_INT*)malloc(coords->valid_spots * sizeof(MKL_INT));
    char** valid_spot_names = (char**)malloc(coords->valid_spots * sizeof(char*));
    MKL_INT* valid_indices = (MKL_INT*)malloc(coords->valid_spots * sizeof(MKL_INT));

    if (!spot_row_valid || !spot_col_valid || !valid_spot_names || !valid_indices) {
        fprintf(stderr, "Error: Memory allocation failed for valid coordinate arrays.\n");
        /* Free allocated memory before exiting */
        free(spot_row_valid); free(spot_col_valid); free(valid_spot_names); free(valid_indices);
        free(expr_to_coord_mapping);
        free_dense_matrix(znorm_matrix); free_spot_coordinates(coords);
        return 1;
    }

    MKL_INT valid_count = 0;
    for (MKL_INT i = 0; i < coords->total_spots; i++) {
        if (coords->valid_mask[i]) {
            spot_row_valid[valid_count] = coords->spot_row[i];
            spot_col_valid[valid_count] = coords->spot_col[i];
            valid_spot_names[valid_count] = strdup(coords->spot_names[i]);
            valid_indices[valid_count] = i;
            valid_count++;
        }
    }

    /* 4. Create distance decay matrix */
    printf("Creating distance decay matrix with max_radius=%lld...\n", (long long)max_radius);
    start_time = get_time();
    DenseMatrix* decay_matrix = create_distance_matrix(max_radius, platform_mode, custom_sigma);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "distance matrix creation");
    
    if (!decay_matrix) {
         fprintf(stderr, "Error: Failed to create decay matrix.\n");
         for(MKL_INT k=0; k<valid_count; ++k) free(valid_spot_names[k]);
         free(spot_row_valid); free(spot_col_valid); free(valid_spot_names); free(valid_indices);
         free(expr_to_coord_mapping);
         free_dense_matrix(znorm_matrix); free_spot_coordinates(coords);
         return 1;
    }

    /* Handle self-weight in the decay matrix */
    if (!include_same_spot) {
        printf("Setting self-comparison weight (decay_matrix[0,0]) to 0.0 as per -s 0.\n");
        decay_matrix->values[0] = 0.0;
    } else {
        printf("Using calculated self-comparison weight: %.6f (include_same_spot=1)\n",
               decay_matrix->values[0]);
    }

    /* 5. Build Spatial Weight Matrix W */
    printf("Building spatial weight matrix (W)...\n");
    start_time = get_time();
    SparseMatrix* W_valid = build_spatial_weight_matrix(
        spot_row_valid, spot_col_valid, coords->valid_spots,
        decay_matrix, max_radius);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "spatial weight matrix construction");

    /* Free decay matrix now that W is built */
    free_dense_matrix(decay_matrix);

    if (!W_valid) {
         fprintf(stderr, "Error: Failed to build spatial weight matrix.\n");
         for(MKL_INT k=0; k<valid_count; ++k) free(valid_spot_names[k]);
         free(spot_row_valid); free(spot_col_valid); free(valid_spot_names); free(valid_indices);
         free(expr_to_coord_mapping);
         free_dense_matrix(znorm_matrix); free_spot_coordinates(coords);
         return 1;
    }

    /* Calculate sum of weights S0 */
    double S0 = 0.0;
    #pragma omp parallel for reduction(+:S0)
    for (MKL_INT i = 0; i < W_valid->nnz; i++) {
        S0 += W_valid->values[i];
    }
    printf("Sum of weights S0: %.6f\n", S0);

    if (fabs(S0) < DBL_EPSILON && coords->valid_spots > 0) {
        fprintf(stderr, "Warning: Sum of weights S0 is near-zero (%.4e). Moran's I results will likely be 0 or NaN/Inf.\n", S0);
    }

    /* 6. Subset Z-normalized matrix to valid spots only */
    printf("Subsetting Z-normalized matrix to valid spots...\n");
    start_time = get_time();
    DenseMatrix* X_calc = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!X_calc) {
        fprintf(stderr, "Error: Memory allocation failed for X_calc structure.\n");
        for(MKL_INT k=0; k<valid_count; ++k) free(valid_spot_names[k]);
        free(spot_row_valid); free(spot_col_valid); free(valid_spot_names); free(valid_indices);
        free(expr_to_coord_mapping);
        free_dense_matrix(znorm_matrix); free_spot_coordinates(coords); free_sparse_matrix(W_valid);
        return 1;
    }
    X_calc->nrows = coords->valid_spots; /* Rows are valid spots */
    X_calc->ncols = znorm_matrix->nrows; /* Columns are genes */
    X_calc->values = (double*)mkl_malloc(X_calc->nrows * X_calc->ncols * sizeof(double), 64);
    X_calc->rownames = (char**)malloc(X_calc->nrows * sizeof(char*));
    X_calc->colnames = (char**)malloc(X_calc->ncols * sizeof(char*));

    if (!X_calc->values || !X_calc->rownames || !X_calc->colnames) {
        fprintf(stderr, "Error: Memory allocation failed for X_calc data.\n");
        if (X_calc->values) mkl_free(X_calc->values);
        free(X_calc->rownames); free(X_calc->colnames); free(X_calc);
        for(MKL_INT k=0; k<valid_count; ++k) free(valid_spot_names[k]);
        free(spot_row_valid); free(spot_col_valid); free(valid_spot_names); free(valid_indices);
        free(expr_to_coord_mapping);
        free_dense_matrix(znorm_matrix); free_spot_coordinates(coords); free_sparse_matrix(W_valid);
        return 1;
    }

    /* Copy gene names (colnames for X_calc) */
    for (MKL_INT j = 0; j < X_calc->ncols; j++) {
        X_calc->colnames[j] = strdup(znorm_matrix->rownames[j]);
    }
    /* Copy valid spot names (rownames for X_calc) */
    for (MKL_INT i = 0; i < X_calc->nrows; i++) {
        X_calc->rownames[i] = strdup(valid_spot_names[i]);
    }

    /* Populate X_calc->values by selecting valid columns from znorm_matrix and transposing */
    printf("Populating X matrix (valid spots x genes)...\n");
    #pragma omp parallel for schedule(static)
    for (MKL_INT gene_idx = 0; gene_idx < X_calc->ncols; ++gene_idx) {
        for (MKL_INT valid_spot_idx = 0; valid_spot_idx < X_calc->nrows; ++valid_spot_idx) {
            MKL_INT original_spot_idx;
            
            if (use_metadata_file && expr_to_coord_mapping) {
                /* For metadata file, use the mapping to get the expression column */
                original_spot_idx = expr_to_coord_mapping[valid_spot_idx];
                if (original_spot_idx < 0) {
                    /* No expression data for this spot, use 0.0 */
                    X_calc->values[valid_spot_idx * X_calc->ncols + gene_idx] = 0.0;
                    continue;
                }
            } else {
                /* For standard Visium/ST data, use the valid_indices directly */
                original_spot_idx = valid_indices[valid_spot_idx];
            }
            
            X_calc->values[valid_spot_idx * X_calc->ncols + gene_idx] =
                znorm_matrix->values[gene_idx * znorm_matrix->ncols + original_spot_idx];
        }
    }
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "data matrix preparation");

    /* Free the full z-normalized matrix and coordinate helpers now */
    free_dense_matrix(znorm_matrix);
    free(spot_row_valid);
    free(spot_col_valid);
    for (MKL_INT i = 0; i < coords->valid_spots; i++) {
        free(valid_spot_names[i]);
    }
    free(valid_spot_names);
    free(valid_indices);
    free(expr_to_coord_mapping);
    free_spot_coordinates(coords);

    /* 7. Calculate Moran's I based on the selected mode */
    printf("Calculating Moran's I...\n");
    start_time = get_time();
    
    if (!calc_pairwise) {
        /* Single gene mode: Calculate Moran's I for each gene */
        printf("Calculating Single-Gene Moran's I for each gene...\n");
        save_single_gene_results(X_calc, W_valid, S0, output_file);
    } else if (!calc_all_vs_all) {
        /* First gene vs. all mode */
        printf("Calculating Moran's I between first gene and all others...\n");
        double* first_gene_results = calculate_first_gene_vs_all(X_calc, W_valid, S0);
        if (first_gene_results) {
            const char** gene_names = (const char**)X_calc->colnames;
            save_first_gene_vs_all_results(first_gene_results, gene_names, X_calc->ncols, output_file);
            mkl_free(first_gene_results); /* Use mkl_free since we're now using mkl_malloc */
        } else {
            fprintf(stderr, "Error: Failed to calculate first gene vs. all.\n");
        }
    } else {
        /* All genes vs. all genes mode */
        printf("Calculating Pairwise Moran's I for all gene pairs...\n");
        DenseMatrix* Moran_I_Result = calculate_morans_i(X_calc, W_valid);
        if (Moran_I_Result) {
            /* Division by S0 is now handled inside calculate_morans_i */
            save_results(Moran_I_Result, output_file);
            free_dense_matrix(Moran_I_Result);
        } else {
            fprintf(stderr, "Error: Failed to calculate pairwise Moran's I.\n");
        }
    }
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "Moran's I calculation");

    /* Free calculation inputs */
    free_dense_matrix(X_calc);
    free_sparse_matrix(W_valid);

    double total_end_time = get_time();
    print_elapsed_time(total_start_time, total_end_time, "TOTAL EXECUTION");
    
    printf("--- MKL Moran's I calculation completed ---\n");
    mkl_thread_free_buffers();
    mkl_free_buffers();
    return 0;
}