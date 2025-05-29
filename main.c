/* main.c - Main program for Moran's I calculation using Intel MKL
 *
 * Version: 1.3.0 (Added Residual Moran's I support and improved structure)
 * Enhanced: Added residual analysis, better error handling, and code organization
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

/* Resource Management Structure */
typedef struct {
    DenseMatrix* vst_matrix;
    DenseMatrix* znorm_matrix;
    SpotCoordinates* spot_coords;
    DenseMatrix* decay_matrix;
    SparseMatrix* W_matrix;
    DenseMatrix* X_calc;
    DenseMatrix* observed_results;
    PermutationResults* perm_results;
    
    /* Residual analysis resources */
    CellTypeMatrix* celltype_matrix;
    ResidualResults* residual_results;
    
    /* Helper arrays */
    MKL_INT* valid_spot_indices;
    MKL_INT* valid_spot_rows;
    MKL_INT* valid_spot_cols;
    char** valid_spot_names;
    MKL_INT num_valid_spots;
} AnalysisResources;

/* Command Line Arguments Structure */
typedef struct {
    char input_file[BUFFER_SIZE];
    char output_prefix[BUFFER_SIZE];
    char meta_file[BUFFER_SIZE];
    char id_column[BUFFER_SIZE];
    char x_column[BUFFER_SIZE];
    char y_column[BUFFER_SIZE];
    double custom_sigma;
    int use_metadata_file;
    int run_toy_example;
    
    /* Residual analysis specific arguments */
    char celltype_file[BUFFER_SIZE];
    char celltype_id_col[BUFFER_SIZE];
    char celltype_type_col[BUFFER_SIZE];
    char celltype_x_col[BUFFER_SIZE];
    char celltype_y_col[BUFFER_SIZE];
    char spot_id_col[BUFFER_SIZE];
} CommandLineArgs;

/* Forward declarations for all functions */
void print_main_help(const char* program_name);
int parse_weight_format(const char* format_str);
int parse_analysis_mode(const char* mode_str);
int parse_celltype_format(const char* format_str);
char** extract_spot_names_from_expression(const DenseMatrix* expr_matrix, MKL_INT* n_spots_out);

/* Toy example functions */
static inline MKL_INT grid_to_1d_idx(MKL_INT r, MKL_INT c, MKL_INT num_grid_cols);
DenseMatrix* create_theoretical_toy_moran_i_matrix_2d(MKL_INT n_genes, char** gene_names);
SparseMatrix* create_toy_W_matrix_2d(MKL_INT num_grid_rows, MKL_INT num_grid_cols);
DenseMatrix* create_toy_X_calc_matrix_2d(MKL_INT num_grid_rows, MKL_INT num_grid_cols, MKL_INT n_genes);
int run_toy_example_2d(const char* output_prefix_toy, MoransIConfig* config);

/* Analysis functions */
static void initialize_resources(AnalysisResources* resources);
static void cleanup_resources(AnalysisResources* resources);
static void initialize_command_args(CommandLineArgs* args);
static int parse_command_line_arguments(int argc, char* argv[], MoransIConfig* config, CommandLineArgs* args);
static int validate_and_initialize_config(MoransIConfig* config, const CommandLineArgs* args);
static int load_and_process_expression_data(const char* input_file, AnalysisResources* resources);
static int setup_spatial_analysis(const MoransIConfig* config, const CommandLineArgs* args, AnalysisResources* resources);
static int setup_builtin_spatial_weights(const MoransIConfig* config, const CommandLineArgs* args, AnalysisResources* resources);
static int setup_custom_spatial_weights(const MoransIConfig* config, AnalysisResources* resources);
static int prepare_valid_spot_arrays(AnalysisResources* resources);
static int map_spots_to_expression(AnalysisResources* resources);
static int prepare_calculation_matrix_builtin(AnalysisResources* resources);
static int prepare_calculation_matrix_custom(AnalysisResources* resources);
static int load_celltype_data(const MoransIConfig* config, const CommandLineArgs* args, AnalysisResources* resources);
static int run_moran_analysis(const MoransIConfig* config, AnalysisResources* resources, const char* output_prefix);
static int run_residual_analysis(const MoransIConfig* config, AnalysisResources* resources, const char* output_prefix);
static int run_permutation_analysis(const MoransIConfig* config, AnalysisResources* resources, const char* output_prefix);
static void print_configuration_summary(const MoransIConfig* config, const CommandLineArgs* args);

void print_elapsed_time(double start_time, double end_time, const char* operation) {
    double elapsed = end_time - start_time;
    printf("Time for %s: %.6f seconds\n", operation, elapsed);
}

/* Initialize resources structure */
static void initialize_resources(AnalysisResources* resources) {
    if (!resources) return;
    memset(resources, 0, sizeof(AnalysisResources));
}

/* Clean up all allocated resources */
static void cleanup_resources(AnalysisResources* resources) {
    if (!resources) return;
    
    if (resources->vst_matrix) {
        free_dense_matrix(resources->vst_matrix);
        resources->vst_matrix = NULL;
    }
    if (resources->znorm_matrix) {
        free_dense_matrix(resources->znorm_matrix);
        resources->znorm_matrix = NULL;
    }
    if (resources->spot_coords) {
        free_spot_coordinates(resources->spot_coords);
        resources->spot_coords = NULL;
    }
    if (resources->decay_matrix) {
        free_dense_matrix(resources->decay_matrix);
        resources->decay_matrix = NULL;
    }
    if (resources->W_matrix) {
        free_sparse_matrix(resources->W_matrix);
        resources->W_matrix = NULL;
    }
    if (resources->X_calc) {
        free_dense_matrix(resources->X_calc);
        resources->X_calc = NULL;
    }
    if (resources->observed_results) {
        free_dense_matrix(resources->observed_results);
        resources->observed_results = NULL;
    }
    if (resources->perm_results) {
        free_permutation_results(resources->perm_results);
        resources->perm_results = NULL;
    }
    if (resources->celltype_matrix) {
        free_celltype_matrix(resources->celltype_matrix);
        resources->celltype_matrix = NULL;
    }
    if (resources->residual_results) {
        free_residual_results(resources->residual_results);
        resources->residual_results = NULL;
    }
    
    if (resources->valid_spot_names) {
        for (MKL_INT k = 0; k < resources->num_valid_spots; k++) {
            if (resources->valid_spot_names[k]) {
                free(resources->valid_spot_names[k]);
                resources->valid_spot_names[k] = NULL;
            }
        }
        free(resources->valid_spot_names);
        resources->valid_spot_names = NULL;
    }
    if (resources->valid_spot_indices) {
        free(resources->valid_spot_indices);
        resources->valid_spot_indices = NULL;
    }
    if (resources->valid_spot_rows) {
        free(resources->valid_spot_rows);
        resources->valid_spot_rows = NULL;
    }
    if (resources->valid_spot_cols) {
        free(resources->valid_spot_cols);
        resources->valid_spot_cols = NULL;
    }
    
    resources->num_valid_spots = 0;
    memset(resources, 0, sizeof(AnalysisResources));
}

/* Initialize command line arguments with defaults */
static void initialize_command_args(CommandLineArgs* args) {
    if (!args) return;
    memset(args, 0, sizeof(CommandLineArgs));
    strcpy(args->id_column, "cell_ID");
    strcpy(args->x_column, "sdimx");
    strcpy(args->y_column, "sdimy");
    strcpy(args->celltype_id_col, "cell_ID");
    strcpy(args->celltype_type_col, "cellType");
    strcpy(args->celltype_x_col, "sdimx");
    strcpy(args->celltype_y_col, "sdimy");
    strcpy(args->spot_id_col, "spot_id");
    args->custom_sigma = 0.0;
    args->use_metadata_file = 0;
    args->run_toy_example = 0;
}

/* Parse command line arguments */
static int parse_command_line_arguments(int argc, char* argv[], MoransIConfig* config, 
                                       CommandLineArgs* args) {
    if (!config || !args) {
        fprintf(stderr, "Error: NULL parameters in parse_command_line_arguments\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    if (argc == 1 || (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))) {
        print_main_help(argv[0]);
        return MORANS_I_SUCCESS; // Special case - help requested
    }
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -i\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->input_file, argv[i], BUFFER_SIZE - 1); 
            args->input_file[BUFFER_SIZE - 1] = '\0';
            
        } else if (strcmp(argv[i], "-o") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -o\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->output_prefix, argv[i], BUFFER_SIZE - 1); 
            args->output_prefix[BUFFER_SIZE - 1] = '\0';
            
        } else if (strcmp(argv[i], "--run-toy-example") == 0) {
            args->run_toy_example = 2;
            
        } else if (strcmp(argv[i], "-c") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -c\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->meta_file, argv[i], BUFFER_SIZE - 1); 
            args->meta_file[BUFFER_SIZE - 1] = '\0';
            args->use_metadata_file = 1;
            
        } else if (strcmp(argv[i], "--id-col") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --id-col\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->id_column, argv[i], BUFFER_SIZE - 1); 
            args->id_column[BUFFER_SIZE - 1] = '\0';
            
        } else if (strcmp(argv[i], "--x-col") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --x-col\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->x_column, argv[i], BUFFER_SIZE - 1); 
            args->x_column[BUFFER_SIZE - 1] = '\0';
            
        } else if (strcmp(argv[i], "--y-col") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --y-col\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->y_column, argv[i], BUFFER_SIZE - 1); 
            args->y_column[BUFFER_SIZE - 1] = '\0';
            
        } else if (strcmp(argv[i], "--scale") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --scale\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->coord_scale = load_double_value(argv[i], "--scale");
            if (isnan(config->coord_scale) || config->coord_scale <= 0) { 
                fprintf(stderr, "Error: --scale must be a positive number.\n"); 
                return MORANS_I_ERROR_PARAMETER;
            }
            
        } else if (strcmp(argv[i], "--sigma") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --sigma\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            args->custom_sigma = load_double_value(argv[i], "--sigma");
            if (isnan(args->custom_sigma)) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            
        } else if (strcmp(argv[i], "-r") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -r\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->max_radius = load_positive_value(argv[i], "-r", 1, 1000);
            if (config->max_radius < 0) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            
        } else if (strcmp(argv[i], "-w") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -w\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            if (config->custom_weights_file) free(config->custom_weights_file);
            config->custom_weights_file = strdup(argv[i]);
            if (!config->custom_weights_file) { 
                perror("strdup custom_weights_file"); 
                return MORANS_I_ERROR_MEMORY; 
            }
            config->platform_mode = CUSTOM_WEIGHTS;
            
        } else if (strcmp(argv[i], "--weight-format") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --weight-format\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->weight_format = parse_weight_format(argv[i]);
            
        } else if (strcmp(argv[i], "--normalize-weights") == 0) {
            config->normalize_weights = 1;
            
        } else if (strcmp(argv[i], "--row-normalize") == 0) {
            if (i + 1 < argc && (strcmp(argv[i + 1], "0") == 0 || strcmp(argv[i + 1], "1") == 0)) {
                // Has explicit value
                i++;
                config->row_normalize_weights = load_positive_value(argv[i], "--row-normalize", 0, 1);
                if (config->row_normalize_weights < 0) {
                    return MORANS_I_ERROR_PARAMETER;
                }
            } else {
                // No explicit value, default to 1 (enable)
                config->row_normalize_weights = 1;
            }
            
        } else if (strcmp(argv[i], "-p") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -p\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->platform_mode = load_positive_value(argv[i], "-p", 0, 3);
            if (config->platform_mode < 0) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            
        } else if (strcmp(argv[i], "-b") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -b\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->calc_pairwise = load_positive_value(argv[i], "-b", 0, 1);
            if (config->calc_pairwise < 0) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            
        } else if (strcmp(argv[i], "-g") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -g\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->calc_all_vs_all = load_positive_value(argv[i], "-g", 0, 1);
            if (config->calc_all_vs_all < 0) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            
        } else if (strcmp(argv[i], "-s") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -s\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->include_same_spot = load_positive_value(argv[i], "-s", 0, 1);
            if (config->include_same_spot < 0) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            
        } else if (strcmp(argv[i], "-t") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -t\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->n_threads = load_positive_value(argv[i], "-t", 1, 1024);
            if (config->n_threads < 0) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            
        } else if (strcmp(argv[i], "-m") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for -m\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->mkl_n_threads = load_positive_value(argv[i], "-m", 1, 1024);
            if (config->mkl_n_threads < 0) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            
        /* Permutation test options */
        } else if (strcmp(argv[i], "--run-perm") == 0) { 
            config->run_permutations = 1;
            
        } else if (strcmp(argv[i], "--num-perm") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --num-perm\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->num_permutations = load_positive_value(argv[i], "--num-perm", 1, 10000000);
            if (config->num_permutations < 0) { 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->run_permutations = 1;
            
        } else if (strcmp(argv[i], "--perm-seed") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --perm-seed\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            long seed_val = strtol(argv[i], NULL, 10); 
            if (errno == ERANGE || seed_val < 0 || (unsigned long)seed_val > UINT_MAX) { 
                fprintf(stderr, "Error: Invalid seed value for --perm-seed '%s'.\n", argv[i]); 
                return MORANS_I_ERROR_PARAMETER;
            }
            config->perm_seed = (unsigned int)seed_val;
            config->run_permutations = 1;
            
        } else if (strcmp(argv[i], "--perm-out-z") == 0) {
            if (i + 1 < argc && (strcmp(argv[i + 1], "0") == 0 || strcmp(argv[i + 1], "1") == 0)) {
                i++;
                config->perm_output_zscores = load_positive_value(argv[i], "--perm-out-z", 0, 1);
                if (config->perm_output_zscores < 0) return MORANS_I_ERROR_PARAMETER;
            } else {
                config->perm_output_zscores = 1;
            }
            config->run_permutations = 1;
            
        } else if (strcmp(argv[i], "--perm-out-p") == 0) {
            if (i + 1 < argc && (strcmp(argv[i + 1], "0") == 0 || strcmp(argv[i + 1], "1") == 0)) {
                i++;
                config->perm_output_pvalues = load_positive_value(argv[i], "--perm-out-p", 0, 1);
                if (config->perm_output_pvalues < 0) return MORANS_I_ERROR_PARAMETER;
            } else {
                config->perm_output_pvalues = 1;
            }
            config->run_permutations = 1;
            
        /* Residual analysis options */
        } else if (strcmp(argv[i], "--analysis-mode") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --analysis-mode\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->residual_config.analysis_mode = parse_analysis_mode(argv[i]);
            if (config->residual_config.analysis_mode < 0) {
                return MORANS_I_ERROR_PARAMETER;
            }
            
        } else if (strcmp(argv[i], "--celltype-file") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --celltype-file\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->celltype_file, argv[i], BUFFER_SIZE - 1);
            args->celltype_file[BUFFER_SIZE - 1] = '\0';
            if (config->residual_config.celltype_file) free(config->residual_config.celltype_file);
            config->residual_config.celltype_file = strdup(argv[i]);
            if (!config->residual_config.celltype_file) {
                perror("strdup celltype_file");
                return MORANS_I_ERROR_MEMORY;
            }
            config->residual_config.analysis_mode = ANALYSIS_MODE_RESIDUAL;
            
        } else if (strcmp(argv[i], "--celltype-format") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --celltype-format\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->residual_config.celltype_format = parse_celltype_format(argv[i]);
            if (config->residual_config.celltype_format < 0) {
                return MORANS_I_ERROR_PARAMETER;
            }
            
        } else if (strcmp(argv[i], "--celltype-id-col") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --celltype-id-col\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->celltype_id_col, argv[i], BUFFER_SIZE - 1);
            args->celltype_id_col[BUFFER_SIZE - 1] = '\0';
            if (config->residual_config.celltype_id_col) free(config->residual_config.celltype_id_col);
            config->residual_config.celltype_id_col = strdup(argv[i]);
            if (!config->residual_config.celltype_id_col) {
                perror("strdup celltype_id_col");
                return MORANS_I_ERROR_MEMORY;
            }
            
        } else if (strcmp(argv[i], "--celltype-type-col") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --celltype-type-col\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->celltype_type_col, argv[i], BUFFER_SIZE - 1);
            args->celltype_type_col[BUFFER_SIZE - 1] = '\0';
            if (config->residual_config.celltype_type_col) free(config->residual_config.celltype_type_col);
            config->residual_config.celltype_type_col = strdup(argv[i]);
            if (!config->residual_config.celltype_type_col) {
                perror("strdup celltype_type_col");
                return MORANS_I_ERROR_MEMORY;
            }
            
        } else if (strcmp(argv[i], "--celltype-x-col") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --celltype-x-col\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->celltype_x_col, argv[i], BUFFER_SIZE - 1);
            args->celltype_x_col[BUFFER_SIZE - 1] = '\0';
            if (config->residual_config.celltype_x_col) free(config->residual_config.celltype_x_col);
            config->residual_config.celltype_x_col = strdup(argv[i]);
            if (!config->residual_config.celltype_x_col) {
                perror("strdup celltype_x_col");
                return MORANS_I_ERROR_MEMORY;
            }
            
        } else if (strcmp(argv[i], "--celltype-y-col") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --celltype-y-col\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->celltype_y_col, argv[i], BUFFER_SIZE - 1);
            args->celltype_y_col[BUFFER_SIZE - 1] = '\0';
            if (config->residual_config.celltype_y_col) free(config->residual_config.celltype_y_col);
            config->residual_config.celltype_y_col = strdup(argv[i]);
            if (!config->residual_config.celltype_y_col) {
                perror("strdup celltype_y_col");
                return MORANS_I_ERROR_MEMORY;
            }
            
        } else if (strcmp(argv[i], "--spot-id-col") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --spot-id-col\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            strncpy(args->spot_id_col, argv[i], BUFFER_SIZE - 1);
            args->spot_id_col[BUFFER_SIZE - 1] = '\0';
            if (config->residual_config.spot_id_col) free(config->residual_config.spot_id_col);
            config->residual_config.spot_id_col = strdup(argv[i]);
            if (!config->residual_config.spot_id_col) {
                perror("strdup spot_id_col");
                return MORANS_I_ERROR_MEMORY;
            }
            
        } else if (strcmp(argv[i], "--include-intercept") == 0) {
            if (i + 1 < argc && (strcmp(argv[i + 1], "0") == 0 || strcmp(argv[i + 1], "1") == 0)) {
                i++;
                config->residual_config.include_intercept = load_positive_value(argv[i], "--include-intercept", 0, 1);
                if (config->residual_config.include_intercept < 0) return MORANS_I_ERROR_PARAMETER;
            } else {
                config->residual_config.include_intercept = 1;
            }
            
        } else if (strcmp(argv[i], "--regularization") == 0) {
            if (++i >= argc) { 
                fprintf(stderr, "Error: Missing value for --regularization\n"); 
                return MORANS_I_ERROR_PARAMETER; 
            }
            config->residual_config.regularization_lambda = load_double_value(argv[i], "--regularization");
            if (isnan(config->residual_config.regularization_lambda) || 
                config->residual_config.regularization_lambda < 0) {
                fprintf(stderr, "Error: --regularization must be a non-negative number\n");
                return MORANS_I_ERROR_PARAMETER;
            }
            
        } else if (strcmp(argv[i], "--normalize-residuals") == 0) {
            if (i + 1 < argc && (strcmp(argv[i + 1], "0") == 0 || strcmp(argv[i + 1], "1") == 0)) {
                i++;
                config->residual_config.normalize_residuals = load_positive_value(argv[i], "--normalize-residuals", 0, 1);
                if (config->residual_config.normalize_residuals < 0) return MORANS_I_ERROR_PARAMETER;
            } else {
                config->residual_config.normalize_residuals = 1;
            }
            
        } else {
            fprintf(stderr, "Error: Unknown parameter '%s'. Use -h for help.\n", argv[i]);
            return MORANS_I_ERROR_PARAMETER;
        }
    }
    
    return MORANS_I_SUCCESS;
}

/* Validate configuration and file accessibility */
static int validate_and_initialize_config(MoransIConfig* config, const CommandLineArgs* args) {
    if (!config || !args) {
        fprintf(stderr, "Error: NULL parameters in validate_and_initialize_config\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    /* Initialize MKL/OpenMP environment */
    if (initialize_morans_i(config) != MORANS_I_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize Moran's I MKL/OpenMP environment.\n");
        return MORANS_I_ERROR_COMPUTATION;
    }
    
    /* Validate output prefix */
    if (strlen(args->output_prefix) == 0) { 
        fprintf(stderr, "Error: Output file prefix (-o) must be specified.\n"); 
        return MORANS_I_ERROR_PARAMETER;
    }

    /* Handle custom weights settings */
    if (config->custom_weights_file && config->platform_mode != CUSTOM_WEIGHTS) {
        printf("Info: Custom weight file provided, setting platform mode to CUSTOM_WEIGHTS\n");
        config->platform_mode = CUSTOM_WEIGHTS;
    }

    /* Validate custom weight matrix settings */
    if (config->platform_mode == CUSTOM_WEIGHTS) {
        if (!config->custom_weights_file) {
            fprintf(stderr, "Error: Custom weights platform mode requires -w <weight_file>\n");
            return MORANS_I_ERROR_PARAMETER;
        }
        if (access(config->custom_weights_file, R_OK) != 0) {
            fprintf(stderr, "Error: Cannot access custom weight matrix file '%s': %s\n", 
                    config->custom_weights_file, strerror(errno));
            return MORANS_I_ERROR_FILE;
        }
    }
    
    if (config->normalize_weights && config->platform_mode != CUSTOM_WEIGHTS) {
        fprintf(stderr, "Warning: --normalize-weights only applies to custom weight matrices\n");
    }

    if (config->row_normalize_weights && config->platform_mode != CUSTOM_WEIGHTS) {
        fprintf(stderr, "Warning: --row-normalize only applies to custom weight matrices\n");
    }

    /* Handle metadata file requirements for single cell mode */
    if (args->use_metadata_file && config->platform_mode != SINGLE_CELL && config->platform_mode != CUSTOM_WEIGHTS) {
        printf("Info: Metadata file provided, setting platform mode to SINGLE_CELL\n");
        config->platform_mode = SINGLE_CELL;
    }

    /* For non-toy runs, validate input file */
    if (args->run_toy_example == 0) {
        if (strlen(args->input_file) == 0) { 
            fprintf(stderr, "Error: Input file (-i) must be specified for standard run.\n"); 
            return MORANS_I_ERROR_PARAMETER;
        }
        if (access(args->input_file, R_OK) != 0) { 
            fprintf(stderr, "Error: Cannot access input file '%s': %s\n", args->input_file, strerror(errno)); 
            return MORANS_I_ERROR_FILE;
        }
        
        /* Validate metadata file if provided */
        if (args->use_metadata_file) {
            if (strlen(args->meta_file) == 0) { 
                fprintf(stderr, "Error: Metadata file path is empty despite -c being used.\n"); 
                return MORANS_I_ERROR_PARAMETER;
            }
            if (access(args->meta_file, R_OK) != 0) { 
                fprintf(stderr, "Error: Cannot access metadata file '%s': %s\n", args->meta_file, strerror(errno)); 
                return MORANS_I_ERROR_FILE;
            }
        } else {
            if (config->platform_mode == SINGLE_CELL) { 
                fprintf(stderr, "Error: Platform mode SINGLE_CELL selected, but no metadata file (-c) provided.\n"); 
                return MORANS_I_ERROR_PARAMETER;
            }
        }

        /* Validate cell type file for residual analysis */
        if (config->residual_config.analysis_mode == ANALYSIS_MODE_RESIDUAL) {
            if (!config->residual_config.celltype_file || strlen(args->celltype_file) == 0) {
                fprintf(stderr, "Error: Residual analysis mode requires --celltype-file\n");
                return MORANS_I_ERROR_PARAMETER;
            }
            if (access(config->residual_config.celltype_file, R_OK) != 0) {
                fprintf(stderr, "Error: Cannot access cell type file '%s': %s\n", 
                        config->residual_config.celltype_file, strerror(errno));
                return MORANS_I_ERROR_FILE;
            }
        }
    }
    
    return MORANS_I_SUCCESS;
}

/* Load and process expression data */
static int load_and_process_expression_data(const char* input_file, AnalysisResources* resources) {
    if (!input_file || !resources) {
        fprintf(stderr, "Error: NULL parameters in load_and_process_expression_data\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    double start_time, end_time;
    
    printf("Loading gene expression data from %s...\n", input_file);
    start_time = get_time();
    resources->vst_matrix = read_vst_file(input_file);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "gene expression data loading");
    
    if (!resources->vst_matrix) { 
        fprintf(stderr, "Error: Failed to load gene expression data.\n"); 
        return MORANS_I_ERROR_FILE;
    }
    
    printf("Loaded data matrix: %lld genes x %lld spots/cells.\n", 
           (long long)resources->vst_matrix->nrows, (long long)resources->vst_matrix->ncols);
    
    if (resources->vst_matrix->nrows == 0 || resources->vst_matrix->ncols == 0) { 
        fprintf(stderr, "Error: Loaded expression matrix is empty.\n"); 
        return MORANS_I_ERROR_FILE;
    }

    /* Clean non-finite values */
    MKL_INT total_elements = resources->vst_matrix->nrows * resources->vst_matrix->ncols; 
    MKL_INT non_finite_count = 0; 
    #pragma omp parallel for reduction(+:non_finite_count)
    for (MKL_INT i = 0; i < total_elements; i++) {
        if (!isfinite(resources->vst_matrix->values[i])) { 
            resources->vst_matrix->values[i] = 0.0; 
            non_finite_count++;
        }
    }
    if (non_finite_count > 0) {
        printf("Warning: Found and replaced %lld non-finite values with 0.0 in expression data.\n", 
               (long long)non_finite_count);
    }

    /* Z-normalize the data */
    printf("Performing Z-normalization (gene-wise)...\n");
    start_time = get_time();
    resources->znorm_matrix = z_normalize(resources->vst_matrix);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "Z-normalization");
    
    /* Free VST matrix as we no longer need it */
    free_dense_matrix(resources->vst_matrix);
    resources->vst_matrix = NULL;
    
    if (!resources->znorm_matrix) { 
        fprintf(stderr, "Error: Z-normalization failed.\n"); 
        return MORANS_I_ERROR_COMPUTATION;
    }
    
    return MORANS_I_SUCCESS;
}

/* Load cell type data for residual analysis */
static int load_celltype_data(const MoransIConfig* config, const CommandLineArgs* args, AnalysisResources* resources) {
    if (!config || !args || !resources) {
        fprintf(stderr, "Error: NULL parameters in load_celltype_data\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (config->residual_config.analysis_mode != ANALYSIS_MODE_RESIDUAL) {
        return MORANS_I_SUCCESS; // Not needed for standard analysis
    }

    if (!config->residual_config.celltype_file) {
        fprintf(stderr, "Error: Cell type file required for residual analysis\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    double start_time, end_time;
    printf("Loading cell type data for residual analysis from %s...\n", config->residual_config.celltype_file);
    start_time = get_time();

    if (config->residual_config.celltype_format == CELLTYPE_FORMAT_SINGLE_CELL) {
        resources->celltype_matrix = read_celltype_singlecell_file(
            config->residual_config.celltype_file,
            config->residual_config.celltype_id_col,
            config->residual_config.celltype_type_col,
            config->residual_config.celltype_x_col,
            config->residual_config.celltype_y_col
        );
    } else {
        resources->celltype_matrix = read_celltype_deconvolution_file(
            config->residual_config.celltype_file,
            config->residual_config.spot_id_col
        );
    }

    end_time = get_time();
    print_elapsed_time(start_time, end_time, "cell type data loading");

    if (!resources->celltype_matrix) {
        fprintf(stderr, "Error: Failed to load cell type data\n");
        return MORANS_I_ERROR_FILE;
    }

    printf("Loaded cell type matrix: %lld spots/cells x %lld cell types\n",
           (long long)resources->celltype_matrix->nrows, (long long)resources->celltype_matrix->ncols);

    /* Validate cell type matrix against expression data */
    if (validate_celltype_matrix(resources->celltype_matrix, resources->znorm_matrix) != MORANS_I_SUCCESS) {
        fprintf(stderr, "Error: Cell type matrix validation failed\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    return MORANS_I_SUCCESS;
}

/* Setup spatial analysis components */
static int setup_spatial_analysis(const MoransIConfig* config, const CommandLineArgs* args, 
                                 AnalysisResources* resources) {
    if (!config || !args || !resources || !resources->znorm_matrix) {
        fprintf(stderr, "Error: NULL parameters in setup_spatial_analysis\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    /* Skip coordinate processing for custom weights mode */
    if (config->platform_mode != CUSTOM_WEIGHTS) {
        return setup_builtin_spatial_weights(config, args, resources);
    } else {
        return setup_custom_spatial_weights(config, resources);
    }
}

/* Setup spatial analysis with built-in weight calculation */
static int setup_builtin_spatial_weights(const MoransIConfig* config, const CommandLineArgs* args, 
                                        AnalysisResources* resources) {
    double start_time, end_time;
    
    printf("Preparing spatial coordinate data...\n");
    start_time = get_time();
    if (args->use_metadata_file) {
        resources->spot_coords = read_coordinates_file(args->meta_file, args->id_column, 
                                                      args->x_column, args->y_column, 
                                                      config->coord_scale);
    } else {
        resources->spot_coords = extract_coordinates(resources->znorm_matrix->colnames, 
                                                    resources->znorm_matrix->ncols);
    }
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "spatial coordinate processing");
    
    if (!resources->spot_coords) { 
        fprintf(stderr, "Error: Failed to obtain or process spot coordinates.\n"); 
        return MORANS_I_ERROR_COMPUTATION;
    }
    
    printf("Processed %lld total spot coordinate entries, %lld are valid.\n", 
           (long long)resources->spot_coords->total_spots, 
           (long long)resources->spot_coords->valid_spots);
    
    if (resources->spot_coords->valid_spots == 0) { 
        fprintf(stderr, "Error: No valid spot coordinates found. Cannot proceed.\n"); 
        return MORANS_I_ERROR_COMPUTATION;
    }

    /* Determine sigma for decay function */
    double sigma_for_decay = args->custom_sigma; 
    if (config->platform_mode == SINGLE_CELL && sigma_for_decay <= 0.0) {
        printf("Inferring sigma for RBF kernel from single-cell coordinate data...\n");
        sigma_for_decay = infer_sigma_from_data(resources->spot_coords, config->coord_scale);
        if (sigma_for_decay <= 0) { 
            fprintf(stderr, "Warning: Failed to infer positive sigma (got %.2f), using default of 50.0 for SC.\n", 
                    sigma_for_decay); 
            sigma_for_decay = 50.0;
        }
    }
    
    /* Prepare helper arrays for valid spots */
    if (prepare_valid_spot_arrays(resources) != MORANS_I_SUCCESS) {
        return MORANS_I_ERROR_MEMORY;
    }
    
    /* Map coordinates to expression data */
    if (map_spots_to_expression(resources) != MORANS_I_SUCCESS) {
        return MORANS_I_ERROR_COMPUTATION;
    }

    /* Create distance decay matrix */
    printf("Creating distance decay matrix (max_radius_grid_units=%d)...\n", config->max_radius);
    start_time = get_time();
    resources->decay_matrix = create_distance_matrix(config->max_radius, config->platform_mode, 
                                                    sigma_for_decay, config->coord_scale);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "distance decay matrix creation");
    
    if (!resources->decay_matrix) { 
        fprintf(stderr, "Error: Failed to create distance decay lookup matrix.\n"); 
        return MORANS_I_ERROR_COMPUTATION;
    }
    
    if (!config->include_same_spot && resources->decay_matrix->nrows > 0 && resources->decay_matrix->ncols > 0) {
        printf("Excluding self-comparisons: setting decay_matrix[0,0] to 0.0.\n");
        resources->decay_matrix->values[0] = 0.0;
    }

    /* Build spatial weight matrix */
    printf("Building spatial weight matrix W for %lld valid spots...\n", (long long)resources->num_valid_spots);
    start_time = get_time();
    resources->W_matrix = build_spatial_weight_matrix(resources->valid_spot_rows, 
                                                     resources->valid_spot_cols, 
                                                     resources->num_valid_spots, 
                                                     resources->decay_matrix, 
                                                     config->max_radius,
                                                     config->row_normalize_weights);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "spatial weight matrix W construction");
    
    /* Free decay matrix as we no longer need it */
    free_dense_matrix(resources->decay_matrix);
    resources->decay_matrix = NULL;
    
    if (!resources->W_matrix) { 
        fprintf(stderr, "Error: Failed to build spatial weight matrix W.\n"); 
        return MORANS_I_ERROR_COMPUTATION;
    }

    /* Prepare final calculation matrix */
    if (prepare_calculation_matrix_builtin(resources) != MORANS_I_SUCCESS) {
        return MORANS_I_ERROR_COMPUTATION;
    }
    
    return MORANS_I_SUCCESS;
}

/* Setup spatial analysis with custom weights */
static int setup_custom_spatial_weights(const MoransIConfig* config, AnalysisResources* resources) {
    double start_time, end_time;
    
    printf("Using custom weight matrix mode - preparing calculation matrix...\n");
    start_time = get_time();
    
    /* Create X_calc by transposing znorm_matrix from genes x spots to spots x genes */
    if (prepare_calculation_matrix_custom(resources) != MORANS_I_SUCCESS) {
        return MORANS_I_ERROR_MEMORY;
    }
    
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "matrix transposition for custom weights");
    
    /* Free znorm matrix as we no longer need it */
    free_dense_matrix(resources->znorm_matrix);
    resources->znorm_matrix = NULL;
    
    /* Load custom weight matrix */
    printf("Loading custom weight matrix from %s...\n", config->custom_weights_file);
    start_time = get_time();
    
    MKL_INT n_spots_for_weights;
    char** spot_names_for_weights = extract_spot_names_from_expression(resources->X_calc, &n_spots_for_weights);
    
    if (!spot_names_for_weights || n_spots_for_weights != resources->X_calc->nrows) {
        fprintf(stderr, "Error: Failed to extract spot names for weight matrix mapping\n");
        if (spot_names_for_weights) free(spot_names_for_weights);
        return MORANS_I_ERROR_COMPUTATION;
    }
    
    resources->W_matrix = read_custom_weight_matrix(config->custom_weights_file, 
                                                   config->weight_format,
                                                   spot_names_for_weights, 
                                                   n_spots_for_weights);
    
    free(spot_names_for_weights);  // Only free the array, not the strings (they're references)
    
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "custom weight matrix loading");
    
    if (!resources->W_matrix) {
        fprintf(stderr, "Error: Failed to load custom weight matrix.\n");
        return MORANS_I_ERROR_FILE;
    }
    
    /* Optionally normalize weights */
    if (config->normalize_weights) {
        double S0_before = calculate_weight_sum(resources->W_matrix);
        if (fabs(S0_before) > ZERO_STD_THRESHOLD) {
            printf("Normalizing custom weights (S0_before = %.6f)...\n", S0_before);
            for (MKL_INT i = 0; i < resources->W_matrix->nnz; i++) {
                resources->W_matrix->values[i] /= S0_before;
            }
            double S0_after = calculate_weight_sum(resources->W_matrix);
            printf("Weights normalized (S0_after = %.6f)\n", S0_after);
        } else {
            fprintf(stderr, "Warning: Cannot normalize weights with S0 near zero (%.6e)\n", S0_before);
        }
    }
    
    return MORANS_I_SUCCESS;
}

/* Helper functions for spatial setup */
static int prepare_valid_spot_arrays(AnalysisResources* resources) {
    resources->num_valid_spots = resources->spot_coords->valid_spots;
    
    resources->valid_spot_indices = (MKL_INT*)malloc((size_t)resources->num_valid_spots * sizeof(MKL_INT));
    resources->valid_spot_rows = (MKL_INT*)malloc((size_t)resources->num_valid_spots * sizeof(MKL_INT));
    resources->valid_spot_cols = (MKL_INT*)malloc((size_t)resources->num_valid_spots * sizeof(MKL_INT));
    resources->valid_spot_names = (char**)calloc(resources->num_valid_spots, sizeof(char*));
    
    if (!resources->valid_spot_indices || !resources->valid_spot_rows || 
        !resources->valid_spot_cols || !resources->valid_spot_names) {
        perror("Memory allocation failed for valid spot helper arrays");
        return MORANS_I_ERROR_MEMORY;
    }
    
    return MORANS_I_SUCCESS;
}

static int map_spots_to_expression(AnalysisResources* resources) {
    MKL_INT v_idx = 0;
    
    for (MKL_INT i = 0; i < resources->spot_coords->total_spots; i++) { 
        if (resources->spot_coords->valid_mask[i]) {
            if (v_idx >= resources->num_valid_spots) { 
                fprintf(stderr, "Error: Index v_idx out of bounds for helper arrays.\n"); 
                return MORANS_I_ERROR_COMPUTATION;
            }
            
            resources->valid_spot_rows[v_idx] = resources->spot_coords->spot_row[i];
            resources->valid_spot_cols[v_idx] = resources->spot_coords->spot_col[i];
            resources->valid_spot_names[v_idx] = strdup(resources->spot_coords->spot_names[i]);
            
            if (!resources->valid_spot_names[v_idx]) { 
                perror("strdup for valid_spot_names_list"); 
                return MORANS_I_ERROR_MEMORY;
            }
            
            /* Find corresponding expression column */
            int expr_col_idx = -1; 
            for (MKL_INT j = 0; j < resources->znorm_matrix->ncols; j++) {
                if (strcmp(resources->spot_coords->spot_names[i], resources->znorm_matrix->colnames[j]) == 0) { 
                    expr_col_idx = j; 
                    break;
                }
            }
            resources->valid_spot_indices[v_idx] = expr_col_idx;
            
            if (expr_col_idx == -1) {
                fprintf(stderr, "Warning: Valid coordinate spot '%s' not found in expression matrix colnames.\n", 
                        resources->spot_coords->spot_names[i]);
            }
            v_idx++;
        }
    }
    
    if (v_idx != resources->num_valid_spots) {
        fprintf(stderr, "Warning: Actual populated valid spots (%lld) differs from initial count (%lld). Adjusting.\n", 
                (long long)v_idx, (long long)resources->num_valid_spots);
        resources->num_valid_spots = v_idx; 
        if (resources->num_valid_spots == 0) { 
            fprintf(stderr, "Error: No valid spots remain. Cannot proceed.\n"); 
            return MORANS_I_ERROR_COMPUTATION; 
        }
    }
    
    return MORANS_I_SUCCESS;
}

static int prepare_calculation_matrix_builtin(AnalysisResources* resources) {
    double start_time, end_time;
    
    printf("Preparing final calculation matrix X_calc (Valid_Spots x Genes)...\n");
    start_time = get_time();
    
    resources->X_calc = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!resources->X_calc) { 
        perror("malloc X_calc struct"); 
        return MORANS_I_ERROR_MEMORY; 
    }
    
    resources->X_calc->nrows = resources->num_valid_spots; 
    resources->X_calc->ncols = resources->znorm_matrix->nrows;
    resources->X_calc->values = (double*)mkl_malloc((size_t)resources->X_calc->nrows * resources->X_calc->ncols * sizeof(double), 64);
    resources->X_calc->rownames = (char**)calloc(resources->X_calc->nrows, sizeof(char*)); 
    resources->X_calc->colnames = (char**)calloc(resources->X_calc->ncols, sizeof(char*)); 
    
    if (!resources->X_calc->values || !resources->X_calc->rownames || !resources->X_calc->colnames) {
        perror("Memory allocation failed for X_calc components"); 
        return MORANS_I_ERROR_MEMORY;
    }

    /* Copy gene names */
    for (MKL_INT g = 0; g < resources->X_calc->ncols; g++) { 
        resources->X_calc->colnames[g] = strdup(resources->znorm_matrix->rownames[g]); 
        if (!resources->X_calc->colnames[g]) {
            perror("strdup X_calc gene name"); 
            return MORANS_I_ERROR_MEMORY;
        }
    }
    
    /* Copy spot names */
    for (MKL_INT i = 0; i < resources->X_calc->nrows; i++) { 
        resources->X_calc->rownames[i] = strdup(resources->valid_spot_names[i]); 
        if (!resources->X_calc->rownames[i]) {
            perror("strdup X_calc spot name"); 
            return MORANS_I_ERROR_MEMORY;
        }
    }

    /* Copy expression data for valid spots */
    #pragma omp parallel for
    for (MKL_INT i = 0; i < resources->X_calc->nrows; i++) { 
        MKL_INT original_expr_col_idx = resources->valid_spot_indices[i]; 
        for (MKL_INT j = 0; j < resources->X_calc->ncols; j++) { 
            if (original_expr_col_idx != -1) {
                resources->X_calc->values[i * resources->X_calc->ncols + j] = 
                    resources->znorm_matrix->values[j * resources->znorm_matrix->ncols + original_expr_col_idx];
            } else {
                resources->X_calc->values[i * resources->X_calc->ncols + j] = 0.0; 
            }
        }
    }
    
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "final X_calc matrix preparation");

    /* Clean up intermediate data structures */
    free_dense_matrix(resources->znorm_matrix);
    resources->znorm_matrix = NULL;
    free_spot_coordinates(resources->spot_coords);
    resources->spot_coords = NULL;
    
    return MORANS_I_SUCCESS;
}

static int prepare_calculation_matrix_custom(AnalysisResources* resources) {
    /* Transpose the gene x spots matrix to spots x genes for X_calc */
    resources->X_calc = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    if (!resources->X_calc) { 
        perror("malloc X_calc struct"); 
        return MORANS_I_ERROR_MEMORY; 
    }
    
    resources->X_calc->nrows = resources->znorm_matrix->ncols; // spots
    resources->X_calc->ncols = resources->znorm_matrix->nrows; // genes
    resources->X_calc->values = (double*)mkl_malloc((size_t)resources->X_calc->nrows * resources->X_calc->ncols * sizeof(double), 64);
    resources->X_calc->rownames = (char**)calloc(resources->X_calc->nrows, sizeof(char*)); // spot names
    resources->X_calc->colnames = (char**)calloc(resources->X_calc->ncols, sizeof(char*)); // gene names
    
    if (!resources->X_calc->values || !resources->X_calc->rownames || !resources->X_calc->colnames) {
        perror("Memory allocation failed for X_calc components"); 
        return MORANS_I_ERROR_MEMORY;
    }

    /* Copy gene names (columns in X_calc) */
    for (MKL_INT g = 0; g < resources->X_calc->ncols; g++) {
        resources->X_calc->colnames[g] = strdup(resources->znorm_matrix->rownames[g]);
        if (!resources->X_calc->colnames[g]) {
            perror("strdup X_calc gene name"); 
            return MORANS_I_ERROR_MEMORY;
        }
    }
    
    /* Copy spot names (rows in X_calc) */
    for (MKL_INT s = 0; s < resources->X_calc->nrows; s++) {
        resources->X_calc->rownames[s] = strdup(resources->znorm_matrix->colnames[s]);
        if (!resources->X_calc->rownames[s]) {
            perror("strdup X_calc spot name"); 
            return MORANS_I_ERROR_MEMORY;
        }
    }
    
    /* Transpose data: znorm_matrix[gene][spot] -> X_calc[spot][gene] */
    #pragma omp parallel for
    for (MKL_INT s = 0; s < resources->X_calc->nrows; s++) {
        for (MKL_INT g = 0; g < resources->X_calc->ncols; g++) {
            resources->X_calc->values[s * resources->X_calc->ncols + g] = 
                resources->znorm_matrix->values[g * resources->znorm_matrix->ncols + s];
        }
    }
    
    return MORANS_I_SUCCESS;
}

/* Run Moran's I analysis based on configuration */
static int run_moran_analysis(const MoransIConfig* config, AnalysisResources* resources, 
                             const char* output_prefix) {
    if (!config || !resources || !output_prefix || !resources->X_calc || !resources->W_matrix) {
        fprintf(stderr, "Error: NULL parameters in run_moran_analysis\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    double start_time, end_time;
    double S0_val = calculate_weight_sum(resources->W_matrix); 
    
    printf("Sum of all weights S0 = %.6f (from %lld NNZ in W)\n", 
           S0_val, (long long)resources->W_matrix->nnz);
    
    if (fabs(S0_val) < DBL_EPSILON && resources->W_matrix->nnz > 0) {
        fprintf(stderr, "Warning: S0 is near-zero. Moran's I results will likely be 0, NaN, or Inf.\n");
    }
    if (resources->W_matrix->nnz == 0 && resources->X_calc->nrows > 0) {
        fprintf(stderr, "Warning: Spatial Weight Matrix W has no non-zero elements.\n");
    }

    printf("Calculating Moran's I based on selected mode%s...\n", 
           config->row_normalize_weights ? " (with row-normalized weights)" : "");
    start_time = get_time();
    
    char result_filename[BUFFER_SIZE]; 
    int status = MORANS_I_SUCCESS;

    if (!config->calc_pairwise) { 
        /* Single-gene mode */
        snprintf(result_filename, BUFFER_SIZE, "%s_single_gene_moran_i.tsv", output_prefix);
        printf("Mode: Single-Gene Moran's I. Output: %s\n", result_filename);
        status = save_single_gene_results(resources->X_calc, resources->W_matrix, S0_val, result_filename, config->row_normalize_weights);
        
    } else if (!config->calc_all_vs_all) { 
        /* First gene vs all mode */
        snprintf(result_filename, BUFFER_SIZE, "%s_first_vs_all_moran_i.tsv", output_prefix);
        printf("Mode: Pairwise Moran's I (First Gene vs All Others). Output: %s\n", result_filename);
        
        if (resources->X_calc->ncols == 0) { 
            fprintf(stderr, "Error: No genes in X_calc for first-vs-all Moran's I.\n"); 
            status = MORANS_I_ERROR_PARAMETER;
        } else {
            double* first_vs_all_results = calculate_first_gene_vs_all(resources->X_calc, resources->W_matrix, S0_val, config->row_normalize_weights); 
            if (first_vs_all_results) {
                status = save_first_gene_vs_all_results(first_vs_all_results, 
                                                       (const char**)resources->X_calc->colnames, 
                                                       resources->X_calc->ncols, result_filename);
                mkl_free(first_vs_all_results);
            } else { 
                fprintf(stderr, "Error: Failed to calculate Moran's I for first gene vs all others.\n"); 
                status = MORANS_I_ERROR_COMPUTATION;
            }
        }
        
    } else { 
        /* All pairs mode */
        snprintf(result_filename, BUFFER_SIZE, "%s_all_pairs_moran_i_raw.tsv", output_prefix);
        printf("Mode: Pairwise Moran's I (All Gene Pairs - Raw Lower Triangular). Output: %s\n", result_filename);
        
        resources->observed_results = calculate_morans_i(resources->X_calc, resources->W_matrix, config->row_normalize_weights);
        if (resources->observed_results) {
            status = save_lower_triangular_matrix_raw(resources->observed_results, result_filename);
        } else {
            fprintf(stderr, "Error: Failed to calculate all-pairs Moran's I.\n");
            status = MORANS_I_ERROR_COMPUTATION;
        }
    }
    
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "Observed Moran's I calculation and saving");
    
    return status;
}

/* Run residual Moran's I analysis */
static int run_residual_analysis(const MoransIConfig* config, AnalysisResources* resources, 
                                const char* output_prefix) {
    if (!config || !resources || !output_prefix) {
        fprintf(stderr, "Error: NULL parameters in run_residual_analysis\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    if (config->residual_config.analysis_mode != ANALYSIS_MODE_RESIDUAL) {
        return MORANS_I_SUCCESS; // Not running residual analysis
    }

    if (!resources->celltype_matrix || !resources->X_calc || !resources->W_matrix) {
        fprintf(stderr, "Error: Missing required data for residual analysis\n");
        return MORANS_I_ERROR_PARAMETER;
    }

    double start_time, end_time;
    printf("--- Running Residual Moran's I Analysis ---\n");

    /* Map cell type matrix to expression matrix spots if needed */
    CellTypeMatrix* mapped_celltype = NULL;
    if (map_celltype_to_expression(resources->celltype_matrix, resources->X_calc, &mapped_celltype) != MORANS_I_SUCCESS) {
        fprintf(stderr, "Error: Failed to map cell type data to expression matrix\n");
        return MORANS_I_ERROR_COMPUTATION;
    }

    start_time = get_time();
    resources->residual_results = calculate_residual_morans_i(resources->X_calc, mapped_celltype, 
                                                            resources->W_matrix, &config->residual_config);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "Residual Moran's I calculation");

    if (!resources->residual_results) {
        fprintf(stderr, "Error: Failed to calculate residual Moran's I\n");
        if (mapped_celltype != resources->celltype_matrix) {
            free_celltype_matrix(mapped_celltype);
        }
        return MORANS_I_ERROR_COMPUTATION;
    }

    /* Save residual analysis results */
    start_time = get_time();
    int save_status = save_residual_results(resources->residual_results, output_prefix);
    end_time = get_time();
    print_elapsed_time(start_time, end_time, "Saving residual analysis results");

    if (mapped_celltype != resources->celltype_matrix) {
        free_celltype_matrix(mapped_celltype);
    }

    if (save_status != MORANS_I_SUCCESS) {
        fprintf(stderr, "Error saving residual analysis results\n");
        return save_status;
    }

    printf("Residual Moran's I analysis completed successfully\n");
    return MORANS_I_SUCCESS;
}

/* Run permutation analysis if requested */
static int run_permutation_analysis(const MoransIConfig* config, AnalysisResources* resources, 
                                   const char* output_prefix) {
    if (!config || !resources || !output_prefix) {
        fprintf(stderr, "Error: NULL parameters in run_permutation_analysis\n");
        return MORANS_I_ERROR_PARAMETER;
    }
    
    if (!config->run_permutations) {
        return MORANS_I_SUCCESS; // Nothing to do
    }
    
    /* Check if we should run standard or residual permutation testing */
    if (config->residual_config.analysis_mode == ANALYSIS_MODE_RESIDUAL) {
        /* Run residual permutation test */
        if (!resources->celltype_matrix || !resources->X_calc || !resources->W_matrix) {
            fprintf(stderr, "Warning: Missing required data for residual permutation test. Skipping.\n");
            return MORANS_I_SUCCESS;
        }

        double start_time, end_time;
        printf("--- Running Residual Permutation Test ---\n");
        
        PermutationParams perm_params; 
        perm_params.n_permutations = config->num_permutations;
        perm_params.seed = config->perm_seed; 
        perm_params.z_score_output = config->perm_output_zscores;
        perm_params.p_value_output = config->perm_output_pvalues;

        start_time = get_time();
        PermutationResults* residual_perm_results = run_residual_permutation_test(
            resources->X_calc, resources->celltype_matrix, resources->W_matrix, 
            &perm_params, &config->residual_config);
        end_time = get_time();
        print_elapsed_time(start_time, end_time, "Residual Permutation Test computation");

        if (residual_perm_results) {
            /* Store results in residual_results structure */
            if (resources->residual_results) {
                resources->residual_results->residual_zscores = residual_perm_results->z_scores;
                resources->residual_results->residual_pvalues = residual_perm_results->p_values;
                resources->residual_results->residual_mean_perm = residual_perm_results->mean_perm;
                resources->residual_results->residual_var_perm = residual_perm_results->var_perm;
                
                /* Clear pointers in the PermutationResults to avoid double-free */
                residual_perm_results->z_scores = NULL;
                residual_perm_results->p_values = NULL;
                residual_perm_results->mean_perm = NULL;
                residual_perm_results->var_perm = NULL;
            }
            
            free_permutation_results(residual_perm_results);
            
            /* Save updated residual results */
            start_time = get_time();
            int save_status = save_residual_results(resources->residual_results, output_prefix);
            end_time = get_time();
            print_elapsed_time(start_time, end_time, "Saving Residual Permutation Test results");
            
            if (save_status != MORANS_I_SUCCESS) {
                fprintf(stderr, "Error saving residual permutation results.\n");
                return save_status;
            }
        } else {
            fprintf(stderr, "Error: Residual permutation test failed to produce results.\n");
            return MORANS_I_ERROR_COMPUTATION;
        }
        
    } else {
        /* Run standard permutation test */
        if (!resources->observed_results || !config->calc_pairwise || !config->calc_all_vs_all) {
            if (!config->calc_pairwise || !config->calc_all_vs_all) {
                printf("Warning: Permutation testing is primarily designed for 'all gene pairs' mode (-b 1 -g 1).\n");
                printf("         Current mode: -b %d -g %d. Skipping permutations.\n", 
                       config->calc_pairwise, config->calc_all_vs_all);
            } else if (!resources->observed_results) {
                printf("Warning: Observed Moran's I calculation failed for all-pairs. Skipping permutations.\n");
            } else {
                printf("Warning: Permutation testing requested but prerequisites not met. Skipping permutations.\n");
            }
            return MORANS_I_SUCCESS;
        }
        
        double start_time, end_time;
        printf("--- Running Standard Permutation Test ---\n");
        
        PermutationParams perm_params; 
        perm_params.n_permutations = config->num_permutations;
        perm_params.seed = config->perm_seed; 
        perm_params.z_score_output = config->perm_output_zscores;
        perm_params.p_value_output = config->perm_output_pvalues;

        start_time = get_time();
        resources->perm_results = run_permutation_test(resources->X_calc, resources->W_matrix, &perm_params, config->row_normalize_weights);
        end_time = get_time();
        print_elapsed_time(start_time, end_time, "Standard Permutation Test computation");

        if (resources->perm_results) {
            start_time = get_time();
            int save_status = save_permutation_results(resources->perm_results, output_prefix);
            end_time = get_time();
            print_elapsed_time(start_time, end_time, "Saving Standard Permutation Test results");
            
            if (save_status != MORANS_I_SUCCESS) {
                fprintf(stderr, "Error saving standard permutation results.\n");
                return save_status;
            }
        } else {
            fprintf(stderr, "Error: Standard permutation test failed to produce results.\n");
            return MORANS_I_ERROR_COMPUTATION;
        }
    }
    
    return MORANS_I_SUCCESS;
}

/* Print configuration summary */
static void print_configuration_summary(const MoransIConfig* config, const CommandLineArgs* args) {
    printf("\nMoran's I calculation utility version %s\n\n", morans_i_mkl_version());
    printf("--- Parameters ---\n");
    printf("Input file: %s\n", args->input_file);
    printf("Output file prefix: %s\n", args->output_prefix);
    
    const char* analysis_mode_names[] = {"Standard", "Residual"};
    printf("Analysis Mode: %s\n", analysis_mode_names[config->residual_config.analysis_mode]);
    
    if (config->residual_config.analysis_mode == ANALYSIS_MODE_RESIDUAL) {
        printf("Cell Type File: %s\n", config->residual_config.celltype_file);
        const char* celltype_format_names[] = {"Deconvolution", "Single Cell"};
        printf("  Cell Type Format: %s\n", celltype_format_names[config->residual_config.celltype_format]);
        printf("  Include Intercept: %s\n", config->residual_config.include_intercept ? "Yes" : "No");
        printf("  Regularization Lambda: %.6f\n", config->residual_config.regularization_lambda);
        printf("  Normalize Residuals: %s\n", config->residual_config.normalize_residuals ? "Yes" : "No");
    }
    
    if (config->platform_mode == CUSTOM_WEIGHTS) {
        printf("Custom Weight Matrix File: %s\n", config->custom_weights_file);
        const char* format_names[] = {"auto", "dense", "sparse_coo", "sparse_tsv"};
        printf("  Weight Format: %s\n", format_names[config->weight_format]);
        printf("  Normalize Weights: %s\n", config->normalize_weights ? "Yes" : "No");
        printf("  Row Normalize Weights: %s\n", config->row_normalize_weights ? "Yes" : "No");
    } else {
        if (args->use_metadata_file) {
            printf("Metadata file: %s\n", args->meta_file);
            printf("  ID column: %s\n", args->id_column); 
            printf("  X coordinate column: %s\n", args->x_column); 
            printf("  Y coordinate column: %s\n", args->y_column);
            printf("  Coordinate scale factor: %.2f\n", config->coord_scale);
        }
        if (args->custom_sigma > 0) {
            printf("Custom Sigma for RBF: %.4f\n", args->custom_sigma);
        } else {
            printf("Custom Sigma for RBF: Not set (will use platform default or infer for SC).\n");
        }
        printf("Max radius (grid units): %d\n", config->max_radius);
        printf("Row Normalize Weights: %s\n", config->row_normalize_weights ? "Yes" : "No");
    }
    
    const char* platform_names[] = {"Visium", "Old ST", "Single Cell", "Custom Weights"};
    printf("Platform: %d (%s)\n", config->platform_mode, 
           config->platform_mode < 4 ? platform_names[config->platform_mode] : "Unknown");
    printf("Mode: %s\n", config->calc_pairwise ? "Pairwise Moran's I" : "Single-Gene Moran's I");
    if (config->calc_pairwise) {
        printf("  Gene Pairs: %s\n", config->calc_all_vs_all ? "All vs All" : "First Gene vs All Others");
    }
    printf("Include Self-Comparisons (w_ii): %s\n", config->include_same_spot ? "Yes" : "No");
    
    if (config->run_permutations) {
        printf("Permutation Testing: Enabled\n");
        printf("  Number of Permutations: %d\n", config->num_permutations);
        printf("  Permutation Seed: %u\n", config->perm_seed);
        printf("  Output Z-scores: %s\n", config->perm_output_zscores ? "Yes" : "No");
        printf("  Output P-values: %s\n", config->perm_output_pvalues ? "Yes" : "No");
    }
    printf("------------------\n");
}

/* Helper function implementations */
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
    printf("  -r <int>\tMaximum grid radius for neighbor search. Default: %d. (Ignored if using custom weights)\n", DEFAULT_MAX_RADIUS);
    printf("  -p <int>\tPlatform type (%d: Visium, %d: Older ST, %d: Single Cell, %d: Custom Weights). Default: %d.\n",
           VISIUM, OLD, SINGLE_CELL, CUSTOM_WEIGHTS, DEFAULT_PLATFORM_MODE);
    printf("  -b <0|1>\tCalculation mode: 0 = Single-gene, 1 = Pairwise. Default: %d.\n",
           DEFAULT_CALC_PAIRWISE);
    printf("  -g <0|1>\tGene selection (if -b 1): 0 = First gene vs all, 1 = All gene pairs. Default: %d.\n",
           DEFAULT_CALC_ALL_VS_ALL);
    printf("  -s <0|1>\tInclude self-comparison (w_ii)? 0 = No, 1 = Yes. Default: %d.\n", DEFAULT_INCLUDE_SAME_SPOT);
    printf("  --row-normalize <0|1>\tNormalize each row of weight matrix to sum to 1? 0 = No, 1 = Yes. Default: %d.\n",
           DEFAULT_ROW_NORMALIZE_WEIGHTS);
    printf("  -t <int>\tSet number of OpenMP threads. Default: %d (or OMP_NUM_THREADS).\n", DEFAULT_NUM_THREADS);
    printf("  -m <int>\tSet number of MKL threads. Default: Value of -t or OpenMP default.\n");
    printf("\nCustom Weight Matrix Options:\n");
    printf("  -w <file>\tCustom weight matrix file. Sets platform mode to CUSTOM_WEIGHTS (%d).\n", CUSTOM_WEIGHTS);
    printf("  --weight-format <format>\tWeight matrix format: auto (default), dense, sparse_coo, sparse_tsv.\n");
    printf("  --normalize-weights\tNormalize custom weights by dividing by sum (S0). Default: No.\n");
    printf("\nSingle-cell Specific Options:\n");
    printf("  -c <file>\tCoordinates/metadata file (TSV). Required for single-cell data.\n");
    printf("  --id-col <name>\tColumn name for cell IDs in metadata. Default: 'cell_ID'.\n");
    printf("  --x-col <name>\tColumn name for X coordinates in metadata. Default: 'sdimx'.\n");
    printf("  --y-col <name>\tColumn name for Y coordinates in metadata. Default: 'sdimy'.\n");
    printf("  --scale <float>\tScaling factor for SC coordinates to integer grid. Default: %.1f.\n", DEFAULT_COORD_SCALE_FACTOR);
    printf("  --sigma <float>\tCustom sigma for RBF kernel (physical units). If <=0, inferred for SC or platform default.\n");
    printf("\nResidual Moran's I Options:\n");
    printf("  --analysis-mode <standard|residual>\tAnalysis mode. Default: standard.\n");
    printf("  --celltype-file <file>\t\tCell type composition/annotation file.\n");
    printf("  --celltype-format <deconv|sc>\t\tFormat: deconvolution or single_cell. Default: single_cell.\n");
    printf("  --celltype-id-col <name>\t\tCell ID column name. Default: cell_ID.\n");
    printf("  --celltype-type-col <name>\t\tCell type column name. Default: cellType.\n");
    printf("  --celltype-x-col <name>\t\tX coordinate column name. Default: sdimx.\n");
    printf("  --celltype-y-col <name>\t\tY coordinate column name. Default: sdimy.\n");
    printf("  --spot-id-col <name>\t\t\tSpot ID column for deconvolution format. Default: spot_id.\n");
    printf("  --include-intercept <0|1>\t\tInclude intercept in regression. Default: 1.\n");
    printf("  --regularization <float>\t\tRidge regularization parameter. Default: 0.0.\n");
    printf("  --normalize-residuals <0|1>\t\tNormalize residuals. Default: 1.\n");
    printf("\nPermutation Test Options:\n");
    printf("  --run-perm\tEnable permutation testing.\n");
    printf("  --num-perm <int>\tNumber of permutations. Default: %d. Implies --run-perm.\n", DEFAULT_NUM_PERMUTATIONS);
    printf("  --perm-seed <int>\tSeed for RNG. Default: Based on system time. Implies --run-perm.\n");
    printf("  --perm-out-z <0|1>\tOutput Z-scores. Default: 1. Implies --run-perm.\n");
    printf("  --perm-out-p <0|1>\tOutput p-values. Default: 1. Implies --run-perm.\n");
    printf("\nToy Example Mode:\n");
    printf("  --run-toy-example\tRuns a small, built-in 2D grid (5x5) example to test functionality.\n"
           "                    \tRequires -o <prefix>. Permutation options can be used.\n");
    printf("\nOutput Format (files named based on <output_prefix>):\n");
    printf("  Single-gene (-b 0): <prefix>_single_gene_moran_i.tsv (Gene, MoranI).\n");
    printf("  Pairwise All (-b 1 -g 1): <prefix>_all_pairs_moran_i_raw.tsv (Observed Moran's I, Raw lower triangular).\n");
    printf("  Pairwise First (-b 1 -g 0): <prefix>_first_vs_all_moran_i.tsv (Gene, MoranI_vs_Gene0).\n");
    printf("  Standard permutation outputs (if enabled): <prefix>_zscores_lower_tri.tsv, <prefix>_pvalues_lower_tri.tsv\n");
    printf("  Residual analysis outputs: <prefix>_residual_morans_i_raw.tsv, <prefix>_regression_coefficients.tsv\n");
    printf("  Residual permutation outputs: <prefix>_residual_zscores_lower_tri.tsv, <prefix>_residual_pvalues_lower_tri.tsv\n");
    printf("\nExample:\n");
    printf("  %s -i expr.tsv -o run1 -r 3 -p 0 -b 1 -g 1 -t 8 --run-perm --num-perm 1000\n", program_name);
    printf("  %s -i expr.tsv -o run2 -w custom_weights.tsv --weight-format dense\n", program_name);
    printf("  %s -i expr.tsv -o run3 --analysis-mode residual --celltype-file celltypes.tsv\n", program_name);
    printf("  %s --run-toy-example -o toy_2d_run --num-perm 100 --perm-seed 42\n\n", program_name);
    printf("Version: %s\n", morans_i_mkl_version());
}

int parse_weight_format(const char* format_str) {
    if (!format_str) return WEIGHT_FORMAT_AUTO;
    
    if (strcmp(format_str, "auto") == 0) return WEIGHT_FORMAT_AUTO;
    if (strcmp(format_str, "dense") == 0) return WEIGHT_FORMAT_DENSE;
    if (strcmp(format_str, "sparse_coo") == 0) return WEIGHT_FORMAT_SPARSE_COO;
    if (strcmp(format_str, "sparse_tsv") == 0) return WEIGHT_FORMAT_SPARSE_TSV;
    
    fprintf(stderr, "Warning: Unknown weight format '%s', using auto-detection\n", format_str);
    return WEIGHT_FORMAT_AUTO;
}

int parse_analysis_mode(const char* mode_str) {
    if (!mode_str) return ANALYSIS_MODE_STANDARD;
    
    if (strcmp(mode_str, "standard") == 0) return ANALYSIS_MODE_STANDARD;
    if (strcmp(mode_str, "residual") == 0) return ANALYSIS_MODE_RESIDUAL;
    
    fprintf(stderr, "Error: Unknown analysis mode '%s'. Use 'standard' or 'residual'\n", mode_str);
    return -1;
}

int parse_celltype_format(const char* format_str) {
    if (!format_str) return CELLTYPE_FORMAT_SINGLE_CELL;
    
    if (strcmp(format_str, "deconv") == 0 || strcmp(format_str, "deconvolution") == 0) {
        return CELLTYPE_FORMAT_DECONVOLUTION;
    }
    if (strcmp(format_str, "sc") == 0 || strcmp(format_str, "single_cell") == 0) {
        return CELLTYPE_FORMAT_SINGLE_CELL;
    }
    
    fprintf(stderr, "Error: Unknown cell type format '%s'. Use 'deconv' or 'sc'\n", format_str);
    return -1;
}

char** extract_spot_names_from_expression(const DenseMatrix* expr_matrix, MKL_INT* n_spots_out) {
    if (!expr_matrix || !n_spots_out) {
        fprintf(stderr, "Error: NULL parameters in extract_spot_names_from_expression\n");
        if (n_spots_out) *n_spots_out = 0;
        return NULL;
    }
    
    MKL_INT n_spots = expr_matrix->nrows;
    *n_spots_out = n_spots;
    
    if (n_spots == 0) {
        return NULL;
    }
    
    char** spot_names = (char**)malloc(n_spots * sizeof(char*));
    if (!spot_names) {
        perror("Failed to allocate spot_names array");
        *n_spots_out = 0;
        return NULL;
    }
    
    for (MKL_INT i = 0; i < n_spots; i++) {
        if (expr_matrix->rownames && expr_matrix->rownames[i]) {
            spot_names[i] = expr_matrix->rownames[i];  // Reference, not copy
        } else {
            spot_names[i] = NULL;
        }
    }
    
    return spot_names;
}

/* ============================================================================
 * TOY EXAMPLE FUNCTIONS
 * ============================================================================ */

/* Helper function for toy examples */
static inline MKL_INT grid_to_1d_idx(MKL_INT r, MKL_INT c, MKL_INT num_grid_cols) {
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
            PermutationParams toy_perm_params;
            toy_perm_params.n_permutations = (config->num_permutations > 0 && config->num_permutations < 5000) ? 
                                            config->num_permutations : 100;
            toy_perm_params.seed = config->perm_seed; 
            toy_perm_params.z_score_output = config->perm_output_zscores;
            toy_perm_params.p_value_output = config->perm_output_pvalues;

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

/* ============================================================================
 * MAIN FUNCTION
 * ============================================================================ */

/* Main function - now clean and focused */
int main(int argc, char* argv[]) {
    double total_start_time = get_time();
    int final_status = MORANS_I_SUCCESS;
    
    /* Initialize structures */
    MoransIConfig config = initialize_default_config();
    CommandLineArgs args;
    AnalysisResources resources;
    initialize_command_args(&args);
    initialize_resources(&resources);
    
    /* Parse command line arguments */
    int parse_status = parse_command_line_arguments(argc, argv, &config, &args);
    if (parse_status != MORANS_I_SUCCESS) {
        final_status = parse_status;
        goto cleanup;
    }
    
    /* Check if help was requested (indicated by empty output prefix after successful parsing) */
    if (argc <= 2 && strlen(args.output_prefix) == 0) {
        final_status = MORANS_I_SUCCESS;
        goto cleanup;
    }
    
    /* Handle toy example mode */
    if (args.run_toy_example == 2) {
        final_status = run_toy_example_2d(args.output_prefix, &config);
        goto cleanup;
    }
    
    /* Validate configuration and initialize */
    final_status = validate_and_initialize_config(&config, &args);
    if (final_status != MORANS_I_SUCCESS) {
        goto cleanup;
    }
    
    /* Print configuration summary */
    print_configuration_summary(&config, &args);
    
    /* Load and process expression data */
    final_status = load_and_process_expression_data(args.input_file, &resources);
    if (final_status != MORANS_I_SUCCESS) {
        goto cleanup;
    }
    
    /* Load cell type data if needed for residual analysis */
    final_status = load_celltype_data(&config, &args, &resources);
    if (final_status != MORANS_I_SUCCESS) {
        goto cleanup;
    }
    
    /* Setup spatial analysis components */
    final_status = setup_spatial_analysis(&config, &args, &resources);
    if (final_status != MORANS_I_SUCCESS) {
        goto cleanup;
    }
    
    /* Run standard Moran's I analysis */
    final_status = run_moran_analysis(&config, &resources, args.output_prefix);
    if (final_status != MORANS_I_SUCCESS) {
        goto cleanup;
    }
    
    /* Run residual analysis if requested */
    final_status = run_residual_analysis(&config, &resources, args.output_prefix);
    if (final_status != MORANS_I_SUCCESS) {
        goto cleanup;
    }
    
    /* Run permutation analysis if requested */
    final_status = run_permutation_analysis(&config, &resources, args.output_prefix);
    
cleanup:
    /* Clean up all resources */
    cleanup_resources(&resources);
    
    /* Free dynamically allocated config strings */
    if (config.custom_weights_file) {
        free(config.custom_weights_file);
        config.custom_weights_file = NULL;
    }
    if (config.residual_config.celltype_file) {
        free(config.residual_config.celltype_file);
        config.residual_config.celltype_file = NULL;
    }
    if (config.residual_config.celltype_id_col) {
        free(config.residual_config.celltype_id_col);
        config.residual_config.celltype_id_col = NULL;
    }
    if (config.residual_config.celltype_type_col) {
        free(config.residual_config.celltype_type_col);
        config.residual_config.celltype_type_col = NULL;
    }
    if (config.residual_config.celltype_x_col) {
        free(config.residual_config.celltype_x_col);
        config.residual_config.celltype_x_col = NULL;
    }
    if (config.residual_config.celltype_y_col) {
        free(config.residual_config.celltype_y_col);
        config.residual_config.celltype_y_col = NULL;
    }
    if (config.residual_config.spot_id_col) {
        free(config.residual_config.spot_id_col);
        config.residual_config.spot_id_col = NULL;
    }

    double total_end_time = get_time();
    print_elapsed_time(total_start_time, total_end_time, "TOTAL EXECUTION");
    printf("--- Moran's I calculation utility finished with status %d ---\n", final_status);
    return final_status;
}