/* morans_i_mkl.h - Header for optimized MKL-based Moran's I implementation */

#ifndef MORANS_I_MKL_H
#define MORANS_I_MKL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>  /* For timing functions */
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_vml.h"  /* For vdDiv and other VML functions */

/* Constants */
#define VISIUM 0
#define OLD 1
#define SINGLE_CELL 2
#define ZERO_STD_THRESHOLD 1e-8
#define WEIGHT_THRESHOLD 1e-12
#define BUFFER_SIZE 1024
#define DEFAULT_MAX_RADIUS 5
#define DEFAULT_PLATFORM_MODE VISIUM
#define DEFAULT_CALC_PAIRWISE 1
#define DEFAULT_CALC_ALL_VS_ALL 1
#define DEFAULT_INCLUDE_SAME_SPOT 0
#define DEFAULT_COORD_SCALE_FACTOR 100.0

/* Structures */
typedef struct {
    double* values;     /* Use mkl_malloc for alignment */
    MKL_INT nrows;
    MKL_INT ncols;
    char** rownames;
    char** colnames;
} DenseMatrix;

typedef struct {
    MKL_INT nrows;
    MKL_INT ncols;
    MKL_INT nnz;
    MKL_INT* row_ptr;   /* CSR format */
    MKL_INT* col_ind;   /* CSR format */
    double* values;     /* CSR format */
    char** rownames;    /* Usually NULL for W */
    char** colnames;    /* Usually NULL for W */
} SparseMatrix;

typedef struct {
    MKL_INT* spot_row;  /* Standard malloc ok */
    MKL_INT* spot_col;  /* Standard malloc ok */
    char** spot_names;  /* Standard malloc ok */
    int* valid_mask;    /* Standard malloc ok */
    MKL_INT total_spots;
    MKL_INT valid_spots;
} SpotCoordinates;

/* Function prototypes */
void print_help(const char* program_name);
int load_positive_value(const char* value_str, const char* param, unsigned int min, unsigned int max);
double load_double_value(const char* value_str, const char* param);

double decay(double d);
DenseMatrix* create_distance_matrix(MKL_INT max_radius, int platform_mode);
SpotCoordinates* extract_coordinates(char** column_names, MKL_INT n_columns);
SpotCoordinates* read_coordinates_file(const char* filename, const char* id_column, 
                                      const char* x_column, const char* y_column, 
                                      double coord_scale);
int map_expression_to_coordinates(DenseMatrix* expr_matrix, SpotCoordinates* coords, 
                                 MKL_INT** mapping);
DenseMatrix* z_normalize(DenseMatrix* data_matrix);
SparseMatrix* build_spatial_weight_matrix(MKL_INT* spot_row_valid, MKL_INT* spot_col_valid,
                                         MKL_INT n_spots_valid, DenseMatrix* distance_matrix,
                                         MKL_INT max_radius);
DenseMatrix* calculate_morans_i(DenseMatrix* X, SparseMatrix* W);
double calculate_single_gene_moran_i(double* gene_data, SparseMatrix* W, MKL_INT n_spots);
double* calculate_first_gene_vs_all(DenseMatrix* X, SparseMatrix* W, double S0);
void save_results(DenseMatrix* result_matrix, const char* output_file);
void save_single_gene_results(DenseMatrix* znorm_data, SparseMatrix* W, double S0, const char* output_file);
void save_first_gene_vs_all_results(double* morans_values, const char** gene_names, MKL_INT n_genes, const char* output_file);
DenseMatrix* read_vst_file(const char* filename);
void free_dense_matrix(DenseMatrix* matrix);
void free_sparse_matrix(SparseMatrix* matrix);
void free_spot_coordinates(SpotCoordinates* coords);
void print_mkl_status(sparse_status_t status, const char* function_name);
double* calculate_morans_i_batch(double* X_data, long long n_genes, long long n_spots,
                                double* W_values, long long* W_row_ptr, long long* W_col_ind,
                                long long W_nnz, int paired_genes);

#endif /* MORANS_I_MKL_H */