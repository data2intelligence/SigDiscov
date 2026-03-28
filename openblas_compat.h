/* openblas_compat.h - OpenBLAS compatibility layer for MKL-dependent code
 *
 * When USE_OPENBLAS is defined, this header provides type definitions and
 * wrapper functions that map MKL-specific APIs to OpenBLAS equivalents.
 *
 * Covers: MKL_INT, memory allocation, threading, sparse BLAS, VML, LAPACK
 */

#ifndef OPENBLAS_COMPAT_H
#define OPENBLAS_COMPAT_H

#ifdef USE_OPENBLAS

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* OpenBLAS headers */
#include <cblas.h>
#include <lapacke.h>

/* ============================================================
 * Type mappings
 * ============================================================ */

/* Use int to match OpenBLAS/LAPACKE lapack_int (32-bit) */
typedef int MKL_INT;

/* Sparse BLAS status codes */
typedef int sparse_status_t;
#define SPARSE_STATUS_SUCCESS           0
#define SPARSE_STATUS_NOT_INITIALIZED   1
#define SPARSE_STATUS_ALLOC_FAILED      2
#define SPARSE_STATUS_INVALID_VALUE     3
#define SPARSE_STATUS_EXECUTION_FAILED  4
#define SPARSE_STATUS_INTERNAL_ERROR    5
#define SPARSE_STATUS_NOT_SUPPORTED     6

/* Sparse operation types */
#define SPARSE_OPERATION_NON_TRANSPOSE  0
#define SPARSE_OPERATION_TRANSPOSE      1

/* Sparse index base */
#define SPARSE_INDEX_BASE_ZERO  0

/* Sparse matrix type */
#define SPARSE_MATRIX_TYPE_GENERAL  0

/* Sparse layout */
#define SPARSE_LAYOUT_ROW_MAJOR  0

/* Matrix descriptor */
struct matrix_descr {
    int type;
};

/* Sparse matrix handle (CSR storage) */
typedef struct {
    MKL_INT nrows;
    MKL_INT ncols;
    MKL_INT *rows_start;
    MKL_INT *rows_end;
    MKL_INT *col_indx;
    double *values;
} sparse_matrix_internal_t;

typedef sparse_matrix_internal_t* sparse_matrix_t;

/* ============================================================
 * Memory allocation (aligned)
 * ============================================================ */

static inline void* mkl_malloc(size_t size, int alignment) {
    void* ptr = NULL;
    if (size == 0) size = 1; /* avoid NULL for zero-size */
    if (posix_memalign(&ptr, (size_t)alignment, size) != 0) return NULL;
    return ptr;
}

static inline void mkl_free(void* ptr) {
    free(ptr);
}

static inline void* mkl_calloc(size_t num, size_t size, int alignment) {
    if (size != 0 && num > SIZE_MAX / size) return NULL; /* overflow check */
    size_t total = num * size;
    void* ptr = mkl_malloc(total, alignment);
    if (ptr) memset(ptr, 0, total);
    return ptr;
}

/* ============================================================
 * Threading
 * ============================================================ */

/* OpenBLAS threading functions (declared in cblas.h or openblas_config.h) */
extern void openblas_set_num_threads(int num_threads);
extern int openblas_get_num_threads(void);

static inline void mkl_set_num_threads(int n) {
    openblas_set_num_threads(n);
}

static inline int mkl_get_max_threads(void) {
    return openblas_get_num_threads();
}

static inline void mkl_set_dynamic(int flag) {
    (void)flag; /* no-op for OpenBLAS */
}

/* ============================================================
 * Sparse BLAS operations (manual CSR implementations)
 * ============================================================ */

/* Create CSR sparse matrix handle */
static inline sparse_status_t mkl_sparse_d_create_csr(
    sparse_matrix_t *A,
    int indexing,
    MKL_INT rows, MKL_INT cols,
    MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, double *values)
{
    (void)indexing;
    sparse_matrix_internal_t *mat = (sparse_matrix_internal_t*)malloc(sizeof(sparse_matrix_internal_t));
    if (!mat) return SPARSE_STATUS_ALLOC_FAILED;
    mat->nrows = rows;
    mat->ncols = cols;
    mat->rows_start = rows_start;
    mat->rows_end = rows_end;
    mat->col_indx = col_indx;
    mat->values = values;
    *A = mat;
    return SPARSE_STATUS_SUCCESS;
}

/* Optimize (no-op) */
static inline sparse_status_t mkl_sparse_optimize(sparse_matrix_t A) {
    (void)A;
    return SPARSE_STATUS_SUCCESS;
}

/* Order column indices (sort within each row) */
static inline sparse_status_t mkl_sparse_order(sparse_matrix_t A) {
    if (!A) return SPARSE_STATUS_INVALID_VALUE;
    /* Simple insertion sort per row for column indices */
    for (MKL_INT i = 0; i < A->nrows; i++) {
        MKL_INT start = A->rows_start[i];
        MKL_INT end = A->rows_end[i];
        for (MKL_INT j = start + 1; j < end; j++) {
            MKL_INT key_col = A->col_indx[j];
            double key_val = A->values[j];
            MKL_INT k = j - 1;
            while (k >= start && A->col_indx[k] > key_col) {
                A->col_indx[k + 1] = A->col_indx[k];
                A->values[k + 1] = A->values[k];
                k--;
            }
            A->col_indx[k + 1] = key_col;
            A->values[k + 1] = key_val;
        }
    }
    return SPARSE_STATUS_SUCCESS;
}

/* Destroy sparse handle (does NOT free the data arrays) */
static inline sparse_status_t mkl_sparse_destroy(sparse_matrix_t A) {
    free(A);
    return SPARSE_STATUS_SUCCESS;
}

/* Sparse matrix-vector multiply: y = alpha * op(A) * x + beta * y */
static inline sparse_status_t mkl_sparse_d_mv(
    int operation, double alpha, sparse_matrix_t A,
    struct matrix_descr descr,
    const double *x, double beta, double *y)
{
    (void)descr;
    if (!A || !x || !y) return SPARSE_STATUS_INVALID_VALUE;

    MKL_INT nrows = A->nrows;

    if (operation == SPARSE_OPERATION_NON_TRANSPOSE) {
        #pragma omp parallel for
        for (MKL_INT i = 0; i < nrows; i++) {
            double sum = 0.0;
            for (MKL_INT j = A->rows_start[i]; j < A->rows_end[i]; j++) {
                sum += A->values[j] * x[A->col_indx[j]];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    } else {
        /* Transpose: y[col] += alpha * val * x[row] */
        for (MKL_INT i = 0; i < A->ncols; i++) y[i] *= beta;
        for (MKL_INT i = 0; i < nrows; i++) {
            for (MKL_INT j = A->rows_start[i]; j < A->rows_end[i]; j++) {
                y[A->col_indx[j]] += alpha * A->values[j] * x[i];
            }
        }
    }
    return SPARSE_STATUS_SUCCESS;
}

/* Sparse matrix-dense matrix multiply: C = alpha * op(A) * B + beta * C
 * layout: SPARSE_LAYOUT_ROW_MAJOR
 * B is (A_cols x columns) row-major with ldb
 * C is (A_rows x columns) row-major with ldc
 */
static inline sparse_status_t mkl_sparse_d_mm(
    int operation, double alpha, sparse_matrix_t A,
    struct matrix_descr descr,
    int layout,
    const double *B, MKL_INT columns, MKL_INT ldb,
    double beta, double *C, MKL_INT ldc)
{
    (void)descr;
    (void)layout; /* assume row-major */
    if (!A || !B || !C) return SPARSE_STATUS_INVALID_VALUE;

    MKL_INT nrows = A->nrows;

    if (operation == SPARSE_OPERATION_NON_TRANSPOSE) {
        #pragma omp parallel for
        for (MKL_INT i = 0; i < nrows; i++) {
            for (MKL_INT k = 0; k < columns; k++) {
                C[i * ldc + k] *= beta;
            }
            for (MKL_INT j = A->rows_start[i]; j < A->rows_end[i]; j++) {
                MKL_INT col = A->col_indx[j];
                double val = alpha * A->values[j];
                for (MKL_INT k = 0; k < columns; k++) {
                    C[i * ldc + k] += val * B[col * ldb + k];
                }
            }
        }
    } else {
        /* Transpose multiply */
        for (MKL_INT i = 0; i < A->ncols; i++) {
            for (MKL_INT k = 0; k < columns; k++) {
                C[i * ldc + k] *= beta;
            }
        }
        for (MKL_INT i = 0; i < nrows; i++) {
            for (MKL_INT j = A->rows_start[i]; j < A->rows_end[i]; j++) {
                MKL_INT col = A->col_indx[j];
                double val = alpha * A->values[j];
                for (MKL_INT k = 0; k < columns; k++) {
                    C[col * ldc + k] += val * B[i * ldb + k];
                }
            }
        }
    }
    return SPARSE_STATUS_SUCCESS;
}

/* ============================================================
 * VML (Vector Math Library) replacements
 * ============================================================ */

/* Element-wise division: r[i] = a[i] / b[i] */
static inline void vdDiv(MKL_INT n, const double *a, const double *b, double *r) {
    for (MKL_INT i = 0; i < n; i++) {
        r[i] = a[i] / b[i];
    }
}

#endif /* USE_OPENBLAS */
#endif /* OPENBLAS_COMPAT_H */
