/* morans_i_internal.h - Internal header for shared helpers across SigDiscov modules
 *
 * This header declares helper functions that were previously static in
 * morans_i_mkl.c but are now shared across multiple translation units as
 * the codebase is split into separate modules.
 *
 * This is NOT part of the public API. Only .c files within SigDiscov
 * should include this header.
 */

#ifndef MORANS_I_INTERNAL_H
#define MORANS_I_INTERNAL_H

#include "morans_i_mkl.h"

#include <ctype.h>    /* for isspace() used by trim_whitespace_inplace */
#include <stdint.h>   /* for SIZE_MAX used by safe_multiply_size_t */
#include <limits.h>
#include <float.h>    /* for DBL_EPSILON, DBL_MAX */
#include <regex.h>    /* for regex_t, regcomp, regexec */
#include <errno.h>    /* for errno, strerror */
#include <unistd.h>   /* for access() */

/* Define M_PI if not provided by math.h */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Enhanced debugging support */
#ifdef DEBUG_BUILD
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "[DEBUG %s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
    #define DEBUG_MATRIX_INFO(mat, name) \
        fprintf(stderr, "[DEBUG] Matrix %s: %lld x %lld, values=%p\n", \
                name, (long long)(mat)->nrows, (long long)(mat)->ncols, (void*)(mat)->values)
#else
    #define DEBUG_PRINT(fmt, ...)
    #define DEBUG_MATRIX_INFO(mat, name)
#endif

/* Maximum reasonable dimensions to prevent memory issues */
#define MAX_REASONABLE_DIM 10000000  /* 10M spots/genes */

/* --- Shared helper function declarations --- */

/**
 * Safe multiplication of two size_t values with overflow detection.
 * Returns 0 on success (result stored in *result), -1 on overflow.
 */
int safe_multiply_size_t(size_t a, size_t b, size_t *result);

/**
 * Validate matrix dimensions (non-negative, within MAX_REASONABLE_DIM).
 * Returns MORANS_I_SUCCESS or MORANS_I_ERROR_PARAMETER.
 */
int validate_matrix_dimensions(MKL_INT nrows, MKL_INT ncols, const char* context);

/**
 * Trim leading and trailing whitespace from a string in place.
 * Returns pointer into the original string (not a new allocation).
 */
char* trim_whitespace_inplace(char* str);

/* --- Spot Name Hash Table helpers --- */

/**
 * djb2 hash function for spot name strings.
 */
unsigned long spot_name_hash(const char* str);

/**
 * Primality test (used internally by get_next_prime).
 */
int is_prime(size_t n);

/**
 * Find the next prime >= n (for hash table bucket sizing).
 */
size_t get_next_prime(size_t n);

/**
 * Create a new spot name hash table sized for approximately num_spots_hint entries.
 * Returns NULL on allocation failure.
 */
SpotNameHashTable* spot_name_ht_create(size_t num_spots_hint);

/**
 * Insert a (name, index) pair into the hash table.
 * Returns 0 on success, -1 on failure.
 */
int spot_name_ht_insert(SpotNameHashTable* ht, const char* name, MKL_INT index);

/**
 * Look up a spot name in the hash table.
 * Returns the associated MKL_INT index, or -1 if not found.
 */
MKL_INT spot_name_ht_find(const SpotNameHashTable* ht, const char* name);

/**
 * Free a spot name hash table and all its entries.
 */
void spot_name_ht_free(SpotNameHashTable* ht);

/* --- String array copy helper --- */

/**
 * Copy an array of strings via strdup.
 * Returns MORANS_I_SUCCESS on success, MORANS_I_ERROR_MEMORY on failure.
 * On failure, already-copied entries in dest[0..i-1] are left as-is
 * (caller is expected to free dest via free_dense_matrix or similar).
 */
int copy_string_array(char** dest, const char** src, MKL_INT count);

/* --- Permutation parameter helpers --- */

/**
 * Initialize PermutationParams from MoransIConfig fields.
 */
static inline PermutationParams config_to_perm_params(const MoransIConfig* config) {
    PermutationParams p;
    p.n_permutations = config->num_permutations;
    p.seed = config->perm_seed;
    p.z_score_output = config->perm_output_zscores;
    p.p_value_output = config->perm_output_pvalues;
    return p;
}

/* --- Permutation worker context structs --- */

/**
 * Context struct that groups the read-only inputs for permutation_worker(),
 * reducing the number of function parameters.
 */
typedef struct {
    const DenseMatrix* X_original;
    const SparseMatrix* W;
    const PermutationParams* params;
    double scaling_factor;
    const DenseMatrix* observed_results;
} PermWorkerContext;

/**
 * Context struct for residual_permutation_worker(), grouping the
 * read-only inputs specific to residual permutation testing.
 */
typedef struct {
    const DenseMatrix* X_original;
    const CellTypeMatrix* Z;
    const SparseMatrix* W;
    const ResidualConfig* config;
    const PermutationParams* params;
    const DenseMatrix* observed_results;
} ResidualPermWorkerContext;

/* --- Permutation result helpers --- */

/**
 * Allocate and initialize a PermutationResults with gene-named matrices.
 * Returns NULL on failure (all allocations cleaned up).
 */
PermutationResults* alloc_permutation_results(MKL_INT n_genes, const char** gene_names,
                                              const PermutationParams* params);

/**
 * Finalize permutation statistics: compute mean, variance, z-scores, p-values
 * from accumulated sums in results->mean_perm and results->var_perm.
 */
void finalize_permutation_statistics(PermutationResults* results,
                                     const DenseMatrix* observed_results,
                                     const PermutationParams* params,
                                     MKL_INT n_genes, int n_perm);

/* --- Random number helpers --- */

/**
 * Unbiased random integer in [0, upper_bound) using rejection sampling.
 * Eliminates modulo bias from rand_r().
 */
static inline MKL_INT rand_range_unbiased(unsigned int* seed, MKL_INT upper_bound) {
    if (upper_bound <= 1) return 0;
    unsigned int threshold = (unsigned int)(-upper_bound) % (unsigned int)upper_bound;
    unsigned int r;
    do {
        r = rand_r(seed);
    } while (r < threshold);
    return (MKL_INT)(r % (unsigned int)upper_bound);
}

/* --- MKL Sparse Handle helper --- */

/**
 * Create an MKL sparse CSR handle from a SparseMatrix and optionally optimize.
 * Returns SPARSE_STATUS_SUCCESS on success. Caller must mkl_sparse_destroy() the handle.
 */
static inline sparse_status_t create_sparse_handle(const SparseMatrix* W, sparse_matrix_t* W_mkl) {
    sparse_status_t status = mkl_sparse_d_create_csr(
        W_mkl, SPARSE_INDEX_BASE_ZERO, W->nrows, W->ncols,
        W->row_ptr, W->row_ptr + 1, W->col_ind, W->values);
    if (status != SPARSE_STATUS_SUCCESS) {
        print_mkl_status(status, "mkl_sparse_d_create_csr");
        return status;
    }
    if (W->nnz > 0) {
        sparse_status_t opt_status = mkl_sparse_optimize(*W_mkl);
        if (opt_status != SPARSE_STATUS_SUCCESS) {
            print_mkl_status(opt_status, "mkl_sparse_optimize");
        }
    }
    return SPARSE_STATUS_SUCCESS;
}

#endif /* MORANS_I_INTERNAL_H */
