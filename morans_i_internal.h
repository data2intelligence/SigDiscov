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

/* Resource Management Pattern */
typedef struct {
    void** ptrs;
    void (**free_funcs)(void*);
    size_t count;
    size_t capacity;
} cleanup_list_t;

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

#endif /* MORANS_I_INTERNAL_H */
