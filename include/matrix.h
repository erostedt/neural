#pragma once

#include <stddef.h>
#include <memory.h>

#define MATRIX_AT(mat, row, col) (mat).elements[(row) * (mat).cols + (col)]
#define MATRIX_ELEMENT_COUNT(mat) (mat).rows *(mat).cols
#define MATRIX_ELEMENT_BYTES(mat) MATRIX_ELEMENT_COUNT(mat) * sizeof(*(mat).elements)
#define MATRIX_ZERO(mat) memset((mat).elements, 0, neural_matrix_element_bytes((mat)))

typedef struct matrix_t
{
    size_t rows;
    size_t cols;
    float *elements;

} matrix_t;

matrix_t matrix_alloc(size_t rows, size_t cols);
