#pragma once

#include <stddef.h>
#include <memory.h>

#define MATRIX_AT(mat, row, col) (mat).elements[(row) * (mat).row_stride + (col) * (mat).col_stride]
#define MATRIX_ELEMENT_COUNT(mat) (mat).rows *(mat).cols
#define MATRIX_ELEMENT_BYTES(mat) MATRIX_ELEMENT_COUNT(mat) * sizeof(*(mat).elements)
#define MATRIX_ZERO(mat) memset((mat).elements, 0, neural_matrix_element_bytes((mat)))

typedef struct matrix_t
{
    size_t rows;
    size_t cols;
    size_t row_stride;
    size_t col_stride;
    float *elements;

} matrix_t;

matrix_t matrix_alloc(size_t rows, size_t cols);
void matrix_transpose(matrix_t* mat);
