#pragma once

#include <memory.h>
#include <stdbool.h>
#include <stddef.h>

#define MATRIX_AT(mat, row, col) (mat).elements[(row) * (mat).cols + (col)]
#define MATRIX_AT_INDEX(mat, index) (mat).elements[(index)]
#define MATRIX_ELEMENT_COUNT(mat) (mat).rows *(mat).cols
#define MATRIX_ELEMENT_BYTES(mat) MATRIX_ELEMENT_COUNT(mat) * sizeof(*(mat).elements)
#define MATRIX_ZERO(mat) memset((mat).elements, 0, MATRIX_ELEMENT_BYTES((mat)))

typedef struct
{
    size_t rows;
    size_t cols;
    double *elements;
} matrix_t;

matrix_t matrix_alloc(size_t rows, size_t cols);
void matrix_free(matrix_t *matrix);
bool matrix_same_shape(matrix_t mat1, matrix_t mat2);
bool matrix_same_shapes(matrix_t mat1, matrix_t mat2, matrix_t mat3);
void matrix_copy(matrix_t dst, matrix_t src);

void matrix_subtract(matrix_t dst, matrix_t lhs, matrix_t rhs);
void matrix_scale(matrix_t matrix, double scalar);
