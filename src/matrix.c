#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "matrix.h"

matrix_t matrix_alloc(size_t rows, size_t cols)
{
    matrix_t matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.elements = malloc(MATRIX_ELEMENT_BYTES(matrix));
    ASSERT(matrix.elements != NULL);
    return matrix;
}

void matrix_free(matrix_t *matrix)
{
    matrix->rows = 0;
    matrix->cols = 0;
    free(matrix->elements);
    matrix->elements = NULL;
}

void matrix_copy(matrix_t dst, matrix_t src)
{
    ASSERT(matrix_same_shape(src, dst));
    memcpy(dst.elements, src.elements, MATRIX_ELEMENT_BYTES(src));
}

void matrix_subtract(matrix_t dst, matrix_t lhs, matrix_t rhs)
{
    ASSERT(matrix_same_shapes(lhs, rhs, dst));
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(lhs); ++i)
    {
        MATRIX_AT_INDEX(dst, i) = MATRIX_AT_INDEX(lhs, i) - MATRIX_AT_INDEX(rhs, i);
    }
}

void matrix_scale(matrix_t dst, matrix_t src, double scalar)
{
    ASSERT(matrix_same_shape(src, dst));
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(src); ++i)
    {
        MATRIX_AT_INDEX(dst, i) = MATRIX_AT_INDEX(src, i) * scalar;
    }
}

bool matrix_same_shape(matrix_t mat1, matrix_t mat2)
{
    return (mat1.rows == mat2.rows) && (mat1.cols == mat2.cols);
}

bool matrix_same_shapes(matrix_t mat1, matrix_t mat2, matrix_t mat3)
{
    return matrix_same_shape(mat1, mat2) && matrix_same_shape(mat1, mat3);
}

matrix_t matrix_alloc_like(matrix_t matrix)
{
    return matrix_alloc(matrix.rows, matrix.cols);
}
