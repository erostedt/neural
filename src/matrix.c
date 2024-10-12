#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

matrix_t matrix_alloc(size_t rows, size_t cols)
{
    matrix_t matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.elements = malloc(MATRIX_ELEMENT_BYTES(matrix));
    assert(matrix.elements != NULL);
    return matrix;
}

void matrix_free(matrix_t *matrix)
{
    matrix->rows = 0;
    matrix->cols = 0;
    free(matrix->elements);
}

void matrix_copy(matrix_t dst, matrix_t src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);
    for (size_t row = 0; row < src.rows; ++row)
    {
        for (size_t col = 0; col < src.cols; ++col)
        {
            MATRIX_AT(dst, row, col) = MATRIX_AT(src, row, col);
        }
    }
}

void matrix_subtract(matrix_t dst, matrix_t src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);
    for (size_t row = 0; row < src.rows; ++row)
    {
        for (size_t col = 0; col < src.cols; ++col)
        {
            MATRIX_AT(dst, row, col) -= MATRIX_AT(src, row, col);
        }
    }
}

void matrix_scale(matrix_t matrix, double scalar)
{
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) *= scalar;
        }
    }
}
