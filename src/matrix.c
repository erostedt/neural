#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "matrix.h"
#include "operations.h"

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
}

void matrix_copy_first_rows(matrix_t dst, matrix_t src, size_t rows)
{
    ASSERT(rows <= src.rows && rows <= dst.rows);
    ASSERT(src.cols == dst.cols);
    size_t byte_count = src.cols * rows * sizeof(*dst.elements);
    memcpy(dst.elements, src.elements, byte_count);
}

void matrix_copy(matrix_t dst, matrix_t src)
{
    ASSERT(matrix_same_shape(src, dst));
    matrix_copy_first_rows(dst, src, dst.rows);
}

void matrix_subtract(matrix_t dst, matrix_t lhs, matrix_t rhs)
{
    ASSERT(matrix_same_shapes(lhs, rhs, dst));
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(lhs); ++i)
    {
        MATRIX_AT_INDEX(dst, i) = MATRIX_AT_INDEX(lhs, i) - MATRIX_AT_INDEX(rhs, i);
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

bool matrix_same_shape(matrix_t mat1, matrix_t mat2)
{
    return (mat1.rows == mat2.rows) && (mat1.cols == mat2.cols);
}

bool matrix_same_shapes(matrix_t mat1, matrix_t mat2, matrix_t mat3)
{
    return matrix_same_shape(mat1, mat2) && matrix_same_shape(mat1, mat3);
}

void matrix_split_into(matrix_t dst1, matrix_t dst2, matrix_t src)
{
    ASSERT(dst1.rows + dst2.rows == src.rows);
    ASSERT(dst1.cols == src.cols);
    ASSERT(dst2.cols == src.cols);

    for (size_t i = 0; i < dst1.rows; ++i)
    {
        vector_copy(row_vector(dst1, i), row_vector(src, i));
    }

    for (size_t i = 0; i < dst2.rows; ++i)
    {
        size_t src_index = dst1.rows + i;
        vector_copy(row_vector(dst2, i), row_vector(src, src_index));
    }
}
