#include <assert.h>
#include <math.h>
#include <stddef.h>

#include "matrix.h"
#include "operations.h"
#include "random.h"
#include "vector.h"

vector_t row_vector(matrix_t mat, size_t row)
{
    return (vector_t){mat.cols, &MATRIX_AT(mat, row, 0)};
}

void matrix_multiply(matrix_t output, matrix_t lhs, matrix_t rhs)
{
    assert(lhs.cols == rhs.rows);
    assert(lhs.rows == output.rows);
    assert(rhs.cols == output.cols);

    for (size_t i = 0; i < lhs.rows; ++i)
    {
        for (size_t j = 0; j < rhs.cols; ++j)
        {
            MATRIX_AT(output, i, j) = 0.0f;
            for (size_t k = 0; k < lhs.cols; ++k)
            {
                MATRIX_AT(output, i, j) += MATRIX_AT(lhs, i, k) * MATRIX_AT(rhs, k, j);
            }
        }
    }
}

float sum_row(matrix_t mat, size_t row)
{
    float sum = 0.0f;
    for (size_t col = 0; col < mat.cols; ++col)
    {
        sum += MATRIX_AT(mat, row, col);
    }
    return sum;
}

void sum_rows(matrix_t mat, vector_t output)
{
    VECTOR_ZERO(output);
    for (size_t row = 0; row < mat.rows; ++row)
    {
        VECTOR_AT(output, row) = sum_row(mat, row);
    }
}

void matrix_randomize_xavier(matrix_t matrix)
{
    float max = sqrtf(6) / (sqrtf(matrix.rows + matrix.cols));
    float min = -max;

    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = uniform(min, max);
        }
    }
}

void matrix_randomize_he(matrix_t matrix)
{
    float std = sqrtf(2.0f / matrix.row_stride);
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = normal(0.0f, std);
        }
    }
}
