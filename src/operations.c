#include <stddef.h>

#include "check.h"
#include "matrix.h"
#include "operations.h"
#include "vector.h"

vector_t row_vector(matrix_t mat, size_t row)
{
    return (vector_t){mat.cols, &MATRIX_AT(mat, row, 0)};
}

void matrix_multiply(matrix_t output, matrix_t lhs, matrix_t rhs)
{
    ASSERT(lhs.cols == rhs.rows);
    ASSERT(lhs.rows == output.rows);
    ASSERT(rhs.cols == output.cols);

    for (size_t i = 0; i < lhs.rows; ++i)
    {
        for (size_t j = 0; j < rhs.cols; ++j)
        {
            MATRIX_AT(output, i, j) = 0.0;
            for (size_t k = 0; k < lhs.cols; ++k)
            {
                MATRIX_AT(output, i, j) += MATRIX_AT(lhs, i, k) * MATRIX_AT(rhs, k, j);
            }
        }
    }
}

void matrix_multiply_ATB(matrix_t output, matrix_t lhs, matrix_t rhs)
{
    ASSERT(lhs.rows == rhs.rows);
    ASSERT(output.rows == lhs.cols);
    ASSERT(output.cols == rhs.cols);

    for (size_t i = 0; i < lhs.cols; ++i)
    {
        for (size_t j = 0; j < rhs.cols; ++j)
        {
            MATRIX_AT(output, i, j) = 0.0;
            for (size_t k = 0; k < lhs.rows; ++k)
            {
                MATRIX_AT(output, i, j) += MATRIX_AT(lhs, k, i) * MATRIX_AT(rhs, k, j);
            }
        }
    }
}
void matrix_multiply_ABT(matrix_t output, matrix_t lhs, matrix_t rhs)
{
    ASSERT(lhs.cols == rhs.cols);
    ASSERT(output.rows == lhs.rows);
    ASSERT(output.cols == rhs.rows);

    for (size_t i = 0; i < lhs.rows; ++i)
    {
        for (size_t j = 0; j < rhs.rows; ++j)
        {
            MATRIX_AT(output, i, j) = 0.0;
            for (size_t k = 0; k < lhs.cols; ++k)
            {
                MATRIX_AT(output, i, j) += MATRIX_AT(lhs, i, k) * MATRIX_AT(rhs, j, k);
            }
        }
    }
}

double sum_row(matrix_t mat, size_t row)
{
    double sum = 0.0;
    for (size_t col = 0; col < mat.cols; ++col)
    {
        sum += MATRIX_AT(mat, row, col);
    }
    return sum;
}

void sum_rows(vector_t output, matrix_t mat)
{
    VECTOR_ZERO(output);
    for (size_t row = 0; row < mat.rows; ++row)
    {
        VECTOR_AT(output, row) = sum_row(mat, row);
    }
}

void one_hot_matrix(matrix_t dst, vector_t classes, size_t class_count)
{
    ASSERT(dst.rows == classes.count);
    ASSERT(dst.cols == class_count);
    MATRIX_ZERO(dst);
    for (size_t i = 0; i < VECTOR_ELEMENT_COUNT(classes); ++i)
    {
        size_t class = VECTOR_AT(classes, i);
        ASSERT(class < class_count);
        MATRIX_AT(dst, i, class) = 1.0;
    }
}

size_t *indices_alloc(size_t count)
{
    size_t *indices = malloc(count * sizeof(count));
    ASSERT(indices != NULL);
    for (size_t i = 0; i < count; ++i)
    {
        indices[i] = i;
    }
    return indices;
}
