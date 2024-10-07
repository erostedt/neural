#include <assert.h>
#include <stddef.h>

#include "operations.h"

neural_vector_t neural_matrix_row_vector(neural_matrix_t matrix, size_t row)
{
    return (neural_vector_t){matrix.cols, &neural_matrix_at(matrix, row, 0)};
}

void neural_multiply_matrix_vector(neural_vector_t output, neural_matrix_t matrix, neural_vector_t vector)
{
    assert(matrix.cols == vector.count);
    assert(matrix.rows == output.count);

    for (size_t row = 0; row < matrix.rows; ++row)
    {
        neural_vector_t row_vector = neural_matrix_row_vector(matrix, row);
        output.elements[row] = neural_vector_dot(row_vector, vector);
    }
}

void neural_multiply_matrix_matrix(neural_matrix_t output, neural_matrix_t lhs, neural_matrix_t rhs)
{
    assert(lhs.cols == rhs.rows);
    assert(lhs.rows == output.rows);
    assert(rhs.cols == output.cols);

    for (size_t i = 0; i < lhs.rows; ++i)
    {
        for (size_t j = 0; j < rhs.cols; ++j)
        {
            neural_matrix_at(output, i, j) = 0.0f;
            for (size_t k = 0; k < lhs.cols; ++k)
            {
                neural_matrix_at(output, i, j) += neural_matrix_at(lhs, i, k) * neural_matrix_at(rhs, k, j);
            }
        }
    }

}
