#include <assert.h>

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
