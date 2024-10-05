#include "utest.h"

#include "matrix.h"
#include "operations.h"
#include "vector.h"

UTEST(matrix, zero)
{
    neural_matrix_t matrix = neural_matrix_zero(2, 3);
    ASSERT_EQ(matrix.rows, 2);
    ASSERT_EQ(matrix.cols, 3);
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            ASSERT_EQ(neural_matrix_at(matrix, row, col), 0);
        }
    }
}

UTEST(matrix, matmul)
{
    neural_matrix_t matrix = {2, 3, (float[6]){1, 2, 3, 4, 5, 6}};
    neural_vector_t vector = {3, (float[3]){7, 8, 9}};
    neural_vector_t dst = neural_vector_zero(2);
    neural_multiply_matrix_vector(dst, matrix, vector);
    ASSERT_EQ(neural_vector_at(dst, 0), 50);
    ASSERT_EQ(neural_vector_at(dst, 1), 122);
}
