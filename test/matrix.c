#include "utest.h"

#include "matrix.h"
#include "operations.h"
#include "vector.h"

UTEST(matrix, zero)
{
    neural_matrix_t matrix = neural_matrix_alloc(2, 3);
    neural_matrix_set_zero(matrix);
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

UTEST(matrix, matmulvec)
{
    neural_matrix_t matrix = {2, 3, (float[6]){1, 2, 3, 4, 5, 6}};
    neural_vector_t vector = {3, (float[3]){7, 8, 9}};
    neural_vector_t dst = neural_vector_alloc(2);
    neural_multiply_matrix_vector(dst, matrix, vector);
    ASSERT_EQ(neural_vector_at(dst, 0), 50);
    ASSERT_EQ(neural_vector_at(dst, 1), 122);
}

UTEST(matrix, matmulmat)
{
    neural_matrix_t lhs = {3, 2, (float[6]){1, 2, 3, 4, 5, 6}};
    neural_matrix_t rhs = {2, 3, (float[6]){7, 8, 9, 10, 11, 12}};
    neural_matrix_t dst = neural_matrix_alloc(3, 3);
    neural_multiply_matrix_matrix(dst, lhs, rhs);
    ASSERT_EQ(neural_matrix_at(dst, 0, 0), 27);
    ASSERT_EQ(neural_matrix_at(dst, 0, 1), 30);
    ASSERT_EQ(neural_matrix_at(dst, 0, 2), 33);
    ASSERT_EQ(neural_matrix_at(dst, 1, 0), 61);
    ASSERT_EQ(neural_matrix_at(dst, 1, 1), 68);
    ASSERT_EQ(neural_matrix_at(dst, 1, 2), 75);
    ASSERT_EQ(neural_matrix_at(dst, 2, 0), 95);
    ASSERT_EQ(neural_matrix_at(dst, 2, 1), 106);
    ASSERT_EQ(neural_matrix_at(dst, 2, 2), 117);
}
