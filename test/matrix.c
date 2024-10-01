#include "utest.h"

#include "matrix.h"

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
