#include "utest.h"

#include "matrix.h"
#include "operations.h"

UTEST(matrix, matmulmat)
{
    matrix_t lhs = matrix_alloc(3, 2);
    lhs.elements = (float[6]){1, 2, 3, 4, 5, 6};
    matrix_t rhs = matrix_alloc(2, 3);
    rhs.elements = (float[6]){7, 8, 9, 10, 11, 12};
    matrix_t dst = matrix_alloc(3, 3);
    matrix_multiply(dst, lhs, rhs);
    ASSERT_EQ(MATRIX_AT(dst, 0, 0), 27);
    ASSERT_EQ(MATRIX_AT(dst, 0, 1), 30);
    ASSERT_EQ(MATRIX_AT(dst, 0, 2), 33);
    ASSERT_EQ(MATRIX_AT(dst, 1, 0), 61);
    ASSERT_EQ(MATRIX_AT(dst, 1, 1), 68);
    ASSERT_EQ(MATRIX_AT(dst, 1, 2), 75);
    ASSERT_EQ(MATRIX_AT(dst, 2, 0), 95);
    ASSERT_EQ(MATRIX_AT(dst, 2, 1), 106);
    ASSERT_EQ(MATRIX_AT(dst, 2, 2), 117);
}

UTEST(matrix, mattranspose)
{
    matrix_t m = matrix_alloc(3, 2);
    m.elements = (float[6]){1, 2, 3, 4, 5, 6};

    matrix_t t = m;
    matrix_transpose(&t);
    ASSERT_EQ(m.rows, 3);
    ASSERT_EQ(m.cols, 2);
    ASSERT_EQ(m.row_stride, 2);
    ASSERT_EQ(m.col_stride, 1);

    ASSERT_EQ(t.rows, 2);
    ASSERT_EQ(t.cols, 3);
    ASSERT_EQ(t.row_stride, 1);
    ASSERT_EQ(t.col_stride, 2);

    for (size_t row = 0; row < 3; ++row)
    {
        for (size_t col = 0; col < 2; ++col)
        {
            ASSERT_EQ(MATRIX_AT(m, row, col), MATRIX_AT(t, col, row));
        }
    }
}
