#include "utest.h"

#include "matrix.h"
#include "operations.h"

UTEST(matrix, matmulmat)
{
    matrix_t lhs = matrix_alloc(3, 2);
    lhs.elements = (double[6]){1, 2, 3, 4, 5, 6};
    matrix_t rhs = matrix_alloc(2, 3);
    rhs.elements = (double[6]){7, 8, 9, 10, 11, 12};
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
