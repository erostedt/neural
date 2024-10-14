#include "utest.h"

#include "comparison.h"
#include "matrix.h"
#include "operations.h"

UTEST(matrix, matrix_multiply)
{
    matrix_t lhs = (matrix_t){3, 2, (double[6]){1, 2, 3, 4, 5, 6}};
    matrix_t rhs = (matrix_t){2, 3, (double[6]){7, 8, 9, 10, 11, 12}};
    matrix_t dst = matrix_alloc(3, 3);

    matrix_multiply(dst, lhs, rhs);
    matrix_t expected_output = (matrix_t){3, 3, (double[9]){27, 30, 33, 61, 68, 75, 95, 106, 117}};

    ASSERT_TRUE(matrix_equals(dst, expected_output));
}

UTEST(matrix, matrix_multiply_ABT)
{
    matrix_t lhs = (matrix_t){3, 2, (double[6]){1, 2, 3, 4, 5, 6}};
    matrix_t rhs = (matrix_t){3, 2, (double[6]){7, 10, 8, 11, 9, 12}};
    matrix_t dst = matrix_alloc(3, 3);

    matrix_multiply_ABT(dst, lhs, rhs);
    matrix_t expected_output = (matrix_t){3, 3, (double[9]){27, 30, 33, 61, 68, 75, 95, 106, 117}};

    ASSERT_TRUE(matrix_equals(dst, expected_output));
}

UTEST(matrix, matrix_multiply_ATB)
{
    matrix_t lhs = (matrix_t){2, 3, (double[6]){1, 3, 5, 2, 4, 6}};
    matrix_t rhs = (matrix_t){2, 3, (double[6]){7, 8, 9, 10, 11, 12}};
    matrix_t dst = matrix_alloc(3, 3);
    matrix_multiply_ATB(dst, lhs, rhs);
    matrix_t expected_output = (matrix_t){3, 3, (double[9]){27, 30, 33, 61, 68, 75, 95, 106, 117}};

    ASSERT_TRUE(matrix_equals(dst, expected_output));
}
