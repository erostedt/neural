#include "utest.h"

#include "comparison.h"
#include "matrix.h"
#include "operations.h"

UTEST(operations, matrix_multiply)
{
    matrix_t lhs = {3, 2, (double[6]){1, 2, 3, 4, 5, 6}};
    matrix_t rhs = {2, 3, (double[6]){7, 8, 9, 10, 11, 12}};
    matrix_t dst = matrix_alloc(3, 3);

    matrix_multiply(dst, lhs, rhs);
    matrix_t expected_outputs = {3, 3, (double[9]){27, 30, 33, 61, 68, 75, 95, 106, 117}};

    ASSERT_TRUE(matrix_equals(dst, expected_outputs));
}

UTEST(operations, matrix_multiply_ABT)
{
    matrix_t lhs = {3, 2, (double[6]){1, 2, 3, 4, 5, 6}};
    matrix_t rhs = {3, 2, (double[6]){7, 10, 8, 11, 9, 12}};
    matrix_t dst = matrix_alloc(3, 3);

    matrix_multiply_ABT(dst, lhs, rhs);
    matrix_t expected_outputs = {3, 3, (double[9]){27, 30, 33, 61, 68, 75, 95, 106, 117}};

    ASSERT_TRUE(matrix_equals(dst, expected_outputs));
}

UTEST(operations, matrix_multiply_ATB)
{
    matrix_t lhs = {2, 3, (double[6]){1, 3, 5, 2, 4, 6}};
    matrix_t rhs = {2, 3, (double[6]){7, 8, 9, 10, 11, 12}};
    matrix_t dst = matrix_alloc(3, 3);
    matrix_multiply_ATB(dst, lhs, rhs);
    matrix_t expected_outputs = {3, 3, (double[9]){27, 30, 33, 61, 68, 75, 95, 106, 117}};

    ASSERT_TRUE(matrix_equals(dst, expected_outputs));
}

UTEST(operations, sum_rows)
{
    matrix_t matrix = {3, 2, (double[]){1, 2, 3, 4, 5, 6}};
    vector_t dst = {3, (double[3]){0}};
    sum_rows(dst, matrix);
    vector_t expected_outputs = {3, (double[]){3, 7, 11}};
    ASSERT_TRUE(vector_equals(dst, expected_outputs));
}

UTEST(operations, row_vector)
{
    matrix_t matrix = {3, 2, (double[]){1, 2, 3, 4, 5, 6}};
    ASSERT_TRUE(vector_equals(row_vector(matrix, 0), (vector_t){2, (double[]){1, 2}}));
    ASSERT_TRUE(vector_equals(row_vector(matrix, 1), (vector_t){2, (double[]){3, 4}}));
    ASSERT_TRUE(vector_equals(row_vector(matrix, 2), (vector_t){2, (double[]){5, 6}}));
}
