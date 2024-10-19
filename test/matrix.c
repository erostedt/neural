#include "utest.h"
#include "comparison.h"

#include "matrix.h"


UTEST(matrix, matrix_at)
{
    matrix_t matrix = {3, 2, (double[]) {1, 2, 3, 4, 5, 6}};
    ASSERT_TRUE(isclose(MATRIX_AT(matrix, 0, 0), 1));
    ASSERT_TRUE(isclose(MATRIX_AT(matrix, 0, 1), 2));
    ASSERT_TRUE(isclose(MATRIX_AT(matrix, 1, 0), 3));
    ASSERT_TRUE(isclose(MATRIX_AT(matrix, 1, 1), 4));
    ASSERT_TRUE(isclose(MATRIX_AT(matrix, 2, 0), 5));
    ASSERT_TRUE(isclose(MATRIX_AT(matrix, 2, 1), 6));
}

UTEST(matrix, matrix_at_index)
{
    matrix_t matrix = {3, 2, (double[]) {1, 2, 3, 4, 5, 6}};
    ASSERT_TRUE(isclose(MATRIX_AT_INDEX(matrix, 0), 1));
    ASSERT_TRUE(isclose(MATRIX_AT_INDEX(matrix, 1), 2));
    ASSERT_TRUE(isclose(MATRIX_AT_INDEX(matrix, 2), 3));
    ASSERT_TRUE(isclose(MATRIX_AT_INDEX(matrix, 3), 4));
    ASSERT_TRUE(isclose(MATRIX_AT_INDEX(matrix, 4), 5));
    ASSERT_TRUE(isclose(MATRIX_AT_INDEX(matrix, 5), 6));
}

UTEST(matrix, matrix_element_count)
{
    matrix_t matrix = {3, 2, (double[]) {1, 2, 3, 4, 5, 6}};
    ASSERT_EQ(MATRIX_ELEMENT_COUNT(matrix), 6);
}

UTEST(matrix, matrix_element_bytes)
{
    matrix_t matrix = {3, 2, (double[]) {1, 2, 3, 4, 5, 6}};
    ASSERT_EQ(MATRIX_ELEMENT_BYTES(matrix), 6 * sizeof(double));
}

UTEST(matrix, matrix_zero)
{
    matrix_t matrix = {3, 2, (double[]) {1, 2, 3, 4, 5, 6}};
    MATRIX_ZERO(matrix);
    matrix_t expected_outputs = {3, 2, (double[6]) {0}};
    ASSERT_TRUE(matrix_equals(matrix, expected_outputs));
}

UTEST(matrix, matrix_alloc)
{
    matrix_t matrix = matrix_alloc(3, 2);
    ASSERT_EQ(matrix.rows, 3);
    ASSERT_EQ(matrix.cols, 2);
}

UTEST(matrix, matrix_alloc_like)
{
    matrix_t matrix = {3, 2, (double[6]){0}};
    matrix_t other = matrix_alloc_like(matrix);
    ASSERT_EQ(other.rows, 3);
    ASSERT_EQ(other.cols, 2);
}

UTEST(matrix, matrix_free)
{
    matrix_t matrix = matrix_alloc(3, 2);
    matrix_free(&matrix);
    ASSERT_EQ(matrix.rows, 0);
    ASSERT_EQ(matrix.cols, 0);
    ASSERT_TRUE(matrix.elements == NULL);
}

UTEST(matrix, matrix_same_shape)
{
    matrix_t matrix1 = matrix_alloc(3, 2);
    matrix_t matrix2 = matrix_alloc(3, 2);
    matrix_t matrix3 = matrix_alloc(2, 3);
    ASSERT_TRUE(matrix_same_shape(matrix1, matrix2));
    ASSERT_TRUE(matrix_same_shape(matrix2, matrix1));
    ASSERT_FALSE(matrix_same_shape(matrix1, matrix3));
    ASSERT_FALSE(matrix_same_shape(matrix3, matrix1));
}

UTEST(matrix, matrix_same_shapes)
{
    matrix_t matrix1 = matrix_alloc(3, 2);
    matrix_t matrix2 = matrix_alloc(3, 2);
    matrix_t matrix3 = matrix_alloc(3, 2);
    matrix_t matrix4 = matrix_alloc(2, 3);
    ASSERT_TRUE(matrix_same_shapes(matrix1, matrix1, matrix3));
    ASSERT_TRUE(matrix_same_shapes(matrix2, matrix1, matrix3));
    ASSERT_FALSE(matrix_same_shapes(matrix4, matrix1, matrix3));
    ASSERT_FALSE(matrix_same_shapes(matrix4, matrix4, matrix3));
}

UTEST(matrix, matrix_copy)
{
    matrix_t src = {3, 2, (double[]) {1, 2, 3, 4, 5, 6}};
    matrix_t dst = {3, 2, (double[6]) {0}};
    matrix_copy(dst, src);
    ASSERT_TRUE(matrix_equals(src, dst));
}

UTEST(matrix, matrix_subtract)
{
    matrix_t lhs = {3, 2, (double[]) {1, 2, 3, 4, 5, 6}};
    matrix_t rhs = {3, 2, (double[]) {6, 5, 4, 3, 2, 1}};
    matrix_t dst = {3, 2, (double[6]) {0}};
    matrix_subtract(dst, lhs, rhs);

    matrix_t expected_outputs = {3, 2, (double[]) {-5, -3, -1, 1, 3, 5}};
    ASSERT_TRUE(matrix_equals(expected_outputs, dst));
}

UTEST(matrix, matrix_scale)
{
    matrix_t matrix = {3, 2, (double[]) {1, 2, 3, 4, 5, 6}};
    matrix_t scaled = {3, 2, (double[6]) {0}};
    double scalar = 3.0;
    matrix_scale(scaled , matrix , scalar);

    matrix_t expected_outputs = {3, 2, (double[]) {3, 6, 9, 12, 15, 18}};

    ASSERT_TRUE(matrix_equals(expected_outputs, scaled));

}

