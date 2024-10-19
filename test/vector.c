#include "comparison.h"
#include "utest.h"

#include "vector.h"

UTEST(vector, vector_alloc)
{
    vector_t vector = vector_alloc(3);
    ASSERT_EQ(vector.count, 3);
}

UTEST(vector, vector_zero)
{
    vector_t vector = {3, (double[]){1.0, 2.0, 3.0}};
    VECTOR_ZERO(vector);
    ASSERT_EQ(vector.count, 3);
    for (size_t i = 0; i < vector.count; ++i)
    {
        ASSERT_EQ(VECTOR_AT(vector, i), 0);
    }
}

UTEST(vector, vector_at)
{
    vector_t vector = {3, (double[]){1.0, 2.0, 3.0}};
    ASSERT_EQ(vector.count, 3);
    ASSERT_EQ(VECTOR_AT(vector, 0), 1.0);
    ASSERT_EQ(VECTOR_AT(vector, 1), 2.0);
    ASSERT_EQ(VECTOR_AT(vector, 2), 3.0);
}

UTEST(vector, vector_element_count)
{
    vector_t vector = {3, (double[]){1.0, 2.0, 3.0}};
    ASSERT_EQ(VECTOR_ELEMENT_COUNT(vector), 3);
}

UTEST(vector, vector_element_bytes)
{
    vector_t vector = {3, (double[]){1.0, 2.0, 3.0}};
    ASSERT_EQ(VECTOR_ELEMENT_BYTES(vector), 3 * sizeof(double));
}


UTEST(vector, vector_free)
{
    vector_t vector = vector_alloc(3);
    vector_free(&vector);
    ASSERT_EQ(vector.count, 0);
    ASSERT_EQ(vector.elements, NULL);
}

UTEST(vector, vector_copy)
{
    vector_t src = {3, (double[]){1.0, 2.0, 3.0}};
    vector_t dst = {3, (double[3]){0}};
    vector_copy(dst, src);

    ASSERT_TRUE(vector_equals(src, dst));
}

UTEST(vector, vector_add)
{
    vector_t lhs = {3, (double[]){1.0, 2.0, 3.0}};
    vector_t rhs = {3, (double[]){4.0, 5.0, 6.0}};
    vector_t dst = {3, (double[3]){0}};
    vector_add(dst, lhs, rhs);

    vector_t expected_outputs = {3, (double[3]){5, 7, 9}};

    ASSERT_TRUE(vector_equals(dst, expected_outputs));
}

UTEST(vector, vector_scale)
{
    vector_t src = {3, (double[]){1.0, 2.0, 3.0}};
    double scalar = 3.0;
    vector_t dst = {3, (double[3]){0}};
    vector_scale(dst, src, scalar);

    vector_t expected_outputs = {3, (double[3]){3, 6, 9}};
    ASSERT_TRUE(vector_equals(dst, expected_outputs));
}

UTEST(vector, vector_same_shape)
{
    vector_t vec1 = {3, (double[]){1.0, 2.0, 3.0}};
    vector_t vec2 = {3, (double[]){4.0, 5.0, 6.0}};
    vector_t vec3 = {2, (double[]){7.0, 8.0}};
    ASSERT_TRUE(vector_same_shape(vec1, vec2));
    ASSERT_TRUE(vector_same_shape(vec2, vec1));
    ASSERT_FALSE(vector_same_shape(vec3, vec2));
    ASSERT_FALSE(vector_same_shape(vec1, vec3));
}

UTEST(vector, vector_same_shapes)
{
    vector_t vec1 = {3, (double[]){1.0, 2.0, 3.0}};
    vector_t vec2 = {3, (double[]){4.0, 5.0, 6.0}};
    vector_t vec3 = {3, (double[]){7.0, 8.0, 9.0}};
    vector_t vec4 = {2, (double[]){10.0, 11.0}};

    ASSERT_TRUE(vector_same_shapes(vec1, vec2, vec3));
    ASSERT_TRUE(vector_same_shapes(vec2, vec2, vec3));
    ASSERT_TRUE(vector_same_shapes(vec3, vec2, vec1));

    ASSERT_FALSE(vector_same_shapes(vec4, vec2, vec1));
    ASSERT_FALSE(vector_same_shapes(vec4, vec4, vec1));
    ASSERT_FALSE(vector_same_shapes(vec1, vec4, vec3));
}

UTEST(vector, vector_argmax)
{
    vector_t vec1 = {3, (double[]){1.0, 2.0, 3.0}};
    vector_t vec2 = {3, (double[]){3.0, 2.0, 1.0}};
    vector_t vec3 = {3, (double[]){1.0, 3.0, 2.0}};
    ASSERT_EQ(vector_argmax(vec1), 2);
    ASSERT_EQ(vector_argmax(vec2), 0);
    ASSERT_EQ(vector_argmax(vec3), 1);
}
