#include "comparison.h"
#include "utest.h"

#include <stdbool.h>

#include "check.h"

#include "vector.h"

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

/*
UTEST(vector, vector_at)
{
    //VECTOR_AT(vec, index)
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_element_count)
{
    //VECTOR_ELEMENT_COUNT(vec)
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_element_bytes)
{
    //VECTOR_ELEMENT_BYTES(vec)
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_alloc)
{
    vector_t vector_alloc(size_t count);
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_free)
{
    void vector_free(vector_t *vector);
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_copy)
{
    void vector_copy(vector_t dst, vector_t src);
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_add)
{
    void vector_add(vector_t dst, vector_t lhs, vector_t rhs);
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_scale)
{
    void vector_scale(vector_t dst, vector_t src, double scalar);
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_same_shape)
{
    bool vector_same_shape(vector_t vec1, vector_t vec2);
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_same_shapes)
{
    bool vector_same_shapes(vector_t vec1, vector_t vec2, vector_t vec3);
    NOT_IMPLEMENTED();
}

UTEST(vector, vector_argmax)
{
    size_t vector_argmax(vector_t vec);
    NOT_IMPLEMENTED();
}
*/
