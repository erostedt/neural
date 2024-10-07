#include <math.h>
#include <stdbool.h>

#include "matrix.h"
#include "utest.h"

#include "vector.h"

bool isclose(float x, float y)
{
    return fabsf(x - y) < 1e-7;
}

UTEST(vector, zero)
{
    vector_t vector = vector_alloc(3);
    VECTOR_ZERO(vector);
    ASSERT_EQ(vector.count, 3);
    for (size_t i = 0; i < vector.count; ++i)
    {
        ASSERT_EQ(VECTOR_AT(vector, i), 0);
    }
}

UTEST(vector, dot)
{
    vector_t v1 = {3, (float[]){1.0f, 2.0f, 3.0f}};
    vector_t v2 = {3, (float[]){4.0f, 5.0f, 6.0f}};
    float dot = vector_dot(v1, v2);
    ASSERT_TRUE(isclose(dot, 32.0f));
}
