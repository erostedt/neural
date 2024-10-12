#include <math.h>
#include <stdbool.h>

#include "utest.h"

#include "vector.h"

bool isclose(double x, double y)
{
    return fabs(x - y) < 1e-7;
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
    vector_t v1 = {3, (double[]){1.0, 2.0, 3.0}};
    vector_t v2 = {3, (double[]){4.0, 5.0, 6.0}};
    double dot = vector_dot(v1, v2);
    ASSERT_TRUE(isclose(dot, 32.0));
}
