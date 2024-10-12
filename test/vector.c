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
