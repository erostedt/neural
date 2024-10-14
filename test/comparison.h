#pragma once

#include <math.h>
#include <stdbool.h>

#include "matrix.h"
#include "vector.h"

static inline bool isclose(double x, double y)
{
    return fabs(x - y) < 1e-8;
}

static inline bool vector_equals(vector_t vec1, vector_t vec2)
{
    if (!vector_same_shape(vec1, vec2))
    {
        return false;
    }

    for (size_t i = 0; i < VECTOR_ELEMENT_COUNT(vec1); ++i)
    {
        if (!isclose(VECTOR_AT(vec1, i), VECTOR_AT(vec2, i)))
        {
            return false;
        }
    }
    return true;
}

static inline bool matrix_equals(matrix_t mat1, matrix_t mat2)
{
    if (!matrix_same_shape(mat1, mat2))
    {
        return false;
    }

    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(mat1); ++i)
    {
        if (!isclose(MATRIX_AT_INDEX(mat1, i), MATRIX_AT_INDEX(mat2, i)))
        {
            return false;
        }
    }
    return true;
}
