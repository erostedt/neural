#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vector.h"


neural_vector_t neural_vector_alloc(size_t count)
{
    neural_vector_t vector;
    vector.count = count;
    vector.elements = malloc(neural_vector_element_bytes(vector));
    assert(vector.elements != NULL);
    return vector;
}

float neural_vector_dot(neural_vector_t v1, neural_vector_t v2)
{
    assert(v1.count == v2.count);
    size_t n = v1.count;
    float dot = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        dot += v1.elements[i] * v2.elements[i];
    }
    return dot;
}


void neural_vector_add(neural_vector_t dst, neural_vector_t lhs, neural_vector_t rhs)
{
    assert(lhs.count == rhs.count);
    assert(dst.count == rhs.count);
    size_t n = lhs.count;
    for (size_t i = 0; i < n; ++i)
    {
        dst.elements[i] = lhs.elements[i] + rhs.elements[i];
    }
}
