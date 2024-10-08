#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vector.h"


vector_t vector_alloc(size_t count)
{
    vector_t vector;
    vector.count = count;
    vector.elements = malloc(VECTOR_ELEMENT_BYTES(vector));
    assert(vector.elements != NULL);
    return vector;
}

void vector_free(vector_t* vector)
{
    vector->count = 0;
    free(vector->elements);
}

float vector_dot(vector_t v1, vector_t v2)
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


void vector_add(vector_t dst, vector_t lhs, vector_t rhs)
{
    assert(lhs.count == rhs.count);
    assert(dst.count == rhs.count);
    size_t n = lhs.count;
    for (size_t i = 0; i < n; ++i)
    {
        dst.elements[i] = lhs.elements[i] + rhs.elements[i];
    }
}

void vector_copy(vector_t dst, vector_t src)
{
    assert(dst.count == src.count);
    for (size_t i = 0; i < src.count; ++i)
    {
        VECTOR_AT(dst, i) = VECTOR_AT(src, i);
    }
}
