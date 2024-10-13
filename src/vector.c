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

void vector_free(vector_t *vector)
{
    vector->count = 0;
    free(vector->elements);
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

bool vector_same_shape(vector_t vec1, vector_t vec2)
{
    return vec1.count == vec2.count;
}

bool vector_same_shapes(vector_t vec1, vector_t vec2, vector_t vec3)
{
    return vector_same_shape(vec1, vec2) && vector_same_shape(vec1, vec3);
}
