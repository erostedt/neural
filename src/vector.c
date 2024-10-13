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

void vector_one_hot(vector_t vec, size_t class)
{
    assert(class < VECTOR_ELEMENT_COUNT(vec));
    VECTOR_ZERO(vec);
    VECTOR_AT(vec, class) = 1.0f;
}

size_t vector_argmax(vector_t vec)
{
    assert(VECTOR_ELEMENT_COUNT(vec) > 0);
    size_t amax = 0;
    size_t max = VECTOR_AT(vec, 0);
    for (size_t i = 1; i < VECTOR_ELEMENT_COUNT(vec); ++i)
    {
        if (VECTOR_AT(vec, i) > max)
        {
            max = VECTOR_AT(vec, i);
            amax = i;
        }
    }
    return amax;
}
