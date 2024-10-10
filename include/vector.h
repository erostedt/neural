#pragma once

#include <memory.h>
#include <stddef.h>

#define VECTOR_AT(vec, index) (vec).elements[(index)]
#define VECTOR_ELEMENT_COUNT(vec) (vec).count
#define VECTOR_ELEMENT_BYTES(vec) VECTOR_ELEMENT_COUNT(vec) * sizeof(*(vec).elements)
#define VECTOR_ZERO(vector) memset((vector).elements, 0, VECTOR_ELEMENT_BYTES((vector)))

typedef struct
{
    size_t count;
    float *elements;

} vector_t;

vector_t vector_alloc(size_t count);
void vector_free(vector_t *vector);
float vector_dot(vector_t v1, vector_t v2);
void vector_add(vector_t dst, vector_t lhs, vector_t rhs);
void vector_copy(vector_t dst, vector_t src);
