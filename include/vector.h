#pragma once

#include <memory.h>
#include <stdbool.h>
#include <stddef.h>

#define VECTOR_AT(vec, index) (vec).elements[(index)]
#define VECTOR_ELEMENT_COUNT(vec) (vec).count
#define VECTOR_ELEMENT_BYTES(vec) VECTOR_ELEMENT_COUNT(vec) * sizeof(*(vec).elements)
#define VECTOR_ZERO(vector) memset((vector).elements, 0, VECTOR_ELEMENT_BYTES((vector)))

typedef struct
{
    size_t count;
    double *elements;
} vector_t;

vector_t vector_alloc(size_t count);
void vector_free(vector_t *vector);
void vector_copy(vector_t dst, vector_t src);
void vector_add(vector_t dst, vector_t lhs, vector_t rhs);
bool vector_same_shape(vector_t vec1, vector_t vec2);
bool vector_same_shapes(vector_t vec1, vector_t vec2, vector_t vec3);
void vector_permute(vector_t vec, const size_t *indices);

size_t vector_argmax(vector_t vec);
