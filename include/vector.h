#pragma once
#include <stddef.h>

#define neural_vector_at(vec, index) (vec).elements[(index)]
#define neural_vector_element_count(vec) (vec).count
#define neural_vector_element_bytes(vec) neural_vector_element_count(vec) * sizeof(*(vec).elements)

typedef struct neural_vector_t
{
    size_t count;
    float *elements;

} neural_vector_t;

neural_vector_t neural_vector_zero(size_t count);
float neural_vector_dot(neural_vector_t v1, neural_vector_t v2);
