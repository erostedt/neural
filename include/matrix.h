#pragma once

#include <stddef.h>
#include <memory.h>

#define neural_matrix_at(mat, row, col) (mat).elements[(row) * (mat).cols + (col)]
#define neural_matrix_set_zero(mat) memset((mat).elements, 0, neural_matrix_element_bytes((mat)))
#define neural_matrix_element_count(mat) (mat).rows *(mat).cols
#define neural_matrix_element_bytes(mat) neural_matrix_element_count(mat) * sizeof(*(mat).elements)

typedef struct neural_matrix_t
{
    size_t rows;
    size_t cols;
    float *elements;

} neural_matrix_t;

neural_matrix_t neural_matrix_alloc(size_t rows, size_t cols);
void neural_matrix_random_uniform(neural_matrix_t matrix, float min, float max);
