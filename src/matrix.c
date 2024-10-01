#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

float uniform(float min, float max)
{
    assert(min <= max);
    float u = (float)rand() / RAND_MAX;
    return min + u * (max - min);
}

neural_matrix_t neural_matrix_zero(size_t rows, size_t cols)
{
    neural_matrix_t matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.elements = malloc(neural_matrix_element_bytes(matrix));
    assert(matrix.elements != NULL);
    memset(matrix.elements, 0, neural_matrix_element_bytes(matrix));
    return matrix;
}

void neural_matrix_random_uniform(neural_matrix_t matrix, float min, float max)
{
    for (size_t i = 0; i < neural_matrix_element_count(matrix); ++i)
    {
        matrix.elements[i] = uniform(min, max);
    }
}
