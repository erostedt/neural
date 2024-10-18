#include "initialization.h"
#include "matrix.h"
#include "random.h"

void matrix_initialize_xavier(matrix_t matrix)
{
    double max = sqrt(6) / (sqrt(matrix.rows + matrix.cols));
    double min = -max;

    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(matrix); ++i)
    {
        MATRIX_AT_INDEX(matrix, i) = uniform(min, max);
    }
}

void matrix_initialize_he(matrix_t matrix)
{
    double std = sqrt(2.0 / matrix.rows);
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(matrix); ++i)
    {
        MATRIX_AT_INDEX(matrix, i) = normal(0.0, std);
    }
}
