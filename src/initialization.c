#include "initialization.h"
#include "random.h"

void matrix_randomize_xavier(matrix_t matrix)
{
    double max = sqrt(6) / (sqrt(matrix.rows + matrix.cols));
    double min = -max;

    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = uniform(min, max);
        }
    }
}

void matrix_randomize_he(matrix_t matrix)
{
    double std = sqrt(2.0 / matrix.rows);
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = normal(0.0, std);
        }
    }
}
