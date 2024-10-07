#include <assert.h>
#include <stddef.h>

#include "operations.h"

void matrix_multiply(matrix_t output, matrix_t lhs, matrix_t rhs)
{
    assert(lhs.cols == rhs.rows);
    assert(lhs.rows == output.rows);
    assert(rhs.cols == output.cols);

    for (size_t i = 0; i < lhs.rows; ++i)
    {
        for (size_t j = 0; j < rhs.cols; ++j)
        {
            MATRIX_AT(output, i, j) = 0.0f;
            for (size_t k = 0; k < lhs.cols; ++k)
            {
                MATRIX_AT(output, i, j) += MATRIX_AT(lhs, i, k) * MATRIX_AT(rhs, k, j);
            }
        }
    }

}
