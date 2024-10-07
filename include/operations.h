#pragma once

#include "matrix.h"
#include "vector.h"

vector_t row_vector(matrix_t mat, size_t row);
void matrix_multiply(matrix_t output, matrix_t lhs, matrix_t rhs);
float sum_row(matrix_t mat, size_t row);
void sum_rows(matrix_t mat, vector_t output);
