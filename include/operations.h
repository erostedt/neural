#pragma once

#include "matrix.h"
#include "vector.h"

void neural_multiply_matrix_vector(neural_vector_t output, neural_matrix_t matrix, neural_vector_t vector);
void neural_multiply_matrix_matrix(neural_matrix_t output, neural_matrix_t lhs, neural_matrix_t rhs);
