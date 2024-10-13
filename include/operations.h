#pragma once

#include "matrix.h"
#include "vector.h"

vector_t row_vector(matrix_t mat, size_t row);
void matrix_multiply(matrix_t output, matrix_t lhs, matrix_t rhs);
void matrix_multiply_ATB(matrix_t output, matrix_t lhs, matrix_t rhs);
void matrix_multiply_ABT(matrix_t output, matrix_t lhs, matrix_t rhs);
double sum_row(matrix_t mat, size_t row);
void sum_rows(vector_t output, matrix_t mat);
void one_hot_matrix(matrix_t dst, vector_t classes, size_t class_count);
