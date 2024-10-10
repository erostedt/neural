#pragma once

#include "matrix.h"

typedef enum
{
    RELU,
    SIGMOID,
    LINEAR,
} activation_type_t;

void activate(matrix_t matrix, activation_type_t activation_type);
void activate_derivative(matrix_t matrix, matrix_t upstream_gradient, activation_type_t activation_type);
