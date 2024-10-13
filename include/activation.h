#pragma once

#include "matrix.h"

typedef enum
{
    RELU,
    SIGMOID,
    LINEAR,
    TANH,
    SOFTMAX,
} activation_type_t;

void activate(matrix_t matrix, activation_type_t activation_type);
void activate_gradient(matrix_t dst, matrix_t activations, matrix_t upstream_gradient,
                       activation_type_t activation_type);
