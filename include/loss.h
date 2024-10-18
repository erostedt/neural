#pragma once

#include "matrix.h"

typedef enum
{
    MSE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
} loss_type_t;

typedef struct
{
    double value;
    matrix_t gradient;
} loss_t;

loss_t loss_alloc(size_t input_count, size_t feature_count);
void loss_free(loss_t *loss);
void loss_calculate(loss_t *loss, loss_type_t loss_type, matrix_t y_pred, matrix_t y_true);
