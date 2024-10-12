#pragma once
#include "assert.h"

#include "matrix.h"

typedef struct
{
    double value;
    matrix_t gradient;
} loss_t;

loss_t loss_alloc(size_t input_count, size_t feature_count);
void loss_free(loss_t *loss);
void loss_mse(loss_t *loss, matrix_t y_true, matrix_t y_pred);
