#include <assert.h>
#include <math.h>

#include "loss.h"
#include "matrix.h"

loss_t loss_alloc(size_t input_count, size_t feature_count)
{
    loss_t loss;
    loss.value = 0.0;
    loss.gradient = matrix_alloc(input_count, feature_count);
    MATRIX_ZERO(loss.gradient);
    return loss;
}

void loss_free(loss_t *loss)
{
    matrix_free(&loss->gradient);
}

void loss_mse(loss_t *loss, matrix_t y_true, matrix_t y_pred)
{
    loss->value = 0.0;
    MATRIX_ZERO(loss->gradient);
    matrix_copy(loss->gradient, y_pred);
    matrix_subtract(loss->gradient, loss->gradient, y_true);

    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(loss->gradient); ++i)
    {
        loss->value += MATRIX_AT_INDEX(loss->gradient, i) * MATRIX_AT_INDEX(loss->gradient, i);
    }

    matrix_scale(loss->gradient, 2.0 / MATRIX_ELEMENT_COUNT(y_true));
    loss->value /= y_pred.rows;
}

double clamp(double value, double min, double max)
{
    assert(min <= max);
    return fmax(fmin(value, max), min);
}

void loss_binary_cross_entropy(loss_t *loss, matrix_t y_true, matrix_t y_pred)
{
    loss->value = 0.0;
    MATRIX_ZERO(loss->gradient);
    matrix_copy(loss->gradient, y_pred);
    matrix_subtract(loss->gradient, loss->gradient, y_true);

    double min = 1e-15;
    double max = 1.0 - min;
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(loss->gradient); ++i)
    {
        double label = MATRIX_AT_INDEX(y_true, i);
        double pred = clamp(MATRIX_AT_INDEX(y_pred, i), min, max);
        loss->value -= label * log(pred) + (1.0 - label) * log(1.0 - pred);
    }

    matrix_scale(loss->gradient, 1.0 / MATRIX_ELEMENT_COUNT(y_true));
    loss->value /= y_pred.rows;
}

void loss_calculate(loss_t *loss, loss_type_t loss_type, matrix_t y_true, matrix_t y_pred)
{
    assert(matrix_same_shapes(y_pred, y_true, loss->gradient));
    assert(y_pred.cols == y_true.cols);
    assert(loss->gradient.rows == y_true.rows);
    assert(loss->gradient.cols == y_true.cols);

    switch (loss_type)
    {
    case MSE:
        loss_mse(loss, y_true, y_pred);
        return;
    case BINARY_CROSS_ENTROPY:
        loss_binary_cross_entropy(loss, y_true, y_pred);
        return;
        assert(0);
    }
}
