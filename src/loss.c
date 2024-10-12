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
    assert(y_pred.rows == y_true.rows);
    assert(y_pred.cols == y_true.cols);
    assert(loss->gradient.rows == y_true.rows);
    assert(loss->gradient.cols == y_true.cols);

    loss->value = 0.0;
    MATRIX_ZERO(loss->gradient);
    matrix_copy(loss->gradient, y_pred);
    matrix_subtract(loss->gradient, y_true);

    for (size_t row = 0; row < loss->gradient.rows; ++row)
    {
        for (size_t col = 0; col < loss->gradient.cols; ++col)
        {
            loss->value += MATRIX_AT(loss->gradient, row, col) * MATRIX_AT(loss->gradient, row, col);
        }
    }

    matrix_scale(loss->gradient, 2.0 / MATRIX_ELEMENT_COUNT(y_true));
}
