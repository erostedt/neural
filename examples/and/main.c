#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "loss.h"
#include "matrix.h"
#include "layer.h"

matrix_t make_and_features()
{
    matrix_t features = matrix_alloc(4, 2);
    MATRIX_AT(features, 0, 0) = 0;
    MATRIX_AT(features, 0, 1) = 0;

    MATRIX_AT(features, 1, 0) = 1;
    MATRIX_AT(features, 1, 1) = 0;

    MATRIX_AT(features, 2, 0) = 0;
    MATRIX_AT(features, 2, 1) = 1;

    MATRIX_AT(features, 3, 0) = 1;
    MATRIX_AT(features, 3, 1) = 1;
    return features;
}

matrix_t make_and_targets()
{
    matrix_t targets = matrix_alloc(4, 1);
    MATRIX_AT(targets, 0, 0) = 0;
    MATRIX_AT(targets, 1, 0) = 0;
    MATRIX_AT(targets, 2, 0) = 0;
    MATRIX_AT(targets, 3, 0) = 1;
    return targets;
}

int main()
{
    srand(time(0));
    layer_t layer = layer_alloc(4, 2, 1);
    layer_randomize(&layer);
    loss_t loss = loss_alloc(4, 1);
    matrix_t inputs = make_and_features();
    matrix_t targets = make_and_targets();
    for (size_t i = 0; i < 10000; ++i)
    {
        layer_forward(&layer, inputs);
        matrix_t pred = layer.outputs;
        loss_mse(&loss, targets, pred);
        layer_backward(&layer, loss.gradient);
        layer_update(&layer, 1e-1f);
        printf("loss: %f\n", loss.value);
    }
    layer_forward(&layer, inputs);
    matrix_t pred = layer.outputs;
    printf("%f & %f = %f\n", MATRIX_AT(inputs, 0, 0), MATRIX_AT(inputs, 0, 1), MATRIX_AT(pred, 0, 0));
    printf("%f & %f = %f\n", MATRIX_AT(inputs, 1, 0), MATRIX_AT(inputs, 1, 1), MATRIX_AT(pred, 1, 0));
    printf("%f & %f = %f\n", MATRIX_AT(inputs, 2, 0), MATRIX_AT(inputs, 2, 1), MATRIX_AT(pred, 2, 0));
    printf("%f & %f = %f\n", MATRIX_AT(inputs, 3, 0), MATRIX_AT(inputs, 3, 1), MATRIX_AT(pred, 3, 0));
}
