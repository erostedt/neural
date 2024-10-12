#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

void set_features(matrix_t features, double min, double max)
{
    assert(features.cols == 1);
    for (size_t i = 0; i < features.rows; ++i)
    {
        MATRIX_AT(features, i, 0) = uniform(min, max);
    }
}

void set_targets(matrix_t features, matrix_t targets)
{
    assert(features.cols == 1);
    assert(targets.cols == 1);
    assert(targets.cols == features.cols);
    for (size_t i = 0; i < features.rows; ++i)
    {
        MATRIX_AT(targets, i, 0) = MATRIX_AT(features, i, 0) * MATRIX_AT(features, i, 0);
    }
}

void linspace(matrix_t features, double min, double max)
{
    assert(features.cols == 1);
    for (size_t i = 0; i < features.rows; ++i)
    {
        MATRIX_AT(features, i, 0) = min + (max - min) * i / features.rows;
    }
}

int main()
{
    srand(37);
    size_t batch_size = 8;
    layer_spec_t spec[2];
    spec[0] = (layer_spec_t){5, RELU};
    spec[1] = (layer_spec_t){1, LINEAR};
    network_t network = network_alloc(batch_size, 1, spec, 2);
    network_summary(&network);

    matrix_t inputs = matrix_alloc(batch_size, 1);
    matrix_t targets = matrix_alloc(batch_size, 1);
    double min = -1.0;
    double max = 1.0;

    for (size_t i = 0; i < 10000; ++i)
    {
        set_features(inputs, min, max);
        set_targets(inputs, targets);
        network_train(&network, inputs, targets, 1e-3);
        printf("loss: %lf\n", network.loss.value);
    }
    linspace(inputs, min, max);
    set_targets(inputs, targets);
    matrix_t pred = network_forward(&network, inputs);
    for (size_t i = 0; i < pred.rows; ++i)
    {
        printf("(%lf)^2 = %lf (%lf)\n", MATRIX_AT(inputs, i, 0), MATRIX_AT(pred, i, 0), MATRIX_AT(targets, i, 0));
    }
}
