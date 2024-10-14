#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

void set_features(matrix_t features, double min, double max)
{
    ASSERT(features.cols == 1);
    for (size_t i = 0; i < features.rows; ++i)
    {
        MATRIX_AT(features, i, 0) = uniform(min, max);
    }
}

void set_targets(matrix_t features, matrix_t targets)
{
    ASSERT(features.cols == 1);
    ASSERT(targets.cols == 1);
    ASSERT(targets.cols == features.cols);
    for (size_t i = 0; i < features.rows; ++i)
    {
        MATRIX_AT(targets, i, 0) = MATRIX_AT(features, i, 0) * MATRIX_AT(features, i, 0);
    }
}

void linspace(matrix_t features, double min, double max)
{
    ASSERT(features.cols == 1);
    for (size_t i = 0; i < features.rows; ++i)
    {
        MATRIX_AT(features, i, 0) = min + (max - min) * i / features.rows;
    }
}

int main()
{
    srand(37);
    size_t batch_size = 4;
    layer_spec_t layers[] = {
        LAYER_RELU(5),
        LAYER_RELU(5),
        LAYER_RELU(5),
        LAYER_LINEAR(1),
    };
    network_t network = network_alloc(batch_size, 1, layers, ARRAY_LEN(layers), MSE);

    network_summary(&network);

    matrix_t inputs = matrix_alloc(batch_size, 1);
    matrix_t targets = matrix_alloc(batch_size, 1);
    double min = -1.0;
    double max = 1.0;

    double learning_rate = 1e-3;
    for (size_t i = 0; i < 10000; ++i)
    {
        set_features(inputs, min, max);
        set_targets(inputs, targets);
        network_train(&network, inputs, targets, learning_rate, i);
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
