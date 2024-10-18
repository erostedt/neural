#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])

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
    const size_t SAMPLE_COUNT = 300;
    const size_t BATCH_SIZE = 16;
    const size_t INPUT_SIZE = 1;
    const size_t OUTPUT_SIZE = 1;
    const double LEARNING_RATE = 1e-3;
    const size_t EPOCHS = 10000;
    const size_t SEED = 37;
    const loss_type_t loss = MSE;

    srand(SEED);
    layer_type_t layers[] = {
        LAYER_RELU(8),
        LAYER_RELU(16),
        LAYER_LINEAR(1),
    };
    network_t network = network_alloc(BATCH_SIZE, INPUT_SIZE, layers, ARRAY_LEN(layers), loss);

    network_summary(&network);

    matrix_t inputs = matrix_alloc(SAMPLE_COUNT, INPUT_SIZE);
    matrix_t targets = matrix_alloc(SAMPLE_COUNT, OUTPUT_SIZE);
    double min = -1.0;
    double max = 1.0;
    set_features(inputs, min, max);
    set_targets(inputs, targets);

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    network_train(&network, inputs, targets, optimizer, EPOCHS);

    matrix_t test_inputs = matrix_alloc(BATCH_SIZE, INPUT_SIZE);
    matrix_t test_targets = matrix_alloc(BATCH_SIZE, INPUT_SIZE);
    linspace(test_inputs, min, max);
    set_targets(test_inputs, test_targets);

    matrix_t pred = network_forward(&network, test_inputs);
    for (size_t i = 0; i < pred.rows; ++i)
    {
        printf("(%lf)^2 = %lf (%lf)\n", MATRIX_AT(test_inputs, i, 0), MATRIX_AT(pred, i, 0),
               MATRIX_AT(test_targets, i, 0));
    }
}
