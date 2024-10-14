#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

#include "iris.h"
#include "matrix.h"

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])

int main()
{
    const size_t BATCH_SIZE = 8;
    const size_t INPUT_SIZE = 4;
    const size_t OUTPUT_SIZE = 3;
    const double LEARNING_RATE = 1e-4;
    const size_t EPOCHS = 2000;
    const size_t SEED = 37;
    const loss_type_t loss = CATEGORICAL_CROSS_ENTROPY;

    srand(SEED);

    layer_spec_t layers[] = {
        LAYER_RELU(8),
        LAYER_RELU(16),
        LAYER_RELU(32),
        LAYER_SOFTMAX(OUTPUT_SIZE),
    };

    matrix_t train_features = matrix_alloc(142, INPUT_SIZE);
    matrix_t test_features = matrix_alloc(8, INPUT_SIZE);

    matrix_t targets = matrix_alloc(species.count, OUTPUT_SIZE);
    one_hot_matrix(targets, species, OUTPUT_SIZE);
    matrix_t train_targets = matrix_alloc(142, OUTPUT_SIZE);
    matrix_t test_targets = matrix_alloc(8, OUTPUT_SIZE);

    matrix_split_into(train_features, test_features, features);
    matrix_split_into(train_targets, test_targets, targets);

    network_t network = network_alloc(BATCH_SIZE, INPUT_SIZE, layers, ARRAY_LEN(layers), loss);
    network_summary(&network);

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    network_train(&network, train_features, train_targets, optimizer, EPOCHS);

    matrix_t pred = network_forward(&network, test_features);
    for (size_t row = 0; row < pred.rows; ++row)
    {
        for (size_t col = 0; col < pred.cols; ++col)
        {
            printf("%.2lf (%.2lf), ", MATRIX_AT(pred, row, col), MATRIX_AT(test_targets, row, col));
        }
        printf("\n");
    }
}
