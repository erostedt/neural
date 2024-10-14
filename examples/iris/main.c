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
    network_t network = network_alloc(BATCH_SIZE, INPUT_SIZE, layers, ARRAY_LEN(layers), loss);
    network_summary(&network);
    matrix_t targets = matrix_alloc(species.count, OUTPUT_SIZE);
    one_hot_matrix(targets, species, OUTPUT_SIZE);

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    network_train(&network, features, targets, optimizer, EPOCHS);

    matrix_t test = {BATCH_SIZE, INPUT_SIZE, features.elements};
    matrix_t pred = network_forward(&network, test);
    for (size_t row = 0; row < pred.rows; ++row)
    {
        for (size_t col = 0; col < pred.cols; ++col)
        {
            printf("%.2lf (%.2lf), ", MATRIX_AT(pred, row, col), MATRIX_AT(targets, row, col));
        }
        printf("\n");
    }
}
