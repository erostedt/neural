#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

#include "iris.h"
#include "layer.h"
#include "loss.h"
#include "matrix.h"
#include "network.h"
#include "operations.h"
#include "random.h"
#include "vector.h"

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])

int main()
{
    const size_t SAMPLE_COUNT = 150;
    const size_t BATCH_SIZE = 16;
    const size_t INPUT_SIZE = 4;
    const double TRAINING_FRACTION = 0.7;
    const size_t OUTPUT_SIZE = 3;
    const double LEARNING_RATE = 1e-4;
    const size_t EPOCHS = 10000;
    const size_t SEED = 37;
    const loss_type_t loss = CATEGORICAL_CROSS_ENTROPY;

    srand(SEED);

    layer_spec_t layers[] = {
        LAYER_RELU(8),
        LAYER_RELU(16),
        LAYER_SOFTMAX(OUTPUT_SIZE),
    };

    size_t *indices = indices_alloc(SAMPLE_COUNT);
    shuffle(indices, SAMPLE_COUNT);
    matrix_permute_rows(features, indices);
    vector_permute(species, indices);

    const size_t training_samples = (size_t)(TRAINING_FRACTION * SAMPLE_COUNT);
    const size_t testing_samples = SAMPLE_COUNT - training_samples;
    matrix_t train_features = matrix_alloc(training_samples, INPUT_SIZE);
    matrix_t test_features = matrix_alloc(testing_samples, INPUT_SIZE);

    matrix_t targets = matrix_alloc(species.count, OUTPUT_SIZE);
    one_hot_matrix(targets, species, OUTPUT_SIZE);

    matrix_t train_targets = matrix_alloc(training_samples, OUTPUT_SIZE);
    matrix_t test_targets = matrix_alloc(testing_samples, OUTPUT_SIZE);

    matrix_split_into(train_features, test_features, features);
    matrix_split_into(train_targets, test_targets, targets);

    network_t network = network_alloc(BATCH_SIZE, INPUT_SIZE, layers, ARRAY_LEN(layers), loss);
    network_summary(&network);

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    network_train(&network, train_features, train_targets, optimizer, EPOCHS);

    matrix_t pred = matrix_alloc(testing_samples, OUTPUT_SIZE);
    network_predict(&network, test_features, pred);

    size_t corrects = 0;
    for (size_t row = 0; row < pred.rows; ++row)
    {
        size_t y_pred = vector_argmax(row_vector(pred, row));
        size_t y_true = VECTOR_AT(species, training_samples + row);
        printf("%zu (%zu)\n", y_pred, y_true);
        if (y_pred == y_true)
        {
            ++corrects;
        }
    }
    printf("Accuracy: %lf\n", (double)corrects / pred.rows);
}
