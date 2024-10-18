#include <stdio.h>

#include <neural.h>

#include "dataset.h"
#include "iris.h"

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])

int main()
{
    const size_t SAMPLE_COUNT = features.rows;
    const size_t INPUT_SIZE = features.cols;
    const size_t BATCH_SIZE = 16;
    const double TRAINING_FRACTION = 0.7;
    const size_t OUTPUT_SIZE = 3;
    const double LEARNING_RATE = 1e-4;
    const size_t EPOCHS = 6000;
    const size_t SEED = 37;
    const loss_type_t LOSS = CATEGORICAL_CROSS_ENTROPY;

    set_seed(SEED);

    layer_spec_t layers[] = {
        LAYER_RELU(8),
        LAYER_RELU(16),
        LAYER_SOFTMAX(OUTPUT_SIZE),
    };

    standardize(features);
    matrix_t targets = matrix_alloc(species.count, OUTPUT_SIZE);
    one_hot_encode(targets, species, OUTPUT_SIZE);

    dataset_t dataset = train_test_split(features, targets, TRAINING_FRACTION);

    network_t network = network_alloc(BATCH_SIZE, INPUT_SIZE, layers, ARRAY_LEN(layers), LOSS);
    network_summary(&network);

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    network_train(&network, dataset.train_features, dataset.train_targets, optimizer, EPOCHS);

    matrix_t pred = matrix_alloc_like(dataset.test_targets);
    network_predict(&network, dataset.test_features, pred);

    size_t corrects = 0;
    for (size_t row = 0; row < pred.rows; ++row)
    {
        size_t y_pred = vector_argmax(row_vector(pred, row));
        size_t y_true = vector_argmax(row_vector(dataset.test_targets, row));
        printf("%zu (%zu)\n", y_pred, y_true);
        if (y_pred == y_true)
        {
            ++corrects;
        }
    }
    printf("Accuracy: %lf\n", (double)corrects / pred.rows);
}
