#include <stdlib.h>

#include <neural.h>

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])

matrix_t make_features(size_t sample_count, double min, double max)
{
    matrix_t features = matrix_alloc(sample_count, 1);
    ASSERT(features.cols == 1);
    for (size_t i = 0; i < features.rows; ++i)
    {
        MATRIX_AT(features, i, 0) = uniform(min, max);
    }
    return features;
}

matrix_t make_targets(matrix_t features)
{
    ASSERT(features.cols == 1);

    matrix_t targets = matrix_alloc_like(features);
    for (size_t i = 0; i < features.rows; ++i)
    {
        MATRIX_AT(targets, i, 0) = MATRIX_AT(features, i, 0) * MATRIX_AT(features, i, 0);
    }
    return targets;
}

int main()
{
    const size_t SAMPLE_COUNT = 1000;
    const size_t BATCH_SIZE = 16;
    const size_t INPUT_SIZE = 1;
    const size_t OUTPUT_SIZE = 1;
    const double TRAIN_FRACTION = 0.9;
    const double LEARNING_RATE = 1e-3;
    const size_t EPOCHS = 1000;
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

    double min = -1.0;
    double max = 1.0;
    matrix_t features = make_features(SAMPLE_COUNT, min, max);
    matrix_t targets = make_targets(features);

    dataset_t dataset = train_test_split(features, targets, TRAIN_FRACTION);

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    network_train(&network, dataset.train_features, dataset.train_targets, optimizer, EPOCHS);

    matrix_t pred = matrix_alloc_like(dataset.test_targets);
    network_predict(&network, dataset.test_targets, pred);
    for (size_t i = 0; i < pred.rows; ++i)
    {
        printf("(%lf)^2 = %lf (%lf)\n", MATRIX_AT(dataset.test_features, i, 0), MATRIX_AT(pred, i, 0),
               MATRIX_AT(dataset.test_targets, i, 0));
    }
}
