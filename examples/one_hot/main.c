#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])

void set_inputs(matrix_t inputs)
{
    for (size_t row = 0; row < inputs.rows; ++row)
    {
        MATRIX_AT(inputs, row, 0) = row;
    }
}

void set_targets(matrix_t targets)
{
    for (size_t row = 0; row < targets.rows; ++row)
    {
        vector_one_hot(row_vector(targets, row), row);
    }
}

int main()
{
    const size_t BATCH_SIZE = 4;
    const size_t INPUT_SIZE = 1;
    const size_t OUTPUT_SIZE = 4;
    const double LEARNING_RATE = 1e-1;
    const size_t EPOCHS = 10000;
    const size_t SEED = 37;
    const loss_type_t loss = CATEGORICAL_CROSS_ENTROPY;

    srand(SEED);

    layer_spec_t layers[] = {
        LAYER_SIGMOID(8),
        LAYER_SIGMOID(16),
        LAYER_SOFTMAX(OUTPUT_SIZE),
    };
    network_t network = network_alloc(BATCH_SIZE, INPUT_SIZE, layers, ARRAY_LEN(layers), loss);
    matrix_t inputs = matrix_alloc(BATCH_SIZE, INPUT_SIZE);
    matrix_t targets = matrix_alloc(BATCH_SIZE, OUTPUT_SIZE);
    set_inputs(inputs);
    set_targets(targets);

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    network_train(&network, inputs, targets, optimizer, EPOCHS);

    matrix_t pred = network_forward(&network, inputs);
    size_t corrects = 0;
    for (size_t row = 0; row < pred.rows; ++row)
    {
        size_t y_pred = vector_argmax(row_vector(pred, row));
        size_t y_true = vector_argmax(row_vector(targets, row));
        printf("%zu (%zu)\n", y_pred, y_true);
        if (y_pred == y_true)
        {
            ++corrects;
        }
    }
    printf("Accuracy: %lf\n", (double)corrects / pred.rows);
}
