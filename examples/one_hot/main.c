#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

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
    const double LEARNING_RATE = 1e-3;
    const size_t EPOCHS = 10000;
    const size_t SEED = 37;
    const loss_type_t loss = CATEGORICAL_CROSS_ENTROPY;

    srand(SEED);

    layer_spec_t layers[] = {
        LAYER_RELU(4),
        LAYER_RELU(4),
        LAYER_RELU(4),
        LAYER_SOFTMAX(OUTPUT_SIZE),
    };
    network_t network = network_alloc(BATCH_SIZE, INPUT_SIZE, layers, ARRAY_LEN(layers), loss);
    network_summary(&network);
    matrix_t inputs = matrix_alloc(BATCH_SIZE, INPUT_SIZE);
    matrix_t targets = matrix_alloc(BATCH_SIZE, OUTPUT_SIZE);
    set_inputs(inputs);
    set_targets(targets);

    for (size_t i = 0; i < EPOCHS; ++i)
    {
        network_train(&network, inputs, targets, LEARNING_RATE, i);
        printf("loss: %lf\n", network.loss.value);
    }

    matrix_t pred = network_forward(&network, inputs);
    for (size_t row = 0; row < pred.rows; ++row)
    {
        for (size_t col = 0; col < pred.cols; ++col)
        {
            printf("%.2lf (%.2lf), ", MATRIX_AT(pred, row, col), MATRIX_AT(targets, row, col));
        }
        printf("\n");
    }
}
