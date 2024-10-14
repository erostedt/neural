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
    srand(37);
    size_t batch_size = 4;
    size_t input_count = 1;
    size_t output_count = 4;

    layer_spec_t layers[] = {
        LAYER_RELU(4),
        LAYER_RELU(4),
        LAYER_RELU(4),
        LAYER_SOFTMAX(output_count),
    };
    network_t network = network_alloc(batch_size, input_count, layers, ARRAY_LEN(layers), CATEGORICAL_CROSS_ENTROPY);
    network_summary(&network);
    matrix_t inputs = matrix_alloc(batch_size, input_count);
    matrix_t targets = matrix_alloc(batch_size, output_count);
    set_inputs(inputs);
    set_targets(targets);
    double learning_rate = 1e-3;

    for (size_t i = 0; i < 10000; ++i)
    {
        network_train(&network, inputs, targets, learning_rate, i);
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
