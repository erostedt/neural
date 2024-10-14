#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])

int main()
{
    const size_t BATCH_SIZE = 4;
    const size_t INPUT_SIZE = 2;
    const size_t OUTPUT_SIZE = 1;
    const size_t HIDDEN_NODES = 4;
    const double LEARNING_RATE = 1e-3;
    const size_t EPOCHS = 10000;
    const size_t SEED = 37;
    const loss_type_t loss = BINARY_CROSS_ENTROPY;

    srand(SEED);
    layer_spec_t layers[] = {
        LAYER_SIGMOID(HIDDEN_NODES),
        LAYER_SIGMOID(OUTPUT_SIZE),
    };
    network_t network = network_alloc(BATCH_SIZE, INPUT_SIZE, layers, ARRAY_LEN(layers), loss);

    matrix_t inputs = (matrix_t){BATCH_SIZE, INPUT_SIZE, (double[]){0, 0, 1, 0, 0, 1, 1, 1}};
    matrix_t targets = (matrix_t){BATCH_SIZE, OUTPUT_SIZE, (double[]){0, 1, 1, 0}};

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    for (size_t i = 0; i < EPOCHS; ++i)
    {
        network_train(&network, inputs, targets, optimizer, i);
        printf("loss: %lf\n", network.loss.value);
    }

    matrix_t pred = network_forward(&network, inputs);
    for (size_t i = 0; i < pred.rows; ++i)
    {
        printf("%lf ^ %lf = %lf (%lf)\n", MATRIX_AT(inputs, i, 0), MATRIX_AT(inputs, i, 1), MATRIX_AT(pred, i, 0),
               MATRIX_AT(targets, i, 0));
    }
}
