#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

#define ARRAY_LEN(arr) sizeof((arr)) / sizeof((arr)[0])

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
    matrix_t inputs = {BATCH_SIZE, INPUT_SIZE, (double[]){0, 1, 2, 3}};
    matrix_t targets = {BATCH_SIZE, OUTPUT_SIZE, (double[]){1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};

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
