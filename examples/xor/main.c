#include <stdio.h>
#include <stdlib.h>

#include <neural.h>

int main()
{
    srand(37);
    layer_spec_t layers[] = {
        SIGMOID_LAYER(4),
        SIGMOID_LAYER(1),
    };
    network_t network = network_alloc(4, 2, layers, ARRAY_LEN(layers), BINARY_CROSS_ENTROPY);

    matrix_t inputs = (matrix_t){4, 2, (double[]){0, 0, 1, 0, 0, 1, 1, 1}};
    matrix_t targets = (matrix_t){4, 1, (double[]){0, 1, 1, 0}};
    for (size_t i = 0; i < 10000; ++i)
    {
        network_train(&network, inputs, targets, 1e-0);
        printf("loss: %lf\n", network.loss.value);
    }

    matrix_t pred = network_forward(&network, inputs);
    for (size_t i = 0; i < pred.rows; ++i)
    {
        printf("%lf ^ %lf = %lf (%lf)\n", MATRIX_AT(inputs, i, 0), MATRIX_AT(inputs, i, 1), MATRIX_AT(pred, i, 0),
               MATRIX_AT(targets, i, 0));
    }
}
