#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "loss.h"
#include "matrix.h"
#include "network.h"

int main()
{
    srand(37);
    layer_spec_t spec[2];
    spec[0] = (layer_spec_t){2, 4, SIGMOID};
    spec[1] = (layer_spec_t){4, 1, SIGMOID};
    network_t network = network_alloc(4, spec, 2);
    matrix_t inputs = (matrix_t){4, 2, (double[]){0, 0, 1, 0, 0, 1, 1, 1}};
    matrix_t targets = (matrix_t){4, 1, (double[]){0, 0, 0, 1}};
    for (size_t i = 0; i < 10000; ++i)
    {
        network_train(&network, inputs, targets, 1e-0);
        printf("loss: %lf\n", network.loss.value);
    }

    matrix_t pred = network_forward(&network, inputs);
    for (size_t i = 0; i < pred.rows; ++i)
    {
        printf("%lf & %lf = %lf (%lf)\n", MATRIX_AT(inputs, i, 0), MATRIX_AT(inputs, i, 1), MATRIX_AT(pred, i, 0),
               MATRIX_AT(targets, i, 0));
    }
}
