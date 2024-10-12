#include <stdio.h>
#include <stdlib.h>

#include "loss.h"
#include "matrix.h"
#include "network.h"

matrix_t make_or_features()
{
    matrix_t features = matrix_alloc(4, 2);
    MATRIX_AT(features, 0, 0) = 0;
    MATRIX_AT(features, 0, 1) = 0;

    MATRIX_AT(features, 1, 0) = 1;
    MATRIX_AT(features, 1, 1) = 0;

    MATRIX_AT(features, 2, 0) = 0;
    MATRIX_AT(features, 2, 1) = 1;

    MATRIX_AT(features, 3, 0) = 1;
    MATRIX_AT(features, 3, 1) = 1;
    return features;
}

matrix_t make_or_targets()
{
    matrix_t targets = matrix_alloc(4, 1);
    MATRIX_AT(targets, 0, 0) = 0;
    MATRIX_AT(targets, 1, 0) = 1;
    MATRIX_AT(targets, 2, 0) = 1;
    MATRIX_AT(targets, 3, 0) = 1;
    return targets;
}

int main()
{
    srand(37);
    layer_spec_t spec[2];
    spec[0] = (layer_spec_t){2, 4, SIGMOID};
    spec[1] = (layer_spec_t){4, 1, SIGMOID};
    network_t network = network_alloc(4, spec, 2);
    network_summary(&network);
    matrix_t inputs = make_or_features();
    matrix_t targets = make_or_targets();
    for (size_t i = 0; i < 10000; ++i)
    {
        network_train(&network, inputs, targets, 1e-1);
        printf("loss: %lf\n", network.loss.value);
    }

    matrix_t pred = network_forward(&network, inputs);
    for (size_t i = 0; i < pred.rows; ++i)
    {
        printf("%lf | %lf = %lf (%lf)\n", MATRIX_AT(inputs, i, 0), MATRIX_AT(inputs, i, 1), MATRIX_AT(pred, i, 0),
               MATRIX_AT(targets, i, 0));
    }
}
