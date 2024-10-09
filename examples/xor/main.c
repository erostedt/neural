#include <stdio.h>
#include <stdlib.h>

#include "loss.h"
#include "matrix.h"
#include "network.h"

matrix_t make_xor_features()
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

matrix_t make_xor_targets()
{
    matrix_t targets = matrix_alloc(4, 1);
    MATRIX_AT(targets, 0, 0) = 0;
    MATRIX_AT(targets, 1, 0) = 1;
    MATRIX_AT(targets, 2, 0) = 1;
    MATRIX_AT(targets, 3, 0) = 0;
    return targets;
}

int main()
{
    srand(37);
    network_t network = network_alloc(4, (size_t[4]){2, 4, 1}, 3);
    matrix_t inputs = make_xor_features();
    matrix_t targets = make_xor_targets();
    for (size_t i = 0; i < 10000; ++i)
    {
        network_train(&network, inputs, targets, 1e-0f);
        printf("loss: %f\n", network.loss.value);
    }

    matrix_t pred = network_forward(&network, inputs);
    printf("%f ^ %f = %f\n", MATRIX_AT(inputs, 0, 0), MATRIX_AT(inputs, 0, 1), MATRIX_AT(pred, 0, 0));
    printf("%f ^ %f = %f\n", MATRIX_AT(inputs, 1, 0), MATRIX_AT(inputs, 1, 1), MATRIX_AT(pred, 1, 0));
    printf("%f ^ %f = %f\n", MATRIX_AT(inputs, 2, 0), MATRIX_AT(inputs, 2, 1), MATRIX_AT(pred, 2, 0));
    printf("%f ^ %f = %f\n", MATRIX_AT(inputs, 3, 0), MATRIX_AT(inputs, 3, 1), MATRIX_AT(pred, 3, 0));
}
