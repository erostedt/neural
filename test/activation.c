#include "comparison.h"
#include "utest.h"

#include "activation.h"
#include "matrix.h"

UTEST(activation, sigmoid)
{
    matrix_t outputs = {2, 2, (double[]){-100.0, 0.0, 1.0, 100.0}};
    matrix_t activations = {2, 2, (double[4]){0}};
    matrix_t expected_activations = {2, 2, (double[]){0.0, 0.5, 0.7310585786300048793, 1.0}};

    activate(activations, outputs, SIGMOID);
    ASSERT_TRUE(matrix_equals(activations, expected_activations));
}

UTEST(activation_gradient, sigmoid)
{
    matrix_t activations = {2, 2, (double[]){0.0, 0.5, 0.7310585786300048793, 1.0}};
    matrix_t gradients = {2, 2, (double[4]){1.0, 2.0, 3.0, 4.0}};
    matrix_t outputs = {2, 2, (double[4]){0}};

    matrix_t expected_outputs = {2, 2, (double[]){0.0, 0.5, 0.5898357997244455576, 0.0}};

    activate_gradient(outputs, activations, gradients, SIGMOID);
    ASSERT_TRUE(matrix_equals(outputs, expected_outputs));
}

UTEST(activation, relu)
{
    matrix_t outputs = {3, 2, (double[]){-100.0, -1.0, 0.0, 1.0, 100.0, 1000.0}};
    matrix_t activations = {3, 2, (double[6]){0}};
    matrix_t expected_activations = {3, 2, (double[]){0.0, 0.0, 0.0, 1.0, 100.0, 1000.0}};

    activate(activations, outputs, RELU);
    ASSERT_TRUE(matrix_equals(activations, expected_activations));
}

UTEST(activation_gradient, relu)
{
    matrix_t activations = {3, 2, (double[]){-100.0, -1.0, 0.0, 1.0, 100.0, 1000.0}};
    matrix_t gradients = {3, 2, (double[6]){-1.0, 2.0, -3.0, 4.0, -5.0, 6.0}};
    matrix_t outputs = {3, 2, (double[6]){0}};

    matrix_t expected_outputs = {3, 2, (double[]){0.0, 0.0, -3.0, 4.0, -5.0, 6.0}};

    activate_gradient(outputs, activations, gradients, RELU);
    ASSERT_TRUE(matrix_equals(outputs, expected_outputs));
}

UTEST(activation, tanh)
{
    matrix_t outputs = {3, 2, (double[]){-1000.0, -log(2), 0.0, log(2), log(3), 1000.0}};
    matrix_t activations = {3, 2, (double[6]){0}};
    matrix_t expected_activations = {3, 2, (double[]){-1.0, -0.6, 0.0, 0.6, 0.8, 1.0}};

    activate(activations, outputs, TANH);
    ASSERT_TRUE(matrix_equals(activations, expected_activations));
}

UTEST(activation_gradient, tanh)
{
    matrix_t activations = {3, 2, (double[]){-1.0, -0.6, 0.0, 0.6, 0.8, 1.0}};
    matrix_t gradients = {3, 2, (double[6]){-1.0, 2.0, -3.0, 4.0, -5.0, 6.0}};
    matrix_t outputs = {3, 2, (double[6]){0}};

    matrix_t expected_outputs = {3, 2, (double[]){0.0, 1.28, -3, 2.56, -1.8, 0.0}};

    activate_gradient(outputs, activations, gradients, TANH);
    ASSERT_TRUE(matrix_equals(outputs, expected_outputs));
}
