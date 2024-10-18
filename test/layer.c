#include "utest.h"

#include "layer.h"
#include "matrix.h"
#include "vector.h"

UTEST(layer, and)
{
    const size_t BATCH_SIZE = 4;
    const size_t INPUT_SIZE = 2;
    layer_t layer = layer_alloc(BATCH_SIZE, INPUT_SIZE, (layer_type_t){1, SIGMOID});
    MATRIX_AT(layer.weights, 0, 0) = 2.0;
    MATRIX_AT(layer.weights, 1, 0) = 2.0;
    VECTOR_AT(layer.biases, 0) = -3.0;
    matrix_t and = {BATCH_SIZE, INPUT_SIZE, (double[]){0, 0, 1, 0, 0, 1, 1, 1}};
    layer_forward(&layer, and);
    matrix_t pred = layer.outputs;

    ASSERT_LT(MATRIX_AT(pred, 0, 0), 0.5);
    ASSERT_LT(MATRIX_AT(pred, 1, 0), 0.5);
    ASSERT_LT(MATRIX_AT(pred, 2, 0), 0.5);
    ASSERT_GT(MATRIX_AT(pred, 3, 0), 0.5);
}

UTEST(layer, or)
{
    const size_t BATCH_SIZE = 4;
    const size_t INPUT_SIZE = 2;
    layer_t layer = layer_alloc(BATCH_SIZE, INPUT_SIZE, (layer_type_t){1, SIGMOID});
    MATRIX_AT(layer.weights, 0, 0) = 2.0;
    MATRIX_AT(layer.weights, 1, 0) = 2.0;
    VECTOR_AT(layer.biases, 0) = -1.0;
    matrix_t or = {BATCH_SIZE, INPUT_SIZE, (double[]){0, 0, 1, 0, 0, 1, 1, 1}};
    layer_forward(&layer, or);
    matrix_t pred = layer.outputs;

    ASSERT_LT(MATRIX_AT(pred, 0, 0), 0.5);
    ASSERT_GT(MATRIX_AT(pred, 1, 0), 0.5);
    ASSERT_GT(MATRIX_AT(pred, 2, 0), 0.5);
    ASSERT_GT(MATRIX_AT(pred, 3, 0), 0.5);
}

UTEST(layer, not)
{
    const size_t BATCH_SIZE = 2;
    const size_t INPUT_SIZE = 1;
    layer_t layer = layer_alloc(BATCH_SIZE, INPUT_SIZE, (layer_type_t){1, SIGMOID});
    MATRIX_AT(layer.weights, 0, 0) = -2.0;
    VECTOR_AT(layer.biases, 0) = 1.0;
    matrix_t or = {BATCH_SIZE, INPUT_SIZE, (double[]){0, 1}};
    layer_forward(&layer, or);
    matrix_t pred = layer.outputs;

    ASSERT_GT(MATRIX_AT(pred, 0, 0), 0.5);
    ASSERT_LT(MATRIX_AT(pred, 1, 0), 0.5);
}

/*
LAYER_RELU(neuron_count)
LAYER_LINEAR(neuron_count)
LAYER_TANH(neuron_count)
LAYER_SIGMOID(neuron_count)
LAYER_SOFTMAX(neuron_count)
*/
