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

UTEST(layer, macros)
{
    layer_type_t layers[] = {LAYER_RELU(3), LAYER_LINEAR(5), LAYER_TANH(30), LAYER_SIGMOID(12), LAYER_SOFTMAX(4)};

    ASSERT_EQ(layers[0].activation, RELU);
    ASSERT_EQ(layers[0].neuron_count, 3);

    ASSERT_EQ(layers[1].activation, LINEAR);
    ASSERT_EQ(layers[1].neuron_count, 5);

    ASSERT_EQ(layers[2].activation, TANH);
    ASSERT_EQ(layers[2].neuron_count, 30);

    ASSERT_EQ(layers[3].activation, SIGMOID);
    ASSERT_EQ(layers[3].neuron_count, 12);

    ASSERT_EQ(layers[4].activation, SOFTMAX);
    ASSERT_EQ(layers[4].neuron_count, 4);
}
