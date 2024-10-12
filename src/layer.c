#include <assert.h>
#include <stddef.h>

#include "activation.h"
#include "layer.h"
#include "matrix.h"
#include "operations.h"
#include "vector.h"

layer_t layer_alloc(size_t batch_size, size_t input_count, layer_spec_t spec)
{
    layer_t layer;
    size_t neuron_count = spec.neuron_count;
    layer.weights = matrix_alloc(input_count, neuron_count);
    layer.biases = vector_alloc(neuron_count);
    layer.outputs = matrix_alloc(batch_size, neuron_count);
    layer.d_inputs = matrix_alloc(batch_size, input_count);
    layer.d_weights = matrix_alloc(input_count, neuron_count);
    layer.d_biases = vector_alloc(neuron_count);
    layer.activation = spec.activation;
    return layer;
}

void layer_free(layer_t *layer)
{
    matrix_free(&layer->weights);
    vector_free(&layer->biases);
    matrix_free(&layer->d_weights);
    vector_free(&layer->d_biases);
}

matrix_t layer_forward(layer_t *layer, matrix_t inputs)
{
    layer->inputs = inputs;
    // z = XW + b
    matrix_multiply(layer->outputs, layer->inputs, layer->weights);
    for (size_t row = 0; row < layer->outputs.rows; ++row)
    {
        vector_t output = row_vector(layer->outputs, row);
        vector_add(output, output, layer->biases);
    }
    activate(layer->outputs, layer->activation);
    return layer->outputs;
}

matrix_t layer_backward(layer_t *layer, matrix_t upstream_gradient)
{

    // d_inputs = dZ W^T
    // d_weights = X^T dZ
    // d_biases = sum_i dZ_i

    activate_gradient(layer->outputs, upstream_gradient, layer->activation);

    matrix_multiply_ABT(layer->d_inputs, upstream_gradient, layer->weights);
    matrix_multiply_ATB(layer->d_weights, layer->inputs, upstream_gradient);

    sum_rows(upstream_gradient, layer->d_biases);
    return layer->d_inputs;
}

void layer_randomize(layer_t *layer)
{
    switch (layer->activation)
    {
    case SIGMOID:
    case LINEAR:
    case TANH:
    case SOFTMAX:
        matrix_randomize_xavier(layer->weights);
        break;
    case RELU:
        matrix_randomize_he(layer->weights);
        break;
    }
    VECTOR_ZERO(layer->biases);
}

void layer_update(layer_t *layer, double learning_rate)
{
    for (size_t row = 0; row < layer->d_weights.rows; ++row)
    {
        for (size_t col = 0; col < layer->d_weights.cols; ++col)
        {
            MATRIX_AT(layer->weights, row, col) -= learning_rate * MATRIX_AT(layer->d_weights, row, col);
        }
    }

    for (size_t i = 0; i < layer->d_biases.count; ++i)
    {
        VECTOR_AT(layer->biases, i) -= learning_rate * VECTOR_AT(layer->d_biases, i);
    }
}
