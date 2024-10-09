#include <assert.h>
#include <math.h>
#include <stddef.h>

#include "layer.h"
#include "matrix.h"
#include "operations.h"
#include "vector.h"

#define MAX(x, y) x > y ? x : y
#define UNREACHABLE(message) assert(0 && message)

layer_t layer_alloc(size_t batch_size, layer_spec_t spec)
{
    layer_t layer;
    size_t num_inputs = spec.num_inputs;
    size_t num_neurons = spec.num_neurons;
    layer.weights = matrix_alloc(num_inputs, num_neurons);
    layer.biases = vector_alloc(num_neurons);
    layer.outputs = matrix_alloc(batch_size, num_neurons);
    layer.d_inputs = matrix_alloc(batch_size, num_inputs);
    layer.d_weights = matrix_alloc(num_inputs, num_neurons);
    layer.d_biases = vector_alloc(num_neurons);
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

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

void activate_sigmoid(matrix_t matrix)
{
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = sigmoidf(MATRIX_AT(matrix, row, col));
        }
    }
}

void activate_sigmoid_derivative(matrix_t sigmoid_output)
{
    for (size_t row = 0; row < sigmoid_output.rows; ++row)
    {
        for (size_t col = 0; col < sigmoid_output.cols; ++col)
        {
            float sigmoid = MATRIX_AT(sigmoid_output, row, col);
            MATRIX_AT(sigmoid_output, row, col) = sigmoid * (1.0f - sigmoid);
        }
    }
}

void activate_relu(matrix_t matrix)
{
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = MAX(MATRIX_AT(matrix, row, col), 0.0f);
        }
    }
}

void activate_relu_derivative(matrix_t relu_output)
{
    for (size_t row = 0; row < relu_output.rows; ++row)
    {
        for (size_t col = 0; col < relu_output.cols; ++col)
        {
            MATRIX_AT(relu_output, row, col) = MATRIX_AT(relu_output, row, col) > 0.0f ? 1.0f : 0.0f;
        }
    }
}

void activate(matrix_t matrix, activation_type_t activation_type)
{
    switch (activation_type)
    {
    case SIGMOID:
        activate_sigmoid(matrix);
        return;
    case RELU:
        activate_relu(matrix);
        return;
    case NONE:
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
}

void activate_derivative(matrix_t matrix, activation_type_t activation_type)
{
    switch (activation_type)
    {
    case SIGMOID:
        activate_sigmoid_derivative(matrix);
        return;
    case RELU:
        activate_relu_derivative(matrix);
        return;
    case NONE:
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
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

    activate_derivative(layer->outputs, layer->activation);
    matrix_element_multiply(upstream_gradient, layer->outputs);

    matrix_transpose(&layer->weights);
    matrix_multiply(layer->d_inputs, upstream_gradient, layer->weights);
    matrix_transpose(&layer->weights);

    matrix_transpose(&layer->inputs);
    matrix_multiply(layer->d_weights, layer->inputs, upstream_gradient);
    matrix_transpose(&layer->inputs);

    sum_rows(upstream_gradient, layer->d_biases);
    return layer->d_inputs;
}

void layer_randomize(layer_t *layer)
{
    switch (layer->activation)
    {
    case SIGMOID:
    case NONE:
        matrix_randomize_xavier(layer->weights);
        break;
    case RELU:
        matrix_randomize_he(layer->weights);
        break;
    }
    VECTOR_ZERO(layer->biases);
}

void layer_update(layer_t *layer, float learning_rate)
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
