#include <stddef.h>

#include "layer.h"
#include "operations.h"



layer_t layer_alloc(size_t batch_size, size_t num_inputs, size_t num_neurons)
{
    layer_t layer;
    layer.weights = matrix_alloc(num_inputs, num_neurons);
    layer.biases = vector_alloc(num_neurons);
    layer.outputs = matrix_alloc(batch_size, num_neurons);
    layer.d_weights = matrix_alloc(num_inputs, num_neurons);
    layer.d_biases = vector_alloc(num_neurons);
    return layer;
}

void layer_free(layer_t* layer)
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
    return layer->outputs;
}


matrix_t layer_backward(layer_t *layer, matrix_t upstream_gradient)
{
    // d_inputs = dZ W^T 
    // d_weights = X^T dZ
    // d_biases = sum_i dZ_i
    matrix_transpose(&layer->weights);
    matrix_multiply(layer->d_inputs, upstream_gradient, layer->weights);
    matrix_transpose(&layer->weights);

    matrix_transpose(&layer->inputs);
    matrix_multiply(layer->d_weights, layer->inputs, upstream_gradient);
    matrix_transpose(&layer->inputs);

    sum_rows(upstream_gradient, layer->d_biases);
    return layer->d_inputs;
}

