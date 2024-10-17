#include <math.h>
#include <stddef.h>

#include "activation.h"
#include "check.h"
#include "initialization.h"
#include "layer.h"
#include "matrix.h"
#include "operations.h"
#include "vector.h"

layer_t layer_alloc(size_t batch_size, size_t input_count, layer_spec_t spec)
{
    layer_t layer;
    size_t neuron_count = spec.neuron_count;
    layer.weights = matrix_alloc(input_count, neuron_count);
    layer.d_weights = matrix_alloc(input_count, neuron_count);
    layer.biases = vector_alloc(neuron_count);
    layer.d_biases = vector_alloc(neuron_count);

    layer.outputs = matrix_alloc(batch_size, neuron_count);
    layer.d_outputs = matrix_alloc(batch_size, neuron_count);
    layer.activations = matrix_alloc(batch_size, neuron_count);

    layer.d_inputs = matrix_alloc(batch_size, input_count);
    layer.activation = spec.activation;

    layer.state.m_weights = matrix_alloc(input_count, neuron_count);
    layer.state.v_weights = matrix_alloc(input_count, neuron_count);
    layer.state.m_biases = vector_alloc(neuron_count);
    layer.state.v_biases = vector_alloc(neuron_count);

    MATRIX_ZERO(layer.state.m_weights);
    MATRIX_ZERO(layer.state.v_weights);
    VECTOR_ZERO(layer.state.m_biases);
    VECTOR_ZERO(layer.state.v_biases);

    return layer;
}

void layer_free(layer_t *layer)
{
    matrix_free(&layer->weights);
    vector_free(&layer->biases);
    matrix_free(&layer->d_weights);
    vector_free(&layer->d_biases);
    matrix_free(&layer->outputs);
    matrix_free(&layer->d_outputs);

    matrix_free(&layer->state.m_weights);
    matrix_free(&layer->state.v_weights);
    vector_free(&layer->state.m_biases);
    vector_free(&layer->state.v_biases);
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
    activate(layer->activations, layer->outputs, layer->activation);
    return layer->activations;
}

matrix_t layer_backward(layer_t *layer, matrix_t upstream_gradient)
{

    // d_inputs = dZ W^T
    // d_weights = X^T dZ
    // d_biases = sum_i dZ_i

    activate_gradient(layer->d_outputs, layer->activations, upstream_gradient, layer->activation);

    matrix_multiply_ABT(layer->d_inputs, layer->d_outputs, layer->weights);
    matrix_multiply_ATB(layer->d_weights, layer->inputs, layer->d_outputs);

    sum_rows(layer->d_biases, layer->d_outputs);
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

void layer_update(layer_t *layer, adam_parameters_t optimizer, size_t epoch)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(layer->d_weights); ++i)
    {
        MATRIX_AT_INDEX(layer->weights, i) -= optimizer.learning_rate * MATRIX_AT_INDEX(layer->d_weights, i);
    }
    for (size_t i = 0; i < VECTOR_ELEMENT_COUNT(layer->d_biases); ++i)
    {
        VECTOR_AT(layer->biases, i) -= optimizer.learning_rate * MATRIX_AT_INDEX(layer->d_biases, i);
    }
    /*
    adam_state_t *state = &layer->state;
    ASSERT(matrix_same_shape(state->m_weights, state->v_weights));
    ASSERT(matrix_same_shapes(state->m_weights, layer->weights, layer->d_weights));

    ASSERT(vector_same_shape(state->m_biases, state->v_biases));
    ASSERT(vector_same_shapes(state->m_biases, layer->biases, layer->d_biases));

    double m_hat_scale = 1.0 / (1.0 - pow(optimizer.beta1, epoch));
    double v_hat_scale = 1.0 / (1.0 - pow(optimizer.beta2, epoch));

    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(layer->d_weights); ++i)
    {
        double g = MATRIX_AT_INDEX(layer->d_weights, i);
        double m = MATRIX_AT_INDEX(state->m_weights, i);
        double v = MATRIX_AT_INDEX(state->v_weights, i);

        m = optimizer.beta1 * m + (1.0 - optimizer.beta1) * g;
        v = optimizer.beta2 * v + (1.0 - optimizer.beta2) * (g * g);

        double m_hat = m * m_hat_scale;
        double v_hat = v * v_hat_scale;

        MATRIX_AT_INDEX(layer->weights, i) -= optimizer.learning_rate * m_hat / (sqrt(v_hat) + optimizer.epsilon);

        MATRIX_AT_INDEX(state->m_weights, i) = m;
        MATRIX_AT_INDEX(state->v_weights, i) = v;
    }

    for (size_t i = 0; i < VECTOR_ELEMENT_COUNT(layer->d_biases); ++i)
    {
        double g = VECTOR_AT(layer->d_biases, i);
        double m = VECTOR_AT(state->m_biases, i);
        double v = VECTOR_AT(state->v_biases, i);

        m = optimizer.beta1 * m + (1.0 - optimizer.beta1) * g;
        v = optimizer.beta2 * v + (1.0 - optimizer.beta2) * (g * g);

        double m_hat = m * m_hat_scale;
        double v_hat = v * v_hat_scale;

        VECTOR_AT(layer->biases, i) -= optimizer.learning_rate * m_hat / (sqrt(v_hat) + optimizer.epsilon);

        VECTOR_AT(state->m_biases, i) = m;
        VECTOR_AT(state->v_biases, i) = v;
    }
    */
}
