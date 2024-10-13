#pragma once
#include <stddef.h>

#include "activation.h"
#include "matrix.h"
#include "vector.h"

#define RELU_LAYER(neuron_count)                                                                                       \
    (layer_spec_t)                                                                                                     \
    {                                                                                                                  \
        (neuron_count), RELU                                                                                           \
    }
#define LINEAR_LAYER(neuron_count)                                                                                     \
    (layer_spec_t)                                                                                                     \
    {                                                                                                                  \
        (neuron_count), LINEAR                                                                                         \
    }
#define TANH_LAYER(neuron_count)                                                                                       \
    (layer_spec_t)                                                                                                     \
    {                                                                                                                  \
        (neuron_count), TANH                                                                                           \
    }
#define SIGMOID_LAYER(neuron_count)                                                                                    \
    (layer_spec_t)                                                                                                     \
    {                                                                                                                  \
        (neuron_count), SIGMOID                                                                                        \
    }

#define SOFTMAX_LAYER(neuron_count)                                                                                    \
    (layer_spec_t)                                                                                                     \
    {                                                                                                                  \
        (neuron_count), SOFTMAX                                                                                        \
    }

typedef struct
{
    size_t neuron_count;
    activation_type_t activation;
} layer_spec_t;

typedef struct
{
    double beta1;
    double beta2;
    double epsilon;

} adam_parameters_t;

typedef struct
{
    matrix_t m_weights;
    matrix_t v_weights;
    vector_t m_biases;
    vector_t v_biases;
} adam_state_t;

typedef struct
{
    matrix_t inputs;
    matrix_t d_inputs;

    matrix_t weights;
    matrix_t d_weights;

    vector_t biases;
    vector_t d_biases;

    matrix_t outputs;
    matrix_t d_outputs;

    matrix_t activations;
    activation_type_t activation;

    adam_state_t state;
} layer_t;

layer_t layer_alloc(size_t batch_size, size_t input_count, layer_spec_t spec);
void layer_free(layer_t *layer);
matrix_t layer_forward(layer_t *layer, matrix_t inputs);
matrix_t layer_backward(layer_t *layer, matrix_t upstream_gradient);
void layer_update(layer_t *layer, double learning_rate, size_t epoch);
void layer_randomize(layer_t *layer);
