#pragma once
#include <stddef.h>

#include "activation.h"
#include "matrix.h"
#include "vector.h"

typedef struct
{
    size_t num_inputs;
    size_t num_neurons;
    activation_type_t activation;
} layer_spec_t;

typedef struct
{
    matrix_t inputs;
    matrix_t weights;
    vector_t biases;
    matrix_t outputs;
    matrix_t d_inputs;
    matrix_t d_weights;
    vector_t d_biases;
    activation_type_t activation;
} layer_t;

layer_t layer_alloc(size_t batch_size, layer_spec_t spec);
void layer_free(layer_t *layer);
matrix_t layer_forward(layer_t *layer, matrix_t inputs);
matrix_t layer_backward(layer_t *layer, matrix_t upstream_gradient);
void layer_update(layer_t *layer, double learning_rate);
void layer_randomize(layer_t *layer);
