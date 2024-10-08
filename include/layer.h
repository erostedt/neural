#pragma once
#include <stddef.h>

#include "matrix.h"
#include "vector.h"


typedef struct layer_t
{
    matrix_t inputs;
    matrix_t weights;
    vector_t biases;
    matrix_t outputs;
    matrix_t d_inputs;
    matrix_t d_weights;
    vector_t d_biases;
} layer_t;


layer_t layer_alloc(size_t batch_size, size_t num_inputs, size_t num_neurons);
void layer_free(layer_t* layer);
matrix_t layer_forward(layer_t *layer, matrix_t inputs);
matrix_t layer_backward(layer_t *layer, matrix_t upstream_gradient);
void layer_update(layer_t *layer, float learning_rate);
void layer_randomize(layer_t *layer);
