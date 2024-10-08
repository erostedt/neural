#pragma once
#include "layer.h"
#include "loss.h"
#include "matrix.h"

typedef struct network_t
{
    layer_t layer;
    loss_t loss;
} network_t;


network_t network_alloc(size_t batch_size, size_t num_inputs, size_t num_outputs);
void network_free(network_t* network);
matrix_t network_forward(network_t* network, matrix_t inputs);
void network_train(network_t* network, matrix_t inputs, matrix_t targets, float learning_rate);

