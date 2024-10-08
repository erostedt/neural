#pragma once
#include "layer.h"
#include "loss.h"
#include "matrix.h"

typedef struct network_t
{
    layer_t* layers;
    size_t layer_count;
    loss_t loss;
} network_t;


network_t network_alloc(size_t batch_size, size_t* layer_sizes, size_t layer_count);
void network_free(network_t* network);
matrix_t network_forward(network_t* network, matrix_t inputs);
void network_train(network_t* network, matrix_t inputs, matrix_t targets, float learning_rate);
void network_summary(network_t* network);

