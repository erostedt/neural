#pragma once
#include "layer.h"
#include "loss.h"
#include "matrix.h"

typedef struct
{
    layer_t *layers;
    size_t layer_count;
    loss_t loss;
} network_t;

network_t network_alloc(size_t batch_size, layer_spec_t *layer_specs, size_t size);
void network_free(network_t *network);
matrix_t network_forward(network_t *network, matrix_t inputs);
void network_train(network_t *network, matrix_t inputs, matrix_t targets, float learning_rate);
void network_summary(network_t *network);
