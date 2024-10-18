#pragma once
#include "layer.h"
#include "loss.h"
#include "matrix.h"

typedef struct
{
    layer_t *layers;
    size_t layer_count;
    loss_t loss;
    loss_type_t loss_type;
    matrix_t temp_buffer;
} network_t;

network_t network_alloc(size_t batch_size, size_t input_count, const layer_type_t *layer_types, size_t size,
                        loss_type_t loss);
void network_free(network_t *network);
matrix_t network_forward(network_t *network, matrix_t inputs);
void network_predict(network_t *network, matrix_t inputs, matrix_t prediction);
void network_train(network_t *network, matrix_t inputs, matrix_t targets, adam_parameters_t optimizer, size_t epochs);
void network_summary(const network_t *network);
