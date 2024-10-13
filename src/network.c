#include <stdio.h>

#include "layer.h"
#include "loss.h"
#include "matrix.h"
#include "network.h"
#include "stdlib.h"
#include "vector.h"

network_t network_alloc(size_t batch_size, size_t input_count, layer_spec_t *layer_specs, size_t size, loss_type_t loss)
{
    assert(size > 0);

    network_t network;
    network.layers = malloc(size * sizeof(layer_t));

    for (size_t layer_index = 0; layer_index < size; ++layer_index)
    {
        layer_spec_t spec = layer_specs[layer_index];
        layer_t layer = layer_alloc(batch_size, input_count, spec);
        input_count = spec.neuron_count;
        layer_randomize(&layer);
        network.layers[layer_index] = layer;
    }
    network.layer_count = size;
    network.loss = loss_alloc(batch_size, layer_specs[size - 1].neuron_count);
    network.loss_type = loss;
    return network;
}

void network_free(network_t *network)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer_free(&network->layers[i]);
    }
    loss_free(&network->loss);
    free(network->layers);
    network->layer_count = 0;
}

matrix_t network_forward(network_t *network, matrix_t inputs)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        inputs = layer_forward(&network->layers[i], inputs);
    }
    return inputs;
}

matrix_t network_backward(network_t *network, matrix_t loss_gradient)
{
    for (int i = network->layer_count - 1; i >= 0; --i)
    {
        loss_gradient = layer_backward(&network->layers[i], loss_gradient);
    }
    return loss_gradient;
}

void network_update(network_t *network, double learning_rate, size_t epoch)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer_update(&network->layers[i], learning_rate, epoch + 1);
    }
}

void network_train(network_t *network, matrix_t inputs, matrix_t targets, double learning_rate, size_t epoch)
{
    matrix_t pred = network_forward(network, inputs);
    loss_calculate(&network->loss, network->loss_type, targets, pred);
    network_backward(network, network->loss.gradient);
    network_update(network, learning_rate, epoch);
}

void network_summary(network_t *network)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        matrix_t w = network->layers[i].weights;
        vector_t b = network->layers[i].biases;
        printf("Layer %zu: weights (%zu, %zu), biases %zu\n", i + 1, w.rows, w.cols, b.count);
    }
    fflush(stdout);
}
