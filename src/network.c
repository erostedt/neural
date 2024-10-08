#include "network.h"

network_t network_alloc(size_t batch_size, size_t num_inputs, size_t num_outputs)
{
    network_t network;
    network.layer = layer_alloc(batch_size, num_inputs, num_outputs);
    layer_randomize(&network.layer);
    network.loss = loss_alloc(4, 1);
    return network;
}

matrix_t network_forward(network_t* network, matrix_t inputs)
{
    return layer_forward(&network->layer, inputs);
}

matrix_t network_backward(network_t* network, matrix_t loss_gradient)
{
    matrix_t gradient = layer_backward(&network->layer, loss_gradient);
    return gradient;
}

void network_update(network_t* network, float learning_rate)
{
    layer_update(&network->layer, learning_rate);
}

void network_train(network_t* network, matrix_t inputs, matrix_t targets, float learning_rate)
{
    matrix_t pred = network_forward(network, inputs);
    loss_mse(&network->loss, targets, pred);
    network_backward(network, network->loss.gradient);
    network_update(network, learning_rate);
}
