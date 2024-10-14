#include <stdio.h>

#include "check.h"
#include "layer.h"
#include "loss.h"
#include "matrix.h"
#include "network.h"
#include "operations.h"
#include "stdlib.h"
#include "vector.h"

network_t network_alloc(size_t batch_size, size_t input_count, layer_spec_t *layer_specs, size_t size, loss_type_t loss)
{
    ASSERT(size > 0);

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

void network_update(network_t *network, adam_parameters_t optimizer, size_t epoch)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer_update(&network->layers[i], optimizer, epoch + 1);
    }
}

size_t network_batch_size(network_t *network)
{
    ASSERT(network->layer_count > 0);
    return network->layers[0].outputs.rows;
}

void shuffle(size_t *elements, size_t size)
{
    if (size < 2)
    {
        return;
    }

    for (size_t i = size - 1; i > 0; --i)
    {
        size_t j = rand() % (i + 1);
        size_t temp = elements[i];
        elements[i] = elements[j];
        elements[j] = temp;
    }
}

size_t *indices_alloc(size_t count)
{
    size_t *indices = malloc(count * sizeof(count));
    ASSERT(indices != NULL);
    for (size_t i = 0; i < count; ++i)
    {
        indices[i] = i;
    }
    return indices;
}

void batch_copy(matrix_t dst, matrix_t src, size_t *indices, size_t batch_size)
{
    for (size_t i = 0; i < batch_size; ++i)
    {
        size_t index = indices[i];
        vector_copy(row_vector(dst, i), row_vector(src, index));
    }
}

void network_train(network_t *network, matrix_t inputs, matrix_t targets, adam_parameters_t optimizer, size_t epochs)
{
    network_summary(network);
    size_t batch_size = network_batch_size(network);
    size_t input_size = inputs.cols;
    size_t target_size = targets.cols;
    matrix_t input_batch = matrix_alloc(batch_size, input_size);
    matrix_t target_batch = matrix_alloc(batch_size, target_size);

    size_t *indices = indices_alloc(inputs.rows);
    shuffle(indices, inputs.rows);

    ASSERT(inputs.rows >= batch_size);
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        double loss_value = 0.0f;
        size_t i = 0;
        for (; i < inputs.rows / batch_size; ++i)
        {
            size_t offset = i * batch_size;
            batch_copy(input_batch, inputs, indices + offset, batch_size);
            batch_copy(target_batch, targets, indices + offset, batch_size);

            matrix_t pred = network_forward(network, input_batch);
            loss_calculate(&network->loss, network->loss_type, target_batch, pred);
            loss_value += network->loss.value;
            network_backward(network, network->loss.gradient);
            network_update(network, optimizer, epoch);
        }
        loss_value /= (i + 1);
        printf("loss: %lf\n", loss_value);
    }

    matrix_free(&input_batch);
    matrix_free(&target_batch);
    free(indices);
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
