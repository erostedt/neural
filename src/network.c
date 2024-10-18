#include <stdio.h>

#include "check.h"
#include "layer.h"
#include "loss.h"
#include "matrix.h"
#include "network.h"
#include "operations.h"
#include "random.h"
#include "stdlib.h"
#include "vector.h"

network_t network_alloc(size_t batch_size, size_t input_count, const layer_spec_t *layer_specs, size_t size, loss_type_t loss)
{
    ASSERT(size > 0);

    network_t network;

    network.loss = loss_alloc(batch_size, layer_specs[size - 1].neuron_count);
    network.loss_type = loss;
    network.temp_buffer = matrix_alloc(batch_size, input_count);

    network.layer_count = size;
    network.layers = malloc(size * sizeof(layer_t));
    for (size_t layer_index = 0; layer_index < size; ++layer_index)
    {
        layer_spec_t spec = layer_specs[layer_index];
        layer_t layer = layer_alloc(batch_size, input_count, spec);
        input_count = spec.neuron_count;
        layer_initialize(&layer);
        network.layers[layer_index] = layer;
    }
    return network;
}

void network_free(network_t *network)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        layer_free(&network->layers[i]);
    }
    loss_free(&network->loss);
    matrix_free(&network->temp_buffer);
    free(network->layers);
    network->layer_count = 0;
}

size_t network_batch_size(network_t *network)
{
    ASSERT(network->layer_count > 0);
    return network->layers[0].outputs.rows;
}

matrix_t network_forward(network_t *network, matrix_t inputs)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        inputs = layer_forward(&network->layers[i], inputs);
    }
    return inputs;
}

void network_predict(network_t *network, matrix_t inputs, matrix_t prediction)
{
    size_t batch_size = network_batch_size(network);
    size_t sample_count = inputs.rows;
    size_t input_size = inputs.cols;
    size_t prediction_size = prediction.cols;
    for (size_t i = 0; i < sample_count / batch_size; ++i)
    {
        matrix_t batch = {batch_size, input_size, inputs.elements + i * batch_size * input_size};
        matrix_t dst = {batch_size, prediction_size, prediction.elements + i * batch_size * prediction_size};
        matrix_copy(dst, network_forward(network, batch));
    }

    if (sample_count % batch_size == 0)
    {
        return;
    }

    matrix_t batch = network->temp_buffer;
    printf("%zu, %zu\n", inputs.rows, inputs.cols);
    printf("%zu, %zu\n", batch.rows, batch.cols);
    printf("%zu, %zu\n", network->temp_buffer.rows, network->temp_buffer.cols);
    MATRIX_ZERO(batch);
    size_t remainder = sample_count % batch_size;
    {
        size_t start = input_size * batch_size * (sample_count / batch_size);
        matrix_t src = {remainder, input_size, inputs.elements + start};
        matrix_t dst = {remainder, input_size, batch.elements};
        matrix_copy(dst, src);
    }

    matrix_t tail = network_forward(network, batch);
    {
        size_t start_pred = prediction_size * batch_size * (sample_count / batch_size);
        matrix_t src = {remainder, input_size, tail.elements};
        matrix_t dst = {remainder, input_size, prediction.elements + start_pred};
        matrix_copy(dst, src);
    }
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

void batch_copy(matrix_t dst, matrix_t src, const size_t *indices, size_t batch_size)
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

    ASSERT(inputs.rows >= batch_size);
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        shuffle(indices, inputs.rows);
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

void network_summary(const network_t *network)
{
    for (size_t i = 0; i < network->layer_count; ++i)
    {
        matrix_t w = network->layers[i].weights;
        vector_t b = network->layers[i].biases;
        printf("Layer %zu: weights (%zu, %zu), biases %zu\n", i + 1, w.rows, w.cols, b.count);
    }
    fflush(stdout);
}
