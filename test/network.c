#include "matrix.h"
#include "utest.h"
#include "comparison.h"

#include "check.h"
#include "vector.h"
#include "network.h"

UTEST(network, network_alloc)
{
    layer_type_t layers[] = {
        LAYER_SIGMOID(1),
        LAYER_RELU(2),
        LAYER_TANH(3),
        LAYER_LINEAR(4),
        LAYER_SOFTMAX(5),
    };

    size_t batch_size = 4;
    size_t size = 5;
    size_t input_count = 2;
    loss_type_t loss = MSE;


    network_t network =  network_alloc(batch_size, input_count, layers, size, loss);
    ASSERT_EQ(network.layer_count, size);
    ASSERT_TRUE(matrix_same_shape(network.temp_buffer, (matrix_t){4, 2}));

    ASSERT_TRUE(matrix_same_shape(network.layers[0].d_inputs, (matrix_t){4, 2}));
    ASSERT_TRUE(matrix_same_shape(network.layers[1].d_inputs, (matrix_t){4, 1}));
    ASSERT_TRUE(matrix_same_shape(network.layers[2].d_inputs, (matrix_t){4, 2}));
    ASSERT_TRUE(matrix_same_shape(network.layers[3].d_inputs, (matrix_t){4, 3}));
    ASSERT_TRUE(matrix_same_shape(network.layers[4].d_inputs, (matrix_t){4, 4}));

    ASSERT_TRUE(matrix_same_shapes(network.layers[0].weights, network.layers[0].d_weights, (matrix_t){2, 1}));
    ASSERT_TRUE(matrix_same_shapes(network.layers[1].weights, network.layers[1].d_weights, (matrix_t){1, 2}));
    ASSERT_TRUE(matrix_same_shapes(network.layers[2].weights, network.layers[2].d_weights, (matrix_t){2, 3}));
    ASSERT_TRUE(matrix_same_shapes(network.layers[3].weights, network.layers[3].d_weights, (matrix_t){3, 4}));
    ASSERT_TRUE(matrix_same_shapes(network.layers[4].weights, network.layers[4].d_weights, (matrix_t){4, 5}));

    ASSERT_TRUE(matrix_same_shapes(network.layers[0].state.m_weights, network.layers[0].state.v_weights, (matrix_t){2, 1}));
    ASSERT_TRUE(matrix_same_shapes(network.layers[1].state.m_weights, network.layers[1].state.v_weights, (matrix_t){1, 2}));
    ASSERT_TRUE(matrix_same_shapes(network.layers[2].state.m_weights, network.layers[2].state.v_weights, (matrix_t){2, 3}));
    ASSERT_TRUE(matrix_same_shapes(network.layers[3].state.m_weights, network.layers[3].state.v_weights, (matrix_t){3, 4}));
    ASSERT_TRUE(matrix_same_shapes(network.layers[4].state.m_weights, network.layers[4].state.v_weights, (matrix_t){4, 5}));

    ASSERT_TRUE(matrix_same_shape(network.layers[0].outputs, (matrix_t){4, 1}));
    ASSERT_TRUE(matrix_same_shape(network.layers[1].outputs, (matrix_t){4, 2}));
    ASSERT_TRUE(matrix_same_shape(network.layers[2].outputs, (matrix_t){4, 3}));
    ASSERT_TRUE(matrix_same_shape(network.layers[3].outputs, (matrix_t){4, 4}));
    ASSERT_TRUE(matrix_same_shape(network.layers[4].outputs, (matrix_t){4, 5}));

    ASSERT_TRUE(matrix_same_shapes(network.layers[0].outputs, network.layers[0].d_outputs, network.layers[0].activations));
    ASSERT_TRUE(matrix_same_shapes(network.layers[1].outputs, network.layers[1].d_outputs, network.layers[1].activations));
    ASSERT_TRUE(matrix_same_shapes(network.layers[2].outputs, network.layers[2].d_outputs, network.layers[2].activations));
    ASSERT_TRUE(matrix_same_shapes(network.layers[3].outputs, network.layers[3].d_outputs, network.layers[3].activations));
    ASSERT_TRUE(matrix_same_shapes(network.layers[4].outputs, network.layers[4].d_outputs, network.layers[4].activations));

    ASSERT_TRUE(vector_same_shapes(network.layers[0].biases, network.layers[0].d_biases, (vector_t){1}));
    ASSERT_TRUE(vector_same_shapes(network.layers[1].biases, network.layers[1].d_biases, (vector_t){2}));
    ASSERT_TRUE(vector_same_shapes(network.layers[2].biases, network.layers[2].d_biases, (vector_t){3}));
    ASSERT_TRUE(vector_same_shapes(network.layers[3].biases, network.layers[3].d_biases, (vector_t){4}));
    ASSERT_TRUE(vector_same_shapes(network.layers[4].biases, network.layers[4].d_biases, (vector_t){5}));

    ASSERT_TRUE(vector_same_shapes(network.layers[0].biases, network.layers[0].state.m_biases, network.layers[0].state.v_biases));
    ASSERT_TRUE(vector_same_shapes(network.layers[1].biases, network.layers[1].state.m_biases, network.layers[1].state.v_biases));
    ASSERT_TRUE(vector_same_shapes(network.layers[2].biases, network.layers[2].state.m_biases, network.layers[2].state.v_biases));
    ASSERT_TRUE(vector_same_shapes(network.layers[3].biases, network.layers[3].state.m_biases, network.layers[3].state.v_biases));
    ASSERT_TRUE(vector_same_shapes(network.layers[4].biases, network.layers[4].state.m_biases, network.layers[4].state.v_biases));
}
