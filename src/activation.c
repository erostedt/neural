#include <assert.h>
#include <math.h>

#include "activation.h"
#include "matrix.h"

#define UNREACHABLE(message) assert(0 && message)

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

void activate_sigmoid(matrix_t dst, matrix_t outputs)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(outputs); ++i)
    {
        MATRIX_AT_INDEX(dst, i) = sigmoid(MATRIX_AT_INDEX(outputs, i));
    }
}

void activate_sigmoid_gradient(matrix_t dst, matrix_t activations, matrix_t upstream_gradient)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(activations); ++i)
    {
        double sig = MATRIX_AT_INDEX(activations, i);
        MATRIX_AT_INDEX(dst, i) = sig * (1.0 - sig) * MATRIX_AT_INDEX(upstream_gradient, i);
    }
}

void activate_softmax(matrix_t dst, matrix_t outputs)
{
    for (size_t row = 0; row < outputs.rows; ++row)
    {
        double max_value = MATRIX_AT(outputs, row, 0);
        for (size_t col = 1; col < outputs.cols; ++col)
        {
            if (MATRIX_AT(outputs, row, col) > max_value)
            {
                max_value = MATRIX_AT(outputs, row, col);
            }
        }
        double sum = 0.0;
        for (size_t col = 0; col < outputs.cols; ++col)
        {
            sum += exp(MATRIX_AT(outputs, row, col) - max_value);
        }

        for (size_t col = 0; col < outputs.cols; ++col)
        {
            MATRIX_AT(dst, row, col) = exp(MATRIX_AT(outputs, row, col) - max_value) / sum;
        }
    }
}

void activate_softmax_gradient(matrix_t dst, matrix_t activations, matrix_t upstream_gradient)
{
    for (size_t row = 0; row < activations.rows; ++row)
    {
        for (size_t col = 0; col < activations.cols; ++col)
        {
            double grad_sum = 0.0;
            for (size_t class = 0; class < activations.cols; ++class)
            {
                if (col == class)
                {
                    grad_sum += MATRIX_AT(upstream_gradient, row, class) * MATRIX_AT(activations, row, col) *
                                (1.0 - MATRIX_AT(activations, row, col));
                }
                else
                {
                    grad_sum -= MATRIX_AT(upstream_gradient, row, class) * MATRIX_AT(activations, row, col) *
                                MATRIX_AT(activations, row, class);
                }
            }

            MATRIX_AT(dst, row, col) *= grad_sum * MATRIX_AT(upstream_gradient, row, col);
        }
    }
}

void activate_relu(matrix_t dst, matrix_t ouputs)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(ouputs); ++i)
    {
        if (MATRIX_AT_INDEX(ouputs, i) < 0.0)
        {
            MATRIX_AT_INDEX(dst, i) = 0.0;
        }
        MATRIX_AT_INDEX(dst, i) = (MATRIX_AT_INDEX(ouputs, i) < 0.0) ? 0.0 : MATRIX_AT_INDEX(ouputs, i);
    }
}

void activate_relu_gradient(matrix_t dst, matrix_t activations, matrix_t upstream_gradient)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(activations); ++i)
    {
        MATRIX_AT_INDEX(dst, i) = (MATRIX_AT_INDEX(activations, i) < 0.0) ? 0.0 : MATRIX_AT_INDEX(upstream_gradient, i);
    }
}

void activate_tanh(matrix_t dst, matrix_t outputs)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(outputs); ++i)
    {
        MATRIX_AT_INDEX(dst, i) = tanh(MATRIX_AT_INDEX(outputs, i));
    }
}

void activate_tanh_gradient(matrix_t dst, matrix_t activations, matrix_t upstream_gradient)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(activations); ++i)
    {
        double tanh_ = MATRIX_AT_INDEX(activations, i);
        MATRIX_AT_INDEX(dst, i) *= (1.0 - (tanh_ * tanh_)) * MATRIX_AT_INDEX(upstream_gradient, i);
    }
}

void activate(matrix_t dst, matrix_t outputs, activation_type_t activation_type)
{
    assert(matrix_same_shape(dst, outputs));
    switch (activation_type)
    {
    case SIGMOID:
        activate_sigmoid(dst, outputs);
        return;
    case RELU:
        activate_relu(dst, outputs);
        return;
    case TANH:
        activate_tanh(dst, outputs);
        return;
    case LINEAR:
        matrix_copy(dst, outputs);
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
}

void activate_gradient(matrix_t dst, matrix_t activations, matrix_t upstream_gradient,
                       activation_type_t activation_type)
{
    assert(matrix_same_shapes(dst, activations, upstream_gradient));
    switch (activation_type)
    {
    case SIGMOID:
        activate_sigmoid_gradient(dst, activations, upstream_gradient);
        return;
    case RELU:
        activate_relu_gradient(dst, activations, upstream_gradient);
        return;
    case TANH:
        activate_tanh_gradient(dst, activations, upstream_gradient);
        return;
    case LINEAR:
        matrix_copy(dst, upstream_gradient);
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
}
