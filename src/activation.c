#include <assert.h>
#include <math.h>

#include "activation.h"
#include "matrix.h"

#define UNREACHABLE(message) assert(0 && message)

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

void activate_sigmoid(matrix_t matrix)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(matrix); ++i)
    {
        MATRIX_AT_INDEX(matrix, i) = sigmoid(MATRIX_AT_INDEX(matrix, i));
    }
}

void activate_sigmoid_gradient(matrix_t dst, matrix_t sigmoid_output, matrix_t upstream_gradient)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(sigmoid_output); ++i)
    {
        double sig = MATRIX_AT_INDEX(sigmoid_output, i);
        MATRIX_AT_INDEX(dst, i) = sig * (1.0 - sig) * MATRIX_AT_INDEX(upstream_gradient, i);
    }
}

void activate_softmax(matrix_t matrix)
{
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        double max_value = MATRIX_AT(matrix, row, 0);
        for (size_t col = 1; col < matrix.cols; ++col)
        {
            if (MATRIX_AT(matrix, row, col) > max_value)
            {
                max_value = MATRIX_AT(matrix, row, col);
            }
        }
        double sum = 0.0;
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            sum += exp(MATRIX_AT(matrix, row, col) - max_value);
        }

        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = exp(MATRIX_AT(matrix, row, col) - max_value) / sum;
        }
    }
}

void activate_softmax_gradient(matrix_t dst, matrix_t softmax_output, matrix_t upstream_gradient)
{
    for (size_t row = 0; row < softmax_output.rows; ++row)
    {
        for (size_t col = 0; col < softmax_output.cols; ++col)
        {
            double grad_sum = 0.0;
            for (size_t class = 0; class < softmax_output.cols; ++class)
            {
                if (col == class)
                {
                    grad_sum += MATRIX_AT(upstream_gradient, row, class) * MATRIX_AT(softmax_output, row, col) *
                                (1.0 - MATRIX_AT(softmax_output, row, col));
                }
                else
                {
                    grad_sum -= MATRIX_AT(upstream_gradient, row, class) * MATRIX_AT(softmax_output, row, col) *
                                MATRIX_AT(softmax_output, row, class);
                }
            }

            MATRIX_AT(dst, row, col) *= grad_sum * MATRIX_AT(upstream_gradient, row, col);
        }
    }
}

void activate_relu(matrix_t matrix)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(matrix); ++i)
    {
        if (MATRIX_AT_INDEX(matrix, i) < 0.0)
        {
            MATRIX_AT_INDEX(matrix, i) = 0.0;
        }
    }
}

void activate_relu_gradient(matrix_t dst, matrix_t relu_output, matrix_t upstream_gradient)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(relu_output); ++i)
    {
        MATRIX_AT_INDEX(dst, i) = (MATRIX_AT_INDEX(relu_output, i) < 0.0) ? 0.0 : MATRIX_AT_INDEX(upstream_gradient, i);
    }
}

void activate_tanh(matrix_t matrix)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(matrix); ++i)
    {
        MATRIX_AT_INDEX(matrix, i) = tanh(MATRIX_AT_INDEX(matrix, i));
    }
}

void activate_tanh_gradient(matrix_t dst, matrix_t tanh_output, matrix_t upstream_gradient)
{
    for (size_t i = 0; i < MATRIX_ELEMENT_COUNT(tanh_output); ++i)
    {
        double tanh_ = MATRIX_AT_INDEX(tanh_output, i);
        MATRIX_AT_INDEX(dst, i) *= (1.0 - (tanh_ * tanh_)) * MATRIX_AT_INDEX(upstream_gradient, i);
    }
}

void activate(matrix_t matrix, activation_type_t activation_type)
{
    switch (activation_type)
    {
    case SIGMOID:
        activate_sigmoid(matrix);
        return;
    case RELU:
        activate_relu(matrix);
        return;
    case TANH:
        activate_tanh(matrix);
        return;
    case LINEAR:
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
}

void activate_gradient(matrix_t dst, matrix_t activations, matrix_t upstream_gradient,
                       activation_type_t activation_type)
{
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
