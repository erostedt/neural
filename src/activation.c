#include <assert.h>
#include <math.h>

#include "activation.h"

#define UNREACHABLE(message) assert(0 && message)

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

void activate_sigmoid(matrix_t matrix)
{
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = sigmoidf(MATRIX_AT(matrix, row, col));
        }
    }
}

void activate_sigmoid_derivative(matrix_t sigmoid_output, matrix_t upstream_gradient)
{
    for (size_t row = 0; row < sigmoid_output.rows; ++row)
    {
        for (size_t col = 0; col < sigmoid_output.cols; ++col)
        {
            float sigmoid = MATRIX_AT(sigmoid_output, row, col);
            MATRIX_AT(upstream_gradient, row, col) *= sigmoid * (1.0f - sigmoid);
        }
    }
}

void activate_relu(matrix_t matrix)
{
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            if (MATRIX_AT(matrix, row, col) < 0.0f)
            {
                MATRIX_AT(matrix, row, col) = 0.0f;
            }
        }
    }
}

void activate_relu_derivative(matrix_t relu_output, matrix_t upstream_gradient)
{
    for (size_t row = 0; row < relu_output.rows; ++row)
    {
        for (size_t col = 0; col < relu_output.cols; ++col)
        {
            if (MATRIX_AT(relu_output, row, col) < 0.0f)
            {
                MATRIX_AT(upstream_gradient, row, col) = 0.0f;
            }
        }
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
    case LINEAR:
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
}

void activate_derivative(matrix_t matrix, matrix_t upstream_gradient, activation_type_t activation_type)
{
    switch (activation_type)
    {
    case SIGMOID:
        activate_sigmoid_derivative(matrix, upstream_gradient);
        return;
    case RELU:
        activate_relu_derivative(matrix, upstream_gradient);
        return;
    case LINEAR:
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
}
