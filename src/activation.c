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
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = sigmoid(MATRIX_AT(matrix, row, col));
        }
    }
}

void activate_sigmoid_gradient(matrix_t sigmoid_output, matrix_t upstream_gradient)
{
    for (size_t row = 0; row < sigmoid_output.rows; ++row)
    {
        for (size_t col = 0; col < sigmoid_output.cols; ++col)
        {
            double sig = MATRIX_AT(sigmoid_output, row, col);
            MATRIX_AT(upstream_gradient, row, col) *= sig * (1.0 - sig);
        }
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

void activate_softmax_gradient(matrix_t softmax_output, matrix_t upstream_gradient)
{
    matrix_subtract(upstream_gradient, softmax_output, upstream_gradient);
}

void activate_relu(matrix_t matrix)
{
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            if (MATRIX_AT(matrix, row, col) < 0.0)
            {
                MATRIX_AT(matrix, row, col) = 0.0;
            }
        }
    }
}

void activate_relu_gradient(matrix_t relu_output, matrix_t upstream_gradient)
{
    for (size_t row = 0; row < relu_output.rows; ++row)
    {
        for (size_t col = 0; col < relu_output.cols; ++col)
        {
            if (MATRIX_AT(relu_output, row, col) < 0.0)
            {
                MATRIX_AT(upstream_gradient, row, col) = 0.0;
            }
        }
    }
}

void activate_tanh(matrix_t matrix)
{
    for (size_t row = 0; row < matrix.rows; ++row)
    {
        for (size_t col = 0; col < matrix.cols; ++col)
        {
            MATRIX_AT(matrix, row, col) = tanh(MATRIX_AT(matrix, row, col));
        }
    }
}

void activate_tanh_gradient(matrix_t tanh_output, matrix_t upstream_gradient)
{
    for (size_t row = 0; row < tanh_output.rows; ++row)
    {
        for (size_t col = 0; col < tanh_output.cols; ++col)
        {
            double tanh_ = MATRIX_AT(tanh_output, row, col);
            MATRIX_AT(upstream_gradient, row, col) *= 1.0 - (tanh_ * tanh_);
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
    case TANH:
        activate_tanh(matrix);
        return;
    case LINEAR:
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
}

void activate_gradient(matrix_t matrix, matrix_t upstream_gradient, activation_type_t activation_type)
{
    switch (activation_type)
    {
    case SIGMOID:
        activate_sigmoid_gradient(matrix, upstream_gradient);
        return;
    case RELU:
        activate_relu_gradient(matrix, upstream_gradient);
        return;
    case TANH:
        activate_tanh_gradient(matrix, upstream_gradient);
        return;
    case LINEAR:
        return;
    default:
        UNREACHABLE("Unexpected activation");
    }
}
