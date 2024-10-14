#pragma once

#include "matrix.h"
#include "vector.h"

typedef struct
{
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;

} adam_parameters_t;

typedef struct
{
    matrix_t m_weights;
    matrix_t v_weights;
    vector_t m_biases;
    vector_t v_biases;
} adam_state_t;

static inline adam_parameters_t optimizer_default(double learning_rate)
{
    return (adam_parameters_t){.learning_rate = learning_rate, .beta1 = 0.9, .beta2 = 0.999, .epsilon = 1e-8};
}
