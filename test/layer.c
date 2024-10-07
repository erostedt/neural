#include "utest.h"

#include "matrix.h"
#include "vector.h"
#include "layer.h"

matrix_t make_and()
{
    matrix_t and = matrix_alloc(4, 2);
    MATRIX_AT(and, 0, 0) = 0;
    MATRIX_AT(and, 0, 1) = 0;

    MATRIX_AT(and, 1, 0) = 0;
    MATRIX_AT(and, 1, 1) = 0;

    MATRIX_AT(and, 2, 0) = 0;
    MATRIX_AT(and, 2, 1) = 0;

    MATRIX_AT(and, 3, 0) = 1;
    MATRIX_AT(and, 3, 1) = 1;
    return and;
}

UTEST(layer, forward)
{
    layer_t layer = layer_alloc(4, 2, 1);
    MATRIX_AT(layer.weights, 0, 0) = 1.0f;
    MATRIX_AT(layer.weights, 1, 0) = 1.0f;
    VECTOR_AT(layer.biases, 0) = -1.0f;
    matrix_t and = make_and();
    layer_input(&layer, and);
    layer_forward(&layer);
    matrix_t pred = layer.outputs;
    ASSERT_LT(MATRIX_AT(pred, 0, 0), 0);
    ASSERT_LT(MATRIX_AT(pred, 1, 0), 0);
    ASSERT_LT(MATRIX_AT(pred, 2, 0), 0);
    ASSERT_GT(MATRIX_AT(pred, 3, 0), 0);
}

/*
AND (20x1+20x2–30)

OR (20x1+20x2–10)

NOT (-20x1+10)
*/
