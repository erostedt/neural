#include "utest.h"

#include "layer.h"
#include "matrix.h"
#include "vector.h"

matrix_t make_and()
{
    matrix_t and = matrix_alloc(4, 2);
    MATRIX_AT(and, 0, 0) = 0;
    MATRIX_AT(and, 0, 1) = 0;

    MATRIX_AT(and, 1, 0) = 1;
    MATRIX_AT(and, 1, 1) = 0;

    MATRIX_AT(and, 2, 0) = 0;
    MATRIX_AT(and, 2, 1) = 1;

    MATRIX_AT(and, 3, 0) = 1;
    MATRIX_AT(and, 3, 1) = 1;
    return and;
}

UTEST(layer, forward)
{
    layer_t layer = layer_alloc(4, 2, (layer_type_t){1, SIGMOID});
    MATRIX_AT(layer.weights, 0, 0) = 20.0;
    MATRIX_AT(layer.weights, 1, 0) = 20.0;
    VECTOR_AT(layer.biases, 0) = -30.0;
    matrix_t and = make_and();
    layer_forward(&layer, and);
    matrix_t pred = layer.outputs;

    ASSERT_LT(MATRIX_AT(pred, 0, 0), 0.5);
    ASSERT_LT(MATRIX_AT(pred, 1, 0), 0.5);
    ASSERT_LT(MATRIX_AT(pred, 2, 0), 0.5);
    ASSERT_GT(MATRIX_AT(pred, 3, 0), 0.5);
}

/*
AND (20x1+20x2–30)

OR (20x1+20x2–10)

NOT (-20x1+10)
*/
