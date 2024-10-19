#include "comparison.h"
#include "utest.h"

#include "loss.h"

UTEST(dataset, loss_alloc)
{
    loss_t loss = loss_alloc(3, 5);
    ASSERT_EQ(loss.gradient.rows, 3);
    ASSERT_EQ(loss.gradient.cols, 5);
}

UTEST(dataset, loss_calculate_mse)
{
    loss_t loss = loss_alloc(3, 2);
    matrix_t y_pred = {3, 2, (double[]){0, 1, 2, 3, 4, 5}};
    matrix_t y_true = {3, 2, (double[]){5, 4, 3, 2, 1, 0}};
    loss_calculate(&loss, MSE, y_pred, y_true);

    double expected_loss = 70.0 / 6.0;
    matrix_t expected_gradient = {3, 2,
                                  (double[]){-10.0 / 6.0, -6.0 / 6.0, -2.0 / 6.0, 2.0 / 6.0, 6.0 / 6.0, 10.0 / 6.0}};

    ASSERT_TRUE(isclose(loss.value, expected_loss));
    ASSERT_TRUE(matrix_equals(loss.gradient, expected_gradient));
}

UTEST(dataset, loss_calculate_binary_cross_entropy)
{
    loss_t loss = loss_alloc(3, 2);
    matrix_t y_pred = {3, 2, (double[]){0, 0.25, 0.5, 0.25, 0.5, 1}};
    matrix_t y_true = {3, 2, (double[]){0, 0, 0, 1, 1, 1}};
    loss_calculate(&loss, BINARY_CROSS_ENTROPY, y_pred, y_true);

    double expected_loss = log(64.0 / 3.0) / 6.0;
    matrix_t expected_gradient = {3, 2, (double[]){0.0, 2.0 / 9.0, 1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 0.0}};

    ASSERT_TRUE(isclose(loss.value, expected_loss));
    ASSERT_TRUE(matrix_equals(loss.gradient, expected_gradient));
}

UTEST(dataset, loss_calculate_categorical_cross_entropy)
{
    loss_t loss = loss_alloc(3, 2);
    matrix_t y_pred = {3, 2, (double[]){0, 1, 1, 0, 0.25, 0.75}};
    matrix_t y_true = {3, 2, (double[]){0, 1, 1, 0, 0, 1}};
    loss_calculate(&loss, CATEGORICAL_CROSS_ENTROPY, y_pred, y_true);

    double expected_loss = log(4.0 / 3.0) / 6.0;
    matrix_t expected_gradient = {3, 2, (double[]){0.0, -1.0 / 6.0, -1.0 / 6.0, 0.0, 0.0, -2.0 / 9.0}};

    ASSERT_TRUE(isclose(loss.value, expected_loss));
    ASSERT_TRUE(matrix_equals(loss.gradient, expected_gradient));
}

/*
 */
