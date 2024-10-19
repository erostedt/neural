#include "comparison.h"
#include "utest.h"

#include "dataset.h"
#include "matrix.h"
#include "vector.h"

UTEST(dataset, range)
{
    size_t* indices = range(10);
    for (size_t i = 0; i < 10; ++i)
    {
        ASSERT_EQ(i, indices[i]);
    }
    free(indices);
}

UTEST(dataset, one_hot_encode)
{
    vector_t classes = {5, (double[]) {0, 1, 2, 1, 0}};
    matrix_t outputs = {5, 3, (double[15]) {0}};

    one_hot_encode(outputs, classes, 3);

    matrix_t expected_outputs = {5, 3, (double[])
        {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        }
    };
    ASSERT_TRUE(matrix_equals(outputs, expected_outputs));
}

UTEST(dataset, train_test_split)
{
    matrix_t features = {10, 2, (double[20]) 
        {
            0, 0,
            1, 1,
            2, 2,
            3, 3,
            4, 4,
            5, 5,
            6, 6,
            7, 7,
            8, 8,
            9, 9,
        }
    };
    matrix_t targets = {10, 1, (double[10]) { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }
    };

    dataset_t dataset = train_test_split(features, targets, 0.7);
    ASSERT_TRUE(dataset.train_features.rows == 7 && dataset.train_features.cols == 2);
    ASSERT_TRUE(dataset.train_targets.rows == 7 && dataset.train_targets.cols == 1);
    ASSERT_TRUE(dataset.test_features.rows == 3 && dataset.test_features.cols == 2);
    ASSERT_TRUE(dataset.test_targets.rows == 3 && dataset.test_targets.cols == 1);
}

UTEST(dataset, calculate_standardization)
{
    matrix_t features = {4, 2, (double[]) {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0}
    };
    standardization_t standardization = calculate_standardization(features);

    ASSERT_EQ(standardization.means.count, 2);
    ASSERT_TRUE(vector_equals(standardization.means, (vector_t){2, (double[]){4.0, 5.0}}));

    ASSERT_EQ(standardization.standard_deviations.count, 2);
    ASSERT_TRUE(vector_equals(standardization.standard_deviations, (vector_t){2, (double[]){2.5819888974716112567, 2.5819888974716112567}}));
}

UTEST(dataset, standardize)
{
    matrix_t features = {4, 2, (double[]) {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0}
    };
    matrix_t dst = {4, 2, (double[8]) {0}};
    standardize(dst, features, calculate_standardization(features));

    vector_t expected_means = {2, (double[2]) {0}};
    vector_t expected_stds = {2, (double[2]) {1.0, 1.0}};

    vector_t means = {2, (double[2]) {0}};
    vector_t stds = {2, (double[2]) {0}};

    for (size_t row = 0; row < dst.rows; ++row)
    {
        for (size_t col = 0; col < dst.cols; ++col)
        {
            VECTOR_AT(means, col) += MATRIX_AT(dst, row, col);
        }
    }

    for (size_t row = 0; row < dst.rows; ++row)
    {
        for (size_t col = 0; col < dst.cols; ++col)
        {
            VECTOR_AT(stds, col) += (MATRIX_AT(dst, row, col) * MATRIX_AT(dst, row, col));
        }
    }

    for (size_t col = 0; col < dst.cols; ++col)
    {
        VECTOR_AT(means, col) /= dst.rows;
        VECTOR_AT(stds, col) = sqrt(VECTOR_AT(stds, col) / (dst.rows-1));
    }

    ASSERT_TRUE(vector_equals(means, expected_means));
    ASSERT_TRUE(vector_equals(stds, expected_stds));
}
