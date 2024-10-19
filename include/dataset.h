#pragma once

#include <stddef.h>

#include "matrix.h"
#include "vector.h"

typedef struct
{
    matrix_t train_features;
    matrix_t train_targets;
    matrix_t test_features;
    matrix_t test_targets;
} dataset_t;

typedef struct
{
    vector_t means;
    vector_t standard_deviations;
} standardization_t;

size_t *range(size_t count);
void one_hot_encode(matrix_t dst, vector_t classes, size_t class_count);
dataset_t train_test_split(matrix_t features, matrix_t targets, double training_fraction);
standardization_t calculate_standardization(matrix_t features);
void standardize(matrix_t dst, matrix_t features, standardization_t standardization);
