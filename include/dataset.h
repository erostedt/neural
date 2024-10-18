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

size_t *range(size_t count);
void permute_rows(matrix_t matrix, const size_t *indices);
void one_hot_encode(matrix_t dst, vector_t classes, size_t class_count);
dataset_t train_test_split(matrix_t features, matrix_t targets, double training_fraction);
void standardize(matrix_t features);
