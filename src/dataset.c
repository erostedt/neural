#include <stdlib.h>

#include "check.h"
#include "operations.h"
#include "random.h"
#include "dataset.h"


size_t *range(size_t count)
{
    size_t *indices = malloc(count * sizeof(count));
    ASSERT(indices != NULL);
    for (size_t i = 0; i < count; ++i)
    {
        indices[i] = i;
    }
    return indices;
}


void permute_rows(matrix_t dataset, const size_t *indices)
{
    if (dataset.rows < 2)
    {
        return;
    }

    vector_t temp = vector_alloc(dataset.cols);
    for (size_t i = 0; i < dataset.rows; ++i)
    {
        size_t j = indices[i];
        vector_copy(temp, row_vector(dataset, i));
        vector_copy(row_vector(dataset, i), row_vector(dataset, j));
        vector_copy(row_vector(dataset, j), temp);
    }
    vector_free(&temp);
}


void one_hot_encode(matrix_t dst, vector_t classes, size_t class_count)
{
    ASSERT(dst.rows == classes.count);
    ASSERT(dst.cols == class_count);
    MATRIX_ZERO(dst);
    for (size_t i = 0; i < VECTOR_ELEMENT_COUNT(classes); ++i)
    {
        size_t class = VECTOR_AT(classes, i);
        ASSERT(class < class_count);
        MATRIX_AT(dst, i, class) = 1.0;
    }
}



void split_into(matrix_t dst1, matrix_t dst2, matrix_t src)
{
    ASSERT(dst1.rows + dst2.rows == src.rows);
    ASSERT(dst1.cols == src.cols);
    ASSERT(dst2.cols == src.cols);

    for (size_t i = 0; i < dst1.rows; ++i)
    {
        vector_copy(row_vector(dst1, i), row_vector(src, i));
    }

    for (size_t i = 0; i < dst2.rows; ++i)
    {
        size_t src_index = dst1.rows + i;
        vector_copy(row_vector(dst2, i), row_vector(src, src_index));
    }
}


dataset_t train_test_split(matrix_t features, matrix_t targets, double train_fraction)
{
    ASSERT(features.rows == targets.rows);
    ASSERT(train_fraction >= 0.0 && train_fraction <= 1.0);

    size_t sample_count = features.rows;
    size_t train_samples = (size_t)(train_fraction * sample_count);
    size_t test_samples = sample_count - train_samples;
    size_t input_size = features.cols;
    size_t output_size = targets.cols;

    size_t *indices = range(sample_count);
    shuffle(indices, sample_count);
    permute_rows(features, indices);
    permute_rows(targets, indices);

    matrix_t train_features = matrix_alloc(train_samples, input_size);
    matrix_t test_features = matrix_alloc(test_samples, input_size);

    matrix_t train_targets = matrix_alloc(train_samples, output_size);
    matrix_t test_targets = matrix_alloc(test_samples, output_size);

    split_into(train_features, test_features, features);
    split_into(train_targets, test_targets, targets);

    return (dataset_t){train_features, train_targets, test_features, test_targets};
}


