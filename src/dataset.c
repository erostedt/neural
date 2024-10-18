#include <stdlib.h>

#include "check.h"
#include "operations.h"
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

