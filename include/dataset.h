#pragma once

#include <stddef.h>

#include "matrix.h"
#include "vector.h"

size_t *range(size_t count);
void permute_rows(matrix_t dataset, const size_t *indices);
void one_hot_encode(matrix_t dst, vector_t classes, size_t class_count);
