#include "vector.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    neural_vector_t v1 = {3, (float[]){1.0f, 2.0f, 3.0f}};
    neural_vector_t v2 = {3, (float[]){4.0f, 5.0f, 6.0f}};
    float dot = neural_vector_dot(v1, v2);
    printf("[1, 2, 3] * [4, 5, 6] = %.0f\n", dot);
    return 0;
}
