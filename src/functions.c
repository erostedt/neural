#include <math.h>

#include "functions.h"

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
