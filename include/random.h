#include <math.h>
#include <stdlib.h>

static float uniform(float min, float max)
{
    float x = (float)rand() / RAND_MAX;
    return min + x * (max - min);
}

static float normal(float mean, float std)
{
    float u1 = uniform(0.0f, 1.0f);
    float u2 = uniform(0.0f, 1.0f);

    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z0 * std + mean;
}
