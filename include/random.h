#pragma once
#include <math.h>
#include <stdlib.h>

static double uniform(double min, double max)
{
    double x = (double)rand() / RAND_MAX;
    return min + x * (max - min);
}

static double normal(double mean, double std)
{
    double u1 = uniform(0.0, 1.0);
    double u2 = uniform(0.0, 1.0);

    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z * std + mean;
}
