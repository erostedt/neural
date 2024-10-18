#pragma once
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT_MSG(expr, msg) _assert(#expr, expr, __FILE__, __LINE__, msg)
#define ASSERT(expr) _assert(#expr, expr, __FILE__, __LINE__, NULL)
#define UNREACHABLE(msg) _unreachable(__FILE__, __LINE__, msg)
#define NOT_IMPLEMENTED() _not_implemented(__FILE__, __FUNCTION__)

static void _assert(const char *expr_str, bool expr, const char *file, int line, const char *msg)
{
    if (!expr)
    {
        if (msg != NULL)
        {
            fprintf(stderr, "Assert failed:\t%s\n", msg);
        }
        else
        {
            fprintf(stderr, "Assert failed\n");
        }

        fprintf(stderr, "Expected:\t%s\n", expr_str);
        fprintf(stderr, "Source:\t\t%s, line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}

static void _unreachable(const char *file, int line, const char *msg)
{
    fprintf(stderr, "Line:\t[%d], in file [%s] should never be reached!\n", line, file);
    fprintf(stderr, "Message:\t%s\n", msg);
    exit(EXIT_FAILURE);
}

static void _not_implemented(const char *file, const char* function)
{
    fprintf(stderr, "Funtion:\t[%s], in file [%s] is not implemented!\n", function, file);
    exit(EXIT_FAILURE);
}
