#ifndef LLM_LAB_TEST_SUPPORT_H
#define LLM_LAB_TEST_SUPPORT_H

#include <stdio.h>
#include <stdlib.h>

#define TEST_ASSERT(condition)                                                                     \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "Assertion failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);    \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (0)

#endif
