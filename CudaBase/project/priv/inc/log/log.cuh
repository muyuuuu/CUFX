#ifndef __LOG_CUH__
#define __LOG_CUH__

#include <cstdio>

void ErrorBackTrace(cudaError_t status_code, const char *file, int line_idx);

#define ErrorHandleWithLabel(ret, label)                                                                               \
    do {                                                                                                               \
        if (cudaSuccess != ret) {                                                                                      \
            ErrorBackTrace(ret, __FILE__, __LINE__);                                                                   \
            goto label;                                                                                                \
        }                                                                                                              \
    } while (0)

#define ErrorHandleNoLabel(ret)                                                                                        \
    do {                                                                                                               \
        if (cudaSuccess != ret) { ErrorBackTrace(ret, __FILE__, __LINE__); }                                           \
    } while (0)

#endif