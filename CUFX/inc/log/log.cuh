#ifndef __LOG_CUH__
#define __LOG_CUH__

#include "data_type.cuh"
#include <cstdio>
#include <string>

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
        if (cudaSuccess != ret) {                                                                                      \
            ErrorBackTrace(ret, __FILE__, __LINE__);                                                                   \
        }                                                                                                              \
    } while (0)

typedef enum CpuLogLevel {
    CpuLogLevelInvalid = 1,
    CpuLogLevelInfo = 2,
    CpuLogLevelError = 3,
} CpuLogLevel;

#define LOG(tag, format, ...)                                                                                          \
    if (CpuLogLevelInfo == tag)                                                                                        \
        printf("[%s] [%s %s %d] " format, "[I] ", __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__);                    \
    else                                                                                                               \
        printf("[%s] [%s %s %d] " format, "[E] ", __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__);                    \
    fflush(stdout)

#endif