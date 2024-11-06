#ifndef __LOG_CUH__
#define __LOG_CUH__

#include "data_type.cuh"
#include <cstdio>
#include <string>

void ErrorBackTrace(cudaError_t status_code, const char *file, int line_idx);

#define CUDA_CHECK(ret)                                                                                                \
    do {                                                                                                               \
        if (cudaSuccess != ret) {                                                                                      \
            ErrorBackTrace(ret, __FILE__, __LINE__);                                                                   \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK_NO_RET(ret)                                                                                         \
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
        printf("Level : %s  " format, "[I]", ##__VA_ARGS__);                                                           \
    else                                                                                                               \
        printf("Level : %s  " format, "[E]", ##__VA_ARGS__);                                                           \
    fflush(stdout)

#define LOGE(format, ...) LOG(CpuLogLevelError, format, ##__VA_ARGS__)

#define LOGI(format, ...) LOG(CpuLogLevelInfo, format, ##__VA_ARGS__)

#endif