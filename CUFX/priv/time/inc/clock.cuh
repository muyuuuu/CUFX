#ifndef __CLOCK_H__
#define __CLOCK_H__

#include "log.cuh"

#include <string>

#define GPU_TIME_START(tag, times, cuda_stream)                                                                        \
    float tag##_avg_time = 0.0f;                                                                                       \
    float tag##_sum_time = 0.0f;                                                                                       \
    cudaError_t tag##_ret;                                                                                             \
    for (int i = 0; i < times; i++) {                                                                                  \
        cudaEvent_t tag##_start, tag##_end;                                                                            \
        tag##_ret = cudaEventCreate(&(tag##_start));                                                                   \
        tag##_ret = cudaEventCreate(&(tag##_end));                                                                     \
        tag##_ret = cudaEventRecord(tag##_start, cuda_stream);                                                         \
        ErrorHandleNoLabel(tag##_ret)

#define GPU_TIME_END(tag, times, cuda_stream)                                                                          \
    tag##_ret = cudaEventRecord(tag##_end, 0);                                                                         \
    tag##_ret = cudaGetLastError();                                                                                    \
    tag##_ret = cudaEventSynchronize(tag##_end);                                                                       \
    tag##_ret = cudaEventElapsedTime(&(tag##_avg_time), tag##_start, tag##_end);                                       \
    tag##_sum_time += tag##_avg_time;                                                                                  \
    tag##_ret = cudaEventDestroy(tag##_start);                                                                         \
    tag##_ret = cudaEventDestroy(tag##_end);                                                                           \
    ErrorHandleWithLabel(tag##_ret, EXIT)                                                                              \
    }                                                                                                                  \
    printf(#tag "use %.4f ms \n", tag##_sum_time / (1.0f * times))

#endif