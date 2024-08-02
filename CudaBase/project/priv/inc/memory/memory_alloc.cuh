#ifndef __MEMORY_ALLOC_CUH__
#define __MEMORY_ALLOC_CUH__

#include "data_type.cuh"
#include "log.cuh"
#include "matrix.cuh"

template <typename T>
cudaError_t MallocMem(Matrix<T> &matrix);

template <typename T>
cudaError_t UVAAllocMem(Matrix<T> &matrix);

template <typename T>
cudaError_t AllocMem(Matrix<T> &matrix);

#endif __MEMORY_ALLOC_CUH__