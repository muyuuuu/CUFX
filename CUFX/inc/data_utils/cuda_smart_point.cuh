#ifndef __CUDA_SMART_POINT_CUH__
#define __CUDA_SMART_POINT_CUH__

#include "log.cuh"
#include "data_type.cuh"

template <typename T>
T *CudaAlloc(std::size_t size) {
    T *ptr = nullptr;
    CUDA_CHECK_NO_RET(cudaMalloc((void **)&ptr, size));
    return ptr;
}

template <typename T>
class CudaDeleter {
    void operator()(T *ptr) {
        CUDA_CHECK_NO_RET(cudaFree(ptr));
    }
};

template <typename T>
std::shared_ptr<T> CudaSharedPtr(std::size_t num) {
    return std::shared_ptr<T>(CudaAlloc<T>(sizeof(T) * num), CudaDeleter<T>());
}

template <typename T>
std::unique_ptr<T, CudaDeleter<T>> CudaUniquePtr(std::size_t num) {
    return std::unique_ptr<T, CudaDeleter<T>>(CudaAlloc<T>(sizeof(T) * num), CudaDeleter<T>());
}

#endif // __CUDA_SMART_POINT_CUH__