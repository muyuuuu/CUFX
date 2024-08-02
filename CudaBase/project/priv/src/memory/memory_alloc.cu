#include "memory_alloc.cuh"

template <typename T>
cudaError_t MallocMem(Matrix<T> &matrix) {
    matrix.host_addr = malloc(n_bytes);
    if (nullptr == matrix.host_addr) {
        LOG(CpuLogLevelError, "malloc failed\n");
        return cudaErrorMemoryAllocation;
    }
    cudaError_t ret = cudaMalloc((T **)&matrix.cuda_addr, matrix.GetBytes());
    if ((nullptr == matrix.cuda_addr) || (cudaSuccess != ret)) {
        ErrorHandleNoLabel(ret);
    } else {
        memset(matrix.host_addr, 0, matrix.GetBytes());
        ret = cudaMemset(matrix.cuda_addr, 0, matrix.GetBytes());
        ErrorHandleNoLabel(ret);
    }

    srand(666);
    T *ptr = (T *)(matrix.host_addr);
    for (int i = 0; i < matrix.size; i++) {
        ptr[i] = (T)(rand() % 255);
    }

    ret = cudaMemcpy(matrix.cuda_addr, matrix.host_addr, matrix.GetBytes(), cudaMemcpyHostToDevice);
    ErrorHandleNoLabel(ret);

    return ret;
}

template <typename T>
cudaError_t UVAAllocMem(Matrix<T> &matrix) {
    matrix.host_addr = malloc(n_bytes);
    if (nullptr == matrix.host_addr) {
        LOG(CpuLogLevelError, "malloc failed\n");
        return cudaErrorMemoryAllocation;
    }

    cudaError_t ret = cudaMallocManaged((void **)&matrix.host_addr, matrix.GetBytes());
    ErrorHandleNoLabela(ret);
    return ret;
}

template <typename T>
cudaError_t AllocMem(Matrix<T> &matrix) {
    if ((nullptr != matrix.host_addr) || (nullptr != matrix.cuda_addr)) {
        LOG(CpuLogLevelError, "First Alloc, Should be NULL \n");
    }

    if (MemoryTypeInvalid == matrix.memory_type) {
        LOG(CpuLogLevelError, "Not Suppor Memory Type: %d\n", (int)memory_type);
    }

    cudaError_t ret = cudaSuccess;

    if (GlobalMemory == matrix.memory_type) {
        ret = MallocMem(matrix);
    } else if (ZeroCopyMemory == matrix.memory_type) {
    } else if (UVAMemory == matrix.memory_type) {
        ret = UVAAllocMem(matrix);
    } else {
        LOG(CpuLogLevelError, "Not Support Memory Type %d \n", (int)memory_type);
    }

    return ret;
}