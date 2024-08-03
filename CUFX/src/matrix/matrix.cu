#include "matrix.cuh"

Matrix::Matrix(const ElemType &elem_type, const Shape &shape, const MemoryType &memory_type, const IsAsync &is_async) {
    this->width = shape.width;
    this->height = shape.height;
    this->channel = shape.channel;
    this->cuda_addr = nullptr;
    this->host_addr = nullptr;
    this->elem_type = elem_type;
    this->is_async = is_async;
    this->is_matrix_valid = false;
    this->size = this->width * this->height * this->channel;
}

template <typename T>
size_t Matrix::GetBytes() {
    return this->height * this->width * this->channel * sizeof(T);
}

template <typename T>
cudaError_t Matrix::MatrixCreate() {
    if ((nullptr != this->host_addr) || (nullptr != this->cuda_addr)) {
        LOG(CpuLogLevelError, "First Alloc, Should be NULL \n");
    }

    if (MemoryTypeInvalid == this->memory_type) {
        LOG(CpuLogLevelError, "Not Suppor Memory Type: %d\n", (int)memory_type);
    }

    cudaError_t ret = cudaSuccess;

    if (GlobalMemory == this->memory_type) {
        ret = MallocMem<T>();
    } else if (ZeroCopyMemory == this->memory_type) {
    } else if (UVAMemory == this->memory_type) {
        ret = UVAAllocMem<T>();
    } else {
        LOG(CpuLogLevelError, "Not Support Memory Type %d \n", (int)memory_type);
    }

    if (cudaSuccess == ret) {
        is_matrix_valid = 1;
    }

    return ret;
}

template <typename T>
cudaError_t Matrix::MallocMem() {
    this->host_addr = malloc(this->GetBytes<T>());
    if (nullptr == this->host_addr) {
        LOG(CpuLogLevelError, "malloc failed\n");
        return cudaErrorMemoryAllocation;
    }
    cudaError_t ret = cudaMalloc((T **)&this->cuda_addr, this->GetBytes<T>());
    if ((nullptr == this->cuda_addr) || (cudaSuccess != ret)) {
        ErrorHandleNoLabel(ret);
    } else {
        memset(this->host_addr, 0, this->GetBytes<T>());
        ret = cudaMemset(this->cuda_addr, 0, this->GetBytes<T>());
        ErrorHandleNoLabel(ret);
    }

    srand(666);
    T *ptr = (T *)(this->host_addr);
    for (int i = 0; i < this->size; i++) {
        ptr[i] = (T)(rand() % 255);
    }

    ret = cudaMemcpy(this->cuda_addr, this->host_addr, this->GetBytes<T>(), cudaMemcpyHostToDevice);
    ErrorHandleNoLabel(ret);

    return ret;
}

template <typename T>
cudaError_t Matrix::UVAAllocMem() {
    this->host_addr = malloc(this->GetBytes<T>());
    if (nullptr == this->host_addr) {
        LOG(CpuLogLevelError, "malloc failed\n");
        return cudaErrorMemoryAllocation;
    }

    cudaError_t ret = cudaMallocManaged((void **)&this->host_addr, this->GetBytes<T>());
    ErrorHandleNoLabel(ret);
    return ret;
}

Matrix::~Matrix() {
    if (nullptr != cuda_addr) {
        if (GlobalMemory == memory_type) {
            cudaFree(cuda_addr);
        } else {
            cudaFreeHost(cuda_addr);
        }
        cuda_addr = nullptr;
    }
    if (nullptr != host_addr) {
        free(host_addr);
        host_addr = nullptr;
    }
}
