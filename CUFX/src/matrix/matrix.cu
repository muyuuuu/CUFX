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
    this->memory_type = memory_type;
    this->size = this->width * this->height * this->channel;
}

cudaError_t Matrix::MatrixCreate() {
    if ((nullptr != this->host_addr) || (nullptr != this->cuda_addr)) {
        LOG(CpuLogLevelError, "First Alloc, Address Should be NULL \n");
    }

    if (MemoryType::MemoryTypeInvalid == this->memory_type) {
        LOG(CpuLogLevelError, "Not Suppor Memory Type: %d\n",
            static_cast<typename std::underlying_type_t<MemoryType>>(memory_type));
    }

    cudaError_t ret = cudaSuccess;

    // 分配全局内存
    if (MemoryType::GlobalMemory == this->memory_type) {
        switch (this->elem_type) {
        case ElemType::ElemInt:
            ret = MallocMem<int>();
            break;
        case ElemType::ElemFloat:
            ret = MallocMem<float>();
        default:
            break;
        }
    } else if (MemoryType::ZeroCopyMemory == this->memory_type) { // 零拷贝内存还没有实现
    } else if (MemoryType::UVAMemory == this->memory_type) {      // 统一内存
        switch (this->elem_type) {
        case ElemType::ElemInt:
            ret = UVAAllocMem<int>();
            break;
        case ElemType::ElemFloat:
            ret = UVAAllocMem<float>();
        default:
            break;
        }
    } else {
        LOGE("Not Support Memory Type %d \n", static_cast<typename std::underlying_type_t<MemoryType>>(memory_type));
    }

    // 内存都分配后  矩阵才合法
    if (cudaSuccess == ret) {
        is_matrix_valid = 1;
    }

    return ret;
}

template <typename T>
cudaError_t Matrix::MallocMem<T>() {
    this->host_addr = malloc(this->GetBytes<T>());
    if (nullptr == this->host_addr) {
        LOGE("malloc failed\n");
        return cudaErrorMemoryAllocation;
    }

    cudaError_t ret = cudaMalloc((T **)&this->cuda_addr, this->GetBytes<T>());
    if ((nullptr == this->cuda_addr) || (cudaSuccess != ret)) {
        ErrorHandleNoLabel(ret);
    } else {
        memset(this->host_addr, 0, this->GetBytes<T>());
        ret = cudaMemset(this->cuda_addr, 0, this->GetBytes<T>());
        ErrorHandleNoLabel(ret);

        // 只有内存都创建成功时 才给 CPU 内存分配随机数
        if (cudaSuccess == ret) {
            RandomData<T>((T *)this->host_addr, this->height, this->width, this->channel);

            ret = cudaMemcpy(this->cuda_addr, this->host_addr, this->GetBytes<T>(), cudaMemcpyHostToDevice);
            ErrorHandleNoLabel(ret);
        }
    }

    return ret;
}

template <typename T>
cudaError_t Matrix::UVAAllocMem<T>() {
    this->host_addr = malloc(this->GetBytes<T>());
    if (nullptr == this->host_addr) {
        LOGE("malloc failed\n");
        return cudaErrorMemoryAllocation;
    } else {
        RandomData<T>((T *)this->host_addr, this->height, this->width, this->channel);
    }

    cudaError_t ret = cudaMallocManaged((void **)&this->host_addr, this->GetBytes<T>());
    ErrorHandleNoLabel(ret);
    return ret;
}

Matrix::~Matrix() {
    if (nullptr != cuda_addr) {
        if (MemoryType::GlobalMemory == memory_type || MemoryType::UVAMemory == memory_type) {
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
