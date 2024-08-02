#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

#include "data_type.cuh"
#include "memory_alloc.cuh"

template <typename T>
class Matrix {
public:
    int width;
    int height;
    int channel;
    MemoryType memory_type;
    ElemType elem_type;
    IsAsync is_async;
    void *host_addr;
    void *cuda_addr;
    size_t elem_byte;
    bool is_matrix_valid;
    Matrix() = delete;
    Matrix(const ElemType &elem_type, const Shape &shape, const MemoryType &memory_type, const IsAsync &is_async) {
        this->width = shape.width;
        this->height = shape.height;
        this->channel = shape.channel;
        this->cuda_addr = nullptr;
        this->host_addr = nullptr;
        this->elem_type = elem_type;
        this->is_async = is_async;
        is_matrix_valid = false;
    }

    size_t GetBytes() const {
        return this->height * this->width * this->channel * sizeof(T);
    }

    void MatrixCreate() {
        if (cudaSuccess == AllocMem<T>(this)) {
            is_matrix_valid = true;
        }
    }

    ~Matrix() {
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
};

#endif