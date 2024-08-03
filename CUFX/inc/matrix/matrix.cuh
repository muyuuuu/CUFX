#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

#include "data_type.cuh"
#include "log.cuh"

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
    size_t size;
    bool is_matrix_valid;
    Matrix() = delete;
    Matrix(const ElemType &elem_type, const Shape &shape, const MemoryType &memory_type, const IsAsync &is_async);

    template <typename T>
    size_t GetBytes();

    template <typename T>
    cudaError_t MatrixCreate();

    template <typename T>
    cudaError_t MallocMem();

    template <typename T>
    cudaError_t UVAAllocMem();

    ~Matrix();
};

#endif