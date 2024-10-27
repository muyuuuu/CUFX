#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

#include "data_random.cuh"
#include "data_type.cuh"
#include "log.cuh"

class Matrix {
public:
    // 尺寸信息
    int width;
    int height;
    int channel;

    // 元素信息
    MemoryType memory_type;
    ElemType elem_type;
    IsAsync is_async;

    // 地址
    void *host_addr;
    void *cuda_addr;

    // 字节信息
    size_t elem_byte;
    size_t size;
    bool is_matrix_valid;

    // 禁用拷贝
    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;

    // 构造函数
    Matrix(const ElemType &elem_type, const Shape &shape, const MemoryType &memory_type, const IsAsync &is_async);

    // 获取字节数
    template <typename T>
    size_t GetBytes();

    ~Matrix();

    // 创建矩阵
    cudaError_t MatrixCreate();

private:
    // 分配内存
    template <typename T>
    cudaError_t MallocMem();

    // 统一内存分配
    template <typename T>
    cudaError_t UVAAllocMem();
};

#endif