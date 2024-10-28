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
    size_t GetBytes() const {
        CheckType<T>();
        return this->height * this->width * this->channel * sizeof(T);
    }

    // 访问矩阵元素
    template <typename T>
    T At(int h, int w, int c) const {
        CheckType<T>();
        T *data = reinterpret_cast<T *>(host_addr);
        return data[h * this->width * this->channel + this->channel * w + c];
    }

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

    template <typename T>
    void CheckType() const {
        switch (this->elem_type) {
        case ElemType::ElemInt:
            assert(std::is_integral<T>::value);
            break;
        case ElemType::ElemFloat:
            assert(std::is_floating_point<T>::value);
            break;
        default:
            break;
        }
    }
};

#endif