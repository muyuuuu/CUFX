#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

#include "data_random.cuh"
#include "data_type.cuh"
#include "log.cuh"

#include <stdexcept>

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

    // 用于卷积
    int batch_size;
    int h_stride;
    int w_stride;

    // 禁用拷贝
    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;

    // 构造函数
    Matrix(const ElemType &elem_type, const Shape &shape, const MemoryType &memory_type, const IsAsync &is_async,
           const int _batch_size = 1, const int _h_stride = 1, const int _w_stride = 1);

    // 获取字节数
    template <typename T>
    size_t GetBytes() const {
        CheckType<T>();
        return this->height * this->width * this->channel * sizeof(T);
    }

    // 访问矩阵元素
    template <typename T>
    T &At(int h, int w, int c) {
        if (h >= height && w >= width && c >= channel) {
            LOGE("memory out of range ! %d %d %d %d\n", h, height, w, width);
            throw std::out_of_range("memory out of range");
        }
        CheckType<T>();
        T *data = reinterpret_cast<T *>(host_addr);
        return data[h * this->width * this->channel + this->channel * w + c];
    }

    template <typename T>
    const T &At(int h, int w, int c) const {
        if (h >= height && w >= width && c >= channel) {
            LOGE("memory out of range ! %d %d %d %d\n", h, height, w, width);
            throw std::out_of_range("memory out of range");
        }
        CheckType<T>();
        T *data = reinterpret_cast<T *>(host_addr);
        return data[h * this->width * this->channel + this->channel * w + c];
    }

    template <typename T>
    T &At(int idx) {
        if (idx >= batch_size * height * channel * width) {
            LOGE("memory out of range ! %d\n", idx);
            throw std::out_of_range("memory out of range");
        }
        CheckType<T>();
        T *data = reinterpret_cast<T *>(host_addr);
        return data[idx];
    }

    template <typename T>
    const T &At(int idx) const {
        if (idx >= batch_size * height * channel * width) {
            LOGE("memory out of range ! %d\n", idx);
            throw std::out_of_range("memory out of range");
        }
        CheckType<T>();
        T *data = reinterpret_cast<T *>(host_addr);
        return data[idx];
    }

    template <typename T>
    T *GetCudaData() const {
        CheckType<T>();
        T *data = reinterpret_cast<T *>(cuda_addr);
        return data;
    }

    // 内存同步
    template <typename T>
    cudaError_t SyncToHost() {
        return cudaMemcpy(this->host_addr, this->cuda_addr, this->GetBytes<T>(), cudaMemcpyDeviceToHost);
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