#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

#include "elem_type.cuh"

template <typename T>
struct Matrix {
    int width;
    int height;
    int channel;
    T *host_addr;
    T *cuda_addr;
    size_t elem_byte;
    Matrix() {
        host_addr = nullptr;
        cuda_addr = nullptr;
        size = 0;
        elem_byte = sizeof(T);
    }
};

#endif