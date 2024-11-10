#ifndef __COMPARE__CUH__
#define __COMPARE__CUH__

#include "matrix.cuh"
#include "log.cuh"

template <typename T>
int CompareScalar(T val1, T val2) {
    if (val1 == val2) {
        return 1;
    }
    return 0;
}

template <typename T>
int CompareMatrix(const Matrix &src1, const Matrix &src2);

#endif