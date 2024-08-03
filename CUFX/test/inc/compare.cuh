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
int CompareMatrix(const Matrix &src1, const Matrix &src2) {
    if (src1.GetBytes<T>() != src2.GetBytes<T>()) {
        LOG(CpuLogLevelError, "matrix size not equal");
        return -1;
    }
    bool flag = true;
    for (size_t i = 0; i < src1.GetBytes<T>(); i++) {
        unsigned char v1 = *((unsigned char *)(src1.host_addr) + i);
        unsigned char v2 = *((unsigned char *)(src2.host_addr) + i);
        if (v1 != v2) {
            flag = false;
            break;
        }
    }
    if (flag) {
        LOG(CpuLogLevelInfo, " Compare Success \n");
    } else {
        LOG(CpuLogLevelInfo, " Compare Failed \n");
    }
    return 0;
}

#endif