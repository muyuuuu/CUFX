#ifndef __REDUC_SUM_CUH__
#define __REDUC_SUM_CUH__

#include "matrix.cuh"

template <typename T>
cudaError_t ReductSum(const Matrix<T> &src, const T &val);

#endif __REDUC_SUM_CUH__