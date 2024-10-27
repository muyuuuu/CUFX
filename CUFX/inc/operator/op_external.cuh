#ifndef __EXTERNAL_CUH__
#define __EXTERNAL_CUH__

#include "matrix.cuh"

cudaError_t ReductSum(const Matrix &src, void *val);

#endif