#ifndef __EXTERNAL_CUH__
#define __EXTERNAL_CUH__

#include "matrix.cuh"
#include "data_type.cuh"
#include <cuda_runtime.h>

cudaError_t ReductSum(const Matrix &src, void *val);

#endif