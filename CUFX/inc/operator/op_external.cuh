#ifndef __EXTERNAL_CUH__
#define __EXTERNAL_CUH__

#include "matrix.cuh"
#include "data_type.cuh"
#include <cuda_runtime.h>

cudaError_t ReductSum(const Matrix &src, void *val);

cudaError_t GEMM(const Matrix &src1, const Matrix &src2, const Matrix &dst);

#endif