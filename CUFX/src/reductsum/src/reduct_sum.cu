#include "external.cuh"

template <typename T>
void ReductSumImpl(T *src) {
}

cudaError_t ReductSum(const Matrix &src, void *val) {
    if (ElemInt == src.elem_type) {
        ReductSumImpl<int>((int *)(src.host_addr));
    }
    return cudaSuccess;
}