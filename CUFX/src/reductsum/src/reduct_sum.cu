#include "op_external.cuh"

template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value, void *> = nullptr>
cudaError_t ReductSumCImpl(T const *src, T *res) {
    *res = static_cast<T>(1.1f);
    return cudaSuccess;
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, void *> = nullptr>
cudaError_t ReductSumCImpl(T const *src, u_long *res) {
    return cudaSuccess;
}

cudaError_t ReductSum(const Matrix &src, void *val) {
    cudaError_t ret = cudaSuccess;

    // if (ElemType::ElemInt == src.elem_type) {
    //     ret = ReductSumCImpl<int>((int *)(src.host_addr), reinterpret_cast<u_long *>(val));
    // } else if (ElemType::ElemFloat == src.elem_type) {
    //     ret = ReductSumCImpl<float>((float *)(src.host_addr), reinterpret_cast<float *>(val));
    // }
    // ErrorHandleNoLabel(ret);
    return ret;
}