#include "compare.cuh"
#include "test_case.cuh"
#include "op_external.cuh"
#include "matrix.cuh"
#include "clock.cuh"

template <typename T>
int GemmCRun(const Matrix &src1, const Matrix &src2, Matrix &dst) {
    if (src1.width != src2.height) {
        LOGE("dimension error\n");
        return -1;
    }

    for (int h = 0; h < src1.height; h++) {
        for (int w = 0; w < src2.width; w++) {
            float sum = 0.0f;
            for (int k = 0; k < src1.width; k++) {
                sum += src1.At<T>(h, k, 0) * src2.At<T>(k, w, 0);
            }
            dst.At<T>(h, w, 0) = sum;
        }
    }

    return 0;
}

TestCase(CudaOp, GemmFloat) {
    const int dim_h = 512;
    const int dim_k = 512;
    const int dim_w = 512;

    // const int dim_h = 2048;
    // const int dim_k = 1024;
    // const int dim_w = 2048;

    Matrix src1{ElemType::ElemFloat, {dim_h, dim_k, 1}, MemoryType::GlobalMemory, IsAsync::IsAsyncFalse};
    Matrix src2{ElemType::ElemFloat, {dim_k, dim_w, 1}, MemoryType::GlobalMemory, IsAsync::IsAsyncFalse};

    Matrix dst1{ElemType::ElemFloat, {dim_h, dim_w, 1}, MemoryType::GlobalMemory, IsAsync::IsAsyncFalse};
    Matrix dst2{ElemType::ElemFloat, {dim_h, dim_w, 1}, MemoryType::GlobalMemory, IsAsync::IsAsyncFalse};
    CUDA_CHECK_NO_RET(src1.MatrixCreate());
    CUDA_CHECK_NO_RET(src2.MatrixCreate());
    CUDA_CHECK_NO_RET(dst1.MatrixCreate());
    CUDA_CHECK_NO_RET(dst2.MatrixCreate());

    cudaError_t cuda_ret;
    int c_ret;

    // cuda run
    {
        cuda_ret = Gemm(src1, src2, dst1);
        ASSERT_EQ(cuda_ret, 0);
    }

    // C run
    {
        ProfileTime time{"Gemm"};
        time.StartCPUTime();
        c_ret = GemmCRun<float>(src1, src2, dst2);
        time.EndCPUTime();
        ASSERT_EQ(c_ret, 0);
    }

    c_ret = CompareMatrix<float>(dst1, dst2);
    ASSERT_EQ(c_ret, 0);
}
