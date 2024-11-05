#include "compare.cuh"
#include "test_case.cuh"
#include "op_external.cuh"
#include "matrix.cuh"

template <typename T, typename U>
T ReductSumCRun(const Matrix &src, T &val) {
    if (val != 0) {
        val = 0;
    }

    for (int h = 0; h < src.height; h++) {
        for (int w = 0; w < src.width; w++) {
            for (int c = 0; c < src.channel; c++) {
                val += src.At<U>(h, w, c);
            }
        }
    }

    return 0;
}

TestCase(CudaOp, ReductSumFloat) {
    Matrix src{ElemType::ElemFloat, {10, 40, 3}, MemoryType::GlobalMemory, IsAsync::IsAsyncFalse};
    cudaError_t cuda_ret = src.MatrixCreate();

    src.GetBytes<float>();

    int c_ret = 0;
    ASSERT_EQ(cuda_ret, 0);

    float c_res = -1;
    float cuda_res = -1;

    // cuda run
    {
        cuda_ret = ReductSum(src, &cuda_res);
        ASSERT_EQ(cuda_ret, 0);
    }

    // C run
    {
        c_ret = ReductSumCRun<float, float>(src, c_res);
        ASSERT_EQ(c_ret, 0); // 运行成功应该返回 0
    }

    ASSERT_NEAREQ(c_res, cuda_res);
}

TestCase(CudaOp, ReductSumInt) {
    Matrix src{ElemType::ElemInt, {4, 4, 3}, MemoryType::GlobalMemory, IsAsync::IsAsyncFalse};
    cudaError_t cuda_ret = src.MatrixCreate();

    src.GetBytes<int>();

    int c_ret = 0;
    ASSERT_EQ(cuda_ret, 0);

    ulong c_res = 1;
    ulong cuda_res = 1;

    // cuda run
    {
        cuda_ret = ReductSum(src, &cuda_res);
        ASSERT_EQ(cuda_ret, 0);
    }

    // C run
    {
        c_ret = ReductSumCRun<ulong, int>(src, c_res);
        ASSERT_EQ(c_ret, 0); // 运行成功应该返回 0
    }

    ASSERT_EQ(c_res, cuda_res);
}