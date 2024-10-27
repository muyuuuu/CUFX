#include "compare.cuh"
#include "test_case.cuh"
#include "op_external.cuh"

TestCase(CudaOp, ReductSum) {
    Matrix src{ElemType::ElemFloat, {512, 512, 3}, MemoryType::GlobalMemory, IsAsync::IsAsyncFalse};
    cudaError_t ret = src.MatrixCreate();
    ASSERT_EQ(ret, 0);

    float c_res = -1;
    float cuda_res = -1;

    ret = ReductSum(src, &c_res);

    std::cerr << c_res << "\n";
}