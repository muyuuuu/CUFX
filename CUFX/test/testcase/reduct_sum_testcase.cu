#include "testcase.cuh"
#include "compare.cuh"

int TestReductSum() {
    Shape shape{512, 512, 1};
    ElemType elem_type = ElemInt;
    MemoryType mem_type = GlobalMemory;
    IsAsync is_async = IsAsyncFalse;
    Matrix src{elem_type, shape, mem_type, is_async};

    int val1 = 0;
    int val2 = 0;

    cudaError_t ret = ReductSum(src, &val1);

    int res = 0;
    UNIT_TEST(0 == CompareScalar<int>(val1, val2), res);
    return res;
}