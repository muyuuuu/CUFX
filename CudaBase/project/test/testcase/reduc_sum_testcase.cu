#include "reduct_sum.cuh"
#include "test_all.cuh"

void TestReductSum() {
    Shape shape{512, 512, 1};
    ElemType elem_type = ElemInt;
    MemoryType mem_type = GlobalMemory;
    IsAsync is_async = IsAsyncFalse;
    Matrix<int> src{elem_type, shape, mem_type, is_async};

    int val1 = 0;
    int val2 = 0;

    ReductSum<int>(src, val1);

    if (val1 == val2) {
        LOG(CpuLogLevelInfo, " ======= Passed ======= \n");
    } else {
        LOG(CpuLogLevelInfo, " ======= Failed ======= \n");
    }
}