#include <cstdio>

#define MATRIX_NUM 512000

template <typename T, size_t N>
__global__ void MatrixSum(const T *src1, const T *src2, T *dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) { return; }

    dst[idx] = src1[idx] + src2[idx];
    return;
}

template <typename T, size_t N>
void Compare(const T *src1, const T *src2) {
    bool flag = true;
    for (int i = 0; i < N; i++) {
        if (src1[i] != src2[i]) { flag = false; }
        if (false == flag) { break; }
    }
    if (flag) {
        printf(" Compare Success \n");
    } else {
        printf(" Compare Failed \n");
    }
}

void Func() {
    size_t n_bytes = sizeof(int) * MATRIX_NUM;

    int *src1_host = (int *)malloc(n_bytes);
    int *src2_host = (int *)malloc(n_bytes);
    int *dst1_host = (int *)malloc(n_bytes);

    int *src1_cuda = nullptr;
    int *src2_cuda = nullptr;
    int *dst1_cuda = nullptr;
    int *dst2_unify = nullptr;

    cudaMalloc((int **)&src1_cuda, n_bytes);
    cudaMalloc((int **)&src2_cuda, n_bytes);
    cudaMalloc((int **)&dst1_cuda, n_bytes);

    for (int i = 0; i < MATRIX_NUM; i++) {
        src1_host[i] = rand() % 255;
        src2_host[i] = rand() % 255;
    }

    cudaMemcpy(src1_cuda, src1_host, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(src2_cuda, src2_host, n_bytes, cudaMemcpyHostToDevice);
    cudaMallocManaged((void **)&dst2_unify, n_bytes);

    for (int i = 0; i < 3; i++) {
        float gpu_time = 0.0f;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);

        MatrixSum<int, MATRIX_NUM><<<500, 1024>>>(src1_cuda, src2_cuda, dst1_cuda);

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);

        cudaEventElapsedTime(&gpu_time, start, end); // 记录 kernel 执行的事件
        printf(" In Epoch %d, Host Memory Cost %.4f ms \n", i + 1, gpu_time);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    struct timeval stop, start;
    gettimeofday(&start, NULL);
    cudaMemcpy(dst1_host, dst1_cuda, n_bytes, cudaMemcpyDeviceToHost);
    gettimeofday(&stop, NULL);

    long time = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf(" cudaMemcpy Cost %.4f ms \n", 1.0 * time / 1000.0);

    for (int i = 0; i < 3; i++) {
        float gpu_time = 0.0f;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);

        MatrixSum<int, MATRIX_NUM><<<500, 1024>>>(src1_cuda, src2_cuda, dst2_unify);

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&gpu_time, start, end); // 记录 kernel 执行的事件
        printf(" In Epoch %d, Unify Memory Cost %.4f ms \n", i + 1, gpu_time);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    Compare<int, MATRIX_NUM>(dst1_host, dst2_unify);

    return;
}