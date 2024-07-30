#include <cstdio>

#include "../tools/common.cuh"
#include "../tools/matrix.cuh"

const int global_size = 512000;
const int local_size = 1024;
const int num_tasks = 10;

template <typename T, size_t size, int N>
__global__ void MatrixAddScalar(T *src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) { return; }

    src[idx] += N * tan((float)(N / 10));
}

__global__ void kernel_1() {
    double sum = 0.0;
    for (int i = 0; i < global_size; i++) sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_2() {
    double sum = 0.0;
    for (int i = 0; i < global_size; i++) sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_3() {
    double sum = 0.0;
    for (int i = 0; i < global_size; i++) sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_4() {
    double sum = 0.0;
    for (int i = 0; i < global_size; i++) sum = sum + tan(0.1) * tan(0.1);
}

void Func1() {
    cudaError_t ret;
    TMatrix<int> src;
    const int step = global_size / num_tasks;

    const size_t n_bytes = sizeof(int) * global_size;
    src.host_addr = (int *)malloc(n_bytes);
    ret = cudaMalloc((int **)&src.cuda_addr, n_bytes);
    ErrorHandleNoLabel(ret);

    cudaStream_t stream_arr[num_tasks];

    for (int i = 0; i < num_tasks; i++) { cudaStreamCreate(&(stream_arr[i])); }

    int *data = (int *)src.host_addr;
    for (int i = 0; i < global_size; i++) { data[i] = rand() % 255; }

    printf(" src[277] = %d\n", data[277]);

    cudaEvent_t start, stop;
    ret = cudaEventCreate(&start);
    ErrorHandleNoLabel(ret);
    ret = cudaEventCreate(&stop);
    ret = cudaEventRecord(start);
    ErrorHandleNoLabel(ret);

    for (int i = 0; i < num_tasks; i++) {
        int offset = i * step;

        ret = cudaMemcpyAsync(src.cuda_addr + offset, src.host_addr + offset, step, cudaMemcpyHostToDevice,
                              stream_arr[i]);
        ErrorHandleNoLabel(ret);

        MatrixAddScalar<int, step, 13><<<step, local_size, 0, stream_arr[i]>>>(src.cuda_addr + offset);

        ret = cudaMemcpyAsync(src.host_addr + offset, src.cuda_addr + offset, step, cudaMemcpyDeviceToHost,
                              stream_arr[i]);
        ErrorHandleNoLabel(ret);
    }

    for (int i = 0; i < num_tasks; i++) { ret = cudaStreamSynchronize(stream_arr[i]); }

    cudaEventRecord(stop);
    // CPU Tasks
    ret = cudaEventSynchronize(stop);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(" Block MatrixAddScalar elapsed time: %.4f ms\n", elapsed_time);

    for (int i = 0; i < num_tasks; i++) {
        ret = cudaStreamDestroy(stream_arr[i]);
        ErrorHandleNoLabel(ret);
    }
    printf(" src[277] = %d\n", data[277]);

    ret = cudaEventRecord(start);
    MatrixAddScalar<int, global_size, 13><<<global_size, local_size>>>(src.cuda_addr);
    ret = cudaEventRecord(stop);
    ret = cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(" Total MatrixAddScalar elapsed time: %.4f ms\n", elapsed_time);

    free(src.host_addr);
    src.host_addr = nullptr;

    cudaFree(src.cuda_addr);
    src.cuda_addr = nullptr;
}

void Func2() {
    cudaError_t ret;
    cudaStream_t stream_arr[num_tasks];
    for (int i = 0; i < num_tasks; i++) { cudaStreamCreate(&(stream_arr[i])); }

    cudaEvent_t wait_tokens[num_tasks];
    for (int i = 0; i < num_tasks; i++) { cudaEventCreateWithFlags(&wait_tokens[i], cudaEventDisableTiming); }

    cudaEvent_t start, stop;
    ret = cudaEventCreate(&start);
    ErrorHandleNoLabel(ret);
    ret = cudaEventCreate(&stop);
    ret = cudaEventRecord(start);

    for (int i = 0; i < num_tasks; i++) {
        kernel_1<<<1, 1, 0, stream_arr[i]>>>();
        kernel_2<<<1, 1, 0, stream_arr[i]>>>();
        kernel_3<<<1, 1, 0, stream_arr[i]>>>();

        cudaEventRecord(wait_tokens[i]);
        cudaStreamWaitEvent(stream_arr[i], wait_tokens[i], 0);

        kernel_4<<<1, 1, 0, stream_arr[i]>>>();
    }
    cudaEventRecord(stop);
    // CPU Tasks
    ret = cudaEventSynchronize(stop);
    ErrorHandleNoLabel(ret);

    for (int i = 0; i < num_tasks; i++) {
        ret = cudaStreamDestroy(stream_arr[i]);
        ErrorHandleNoLabel(ret);
    }

    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(" Func2 elapsed time: %.4f ms\n", elapsed_time);
}

// 回调函数原型，必须符合 cudaStreamCallback_t 的定义
void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *userData) {
    int *p = (int *)userData;
    if (status == cudaSuccess) {
        printf("Callback from stream %d, Wait Event Success \n", *p);
    } else {
        printf("Error in stream callback: %s\n", cudaGetErrorString(status));
    }
}

void Func3() {
    cudaError_t ret;
    cudaStream_t stream_arr[num_tasks];
    for (int i = 0; i < num_tasks; i++) { cudaStreamCreate(&(stream_arr[i])); }

    cudaEvent_t wait_tokens[num_tasks];
    for (int i = 0; i < num_tasks; i++) { cudaEventCreateWithFlags(&wait_tokens[i], cudaEventDisableTiming); }

    cudaEvent_t start, stop;
    ret = cudaEventCreate(&start);
    ErrorHandleNoLabel(ret);
    ret = cudaEventCreate(&stop);
    ret = cudaEventRecord(start);

    int stream_idx[num_tasks];

    for (int i = 0; i < num_tasks; i++) {
        stream_idx[i] = i;
        kernel_1<<<1, 1, 0, stream_arr[i]>>>();
        kernel_2<<<1, 1, 0, stream_arr[i]>>>();
        kernel_3<<<1, 1, 0, stream_arr[i]>>>();

        cudaEventRecord(wait_tokens[i]);
        cudaStreamWaitEvent(stream_arr[i], wait_tokens[i], 0);

        kernel_4<<<1, 1, 0, stream_arr[i]>>>();
        cudaStreamAddCallback(stream_arr[i], my_callback, stream_idx + i, 0);
    }
    cudaEventRecord(stop);
    // CPU Tasks ...
    ret = cudaEventSynchronize(stop);
    ErrorHandleNoLabel(ret);

    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(" Func3 elapsed time: %.4f ms\n", elapsed_time);
}

int main() {
    Func1();
    Func2();
    Func3();
    return 0;
}