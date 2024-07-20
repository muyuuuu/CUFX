#include <cstdio>

#include "../tools/common.cuh"

#define NUM 10

__device__ int bias[NUM];

extern __shared__ int dynamic_array[];

__constant__ int c_val_0;
__constant__ int c_val_1 = 3;

__global__ void ConstantMemFunc() {
    printf(" ==== Constant Static Mem Test ==== \n");
    printf(" c_val_0 + c_val_1 = %d\n", c_val_0 + c_val_1);
}

__global__ void GloablStaticMemFunc() {
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bias[thread_idx] += 1;
}

__global__ void DynamicStaticMemFunc() {
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= NUM) { return; }

    if (thread_idx < NUM) { dynamic_array[thread_idx] = bias[thread_idx]; }

    __syncthreads();

    if (NUM - 1 == thread_idx) {
        printf(" ==== Dynamic Static Mem Test ==== \n");
        for (int i = 0; i < NUM; i++) { printf("arr[%d] = %d\n", i, dynamic_array[i]); }
    }
}

__global__ void SharedStaticMemFunc() {
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= NUM) { return; }

    __shared__ int static_array[NUM];
    if (thread_idx < NUM) { static_array[thread_idx] = bias[thread_idx]; }

    __syncthreads();

    if (NUM - 1 == thread_idx) {
        printf(" ==== Shared Static Mem Test ==== \n");
        for (int i = 0; i < NUM; i++) { printf("arr[%d] = %d\n", i, static_array[i]); }
    }
}

int main() {
    dim3 global_size(1);
    dim3 local_size(NUM);
    cudaError_t ret;

    int val = 7;

    ret = SetGPU();
    ErrorHandleWithLabel(ret, EXIT);

    int arr[NUM];
    for (int i = 0; i < NUM; i++) { arr[i] = i - 1; }

    ret = cudaMemcpyToSymbol(bias, arr, sizeof(int) * NUM);
    ErrorHandleWithLabel(ret, EXIT);

    GloablStaticMemFunc<<<global_size, local_size>>>();
    ret = cudaDeviceSynchronize();
    ErrorHandleWithLabel(ret, EXIT);

    ret = cudaMemcpyFromSymbol(arr, bias, sizeof(int) * NUM);
    ErrorHandleWithLabel(ret, EXIT);

    for (int i = 0; i < NUM; i++) { printf("arr[%d] = %d\n", i, arr[i]); }

    SharedStaticMemFunc<<<global_size, local_size>>>();
    ret = cudaDeviceSynchronize();
    ErrorHandleWithLabel(ret, EXIT);

    DynamicStaticMemFunc<<<global_size, local_size, NUM>>>();
    ret = cudaDeviceSynchronize();
    ErrorHandleWithLabel(ret, EXIT);

    ret = cudaMemcpyToSymbol(c_val_0, &val, sizeof(int));
    ConstantMemFunc<<<1, 1>>>();
    ret = cudaDeviceSynchronize();

EXIT:
    return 0;
}
