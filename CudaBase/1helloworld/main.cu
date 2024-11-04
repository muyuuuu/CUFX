#include <cuda_runtime.h>
#include <iostream>

template <typename T>
__global__ void ReductSumKernel(T *input, T *output, int N) {
    extern __shared__ float local_arr[]; // Declare dynamic shared memory

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (global_idx < N) {
        local_arr[tid] = input[global_idx];
    } else {
        local_arr[tid] = 0; // Handle out-of-bounds
    }

    __syncthreads(); // Ensure all threads have loaded their data

    // Perform reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_arr[tid] += local_arr[tid + stride];
        }
        __syncthreads();
    }

    // Write result to output
    if (tid == 0) {
        output[blockIdx.x] = local_arr[0];
    }
}

void launchKernel(float *input, float *output, int N, dim3 grid_size, dim3 block_size) {
    size_t shared_memory_size = block_size.x * sizeof(float); // Size for dynamic shared memory
    ReductSumKernel<float><<<grid_size, block_size, shared_memory_size>>>(input, output, N);
}

void launchKernelInt(int *input, int *output, int N, dim3 grid_size, dim3 block_size) {
    size_t shared_memory_size = block_size.x * sizeof(int); // Size for dynamic shared memory
    ReductSumKernel<int><<<grid_size, block_size, shared_memory_size>>>(input, output, N);
}

int main() {
    // Example of preparing data and launching the kernel for float type
    float *d_input_float, *d_output_float;
    // Allocate and initialize d_input_float and d_output_float...
    launchKernel(d_input_float, d_output_float, 1, {1, 1}, {1, 1});

    // // Example of preparing data and launching the kernel for int type
    // int *d_input_int, *d_output_int;
    // // Allocate and initialize d_input_int and d_output_int...
    // launchKernelInt(d_input_int, d_output_int, 1, {1, 1}, {1, 1});

    return 0;
}
