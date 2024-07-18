#include <cstdio>

__global__ void func_1d() {
    const int block_id  = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int block_dim = blockDim.x;

    printf("Thread id is %d \n", block_id * block_dim + thread_id);
    return;
}

__global__ void func_2d() {
    const int grid_dim_x       = gridDim.x;
    const int block_id         = blockIdx.y * grid_dim_x + blockIdx.x;
    const int thread_local_id  = threadIdx.x + threadIdx.y * blockDim.x;
    const int thread_id        = thread_local_id + block_id * blockDim.x * blockDim.y;

    printf("Thread id is %d \n", thread_id);
    return;
}

__global__ void func_3d() {
    const int block_id        = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    const int thread_local_id = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_id = thread_local_id + block_id * blockDim.x * blockDim.y * blockDim.z;

    printf("Thread id is %d \n", thread_id);
    return;
}

int main() {

    printf("========== 1D ========== \n");
    dim3 grid_size_1d(2);
    dim3 block_size_1d(4);
    func_1d<<<grid_size_1d, block_size_1d>>>();
    cudaDeviceSynchronize();

    printf("========== 2D ========== \n");
    dim3 grid_size_2d(2, 2);
    dim3 block_size_2d(3, 3);
    func_2d<<<grid_size_2d, block_size_2d>>>();
    cudaDeviceSynchronize();

    printf("========== 3D ========== \n");
    dim3 grid_size_3d(2, 1, 2);
    dim3 block_size_3d(1, 1, 3);
    func_3d<<<grid_size_3d, block_size_3d>>>();
    cudaDeviceSynchronize();

    return 0;
}