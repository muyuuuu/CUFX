#include <cstdio>

__global__ void func() {
    const int grid_dim  = gridDim.x;
    const int block_dim = blockDim.x;

    const int block_idx  = blockIdx.x;
    const int thread_idx = threadIdx.x;

    if (0 == thread_idx && 0 == block_idx) {
        printf(" ==== Grid Dim is [%d, %d] ====\n", grid_dim, block_dim);
    }

    printf(" Local thread is %d, in Block %d, Thread ID is %d\n", 
            thread_idx, 
            block_idx,
            block_idx * block_dim + thread_idx);
}

int main() {
    func<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}