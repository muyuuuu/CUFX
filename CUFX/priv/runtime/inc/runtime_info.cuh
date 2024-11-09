#ifndef __RUNTIME_INFO_CUH__
#define __RUNTIME_INFO_CUH__

cudaError_t SetGPU();

int GetCores(cudaDeviceProp &prop);

inline dim3 GetGridSize(const int width, const int height, const int block_width, const int block_height) {
    dim3 gridDim((width + block_width - 1) / block_width, (height + block_height - 1) / block_height);
    return gridDim;
}

#endif