#ifndef __RUNTIME_INFO_CUH__
#define __RUNTIME_INFO_CUH__

cudaError_t SetGPU();

int GetCores(cudaDeviceProp &prop);

inline dim3 GetGridSize(const int width, const int height, const int block_width, const int block_height) {
    dim3 gridDim((width + block_width - 1) / block_width, (height + block_height - 1) / block_height);
    return gridDim;
}

__device__ inline float4 ReadFloat4(void *pointer) {
    return reinterpret_cast<float4 *>(pointer)[0];
}

__device__ inline int GetOffset(int r, int w, int c) {
    return r * w + c;
}

#endif