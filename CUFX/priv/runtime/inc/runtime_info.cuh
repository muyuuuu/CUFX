#ifndef __RUNTIME_INFO_CUH__
#define __RUNTIME_INFO_CUH__

cudaError_t SetGPU();

int GetCores(cudaDeviceProp &prop);

#endif