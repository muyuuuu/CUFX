cudaError_t SetGPU() {
    int n_device = 0;
    int i_device = 0;
    cudaError_t ret;
    
    ret = cudaGetDeviceCount(&n_device);
    if ((0 == n_device) || (cudaSuccess != ret)) {
        printf(" cudaGetDeviceCount got error !\n");
    } else {
        printf(" cudaGetDeviceCount   get [%d] device !\n", n_device);
    }

    ret = cudaSetDevice(i_device);
    if (cudaSuccess != ret) {
        printf(" cudaSetDevice got error !\n");
    } else {
        printf(" cudaSetDevice set device [%d] to run !\n", i_device);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i_device);
    printf(" device name       \t %s\n", prop.name);
    printf(" device global mem \t %lu MB\n", prop.totalGlobalMem / 1024 / 1024);
    printf(" device const  mem \t %lu KB\n", prop.totalConstMem / 1024);
    printf(" device sms        \t %d \n", prop.multiProcessorCount);

    return ret;
}

void ErrorBackTrace(cudaError_t status_code, const char* file, int line_idx) {
    if (status_code != cudaSuccess) {
        printf("CUDA ERROR: \n \t code = %d\n\t name = %s\n\t desc = %s\n\t file = %s\n\t line = %d\n", 
                status_code, cudaGetErrorName(status_code), cudaGetErrorString(status_code), file, line_idx);
    }
}

// do while(0) 技巧：https://muyuuuu.github.io/2024/02/03/define-macro/
#define ErrorHandleWithLabel(ret, label)                  \
    do {                                                  \
        if(cudaSuccess != ret) {                          \
            ErrorBackTrace(ret, __FILE__, __LINE__);      \
            goto label;                                   \
        }                                                 \
    } while(0)

#define ErrorHandleNoLabel(ret)                           \
    do {                                                  \
        if(cudaSuccess != ret) {                          \
            ErrorBackTrace(ret, __FILE__, __LINE__);      \
        }                                                 \
    } while(0)
