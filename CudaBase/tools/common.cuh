#include <sys/time.h>
#include <stdio.h>

int GetCores(cudaDeviceProp &prop) {
    int cores = 0;
    int mp = prop.multiProcessorCount;

    switch (prop.major) {
    case 2: // Fermi
        if (prop.minor == 1) {
            cores = mp * 48;
        } else {
            cores = mp * 32;
        }
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if ((prop.minor == 1) || (prop.minor == 2)) {
            cores = mp * 128;
        } else if (prop.minor == 0) {
            cores = mp * 64;
        } else {
            printf("Unknown device type\n");
        }
        break;
    case 7: // Volta and Turing
        if ((prop.minor == 0) || (prop.minor == 5)) {
            cores = mp * 64;
        } else {
            printf("Unknown device type\n");
        }
        break;
    case 8: // Ampere
        if (prop.minor == 0) {
            cores = mp * 64;
        } else if (prop.minor == 6) {
            cores = mp * 128;
        } else if (prop.minor == 9) {
            cores = mp * 128; // ada lovelace
        } else {
            printf("Unknown device type\n");
        }
        break;
    case 9: // Hopper
        if (prop.minor == 0) {
            cores = mp * 128;
        } else {
            printf("Unknown device type\n");
        }
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

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
    printf(" device name          \t %s\n", prop.name);
    printf(" device global mem    \t %lu MB\n", prop.totalGlobalMem / 1024 / 1024);
    printf(" device const  mem    \t %lu KB\n", prop.totalConstMem / 1024);
    printf(" device sms           \t %d \n", prop.multiProcessorCount);
    printf(" Cores                \t %d \n", GetCores(prop));
    printf(" Max threads per block:  %d\n", prop.maxThreadsPerBlock);
    return ret;
}

void ErrorBackTrace(cudaError_t status_code, const char *file, int line_idx) {
    if (status_code != cudaSuccess) {
        printf("CUDA ERROR: \n \t code = %d\n\t name = %s\n\t desc = %s\n\t file = %s\n\t line = %d\n",
               status_code, cudaGetErrorName(status_code), cudaGetErrorString(status_code), file, line_idx);
    }
}

// do while(0) 技巧：https://muyuuuu.github.io/2024/02/03/define-macro/
#define ErrorHandleWithLabel(ret, label)             \
    do {                                             \
        if (cudaSuccess != ret) {                    \
            ErrorBackTrace(ret, __FILE__, __LINE__); \
            goto label;                              \
        }                                            \
    } while (0)

#define ErrorHandleNoLabel(ret)                      \
    do {                                             \
        if (cudaSuccess != ret) {                    \
            ErrorBackTrace(ret, __FILE__, __LINE__); \
        }                                            \
    } while (0)
