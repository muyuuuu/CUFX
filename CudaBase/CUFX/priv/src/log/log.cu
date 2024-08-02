#include "log.cuh"

void ErrorBackTrace(cudaError_t status_code, const char *file, int line_idx) {
    if (status_code != cudaSuccess) {
        printf("CUDA ERROR: \n \t code = %d\n\t name = %s\n\t desc = %s\n\t file = %s\n\t line = %d\n", status_code,
               cudaGetErrorName(status_code), cudaGetErrorString(status_code), file, line_idx);
    }
}
