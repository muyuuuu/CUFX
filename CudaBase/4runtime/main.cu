#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>

#include "../tools/common.cuh"
#include "../tools/matrix.cuh"

template <typename T>
cudaError_t AllocMem(TMatrix<T> &matrix, int size) {
    cudaError_t ret;
    size_t n_bytes = matrix.size * sizeof(T);
    matrix.host_addr = (T *)malloc(n_bytes);
    ret = cudaMalloc((T **)&matrix.cuda_addr, n_bytes);
    if ((nullptr == matrix.host_addr) || (nullptr == matrix.cuda_addr) || (ret != cudaSuccess)) {
        printf(" Alloc memory failed \n");
        ErrorHandleNoLabel(ret);
        return cudaErrorInvalidValue;
    } else {
        memset(matrix.host_addr, 0, n_bytes);
        ret = cudaMemset(matrix.cuda_addr, 0, n_bytes);
        ErrorHandleNoLabel(ret);
    }

    srand(666);
    T *ptr = matrix.host_addr;
    for (int i = 0; i < matrix.size; i++) { ptr[i] = (T)(rand() % 255); }

    ret = cudaMemcpy(matrix.cuda_addr, matrix.host_addr, n_bytes, cudaMemcpyHostToDevice);
    ErrorHandleNoLabel(ret);

    return ret;
}

template <typename T>
cudaError_t InitMatrix(TMatrix<T> &matrix, int size = 512) {
    matrix.size = size;
    cudaError_t ret = cudaErrorInvalidValue;

    ret = AllocMem<T>(matrix, size);

    if (cudaSuccess != ret) {
        if (matrix.host_addr != nullptr) { free(matrix.host_addr); }
        if (matrix.cuda_addr != nullptr) { free(matrix.cuda_addr); }
        ErrorHandleNoLabel(ret);
    } else {
        printf("Init Matrix Success\n");
    }

    return ret;
}

template <typename T>
void FreeMatirx(TMatrix<T> &matrix) {
    if (nullptr != matrix.host_addr) { free(matrix.host_addr); }
    if (nullptr != matrix.cuda_addr) { cudaFree(matrix.cuda_addr); }
}

template <typename T>
void MatrixCPUAdd(T *src1, T *src2, T *dst, const int size) {
    for (int i = 0; i < size; i++) { dst[i] = src1[i] + src2[i]; }
}

template <typename T>
__global__ void MatrixGPUAdd(T *src1, T *src2, T *dst, const int size) {
    const int block_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = tid + block_id * blockDim.x;
    if (idx >= size) { return; }

    dst[idx] = src1[idx] + src2[idx];
}

template <typename T>
void MatrixCompare(const TMatrix<T> &m1, const TMatrix<T> &m2, int n_bytes) {
    if (m1.size != m2.size) {
        printf("Matrix dim not same!\n");
    } else {
        bool flag = true;
        for (int i = 0; i < n_bytes; i++) {
            unsigned char v1 = *((unsigned char *)(m1.host_addr) + i);
            unsigned char v2 = *((unsigned char *)(m2.host_addr) + i);
            if (v1 != v2) {
                flag = false;
                break;
            }
        }
        if (flag) {
            printf(" Compare Success \n");
        } else {
            printf(" Compare Failed \n");
        }
    }
}

int main() {
    cudaError_t ret;

    float gpu_time = 0.0f;
    const int EPOCHS = 5;

    TMatrix<int> m1;
    TMatrix<int> m2;
    TMatrix<int> m3;
    TMatrix<int> m4;

    int global_size = 5120000;
    int local_size = 1024;

    dim3 grid_size_1d((global_size + local_size - 1) / local_size);
    dim3 block_size_1d(local_size);

    size_t n_bytes = global_size * m1.elem_byte;

    ret = SetGPU();
    ErrorHandleWithLabel(ret, EXIT);

    ret = InitMatrix(m1, global_size);
    ErrorHandleWithLabel(ret, EXIT);

    ret = InitMatrix(m2, global_size);
    ErrorHandleWithLabel(ret, EXIT);

    ret = InitMatrix(m3, global_size);
    ErrorHandleWithLabel(ret, EXIT);

    ret = InitMatrix(m4, global_size);
    ErrorHandleWithLabel(ret, EXIT);

    for (int i = 0; i < EPOCHS; i++) {
        cudaEvent_t start, end;
        ret = cudaEventCreate(&start);
        ret = cudaEventCreate(&end);
        ret = cudaEventRecord(start, 0);
        ErrorHandleWithLabel(ret, EXIT);

        cudaEventQuery(start);

        MatrixGPUAdd<int><<<grid_size_1d, block_size_1d>>>(m1.cuda_addr, m2.cuda_addr, m3.cuda_addr, global_size);

        ret = cudaEventRecord(end, 0);
        ret = cudaGetLastError();
        ret = cudaEventSynchronize(end);

        ret = cudaEventElapsedTime(&gpu_time, start, end);
        printf(" In Epoch %d, GPU Cost %.4f ms \n", i + 1, gpu_time);
        ErrorHandleWithLabel(ret, EXIT);

        ret = cudaEventDestroy(start);
        ret = cudaEventDestroy(end);
        ErrorHandleWithLabel(ret, EXIT);
    }

    ret = cudaMemcpy(m3.host_addr, m3.cuda_addr, n_bytes, cudaMemcpyDeviceToHost);
    ErrorHandleWithLabel(ret, EXIT);

    for (int i = 0; i < EPOCHS; i++) {
        struct timeval stop, start;
        gettimeofday(&start, NULL);
        MatrixCPUAdd<int>(m1.host_addr, m2.host_addr, m4.host_addr, global_size);
        gettimeofday(&stop, NULL);

        long time = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;

        printf(" In Epoch %d, CPU Cost %.4f ms \n", i + 1, 1.0 * time / 1000.0);
    }

    MatrixCompare(m3, m4, n_bytes);

EXIT:

    FreeMatirx(m1);
    FreeMatirx(m2);
    FreeMatirx(m3);

    return 0;
}