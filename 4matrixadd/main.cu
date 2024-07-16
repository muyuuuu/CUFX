#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>

#include "../tools/common.cuh"
#include "../tools/matrix.cuh"

size_t GetSize(int size, ElemType type) {
    if (type == ElemInt) {
        return sizeof(int) * size;
    } else if (type == ElemFloat) {
        return sizeof(float) * size;
    }
    return 0;
}

template<typename T>
cudaError_t AllocMem(Matrix& matrix, int size) {
    cudaError_t ret;
    size_t n_bytes = GetSize(matrix.size, matrix.elem_type);
    matrix.host_addr = malloc(n_bytes);
    ret = cudaMalloc((int**)&matrix.cuda_addr, n_bytes);
    if ((nullptr == matrix.host_addr) || (nullptr == matrix.cuda_addr) || (ret != cudaSuccess)) {
        printf(" Alloc memory failed \n");
        ErrorHandleNoLabel(ret);
        return cudaErrorInvalidValue;
    }
    else {
        memset(matrix.host_addr, 0, n_bytes);
        ret = cudaMemset(matrix.cuda_addr, 0, n_bytes);
        ErrorHandleNoLabel(ret);
    }

    srand(666);
    T* ptr = (T*)(matrix.host_addr);
    for (int i = 0; i < matrix.size; i++) {
        ptr[i] = (T)(rand() % 255);
    }

    ret = cudaMemcpy(matrix.cuda_addr, matrix.host_addr, n_bytes, cudaMemcpyHostToDevice);
    ErrorHandleNoLabel(ret);

    return ret;
}

cudaError_t InitMatrix(Matrix &matrix, int size=512, ElemType type=ElemInt) {
    matrix.size = size;
    matrix.elem_type = type;
    cudaError_t ret = cudaErrorInvalidValue;
    
    if (type == ElemInvalid) {
        printf("Please Set Matrix Elem Type\n");
        return ret;
    } else if (ElemInt == type) {
        ret = AllocMem<int>(matrix, size);
    } else if (ElemFloat == type) {
        ret = AllocMem<float>(matrix, size);
    } else {
        printf("Not Supprot Matrix Elem Type [%d] \n", (int)type);
        return ret;
    }

    if (cudaSuccess != ret) {
        if (matrix.host_addr != nullptr) {
            free(matrix.host_addr);
        }
        if (matrix.cuda_addr != nullptr) {
            free(matrix.cuda_addr);
        }
        ErrorHandleNoLabel(ret);
    }
    else {
        printf("Init Matrix Success\n");
    }

    return ret;
}

void FreeMatirx(Matrix& matrix) {
    if (nullptr != matrix.host_addr) {
        free(matrix.host_addr);
    }
    if (nullptr != matrix.cuda_addr) {
        cudaFree(matrix.cuda_addr);
    }
}

template<typename T>
void MatrixCPUAdd(void* m1, void* m2, void* m3, const int size) {

    T* src1 = (T*)(m1);
    T* src2 = (T*)(m2);
    T* dst = (T*)(m3);

    for (int i = 0; i < size; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

template<typename T>
__global__ void MatrixGPUAdd(void* m1, void* m2, void* m3, const int size) {
    const int block_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = tid + block_id * blockDim.x;
    if (idx >= size) {
        return;
    }

    T* src1 = (T*)(m1);
    T* src2 = (T*)(m2);
    T* dst = (T*)(m3);

    dst[idx] = src1[idx] + src2[idx];
}

void MatrixCompare(const Matrix& m1, const Matrix& m2, int n_bytes) {
    if (m1.size != m2.size) {
        printf("Matrix dim not same!\n");
    } else {
        bool flag = true;
        for (int i = 0; i < n_bytes; i++) {
            unsigned char v1 = *((unsigned char*)(m1.host_addr) + i);
            unsigned char v2 = *((unsigned char*)(m2.host_addr) + i);
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

    Matrix m1;
    Matrix m2;
    Matrix m3;
    Matrix m4;

    int global_size = 5120000;
    // int local_size = 3200;
    int local_size = 32;
    ElemType type = ElemFloat;

    dim3 grid_size_1d(global_size / local_size);
    dim3 block_size_1d(local_size);

    size_t n_bytes =  type == ElemInt ? global_size * sizeof(int) : global_size * sizeof(float);
    cudaError_t ret;

    ret = SetGPU();
    ErrorHandleWithLabel(ret, EXIT);

    ret = InitMatrix(m1, global_size, type);
    ErrorHandleWithLabel(ret, EXIT);

    ret = InitMatrix(m2, global_size, type);
    ErrorHandleWithLabel(ret, EXIT);

    ret = InitMatrix(m3, global_size, type);
    ErrorHandleWithLabel(ret, EXIT);

    ret = InitMatrix(m4, global_size, type);
    ErrorHandleWithLabel(ret, EXIT);

    if (ElemInt == type) {
        MatrixGPUAdd<int><<<grid_size_1d, block_size_1d>>>(m1.cuda_addr, m2.cuda_addr, m3.cuda_addr, global_size);
    } else if (ElemFloat == type) {
        MatrixGPUAdd<float><<<grid_size_1d, block_size_1d>>>(m1.cuda_addr, m2.cuda_addr, m3.cuda_addr, global_size);
    }

    ret = cudaGetLastError();
    ErrorHandleWithLabel(ret, EXIT);

    ret = cudaDeviceSynchronize();
    ErrorHandleWithLabel(ret, EXIT);

    ret = cudaMemcpy(m3.host_addr, m3.cuda_addr, n_bytes, cudaMemcpyDeviceToHost);
    ErrorHandleWithLabel(ret, EXIT);

    if (ElemInt == type) {
        MatrixCPUAdd<int>(m1.host_addr, m2.host_addr, m4.host_addr, global_size);
    } else if (ElemFloat == type) {
        MatrixCPUAdd<float>(m1.host_addr, m2.host_addr, m4.host_addr, global_size);
    }

    MatrixCompare(m3, m4, n_bytes);

EXIT:

    FreeMatirx(m1);
    FreeMatirx(m2);
    FreeMatirx(m3);

    return 0;
}