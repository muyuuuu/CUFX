#include "op_external.cuh"
#include "clock.cuh"
#include "runtime_info.cuh"

template <typename T>
__device__ T *SharedMemoryProxy() {
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T *>(memory);
}

template <typename T>
__global__ void GemmKernel(T *src1, T *src2, T *dst, std::size_t h, std::size_t k, std::size_t w) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int height = blockIdx.y * blockDim.y + ty;
    const int width = blockIdx.x * blockDim.x + tx;

    if (width >= w || height >= h) {
        return;
    }

    float sum = 0.0f;

    for (int i = 0; i < k; i++) {
        sum += src1[height * k + i] * src2[w * i + width];
    }

    dst[height * w + width] = sum;
}

// block_size 32
template <typename T, int block_size>
__global__ void GemmSharedMemKernel(T *src1, T *src2, T *dst, std::size_t h, std::size_t k, std::size_t w) {
    __shared__ float local_src1[block_size * block_size];
    __shared__ float local_src2[block_size * block_size];

    // auto local_src1 = SharedMemoryProxy<T>();
    // auto local_src2 = SharedMemoryProxy<T>();

    const uint block_idy = blockIdx.y;
    const uint block_idx = blockIdx.x;

    const int thread_idy = threadIdx.x;
    const int thread_idx = threadIdx.y;

    src1 += block_idy * k * block_size;
    src2 += block_idx * block_size;
    dst += (block_size * block_idx + block_size * block_idy * w);

    float sum = 0.0f;

    for (int i = 0; i < k; i += block_size) {
        local_src1[thread_idy * block_size + thread_idx] = src1[thread_idy * k + thread_idx];
        local_src2[thread_idy * block_size + thread_idx] = src2[thread_idy * w + thread_idx];
        __syncthreads();

        for (int j = 0; j < block_size; j++) {
            sum += local_src1[thread_idy * block_size + j] * local_src2[j * block_size + thread_idx];
        }
        __syncthreads();

        src1 += block_size;
        src2 += w * block_size;
    }
    dst[thread_idy * w + thread_idx] = sum;
}

template <typename T>
cudaError_t GemmImpl(const Matrix &src1, const Matrix &src2, Matrix &dst) {
    cudaError_t ret = cudaSuccess;

    int src1_height = src1.height;
    int src1_width = src1.width;
    int src2_width = src2.width;

    const int local_height = 32;
    const int local_width = 32;

    dim3 grid_size = GetGridSize(dst.width, dst.height, local_width, local_height);
    dim3 block_size(local_width, local_height);

    ProfileTime time{"Gemm"};
    time.StartGpuTime();
    // GemmKernel<T><<<grid_size, block_size>>>(src1.GetCudaData<T>(), src2.GetCudaData<T>(), dst.GetCudaData<T>(),
    //                                          src1_height, src1_width, src2_width);
    // GemmSharedMemKernel<T, local_height><<<grid_size, block_size, sizeof(T) * local_height * local_height>>>(
    //     src1.GetCudaData<T>(), src2.GetCudaData<T>(), dst.GetCudaData<T>(), src1_height, src2_width, src1_width);

    GemmSharedMemKernel<T, local_height><<<grid_size, block_size>>>(
        src1.GetCudaData<T>(), src2.GetCudaData<T>(), dst.GetCudaData<T>(), src1_height, src1_width, src2_width);

    time.EndGpuTime();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(dst.SyncToHost<T>());

    return ret;
}

cudaError_t Gemm(const Matrix &src1, const Matrix &src2, Matrix &dst) {
    cudaError_t ret = cudaSuccess;

    if (src1.elem_type != ElemType::ElemFloat || src2.elem_type != ElemType::ElemFloat
        || dst.elem_type != ElemType::ElemFloat) {
        LOGE("only support float matrix for gemm now \n");
        return cudaErrorInvalidValue;
    }

    ret = GemmImpl<float>(src1, src2, dst);
    return ret;
}
