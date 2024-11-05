#include "op_external.cuh"
#include <memory>

template <typename T>
__device__ T *SharedMemoryProxy() {
    // do we need an __align__() here? I don't think so...
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T *>(memory);
}

template <typename T1, typename T_matrix>
__global__ void ReductSumKernel(const T_matrix *input, T1 *output, const int h, const int w) {
    auto local_arr = SharedMemoryProxy<T1>();

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int height = blockIdx.y * blockDim.y + ty;
    const int width = blockIdx.x * blockDim.x + tx;

    // Load data into shared memory
    if (height < h && width < w) {
        local_arr[ty * blockDim.x + tx] = input[height * w + width];
    } else {
        local_arr[ty * blockDim.x + tx] = 0;
    }

    __syncthreads();

    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        int idx = ty * blockDim.x + tx;
        if (idx < stride) {
            local_arr[idx] += local_arr[idx + stride];
        }
        __syncthreads();
    }

    if (0 == tx && 0 == ty) {
        printf("======= %.4f \n", local_arr[0]);
        output[blockIdx.x + blockIdx.y * gridDim.x] = local_arr[0];
    }

    return;
}

template <typename T1, typename T_matrix>
cudaError_t ReductSumImpl(const Matrix &src, void *val) {
    cudaError_t ret = cudaSuccess;

    int global_height = src.height;
    int global_width = src.width * src.channel;

    int local_height = 32;
    int local_width = 32;

    dim3 grid_size((global_width + local_width - 1) / local_width, (global_height + local_height - 1) / local_height);
    dim3 block_size(local_width, local_height);

    int num_block_size = grid_size.x * grid_size.y;

    T1 *cuda_output = nullptr;
    ret = cudaMalloc((T1 **)&cuda_output, num_block_size * sizeof(T1));
    CUDA_CHECK(ret);

    ReductSumKernel<T1, T_matrix><<<grid_size, block_size, local_width * local_height * sizeof(T1)>>>(
        src.GetCudaData<T_matrix>(), cuda_output, src.height, src.width * src.channel);

    ret = cudaGetLastError();
    CUDA_CHECK(ret);

    ret = cudaDeviceSynchronize();
    CUDA_CHECK(ret);

    auto c_output = std::make_unique<T1[]>(num_block_size);
    ret = cudaMemcpy(c_output.get(), cuda_output, num_block_size * sizeof(T1), cudaMemcpyDeviceToHost);
    CUDA_CHECK(ret);

    auto res = c_output[0];
    for (int i = 1; i < num_block_size; i++) {
        res += c_output[i];
    }

    *(T1 *)val = res;
    ret = cudaFree(cuda_output);
    CUDA_CHECK(ret);
    return ret;
}

cudaError_t ReductSum(const Matrix &src, void *val) {
    cudaError_t ret = cudaSuccess;

    switch (src.elem_type) {
    case ElemType::ElemInt: {
        {
            ret = ReductSumImpl<uint32_t, int>(src, val);
        }
    } break;
    case ElemType::ElemFloat: {
        {
            ret = ReductSumImpl<float, float>(src, val);
        }
    } break;
    default:
        break;
    }

    return ret;
}