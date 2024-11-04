#include "op_external.cuh"
#include <memory>

template <typename T>
__device__ T *shared_memory_proxy() {
    // do we need an __align__() here? I don't think so...
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T *>(memory);
}

template <typename T1, typename T_matrix>
__global__ void ReductSumKernel(const T_matrix *input, T1 *output, const int h, const int w) {
    auto local_arr = shared_memory_proxy<T1>();

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int height = blockDim.y * blockIdx.y + ty;
    const int width = blockDim.x * blockIdx.x + tx;

    // Load data into shared memory
    if (height < h && width < w) {
        local_arr[ty * blockDim.x + tx] = input[w * height + width];
    } else {
        local_arr[ty * blockDim.x + tx] = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tx < stride && 0 == ty) {
            local_arr[tx] += local_arr[tx + stride];
        }
        __syncthreads();
    }

    if (0 == tx && 0 == ty) {
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

    dim3 grid_size((global_height + local_height - 1) / local_height, (global_width + local_width - 1) / local_width);
    dim3 block_size(local_height, local_width);

    int num_block_size = (grid_size.x / block_size.x) * (grid_size.y / block_size.y);

    T1 *cuda_output = nullptr;
    auto c_output = std::make_unique<T1[]>(num_block_size);
    ret = cudaMalloc(&cuda_output, num_block_size * sizeof(T1));
    ErrorHandleNoLabel(ret);

    ReductSumKernel<T1, T_matrix>
        <<<grid_size, block_size, num_block_size>>>(src.GetData<T_matrix>(), cuda_output, src.height, src.width);

    ret = cudaDeviceSynchronize();

    ErrorHandleNoLabel(ret);
    ret = cudaMemcpy(c_output.get(), cuda_output, num_block_size * sizeof(T1), cudaMemcpyDeviceToHost);
    auto res = c_output[0];
    for (int i = 1; i < num_block_size; i++) {
        res += c_output[i];
    }
    *(T1 *)val = res;
    ret = cudaFree(cuda_output);
    ErrorHandleNoLabel(ret);
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