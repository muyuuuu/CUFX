#include "op_external.cuh"
#include "clock.cuh"
#include "runtime_info.cuh"

__device__ int GetIdxKernel(int b = 0, int c = 0, int h = 0, int w = 0, int channel = 0, int height = 0,
                            int width = 0) {
    return b * channel * height * width + c * height * width + h * width + w;
}

template <typename T>
__global__ void NaiveConvKernel(int batch_size, int src_channel, int src_height, int src_width, int dst_channel,
                                int dst_height, int dst_width, int h_stride, int w_stride, int kernel_height,
                                int kernel_width, T *src, T *kernel, T *dst) {
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_z = blockIdx.z; // 第 z 个 batch 的数据

    if (global_x >= dst_width || global_y >= dst_height || global_z >= batch_size) {
        return;
    }

    for (int dst_c = 0; dst_c < dst_channel; dst_c++) {
        float sum = 0.0f;

        int src_h = global_y * h_stride;
        int src_w = global_x * w_stride;

        for (int src_c = 0; src_c < src_channel; src_c++) {
            for (int k_h = 0; k_h < kernel_height; k_h++) {
                for (int k_w = 0; k_w < kernel_width; k_w++) {
                    int src_h_k = src_h + k_h;
                    int src_w_k = src_w + k_w;

                    if (0 <= src_h_k && 0 <= src_w_k && src_w_k < src_width && src_h_k < src_height) {
                        auto ker_val = kernel[GetIdxKernel(0, dst_c, k_h, k_w, 0, kernel_height, kernel_width)];
                        auto src_val =
                            src[GetIdxKernel(global_z, src_c, src_h_k, src_w_k, src_channel, src_height, src_width)];
                        sum += ker_val * src_val;
                    }
                }
            }
        }
        dst[GetIdxKernel(global_z, dst_c, global_y, global_x, dst_channel, dst_height, dst_width)] = sum;
    }

    return;
}

template <typename T, int step, int kernel_height, int kernel_width>
__global__ void Img2ColConvKernel(int batch_size, int src_channel, int src_height, int src_width, int dst_channel,
                                  int dst_height, int dst_width, int h_stride, int w_stride, T *src, T *kernel,
                                  T *dst) {
    const int batch_idx = blockIdx.x; // 第 z 个 batch 的数据
    const int dst_c = blockIdx.y;
    const int out_h = blockIdx.z / (dst_width / step);
    const int out_w = blockIdx.z % (dst_width / step) * step;
    if (out_w >= dst_width || out_h >= dst_height || dst_c >= dst_channel || batch_idx >= batch_size) {
        return;
    }

    __shared__ T local_src[kernel_height * kernel_width * step];
    __shared__ T local_ker[kernel_height * kernel_width];

    const int k_h = (threadIdx.x % step) / kernel_width;
    const int k_w = (threadIdx.x % step) % kernel_width;
    local_ker[threadIdx.x % step] = kernel[dst_c * kernel_width * kernel_height + k_h * kernel_width + k_w];

    T result = 0.0;
    for (int src_c = 0; src_c < src_channel; src_c++) {
        for (int k_h = 0; k_h < kernel_height; k_h++) {
            for (int k_w = 0; k_w < kernel_width; k_w++) {
                const int local_src_h = k_h * kernel_width + k_w;
                const int src_h = out_h + k_h;
                const int src_w = out_w + threadIdx.x + k_w;

                if (src_w < src_width && src_h < src_height) {
                    local_src[local_src_h * step + threadIdx.x] =
                        src[batch_idx * src_channel * src_height * src_width + src_c * src_height * src_width
                            + src_h * src_width + src_w];
                }
            }
        }

        __syncthreads();

        for (int k = 0; k < kernel_height * kernel_width; ++k) {
            result += local_src[k * step + threadIdx.x] * local_ker[k];
        }
    }

    if (out_w + threadIdx.x < dst_width) {
        dst[batch_idx * dst_channel * dst_height * dst_width + dst_c * dst_height * dst_width + out_h * dst_width
            + out_w + threadIdx.x] = result;
    }

    return;
}

template <typename T>
cudaError_t ConvImpl(const Matrix &src, const Matrix &kernel, Matrix &dst) {
    cudaError_t ret = cudaSuccess;

    const int batch_size = src.batch_size;
    const int src_channel = src.channel;
    const int src_height = src.height;
    const int src_width = src.width;

    const int dst_channel = dst.channel;
    const int dst_height = dst.height;
    const int dst_width = dst.width;

    const int h_stride = kernel.h_stride;
    const int w_stride = kernel.w_stride;

    ProfileTime time{"Conv"};
    time.StartGpuTime();

    // {
    //     const int block_x = 32;
    //     const int block_y = 32;
    //     const int kernel_height = kernel.height;
    //     const int kernel_width = kernel.width;
    //     dim3 grid_size = Get3DGridSize(dst.width, dst.height, block_x, block_y, dst.batch_size);
    //     dim3 block_size(block_x, block_y);
    //     NaiveConvKernel<T><<<grid_size, block_size>>>(
    //         batch_size, src_channel, src_height, src_width, dst_channel, dst_height, dst_width, h_stride, w_stride,
    //         kernel_height, kernel_width, src.GetCudaData<T>(), kernel.GetCudaData<T>(), dst.GetCudaData<T>());
    // }

    {
        const int step = 128;
        const int kernel_height = 3;
        const int kernel_width = 3;

        dim3 grid_size(dst.batch_size, dst_channel, dst.height * dst.width / step);
        dim3 block_size(step);

        Img2ColConvKernel<T, step, kernel_height, kernel_width><<<grid_size, block_size>>>(
            batch_size, src_channel, src_height, src_width, dst_channel, dst_height, dst_width, h_stride, w_stride,
            src.GetCudaData<T>(), kernel.GetCudaData<T>(), dst.GetCudaData<T>());
    }

    time.EndGpuTime();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(dst.SyncToHost<T>());

    return ret;
}

cudaError_t Conv(const Matrix &src, const Matrix &kernel, Matrix &dst) {
    cudaError_t ret = cudaSuccess;

    if (src.elem_type != ElemType::ElemFloat || dst.elem_type != ElemType::ElemFloat) {
        LOGE("only support float matrix for conv now \n");
        return cudaErrorInvalidValue;
    }

    cudaMemset(dst.cuda_addr, 0, dst.GetBytes<float>());

    ret = ConvImpl<float>(src, kernel, dst);
    return ret;
}
