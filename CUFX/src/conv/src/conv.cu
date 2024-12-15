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

    for (int dst_c = 0; dst_c <= dst_channel; dst_c++) {
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
                            src[(GetIdxKernel(global_z, src_c, src_h_k, src_w_k, src_channel, src_height, src_width))];
                        sum += ker_val * src_val;
                    }
                }
            }
        }
        dst[GetIdxKernel(global_z, dst_c, global_y, global_x, dst_channel, dst_height, dst_width)] = sum;
    }

    return;
}

template <typename T>
cudaError_t ConvImpl(const Matrix &src, const Matrix &kernel, Matrix &dst) {
    cudaError_t ret = cudaSuccess;

    const int block_x = 32;
    const int block_y = 32;

    dim3 grid_size = Get3DGridSize(dst.width, dst.height, block_x, block_y, dst.batch_size);
    dim3 block_size(block_x, block_y);

    const int batch_size = src.batch_size;
    const int src_channel = src.channel;
    const int src_height = src.height;
    const int src_width = src.width;

    const int dst_channel = dst.channel;
    const int dst_height = dst.height;
    const int dst_width = dst.width;

    const int h_stride = kernel.h_stride;
    const int w_stride = kernel.w_stride;
    const int kernel_height = kernel.height;
    const int kernel_width = kernel.width;

    ProfileTime time{"Conv"};
    time.StartGpuTime();

    NaiveConvKernel<T><<<grid_size, block_size>>>(
        batch_size, src_channel, src_height, src_width, dst_channel, dst_height, dst_width, h_stride, w_stride,
        kernel_height, kernel_width, src.GetCudaData<T>(), kernel.GetCudaData<T>(), dst.GetCudaData<T>());

    time.EndGpuTime(100);
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

    ret = ConvImpl<float>(src, kernel, dst);
    return ret;
}
