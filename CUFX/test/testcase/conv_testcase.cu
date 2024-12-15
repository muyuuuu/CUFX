#include "compare.cuh"
#include "test_case.cuh"
#include "op_external.cuh"
#include "matrix.cuh"
#include "clock.cuh"

inline int GetIdx(int b = 0, int c = 0, int h = 0, int w = 0, int channel = 0, int height = 0, int width = 0) {
    return b * channel * height * width + c * height * width + h * width + w;
}

template <typename T>
int ConvCPU(const Matrix &src, const Matrix &kernel, Matrix &dst) {
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

    // 以 dst 进行循环
    for (int b = 0; b < batch_size; b++) {
        for (int dst_c = 0; dst_c < dst_channel; dst_c++) {
            for (int dst_h = 0; dst_h < dst_height; dst_h++) {
                for (int dst_w = 0; dst_w < dst_width; dst_w++) {
                    float sum{0.0f};

                    int src_h = dst_h * h_stride;
                    int src_w = dst_w * w_stride;

                    // 处理 src 的通道
                    for (int src_c = 0; src_c < src_channel; src_c++) {
                        // 循环 kernel
                        for (int k_h = 0; k_h < kernel_height; k_h++) {
                            for (int k_w = 0; k_w < kernel_width; k_w++) {
                                int src_h_k = src_h + k_h;
                                int src_w_k = src_w + k_w;

                                if (0 <= src_h_k && 0 <= src_w_k && src_w_k < src_width && src_h_k < src_height) {
                                    auto ker_val =
                                        kernel.At<T>(GetIdx(0, dst_c, k_h, k_w, 0, kernel_height, kernel_width));
                                    auto src_val = src.At<T>(
                                        GetIdx(b, src_c, src_h_k, src_w_k, src_channel, src_height, src_width));
                                    sum += ker_val * src_val;
                                }
                            }
                        }
                    }

                    dst.At<T>(GetIdx(b, dst_c, dst_h, dst_w, dst_channel, dst_height, dst_width)) = sum;
                }
            }
        }
    }
    return 0;
}

TestCase(CudaOp, ConvFloat) {
    const int batch_size = 2;
    const int src_height = 100;
    const int src_width = 100;
    const int src_channel = 12;

    const int kernel_channel = 24;
    const int kernel_height = 3;
    const int kernel_width = 3;
    const int h_stride = 1;
    const int w_stride = 1;

    const int dst_height = (src_height - kernel_height) / h_stride + 1;
    const int dst_width = (src_width - kernel_width) / w_stride + 1;

    Matrix src{ElemType::ElemFloat,
               {src_height, src_width, src_channel},
               MemoryType::GlobalMemory,
               IsAsync::IsAsyncFalse,
               batch_size};

    Matrix kernel{ElemType::ElemFloat,
                  {kernel_height, kernel_width, kernel_channel},
                  MemoryType::GlobalMemory,
                  IsAsync::IsAsyncFalse,
                  batch_size,
                  h_stride,
                  w_stride};

    Matrix dst1{ElemType::ElemFloat,
                {dst_height, dst_width, kernel_channel},
                MemoryType::GlobalMemory,
                IsAsync::IsAsyncFalse,
                batch_size};

    Matrix dst2{ElemType::ElemFloat,
                {dst_height, dst_width, kernel_channel},
                MemoryType::GlobalMemory,
                IsAsync::IsAsyncFalse,
                batch_size};

    CUDA_CHECK_NO_RET(src.MatrixCreate());
    CUDA_CHECK_NO_RET(kernel.MatrixCreate());
    CUDA_CHECK_NO_RET(dst1.MatrixCreate());
    CUDA_CHECK_NO_RET(dst2.MatrixCreate());

    cudaError_t cuda_ret;
    int c_ret{-1};

    // cuda run
    {
        cuda_ret = Conv(src, kernel, dst1);
        ASSERT_EQ(cuda_ret, 0);
    }

    // C run
    {
        ProfileTime time{"Conv"};
        time.StartCPUTime();
        c_ret = ConvCPU<float>(src, kernel, dst2);
        time.EndCPUTime();
        ASSERT_EQ(c_ret, 0);
    }

    c_ret = CompareMatrix<float>(dst1, dst2);
    ASSERT_EQ(c_ret, 0);
}
