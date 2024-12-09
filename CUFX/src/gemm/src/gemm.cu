#include "op_external.cuh"
#include "clock.cuh"
#include "runtime_info.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

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
    __shared__ T local_src1[block_size * block_size];
    __shared__ T local_src2[block_size * block_size];

    const uint block_idx = blockIdx.x;
    const uint block_idy = blockIdx.y;

    const int thread_idx = threadIdx.x; // % block_size;
    const int thread_idy = threadIdx.y; // / block_size;

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

template <typename T, int bm, int bn, int bk, int tm>
__global__ void GemmSharedMem1DKernel(T *src1, T *src2, T *dst, std::size_t h, std::size_t k, std::size_t w) {
    __shared__ T local_src1[bm * bk];
    __shared__ T local_src2[bk * bn];

    const uint block_idx = blockIdx.x;
    const uint block_idy = blockIdx.y;

    const int thread_idx = threadIdx.x % bn;
    const int thread_idy = threadIdx.x / bn;

    src1 += block_idy * k * bm;
    src2 += block_idx * bn;
    dst += (bm * block_idy * w + bn * block_idx);

    int global_src1 = block_idy * k * bm;
    int global_src2 = bn * block_idx;

    int src1_inner_row = threadIdx.x / bk;
    int src1_inner_col = threadIdx.x % bk;
    int src2_inner_row = threadIdx.x / bn;
    int src2_inner_col = threadIdx.x % bn;

    float thread_res[tm] = {0.0f};

    for (int bk_idx = 0; bk_idx < k; bk_idx += bk) {
        local_src1[src1_inner_row * bk + src1_inner_col] = global_src1 + src1_inner_row * k + src1_inner_col < h * k ?
                                                               src1[src1_inner_row * k + src1_inner_col] :
                                                               0.0f;
        local_src2[src2_inner_row * bn + src2_inner_col] = global_src2 + src2_inner_row * w + src2_inner_col < w * k ?
                                                               src2[src2_inner_row * w + src2_inner_col] :
                                                               0.0f;
        __syncthreads();

        src1 += bk;
        src2 += bk * w;
        global_src1 += bk;
        global_src2 += bk * w;

        for (int idx = 0; idx < bk; idx++) {
            float val = local_src2[idx * bn + thread_idx];
            for (int res_idx = 0; res_idx < tm; res_idx++) {
                thread_res[res_idx] += val * local_src1[(thread_idy * tm + res_idx) * bk + idx];
            }
        }

        __syncthreads();
    }

    for (int res_idx = 0; res_idx < tm; res_idx++) {
        if (block_idy * bm + thread_idy * tm + res_idx < h && block_idx * bn + thread_idx < w) {
            dst[(thread_idy * tm + res_idx) * w + thread_idx] = thread_res[res_idx];
        }
    }

    return;
}

template <typename T, int bm, int bn, int bk, int tm, int tn>
__global__ void __launch_bounds__((bm * bn) / (tm * tn), 1)
    GemmSharedMem2DKernel(T *src1, T *src2, T *dst, std::size_t h, std::size_t k, std::size_t w) {
    __shared__ T local_src1[bm * bk];
    __shared__ T local_src2[bk * bn];

    const uint block_idx = blockIdx.x;
    const uint block_idy = blockIdx.y;

    const int thread_idx = threadIdx.x % (bn / tn);
    const int thread_idy = threadIdx.x / (bn / tn);

    src1 += block_idy * k * bm;
    src2 += block_idx * bn;
    dst += (bm * block_idy * w + bn * block_idx);

    int global_src1 = block_idy * bm * k;
    int global_src2 = bn * block_idx;

    int src1_inner_row = threadIdx.x / bk;
    int src1_inner_col = threadIdx.x % bk;
    int src2_inner_row = threadIdx.x / bn;
    int src2_inner_col = threadIdx.x % bn;

    float thread_res[tm * tn] = {0.0f};

    for (int bk_idx = 0; bk_idx < k; bk_idx += bk) {
        for (uint load_offset = 0; load_offset < bm / tm; load_offset++) {
            local_src1[(src1_inner_row + load_offset * tm) * bk + src1_inner_col] =
                src1[(src1_inner_row + load_offset * tm) * k + src1_inner_col];
        }

        for (uint load_offset = 0; load_offset < bk; load_offset++) {
            local_src2[(src2_inner_row + load_offset) * bn + src2_inner_col] =
                src2[(src2_inner_row + load_offset) * w + src2_inner_col];
        }

        __syncthreads();

        src1 += bk;
        src2 += bk * w;
        global_src1 += bk;
        global_src2 += bk * w;

        for (int idx = 0; idx < bk; idx++) {
            for (int tm_idx = 0; tm_idx < tm; tm_idx++) {
                float val = local_src1[(thread_idy * tm + tm_idx) * bk + idx];
                for (int tn_idx = 0; tn_idx < tn; tn_idx++) {
                    thread_res[tm_idx * tn + tn_idx] += val * local_src2[idx * bn + thread_idx * tn + tn_idx];
                }
            }
        }

        __syncthreads();
    }

    for (int tm_idx = 0; tm_idx < tm; tm_idx++) {
        for (int tn_idx = 0; tn_idx < tn; tn_idx++) {
            if (block_idy * bm + thread_idy * tm + tm_idx < h && block_idx * bn + thread_idx * tn + tn_idx < w) {
                dst[(thread_idy * tm + tm_idx) * w + thread_idx * tn + tn_idx] = thread_res[tm_idx * tn + tn_idx];
            }
        }
    }

    return;
}

template <typename T, int bm, int bn, int bk, int tm, int tn>
__global__ void GemmSharedMem2DVecKernel(T *src1, T *src2, T *dst, std::size_t h, std::size_t k, std::size_t w) {
    __shared__ T local_src1[bm * bk];
    __shared__ T local_src2[bk * bn];

    const uint block_idy = blockIdx.y;
    const uint block_idx = blockIdx.x;

    const int thread_idx = threadIdx.x % (bn / tn);
    const int thread_idy = threadIdx.x / (bn / tn);

    src1 += block_idy * k * bm;
    src2 += block_idx * bn;
    dst += (bm * block_idy * w + bn * block_idx);

    int src1_inner_row = threadIdx.x / (bk / 4);
    int src1_inner_col = threadIdx.x % (bk / 4);
    int src2_inner_row = threadIdx.x / (bn / 4);
    int src2_inner_col = threadIdx.x % (bn / 4);

    float thread_res[tm * tn] = {0.0f};

    // save transpose
    float src1_transpose[4] = {0.f};

    // save tm and tn
    float reg_src1[tm];
    float reg_src2[tn];

    for (int bk_idx = 0; bk_idx < k; bk_idx += bk) {
        for (uint load_offset = 0; load_offset < bm / tm; load_offset += 4) {
            FETCH_FLOAT4(src1_transpose) =
                FETCH_FLOAT4(src1[(src1_inner_row + load_offset * tm) * k + 4 * src1_inner_col]);
            local_src1[src1_inner_row + tm * load_offset + bm * (4 * src1_inner_col + 0)] = src1_transpose[0];
            local_src1[src1_inner_row + tm * load_offset + bm * (4 * src1_inner_col + 1)] = src1_transpose[1];
            local_src1[src1_inner_row + tm * load_offset + bm * (4 * src1_inner_col + 2)] = src1_transpose[2];
            local_src1[src1_inner_row + tm * load_offset + bm * (4 * src1_inner_col + 3)] = src1_transpose[3];
        }

        for (uint load_offset = 0; load_offset < bk; load_offset += 4) {
            FETCH_FLOAT4(local_src2[(src2_inner_row + load_offset) * bn + 4 * src2_inner_col]) =
                FETCH_FLOAT4(src2[(src2_inner_row + load_offset) * w + 4 * src2_inner_col]);
        }

        __syncthreads();

        src1 += bk;
        src2 += bk * w;

        for (int idx = 0; idx < bk; idx++) {
            for (int i = 0; i < tm; i += 4) {
                FETCH_FLOAT4(reg_src1[i]) = FETCH_FLOAT4(local_src1[bm * idx + thread_idy * tm + i]);
            }

            for (int i = 0; i < tn; i += 4) {
                FETCH_FLOAT4(reg_src2[i]) = FETCH_FLOAT4(local_src2[bn * idx + thread_idx * tn + i]);
            }

            for (int m = 0; m < tm; m++) {
                for (int n = 0; n < tn; n++) {
                    thread_res[m * tn + n] += reg_src1[m] * reg_src2[n];
                }
            }
        }

        __syncthreads();
    }

    for (int m = 0; m < tm; m++) {
        for (int n = 0; n < tn; n += 4) {
            FETCH_FLOAT4(dst[(thread_idy * tm + m) * w + thread_idx * tn + n]) = FETCH_FLOAT4(thread_res[m * tn + n]);
        }
    }

    return;
}

template <typename T, int bm, int bn, int bk, int tm, int tn>
__global__ void GemmSharedMem2DVecDoubleBufKernel(T *src1, T *src2, T *dst, std::size_t h, std::size_t k,
                                                  std::size_t w) {
    __shared__ T local_src1[2][bm * bk];
    __shared__ T local_src2[2][bk * bn];

    const uint block_idy = blockIdx.y;
    const uint block_idx = blockIdx.x;

    const int thread_idx = threadIdx.x % (bn / tn);
    const int thread_idy = threadIdx.x / (bn / tn);

    src1 += block_idy * k * bm;
    src2 += block_idx * bn;
    dst += (bm * block_idy * w + bn * block_idx);

    int src1_inner_row = threadIdx.x / (bk / 4);
    int src1_inner_col = threadIdx.x % (bk / 4);
    int src2_inner_row = threadIdx.x / (bn / 4);
    int src2_inner_col = threadIdx.x % (bn / 4);

    float thread_res[tm * tn] = {0.0f};

    // save transpose
    float src1_transpose[4] = {0.f};

    // save tm and tn
    float reg_src1[2][tm] = {0.0f};
    float reg_src2[2][tn] = {0.0f};

    int rw_idx = 0;

    for (int bk_idx = 0; bk_idx < k; bk_idx += bk) {
        for (uint load_offset = 0; load_offset < bm / tm; load_offset += 4) {
            FETCH_FLOAT4(src1_transpose) =
                FETCH_FLOAT4(src1[(src1_inner_row + load_offset * tm) * k + 4 * src1_inner_col]);
            local_src1[rw_idx][src1_inner_row + tm * load_offset + bm * (4 * src1_inner_col + 0)] = src1_transpose[0];
            local_src1[rw_idx][src1_inner_row + tm * load_offset + bm * (4 * src1_inner_col + 1)] = src1_transpose[1];
            local_src1[rw_idx][src1_inner_row + tm * load_offset + bm * (4 * src1_inner_col + 2)] = src1_transpose[2];
            local_src1[rw_idx][src1_inner_row + tm * load_offset + bm * (4 * src1_inner_col + 3)] = src1_transpose[3];
        }

        for (uint load_offset = 0; load_offset < bk; load_offset += 4) {
            FETCH_FLOAT4(local_src2[rw_idx][(src2_inner_row + load_offset) * bn + 4 * src2_inner_col]) =
                FETCH_FLOAT4(src2[(src2_inner_row + load_offset) * w + 4 * src2_inner_col]);
        }

        __syncthreads();

        src1 += bk;
        src2 += bk * w;

        for (int idx = 0; idx < bk; idx++) {
            for (int i = 0; i < tm; i += 4) {
                FETCH_FLOAT4(reg_src1[rw_idx][i]) = FETCH_FLOAT4(local_src1[rw_idx][bm * idx + thread_idy * tm + i]);
            }

            for (int i = 0; i < tn; i += 4) {
                FETCH_FLOAT4(reg_src2[rw_idx][i]) = FETCH_FLOAT4(local_src2[rw_idx][bn * idx + thread_idx * tn + i]);
            }

            for (int m = 0; m < tm; m++) {
                for (int n = 0; n < tn; n++) {
                    thread_res[m * tn + n] += reg_src1[rw_idx][m] * reg_src2[rw_idx][n];
                }
            }
        }

        rw_idx = 1 - rw_idx;
    }

    for (int m = 0; m < tm; m++) {
        for (int n = 0; n < tn; n += 4) {
            FETCH_FLOAT4(dst[(thread_idy * tm + m) * w + thread_idx * tn + n]) = FETCH_FLOAT4(thread_res[m * tn + n]);
        }
    }

    return;
}

template <typename T>
cudaError_t GemmImpl(const Matrix &src1, const Matrix &src2, Matrix &dst) {
    cudaError_t ret = cudaSuccess;

    int src1_height = src1.height;
    int src1_width = src1.width;
    int src2_width = src2.width;

    // dim3 grid_size = GetGridSize(dst.width, dst.height, local_width, local_height);
    // dim3 block_size(local_width, local_height);

    ProfileTime time{"Gemm"};
    time.StartGpuTime();

    // {
    //     const int local_height = 32;
    //     const int local_width = 32;
    //     dim3 grid_size = GetGridSize(dst.width, dst.height, local_width, local_height);
    //     dim3 block_size(local_width, local_height);

    //     // GemmKernel<T><<<grid_size, block_size>>>(src1.GetCudaData<T>(), src2.GetCudaData<T>(),
    //     dst.GetCudaData<T>(),
    //     //                                          src1_height, src1_width, src2_width);

    //     GemmSharedMemKernel<T, local_height><<<grid_size, block_size>>>(
    //         src1.GetCudaData<T>(), src2.GetCudaData<T>(), dst.GetCudaData<T>(), src1_height, src1_width, src2_width);
    // }

    // block 1D
    // {
    //     const int block_m = 64;
    //     const int block_n = 64;
    //     const int block_k = 8;
    //     const int per_elem_thread = 8;

    //     dim3 grid_size = GetGridSize(dst.width, dst.height, block_n, block_m);
    //     dim3 block_size(block_n * block_m / per_elem_thread);

    //     GemmSharedMem1DKernel<T, block_m, block_n, block_k, per_elem_thread><<<grid_size, block_size>>>(
    //         src1.GetCudaData<T>(), src2.GetCudaData<T>(), dst.GetCudaData<T>(), src1_height, src1_width, src2_width);
    // }

    // block 2D
    // {
    //     const int block_m = 64;
    //     const int block_n = 64;
    //     const int block_k = 8;
    //     const int per_elem_thread_x = 8;
    //     const int per_elem_thread_y = 8;

    //     dim3 grid_size = GetGridSize(dst.width, dst.height, block_n, block_m);
    //     dim3 block_size(block_n * block_m / per_elem_thread_x / per_elem_thread_y);

    //     GemmSharedMem2DKernel<T, block_m, block_n, block_k, per_elem_thread_x, per_elem_thread_y>
    //         <<<grid_size, block_size>>>(src1.GetCudaData<T>(), src2.GetCudaData<T>(), dst.GetCudaData<T>(),
    //         src1_height,
    //                                     src1_width, src2_width);
    // }

    // block 2D vec
    // {
    //     const int block_m = 64;
    //     const int block_n = 64;
    //     const int block_k = 8;
    //     const int per_elem_thread_x = 8;
    //     const int per_elem_thread_y = 8;

    //     dim3 grid_size = GetGridSize(dst.width, dst.height, block_n, block_m);
    //     dim3 block_size(block_n * block_m / per_elem_thread_x / per_elem_thread_y);

    //     GemmSharedMem2DVecKernel<T, block_m, block_n, block_k, per_elem_thread_x, per_elem_thread_y>
    //         <<<grid_size, block_size>>>(src1.GetCudaData<T>(), src2.GetCudaData<T>(), dst.GetCudaData<T>(),
    //         src1_height,
    //                                     src1_width, src2_width);
    // }

    // block 2D vec adn double buffer
    {
        const int block_m = 64;
        const int block_n = 64;
        const int block_k = 8;
        const int per_elem_thread_x = 8;
        const int per_elem_thread_y = 8;

        dim3 grid_size = GetGridSize(dst.width, dst.height, block_n, block_m);
        dim3 block_size(block_n * block_m / per_elem_thread_x / per_elem_thread_y);

        for (int i = 0; i < 100; i++) {
            GemmSharedMem2DVecDoubleBufKernel<T, block_m, block_n, block_k, per_elem_thread_x, per_elem_thread_y>
                <<<grid_size, block_size>>>(src1.GetCudaData<T>(), src2.GetCudaData<T>(), dst.GetCudaData<T>(),
                                            src1_height, src1_width, src2_width);
        }
    }

    time.EndGpuTime(100);
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
