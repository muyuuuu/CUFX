#include "compare.cuh"

template <typename T>
int CompareMatrix(const Matrix &src1, const Matrix &src2) {
    if (src1.GetBytes<T>() != src2.GetBytes<T>()) {
        LOG(CpuLogLevelError, "matrix size not equal");
        return -1;
    }
    bool flag = true;
    for (size_t i = 0; i < src1.GetBytes<T>(); i++) {
        unsigned char v1 = *((unsigned char *)(src1.host_addr) + i);
        unsigned char v2 = *((unsigned char *)(src2.host_addr) + i);
        if (v1 != v2) {
            flag = false;
            break;
        }
    }
    if (flag) {
        LOG(CpuLogLevelInfo, " Compare Success \n");
    } else {
        LOG(CpuLogLevelInfo, " Compare Failed \n");
    }
    return 0;
}

template <>
int CompareMatrix<float>(const Matrix &src1, const Matrix &src2) {
    if (src1.GetBytes<float>() != src2.GetBytes<float>()) {
        LOGE("matrix size not equal");
        return -1;
    }

    if (src1.height != src2.height || src1.width != src2.width) {
        LOGE("matrix size not equal");
        return -1;
    }

    for (std::size_t b = 0; b < src1.batch_size; b++) {
        for (std::size_t i = 0; i < src1.height; i++) {
            for (std::size_t j = 0; j < src1.width; j++) {
                for (std::size_t c = 0; c < src1.channel; c++) {
                    int idx = b * src1.height * src1.width * src1.channel + i * src1.width * src1.channel
                              + j * src1.channel + c;
                    float v1 = src1.At<float>(idx);
                    float v2 = src2.At<float>(idx);

                    if (std::fabs(v1 - v2) > 1e-4) {
                        LOGE("mis at (h, w, c) (%zu %zu %zu) v1 = %.4f v2 = %.4f, early return.\n", i, j, c, v1, v2);
                        return -1;
                    }
                }
            }
        }
    }

    LOG(CpuLogLevelInfo, " Compare Success \n");
    return 0;
}