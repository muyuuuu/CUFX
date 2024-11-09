#include "clock.cuh"

void ProfileTime::StartCPUTime() {
    m_cpu_start_time = std::chrono::high_resolution_clock::now();
}

void ProfileTime::EndCPUTime() {
    m_cpu_end_time = std::chrono::high_resolution_clock::now();
    auto dur = m_cpu_end_time - m_cpu_start_time;
    auto i_millis = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
    LOGI("%s [CPU] cost %.4f ms\n", m_tag.c_str(), static_cast<float>(i_millis.count()));
}

void ProfileTime::StartGpuTime() {
    CUDA_CHECK_NO_RET(cudaEventCreate(&m_gpu_start));
    CUDA_CHECK_NO_RET(cudaEventCreate(&m_gpu_end));

    CUDA_CHECK_NO_RET(cudaEventRecord(m_gpu_start, 0));
}

void ProfileTime::EndGpuTime() {
    CUDA_CHECK_NO_RET(cudaEventRecord(m_gpu_end, 0));
    CUDA_CHECK_NO_RET(cudaGetLastError());
    CUDA_CHECK_NO_RET(cudaEventSynchronize(m_gpu_end));
    CUDA_CHECK_NO_RET(cudaEventElapsedTime(&m_gpu_time, m_gpu_start, m_gpu_end));
    LOGI("%s [GPU] cost %.4f ms\n", m_tag.c_str(), m_gpu_time);
}

ProfileTime::~ProfileTime() {
    cudaEventDestroy(m_gpu_start);
    cudaEventDestroy(m_gpu_end);
}
