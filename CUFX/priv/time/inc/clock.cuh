#ifndef __CLOCK_H__
#define __CLOCK_H__

#include "log.cuh"

#include <string>
#include <chrono>

class ProfileTime {
public:
    ProfileTime(const std::string &tag) : m_tag{tag} {
    }

    ProfileTime(const ProfileTime &other) = delete;
    ProfileTime(ProfileTime &&other) = delete;

    void StartCPUTime();

    void EndCPUTime();

    void StartGpuTime();

    void EndGpuTime();

    ~ProfileTime();

private:
    std::string m_tag;
    cudaEvent_t m_gpu_start;
    cudaEvent_t m_gpu_end;
    float m_gpu_time;

    decltype(std::chrono::high_resolution_clock::now()) m_cpu_start_time;
    decltype(std::chrono::high_resolution_clock::now()) m_cpu_end_time;
};

#endif