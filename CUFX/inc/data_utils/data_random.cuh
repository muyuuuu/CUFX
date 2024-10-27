#ifndef _DATA_RANDOM_H__
#define _DATA_RANDOM_H__

#include "data_type.cuh"

#include <random>

static std::mt19937 mt(20230701);

template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value, void *> = nullptr>
void RandomData(T *data, std::size_t size) {
    std::uniform_real_distribution<float> float_dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < size; i++) {
        data[i] = float_dist(mt);
    }
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, void *> = nullptr>
void RandomData(T *data, std::size_t size) {
    std::uniform_int_distribution<int> int_dist(0, 255);
    for (std::size_t i = 0; i < size; i++) {
        data[i] = int_dist(mt);
    }
}

#endif