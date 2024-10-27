#ifndef _DATA_RANDOM_H__
#define _DATA_RANDOM_H__

#include "data_type.cuh"

#include <random>
#include <iostream>

static std::mt19937 mt(20230710);

template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value, void *> = nullptr>
void RandomData(T *data, int height, int width, int channel) {
    std::uniform_real_distribution<float> float_dist(0.0, 1.0);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int c = 0; c < channel; c++) {
                data[i * width * channel + j * channel + c] = float_dist(mt);
            }
        }
    }
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, void *> = nullptr>
void RandomData(T *data, int height, int width, int channel) {
    std::uniform_int_distribution<int> int_dist(0, 255);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int c = 0; c < channel; c++) {
                data[i * width * channel + j * channel + c] = int_dist(mt);
            }
        }
    }
}

#endif