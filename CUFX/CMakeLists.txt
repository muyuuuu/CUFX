cmake_minimum_required(VERSION 3.1)

project(CUFX LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -dlcm=ca")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17")

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA INCLUDE FILES: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
else()
    message(STATUS "Not find CUDA")
endif()

add_subdirectory(src)

add_subdirectory(test)