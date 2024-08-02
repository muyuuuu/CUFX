cmake_minimum_required(VERSION 3.1)

project(main LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -dlcm=ca")

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA INCLUDE FILES: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
else()
    message(STATUS "Not find CUDA")
endif()

# add_subdirectory(priv)

# add_subdirectory(src)


add_subdirectory(inc)

add_subdirectory(test)