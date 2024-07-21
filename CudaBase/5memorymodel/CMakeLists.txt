cmake_minimum_required(VERSION 3.10)

project(main LANGUAGES CXX CUDA) # 指定项目和语言

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -dlcm=ca")

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA INCLUDE FILES: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
else()
    message(STATUS "Not find CUDA")
endif()

include_directories(main PRIVATE ${CUDA_INCLUDE_DIRS})
add_executable(main main.cu ../tools/common.cuh)
target_link_libraries(main PRIVATE ${CUDA_LIBRARIES})

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")
message(STATUS "${CMAKE_BINARY_DIR}")
install(TARGETS main RUNTIME DESTINATION bin)