file(GLOB_RECURSE PROJECT_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cuh")  
message(STATUS "PROJECT_INC_DIR = ${PROJECT_INC_DIR}")

include_directories(${PROJECT_INC_DIR})

add_library(cuda_project SHARED ${ALL_MODULE_SRC})