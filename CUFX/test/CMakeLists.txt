include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/testcase SRC_LIST)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc/data_type)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc/log)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc/matrix)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc/operator)

add_executable(test_cufx ${SRC_LIST} main.cu)

target_link_libraries(test_cufx PRIVATE cufx)