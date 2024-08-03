message(STATUS  "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/testcase SRC_LIST)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test_all.cuh)

add_executable(test_main ${SRC_LIST} main.cu)

# target_link_libraries(test_project PRIVATE project)