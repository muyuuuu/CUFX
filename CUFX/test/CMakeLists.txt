set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

# inc module
get_property(PROJECT_INC_DIR GLOBAL PROPERTY "PROJECT_INC_DIR")
message(STATUS "[TEST] PROJECT_INC_DIR = ${PROJECT_INC_DIR}")
include_directories(${PROJECT_INC_DIR})

# priv module
get_property(PROJECT_PRIV_INC GLOBAL PROPERTY "PROJECT_PRIV_INC")
include_directories(${PROJECT_PRIV_INC})
message(STATUS "[TEST] PROJECT_PRIV_INC = ${PROJECT_PRIV_INC}")

# testcase module
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/unittest/inc)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/unittest/src SRC_LIST)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/testcase SRC_LIST)

add_executable(test_cufx ${SRC_LIST} main.cu)
target_link_libraries(test_cufx PRIVATE cufx)