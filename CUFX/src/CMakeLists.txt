# ========================  PUBLIC HEAD FILES ==============
set(PROJECT_INC_DIR)
list(APPEND PROJECT_INC_DIR "${CMAKE_SOURCE_DIR}/inc/data_type")
list(APPEND PROJECT_INC_DIR "${CMAKE_SOURCE_DIR}/inc/operator")
list(APPEND PROJECT_INC_DIR "${CMAKE_SOURCE_DIR}/inc/matrix")
list(APPEND PROJECT_INC_DIR "${CMAKE_SOURCE_DIR}/inc/log")

# ========================  PRIVATE HEAD FILES ==============
set_property(GLOBAL PROPERTY PROJECT_PRIV_DIR_HEAD_FILE)

list(APPEND PROJECT_PRIV_DIR_HEAD_FILE "${CMAKE_SOURCE_DIR}/priv/runtime/inc")
aux_source_directory("${CMAKE_SOURCE_DIR}/priv/runtime/src" ALL_MODULE_SRC)

list(APPEND PROJECT_PRIV_DIR_HEAD_FILE "${CMAKE_SOURCE_DIR}/priv/time/inc")
aux_source_directory("${CMAKE_SOURCE_DIR}/priv/time/src" ALL_MODULE_SRC)

list(APPEND PROJECT_PRIV_DIR_HEAD_FILE "${CMAKE_SOURCE_DIR}/priv/testcase/inc")
aux_source_directory("${CMAKE_SOURCE_DIR}/priv/testcase/src" ALL_MODULE_SRC)

# 定义一个添加新模块的方法
macro(add_inc_module module)
    # 搜索代码文件
    set(MODULE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${module}")
    message(STATUS ">>>>>> MODULE_SRC = ${MODULE_SRC}")
    aux_source_directory(${MODULE_SRC} ALL_MODULE_SRC)
endmacro(add_inc_module)

add_inc_module(log)
add_inc_module(matrix)

set(PROJECT_PRIV_DIR ${PROJECT_PRIV_DIR_HEAD_FILE})

# 定义一个添加新模块的方法
macro(add_op_module module)
    # 模块自身的头文件视为私有方法，不对外暴露，添加到 PROJECT_PRIV_DIR 中
    set(MODULE_INC "${CMAKE_CURRENT_SOURCE_DIR}/${module}/inc")
    list(APPEND PROJECT_PRIV_DIR ${MODULE_INC})

    # 搜索代码文件
    set(MODULE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${module}/src")
    aux_source_directory(${MODULE_SRC} ALL_MODULE_SRC)
endmacro(add_op_module)

add_op_module(reductsum)  # 注册模块


# # 制作动态库
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib)
message( STATUS "ALL_MODULE_SRC = ${ALL_MODULE_SRC}")
message( STATUS "ALL_MODULE_INC = ${PROJECT_PRIV_DIR_HEAD_FILE}")


add_library(cufx SHARED ${ALL_MODULE_SRC})
target_include_directories(cufx PRIVATE ${PROJECT_PRIV_DIR})
include_directories(${PROJECT_INC_DIR})