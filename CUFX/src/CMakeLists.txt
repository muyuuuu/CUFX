include("${CMAKE_SOURCE_DIR}/inc/CMakeLists.txt")
include("${CMAKE_SOURCE_DIR}/priv/CMakeLists.txt")

# ====================================================================
# ====================================================================
# ====================================================================
# 定义一个添加新模块的方法
macro(add_common_module module)
    # 搜索代码文件
    set(MODULE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${module}")
    aux_source_directory(${MODULE_SRC} ALL_MODULE_SRC)
endmacro(add_common_module)

add_common_module(log)
add_common_module(matrix)

# ====================================================================
# ====================================================================
# ====================================================================
# 定义一个添加新模块的方法

get_property(PROJECT_PRIV_INC GLOBAL PROPERTY "PROJECT_PRIV_INC")
get_property(PROJECT_PRIV_SRC GLOBAL PROPERTY "PROJECT_PRIV_SRC")

set(MODULE_PRIV_INC)
set(ALL_MODULE_SRC)

list(APPEND MODULE_PRIV_INC ${PROJECT_PRIV_INC})
list(APPEND ALL_MODULE_SRC ${PROJECT_PRIV_SRC})

macro(add_op_module module)
    # 模块自身的头文件视为私有方法，不对外暴露，添加到 MODULE_PRIV_INC 中
    set(MODULE_INC "${CMAKE_CURRENT_SOURCE_DIR}/${module}/inc")
    list(APPEND MODULE_PRIV_INC ${MODULE_INC})

    # 搜索代码文件
    set(MODULE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${module}/src")
    aux_source_directory(${MODULE_SRC} ALL_MODULE_SRC)
endmacro(add_op_module)

add_op_module(reductsum)  # 注册模块


# 制作动态库
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib)
message( STATUS "[SRC] ALL_MODULE_SRC = ${ALL_MODULE_SRC}")
message( STATUS "[SRC] MODULE_PRIV_INC = ${MODULE_PRIV_INC}")

add_library(cufx SHARED ${ALL_MODULE_SRC})
target_include_directories(cufx PRIVATE ${MODULE_PRIV_INC})

get_property(PROJECT_PRIV_INC GLOBAL PROPERTY "PROJECT_PRIV_INC")
include_directories(${PROJECT_INC_DIR})