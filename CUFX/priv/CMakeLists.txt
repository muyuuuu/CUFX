set(PROJECT_PRIV_INC)
set(PROJECT_PRIV_SRC)

# ========================  PRIVATE HEAD FILES ==============
macro(add_priv_dir module)
    list(APPEND PROJECT_PRIV_INC "${CMAKE_SOURCE_DIR}/priv/${module}/inc")
    aux_source_directory("${CMAKE_SOURCE_DIR}/priv/${module}/src" PROJECT_PRIV_SRC)
endmacro(add_priv_dir)

add_priv_dir(runtime)
add_priv_dir(time)

set_property(GLOBAL PROPERTY PROJECT_PRIV_INC ${PROJECT_PRIV_INC})
set_property(GLOBAL PROPERTY PROJECT_PRIV_SRC ${PROJECT_PRIV_SRC})

message( STATUS "[PRIV] PROJECT_PRIV_INC = ${PROJECT_PRIV_INC}")
message( STATUS "[PRIV] PROJECT_PRIV_SRC = ${PROJECT_PRIV_SRC}")