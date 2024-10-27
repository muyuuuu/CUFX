#ifndef __ELEM_TYPE_CUH__
#define __ELEM_TYPE_CUH__

#include <type_traits> // for SFINAE
#include <memory>      // for unique ptr

typedef enum class ElemType : int {
    ElemInvalid = 1,
    ElemInt = 2,
    ElemFloat = 3,
} ElemType;

typedef enum class IsAsync : int {
    IsAsyncInvalid = 1,
    IsAsyncTrue,
    IsAsyncFalse,
} IsAsync;

typedef enum class MemoryType : int {
    MemoryTypeInvalid = 1,
    GlobalMemory,
    ZeroCopyMemory,
    UVAMemory,
} MemoryType;

class Shape {
public:
    Shape() {
        height = 0;
        width = 0;
        channel = 0;
    }
    Shape(int h, int w, int c) : height{h}, width{w}, channel{c} {};

    int height;
    int width;
    int channel;
};

#endif