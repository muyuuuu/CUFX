#ifndef __ELEM_TYPE_CUH__
#define __ELEM_TYPE_CUH__

typedef enum ElemType {
    ElemInvalid = 1,
    ElemInt = 2,
    ElemFloat = 3,
} ElemType;

typedef enum IsAsync {
    IsAsyncInvalid = 1,
    IsAsyncTrue,
    IsAsyncFalse,
} IsAsync;

typedef enum MemoryType {
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

typedef enum CpuLogLevel {
    CpuLogLevelInvalid = 1,
    CpuLogLevelInfo = 2,
    CpuLogLevelError = 3,
} CpuLogLevel;

#endif