typedef enum ElemType {
    ElemInvalid = 1,
    ElemInt = 2,
    ElemFloat = 3,
} ElemType;

typedef struct Matrix {
    int size;
    void* host_addr;
    void* cuda_addr;
    ElemType elem_type;
    Matrix() {
        host_addr = nullptr;
        cuda_addr = nullptr;
    }
} Matrix;

template<typename T>
struct TMatrix {
    int size;
    T* host_addr;
    T* cuda_addr;
    size_t elem_byte;
    TMatrix() {
        host_addr = nullptr;
        cuda_addr = nullptr;
        size = 0;
        elem_byte = sizeof(T);
    }
};
