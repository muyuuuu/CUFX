typedef enum ElemType {
    ElemInvalid = 1,
    ElemInt = 2,
    ElemFloat = 3,
} ElemType;

typedef struct Matrix {
    int size;
    void *host_addr;
    void *cuda_addr;
    ElemType elem_type;
    Matrix() {
        host_addr = nullptr;
        cuda_addr = nullptr;
    }
} Matrix;
