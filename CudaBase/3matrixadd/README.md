# 矩阵加法

注意，代码在这里会悄悄的上一个强度，尽量让代码工程化。代码依然采用 `C` 进行编写，写完了还是感觉 `C++` 面向对象方便一些，后面有机会重构一下代码吧。

## 矩阵设置

在进行矩阵加法时，我们需要为矩阵进行一些基本的设置，比如矩阵的大小，数据类型，以及矩阵数据的存放地址。

为了简单起见，我们以一维矩阵为例：

- 用 `size` 表示矩阵的大小。
- 用枚举类型 `ElemType` 表述矩阵的元素类型，这里只设置了 `int` 和 `float` 类型。由于枚举变量会自动将第一个元素初始化为 0。为了避免创建枚举变量而没有初始化的 bug，所以把枚举中的第一个元素设置为违法类型，且取值为 1。
- 考虑到易用性，为了随时随地的创建矩阵。这里我们将矩阵内存分配到堆空间而非栈空间。用 `host_addr` 表示矩阵元素在 CPU 侧的存储位置，用 `cuda_addr` 表示矩阵元素在 GPU 存储位置。因为如果不是共享内存的话，需要在 CPU 端创建矩阵，并传递给 GPU，在 GPU 端完成计算。

```c
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
        elem_type = ElemInvalid;
        size = 0;
    }
} Matrix;
```

并且提供默认的构造函数，这样创建矩阵时，矩阵内部的元素就不会是随机数，引发奇怪的 `bug`。~~每一行代码都是很长时间 debug 的教训......~~

我们把代码放到 `tools` 目录的 `matrix.cuh` 文件中，`cuh` 可以理解为 `cuda` 的头文件。并在 `main.cu` 中进行引用即可：

```c
#include "../tools/matrix.cuh"
```

## cuda 矩阵加法

因为要做好 cuda 矩阵加法，需要做很多的准备工作，如运行信息查询、矩阵设置、避免内存泄漏、异常日志准备等等等等。为了防止看不下去，直接把 Kernel 函数放在前面，后面的不想看也可以，但我推荐看一看。

首先通过 global_size 设置矩阵的大小为 5120000，并将 block 设置的大小设置为 32，这样每个线程快就有 32 个线程。那么此时，我们就需要设置 5120000 / 32 = 160000 个 grid。这里我们使用 1 维 grid 和 1 维 block。

```c
int global_size = 5120000;
int local_size = 32;
ElemType type = ElemFloat;

dim3 grid_size_1d(global_size / local_size);
dim3 block_size_1d(local_size);
```

为了处理矩阵不同的数据类型，如 int 或者 float 等，我们使用了模板：

```c
template<typename T>
__global__ void MatrixGPUAdd(void* m1, void* m2, void* m3, const int size) {
    const int block_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = tid + block_id * blockDim.x;
    if (idx >= size) {
        return;
    }

    T* src1 = (T*)(m1);
    T* src2 = (T*)(m2);
    T* dst = (T*)(m3);
    
    dst[idx] = src1[idx] + src2[idx];
}
```

因为矩阵加法肯定要传入矩阵的地址，由于将地址声明为空指针，所以还要在这里强制转换为 T 类型的指针进行读取和存储。通过在第二部分计算线程索引的方式得到每一个线程的 id，也就是对应到矩阵中的每个元素的索引，对应位置完成矩阵加法即可。

不过需要加一个判断：当索引超出矩阵大小时会面临内存越界的情况，因为下面这种情况肯定是报错的：

```c
int arr[10];
arr[11] = ...
```

之后在主函数中进行调用即可：

```c
if (ElemInt == type) {
    MatrixGPUAdd<int><<<grid_size_1d, block_size_1d>>>(m1.cuda_addr, m2.cuda_addr, m3.cuda_addr, global_size);
} else if (ElemFloat == type) {
    MatrixGPUAdd<float><<<grid_size_1d, block_size_1d>>>(m1.cuda_addr, m2.cuda_addr, m3.cuda_addr, global_size);
}

ret = cudaGetLastError();  // 捕捉 kernel 函数中可能发生的错误
ret = cudaDeviceSynchronize();
```

至于 `m1.cuda_addr` 这种东西是什么，可以看下文的内存设置，为矩阵申请 GPU 内存。

## 内存设置

这里还没有接触到共享内存的概念，所以需要在 `CPU` 端创建好内存，并传递给 `GPU`，在 `GPU` 端完成矩阵加法。

首先创建一个矩阵，以及初始化矩阵的大小、类型：

```c
Matrix m4;                    // 创建矩阵
int global_size = 5120000;    // 矩阵大小
ElemType type = ElemFloat;    // 矩阵类型
```

而后，设计一个名字叫 `InitMatrix` 的函数去初始化矩阵：

```c
cudaError_t InitMatrix(Matrix &matrix, int size=512, ElemType type=ElemInt) {
    matrix.size = size;
    matrix.elem_type = type;
}
ret = InitMatrix(m4, global_size, type);
```

of course，除了设置矩阵大小和类型，最重要的是为矩阵分配空间和赋值，如果矩阵是 `int` 类型，那么就 `malloc(sizeof(int) * size)`，如果是 `float` 类型，就 `malloc(sizeof(float) * size)`。为了处理不同的类型，我们使用一下模板：

```c
cudaError_t InitMatrix(Matrix &matrix, int size=512, ElemType type=ElemInt) {
    cudaError_t ret = cudaErrorInvalidValue;
    
    matrix.size = size;
    matrix.elem_type = type;

    if (type == ElemInvalid) {
        printf("Please Set Matrix Elem Type\n");
        return ret;
    } else if (ElemInt == type) {  // 调用模板分配函数内存
        ret = AllocMem<int>(matrix, size);
    } else if (ElemFloat == type) {
        ret = AllocMem<float>(matrix, size);
    } else {
        printf("Not Supprot Matrix Elem Type [%d] \n", (int)type);
        return ret;
    }
}
```

在 `AllocMem` 函数中，我们按照字节数对 `matrix.host_addr` 分配内存空间，并完成矩阵元素的初始化：

```c
template<typename T>
cudaError_t AllocMem(Matrix& matrix, int size) {
    cudaError_t ret;
    size_t n_bytes = matrix.size * sizeof(T);
    matrix.host_addr = malloc(n_bytes);          // 开辟内存空间
    srand(666);
    T* ptr = (T*)(matrix.host_addr);             // 获取矩阵首地址
    for (int i = 0; i < matrix.size; i++) {
        ptr[i] = (T)(rand() % 255);              // 初始化范围是 0 到 255
    }
}
```

同时，使用 `cudaMalloc` 创建 `GPU` 内存，使用 `cudaMemcpy(dst, src, byetes_num, KIND)` API 将 CPU 侧的内存拷贝到 GPU 侧，这样就不在 GPU 侧对矩阵进行初始化了。

```c
template<typename T>
cudaError_t AllocMem(Matrix& matrix, int size) {

    ...

    // cudaMemcpyHostToDevice 表示将 Host(CPU) 端拷贝到 GPU 端
    ret = cudaMemcpy(matrix.cuda_addr, matrix.host_addr, n_bytes, cudaMemcpyHostToDevice);
}
```

综上所述，我们可以创建一些矩阵并初始化：

```c
Matrix src_1;   // 输入矩阵
Matrix src_2;   // 输入矩阵
Matrix cpu_dst; // cpu 侧的输出矩阵
Matrix gpu_dst; // gpu 侧的输出矩阵

ret = InitMatrix(src_1, global_size, type);
ret = InitMatrix(src_2, global_size, type);
ret = InitMatrix(cpu_dst, global_size, type);
ret = InitMatrix(gpu_dst, global_size, type);
```

除了内存申请外，还需要在程序退出时设置内存释放的函数，避免内存泄漏：

```c
void FreeMatirx(Matrix& matrix) {
    if (nullptr != matrix.host_addr) {
        free(matrix.host_addr);
    }
    if (nullptr != matrix.cuda_addr) {
        cudaFree(matrix.cuda_addr);   // cuda 释放内存的 api 
    }
}

FreeMatirx(matrix...);
```

在调用之前的矩阵加法的 cuda kernel 函数就大功告成，在学术圈已经可以交差了。但我依然推荐你看一看后面的内存泄漏、日志模块和精度对比，这是工业圈中必不可少的一环。

### 内存泄漏

但是，程序写成这样真的好吗？

在学校里写大作业时，自己组成程序写 `main.cpp` 运行交差就完事了，一般不会有什么问题。但是在工程界，很少有这种服务场景。通常，我们会将代码封装成动态链接库和外部接口，交给框架组统一调用。

假如框架组此时有矩阵加法和高斯滤波两个库。当调用我们提供的矩阵加法库发生异常时，比如内存无法申请，如 `matrix.host_addr` 的结果是空指针等异常，导致我们的库无法得到正确的结果。

假设输入矩阵 `src_1, src_2` 已经创建完毕，在创建输出矩阵 `gpu_dst` 时发生以下异常：

```c
matrix.host_addr = malloc(n_bytes);          // 无法申请内存，返回空指针
T* ptr = (T*)(matrix.host_addr);             // 获取矩阵首地址，为空指针
for (int i = 0; i < matrix.size; i++) {
    ptr[i] = (T)(rand() % 255);              // 空指针赋值，程序会段错误退出
}
```

此时继续执行高斯滤波库，那么 `src_1` 和 `src_2` 这两个输入矩阵的内存依然在堆空间中并没有被释放，**就会导致内存泄漏**。除了使用 C++ 的 RAII 以及智能指针的技术管理这种内存泄漏外，还有一种在教科书中很少提及的 goto 语句。当发生异常时，我们直接跳转到内存释放函数，释放堆空间上的内存，避免泄漏：

```c
template<typename T>
cudaError_t AllocMem(Matrix& matrix, int size) {
    ...
    matrix.host_addr = malloc(n_bytes);
    ret = cudaMalloc((T**)&matrix.cuda_addr, n_bytes);
    if ((nullptr == matrix.host_addr) || (nullptr == matrix.cuda_addr) || (ret != cudaSuccess)) {
        printf(" Alloc memory failed \n");
        ErrorHandleNoLabel(ret);
        return cudaErrorInvalidValue;
    }
    ...
}

int main() {
    ret = InitMatrix(src_1, global_size, type);
    if (ret != cudaSuccess) {
        goto EXIT;
    }
    ret = InitMatrix(src_2, global_size, type);   
    if (ret != cudaSuccess) {
        goto EXIT;
    }
    ret = InitMatrix(cpu_dst, global_size, type); // 发生内存不足的异常
    if (ret != cudaSuccess) {
        goto EXIT;
    }
    ret = InitMatrix(gpu_dst, global_size, type);
    if (ret != cudaSuccess) {
        goto EXIT;
    }

EXIT:
    FreeMatirx(src_1);
    FreeMatirx(src_2);
    FreeMatirx(cpu_dst);
    FreeMatirx(gpu_dst);

    return 0;
}
```

这样，会保证程序的健壮性。在研一上课时，有个工程很强的老师讲到：工业界和学术界的代码完全不一样，需要写很多的异常处理来兜底，此时此刻我信了。

什么？你说每次都写判断、然后 goto 的语法很丑？且无法获取准确的报错信息？

## 报错时的日志模块

在 cuda 运行时可能会发生一些错误，这些错误并不是在编译时期能直接确定的。如果我们对外提供的是 `so` 形式的动态库，出现错误会很难定位。不过我们可以使用 `cudaGetErrorName(cudaError_t status_code)` 这个 `API` 获取错误描述符，使用 `cudaGetErrorName(cudaError_t status_code)` 获取错误的具体描述。可以看下面的例子：

```c
cudaError_t ret = cudaDeviceSynchronize();
printf(" %s\n", cudaGetErrorName(ret)); 
printf(" %s\n", cudaGetErrorString(ret)); 
```

这样，如果阻塞等待 GPU 执行结束时发生了错误，就可以打印出错误的信息，供开发人员调试。那么，我们可不可以让报错代码更友好一些？报错时我想清楚的知道在哪个文件的哪一行出现了问题。

在我之前的文章 [C 语言中的黑魔法：宏](https://muyuuuu.github.io/2024/02/03/define-macro/) 中介绍过，可以使用 `__FILE__` 宏来获取当前文件，使用 `__LINE__` 宏来获取当前行号。所以，我们把错误信息打印封装成一个函数，放到 `tools/common.cuh` 中：

```c
cudaErrot_t ErrorBackTrace(cudaError_t status_code, const char* file, int line_idx) {
    if (status_code != cudaSuccess) {
        printf("CUDA ERROR: \n \t code = %d\n\t name = %s\n\t desc = %s\n\t file = %s\n\t line = %d\n", 
                status_code, cudaGetErrorName(status_code), cudaGetErrorString(status_code), file, line_idx);
    }
    return status_code;
}

ret = InitMatrix(src_1, global_size, type);
ret = ErrorBackTrace(ret, __FILE, __LINE__);
if (ret != cudaSuccess) {
    goto EXIT;
}

ret = InitMatrix(src_2, global_size, type);
ret = ErrorBackTrace(ret, __FILE, __LINE__);
if (ret != cudaSuccess) {
    goto EXIT;
}
```

这样，在运行期间出现错误时，会有很丰富信息提示：

```
CUDA ERROR: 
         code = 101
         name = cudaErrorInvalidDevice
         desc = invalid device ordinal
         file = main.cu
         line = 157
```

但是不是感觉函数的调用不够友好？在每次调用 cuda API 时都需要在后面跟上这三行代码，调用错误检查函数，异常退出释放内存。那么在使用[C 语言中的黑魔法：宏](https://muyuuuu.github.io/2024/02/03/define-macro/)中介绍的 `do while(0)` 宏技巧简化一下代码：

```c
#define ErrorHandleWithLabel(ret, label)                  \
    do {                                                  \
        if(cudaSuccess != ret) {                          \
            ErrorBackTrace(ret, __FILE__, __LINE__);      \
            goto label;                                   \
        }                                                 \
    } while(0)

#define ErrorHandleNoLabel(ret)                           \
    do {                                                  \
        if(cudaSuccess != ret) {                          \
            ErrorBackTrace(ret, __FILE__, __LINE__);      \
        }                                                 \
    } while(0)
```

在 `ErrorHandleWithLabel` 宏中，传入 ret 和 label，发生错误时直接跳转到 label 标签处完成内存的释放，在 `ErrorHandleNoLabel` 宏中，只需要传入 ret，打印报错信息而不需要跳转。这样，代码再次简化：

```c
template<typename T>
cudaError_t AllocMem(Matrix& matrix, int size) {

    ...

    matrix.host_addr = malloc(n_bytes);
    ret = cudaMalloc((T**)&matrix.cuda_addr, n_bytes);
    if ((nullptr == matrix.host_addr) || (nullptr == matrix.cuda_addr) || (ret != cudaSuccess)) {
        printf(" Alloc memory failed \n");
        ErrorHandleNoLabel(ret);        // 报错时无需跳转，返回异常即可，不在这里释放内存
        return cudaErrorInvalidValue;
    }

    ...

}

ret = InitMatrix(m1, global_size, type);
ErrorHandleWithLabel(ret, EXIT);        // 报错时跳转

ret = InitMatrix(m2, global_size, type);
ErrorHandleWithLabel(ret, EXIT);        // 报错时跳转
```

## 精度对比

写到最后别高兴的太早，工业界和上学时最大的差别就是：工业界一定要保证代码的质量，而不是能运行通过出结果，写文档交差拿学分。在程序编译运行通过后，也获得了 GPU 矩阵加法的加速比，但我们一定要保证结果的正确性。

所以我们实现一个 C 版本的矩阵加法程序得到结果，并和 GPU 矩阵加法的结果进行对比。

C 版本矩阵加法，小用一手模板：

```c
template<typename T>
void MatrixCPUAdd(void* m1, void* m2, void* m3, const int size) {

    T* src1 = (T*)(m1);
    T* src2 = (T*)(m2);
    T* dst = (T*)(m3);

    for (int i = 0; i < size; i++) {
        dst[i] = src1[i] + src2[i];
    }
}
```

这里我在想，如果使用面向对象编程，调用入口会不会简单很多......

```c
if (ElemInt == type) {
    MatrixCPUAdd<int>(m1.host_addr, m2.host_addr, m4.host_addr, global_size);
} else if (ElemFloat == type) {
    MatrixCPUAdd<float>(m1.host_addr, m2.host_addr, m4.host_addr, global_size);
}
```

最后我们在提供一个逐字节比对精度的对比函数，来判断两个矩阵的元素是否完全一致：

```c
void MatrixCompare(const Matrix& m1, const Matrix& m2, int n_bytes) {
    if (m1.size != m2.size) {
        printf("Matrix dim not same!\n");
    } else {
        bool flag = true;
        for (int i = 0; i < n_bytes; i++) {
            // 强行使用字节解析地址，得到数值
            unsigned char v1 = *((unsigned char*)(m1.host_addr) + i);
            unsigned char v2 = *((unsigned char*)(m2.host_addr) + i);
            // 数值不相等时退出程序，打印对比失败的信息
            if (v1 != v2) {
                flag = false;
                break;
            }
        }
        if (flag) {
            printf(" Compare Success \n");
        } else {
            printf(" Compare Failed \n");
        }
    }
}

MatrixCompare(cpu_dst, gpu_dst, n_bytes); // 调用接口
```

## 结尾

至此，一个矩阵加法的程序结束了，完成代码参考 `main.cu`。我也在思考一个问题：

- goto 语法那里有没有更好的写法？但是依然需要保证程序异常退出时释放内存，避免内存泄漏
- 如何优化代码？减少类型判断、内存申请 `InitMatrix` 中的繁琐代码。这里我想到的是，将 `Matrix` 也声明为模板类型，避免指针来回来去的转换。