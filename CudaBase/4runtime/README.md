# 运行时信息查询

在内容开始之前，按照上一节的结尾所述：将 `Matrix` 也声明为模板类型，此时矩阵类型为：

```c
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
```

避免指针来回来去的转换，这样会精简一部分代码。有的时候需要查询 GPU 的一些基本信息，如内存信息、计算核心的数量、检测 GPU 的数量，以及存在多张 GPU 时，设置在哪一张 GPU 运行。我们把这些信息查询封装到 `SetGPU()` 函数中，放到 `tools/common.cuh` 文件中。

- `cudaGetDeviceCount` 查询 GPU 的数量
- `cudaSetDevice` 设置程序在哪个 GPU 运行

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, i_device);  // 查询更多的属性，更多请参考官方文档
```

GPU 计算时依赖 GPU 计算核心，计算核心的数量影响计算性能。可以根据 cuda 的版本号得到计算核心数量，封装到 `tools/common.cuh` 文件的 `GetCores` 函数中。

# Cuda 计时

CUDA 所有的操作都是在流中完成的，核函数在 cuda 的流中运行，那么在核函数前后的流中插入两个事件，记录事件的时间点，就得到的核函数的运行时间。

使用事件计时，一个 CUDA 事件是 CUDA 流中的一个标记点，它可以用来检查正在执行的流操作是否已经到达了该点。CUDA 提供了在流中的任意点插入并查询事件完成情况的函数，只有当流中先前的所有操作都执行结束后，记录在该流中的事件才会起作用。

声明和创建一个事件的方式如下：

```c
cudaEvent_t event;
cudaError_t cudaEventCreate(cudaEvent_t* event);
```

一个事件可以使用如下函数进入 CUDA 流的操作队列中：

```c
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
```

下面的函数会在 host(CPU 侧) 中阻塞式地等待一个事件完成

```c
cudaError_t cudaEventSynchronize(cudaEvent_t event);
```

如果想知道两个事件之间的操作所耗费的时间，可以调用：

```c
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop);
```

最后去销毁事件：

```c
cudaError_t cudaEventDestroy(cudaEvent_t event);
```

所以，cuda 端计时程序为：

```c
const int EPOCHS = 5;                // 测 5 次
for (int i = 0; i < EPOCHS; i++) {
    cudaEvent_t start, end;
    ret = cudaEventCreate(&start);
    ret = cudaEventCreate(&end);
    ret = cudaEventRecord(start, 0);   // 插入开始事件
    ErrorHandleWithLabel(ret, EXIT);

    // 调用 kernel 函数
    MatrixGPUAdd<int><<<grid_size_1d, block_size_1d>>>(m1.cuda_addr, m2.cuda_addr, m3.cuda_addr, global_size);

    ret = cudaEventRecord(end, 0);     // 插入结束事件
    ret = cudaGetLastError();
    ret = cudaEventSynchronize(end);   // 阻塞等待结束事件，所以不用在调用 cudaDeviceSynchronize

    ret = cudaEventElapsedTime(&gpu_time, start, end);   // 记录 kernel 执行的事件
    printf(" In Epoch %d, GPU Cost %.4f ms \n", i + 1, gpu_time);
    ErrorHandleWithLabel(ret, EXIT);

    ret = cudaEventDestroy(start);
    ret = cudaEventDestroy(end);
    ErrorHandleWithLabel(ret, EXIT);
}
```

## CPU 计时

CPU 计时就很简单可，因为 C 语言的 clock 那个函数存在计时的 bug，因此采用 `gettimeofday` 来记录事件，同样循环执行 5 次。

```c
for (int i = 0; i < EPOCHS; i++) {
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    MatrixCPUAdd<int>(m1.host_addr, m2.host_addr, m4.host_addr, global_size);
    gettimeofday(&stop, NULL);

    long time = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;

    printf(" In Epoch %d, CPU Cost %.4f ms \n", i + 1, 1.0 * time / 1000.0);
}
```

可以看到 GPU 远远快于 CPU：

```
 cudaGetDeviceCount   get [1] device !
 cudaSetDevice set device [0] to run !
 device name             NVIDIA GeForce RTX 4070
 device global mem       12281 MB
 device const  mem       64 KB
 device sms              46 
 Cores                   5888 
 Max threads per block:  1024
Init Matrix Success
Init Matrix Success
Init Matrix Success
Init Matrix Success
 In Epoch 1, GPU Cost 0.2451 ms 
 In Epoch 2, GPU Cost 0.1740 ms 
 In Epoch 3, GPU Cost 0.1965 ms 
 In Epoch 4, GPU Cost 0.1864 ms 
 In Epoch 5, GPU Cost 0.1946 ms 
 In Epoch 1, CPU Cost 4.6920 ms 
 In Epoch 2, CPU Cost 4.6380 ms 
 In Epoch 3, CPU Cost 4.7680 ms 
 In Epoch 4, CPU Cost 4.9540 ms 
 In Epoch 5, CPU Cost 5.0270 ms 
 Compare Success 
```