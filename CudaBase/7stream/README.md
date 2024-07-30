# 流

> CUDA 流表示一个 GPU 操作队列，队列中的操作将以添加到流中的先后顺序而依次执行。流能够封装一系列异步操作，并保持这些操作在流中排队，使得在前面所有操作启动之后再启动后续的操作。

> 设备操作包括：数据传输和执行 kernel 函数。在 cuda 中，所有的设备操作都在 stream 中执行。

在前面的章节中，代码流程通常是：

- 主机与设备间的数据传输
- 核函数启动
- 等待 GPU 执行结束，将数据从设备传输到主机

所有的 cuda 操作都是在流中进行的，比如内核启动和核函数计时。因为我们没有主动的去创建流，所以也可以猜到流分为显式和隐式的声明和调用：

- 隐式声明的流，我们叫做空流
- 显式声明的流，我们叫做非空流

所以之前的代码都是在空流中完成的，当我们想控制流时，就需要创建非空流。

额外的，在 `4runtime` 那一部分中的计时器，就是将两个时间插入了流，并计算两个时间的时间，得到流中核函数的执行时间。

## 处理异步操作与多流工作

以数据拷贝为例：`cudaMemcpy()` 是一个同步操作，而 `cudaMemcpyAsync()` 是一个异步操作，这个 API 就需要传入一个流参数。流的创建和调用方式如下：

```c
cudaStream_t a;
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
...
kernel_name<<<grid, block, sharedMemSize, stream>>>(argument list);
```

异步的操作很容易去隐藏时间，比如异步启动一个函数放到后台执行，然后去执行别的函数。最后在需要同步的地方等待异步函数执行完毕即可。

```c
for (int i = 0; i < num_tasks; i++) { cudaStreamCreate(&(stream_arr[i])); }     // 创建流

for (int i = 0; i < num_tasks; i++) {
    int start = i * step;
    int end = start + step;
    printf(" Task %d, start = %d, end = %d \n", i + 1, start, end);
    cudaMemcpyAsync(src.cuda_addr, src.host_addr, step * sizeof(int), cudaMemcpyHostToDevice, stream_arr[i]);
    MatrixAddScalar<int, step * sizeof(int), 13>                                // 调用核函数
        <<<step, local_size, 0, stream_arr[i]>>>(src.cuda_addr + start * sizeof(int));
    cudaMemcpyAsync(src.host_addr, src.cuda_addr, step * sizeof(int), cudaMemcpyDeviceToHost, stream_arr[i]);
}

for (int i = 0; i < num_tasks; i++) { cudaStreamSynchronize(stream_arr[i]); }   // 同步流
```

```c
cudaMemcpyAsync()  
MatrixAddScalar()
cudaMemcpyAsync() 
```

以上三个操作都不会阻塞主机，但是在同一个阻塞流中，这三个操作会顺序执行。

这里只是举例说明多流如何工作，上述代码的性能并不是很好：

- 如果有 A B C D 四个不相关的任务，每个任务使用的数据也不相关，可以使用 4 个流进行异步并发处理。即当一次处理多个数据时，使用异步传输数据 `cudaMemcpyAsync` 来提升效率。
- 对于类似矩阵加法这种任务，任务相关且数据相关，4 个流工作的速度会远远略微低于 1 个流


## 流同步

- 空流:
  - 隐式声明，同步所有阻塞流中的操作
- 非空流：
  - 人为定义，在 GPU 上并发执行多个操作
  - 可以分为阻塞流和非阻塞流
    - `cudaStreamCreate` 默认为阻塞流，这就意味着流中操作可被阻塞，直到空流中的操作完成，直到空流的执行结束。同一个流内的 kernel 函数会相互阻塞，不同流的 kernel 函数不会阻塞。
    - 非阻塞流通过 `cudaStreamCreateWithFlags` 创建，并设置 `cudaStreamNonBlocking` 标志。不会因为空流的操作而阻塞。

需要注意的是：在 CUDA 中，启动多个阻塞流，且没有空流的情况下，这些阻塞流之间通常不会相互阻塞。阻塞流中的操作可能会受到空流（即默认流）中操作的影响，但在没有空流参与的情况下，阻塞流内的操作主要受到GPU执行资源和依赖关系的影响。

### 阻塞流和非阻塞流

```c
kernel_1<<<1, 1, 0, stream_1>>>();
kernel_2<<<1, 1>>>();
kernel_3<<<1, 1, 0, stream_2>>>();
```

stream_1 和 stream_2 是阻塞流。具体执行过程是，kernel_1 被启动，控制权返回主机，然后启动 kernel_2，但是此时 kernel_2 不会并不会马上执行，他会等到 kernel_1 执行完毕。

同理启动完 kernel_2 控制权立刻返回给主机，主机继续启动 kernel_3，这时候 kernel_3 也要等待，直到 kernel_2执行完，但是从主机的角度，这三个核都是异步的，启动后控制权马上还给主机。

### 显式同步

与显式同步对应的是隐式同步，如 `cudaMemcpy` 等 API。显式同步的有：

```c
cudaDeviceSynchronize();  // 阻塞，直到设备完成所有操作
cudaStreamSynchronize();  // 流同步，阻塞直到结束
cudaStreamQuery();        // 非阻塞，查询流是否结束
cudaEventSynchronize();   // 阻塞，直到事件完成
cudaEventQuery();         // 非阻塞，查询事件是否完成
cudaStreamWaitEvent();    // 指定的流要等待指定的事件，事件完成后流才能继续
```

不过在调用同步以及等待同步完成的时候，可以执行一些其他任务，做到 CPU 和 GPU 的并行：

```c
cudaEventRecord(stop);
// CPU Tasks ...
ret = cudaEventSynchronize(stop);
```

### 事件同步

假如一个流中有 A B C D 一共 4 个核函数，A B C 核函数是做数据预处理，D 是真正的计算函数。A B C 没有任务依赖关系，而 D 依赖于 A B C。此时可以使用事件同步，A B C 全部异步执行，并且在 C 之后插入一个事件，任务 D 等到这个事件执行后，才能执行：

```c
for (int i = 0; i < num_tasks; i++) {
    kernel_1<<<1, 1, 0, stream_arr[i]>>>();
    kernel_2<<<1, 1, 0, stream_arr[i]>>>();
    kernel_3<<<1, 1, 0, stream_arr[i]>>>();

    cudaEventRecord(wait_tokens[i]);
    cudaStreamWaitEvent(stream_arr[i], wait_tokens[i], 0);

    kernel_4<<<1, 1, 0, stream_arr[i]>>>();
}
```

## 流回调

使用固定 API 完成一个操作：当一个流执行到某一步的时候，调用一个回调函数。比如上面的例子，执行完事件后，回调一个函数，打印数据全部准备好了的信息。

```c
// 回调函数原型，必须符合 cudaStreamCallback_t 的定义
void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *userData) {
    int *p = (int *)userData;
    if (status == cudaSuccess) {
        printf("Callback from stream %d, Wait Event Success \n", *p);
    } else {
        printf("Error in stream callback: %s\n", cudaGetErrorString(status));
    }
}

for (int i = 0; i < num_tasks; i++) {
    stream_idx[i] = i;
    kernel_1<<<1, 1, 0, stream_arr[i]>>>();
    kernel_2<<<1, 1, 0, stream_arr[i]>>>();
    kernel_3<<<1, 1, 0, stream_arr[i]>>>();

    cudaEventRecord(wait_tokens[i]);
    cudaStreamWaitEvent(stream_arr[i], wait_tokens[i], 0);

    kernel_4<<<1, 1, 0, stream_arr[i]>>>();
    cudaStreamAddCallback(stream_arr[i], my_callback, stream_idx + i, 0);  // 回调函数
}
```