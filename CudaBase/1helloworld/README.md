# 安装与 hello world

## 安装

我曾经在我的笔记本上装过无数次的 linux 操作系如，这一次实在不想折腾了，没有选择双系统的安装方式。

网上有两种常见的安装方式：

1. 在 windows 安装，并配置 visual studio 宇宙第一编辑器进行开发。我拒绝了这种安装方式，纯粹是因为 visual studio 这玩意太大了，还得考虑版本适配、插件、编译环境等一系列设置。

2. 在 windows 安装 cuda toolkit。并且安装 wsl2，注意一定是 wsl2，wsl1 可能无法检测到显卡的驱动环境。这样在 wsl2 中执行 nvidia-smi 可以看到显卡信息时，表明安装成功：

```
ljw@WIN-28T22C1R5J4:~/cuda$ nvidia-smi
Thu Jul 18 22:40:32 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 556.12       CUDA Version: 12.5     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4070        On  | 00000000:01:00.0  On |                  N/A |
|  0%   39C    P8              13W / 200W |    604MiB / 12282MiB |     17%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## hello world

首先，cuda 文件是以 `.cu` 作为后缀，所以我们创建 `main.cu` 文件。而后像写 C++ 那样去写代码就可以了，首先包含头文件：

```cpp
#include <stdio.h>
```

而后简单了解下 `cuda` 的工作形式：

1. 在 CPU 侧编写 GPU 端的代码。这个代码一般叫做 kernel 程序，以 `__global__` 开头，返回值类型是 `void`
2. CPU 将 kernel 代码提交到 GPU，CPU 侧以阻塞的形式等待 GPU 执行完毕
3. 继续执行 CPU 侧后面的程序直到结束

写一个简单的 kernel 函数：

```cpp
__global__ void hello_from_gpu()
{
    printf("Hello World from the the GPU\n");
}
```

并在 CPU 侧完成调用和阻塞等待：

```cpp
int main(void)
{
    hello_from_gpu<<<1, 1>>>();  // 创建 1 个 grid 和 1 个 thread 去执行，后面会讲
    cudaDeviceSynchronize();  // 阻塞等待 GPU 端执行完毕
    return 0;
}
```

这样，就完成了最简单的 cuda 程序。

~~实不相瞒，安装这部分，我花了 5 个小时才搞定......~~