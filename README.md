# CUDA 计划

> 如果你有幸学过 `OpenCL` 的话，会发现 `cuda` 异常简单，甚至能用个周六日学完。

想着在下班时间学点什么，刷视频打游戏总不是办法，想来想去决定从 `cuda` 开始了。本系列计划由两部分组成：

- 第一部分是 `cuda` 编程基础
- 第二部分是 `cuda` 优化模型

不过搞完第一部分后我应该会去学 `C++`，把一切基础知识准备就绪后开始学习 `cuda` 优化模型。

## 第一部分

在 `CudaBase` 文件夹下，每个文件夹下配有文档和代码示例。如果没有特别声明，所有编译和运行的方式均为：

```bash
nvcc main.cu
./a.out
```

一开始避免接触什么 SM 流多处理器，较多的知识概念只会让人心生劝退。从最简单的 helloword 开始，先把代码跑起来，在一点点学习那些线程块、内存模型。

1. `1helloworld`，安装与运行 hello world 级别的代码
2. `2`