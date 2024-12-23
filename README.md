# CUDA 计划

> 如果你有幸学过 `OpenCL` 的话，会发现 `cuda` 异常简单，甚至能用个周六日学完。

想着在下班时间学点什么，刷视频打游戏总不是办法。有时候下班都 22 点甚至更晚，但还是要顶住，哪怕学 10 分钟呢？想来想去决定从 `cuda` 开始了，本系列计划由两部分组成：

- 第一部分是 `cuda` 编程基础
- 第二部分是 `cuda` 优化模型

不过搞完第一部分后我应该会去学 `C++`，把一切基础知识准备就绪后开始学习 `cuda` 优化模型。时隔两个月，我学完 `C++` 回来了。

> [一份不错的 C++ 进阶文档](https://github.com/parallel101/cppguidebook)

**下班后有点头晕眼花，但还是在艰难施工中**。 ~~也许哪天学到高大上的写法，比如一些好的日志实现，好的架构组织，我就回来填坑。~~ 如果你有想法，哪怕是练手，想练习一个内存池，也欢迎。

## `CudaBase`

在 `CudaBase` 文件夹下，每个文件夹下配有文档和代码示例。下班时间很少，努力保证代码质量，但文档写的很急 ...... 如果没有特别声明，所有编译和运行的方式均为：

```bash
nvcc main.cu
./a.out
```

后续会过度到 `cmake`，`cmake` 的编译方式为：

```bash
cd build
cmake ..
make install
./bin/main
```

| 文件           | 备注                                                                              |
| -------------- | --------------------------------------------------------------------------------- |
| `1helloworld`  | 安装与运行 hello world 级别的代码                                                 |
| `2threadid`    | 线程模型中的索引计算                                                              |
| `3matrixadd`   | 矩阵加法。小用一手模板，宏的黑魔法，以及避免内存泄漏                              |
| `4runtime`     | 运行时。继续使用模板精简代码，运行时信息查询，函数计时等                          |
| `5memorymodel` | 内存模型。全局内存、局部内存、共享内存、统一内存等，并全部使用 `cmake` 编译和管理 |
| `6arch`        | 计算架构相关的东西，流多处理器、延时隐藏、线程束分化等问题                        |
| `7stream`      | 多流计算，流同步、回调等                                                          |

## `CUFX`

`cuda` 计算框架 `CUFX`，取自单词 `Cuda Framework eXtended` 用来装逼。包含一些算子的实际项目，严格组织代码结构。大概计划是：


- [ ] 线程池
- [ ] 内存池
- [x] 基础日志、数据类型
- [x] 搭建工程结构，编写 cmakelists.txt
- [x] 一些优雅的 C++ 实现
- [x] 一些高性能 cuda 算子
- [x] 可对外提供动态库，直接使用
- [x] 自动化测试框架

| 文件        | 备注                                                         |
| ----------- | ------------------------------------------------------------ |
| `reductsum` | `reduce` 优化，解决 `bank conflict`，交叉寻址，`warp` 展开等 |
| `gemm`      | `gemm` 实践，共享内存、`2d thread tile`、双缓冲技术          |
| `conv`      | 卷积实践，`img2col` 与隐 `gemm` 算法                         |

# 参考

1. [基础](https://github.com/sangyc10/CUDA-code)
2. [内存](https://www.cnblogs.com/moonzzz/p/17621574.html)
3. [多流](https://lulaoshi.info/gpu/python-cuda/streams.html)
4. [进阶，不适用初学者](https://github.com/PaddleJitLab/CUDATutorial/tree/develop)
