# nvprof 性能分析

## 简介

在过去，写完代码的时候只知道加速比，比如比 C 代码快了 10 倍或者 20 倍就觉得可以了。这种加速的方式并不科学，想具体知道算子的性能瓶颈在哪里，加速的效率和性能如何，或者算子调优，就需要 profiling 工具的帮助。

## 用法

需要注意的是，在计算能力大于 `8.0` 的设备上，一些早期教程中的 `nvprof` 命令已经不在支持使用，会有以下警告：

```
======== Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
                  Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
                  Refer https://developer.nvidia.com/tools-overview for more details.
```

此时可以使用 `nsys` 命令