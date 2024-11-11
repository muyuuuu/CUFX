# 算子优化记录

## Reduct

### 朴素实现

```c
for (int s = 1; s < bdim; s *= 2)
{
    if (tid % (2 * s) == 0 && i + s < len)
    {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

这个是最简单的版本，但是会导致线程束分化：对于同一个 `warp` 内的线程，同一时刻必须执行同一指令。所以在同一 `warp` 中，对于 0 到 31 号线程，只有一半的线程满足 `if` 条件，其余线程需要等待，所以会发生线程束分化。线程分配方式如下图所示：

![](./imgs/reduce1.png)

### 避免线程束分化

我们可以修改线程的分配方式，令同一个 `warp` 中的线程执行相同的指令：

![](./imgs/reduce2.png)

此时代码如下：

```c
for (int s = 1; s < bdim; s *= 2)
{
    int index = 2 * s * tid;
    if ((index + s < bdim) && (bdim * bid + s < len))
    {
        sdata[index] += sdata[index + s];
    }
}
```

### 避免 bank conflict (0.5ms)

对于 `cuda` 的动态共享内存，会被划分为 32 个可被同时访问的 bank。共享内存中连续的 128 字节的内容会被分摊到 32 个 bank 中的同一层中。

![](./imgs/bank.png)

- 如果同一个 warp 的多个线程同时访问同一个 bank，那么它们的访问就会被串行化，导致 bank conflict，导致性能降低
- 如果一个线程访问一个 bank 的不同地址，不会被串行化
- 如果多个线程访问同一个 bank 的同一个地址，不会被串行化，会发生广播

所以优化代码的思路：

```c
for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s){
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

避免同一个 warp 内的不同线程访问同一个 bank 的不同地址。

### 减少空闲线程 (0.3ms)

在加载内存的时候，完成一次加法，这样就可以省掉一半的 grid。

### 循环展开 (0.2ms)

由于 `cuda` 是单指令多线程的架构。所以在步长小于 32 的时候，计算会发生在同一个 warp 内，不在需要 `__syncthreads` 同步。

### 参考

晚上下班时间少写的有点精简，具体的优化思路可以参考这里：https://github.com/PaddleJitLab/CUDATutorial/tree/develop/docs/09_optimize_reduce