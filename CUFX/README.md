# 如何添加算子

> 常见的函数和测试框架已经搭建。

1. 在 `inc/operator/external.cuh` 中添加算子的声明，如 `gemm`
2. 在 `src/` 目录下创建对应的文件夹 `gemm`。在 `gemm` 文件夹下添加 `inc` 和 `src` 文件夹。其中 `inc` 文件夹是需要的助手函数，对外不可见；`src` 是算子的实现
3. 在 `src/` 目录中的 `CMakeLists.txt` 中注册模块：`add_op_module(gemm)`，编译模块代码
4. 在 `test/testcase` 目录下写算子的 `testcase` 代码： `gemm_testcase.cu`，测试代码规范：

```cpp
TestCase(CudaOp, Gemm) {   // Gemm 是 TestCase 的名字
    op run ...
    ASSERT_TRUE(ret);
}
```

5. 在 `scripts/run.sh` 中，指定自己的 `TestCase` 名，也就是 `Gemm`，检查测试是否通过

如果有不懂的地方，可以参考 `reduct_sum` 的写法。

## 代码规范

1. 变量名小写，用下划线连接
2. 类和函数首字母大写
3. 宏定义全部大写
4. 不能超过 120 列
5. 其余用给出的 clang-format 自动格式化
6. 可以写中文注释

# 未完工

- [ ] 零拷贝内存
- [ ] 多流时异步操作，需要思考下如何实现
- [ ] matrix 实现的不好，迟早优化掉
- [ ] C++ 新特性应用，类型转换、智能指针、类型擦除、函数式编程用起来

# 编译运行

```
bash scripts/build.sh
bash scripts/run.sh
```

# 架构组织

```
├── CMakeLists.txt
├── README.md
├── inc                                    # 对外头文件，用户可见
│   ├── CMakeLists.txt
│   ├── data_type
│   │   └── data_type.cuh
│   ├── log
│   │   └── log.cuh
│   ├── matrix
│   │   └── matrix.cuh
│   └── operator
│       └── external.cuh
├── priv                                   # 私有文件，用户不可见
│   ├── CMakeLists.txt
│   ├── runtime
│   │   ├── inc
│   │   │   └── runtime_info.cuh
│   │   └── src
│   │       └── runtime_info.cu
│   └── time
│       └── inc
│           └── clock.cuh
├── scripts                                # 编译运行的脚本
│   ├── build.sh
│   └── run.sh
├── src                                    # 算子实现
│   ├── CMakeLists.txt
│   ├── log
│   │   └── log.cu
│   ├── matrix
│   │   └── matrix.cu
│   ├── reductsum
│   │   ├── inc
│   │   └── src
│   │       └── reduct_sum.cu
│   └── transpose
└── test                                   # 测试代码
    ├── CMakeLists.txt
    ├── main.cu
    ├── testcase
    │   └── reduct_sum_testcase.cu
    └── unittest
        ├── inc
        │   ├── compare.cuh
        │   ├── test_case.h
        │   └── test_impl.h
        └── src
            └── test_impl.cpp
```