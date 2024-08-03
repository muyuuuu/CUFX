
# 结构

```
bash build_run.sh  # 编译执行
```

```
├── CMakeLists.txt                                      # 根目录 CMakeLists.txt
├── README.md
├── build_run.sh                                        # 编译运行脚本
├── inc                                    # 外部接口，外部可见
│   ├── data_type                                       # 数据类型
│   │   └── data_type.cuh
│   ├── log                                             # 日志声明
│   │   └── log.cuh                                     
│   ├── matrix                                          # 矩阵声明
│   │   └── matrix.cuh 
│   └── operator                                        # cuda 算子
│       └── external.cuh
├── priv                                   # 内部接口，不对外提供
│   ├── runtime                                         # 运行时信息
│   │   ├── inc
│   │   │   └── runtime_info.cuh
│   │   └── src
│   │       └── runtime_info.cu
│   └── time                                            # 计时函数
│       └── inc
│           └── clock.cuh
├── src                                     # 实现代码目录
│   ├── CMakeLists.txt
│   ├── log                                             # 日志函数实现
│   │   └── log.cu
│   ├── matrix                                          # 矩阵函数实现
│   │   └── matrix.cu
│   ├── reductsum                                       # 归约求和算子
│   │   ├── inc
│   │   └── src
│   │       └── reduct_sum.cu
│   └── transpose
└── test                                    # 测试目录
    ├── CMakeLists.txt
    ├── inc                                             # 测试模块代码
    │   ├── compare.cuh
    │   └── testcase.cuh
    ├── main.cu                                         # 程序入口，执行测试样例
    └── testcase                                        # 测试用例
        └── reduct_sum_testcase.cu
```