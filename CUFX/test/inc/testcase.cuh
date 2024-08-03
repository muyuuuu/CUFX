#ifndef __TESTCASE_CUH__
#define __TESTCASE_CUH__

#include "external.cuh"

#define __TO_STR__(x) #x ":"
#define __TO_REAL__(x) __TO_STR__(x)
// 文件:行号
#define __FILE_LINE__ __FILE__ ":" __TO_REAL__(__LINE__)

#define UNIT_TEST(exper, res)                                                                                          \
    do {                                                                                                               \
        printf("UNIT_TEST" __FILE_LINE__ ":calling " #exper "\n");                                                     \
        if (0 == (exper)) {                                                                                            \
            printf("UNIT_TEST" __FILE_LINE__ ":error \n");                                                             \
            res = 0;                                                                                                   \
        } else {                                                                                                       \
            res = -1;                                                                                                  \
            printf("UNIT_TEST" __FILE_LINE__ ":passed \n");                                                            \
        }                                                                                                              \
    } while (0)

int TestReductSum();

#endif