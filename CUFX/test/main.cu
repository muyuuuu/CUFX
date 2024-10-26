#include "test_case.h"
#include "log.cuh"

int main(int argn, char **argv) {
    if (2 != argn) {
        LOGE("args must be 2, not be %d\n", argn);
        return 0;
    } else {
        Test::RunSingleTestCase(argv[1]);
    }

    return 0;
}