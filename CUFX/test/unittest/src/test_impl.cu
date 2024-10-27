#include "test_case.cuh"
#include "log.cuh"

namespace Test {

void RegisterTestCase(const std::string &base, const std::string &name, const std::function<void()> &func) {
    test_funcs.emplace_back(base, name, func);
}

int RunAllTestCases() {
    int num = 0;
    for (int i = 0; i < test_funcs.size(); i++) {
        TestFunc &t = test_funcs[i];
        LOGI(" >>> Test %s %s \n", t.base.c_str(), t.name.c_str());
        t.func();
        num += 1;
    }
    LOGI(" >>> Total %lu , Passed %d \n", test_funcs.size(), num);
    return 0;
}

int RunSingleTestCase(const std::string &test_case_name) {
    for (int i = 0; i < test_funcs.size(); i++) {
        TestFunc &t = test_funcs[i];
        if (t.name == test_case_name) {
            t.func();
        }
    }
    LOGI(" >>> Single TestCase [%s] Passed.\n", test_case_name.c_str());
    return 0;
}

} // namespace Test