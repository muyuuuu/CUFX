#include "test_case.h"

namespace Test {

void RegisterTestCase(const std::string &base, const std::string &name, const std::function<void()> &func) {
    test_funcs.emplace_back(base, name, func);
}

int RunAllTestCases() {
    int num = 0;
    for (int i = 0; i < test_funcs.size(); i++) {
        TestFunc &t = test_funcs[i];
        std::cout << " >>> Test " << t.base << " " << t.name << std::endl;
        t.func();
        num += 1;
    }
    std::cout << " === Total : " << test_funcs.size() << ", Passed : " << num << std::endl;
    return 0;
}

} // namespace Test