#ifndef _TEST_CASE_H__
#define _TEST_CASE_H__

#include "test_impl.h"

namespace Test {

struct TestFunc {
    std::string base;
    std::string name;
    std::function<void()> func;

    TestFunc(const std::string &_base, const std::string &_name, const std::function<void()> &_func) {
        base = _base;
        name = _name;
        func = _func;
    }
};

static std::vector<TestFunc> test_funcs;

void RegisterTestCase(const std::string &base, const std::string &name, const std::function<void()> &_func);

int RunAllTestCases();

} // namespace Test

#endif // _TEST_CASE_H__