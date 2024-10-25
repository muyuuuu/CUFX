#ifndef _TEST_IMPL_H__
#define _TEST_IMPL_H__

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <functional>

namespace Test {

class TestOp {
public:
    TestOp(const char *f, int l) {
        file_name = f;
        line = l;
        ok = true;
    }

    virtual ~TestOp() {
        if (!ok) {
            std::cerr << file_name << " : " << line << " : " << stream.str() << std::endl;
            exit(-1);
        }
    }

    TestOp &IsTrue(bool b, const char *msg) {
        if (!b) {
            stream << " Assert is True Failed " << msg;
            ok = false;
        }
        return *this;
    }

#define BINARY_OP(name, op)                                                                                            \
    template <typename X, typename Y>                                                                                  \
    TestOp &name(const X &x, const Y &y) {                                                                             \
        if (!(x op y)) {                                                                                               \
            stream << " Assert " << x << " " << #op << " " << y << " Failed ";                                         \
            ok = false;                                                                                                \
        }                                                                                                              \
        return *this;                                                                                                  \
    }

    BINARY_OP(IsEqual, ==);
    BINARY_OP(IsNotEqual, !=);
    BINARY_OP(IsGreatEqual, >=);
    BINARY_OP(IsLittleEqual, <=);
    BINARY_OP(IsGreat, >);
    BINARY_OP(IsLittle, <);

#undef BINARY_OP

private:
    std::string file_name;
    int line;
    bool ok;
    std::stringstream stream;
};

} // namespace Test

#define ASSERT_TRUE(c) Test::TestOp(__FILE__, __LINE__).IsTrue((c), #c)
#define ASSERT_EQ(a, b) Test::TestOp(__FILE__, __LINE__).IsEqual((a), (b))
#define ASSERT_NE(a, b) Test::TestOp(__FILE__, __LINE__).IsNotEqual((a), (b))
#define ASSERT_GE(a, b) Test::TestOp(__FILE__, __LINE__).IsGreatEqual((a), (b))
#define ASSERT_GT(a, b) Test::TestOp(__FILE__, __LINE__).IsGreat((a), (b))
#define ASSERT_LE(a, b) Test::TestOp(__FILE__, __LINE__).IsLittleEqual((a), (b))
#define ASSERT_LT(a, b) Test::TestOp(__FILE__, __LINE__).IsLittle((a), (b))

#define TestCase(base, name)                                                                                           \
    class _test_##base##name {                                                                                         \
    public:                                                                                                            \
        void __Run();                                                                                                  \
        static void Run() {                                                                                            \
            _test_##base##name t;                                                                                      \
            t.__Run();                                                                                                 \
        }                                                                                                              \
    };                                                                                                                 \
    class _register_##base##name {                                                                                     \
    public:                                                                                                            \
        _register_##base##name() {                                                                                     \
            Test::RegisterTestCase(#base, #name, &_test_##base##name::Run);                                            \
        }                                                                                                              \
    };                                                                                                                 \
    _register_##base##name auto_register_##base##name;                                                                 \
    void _test_##base##name::__Run()

#endif // _TEST_IMPL_H__