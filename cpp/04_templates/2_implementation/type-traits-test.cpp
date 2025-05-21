#include <gtest/gtest.h>

TEST(TypeTraitsTests, StdIntegralConstant) {
    // std::integral_constant<class T, T val> is a fundamental building block for template metaprogramming, used to wrap compile-time constants into types. It's commonly used as a base class for many type traits in the standard library.
    //https://timsong-cpp.github.io/cppwp/n4950/meta.help
    std::integral_constant<int, 3> c;
    constexpr int b = 3;
    EXPECT_EQ(c, 3);
    EXPECT_EQ(b, c);
}

TEST(TypeTraitsTests, StdIsSame) {
    static_assert(std::is_same<int, int>::value);
    EXPECT_TRUE((std::is_same<int, int>::value));
    EXPECT_FALSE((std::is_same<int, unsigned int>::value));
    EXPECT_FALSE((std::is_same<int, const int>::value));
}

TEST(TypeTraitsTests, StdIsSameIsAClassTemplate) {
    using Type = std::is_same<std::string, std::vector<bool>>;
    Type obj;
    EXPECT_FALSE(obj.value);
    EXPECT_FALSE(obj());
    EXPECT_TRUE((std::is_same<Type, Type>::value));
}

TEST(TypeTraitsTests, StdIsSameInheritsFromStdIntegralConstant) {
    using Type = std::is_same<std::string, std::string>;
    EXPECT_FALSE(
            (std::is_same<std::integral_constant<bool, true>, Type>::value));
    EXPECT_TRUE(
            (std::is_base_of<std::integral_constant<bool, true>, Type>::value));

}