#include "type-traits-impl.h"

#include <gtest/gtest.h>

TEST(TypeTraitsTests, StdIntegralConstant) {
    // std::integral_constant<class T, T val> is a fundamental building block
    // for template metaprogramming, used to wrap compile-time constants into
    // types. It's commonly used as a base class for many type traits in the
    // standard library.
    // https://timsong-cpp.github.io/cppwp/n4950/meta.help
    const std::integral_constant<int, 3> a;
    constexpr int b = 3;
    EXPECT_EQ(b, 3);
    EXPECT_EQ(a, b);
}

TEST(TypeTraitsTests, MyIntegralConstant) {

    constexpr my::integral_constant<int, 3> a;
    constexpr int b = 3;
    EXPECT_EQ(b, 3);
    EXPECT_EQ(a, b);
    constexpr my::integral_constant_2<int, 3> a2;
    EXPECT_FALSE((std::is_same<decltype(a), decltype(a2)>::value));
}

TEST(TypeTraitsTests, StdTrueFalseTypes) {
    // std::true_type and std::false_type are compile-time boolean
    // they are building block in type traits
    // [What is std::false_type or
    // std::true_type?](https://stackoverflow.com/a/58694801/19634193)
    std::integral_constant<bool, true> a;
    std::true_type b;
    // std::true_type is a specialization of std::integral_constant<bool, true>
    constexpr bool c = true;
    constexpr bool d = b;
    EXPECT_EQ(a, b);
    EXPECT_EQ(b, c);
    EXPECT_EQ(c, d);
    EXPECT_TRUE((std::is_same<decltype(a), decltype(b)>::value));
}

TEST(TypeTraitsTests, StdIsSame) {
    static_assert(std::is_same<int, int>::value);
    EXPECT_TRUE((std::is_same<int, int>::value));
    EXPECT_FALSE((std::is_same<int, unsigned int>::value));
    EXPECT_FALSE((std::is_same<int, const int>::value));
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

TEST(TypeTraitsTests, MyIsSame) {
    EXPECT_EQ((std::is_same<int, int>::value), (my::is_same<int, int>::value));
    EXPECT_EQ((std::is_same<int, unsigned int>::value),
              (my::is_same<int, unsigned int>::value));
    EXPECT_EQ((std::is_same<int, const int>::value),
              (my::is_same<int, const int>::value));
}
