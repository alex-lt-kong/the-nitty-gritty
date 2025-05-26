#include "4_my-allocator-impl.h"

#include <gtest/gtest.h>

TEST(TestMyAllocator, Test1) {
    std::vector<std::string> vec1 = {"0xdeadbeaf"};
    std::vector<std::string, My::Allocator<std::string>> vec2 = {"0xdeadbeaf"};
    EXPECT_EQ(vec1[0], vec1[0]);
}

TEST(TestMyAllocator, Test2) {
    std::vector<std::string> vec1;
    std::vector<std::string, My::Allocator<std::string>> vec2;
    constexpr size_t sz = 65536;
    vec1.reserve(sz);
    vec2.reserve(sz);
    for (int i = 0; i < sz; ++i) {
        vec1.push_back(std::to_string(sz - i));
        vec2.push_back(std::to_string(sz - i));
    }
    for (int i = sz - 1; i >= 0; --i) {
        EXPECT_EQ(vec1[i], vec1[i]);
    }
}
