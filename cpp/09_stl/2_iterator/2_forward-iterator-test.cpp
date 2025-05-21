#include "2_forward-iterator-impl.h"

#include <gtest/gtest.h>

#include <iterator>
#include <print>

template<typename T>
using MyContainer = MyContainerWithForwardIterator<T>;

TEST(MyForwardIterator, BasicUsageShouldWork) {
    auto mc = MyContainer<int>({3, 1, 4, 1, 5, 9, 2, 6});
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    // `its != words.end()`, not `itr < words.end()`
    for (auto itr = mc.begin(); itr != mc.end(); ++itr) {
        std::cout << *itr << " ";
    }
    std::cout << std ::endl;
    const std::string stdout_str = buffer.str();
    EXPECT_EQ(stdout_str, "3 1 4 1 5 9 2 6 \n");
    std::cout.rdbuf(old);
    // static_assert(std::forward_iterator<decltype(mc)::iterator>);
}

TEST(MyForwardIterator, RangeBasedLoopShouldWork) {
    auto mc = MyContainer<std::string>(
            {"Lorem", "ipsum", "dolor", "sit", "amet"});
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    for (const auto &itr: mc) {
        std::cout << itr << " ";
    }
    std::cout << std ::endl;
    const std::string stdout_str = buffer.str();
    EXPECT_EQ(stdout_str, "Lorem ipsum dolor sit amet \n");
    std::cout.rdbuf(old);
}

TEST(MyForwardIterator, BasicUsageShouldWork1) {
    auto mc = MyContainer<int>({3, 1, 4, 1, 5, 9, 2, 6});
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    // `its != words.end()`, not `itr < words.end()`
    for (auto itr = mc.begin(); itr != mc.end(); ++itr) {
        std::cout << *itr << " ";
    }
    std::cout << std ::endl;
    const std::string stdout_str = buffer.str();
    EXPECT_EQ(stdout_str, "3 1 4 1 5 9 2 6 \n");
    std::cout.rdbuf(old);
}
