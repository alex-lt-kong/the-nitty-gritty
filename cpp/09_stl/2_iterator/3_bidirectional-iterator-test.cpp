#include "3_bidirectional-iterator-impl.h"

#include <gtest/gtest.h>

#include <iterator>
#include <print>
#include <ranges>

template<typename T>
using MyContainer = MyContainerWithBidirectionalIterator<T>;

TEST(MyBidirectionalIterator, BasicUsageShouldWork) {
    auto mc = MyContainer<int>({3, 1, 4, 1, 5, 9, 2, 6});
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    // `its != words.end()`, not `itr < words.end()`
    for (auto itr = mc.begin(); itr != mc.end(); ++itr) {
        std::cout << *itr << " ";
    }
    std::cout << std ::endl;
    const std::string stdout = buffer.str();
    EXPECT_EQ(stdout, "3 1 4 1 5 9 2 6 \n");
    std::cout.rdbuf(old);
    static_assert(std::bidirectional_iterator<decltype(mc)::iterator>);
}

TEST(MyBidirectionalIterator,
     IterateOverTheOtherDirectionIsAwkwardButShouldWork) {
    std::vector vec = {3, 1, 4, 1, 5, 9, 2, 6};
    auto mc = MyContainer(vec);
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    auto itg = mc.end();
    auto itv = vec.end();
    while (true) {
        --itg;
        --itv;
        std::cout << *(itg) << " ";
        if (itg == mc.begin()) {
            break;
        }
        EXPECT_EQ(*itg, *itv);
    }
    std::cout << std ::endl;
    const std::string stdout = buffer.str();
    EXPECT_EQ(stdout, "6 2 9 5 1 4 1 3 \n");
    std::cout.rdbuf(old);
    // static_assert(std::bidirectional_iterator<decltype(gc)::iterator>);
}

TEST(MyBidirectionalIterator, RangeBasedLoopShouldWork) {
    auto mc = MyContainer<std::string>(
            {"Lorem", "ipsum", "dolor", "sit", "amet"});
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    for (const auto &itr: mc) {
        std::cout << itr << " ";
    }
    std::cout << std ::endl;
    const std::string stdout = buffer.str();
    EXPECT_EQ(stdout, "Lorem ipsum dolor sit amet \n");
    std::cout.rdbuf(old);
}

TEST(MyBidirectionalIterator, RangeBasedLoopShouldWorkInReverse) {
    auto mc = MyContainer<std::string>(
            {"Lorem", "ipsum", "dolor", "sit", "amet"});
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    for (const auto &itr: std::views::reverse(mc)) {
        std::cout << itr << " ";
    }
    std::cout << std ::endl;
    const std::string stdout = buffer.str();
    EXPECT_EQ(stdout, "amet sit dolor ipsum Lorem \n");
    std::cout.rdbuf(old);
}
