#include "4_random-access-iterator-impl.h"

#include <gtest/gtest.h>

#include <iterator>
#include <print>
#include <ranges>

template<typename T>
using MyContainer = MyContainerWithRandomAccessIterator<T>;

TEST(MyRandomAccessIteratorTests, BasicUsageShouldWork) {
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
    static_assert(std::random_access_iterator<decltype(mc)::iterator>);
}

TEST(MyRandomAccessIteratorTests,
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
    const std::string stdout_str = buffer.str();
    EXPECT_EQ(stdout_str, "6 2 9 5 1 4 1 3 \n");
    std::cout.rdbuf(old);
    // static_assert(std::bidirectional_iterator<decltype(gc)::iterator>);
}

TEST(MyRandomAccessIteratorTests, RangeBasedLoopShouldWork) {
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

TEST(MyRandomAccessIteratorTests, RangeBasedLoopShouldWorkInReverse) {
    auto mc = MyContainer<std::string>(
            {"Lorem", "ipsum", "dolor", "sit", "amet"});
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    for (const auto &itr: std::views::reverse(mc)) {
        std::cout << itr << " ";
    }
    std::cout << std ::endl;
    const std::string stdout_str = buffer.str();
    EXPECT_EQ(stdout_str, "amet sit dolor ipsum Lorem \n");
    std::cout.rdbuf(old);
}


TEST(MyRandomAccessIteratorTests, RandomAccessIteratorShouldWork) {
    std::vector words = {"The",  "quick", "brown", "fox", "jumps",
                         "over", "the",   "lazy",  "dog"};
    auto mc = MyContainer(words);

    auto itw = words.begin();
    auto itm = mc.begin();
    EXPECT_EQ(itw[1], "quick");
    EXPECT_EQ(itm[1], "quick");
    EXPECT_EQ(*(itw + 3), "fox");
    EXPECT_EQ(*(itm + 3), "fox");
    itw += 4;
    itm += 4;
    EXPECT_EQ(itw[0], "jumps");
    EXPECT_EQ(itm[0], "jumps");
    itw -= 2;
    itm -= 2;
    EXPECT_EQ(*itw, "brown");
    EXPECT_EQ(*itm, "brown");
    EXPECT_EQ(*(words.end() - 2), "lazy");
    EXPECT_EQ(*(mc.end() - 2), "lazy");
    EXPECT_EQ(mc.end() - mc.begin(), words.end() - words.begin());
}
