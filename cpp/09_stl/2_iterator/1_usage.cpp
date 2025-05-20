#include <gtest/gtest.h>

#include <iostream>


TEST(IteratorUsage, BasicUsage) {
    std::vector<std::string> words = {"The",  "quick", "brown", "fox", "jumps",
                                      "over", "the",   "lazy",  "dog"};

    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    // `its != words.end()`, not `itr < words.end()`
    for (auto itr = words.begin(); itr != words.end(); ++itr) {
        std::cout << *itr << " ";
    }
    std::cout << std ::endl;
    const std::string stdout = buffer.str();
    EXPECT_EQ(stdout, "The quick brown fox jumps over the lazy dog \n");
    std::cout.rdbuf(old);
}

TEST(IteratorUsage, IterateOverTheOtherDirectionIsAwkwardButShouldWork) {
    std::vector<std::string> words = {"The",  "quick", "brown", "fox", "jumps",
                                      "over", "the",   "lazy",  "dog"};

    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    // `its != words.end()`, not `itr < words.end()`
    auto itr = words.end();
    while (true) {
        --itr;
        std::cout << *(itr) << " ";
        if (itr == words.begin()) {
            break;
        }
    }
    std::cout << std ::endl;
    const std::string stdout = buffer.str();
    EXPECT_EQ(stdout, "dog lazy the over jumps fox brown quick The \n");
    std::cout.rdbuf(old);
}

TEST(IteratorUsage, RangeBasedLoopUsesIterator) {
    // Question, what is the type of words here??
    const auto words = {"The",  "quick", "brown", "fox", "jumps",
                        "over", "the",   "lazy",  "dog"};
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    for (auto &word: words) {
        std::cout << word << " ";
    }
    std::cout << std ::endl;
    const auto stdout = buffer.str();

    EXPECT_EQ(stdout, "The quick brown fox jumps over the lazy dog \n");
    std::cout.rdbuf(old);
}


TEST(IteratorUsage, BidirectionalIterator) {
    // Question, what is the type of words here??
    const auto words = {"The",  "quick", "brown", "fox", "jumps",
                        "over", "the",   "lazy",  "dog"};
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    for (auto &word: words) {
        std::cout << word << " ";
    }
    std::cout << std ::endl;
    const auto stdout = buffer.str();

    EXPECT_EQ(stdout, "The quick brown fox jumps over the lazy dog \n");
    std::cout.rdbuf(old);
}

TEST(IteratorUsage, RandomAccessIterator) {
    // Question, what is the type of words here??
    const auto words = {"The",  "quick", "brown", "fox", "jumps",
                        "over", "the",   "lazy",  "dog"};
    const std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    auto it = words.begin();
    EXPECT_EQ(it[1], "quick");
    it += 2;
    EXPECT_EQ(it[0], "brown");
    EXPECT_EQ(*it, "brown");
    EXPECT_EQ(*(words.end() - 2), "lazy");
    std::cout.rdbuf(old);
}
