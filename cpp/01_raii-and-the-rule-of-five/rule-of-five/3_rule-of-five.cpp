#include "my-string-impl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <print>
#include <utility>

TEST(RuleOfFive, DefaultConstructor) {
  std::string ss;
  my_string ms;
  EXPECT_EQ(ss.size(), ms.size());
  EXPECT_EQ(*ss.c_str(), *ms.c_str());
}

TEST(RuleOfFive, CopyInitialization) {
  constexpr char text[] = "Hello world, I am a 0xDEADBEEF!";
  my_string ms1 = text;
  std::string ss1 = text;
  EXPECT_EQ(ms1, ss1);

  // function-style notation
  my_string ms2(text);
  // uniform initialization:
  my_string ms3{text};

  EXPECT_EQ(ms1, ms1);
  EXPECT_EQ(ms1, ms2);
  EXPECT_EQ(ms2, ms3);

  my_string ms4 = ms3;
  EXPECT_EQ(ms4, ms3);
}

TEST(RuleOfFive, CopyAssignment) {
  // my_string only re-allocates when current size not big enough, so we want to
  // test both cases
  constexpr char text1[] = "The quick brown fox jumps over the lazy dog";
  constexpr char text2[] =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
      "tempor incididunt ut labore et dolore magna aliqua.";
  constexpr char text3[] = "0x0000000FF1CE";
  std::string ss1 = text1;
  my_string ms1 = text1;

  std::string ss2 = text2;
  my_string ms2 = text2;
  ss2 = ss1;
  ms2 = ms1;
  EXPECT_EQ(ss1, ss2);
  EXPECT_EQ(ms1, ms2);

  std::string ss3 = text3;
  my_string ms3 = text3;
  ss3 = ss1;
  ms3 = ms1;
  EXPECT_EQ(ss1, ss3);
  EXPECT_EQ(ms1, ms3);
  EXPECT_EQ(strcmp(ms3.c_str(), ss3.c_str()), 0);
}

TEST(RuleOfFive, MoveInitialization) {
  constexpr char text[] = "The quick brown fox jumps over the lazy dog";
  my_string ms1 = text;
  std::string ss1 = text;
  my_string ms2 = std::move(ms1);
  std::string ss2 = std::move(ss1);
  EXPECT_EQ(strcmp(ss2.c_str(), text), 0);
  EXPECT_EQ(strcmp(ss2.c_str(), ms2.c_str()), 0);
  EXPECT_EQ(ss1.size(), 0);
  EXPECT_EQ(ms1.size(), 0);
}

TEST(RuleOfFive, MoveAssignment) {
  constexpr char text1[] = "The quick brown fox jumps over the lazy dog";
  constexpr char text2[] =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
      "tempor incididunt ut labore et dolore magna aliqua.";
  std::string ss1 = text1;
  my_string ms1 = text1;

  std::string ss2 = text2;
  my_string ms2 = text2;
  ss2 = std::move(ss1);
  ms2 = std::move(ms1);

  EXPECT_EQ(strcmp(ss2.c_str(), text1), 0);
  EXPECT_EQ(strcmp(ms2.c_str(), text1), 0);
  EXPECT_EQ(ss1.size(), 0);
  EXPECT_EQ(ms1.size(), 0);

  ss2 = std::move(ss2);
  ms2 = std::move(ms2);
  // std::string does not have self-assignment protection??
  // EXPECT_EQ(strcmp(ss2.c_str(), text1), 0);
  EXPECT_EQ(strcmp(ms2.c_str(), text1), 0);
}

TEST(RuleOfFive, AdditionOperator) {
  constexpr char text1[] = "Hello, ";
  constexpr char text2[] = "World!";
  std::string ss1 = text1;
  my_string ms1 = text1;
  std::string ss2 = text2;
  my_string ms2 = text2;

  auto ss3 = ss1 + ss2;
  auto ms3 = ms1 + ms2;
  EXPECT_EQ(strcmp(ms3.c_str(), ss3.c_str()), 0);
  ss3 = ss3 + ss3 + ss1 + ss2;
  ms3 = ms3 + ms3 + ms1 + ms2;
  EXPECT_EQ(strcmp(ms3.c_str(), ss3.c_str()), 0);
  ss3 = ss3 + ss1 + ss2;
  ms3 = ms3 + ms1 + ms2;
  EXPECT_EQ(strcmp(ms3.c_str(), ss3.c_str()), 0);
}