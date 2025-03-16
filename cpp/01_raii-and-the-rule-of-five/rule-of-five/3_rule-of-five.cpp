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
  constexpr char text[] = "The quick brown fox jumps over the lazy dog";
  std::string ss1 = text;
  my_string ms1 = text;
  auto ss2 = ss1; // Copy constructor is called here
  auto ms2 = ms1; // Copy constructor is called here
  std::string ss3;
  my_string ms3 =
      "THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG'S BACK 1234567890";
  ss3 = text;
  ms3 = text;

  EXPECT_EQ(ms1, ms3);
  /*
  EXPECT_EQ(p1.getY(), p2.getY());
  Point p3; // Default constructor is called here

  Point p4{3, 4};
  Point p5(65535, -1234);
  p3 = p1; // Copy assignment operator is called here
  EXPECT_EQ(p2.getX(), p3.getX());
  EXPECT_EQ(p2.getY(), p3.getY());
  p5 = p4 = p3;
  EXPECT_EQ(p3.getX(), p4.getX());
  EXPECT_EQ(p3.getY(), p4.getY());
  EXPECT_EQ(p4.getX(), p5.getX());
  EXPECT_EQ(p4.getY(), p5.getY());*/
}
