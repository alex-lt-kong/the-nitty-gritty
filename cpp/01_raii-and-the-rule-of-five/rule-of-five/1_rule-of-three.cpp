#include "point-impl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <print>
#include <utility>

TEST(Constructors, DefaultConstructor) {
  Point p;
  EXPECT_EQ(p.getX(), 0);
  EXPECT_EQ(p.getY(), 0);
}

TEST(Constructors, CopyInitialization) {
  int x = 123;
  int y = 456;
  // LHS is a Point, RHS is an int, how come we can use an int to populate a
  // Point? Answer: C++ implicitly converts x to Point(x). This feels natural,
  // but could lead to unexpected issues
  Point p1 = x;
  // you cant do Point p2 = x, y for sure...
  auto p2 = Point(x, y);
  EXPECT_EQ(p1.getX(), p2.getX());
  EXPECT_NE(p1.getY(), p2.getY());

  // function-style notation
  Point p3(x, y);
  EXPECT_EQ(p1.getX(), p3.getX());
  EXPECT_NE(p1.getY(), p3.getY());

  // function-style notation
  Point p4(x, y);
  EXPECT_EQ(p3.getX(), p4.getX());
  EXPECT_EQ(p3.getY(), p4.getY());

  // uniform initialization:
  const Point p5{x, y};
  EXPECT_EQ(p4.getX(), p5.getX());
  EXPECT_EQ(p4.getY(), p5.getY());

  // Constructor Point(std::pair<int, int> xy) is marked as explicit,
  // you cant do Point p3 = std::pair<int, int>(x, y); anymore
  const auto p6 = Point(std::pair<int, int>(x, y));
  EXPECT_EQ(p5.getX(), p6.getX());
  EXPECT_EQ(p5.getY(), p6.getY());

  auto p7 = p6;
  EXPECT_EQ(p6.getX(), p7.getX());
  EXPECT_EQ(p6.getY(), p7.getY());
}

TEST(Constructors, CopyAssignment) {
  const auto p1 = Point(std::pair<int, int>(123, 456));
  const Point p2 = p1; // Copy constructor is called here
  EXPECT_EQ(p1.getX(), p2.getX());
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
  EXPECT_EQ(p4.getY(), p5.getY());
}
