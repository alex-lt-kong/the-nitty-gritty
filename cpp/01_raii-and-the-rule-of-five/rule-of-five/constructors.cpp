#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <print>
#include <utility>

using namespace std;

class Point {
private:
  int x, y;

public:
  // Default constructor
  // "A default constructor is a constructor which can be called with no
  // arguments."
  // https://en.cppreference.com/w/cpp/language/default_constructor
  Point() {
    x = 0;
    y = 0;
    std::println("Default ctor Point() called, ({}, {})", x, y);
  }

  // Intentionally leave out explicit to make implicit conversion works
  Point(const int xy) {
    x = xy;
    y = xy;
    std::println("Ctor Point(int xy) called, ({}, {})", x, y);
  }

  explicit Point(std::pair<int, int> xy) {
    x = xy.first;
    y = xy.second;
    std::println("Ctor explicit Point(int xy) called, ({}, {})", x, y);
  }

  Point(int x1, int y1) {
    x = x1;
    y = y1;
    std::println("Ctor Point(int x1, int y1) called ({}, {})", x, y);
  }

  // Copy constructor
  Point(const Point &p1) {
    x = p1.x;
    y = p1.y;
    std::println("Copy ctor Point(const Point &p1) called ({}, {})", x, y);
  }

  // Copy assignment operator
  // Return value: a copy assignment operator doesn't need to return a
  // reference. It can return a value, but it is preferable to return by
  // non-const reference
  // https://stackoverflow.com/questions/3105798/why-must-the-copy-assignment-operator-return-a-reference-const-reference
  // Arguments: the below forms are all standard-compliant,
  // but their behaviors are different:
  // operator=(T)
  // operator=(T&)
  // operator=(const T&)
  // operator=(volatile T&)
  // operator=(const volatile T&)
  // https://www.ibm.com/docs/en/i/7.3?topic=only-copy-assignment-operators-c
  Point &operator=(const Point &p1) {
    x = p1.x;
    y = p1.y;
    std::println("Copy assignment operator= called ({}, {})", x, y);
    return *this;
  }

  // the const after the parameter list in the function declaration is a
  // "cv-qualifier". That is, it is used to specify constness or volatility of
  // the function
  // https://en.cppreference.com/w/cpp/language/cv
  // Declaring a member function with the const keyword specifies that the
  // function is a "read-only" function that doesn't modify the object for which
  // it's called.
  // https://learn.microsoft.com/en-us/cpp/cpp/const-cpp?view=msvc-170#const-member-functions
  int getX() const { return x; }
  int getY() const { return y; }
};

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
  const auto p1 = Point(pair<int, int>(123, 456));
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
