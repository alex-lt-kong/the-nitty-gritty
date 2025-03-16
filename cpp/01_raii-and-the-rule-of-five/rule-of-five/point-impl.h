#ifndef POINT_H
#define POINT_H

#include <print>

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

  // Move constructor
  Point(Point &&p1) {
    x = p1.x;
    y = p1.y;

    std::println("Move ctor Point(Point &&p1) called ({}, {})", x, y);
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
#endif // POINT_H
