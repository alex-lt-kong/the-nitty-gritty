#include <algorithm>
#include <concepts>
#include <iostream>
#include <numbers>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

template <typename T>
concept is_shape = requires(T v) {
  { v.area() } -> std::floating_point;
};

class Circle {
private:
  double r;

public:
  Circle(double r) { this->r = r; }
  double area() { return r * r * std::numbers::pi_v<float>; };
};

class Cat {
public:
  Cat() {}
  void meow() { std::cout << "meow!~\n"; };
};

template <is_shape T> float getVolume(T &shape, float height) {
  return shape.area() * height;
}

template <typename T>
constexpr T getMaxWithRequires(const T &a,
                               const T &b) requires std::totally_ordered<T>
/*
 requires std::totally_ordered<T>
 is like
 C#'s class EmployeeList<T> where T : IComparable<T>
 */
{
  return a > b ? a : b;
};

class MyDummy {
private:
  int _a;

public:
  MyDummy(int a) { _a = a; }
  MyDummy() { _a = 1; }
};

int main() {
  MyDummy md1(2), md2(5);
  std::cout << "getMaxWithRequires(): " << getMaxWithRequires(3.5, 3.4) << "\n";
  // No, you can't pass md1 and md2 to getMaxWithRequires() as md1 and md2 is
  // not comparable getMax(md1, md2);

  Circle my_circle(3);
  std::cout << "getVolume(): " << getVolume(my_circle, 7.2) << "\n";
  Cat my_cat;
  // No, you can't pass my_cat to getVolume(), as Cat is not a shape!
  // std::cout << "getVolume(): " << getVolume(my_cat, 7.2) << "\n";
  return 0;
}
