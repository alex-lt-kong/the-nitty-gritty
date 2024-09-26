#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

template <typename T>
constexpr T getMax(const T &a, const T &b) requires std::totally_ordered<T>
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

  getMax(3.5, 3.4);
  // getMax(md1, md2);
  return 0;
}
