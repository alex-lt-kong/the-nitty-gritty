// In C++, perfect forwarding is the act of passing a function’s parameters to
// another function while preserving its reference category (e.g.,
// rvalue/lvalue/etc). It is commonly used by wrapper methods that want to pass
// their parameters through to another function, often a constructor.
// https://devblogs.microsoft.com/oldnewthing/20230727-00/?p=108494

#include <fmt/format.h>
#include <vector>

class MyDummyClass {
  std::vector<int> data;

public:
  MyDummyClass() { data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; }

  // copy constructor
  MyDummyClass(MyDummyClass const &other) : data(other.data) {}

  // move constructor
  MyDummyClass(MyDummyClass &&other) : data(std::move(other.data)) {}

  // copy-assignment
  MyDummyClass &operator=(MyDummyClass const &rhs) {
    if (&rhs == this)
      return *this;
    data = rhs.data;
    return *this;
  }

  // move-assignment
  MyDummyClass &operator=(MyDummyClass &&rhs) {
    if (&rhs == this)
      return *this;
    // While std::move() does not move anything, the below invokes std::vector's
    // move assignment operation, which moves elements
    data = std::move(rhs.data);
    return *this;
  }
  void print() { fmt::print("{}\n", fmt::join(data, ", ")); }
};

void g(MyDummyClass &&t) {
  fmt::print("rvalue overload called\n");
  t.print();
}
void g(MyDummyClass &t) {
  fmt::print("lvalue overload called\n");
  t.print();
}

template <typename T> void non_forwarding_func(T &&t) { g(t); }
// perfectly forwarding func always takes an T &&t
template <typename T> void perfectly_forwarding_func(T &&t) {
  g(std::forward<T>(t));
}

int main() {
  MyDummyClass dc;

  non_forwarding_func(dc);             // dc is an lvalue
  non_forwarding_func(MyDummyClass()); // MyDummyClass() returns an rvalue

  perfectly_forwarding_func(dc);             // dc is an lvalue
  perfectly_forwarding_func(MyDummyClass()); // MyDummyClass() returns an rvalue
  return 0;
}