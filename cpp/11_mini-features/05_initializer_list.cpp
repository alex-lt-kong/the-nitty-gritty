#define FMT_HEADER_ONLY
#include <fmt/ranges.h>

#include <cxxabi.h>
#include <list>
#include <map>
#include <print>
#include <typeinfo>
#include <unordered_map>
#include <vector>

/*
Claude-3.5-Sonnet: abi::__cxa_demangle is not part of the C++ standard. It's a
compiler-specific function provided by the GCC and Clang. This function is part
of the C++ ABI (Application Binary Interface) used by these compilers to
demangle symbol names.
*/
std::string demangle(const char *name) {
  int status = -1;
  char *demangledName = abi::__cxa_demangle(name, NULL, NULL, &status);
  std::string result(status == 0 ? demangledName : name);
  free(demangledName);
  return result;
}

int main() {
  /*
  { 1, 2, 3...} is a very common and convenient syntax in C++, but what
  exactly is it?
  */

  // Example 0: Initializing a std::string
  std::string text{'h', 'e', 'l', 'l', 'o', ',', ' ',
                   'w', 'o', 'r', 'l', 'd', '!'};
  fmt::println("text: {}\n", text);

  // Example 1: Initializing a std::vector
  std::vector vec{1, 2, 3, 4, 5};
  fmt::println("vec: {}", vec);
  vec = {3, 1, 4, 1, 5, 9, 2, 6};
  fmt::println("vec: {}\n", vec);

  // Example 2: Initializing a std::list (i.e., a doubly linked list)
  std::list lst({1.1, 2.2, 3.3, 4.4});
  fmt::println("lst: {}", lst);
  lst = {3.14, 1.414, 2.71, 0.00, -1.00};
  fmt::println("lst: {}\n", lst);

  // Example 3: Initializing a 2D std::vector
  std::vector<std::vector<int>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  fmt::println("matrix: {}\n", matrix);

  // Example 4: Initializing a std::map
  std::map<std::string, double> mp{
      {"one", 1}, {"pi", 3.1415}, {"e", 2.71}, {"http", 80}, {"https", 443}};
  fmt::println("mp: {}\n", mp);

  // Example 5: Initializing a std::unordered_map
  std::unordered_map<std::string, int> ump{
      {"Warren", 94}, {"Donald", 78}, {"Charlie", 99}, {"Elon", 55}};
  fmt::println("ump: {}\n", ump);

  // what is an std::initializer_list
  auto what = {1, 2, 3};
  fmt::println("what is {}: {}\n", demangle(typeid(what).name()), what);

  // Example 6: Initializing a custom container (using std::initializer_list in
  // a constructor)
  class MyClass {
    int a = 0, b = 0, c = 0;

  public:
    MyClass(std::initializer_list<int> il) {
      /*
      init_list does not support subscript operator (i.e., init_list[0]), we can
      only iterate it with an iterator.
      */
      auto it = il.begin();
      int idx = 0;
      while (it != il.end()) {
        switch (idx++) {
        case 0:
          a = *it;
          break;
        case 1:
          b = *it;
          break;
        case 2:
          c = *it;
          break;
        default:
          break;
        }
        ++it;
      }
    }

    void print() { fmt::println("MyClass::print(): {}, {}, {}", a, b, c); }
  };

  MyClass mc{6, 5, 5, 3, 6};
  mc.print();
  // Question: is copy constructor or copy assignment operator called below?
  mc = {-2, -1, 0, 1, 2};
  mc.print();

  return 0;
}
