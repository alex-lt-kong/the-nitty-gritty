#include <cxxabi.h>
#include <iostream>
#include <ostream>
#include <typeinfo>
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

template <typename T> void printType(const T &var) {
  std::cout << "Type: " << typeid(var).name() << std::endl;
}

int main() {
  auto a = 12;
  auto b = -3.141;
  auto c = {1.414, 2.71, 3.14};
  std::vector<float> arr = {3.14, 1.414, 2.71, -0.2};
  std::cout << "a: " << a << "\n"
            << "b: " << b << "\n";
  for (auto const ele : c) {
    std::cout << ele << ", ";
  }
  std::cout << "\n";
  printType(a);
  printType(b);
  printType(c);
  std::cout << std::endl;
  return 0;
}