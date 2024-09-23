#include <iostream>
#include <string>

// Base case for the recursive variadic template
void variadicPrint() { std::cout << std::endl; }

// Variadic template function that takes a parameter pack
template <typename T, typename... Args>
void variadicPrint(T first, Args... args) {
  std::cout << first << " ";
  variadicPrint(args...);
}

template <typename... Args> void variadicPrintWithFolds(Args... args) {
  ((std::cout << args << " "), ...);
  std::cout << "\n";
}

// Variadic template for sum calculation
template <typename T> T variadicSum(T t) { return t; }

template <typename T, typename... Args> T variadicSum(T first, Args... args) {
  return first + variadicSum(args...);
}

// Using fold expression to sum all arguments
template <typename... Args> auto variadicSumWithFolds(Args... args) {
  return (... + args); // unary left fold
}

int main() {
  // Demonstrate print function with different types and number of arguments
  variadicPrint("Hello", "C++", "Parameter", "Pack", 2024, "which is", true);
  variadicPrintWithFolds(1, 2.5, 'a', "mixed types");

  // Demonstrate sum function
  std::cout << "variadicSum():          " << variadicSum(1, 2, 3, 4, 5) << "\n";
  std::cout << "variadicSum():          " << variadicSum(1.1, 2.2, 3.3, 4.5)
            << "\n";
  std::cout << "variadicSumWithFolds(): "
            << variadicSumWithFolds(3.14, 2.414, 2.71) << "\n";

  return 0;
}