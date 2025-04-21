#ifndef FUNCTION_OVERLOADING_H
#define FUNCTION_OVERLOADING_H

#include <print>

// Must specify __attribute__((noinline)) otherwise the compiler could inline
// some implementations below
__attribute__((noinline)) inline void my_print(int x) {
  std::println("x: {}", x);
}

__attribute__((noinline)) inline void my_print(int x, int y) {
  std::println("x: {}, y: {}", x, y);
}

// Variadic template function to handle multiple arguments
template <typename... Args>
__attribute__((noinline)) void my_print(Args... args) {
  /*
   * Fold expression, since C++17.
   * Important to note that this is NOT a recursive function call.
   * The fold expression is expanded at compile time.
   * For example, if you call my_print(1, "Hello world", 3.1415);
   * the template function will be expanded to:
   * std::print("{}, ", 1);
   * std::print("{}, ", "Hello world");
   * std::print("{}, ", 3.1415);
   * std::println();
   */
  ((std::print("{}, ", args)), ...);
  std::println("<end>");
}
#endif // FUNCTION_OVERLOADING_H
