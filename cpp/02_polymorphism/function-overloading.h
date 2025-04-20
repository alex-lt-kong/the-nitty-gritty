#ifndef FUNCTION_OVERLOADING_H
#define FUNCTION_OVERLOADING_H

#include <print>

// Function Overloading: Calls resolved at compile time via name mangling.
inline void my_print(int x) { std::println("x: {}", x); }
inline void my_print(int x, int y) { std::println("x: {}, y: {}", x, y); }

template <typename... Args> void my_print(Args... args) {
  ((std::print("{}, ", args)), ...); // Fold expression to unpack arguments
  std::println();
}
#endif // FUNCTION_OVERLOADING_H
