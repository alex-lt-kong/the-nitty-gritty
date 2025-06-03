#pragma once

namespace my {
template <class T, T v> struct integral_constant {
  static constexpr T value = v;
  using value_type = T;
  using type = integral_constant<T, v>;
  // User-defined conversion function, allowing us to use the struct as a
  // value_type variable
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

// Compile-time Fibonacci calculation using constexpr function
constexpr int fibonacci(const int n) {
  return (n <= 1) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

} // namespace my
