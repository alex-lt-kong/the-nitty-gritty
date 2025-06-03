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

template <class T, T v> struct integral_constant_2 {
  static constexpr T value = v;
  using value_type = T;
  using type = integral_constant<T, v>;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

// The order matters here
template <typename T, typename U>
// Primary template
struct is_same : integral_constant<bool, false> {};
// Full specialization for identical types
template <typename T> struct is_same<T, T> : integral_constant<bool, true> {};

} // namespace my
