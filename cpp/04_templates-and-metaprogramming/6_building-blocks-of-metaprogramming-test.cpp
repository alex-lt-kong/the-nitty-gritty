#include "6_building-blocks-of-metaprogramming-impl.h"

#include <gtest/gtest.h>

TEST(TestBuildingBlocksOfMetaprogramming, Fibonacci) {
  // std::integral_constant<class T, T val> is a fundamental building block
  // for template metaprogramming, used to wrap compile-time constants into
  // types. It's commonly used as a base class for many type traits in the
  // standard library.
  // https://timsong-cpp.github.io/cppwp/n4950/meta.help
  const std::integral_constant<int, 3> a;
  constexpr int b = 10;
  std::cout << my::fibonacci(b) << "\n";
}
